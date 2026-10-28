"""Step 12a: fine-tune only the codebook values of a clustered TinyBERT; indices stay fixed.

The ResNet Step 11a method, extended to nn.Embedding: every clustered weight
(Linear and, in the Linear + embeddings condition, the three embedding tables) is
computed as codebook[indices] in the forward pass, so each codebook entry's gradient
is the sum of its members' gradients. TinyBERT uses LayerNorm, which has no running
statistics, so there is nothing to re-estimate; the model stays in eval() (dropout
off) as in the TinyBERT LoRA recipe. Selection uses the selection split only; both
test sets are evaluated once each, after every variant is selected and saved.
"""

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fault_lora.data.sst2 import SentenceDataset, make_collate
from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import require, require_state_equal
from fault_lora.evaluation.source_hashes import compare_source_hashes, unique_changes
from fault_lora.evaluation.tinybert_clean_eval import environment_record, source_artifacts
from fault_lora.evaluation.tinybert_sources import build_tinybert, evaluate_splits, make_loaders, synthetic_token_batch
from fault_lora.memory.clustering import decode_state, validate_record
from fault_lora.memory.faults import encoding_fingerprint, validate_parameters
from fault_lora.training.baseline import choose_device
from fault_lora.training.codebook_finetune import CodebookLayer, assert_frozen, build_encoding, codebook_changes, codebook_order, effective_state, frozen_snapshot, verify_same_cells
from fault_lora.training.tinybert_recovery import TinyBertRecoveryConfig, lora_training_data, prepare_recovery_sources


SELECTION = "selection"
SWEEPS = {"linear-and-embeddings": "results/tinybert-sst2-fault-sweep-linear-emb-seed1000", "linear-only": "results/tinybert-sst2-fault-sweep-seed42"}
EVALUATION_FIELDS = ["model", "split", "num_examples", "num_correct", "accuracy", "cross_entropy_loss", "gap_to_clean_percentage_points"]


class CodebookEmbedding(nn.Module):
    """An nn.Embedding whose table exists only as codebook[indices]."""

    def __init__(self, base, record):
        super().__init__()
        validate_record(record)
        require(isinstance(base, nn.Embedding) and list(base.weight.shape) == record["shape"], "Codebook record does not fit this embedding")
        require(base.max_norm is None and not base.sparse, "max_norm and sparse embeddings are not supported")
        del base._parameters["weight"]
        self.base = base
        self.register_buffer("indices", record["indices"].detach().clone())
        self.codebook = nn.Parameter(record["codebook"].detach().clone())

    def weight(self):
        return self.codebook[self.indices.long()]

    def forward(self, inputs):
        base = self.base
        return F.embedding(inputs, self.weight(), base.padding_idx, None, base.norm_type, base.scale_grad_by_freq, False)


def attach_tinybert_codebooks(model, encoding):
    """Wrap exactly the encoding's clustered tensors (Linear and Embedding weights)."""
    layers = {}
    for name in sorted(encoding["clustered_tensors"]):
        require(name.endswith(".weight"), f"Unexpected clustered tensor: {name}")
        module_name = name.removesuffix(".weight")
        parent_name, _, child = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child)
        record = encoding["clustered_tensors"][name]
        layer = CodebookEmbedding(base, record) if isinstance(base, nn.Embedding) else CodebookLayer(base, record)
        setattr(parent, child, layer)
        layers[module_name] = layer
    return layers


def set_tinybert_trainable(model, layers, variant):
    """Codebooks always; LayerNorm affine and all biases only where the variant declares them."""
    model.requires_grad_(False)
    names = []
    for module_name, layer in layers.items():
        layer.codebook.requires_grad_(True)
        names.append(f"{module_name}.codebook")
    for name, parameter in model.named_parameters():
        module_name = name.rpartition(".")[0]
        is_layernorm = isinstance(model.get_submodule(module_name.removesuffix(".base") if module_name.endswith(".base") else module_name), nn.LayerNorm)
        if (variant["train_layernorm"] and is_layernorm) or (variant["train_biases"] and name.endswith(".bias") and not is_layernorm):
            parameter.requires_grad_(True)
            names.append(name)
    require({name for name, parameter in model.named_parameters() if parameter.requires_grad} == set(names), "Trainable set differs from the declaration")
    return sorted(names)


@dataclass
class TinyBertCodebookConfig:
    run_id: str = "tinybert-sst2-codebook-finetune-k16-seed42"
    variants: list = field(default_factory=lambda: [
        {"variant_id": "linemb-codebook", "label": "Linear + embeddings, codebooks only", "condition": "linear-and-embeddings", "train_layernorm": False, "train_biases": False},
        {"variant_id": "linemb-codebook-lnbias", "label": "Linear + embeddings, codebooks + LayerNorm + biases", "condition": "linear-and-embeddings", "train_layernorm": True, "train_biases": True},
        {"variant_id": "lin-codebook", "label": "Linear only, codebooks only", "condition": "linear-only", "train_layernorm": False, "train_biases": False}])
    source_sweeps: dict = field(default_factory=lambda: dict(SWEEPS))
    epochs: int = 2
    evaluations_per_epoch: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    schedule: str = "cosine-per-step"
    model_mode: str = "eval"
    seed: int = 42
    expected_training_examples: int = 62349
    device: str = "mps"
    cpu_threads: int = 4
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        validate_parameters(0, self.seed)
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        keys = {"variant_id", "label", "condition", "train_layernorm", "train_biases"}
        if not isinstance(self.variants, list) or not self.variants or any(not isinstance(item, dict) or set(item) != keys for item in self.variants):
            raise ValueError(f"Each variant declares exactly {sorted(keys)}")
        if len({item["variant_id"] for item in self.variants}) != len(self.variants) or any(not re.fullmatch(r"[a-z0-9][a-z0-9-]*", str(item["variant_id"])) for item in self.variants):
            raise ValueError("variant_id values must be distinct simple names")
        if any(item["condition"] not in self.source_sweeps or not isinstance(item["train_layernorm"], bool) or not isinstance(item["train_biases"], bool) for item in self.variants):
            raise ValueError("Each variant needs a declared condition sweep and boolean switches")
        for name, minimum in (("epochs", 1), ("evaluations_per_epoch", 1), ("batch_size", 1), ("expected_training_examples", 1), ("cpu_threads", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, positive in (("learning_rate", True), ("weight_decay", False), ("max_grad_norm", True)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"Invalid {name}")
        if self.schedule != "cosine-per-step" or self.model_mode != "eval":
            raise ValueError("Only the declared cosine-per-step schedule and eval() mode are supported")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def variant_sources(root, config, condition):
    """The verified clustered base of one condition, via the TinyBERT recovery source chain."""
    recovery = TinyBertRecoveryConfig(run_id="unused", base_kind="clustered", condition=condition, pattern_id=None, source_sweep_results=config.source_sweeps[condition],
                                      expected_training_examples=config.expected_training_examples)
    context = prepare_recovery_sources(root, recovery)
    return recovery, context


def train_variant(config, variant, context, training, loaders, device, directory, source_info):
    """Train one variant from its clustered encoding; select on the selection split only."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    sources, encoding, record = context["sources"], context["encoding"], context["record"]
    model = build_tinybert(sources.checkpoint["model_config"], decode_state(encoding))
    layers = attach_tinybert_codebooks(model, encoding)
    trainable = set_tinybert_trainable(model, layers, variant)
    model.to(device).eval()
    snapshot = frozen_snapshot(model)
    original_indices = {name: layer.indices.detach().cpu().clone() for name, layer in layers.items()}
    counts = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable_tensors": len(trainable),
              "codebook_values": sum(layer.codebook.numel() for layer in layers.values()), "clustered_tensors": len(layers)}
    directory.mkdir(parents=True)
    selection_loader = loaders[SELECTION]
    initial = evaluate_splits(model, {SELECTION: selection_loader}, device)[SELECTION]
    require(initial["num_correct"] == record["measurements"][SELECTION]["num_correct"], "Epoch-0 codebook model differs from the clustered sweep reference")

    def payload(point, selection):
        state = effective_state(model, layers)
        return {"format_version": 1, "kind": "tinybert_codebook_finetune_state", "variant": variant, "epoch": point, "selection_metrics": selection, "config": asdict(config), **source_info,
                "codebooks": {f"{name}.weight": layer.codebook.detach().cpu().clone() for name, layer in layers.items()},
                "untouched_state": {name: value for name, value in state.items() if name not in {f"{module}.weight" for module in layers}}}

    save_torch(directory / "best.pt", payload(0.0, initial))
    best_key, best_point = (initial["accuracy"], -initial["cross_entropy_loss"]), 0.0
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True, collate_fn=make_collate(sources.tokenizer, sources.config.max_length),
                        generator=torch.Generator().manual_seed(config.seed))
    steps = len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps * config.epochs)
    boundaries = sorted({math.ceil(steps * (index + 1) / config.evaluations_per_epoch) for index in range(config.evaluations_per_epoch)})
    history = [{"epoch": 0.0, SELECTION: initial, "note": "clustered encoding before any codebook update"}]
    for epoch in range(config.epochs):
        segment, started = {"count": 0, "correct": 0, "loss": 0.0}, time.monotonic()
        for step, batch in enumerate(loader, start=1):
            model.eval()
            inputs = {name: value.to(device) for name, value in batch.items() if name != "labels"}
            targets = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = F.cross_entropy(logits, targets)
            require(math.isfinite(loss.item()), f"Nonfinite training loss at epoch {epoch + 1}, step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            segment["count"] += targets.numel()
            segment["correct"] += (logits.detach().argmax(1) == targets).sum().item()
            segment["loss"] += loss.item() * targets.numel()
            if step in boundaries:
                point = round(epoch + (boundaries.index(step) + 1) / config.evaluations_per_epoch, 6)
                optimizer.zero_grad(set_to_none=True)
                assert_frozen(model, snapshot, trainable)
                require(not any(module.training for module in model.modules()), "Model left eval mode")
                require(all(torch.equal(layer.indices.detach().cpu(), original_indices[name]) for name, layer in layers.items()), "Indices changed during training")
                selection = evaluate_splits(model, {SELECTION: selection_loader}, device)[SELECTION]
                if (selection["accuracy"], -selection["cross_entropy_loss"]) > best_key:
                    best_key, best_point = (selection["accuracy"], -selection["cross_entropy_loss"]), point
                    save_torch(directory / "best.pt", payload(point, selection))
                order = codebook_order({name: layer.codebook for name, layer in layers.items()})
                history.append({"epoch": point, "learning_rate_after": optimizer.param_groups[0]["lr"], "training": {"num_examples": segment["count"], "accuracy": segment["correct"] / segment["count"],
                                "cross_entropy_loss": segment["loss"] / segment["count"]}, SELECTION: selection, "frozen_state_verified": True, "indices_unchanged": True,
                                "codebooks_strictly_increasing": all(item["strictly_increasing"] for item in order.values()), "seconds": time.monotonic() - started})
                print(json.dumps({"event": "tinybert_codebook_evaluation", "variant": variant["variant_id"], "epoch": point, "training_loss": history[-1]["training"]["cross_entropy_loss"],
                                  "selection_accuracy": selection["accuracy"], "selected": best_point, "strictly_increasing": history[-1]["codebooks_strictly_increasing"]}), flush=True)
                segment, started = {"count": 0, "correct": 0, "loss": 0.0}, time.monotonic()
    save_torch(directory / "last.pt", payload(float(config.epochs), history[-1][SELECTION]))
    selected = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    require(selected["epoch"] == best_point, "Selected checkpoint differs from the selection")
    return {"variant": variant, "selected_epoch": best_point, "selected": selected, "history": history, "trainable": trainable, "counts": counts, "model": model, "layers": layers}


def run_tinybert_codebook_finetune(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run directories already exist; choose a new run_id")
    contexts = {condition: variant_sources(root, config, condition)[1] for condition in sorted({item["condition"] for item in config.variants})}
    any_sources = next(iter(contexts.values()))["sources"]
    require(all(context["sources"].manifest["split_sha256"] == any_sources.manifest["split_sha256"] for context in contexts.values()), "Conditions use different splits")
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_codebook_finetune.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        write_json(results_dir / "test_sets.json", any_sources.test_sets)
        environment = {**environment_record(device, config.cpu_threads, ()), "reproducibility": "Seeded data order; codebook gradients are scatter-add sums whose float order may differ on GPU backends."}
        write_json(results_dir / "environment.json", environment)
        train_sentences, train_labels = lora_training_data(any_sources, config)
        training = SentenceDataset(train_sentences, train_labels)
        loaders = make_loaders(any_sources)
        results = {}
        for variant in config.variants:
            context = contexts[variant["condition"]]
            source_info = {"source_encoding_fingerprint": encoding_fingerprint(context["encoding"]), "condition": variant["condition"], "split_sha256": any_sources.manifest["split_sha256"]}
            result = train_variant(config, variant, context, training, loaders, device, checkpoint_dir / variant["variant_id"], source_info)
            results[variant["variant_id"]] = result
            write_json(results_dir / f"history_{variant['variant_id']}.json", result["history"])
            write_json(results_dir / "status.json", {"status": "running", "completed_variants": list(results)})
        write_json(results_dir / "trainable.json", {key: {"label": item["variant"]["label"], "condition": item["variant"]["condition"], **item["counts"], "trainable_names": item["trainable"],
                                                          "selected_epoch": item["selected_epoch"]} for key, item in results.items()})
        order = {key: codebook_order(item["selected"]["codebooks"]) for key, item in results.items()}
        violations = {key: {name: {k: report[name][k] for k in ("adjacent_violations", "inverted_pairs")} for name in report if not report[name]["strictly_increasing"]} for key, report in order.items()}
        write_json(results_dir / "codebook_order.json", {"selected": order, "violations": violations})
        if any(violations.values()):
            # Keep the run and its trained states; never re-sort or remap cells, and save no baseline encoding.
            write_json(results_dir / "status.json", {"status": "stopped_codebook_order_violation", "violations": {k: v for k, v in violations.items() if v}})
            print(json.dumps({"event": "codebook_order_violation", "violations": {k: v for k, v in violations.items() if v}}), flush=True)
            return {"status": "stopped_codebook_order_violation", "violations": violations}
        encodings = {}
        for key, item in results.items():
            context, selected = contexts[item["variant"]["condition"]], item["selected"]
            source = context["encoding"]
            metadata = {"representation": "per-tensor scalar codebooks and indices; codebooks fine-tuned with indices fixed; no faults or adapters", "faults_injected": 0,
                        "codebook_finetune": {"run_id": config.run_id, "variant": item["variant"], "selected_epoch": item["selected_epoch"], "config": asdict(config),
                                              "trainable_parameters": item["counts"]["trainable_parameters"]},
                        "source_encoding_fingerprint": encoding_fingerprint(source), "environment": environment}
            new = build_encoding(source, selected["codebooks"], selected["untouched_state"], metadata)
            path = checkpoint_dir / key / "encoding.pt"
            save_torch(path, new)
            reloaded = torch.load(path, map_location="cpu", weights_only=True)
            cells = verify_same_cells(reloaded, source)
            decoded = decode_state(reloaded)
            # The saved encoding must reconstruct exactly the weights the selected model used.
            check = build_tinybert(context["sources"].checkpoint["model_config"], decode_state(source))
            layers = attach_tinybert_codebooks(check, source)
            with torch.no_grad():
                for name, value in selected["codebooks"].items():
                    layers[name.removesuffix(".weight")].codebook.copy_(value)
                named = dict(check.named_parameters()) | dict(check.named_buffers())
                for name, value in selected["untouched_state"].items():
                    module_name, _, leaf = name.rpartition(".")
                    named[f"{module_name}.base.{leaf}" if module_name in layers else name].copy_(value)
            require_state_equal(decoded, effective_state(check, layers), f"Saved encoding reconstruction ({key})")
            for name, value in source["untouched_state"].items():
                module_name, _, leaf = name.rpartition(".")
                if (f"{module_name}.base.{leaf}" if module_name in layers else name) not in item["trainable"]:
                    require(torch.equal(reloaded["untouched_state"][name], value), f"Frozen state changed: {name}")
            dense_path = checkpoint_dir / key / "dense.pt"
            save_torch(dense_path, {"model_state_dict": decoded, "source_codebook_encoding": str(path.relative_to(root)), "model_config": context["sources"].checkpoint["model_config"]})
            changes = codebook_changes(reloaded, source)
            write_json(results_dir / f"codebook_changes_{key}.json", changes)
            encodings[key] = {"encoding": str(path.relative_to(root)), "encoding_sha256": sha256_file(path), "fingerprint": encoding_fingerprint(reloaded), "dense": str(dense_path.relative_to(root)),
                              "dense_sha256": sha256_file(dense_path), "cells_identical_to_source": cells, "condition": item["variant"]["condition"],
                              "untouched_state_changed": sorted(name for name in source["untouched_state"] if not torch.equal(reloaded["untouched_state"][name], source["untouched_state"][name])),
                              "max_relative_weight_change": max(change["relative_weight_change"] for change in changes.values()),
                              "max_abs_codebook_change": max(change["max_abs_change"] for change in changes.values())}
        write_json(results_dir / "encodings.json", encodings)
        # Test sets are read only now, after every variant's selection and encoding are final.
        models = {"clean": {name: value.detach().clone() for name, value in any_sources.model.state_dict().items()}}
        references = {"clean": next(iter(contexts.values()))["references"]}
        for condition, context in contexts.items():
            models[f"clustered-{condition}"] = decode_state(context["encoding"])
        models.update({key: decode_state(torch.load(root / item["encoding"], map_location="cpu", weights_only=True)) for key, item in encodings.items()})
        evaluation = {}
        for name, state in models.items():
            print(json.dumps({"event": "paired_evaluation_start", "model": name}), flush=True)
            evaluation[name] = evaluate_splits(build_tinybert(any_sources.checkpoint["model_config"], state).to(device), loaders, device)
        for condition, context in contexts.items():
            for split, value in context["references"].items():
                require(evaluation[f"clustered-{condition}"][split]["num_correct"] == value["clustered_no_faults"]["num_correct"], f"Clustered replay differs: {condition} {split}")
                require(evaluation["clean"][split]["num_correct"] == value["clean"]["num_correct"], f"Clean replay differs: {split}")
        clean = evaluation["clean"]
        rows = [{"model": name, "split": split, **values[split], "gap_to_clean_percentage_points": 100 * (clean[split]["accuracy"] - values[split]["accuracy"])} for name, values in evaluation.items() for split in values]
        write_json(results_dir / "paired_evaluation.json", {"models": evaluation, "rows": rows, "selected_epochs": {key: item["selected_epoch"] for key, item in results.items()},
                                                            "note": "The selection split was part of the checkpoint's fine-tuning data and chooses checkpoints; only the two test sets are results."})
        with (results_dir / "paired_evaluation.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVALUATION_FIELDS)
            writer.writeheader()
            writer.writerows({key: row[key] for key in EVALUATION_FIELDS} for row in rows)
        accepted = [change for context in contexts.values() for change in context["sources"].accepted_source_hash_changes] + compare_source_hashes(root, source_code, "Source code changed during the run")
        artifacts = {}
        for condition, context in contexts.items():
            artifacts.update({f"{condition}:{name}": item for name, item in source_artifacts(root, context["sources"].paths, context["sources"].hashes).items()})
        for name, item in artifacts.items():
            require(sha256_file(root / item["path"]) == item["sha256"], f"An input artifact changed: {name}")
        manifest = {"run_id": config.run_id, "source_artifacts": artifacts, "encodings": encodings,
                    "method": "weight = codebook[indices] for every clustered Linear and Embedding weight; only declared parameters train; eval() mode (dropout off); LayerNorm has no running statistics.",
                    "selection_rule": "Highest selection accuracy, then lowest selection cross-entropy; earlier point on exact ties; epoch 0 (clustered encoding) competes.",
                    "test_policy": "Each test set evaluated once per model after every variant was selected and saved; test sets are never pooled.",
                    "ordering": "Codebooks checked for strict order after selection; never re-sorted or remapped.", "accepted_source_hash_changes": unique_changes(accepted),
                    "environment": environment, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "verification.json", verify_tinybert_codebook_finetune(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "selected_epochs": {key: item["selected_epoch"] for key, item in results.items()}, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "tinybert_codebook_finetune_complete", "run_id": config.run_id, "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return {"status": "complete", "evaluation": evaluation, "selected_epochs": {key: item["selected_epoch"] for key, item in results.items()}, "encodings": encodings}
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)})
        raise


def verify_tinybert_codebook_finetune(root, config):
    """Re-read the saved run: selection, ordering, cell identity, reconstruction, records, and hashes."""
    root = Path(root).resolve()
    results_dir = root / config.results_root / config.run_id
    manifest = json.loads((results_dir / "manifest.json").read_text())
    paired = json.loads((results_dir / "paired_evaluation.json").read_text())
    checks = {}
    for variant in config.variants:
        key = variant["variant_id"]
        history = json.loads((results_dir / f"history_{key}.json").read_text())
        selected, best = 0.0, (history[0][SELECTION]["accuracy"], -history[0][SELECTION]["cross_entropy_loss"])
        for item in history[1:]:
            candidate = (item[SELECTION]["accuracy"], -item[SELECTION]["cross_entropy_loss"])
            if candidate > best:
                selected, best = item["epoch"], candidate
        require(len(history) == 1 + config.epochs * config.evaluations_per_epoch and all(item["frozen_state_verified"] and item["indices_unchanged"] for item in history[1:]), f"History incomplete: {key}")
        require(selected == paired["selected_epochs"][key], f"Selected point is not the selection optimum: {key}")
        payload = torch.load(root / config.checkpoint_root / config.run_id / key / "best.pt", map_location="cpu", weights_only=True)
        require(payload["epoch"] == selected and payload["selection_metrics"] == next(item for item in history if item["epoch"] == selected)[SELECTION], f"Selected checkpoint differs: {key}")
        item = manifest["encodings"][key]
        require(sha256_file(root / item["encoding"]) == item["encoding_sha256"] and sha256_file(root / item["dense"]) == item["dense_sha256"], f"Saved artifacts changed: {key}")
        encoding = torch.load(root / item["encoding"], map_location="cpu", weights_only=True)
        require(encoding_fingerprint(encoding) == item["fingerprint"], f"Fingerprint differs: {key}")
        require(all(report["strictly_increasing"] for report in codebook_order({name: record["codebook"] for name, record in encoding["clustered_tensors"].items()}).values()), f"Saved codebooks unsorted: {key}")
        require(all(torch.equal(encoding["clustered_tensors"][name]["codebook"], value) for name, value in payload["codebooks"].items()), f"Encoding codebooks differ from the selected checkpoint: {key}")
        require_state_equal(torch.load(root / item["dense"], map_location="cpu", weights_only=True)["model_state_dict"], decode_state(encoding), f"Dense reconstruction ({key})")
        checks[key] = {"selected_epoch": selected, "strictly_sorted": True, "fingerprint": item["fingerprint"], "dense_matches_encoding": True}
    for name, item in manifest["source_artifacts"].items():
        require(sha256_file(root / item["path"]) == item["sha256"], f"Input artifact changed: {name}")
    with (results_dir / "paired_evaluation.csv").open(newline="") as stream:
        require(len(list(csv.DictReader(stream))) == len(paired["rows"]), "CSV and JSON evaluations disagree")
    accepted = compare_source_hashes(root, json.loads((results_dir / "source_sha256.json").read_text()), "Source code changed after the run")
    return {"status": "passed", "variants": checks, "selection_verified_using_selection_split_only": True, "reference_counts_match_saved": True,
            "source_artifacts_verified": len(manifest["source_artifacts"]), "accepted_source_hash_changes": unique_changes(accepted)}


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tinybert_sst2_codebook_finetune.json"))
    args = parser.parse_args()
    run_tinybert_codebook_finetune(TinyBertCodebookConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
