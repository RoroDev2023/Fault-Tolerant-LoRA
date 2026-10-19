"""Train only LoRA parameters against one frozen base: clean, clustered, or one saved static fault pattern."""

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from fault_lora.data.cifar10 import cifar10_transforms
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import require, require_state_equal
from fault_lora.evaluation.fault_sweep import SweepConfig, prepare_sources as prepare_audited_sources, sweep_plan
from fault_lora.evaluation.source_hashes import compare_source_hashes
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, summarize_fault_pattern, validate_parameters
from fault_lora.models.lora import adapter_state_dict, attach_lora, base_state_dict, load_adapter_state
from fault_lora.models.resnet import build_resnet18
from fault_lora.training.baseline import choose_device, seed_worker


BASE_KINDS = ("clean", "clustered", "faulty", "retrained-clustered", "retrained-faulty")
PATTERN_KINDS = ("faulty", "retrained-faulty")
BASE_LABELS = {"clean": "Clean base", "clustered": "Clustered base", "faulty": "Faulty base", "retrained-clustered": "Retrained clustered base", "retrained-faulty": "Retrained faulty base"}


@dataclass
class RecoveryConfig:
    run_id: str = "resnet18-cifar10-lora-r4-p001-seed52-train42"
    source_sweep_results: str = "results/resnet18-cifar10-fault-sweep-seed42"
    source_audit_results: str = "results/resnet18-cifar10-fault-checks-seed42"
    pattern_id: str | None = "rate03-seed52"
    rank: int = 4
    alpha: float = 4.0
    target_names: list | None = None
    seed: int = 42
    epochs: int = 5
    batch_size: int = 128
    evaluation_batch_size: int = 256
    learning_rate: float = .001
    weight_decay: float = .0001
    max_grad_norm: float = 1.0
    device: str = "mps"
    cpu_threads: int = 4
    num_workers: int = 2
    data_root: str = "data"
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"
    # Step 8 configurations predate these fields; their defaults reproduce that path.
    base_kind: str = "faulty"
    clean_checkpoint_sha256: str | None = None

    def validate(self):
        validate_parameters(0, self.seed)
        if self.base_kind not in BASE_KINDS:
            raise ValueError(f"base_kind must be one of {BASE_KINDS}")
        if (self.pattern_id is not None) != (self.base_kind in PATTERN_KINDS):
            raise ValueError("pattern_id is required for a faulty base and must be null otherwise")
        for name in ("run_id", "pattern_id") if self.base_kind in PATTERN_KINDS else ("run_id",):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", getattr(self, name)):
                raise ValueError(f"{name} must be a simple name")
        if self.clean_checkpoint_sha256 is not None and (not isinstance(self.clean_checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.clean_checkpoint_sha256)):
            raise ValueError("clean_checkpoint_sha256 must be null or a full lowercase SHA-256 digest")
        for name, minimum in (("rank", 1), ("epochs", 1), ("batch_size", 1), ("evaluation_batch_size", 1), ("cpu_threads", 1), ("num_workers", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, positive in (("alpha", True), ("learning_rate", True), ("weight_decay", False), ("max_grad_norm", True)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"Invalid {name}")
        if self.target_names is not None and (not isinstance(self.target_names, list) or not self.target_names
                or any(not isinstance(name, str) or not name for name in self.target_names) or len(set(self.target_names)) != len(self.target_names)):
            raise ValueError("target_names must be null or a nonempty list of distinct layer names")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def prepare_sources(root, config):
    """Verify the audit, sweep provenance, and the configured base before training.

    The record holds the base's previously measured Step 7 validation/test results:
    the selected pattern for a faulty base, the replayed references otherwise.
    Retrained base kinds (Step 11c) are prepared from the Step 11b sweep instead.
    """
    if config.base_kind.startswith("retrained-"):
        from fault_lora.training.retrained_bases import prepare_retrained_sources
        return prepare_retrained_sources(root, config)
    directory = root / config.source_sweep_results
    require(json.loads((directory / "status.json").read_text())["status"] == "complete", "Source sweep is incomplete")
    sweep_config = SweepConfig(**json.loads((directory / "config.json").read_text()))
    require(sweep_config.source_audit_results == config.source_audit_results, "Recovery and sweep use different fault audits")
    plan = sweep_plan(sweep_config)
    require(json.loads((directory / "plan.json").read_text()) == plan, "Saved sweep plan changed")
    records = json.loads((directory / "measurements.json").read_text())
    require(len(records) == len(plan) and [record["pattern_id"] for record in records] == [entry["pattern_id"] for entry in plan], "Saved sweep records are incomplete or reordered")
    references = json.loads((directory / "references.json").read_text())
    faulty = config.base_kind == "faulty"
    if faulty:
        entries = [entry for entry in plan if entry["pattern_id"] == config.pattern_id]
        require(len(entries) == 1 and entries[0]["probability"] > 0, "Recovery requires a saved nonzero sweep pattern")
        record = next(record for record in records if record["pattern_id"] == config.pattern_id)
        require(all(record[name] == entries[0][name] for name in entries[0]), "Selected record differs from the sweep plan")
        require(record == json.loads((directory / "patterns" / f"{config.pattern_id}.json").read_text()), "Individual and aggregate pattern records disagree")
    else:
        # A clean base has no stored cells, so its fault fields are null rather than zero.
        clustered = config.base_kind == "clustered"
        require(plan[0]["probability"] == 0 and all(records[0]["measurements"][name] == references[name]["clustered_no_faults"] for name in ("validation", "test")),
                "Zero-fault sweep record differs from the clustered reference")
        record = {"pattern_id": None, "seed": None, "probability": 0.0 if clustered else None,
                  "counts": {"actual_corrupted_cells": 0 if clustered else None, "realized_fault_rate": 0.0 if clustered else None},
                  "measurements": {name: references[name]["clustered_no_faults" if clustered else "clean"] for name in ("validation", "test")}}
    encoding, clean, metadata, split, _, _, paths, hashes = prepare_audited_sources(root, sweep_config)
    del clean
    metadata = {name: value for name, value in metadata.items() if name != "model_state_dict"}
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, item in manifest["source_artifacts"].items():
        path = root / item["path"]
        require(sha256_file(path) == item["sha256"], f"Sweep source artifact changed: {name}")
        paths[f"sweep_input_{name}"], hashes[f"sweep_input_{name}"] = path, item["sha256"]
    accepted = compare_source_hashes(root, json.loads((directory / "source_sha256.json").read_text()), "Previously used sweep code changed")
    for filename in ["config.json", "plan.json", "status.json", "manifest.json", "measurements.json", "references.json", "source_sha256.json"] + ([f"patterns/{config.pattern_id}.json"] if faulty else []):
        path = directory / filename
        paths[f"sweep_{filename}"], hashes[f"sweep_{filename}"] = path, sha256_file(path)
    require(config.clean_checkpoint_sha256 in (None, hashes["clean"]), "Clean checkpoint differs from the configured SHA-256")
    if faulty:
        paths["selected_pattern"] = root / record["pattern"]
        hashes["selected_pattern"] = record["pattern_sha256"]
        require(sha256_file(paths["selected_pattern"]) == hashes["selected_pattern"], "Selected pattern file changed")
        pattern = torch.load(paths["selected_pattern"], map_location="cpu", weights_only=True)
        require(pattern["seed"] == record["seed"] and pattern["fault_probability"] == record["probability"], "Selected pattern seed or probability changed")
        require(summarize_fault_pattern(encoding, pattern) == record["counts"], "Selected pattern counts differ from Step 7")
    # Accepted recorded-hash differences travel with the record so every caller can report them.
    record = {**record, "accepted_source_hash_changes": accepted}
    return encoding, metadata, split, record, references, paths, hashes


def build_base_state(config, encoding, paths):
    """Return the independent dense state that stays frozen under the adapters.

    clean: the Step 3 validation-selected checkpoint the encoding was clustered from.
    clustered: the decoded Step 4 encoding with no faults. faulty: the same encoding
    read through the saved static pattern, exactly as in Step 8. Retrained kinds
    use the Step 11a encoding and Step 11b patterns.
    """
    if config.base_kind.startswith("retrained-"):
        from fault_lora.training.retrained_bases import build_retrained_state
        return build_retrained_state(config, encoding, paths)
    if config.base_kind == "clean":
        state = torch.load(paths["clean"], map_location="cpu", weights_only=True)["model_state_dict"]
        require(set(state) == set(encoding["clustered_tensors"]) | set(encoding["untouched_state"]), "Clean checkpoint and encoding describe different tensors")
        require(all(torch.equal(state[name], value) for name, value in encoding["untouched_state"].items()), "Encoding was not clustered from this clean checkpoint")
        return {name: value.detach().clone() for name, value in state.items()}
    if config.base_kind == "clustered":
        return decode_state(encoding)
    require(config.base_kind == "faulty", "Unsupported base_kind")
    return apply_fault_pattern(encoding, torch.load(paths["selected_pattern"], map_location="cpu", weights_only=True))


def train_adapter_epoch(model, loader, optimizer, device, max_grad_norm):
    # eval() fixes running means/variances and counters; it does NOT disable gradients.
    # These adapters have no dropout or normalization requiring training mode.
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    criterion = nn.CrossEntropyLoss()
    count, correct, total_loss = 0, 0, 0.0
    started = time.monotonic()
    for batch, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        require(math.isfinite(loss.item()), "Nonfinite adapter training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        count += targets.numel()
        correct += (logits.detach().argmax(1) == targets).sum().item()
        total_loss += loss.item() * targets.numel()
        if batch % 50 == 0 or batch == len(loader):
            print(json.dumps({"event": "adapter_training_progress", "batch": batch, "batches": len(loader), "accuracy": correct / count,
                              "cross_entropy_loss": total_loss / count, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    require(count > 0, "Empty recovery training split")
    return {"num_examples": count, "num_correct": correct, "accuracy": correct / count, "cross_entropy_loss": total_loss / count}


def assert_frozen_base(model, original_state):
    require_state_equal(base_state_dict(model), original_state, "Recovery frozen parameters and buffers")
    adapter_names = set(adapter_state_dict(model))
    require({name for name, parameter in model.named_parameters() if parameter.requires_grad} == adapter_names,
            "Trainable parameter set differs from adapter parameters")
    require(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in adapter_names), "Base parameter received a gradient")


def _restore_adapter(payload, config, root, device):
    require(payload["format_version"] == 1 and payload["kind"] == "static_fault_lora_adapter", "Unsupported recovery checkpoint")
    config.validate()
    encoding, metadata, split, record, _, paths, hashes = prepare_sources(root, config)
    require(payload["split_sha256"] == metadata["split_sha256"] and payload["encoding_fingerprint"] == encoding_fingerprint(encoding), "Adapter checkpoint storage or split changed")
    require(payload["architecture"] == "resnet18" and payload["classes"] == metadata["classes"] and payload["preprocessing"] == metadata["preprocessing"],
            "Adapter checkpoint architecture, classes, or preprocessing differ")
    state = build_base_state(config, encoding, paths)
    model = build_resnet18(metadata["num_classes"], pretrained=False)
    model.load_state_dict(state, strict=True)
    layout = attach_lora(model, config.rank, config.alpha, config.target_names)
    require(layout == payload["adapter_layout"], "Adapter layout differs from saved checkpoint")
    load_adapter_state(model, payload["adapter_state_dict"])
    return model.to(device).eval(), {"metadata": metadata, "split": split, "record": record, "paths": paths, "hashes": hashes, "base_state": state}


def load_recovered_model(checkpoint_path, project_root, *, device="cpu"):
    """Reconstruct the exact base the adapter was trained on, then load adapter-only weights."""
    root = Path(project_root)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, context = _restore_adapter(payload, RecoveryConfig(**payload["config"]), root, device)
    require(payload["source_artifacts"] == {name: {"path": str(path.relative_to(root)), "sha256": context["hashes"][name]} for name, path in context["paths"].items()},
            "Adapter checkpoint source provenance differs")
    # Not part of the saved checkpoint: the source-hash differences accepted while loading it.
    return model, {**payload, "accepted_source_hash_changes": context["record"].get("accepted_source_hash_changes", [])}


def load_adapter_on_base(adapter_checkpoint, base_kind, pattern_id, project_root, *, device="cpu"):
    """Load a saved adapter onto a possibly DIFFERENT base of the same architecture.

    Split, encoding fingerprint, architecture, and the complete adapter layout must
    match the checkpoint; only base_kind/pattern_id may differ. No training.
    """
    payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
    config = replace(RecoveryConfig(**payload["config"]), base_kind=base_kind, pattern_id=pattern_id)
    model, context = _restore_adapter(payload, config, Path(project_root), device)
    return model, payload, {**context, "config": config}


def evaluate_adapter_on_base(adapter_checkpoint, base_kind, pattern_id, project_root, *, device="cpu"):
    """Measure a saved adapter on another base's validation/test splits; evaluation only.

    The same base is also measured with the adapter output zeroed. Nothing is
    trained, selected, or written; these numbers must not choose any setting.
    """
    root, device = Path(project_root), choose_device(device)
    model, payload, context = load_adapter_on_base(adapter_checkpoint, base_kind, pattern_id, root, device=device)
    config, metadata, split, record = context["config"], context["metadata"], context["split"], context["record"]
    source = RecoveryConfig(**payload["config"])
    _, eval_transform = cifar10_transforms(metadata["preprocessing"]["input_size"][-1])
    train_data = CIFAR10(root / config.data_root, train=True, transform=eval_transform, download=False)
    test_data = CIFAR10(root / config.data_root, train=False, transform=eval_transform, download=False)
    require(train_data.classes == test_data.classes == metadata["classes"] and len(train_data) == 50000 and len(test_data) == 10000, "CIFAR-10 class order or size changed")
    datasets = {"validation": Subset(train_data, split["validation"]), "test": Subset(test_data, split["test"]["indices"])}
    loaders = {name: DataLoader(dataset, batch_size=config.evaluation_batch_size, shuffle=False, num_workers=config.num_workers,
                                persistent_workers=config.num_workers > 0, pin_memory=device.type == "cuda") for name, dataset in datasets.items()}
    with_adapter = {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}
    trained = adapter_state_dict(model)
    load_adapter_state(model, {name: torch.zeros_like(value) if ".lora_B." in name else value for name, value in trained.items()})
    without_adapter = {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}
    load_adapter_state(model, trained)
    assert_frozen_base(model, context["base_state"])
    require(all(sha256_file(path) == context["hashes"][name] for name, path in context["paths"].items()), "Adapter evaluation changed an input artifact")
    return {"adapter_checkpoint_sha256": sha256_file(adapter_checkpoint), "adapter_epoch": payload["epoch"], "rank": source.rank, "alpha": source.alpha, "adapter_seed": source.seed,
            "adapter_base": {"base_kind": source.base_kind, "pattern_id": source.pattern_id},
            "evaluation_base": {"base_kind": base_kind, "pattern_id": pattern_id, "fault_probability": record["probability"],
                                "actual_corrupted_cells": record["counts"]["actual_corrupted_cells"], "realized_fault_rate": record["counts"]["realized_fault_rate"]},
            "same_base": (source.base_kind, source.pattern_id) == (base_kind, pattern_id), "device": str(device),
            "accepted_source_hash_changes": record.get("accepted_source_hash_changes", []),
            "with_adapter": with_adapter, "without_adapter": without_adapter,
            "accuracy_gain_percentage_points": {name: 100 * (with_adapter[name]["accuracy"] - without_adapter[name]["accuracy"]) for name in loaders},
            "base_num_correct_difference_from_step7": {name: without_adapter[name]["num_correct"] - record["measurements"][name]["num_correct"] for name in loaders}}


def plot_recovery(history, metrics, references, output_dir):
    with tempfile.TemporaryDirectory(prefix="fault-lora-recovery-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            epochs = [0] + [item["epoch"] for item in history]
            values = [100 * metrics["before"]["validation"]["accuracy"]] + [100 * item["validation"]["accuracy"] for item in history]
            axes[0].plot(epochs, values, "o-", color="#174c78", label="Validation accuracy")
            selected = metrics["selected_epoch"]
            axes[0].scatter([selected], [100 * metrics["after"]["validation"]["accuracy"]], color="#b54d30", s=90, marker="*", zorder=4, label="Selected checkpoint")
            axes[0].set_xlabel("Epoch (0 = zero-output adapter)")
            axes[0].set_xticks(epochs)
            axes[0].set_title("Validation selects the checkpoint")
            test_values = [100 * metrics[stage]["test"]["accuracy"] for stage in ("before", "after")]
            label = BASE_LABELS[metrics["base_kind"]]
            bars = axes[1].bar([label, f"{label} + LoRA"], test_values, width=.55, color=["#a76a2d", "#174c78"])
            axes[1].bar_label(bars, fmt="%.2f%%", padding=4)
            axes[1].set_title("Test: same saved pattern before / after" if metrics["pattern_id"] is not None else "Test: same frozen base before / after")
            for axis, split in zip(axes, ("validation", "test")):
                axis.axhline(100 * references[split]["clean"]["accuracy"], linestyle="--", color="#26734d", label="Original clean reference")
                axis.axhline(100 * references[split]["clustered_no_faults"]["accuracy"], linestyle=":", color="#a76a2d", label="Clustered zero-fault reference")
                axis.set_ylim(0, 100)
                axis.set_ylabel("Classification accuracy (%)")
                axis.grid(axis="y", alpha=.2)
                axis.set_axisbelow(True)
                axis.legend(loc="lower left", fontsize=8)
            config = metrics["config"]
            fig.suptitle(f"ResNet-18 / CIFAR-10: rank-{config['rank']} LoRA recovery, {metrics['pattern_id'] or metrics['base_kind'] + ' base'}")
            fig.text(.5, .02, f"One {'static pattern' if metrics['pattern_id'] is not None else 'frozen base'} and one training seed; no uncertainty estimate. Base parameters and BatchNorm buffers fixed.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .055, 1, .94))
            paths = [Path(output_dir) / f"lora_recovery.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=200)
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def run_recovery(config, project_root):
    config.validate()
    root = Path(project_root)
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Recovery directories already exist; choose a new run_id")
    encoding, metadata, split, record, references, source_paths, source_hashes = prepare_sources(root, config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    started = time.monotonic()
    history = []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_lora_recovery.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        write_json(results_dir / "splits.json", split)
        write_json(results_dir / "preprocessing.json", metadata["preprocessing"])
        if config.pattern_id is not None:
            write_json(results_dir / "fault_counts.json", record["counts"])
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "cpu_threads": config.cpu_threads,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "matplotlib")},
                       "reproducibility": "Seeded adapter initialization, data shuffle, workers and augmentation; bitwise replay across hardware/library versions is not guaranteed."}
        write_json(results_dir / "environment.json", environment)
        sources = {name: {"path": str(path.relative_to(root)), "sha256": source_hashes[name]} for name, path in source_paths.items()}
        original_state = build_base_state(config, encoding, source_paths)
        model = build_resnet18(metadata["num_classes"], pretrained=False)
        model.load_state_dict(original_state, strict=True)
        base_parameters = sum(parameter.numel() for parameter in model.parameters())
        # Synthetic CPU logits provide an exact, full-model zero-output identity check.
        model.eval()
        probe = torch.randn(2, *metadata["preprocessing"]["input_size"], generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            initial_logits = model(probe)
        layout = attach_lora(model, config.rank, config.alpha, config.target_names)
        with torch.no_grad():
            require(torch.equal(initial_logits, model(probe)), "Zero-output adapters changed initial logits")
        model.to(device)
        assert_frozen_base(model, original_state)
        initial_adapters = adapter_state_dict(model)
        adapter_parameters = sum(value.numel() for value in initial_adapters.values())
        adapter_bytes = sum(value.numel() * value.element_size() for value in initial_adapters.values())
        write_json(results_dir / "adapter_layout.json", layout)
        train_transform, eval_transform = cifar10_transforms(metadata["preprocessing"]["input_size"][-1])
        training_data = CIFAR10(root / config.data_root, train=True, transform=train_transform, download=False)
        evaluation_data = CIFAR10(root / config.data_root, train=True, transform=eval_transform, download=False)
        require(training_data.classes == evaluation_data.classes == metadata["classes"] and len(training_data) == len(evaluation_data) == 50000, "CIFAR-10 class order or size changed")
        options = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0, "worker_init_fn": seed_worker, "pin_memory": device.type == "cuda"}
        train_loader = DataLoader(Subset(training_data, split["train"]), batch_size=config.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(config.seed), **options)
        validation_loader = DataLoader(Subset(evaluation_data, split["validation"]), batch_size=config.evaluation_batch_size, shuffle=False,
                                       generator=torch.Generator().manual_seed((config.seed + 1) % 2**32), **options)
        before = {"validation": evaluate_classifier(model, validation_loader, device)}
        require(before["validation"]["num_correct"] == record["measurements"]["validation"]["num_correct"], "Initial validation accuracy differs from the Step 7 measurement of this base")
        write_json(results_dir / "before.json", before)
        checkpoint_metadata = {"format_version": 1, "kind": "static_fault_lora_adapter", "base_kind": config.base_kind, "config": asdict(config), "adapter_layout": layout,
                               "source_artifacts": sources, "encoding_fingerprint": encoding_fingerprint(encoding), "split_sha256": metadata["split_sha256"],
                               "architecture": "resnet18", "dataset": "CIFAR-10", "classes": metadata["classes"], "preprocessing": metadata["preprocessing"]}
        def save_adapter(filename, epoch, validation):
            save_torch(checkpoint_dir / filename, {**checkpoint_metadata, "epoch": epoch, "validation_metrics": validation, "adapter_state_dict": adapter_state_dict(model)})
        save_adapter("initial.pt", 0, before["validation"])
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        # Epoch zero competes with trained adapters, allowing a harmful run to retain baseline.
        best_key = (before["validation"]["accuracy"], -before["validation"]["cross_entropy_loss"])
        best_epoch = 0
        save_adapter("best.pt", 0, before["validation"])
        training_started = time.monotonic()
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.monotonic()
            learning_rate = optimizer.param_groups[0]["lr"]
            print(json.dumps({"event": "adapter_epoch_start", "epoch": epoch, "epochs": config.epochs, "learning_rate": learning_rate}), flush=True)
            training = train_adapter_epoch(model, train_loader, optimizer, device, config.max_grad_norm)
            assert_frozen_base(model, original_state)
            validation = evaluate_classifier(model, validation_loader, device)
            key = (validation["accuracy"], -validation["cross_entropy_loss"])
            if key > best_key:
                best_key, best_epoch = key, epoch
                save_adapter("best.pt", epoch, validation)
            save_adapter("last.pt", epoch, validation)
            history.append({"epoch": epoch, "learning_rate": learning_rate, "training": training, "validation": validation,
                            "frozen_base_verified": True, "seconds": time.monotonic() - epoch_started})
            scheduler.step()
            write_json(results_dir / "history.json", history)
            write_json(results_dir / "status.json", {"status": "running", "completed_epochs": epoch, "selected_epoch": best_epoch})
            print(json.dumps({"event": "adapter_epoch_complete", **history[-1], "selected_epoch": best_epoch}), flush=True)
        training_seconds = time.monotonic() - training_started
        selected_state = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
        load_adapter_state(model, selected_state["adapter_state_dict"])
        # Independent reconstruction via the public loader verifies saved provenance and serialization.
        reloaded, _ = load_recovered_model(checkpoint_dir / "best.pt", root, device=device)
        device_probe = probe.to(device)
        with torch.no_grad():
            require(torch.equal(model(device_probe), reloaded(device_probe)), "Saved adapter reload changed logits")
        del reloaded
        after = {"validation": evaluate_classifier(model, validation_loader, device)}
        require(after["validation"] == selected_state["validation_metrics"], "Selected validation metrics changed on replay")
        # Test data is loaded only after validation-based checkpoint selection is final.
        test_data = CIFAR10(root / config.data_root, train=False, transform=eval_transform, download=False)
        require(test_data.classes == metadata["classes"] and len(test_data) == 10000, "CIFAR-10 test class order or size changed")
        test_loader = DataLoader(Subset(test_data, split["test"]["indices"]), batch_size=config.evaluation_batch_size, shuffle=False,
                                 generator=torch.Generator().manual_seed((config.seed + 2) % 2**32), **options)
        after["test"] = evaluate_classifier(model, test_loader, device)
        assert_frozen_base(model, original_state)
        load_adapter_state(model, initial_adapters)
        before["test"] = evaluate_classifier(model, test_loader, device)
        require(before["test"]["num_correct"] == record["measurements"]["test"]["num_correct"], "Paired test baseline differs from the Step 7 measurement of this base")
        load_adapter_state(model, selected_state["adapter_state_dict"])
        assert_frozen_base(model, original_state)
        require(all(sha256_file(path) == source_hashes[name] for name, path in source_paths.items()), "Recovery changed an input artifact")
        accepted = record.get("accepted_source_hash_changes", []) + compare_source_hashes(root, source_code, "Recovery source code changed during the run")
        metrics = {"run_id": config.run_id, "config": asdict(config), "base_kind": config.base_kind, "pattern_id": config.pattern_id, "fault_seed": record["seed"], "fault_probability": record["probability"],
                   "actual_corrupted_cells": record["counts"]["actual_corrupted_cells"], "realized_fault_rate": record["counts"]["realized_fault_rate"], "selected_epoch": best_epoch,
                   "before": before, "after": after,
                   "recovery": {name: {"accuracy_gain_percentage_points": 100 * (after[name]["accuracy"] - before[name]["accuracy"]),
                                       "cross_entropy_reduction": before[name]["cross_entropy_loss"] - after[name]["cross_entropy_loss"],
                                       "gap_to_clustered_percentage_points": 100 * (references[name]["clustered_no_faults"]["accuracy"] - after[name]["accuracy"]),
                                       "gap_to_clean_percentage_points": 100 * (references[name]["clean"]["accuracy"] - after[name]["accuracy"])} for name in before},
                   "parameters": {"frozen_base": base_parameters, "trainable_adapter": adapter_parameters, "adapter_to_base_fraction": adapter_parameters / base_parameters,
                                  "adapter_tensor_bytes": adapter_bytes, "best_serialized_bytes": (checkpoint_dir / "best.pt").stat().st_size},
                   "best_checkpoint": str((checkpoint_dir / "best.pt").relative_to(root)), "best_checkpoint_sha256": sha256_file(checkpoint_dir / "best.pt"),
                   "training_and_validation_seconds": training_seconds, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "before.json", before)
        write_json(results_dir / "metrics.json", metrics)
        figures = plot_recovery(history, metrics, references, results_dir)
        manifest = {**checkpoint_metadata, "environment": environment, "selection_rule": "Highest validation accuracy, then lowest validation loss; exact ties retain earlier epoch. Includes epoch-zero adapters.",
                    "base_selection": {"clean": "Step 3 validation-selected clean checkpoint; no clustering or faults.", "clustered": "Decoded Step 4 encoding with zero faults.",
                                       "faulty": "Step 4 encoding read through the configured saved static pattern.",
                                       "retrained-clustered": "Step 11a codebook-retrained encoding with zero faults.",
                                       "retrained-faulty": "Step 11a codebook-retrained encoding read through the configured Step 11b static pattern."}[config.base_kind],
                    "pattern_selection": "Configured saved pattern; initial default is first ordinal at p=0.01, not selected by its score." if config.pattern_id is not None else "No fault pattern; pattern_id is null.",
                    "test_policy": "Same fixed base before/after; each model evaluated once on official test after validation-only checkpoint selection.",
                    "frozen_state": "All base weights, biases, BatchNorm affine parameters, running statistics and counters unchanged. Model stays in eval mode with adapter gradients enabled.",
                    "adapter_storage": "Reliable float32 adapter storage, unmerged residual branches; no adapter faults injected; byte estimate excludes optimizer and archive overhead.",
                    "limitations": "One base and one adapter training seed per run; this run alone measures no unseen-pattern transfer, rank effect, or hardware speed/storage-area.",
                    "accepted_source_hash_changes": accepted,
                    "source_artifacts_preserved": True, "base_state_preserved_each_epoch": True, "zero_output_identity_verified": True, "adapter_reload_logits_exact": True,
                    "split_sizes": {"train": len(split["train"]), "validation": len(split["validation"]), "test": len(split["test"]["indices"])},
                    "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures], "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "status.json", {"status": "complete", "completed_epochs": config.epochs, "selected_epoch": best_epoch, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "lora_recovery_complete", "run_id": config.run_id, "before_test": before["test"], "after_test": after["test"], "recovery": metrics["recovery"],
                          "parameters": metrics["parameters"], "selected_epoch": best_epoch, "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return metrics
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_epochs": len(history), "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_lora.json"))
    args = parser.parse_args()
    run_recovery(RecoveryConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
