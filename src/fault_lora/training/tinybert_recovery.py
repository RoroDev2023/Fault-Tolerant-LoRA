"""Adapter-only LoRA recovery on frozen TinyBERT bases (clean, clustered, or faulty).

Sibling of `training/recovery.py` with the same conventions: `attach_linear_lora`
(the existing `LoRALayer`; a capped-rank subclass only where rank exceeds a layer
dimension, i.e. the 312 → 2 classifier) on every nn.Linear, a zero-output start that competes as epoch 0,
frozen-base checks, selection-only checkpoint choice, test sets evaluated only
after selection, adapter-only checkpoints, provenance, and post-run
verification. The base is fully frozen (weights, biases, embeddings, LayerNorm)
and the model stays in eval() mode, which also disables BERT dropout.

Bases come from a verified TinyBERT fault sweep: the Step 10b sweep or a Step
10c condition sweep. `condition` names the encoding: "linear-only" (Step 10b
and Condition A) or "linear-and-embeddings" (Condition B). A clean base has no
condition.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fault_lora.data.sst2 import SentenceDataset, make_collate, read_sst2
from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import require, require_state_equal
from fault_lora.evaluation.fault_sweep import sweep_plan
from fault_lora.evaluation.source_hashes import compare_source_hashes
from fault_lora.evaluation.tinybert_clean_eval import environment_record, source_artifacts, verify_source_artifacts, verify_source_code
from fault_lora.evaluation.tinybert_condition_sweep import ConditionSweepConfig, condition_plan, prepare_condition_sources
from fault_lora.evaluation.tinybert_fault_sweep import TinyBertSweepConfig, prepare_sweep_sources
from fault_lora.evaluation.tinybert_sources import build_tinybert, evaluate_splits, linear_weight_names, make_loaders, synthetic_token_batch
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, summarize_fault_pattern, validate_parameters
from fault_lora.models.lora import adapter_state_dict, base_state_dict, load_adapter_state
from fault_lora.models.tinybert_lora import attach_linear_lora
from fault_lora.training.baseline import choose_device

BASE_KINDS = ("clean", "clustered", "faulty")
CONDITIONS = ("linear-only", "linear-and-embeddings")
CONDITION_TAGS = {None: "none", "linear-only": "lin", "linear-and-embeddings": "linemb"}
SWEEP_CONDITIONS = {"linear-high-rate": "linear-only", "linear-and-embeddings": "linear-and-embeddings"}
CHECKPOINT_KIND = "tinybert_static_fault_lora_adapter"
SELECTION = "selection"


def recovery_run_id(base_kind, condition, pattern_id, rank, seed):
    return f"tinybert-sst2-lora-{base_kind}-{CONDITION_TAGS[condition]}-{pattern_id or 'none'}-r{rank}-train{seed}"


@dataclass
class TinyBertRecoveryConfig:
    run_id: str = "tinybert-sst2-lora-faulty-linemb-rate07-seed1030-r4-train42"
    base_kind: str = "faulty"
    condition: str | None = "linear-and-embeddings"
    pattern_id: str | None = "rate07-seed1030"
    source_sweep_results: str = "results/tinybert-sst2-fault-sweep-linear-emb-seed1000"
    rank: int = 4
    alpha: float = 4.0
    seed: int = 42
    epochs: int = 2
    evaluations_per_epoch: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    schedule: str = "cosine-per-step"
    expected_training_examples: int = 62349
    loss_window_steps: int = 100
    device: str = "cpu"
    cpu_threads: int = 4
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        validate_parameters(0, self.seed)
        if self.base_kind not in BASE_KINDS:
            raise ValueError(f"base_kind must be one of {BASE_KINDS}")
        if (self.pattern_id is not None) != (self.base_kind == "faulty"):
            raise ValueError("pattern_id is required for a faulty base and must be null otherwise")
        if (self.condition is None) != (self.base_kind == "clean") or (self.condition is not None and self.condition not in CONDITIONS):
            raise ValueError(f"condition must be null for a clean base and one of {CONDITIONS} otherwise")
        for name in ("run_id", "pattern_id") if self.base_kind == "faulty" else ("run_id",):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", getattr(self, name)):
                raise ValueError(f"{name} must be a simple name")
        for name, minimum in (("rank", 1), ("epochs", 1), ("evaluations_per_epoch", 1), ("batch_size", 1), ("expected_training_examples", 1), ("loss_window_steps", 1), ("cpu_threads", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, positive in (("alpha", True), ("learning_rate", True), ("weight_decay", False), ("max_grad_norm", True)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"Invalid {name}")
        if self.schedule != "cosine-per-step":
            raise ValueError("Only the declared cosine-per-step schedule is supported")
        if not isinstance(self.source_sweep_results, str) or not self.source_sweep_results:
            raise ValueError("source_sweep_results is required")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def load_sweep(root, config):
    """The verified sweep: its condition, plan, records, references, sources, and encoding."""
    directory = root / config.source_sweep_results
    read = lambda name: json.loads((directory / name).read_text())
    require(read("status.json")["status"] == "complete" and read("verification.json")["status"] == "passed", "Source sweep is incomplete or unverified")
    saved = read("config.json")
    if "condition" in saved:
        sweep_config = ConditionSweepConfig(**saved)
        condition, plan = SWEEP_CONDITIONS[sweep_config.condition], condition_plan(sweep_config)
        sources, clean_metrics, _, encoding = prepare_condition_sources(root, sweep_config)
    else:
        sweep_config = TinyBertSweepConfig(**saved)
        condition, plan = "linear-only", sweep_plan(sweep_config.step7_config())
        sources, clean_metrics, _, encoding = prepare_sweep_sources(root, sweep_config)
    require(read("plan.json") == plan, "Saved sweep plan changed")
    manifest = read("manifest.json")
    verify_source_artifacts(root, manifest, "Source sweep")
    verify_source_code(root, directory, "Source sweep")
    require(manifest["encoding_fingerprint"] == encoding_fingerprint(encoding), "Sweep encoding differs")
    records, references = read("measurements.json"), read("references.json")
    require([record["pattern_id"] for record in records] == [entry["pattern_id"] for entry in plan], "Saved sweep records are incomplete or reordered")
    for name in ("config", "plan", "status", "verification", "manifest", "measurements", "references", "source_sha256"):
        sources.paths[f"sweep_{name}"], sources.hashes[f"sweep_{name}"] = directory / f"{name}.json", sha256_file(directory / f"{name}.json")
    return condition, plan, records, references, sources, clean_metrics, encoding


def prepare_recovery_sources(root, config):
    """Verify the sweep chain and the configured base; return what training and evaluation need."""
    condition, plan, records, references, sources, _, encoding = load_sweep(root, config)
    require(config.base_kind == "clean" or config.condition == condition, f"Configured condition {config.condition} differs from the sweep's {condition}")
    splits = list(sources.datasets)
    pattern = None
    if config.base_kind == "faulty":
        entries = [entry for entry in plan if entry["pattern_id"] == config.pattern_id]
        require(len(entries) == 1 and entries[0]["probability"] > 0, "Recovery requires a saved nonzero sweep pattern")
        record = next(record for record in records if record["pattern_id"] == config.pattern_id)
        directory = root / config.source_sweep_results
        require(record == json.loads((directory / "patterns" / f"{config.pattern_id}.json").read_text()), "Individual and aggregate pattern records disagree")
        sources.paths["selected_pattern"], sources.hashes["selected_pattern"] = root / record["pattern"], record["pattern_sha256"]
        require(sha256_file(sources.paths["selected_pattern"]) == record["pattern_sha256"], "Selected pattern file changed")
        pattern = torch.load(sources.paths["selected_pattern"], map_location="cpu", weights_only=True)
        require(pattern["seed"] == record["seed"] and pattern["fault_probability"] == record["probability"], "Selected pattern seed or probability changed")
        require(summarize_fault_pattern(encoding, pattern) == record["counts"], "Selected pattern counts differ from the sweep")
    else:
        clustered = config.base_kind == "clustered"
        record = {"pattern_id": None, "seed": None, "probability": 0.0 if clustered else None,
                  "counts": {"actual_corrupted_cells": 0 if clustered else None, "realized_fault_rate": 0.0 if clustered else None, "eligible_cells": None},
                  "measurements": {name: references[name]["clustered_no_faults" if clustered else "clean"] for name in splits}}
    return {"condition": condition, "sources": sources, "encoding": encoding, "record": record, "references": references, "pattern": pattern}


def build_recovery_base(config, context):
    """The independent dense state that stays frozen under the adapters."""
    encoding, sources = context["encoding"], context["sources"]
    if config.base_kind == "clean":
        state = {name: value.detach().clone() for name, value in sources.model.state_dict().items()}
        require(set(state) == set(encoding["clustered_tensors"]) | set(encoding["untouched_state"]), "Clean checkpoint and encoding describe different tensors")
        require(all(torch.equal(state[name], value) for name, value in encoding["untouched_state"].items()), "Encoding was not made from this clean checkpoint")
        return state
    if config.base_kind == "clustered":
        return decode_state(encoding)
    return apply_fault_pattern(encoding, context["pattern"])


def lora_training_data(sources, config):
    """The Step 10a LoRA training split: train rows not in the selection split."""
    manifest = sources.manifest
    path = manifest["source_artifacts"]["dataset"]["files"]["data/train-00000-of-00001.parquet"]["path"]
    snapshot = sources.root / path[: -len("data/train-00000-of-00001.parquet")]
    sst2, _ = read_sst2(snapshot, sources.root / sources.config.data_cache / "datasets")
    indices = sources.split["train"]
    require(len(indices) == config.expected_training_examples and not set(indices) & set(sources.split["selection"]), "LoRA training split is not the saved 62,349-row partition")
    return [sst2["train"]["sentences"][index] for index in indices], [sst2["train"]["labels"][index] for index in indices]


def assert_frozen_base(model, original_state):
    require_state_equal(base_state_dict(model), original_state, "Recovery frozen parameters and buffers")
    adapter_names = set(adapter_state_dict(model))
    require({name for name, parameter in model.named_parameters() if parameter.requires_grad} == adapter_names, "Trainable parameter set differs from adapter parameters")
    require(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in adapter_names), "Base parameter received a gradient")
    require(not any(module.training for module in model.modules()), "Model left eval mode; dropout would be active")


def selection_key(metrics):
    return (metrics["accuracy"], -metrics["cross_entropy_loss"])


def checkpoint_metadata(config, context, layout):
    sources, encoding = context["sources"], context["encoding"]
    return {"format_version": 1, "kind": CHECKPOINT_KIND, "base_kind": config.base_kind, "condition": config.condition, "config": asdict(config), "adapter_layout": layout,
            "source_artifacts": source_artifacts(sources.root, sources.paths, sources.hashes), "encoding_fingerprint": encoding_fingerprint(encoding),
            "split_sha256": sources.manifest["split_sha256"], "test_set_sha256": {name: item["sha256"] for name, item in sources.test_sets.items()},
            "architecture": sources.checkpoint["architecture"], "model_config": sources.checkpoint["model_config"], "classes": sources.checkpoint["classes"],
            "preprocessing": sources.checkpoint["preprocessing"]}


def restore_adapter(payload, config, root, device):
    """Rebuild the configured base and load adapter-only weights; validates storage, splits, and layout."""
    require(payload["format_version"] == 1 and payload["kind"] == CHECKPOINT_KIND, "Unsupported TinyBERT recovery checkpoint")
    config.validate()
    context = prepare_recovery_sources(root, config)
    sources = context["sources"]
    require(payload["split_sha256"] == sources.manifest["split_sha256"] and payload["test_set_sha256"] == {name: item["sha256"] for name, item in sources.test_sets.items()}, "Splits or test sets changed")
    require(payload["encoding_fingerprint"] == encoding_fingerprint(context["encoding"]), "Adapter was trained for a different encoding")
    require(payload["model_config"] == sources.checkpoint["model_config"] and payload["classes"] == sources.checkpoint["classes"], "Architecture or classes differ")
    state = build_recovery_base(config, context)
    model = build_tinybert(sources.checkpoint["model_config"], state)
    layout = attach_linear_lora(model, config.rank, config.alpha)
    require(layout == payload["adapter_layout"], "Adapter layout differs from the saved checkpoint")
    load_adapter_state(model, payload["adapter_state_dict"])
    return model.to(device).eval(), {**context, "base_state": state, "config": config}


def load_tinybert_recovered_model(checkpoint_path, project_root, *, device="cpu"):
    """Reconstruct the exact base the adapter was trained on, then load the adapter."""
    root = Path(project_root).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, context = restore_adapter(payload, TinyBertRecoveryConfig(**payload["config"]), root, device)
    sources = context["sources"]
    require(payload["source_artifacts"] == source_artifacts(root, sources.paths, sources.hashes), "Adapter checkpoint source provenance differs")
    return model, payload


def evaluate_tinybert_adapter_on_base(adapter_checkpoint, base_kind, pattern_id, project_root, *, source_sweep_results=None, device="cpu"):
    """Transfer evaluation: a saved adapter on another base of the same encoding; nothing is trained or written.

    The same base is also measured with the adapter output zeroed; that measurement
    must reproduce the sweep's record for the base. Results cover the selection
    split and each test set separately and must not select any setting.
    """
    root, device = Path(project_root).resolve(), choose_device(device)
    payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
    source = TinyBertRecoveryConfig(**payload["config"])
    config = replace(source, base_kind=base_kind, pattern_id=pattern_id, source_sweep_results=source_sweep_results or source.source_sweep_results,
                     condition=None if base_kind == "clean" else source.condition)
    model, context = restore_adapter(payload, config, root, device)
    loaders = make_loaders(context["sources"])
    with_adapter = evaluate_splits(model, loaders, device)
    trained = adapter_state_dict(model)
    load_adapter_state(model, {name: torch.zeros_like(value) if ".lora_B." in name else value for name, value in trained.items()})
    without_adapter = evaluate_splits(model, loaders, device)
    load_adapter_state(model, trained)
    assert_frozen_base(model, context["base_state"])
    record = context["record"]
    differences = {name: without_adapter[name]["num_correct"] - record["measurements"][name]["num_correct"] for name in loaders}
    require(all(value == 0 for value in differences.values()), f"Zero-adapter base differs from the sweep record: {differences}")
    return {"adapter_checkpoint_sha256": sha256_file(adapter_checkpoint), "adapter_selected_at": payload["epoch"], "adapter_base": {"base_kind": source.base_kind, "condition": source.condition, "pattern_id": source.pattern_id},
            "adapter_seed": source.seed, "evaluation_base": {"base_kind": base_kind, "condition": config.condition, "pattern_id": pattern_id, "fault_probability": record["probability"],
                                                              "actual_corrupted_cells": record["counts"]["actual_corrupted_cells"], "realized_fault_rate": record["counts"]["realized_fault_rate"]},
            "same_base": (source.base_kind, source.pattern_id) == (base_kind, pattern_id), "device": str(device), "with_adapter": with_adapter, "without_adapter": without_adapter,
            "accuracy_gain_percentage_points": {name: 100 * (with_adapter[name]["accuracy"] - without_adapter[name]["accuracy"]) for name in loaders},
            "reporting": "selection is for selection only; each test set separately; nothing here may choose a setting"}


def plot_recovery(history, metrics, references, test_names, output_dir):
    with tempfile.TemporaryDirectory(prefix="fault-lora-recovery-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
            points = [0.0] + [item["epoch"] for item in history]
            axes[0].plot(points, [100 * metrics["before"][SELECTION]["accuracy"]] + [100 * item[SELECTION]["accuracy"] for item in history], "o-", color="#174c78", label="Selection accuracy")
            axes[0].scatter([metrics["selected_epoch"]], [100 * metrics["after"][SELECTION]["accuracy"]], color="#b54d30", s=90, marker="*", zorder=4, label="Selected checkpoint")
            axes[0].set_xlabel("Epoch (0 = zero-output adapter)")
            axes[0].set_xticks(points)
            axes[0].set_title("Selection split chooses the checkpoint")
            axes[0].set_ylabel("Accuracy (%)")
            axes[0].legend(loc="lower right", fontsize=8)
            axes[0].grid(alpha=.2)
            for axis, name in zip(axes[1:], test_names):
                values = [100 * metrics[stage][name]["accuracy"] for stage in ("before", "after")]
                bars = axis.bar(["Base, zero adapter", "Base + selected LoRA"], values, width=.55, color=["#a76a2d", "#174c78"])
                axis.bar_label(bars, fmt="%.2f%%", padding=4)
                axis.axhline(100 * references[name]["clean"]["accuracy"], linestyle="--", color="#26734d", label="Original clean model")
                axis.axhline(100 * references[name]["clustered_no_faults"]["accuracy"], linestyle=":", color="#a76a2d", label="Clustered zero-fault reference")
                axis.set_ylim(40, 100)
                axis.set_title(f"{name} (after selection)")
                axis.legend(loc="lower left", fontsize=8)
                axis.grid(axis="y", alpha=.2)
            config = metrics["config"]
            fig.suptitle(f"TinyBERT / SST-2: rank-{config['rank']} LoRA recovery, {config['base_kind']} base, {config['condition'] or 'no condition'}, {config['pattern_id'] or 'no pattern'}")
            fig.text(.5, .02, "One base and one training seed; no uncertainty estimate. Base frozen; model in eval mode.", ha="center", fontsize=8)
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


def verify_tinybert_recovery(root, config):
    """Re-read a completed run: configuration, selection from history, checkpoints, inputs, and code."""
    root = Path(root).resolve()
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    read = lambda name: json.loads((results_dir / name).read_text())
    require(read("config.json") == asdict(config), "Saved configuration differs")
    history, metrics, manifest = read("history.json"), read("metrics.json"), read("manifest.json")
    verify_source_artifacts(root, manifest, "TinyBERT recovery")
    verify_source_code(root, results_dir, "TinyBERT recovery")
    expected_points = [round(epoch + (index + 1) / config.evaluations_per_epoch, 6) for epoch in range(config.epochs) for index in range(config.evaluations_per_epoch)]
    require([item["epoch"] for item in history] == expected_points, "Evaluation points differ from the configuration")
    best_key, best_epoch = selection_key(metrics["before"][SELECTION]), 0.0
    for item in history:
        require(item["frozen_base_verified"] and all(math.isfinite(value) for value in item["training_loss_windows"]), f"Invalid history entry at epoch {item['epoch']}")
        if selection_key(item[SELECTION]) > best_key:
            best_key, best_epoch = selection_key(item[SELECTION]), item["epoch"]
    require(best_epoch == metrics["selected_epoch"], "Selected epoch differs from a recomputation over the selection history")
    require(sha256_file(checkpoint_dir / "best.pt") == metrics["best_checkpoint_sha256"], "Selected checkpoint changed")
    best = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    require(best["epoch"] == best_epoch and best["selection_metrics"] == metrics["after"][SELECTION], "Selected checkpoint metadata differs")
    initial = torch.load(checkpoint_dir / "initial.pt", map_location="cpu", weights_only=True)
    require(all(not value.any() for name, value in initial["adapter_state_dict"].items() if ".lora_B." in name), "Initial adapter is not zero-output")
    for name, measurement in metrics["before"].items():
        require(measurement["num_correct"] == metrics["base_record_num_correct"][name], f"Before-LoRA measurement differs from the sweep record: {name}")
    for name in metrics["after"]:
        for stage in ("before", "after"):
            item = metrics[stage][name]
            require(item["accuracy"] == item["num_correct"] / item["num_examples"], f"Inconsistent {stage} {name}")
    for plot in manifest["plots"]:
        require(sha256_file(root / plot["path"]) == plot["sha256"], f"Plot changed: {plot['path']}")
    return {"status": "passed", "configuration_matches": True, "selection_recomputed_from_history": True, "evaluation_points": len(history), "selected_checkpoint_hash_verified": True,
            "initial_adapter_zero_output": True, "before_matches_sweep_record": True, "training_loss_finite": True, "input_artifact_hashes_verified": len(manifest["source_artifacts"]),
            "source_code_unchanged": True}


def run_tinybert_recovery(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Recovery directories already exist; choose a new run_id")
    context = prepare_recovery_sources(root, config)
    sources, record, references = context["sources"], context["record"], context["references"]
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    started = time.monotonic()
    history = []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_recovery.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        write_json(results_dir / "test_sets.json", sources.test_sets)
        if config.base_kind == "faulty":
            write_json(results_dir / "fault_counts.json", record["counts"])
        environment = {**environment_record(device, config.cpu_threads, ("matplotlib",)),
                       "reproducibility": "Seeded adapter initialization and data order; bitwise replay across hardware, thread counts, or library versions is not guaranteed."}
        write_json(results_dir / "environment.json", environment)
        original_state = build_recovery_base(config, context)
        model = build_tinybert(sources.checkpoint["model_config"], original_state)
        base_parameters = sum(parameter.numel() for parameter in model.parameters())
        probe = synthetic_token_batch(sources.checkpoint["model_config"], seed=0)
        with torch.no_grad():
            initial_logits = model(**probe).logits
        layout = attach_linear_lora(model, config.rank, config.alpha)
        require([item["name"] + ".weight" for item in layout] == linear_weight_names(build_tinybert(sources.checkpoint["model_config"])), "Adapters are not on exactly the nn.Linear layers")
        with torch.no_grad():
            require(torch.equal(initial_logits, model(**probe).logits), "Zero-output adapters changed initial logits")
        model.to(device)
        assert_frozen_base(model, original_state)
        initial_adapters = adapter_state_dict(model)
        adapter_parameters = sum(value.numel() for value in initial_adapters.values())
        write_json(results_dir / "adapter_layout.json", layout)

        train_sentences, train_labels = lora_training_data(sources, config)
        collate = make_collate(sources.tokenizer, sources.config.max_length)
        train_loader = DataLoader(SentenceDataset(train_sentences, train_labels), batch_size=config.batch_size, shuffle=True, collate_fn=collate,
                                  generator=torch.Generator().manual_seed(config.seed))
        loaders = make_loaders(sources)
        selection_loader = loaders[SELECTION]
        before = {SELECTION: evaluate_splits(model, {SELECTION: selection_loader}, device)[SELECTION]}
        require(before[SELECTION]["num_correct"] == record["measurements"][SELECTION]["num_correct"], "Initial selection accuracy differs from the sweep measurement of this base")
        metadata = checkpoint_metadata(config, context, layout)

        def save_adapter(filename, epoch, selection):
            save_torch(checkpoint_dir / filename, {**metadata, "epoch": epoch, "selection_metrics": selection, "adapter_state_dict": adapter_state_dict(model)})

        save_adapter("initial.pt", 0.0, before[SELECTION])
        save_adapter("best.pt", 0.0, before[SELECTION])
        best_key, best_epoch = selection_key(before[SELECTION]), 0.0
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        boundaries = sorted({math.ceil(steps_per_epoch * (index + 1) / config.evaluations_per_epoch) for index in range(config.evaluations_per_epoch)})
        criterion = nn.CrossEntropyLoss()
        epoch_seconds = []
        training_started = time.monotonic()
        for epoch in range(config.epochs):
            epoch_started, train_seconds = time.monotonic(), 0.0
            window, windows, segment = [], [], {"count": 0, "correct": 0, "loss": 0.0}
            interval_started = time.monotonic()
            for step, batch in enumerate(train_loader, start=1):
                model.eval()
                inputs = {name: value.to(device) for name, value in batch.items() if name != "labels"}
                targets = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(**inputs).logits
                loss = criterion(logits, targets)
                require(math.isfinite(loss.item()), f"Nonfinite training loss at epoch {epoch + 1}, step {step}; lr {config.learning_rate} diverged")
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                window.append(loss.item())
                segment["count"] += targets.numel()
                segment["correct"] += (logits.detach().argmax(1) == targets).sum().item()
                segment["loss"] += loss.item() * targets.numel()
                if len(window) == config.loss_window_steps:
                    windows.append(sum(window) / len(window))
                    window = []
                if step in boundaries:
                    if window:
                        windows.append(sum(window) / len(window))
                        window = []
                    train_seconds += time.monotonic() - interval_started
                    point = round(epoch + (boundaries.index(step) + 1) / config.evaluations_per_epoch, 6)
                    optimizer.zero_grad(set_to_none=True)
                    assert_frozen_base(model, original_state)
                    selection = evaluate_splits(model, {SELECTION: selection_loader}, device)[SELECTION]
                    if selection_key(selection) > best_key:
                        best_key, best_epoch = selection_key(selection), point
                        save_adapter("best.pt", point, selection)
                    history.append({"epoch": point, "step": epoch * steps_per_epoch + step, "learning_rate_after": optimizer.param_groups[0]["lr"],
                                    "training": {"num_examples": segment["count"], "accuracy": segment["correct"] / segment["count"], "cross_entropy_loss": segment["loss"] / segment["count"]},
                                    "training_loss_windows": windows, SELECTION: selection, "frozen_base_verified": True, "training_seconds_since_previous_point": train_seconds})
                    write_json(results_dir / "history.json", history)
                    write_json(results_dir / "status.json", {"status": "running", "evaluation_points": len(history), "selected_epoch": best_epoch})
                    print(json.dumps({"event": "selection_evaluation", "epoch": point, "training_loss": history[-1]["training"]["cross_entropy_loss"],
                                      "selection_accuracy": selection["accuracy"], "selected_epoch": best_epoch, "seconds": round(time.monotonic() - epoch_started, 1)}), flush=True)
                    windows, segment, train_seconds = [], {"count": 0, "correct": 0, "loss": 0.0}, 0.0
                    interval_started = time.monotonic()
            epoch_seconds.append({"epoch": epoch + 1, "wall_clock_seconds": time.monotonic() - epoch_started,
                                  "training_only_seconds": sum(item["training_seconds_since_previous_point"] for item in history[-config.evaluations_per_epoch:])})
        save_adapter("last.pt", float(config.epochs), history[-1][SELECTION])
        training_seconds = time.monotonic() - training_started

        selected = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
        load_adapter_state(model, selected["adapter_state_dict"])
        reloaded, _ = load_tinybert_recovered_model(checkpoint_dir / "best.pt", root, device=device)
        device_probe = {name: value.to(device) for name, value in probe.items()}
        with torch.no_grad():
            require(torch.equal(model(**device_probe).logits, reloaded(**device_probe).logits), "Saved adapter reload changed logits")
        del reloaded
        after = {SELECTION: evaluate_splits(model, {SELECTION: selection_loader}, device)[SELECTION]}
        require(after[SELECTION] == selected["selection_metrics"], "Selected selection metrics changed on replay")
        # Test sets are read by the model only after selection is final; each is reported separately.
        tests = {name: loader for name, loader in loaders.items() if name != SELECTION}
        after.update(evaluate_splits(model, tests, device))
        assert_frozen_base(model, original_state)
        load_adapter_state(model, initial_adapters)
        before.update(evaluate_splits(model, tests, device))
        for name in tests:
            require(before[name]["num_correct"] == record["measurements"][name]["num_correct"], f"Paired before-LoRA result differs from the sweep record: {name}")
        load_adapter_state(model, selected["adapter_state_dict"])
        assert_frozen_base(model, original_state)
        require(all(sha256_file(path) == sources.hashes[name] for name, path in sources.paths.items()), "Recovery changed an input artifact")
        compare_source_hashes(root, source_code, "Recovery source code changed during the run")
        all_windows = [value for item in history for value in item["training_loss_windows"]]
        metrics = {
            "run_id": config.run_id, "config": asdict(config), "base_kind": config.base_kind, "condition": config.condition, "pattern_id": config.pattern_id,
            "fault_seed": record["seed"], "fault_probability": record["probability"], "actual_corrupted_cells": record["counts"]["actual_corrupted_cells"],
            "realized_fault_rate": record["counts"]["realized_fault_rate"], "selected_epoch": best_epoch, "before": before, "after": after,
            "base_record_num_correct": {name: record["measurements"][name]["num_correct"] for name in before},
            "recovery": {name: {"accuracy_gain_percentage_points": 100 * (after[name]["accuracy"] - before[name]["accuracy"]),
                                "cross_entropy_reduction": before[name]["cross_entropy_loss"] - after[name]["cross_entropy_loss"],
                                "gap_to_clustered_percentage_points": 100 * (references[name]["clustered_no_faults"]["accuracy"] - after[name]["accuracy"]),
                                "gap_to_clean_percentage_points": 100 * (references[name]["clean"]["accuracy"] - after[name]["accuracy"])} for name in before},
            "training_loss": {"first_window": all_windows[0], "last_window": all_windows[-1], "min_window": min(all_windows), "max_window": max(all_windows), "all_finite": all(math.isfinite(value) for value in all_windows),
                              "half_epoch_means": [item["training"]["cross_entropy_loss"] for item in history],
                              "half_epoch_means_non_increasing": all(later <= earlier for earlier, later in zip([item["training"]["cross_entropy_loss"] for item in history], [item["training"]["cross_entropy_loss"] for item in history][1:]))},
            "timing": {"epochs": epoch_seconds, "training_and_selection_seconds": training_seconds, "steps_per_epoch": steps_per_epoch},
            "parameters": {"frozen_base": base_parameters, "trainable_adapter": adapter_parameters, "adapter_to_base_fraction": adapter_parameters / base_parameters,
                           "adapter_tensor_bytes": sum(value.numel() * value.element_size() for value in initial_adapters.values()), "best_serialized_bytes": (checkpoint_dir / "best.pt").stat().st_size},
            "best_checkpoint": str((checkpoint_dir / "best.pt").relative_to(root)), "best_checkpoint_sha256": sha256_file(checkpoint_dir / "best.pt"),
            "reporting": "Test sets reported separately and never pooled; selection is for checkpoint choice only.",
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(results_dir / "before.json", before)
        write_json(results_dir / "metrics.json", metrics)
        figures = plot_recovery(history, metrics, references, sources.test_names, results_dir)
        write_json(results_dir / "manifest.json", {
            **{key: value for key, value in metadata.items() if key != "model_config"}, "environment": environment,
            "selection_rule": "Highest selection-split accuracy, then lowest selection cross-entropy; exact ties keep the earlier point. Evaluated every half epoch; the epoch-0 zero-output adapter competes.",
            "test_policy": "Both test sets evaluated once each, after selection, for the selected adapter and for the same base with the zero-output adapter.",
            "frozen_state": "All base weights, biases, embeddings, and LayerNorm parameters unchanged; model kept in eval mode (dropout off) with adapter gradients enabled.",
            "schedule": "AdamW; cosine annealing per optimizer step over all epochs to zero; gradient-norm clipping.",
            "adapter_storage": "Reliable float32 adapters, unmerged residual branches; no adapter faults.",
            "limitations": "One base and one training seed per run.", "source_artifacts_preserved": True, "zero_output_identity_verified": True, "adapter_reload_logits_exact": True,
            "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures], "elapsed_seconds": time.monotonic() - started,
        })
        write_json(results_dir / "verification.json", verify_tinybert_recovery(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "selected_epoch": best_epoch, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "tinybert_recovery_complete", "run_id": config.run_id, "selected_epoch": best_epoch, "before": {name: before[name]["accuracy"] for name in before},
                          "after": {name: after[name]["accuracy"] for name in after}, "elapsed_seconds": metrics["elapsed_seconds"]}), flush=True)
        return metrics
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "evaluation_points": len(history), "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_tinybert_recovery(TinyBertRecoveryConfig(**json.loads(args.config.read_text())), project_root)
