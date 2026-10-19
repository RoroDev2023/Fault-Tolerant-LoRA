"""Step 11b: audit and fault sweep on the codebook-retrained ResNet-18 encodings.

Patterns are bound to an encoding fingerprint. Because the retrained encodings keep
the Step 4 indices and mappings, the same seed and probability must give the same
positions and levels; that identity is checked for every pattern before any
evaluation. The same faults are then applied to the Step 4 baseline, the retrained
baseline, and (at selected rates) the retrained ablation.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import re
import tempfile
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from fault_lora.data.cifar10 import cifar10_transforms
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import inspect_transitions, patterns_equal, require, require_state_equal
from fault_lora.evaluation.fault_sweep import SweepConfig, prepare_sources as prepare_sweep_sources, summarize_values, sweep_plan
from fault_lora.evaluation.source_hashes import compare_source_hashes
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern
from fault_lora.models.resnet import build_resnet18
from fault_lora.training.baseline import choose_device
from fault_lora.training.codebook_finetune import codebook_order, verify_same_cells


SPLITS = ("validation", "test")
MODELS = ("step4", "retrained", "ablation")
LABELS = {"step4": "Step 4 clustered baseline", "retrained": "Retrained baseline (codebook+BNstats)", "ablation": "Retrained ablation (codebook+BNaffine)"}
CSV_FIELDS = ["pattern_id", "probability", "seed", "actual_corrupted_cells", "realized_fault_rate", "model", "split", "num_examples", "num_correct", "accuracy", "cross_entropy_loss",
              "fault_damage_percentage_points"]
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
DIGEST = re.compile(r"[0-9a-f]{64}")


def transition_differences(first, second):
    """Per-tensor differences between two patterns' cells; empty when identical."""
    differences = []
    if first["tensor_order"] != second["tensor_order"]:
        return [{"tensor": None, "field": "tensor_order"}]
    for name in first["tensor_order"]:
        a, b = first["tensors"][name], second["tensors"][name]
        for key in ("shape", "num_levels"):
            if a[key] != b[key]:
                differences.append({"tensor": name, "field": key})
        for key in ("positions", "before_levels", "after_levels"):
            if a[key].dtype != b[key].dtype or not torch.equal(a[key], b[key]):
                differences.append({"tensor": name, "field": key, "counts": [a[key].numel(), b[key].numel()]})
    if first["seed"] != second["seed"] or first["fault_probability"] != second["fault_probability"]:
        differences.append({"tensor": None, "field": "seed_or_probability"})
    return differences


def rebind(pattern, encoding):
    """Regenerate a pattern for another encoding with the same seed; require identical cells."""
    other = generate_fault_pattern(encoding, pattern["fault_probability"], seed=pattern["seed"])
    differences = transition_differences(pattern, other)
    require(not differences, f"Pattern cells differ across encodings: {differences[:5]}")
    return other


def extended_plan(step7_plan, probabilities, first_seed, per_rate):
    """Step 7 entries unchanged, then new rates with consecutive new seeds in ascending rate order."""
    require(probabilities == sorted(set(probabilities)) and probabilities[0] > max(entry["probability"] for entry in step7_plan), "New rates must be sorted, unique, and above Step 7")
    used = {entry["seed"] for entry in step7_plan if entry["probability"] > 0}
    ordinal = max(int(entry["pattern_id"][4:6]) for entry in step7_plan)
    plan = [dict(entry) for entry in step7_plan]
    for offset, probability in enumerate(probabilities):
        for repeat in range(per_rate):
            seed = first_seed + offset * per_rate + repeat
            require(seed not in used and 0 <= seed < 2**32, f"Seed {seed} is reused or invalid")
            plan.append({"pattern_id": f"rate{ordinal + offset + 1:02d}-seed{seed}", "probability": float(probability), "seed": seed, "repeat": repeat})
    return plan


def aggregate(records, probabilities_by_model):
    """Per model and rate: accuracy and fault damage (that model at 0% minus at the rate)."""
    result = {}
    for model, probabilities in probabilities_by_model.items():
        zero = next(record for record in records if record["probability"] == 0)["models"][model]
        rates = []
        for probability in probabilities:
            group = [record for record in records if record["probability"] == probability and model in record["models"]]
            require(group, f"No measurements for {model} at {probability}")
            entry = {"probability": probability, "num_patterns": len(group), "seeds": [record["seed"] for record in group],
                     "realized_fault_rate": summarize_values([record["counts"]["realized_fault_rate"] for record in group]), "splits": {}}
            for split in SPLITS:
                entry["splits"][split] = {"accuracy": summarize_values([record["models"][model][split]["accuracy"] for record in group]),
                                          "cross_entropy_loss": summarize_values([record["models"][model][split]["cross_entropy_loss"] for record in group]),
                                          "fault_damage_percentage_points": summarize_values([100 * (zero[split]["accuracy"] - record["models"][model][split]["accuracy"]) for record in group])}
            rates.append(entry)
        result[model] = rates
    return result


def side_by_side(aggregates):
    """Damage table: Step 4 versus retrained baseline at every shared rate."""
    retrained = {entry["probability"]: entry for entry in aggregates["retrained"]}
    rows = []
    for entry in aggregates["step4"]:
        other = retrained[entry["probability"]]
        rows.append({"probability": entry["probability"], "num_patterns": entry["num_patterns"], **{
            f"{split}_{key}": {"step4": entry["splits"][split][key], "retrained": other["splits"][split][key],
                               "step4_minus_retrained_mean": entry["splits"][split][key]["mean"] - other["splits"][split][key]["mean"]}
            for split in SPLITS for key in ("accuracy", "fault_damage_percentage_points")}})
    return rows


def csv_rows(records):
    rows = []
    for record in records:
        for model, measurements in record["models"].items():
            for split in SPLITS:
                row = {"pattern_id": record["pattern_id"], "probability": record["probability"], "seed": record["seed"], **{key: record["counts"][key] for key in ("actual_corrupted_cells", "realized_fault_rate")},
                       "model": model, "split": split, **measurements[split], "fault_damage_percentage_points": record["fault_damage_percentage_points"][model][split]}
                rows.append({key: "" if row[key] is None else str(row[key]) for key in CSV_FIELDS})
    return rows


def load_baselines(root, config):
    """Audited Step 3/4 sources, the Step 7 sweep, and the two pinned Step 11a encodings."""
    sweep_config = SweepConfig(**json.loads((root / config.step7_results / "config.json").read_text()))
    step4, clean, metadata, split, clean_metrics, clustered_metrics, paths, hashes = prepare_sweep_sources(root, sweep_config)
    step7 = root / config.step7_results
    require(json.loads((step7 / "status.json").read_text())["status"] == "complete", "Step 7 sweep is incomplete")
    plan = json.loads((step7 / "plan.json").read_text())
    require(plan == sweep_plan(sweep_config), "Saved Step 7 plan changed")
    records = json.loads((step7 / "measurements.json").read_text())
    require([record["pattern_id"] for record in records] == [entry["pattern_id"] for entry in plan], "Step 7 records are incomplete")
    for record in records:
        path = root / record["pattern"]
        require(sha256_file(path) == record["pattern_sha256"], f"Step 7 pattern changed: {record['pattern_id']}")
        paths[f"step7_pattern_{record['pattern_id']}"], hashes[f"step7_pattern_{record['pattern_id']}"] = path, record["pattern_sha256"]
    finetune = root / config.finetune_results
    require(json.loads((finetune / "status.json").read_text())["status"] == "complete" and json.loads((finetune / "verification.json").read_text())["status"] == "passed",
            "Step 11a run is incomplete or unverified")
    saved = json.loads((finetune / "encodings.json").read_text())
    encodings = {}
    for model, variant, fingerprint in (("retrained", config.baseline_variant, config.baseline_fingerprint), ("ablation", config.ablation_variant, config.ablation_fingerprint)):
        item = saved[variant]
        path = root / item["encoding"]
        require(sha256_file(path) == item["encoding_sha256"], f"Step 11a encoding changed: {variant}")
        encoding = torch.load(path, map_location="cpu", weights_only=True)
        require(encoding_fingerprint(encoding) == item["fingerprint"] == fingerprint, f"Step 11a fingerprint differs from the pinned value: {variant}")
        require(encoding["source_encoding_sha256"] == hashes["encoding"], f"Step 11a encoding was not derived from this Step 4 encoding: {variant}")
        verify_same_cells(encoding, step4)
        require(all(report["strictly_increasing"] for report in codebook_order({name: record["codebook"] for name, record in encoding["clustered_tensors"].items()}).values()),
                f"Step 11a codebooks are not strictly increasing: {variant}")
        dense = torch.load(root / item["dense"], map_location="cpu", weights_only=True)
        require(sha256_file(root / item["dense"]) == item["dense_sha256"], f"Step 11a dense reconstruction changed: {variant}")
        require_state_equal(decode_state(encoding), dense["model_state_dict"], f"Step 11a encoding reconstruction ({variant})")
        encodings[model] = encoding
        paths[f"{model}_encoding"], hashes[f"{model}_encoding"] = path, item["encoding_sha256"]
        paths[f"{model}_dense"], hashes[f"{model}_dense"] = root / item["dense"], item["dense_sha256"]
    paths["step11a_paired_evaluation"] = finetune / "paired_evaluation.json"
    hashes["step11a_paired_evaluation"] = sha256_file(paths["step11a_paired_evaluation"])
    step11a = json.loads(paths["step11a_paired_evaluation"].read_text())["models"]
    return {"step4": step4, "retrained": encodings["retrained"], "ablation": encodings["ablation"], "clean_state": {name: value.detach().cpu().clone() for name, value in clean.state_dict().items()},
            "metadata": metadata, "split": split, "clean_metrics": clean_metrics, "step7_plan": plan, "step7_records": records,
            "step11a_metrics": {"step4": step11a["clustered_step4"], "retrained": step11a[config.baseline_variant], "ablation": step11a[config.ablation_variant]},
            "paths": paths, "hashes": hashes}


def write_common(root, results_dir, config, script, device):
    files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / script, root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
    source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "source_sha256.json", source_code)
    environment = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "cpu_threads": config.cpu_threads,
                   "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "matplotlib")}}
    write_json(results_dir / "environment.json", environment)
    return source_code, environment


@dataclass
class RetrainedAuditConfig:
    run_id: str = "resnet18-cifar10-retrained-fault-checks-seed42"
    step7_results: str = "results/resnet18-cifar10-fault-sweep-seed42"
    finetune_results: str = "results/resnet18-cifar10-codebook-finetune-k16-seed42"
    baseline_variant: str = "codebook-bnstats"
    baseline_fingerprint: str = "18e55d67952361ebb6c47a4dc29887d17df2f367e2cf3fc22c116b72b9075220"
    ablation_variant: str = "codebook-bnaffine"
    ablation_fingerprint: str = "e088d98014a97c630f99bbee2d93a4766087955506a5ac0393e042ddd8c6f0ce"
    audit_probability: float = .01
    audit_seed: int = 42
    cpu_threads: int = 4
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        for name in ("run_id", "baseline_variant", "ablation_variant"):
            if not isinstance(getattr(self, name), str) or not NAME.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a simple name")
        for name in ("baseline_fingerprint", "ablation_fingerprint"):
            if not isinstance(getattr(self, name), str) or not DIGEST.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a full SHA-256 fingerprint")
        if isinstance(self.audit_probability, bool) or not isinstance(self.audit_probability, Real) or not 0 < self.audit_probability < 1:
            raise ValueError("audit_probability must be in (0, 1)")
        if isinstance(self.audit_seed, bool) or not isinstance(self.audit_seed, int) or not 0 <= self.audit_seed < 2**32 - 1:
            raise ValueError("Invalid audit_seed")
        if isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int) or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")


def run_retrained_fault_checks(config, project_root):
    """The eight Step 6 checks, on the retrained baseline encoding."""
    config.validate()
    root = Path(project_root)
    torch.set_num_threads(config.cpu_threads)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Audit directories already exist; choose a new run_id")
    sources = load_baselines(root, config)
    encoding, step4 = sources["retrained"], sources["step4"]
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    started, checks = time.monotonic(), {}
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})

    def passed(name, details):
        checks[name] = {"passed": True, **details}
        print(json.dumps({"event": "retrained_audit_check_passed", "check": name}), flush=True)

    try:
        source_code, _ = write_common(root, results_dir, config, "scripts/check_retrained_faults.py", torch.device("cpu"))
        fingerprint = encoding_fingerprint(encoding)
        baseline = decode_state(encoding)
        passed("source_integrity", {"encoding_fingerprint": fingerprint, "step4_fingerprint": encoding_fingerprint(step4), "cells_identical_to_step4": len(encoding["clustered_tensors"]),
                                    "dense_reconstruction_exact": True, "codebooks_strictly_increasing": True, "state_tensor_count": len(baseline)})
        zero = generate_fault_pattern(encoding, 0, seed=config.audit_seed)
        zero_state = apply_fault_pattern(encoding, zero)
        require_state_equal(zero_state, baseline, "Zero-fault identity")
        zero_details = inspect_transitions(encoding, zero, zero_state)
        require(zero_details["actual_corrupted_cells"] == 0, "Zero pattern selected cells")
        model = build_resnet18(num_classes=sources["metadata"]["num_classes"], pretrained=False).eval().requires_grad_(False)
        inputs = torch.randn(2, *sources["metadata"]["preprocessing"]["input_size"], generator=torch.Generator().manual_seed(0))
        model.load_state_dict(baseline, strict=True)
        with torch.inference_mode():
            baseline_logits = model(inputs)
        model.load_state_dict(zero_state, strict=True)
        with torch.inference_mode():
            zero_logits = model(inputs)
        require(torch.equal(zero_logits, baseline_logits) and bool(torch.isfinite(zero_logits).all()), "Zero-fault logits differ")
        passed("zero_fault_identity", {**zero_details, "synthetic_input_seed": 0, "logits_exactly_equal": True})
        del zero, zero_state, model
        # Saved pattern: same probability and seed as Step 5; it must carry Step 5's exact cells.
        pattern_path = checkpoint_dir / "pattern.pt"
        save_torch(pattern_path, generate_fault_pattern(encoding, config.audit_probability, seed=config.audit_seed))
        pattern = torch.load(pattern_path, map_location="cpu", weights_only=True)
        step5 = torch.load(sources["paths"]["pattern"], map_location="cpu", weights_only=True)
        require(step5["seed"] == config.audit_seed and step5["fault_probability"] == config.audit_probability, "Audit seed or probability differs from Step 5")
        differences = transition_differences(pattern, step5)
        require(not differences, f"Saved audit pattern differs from Step 5: {differences[:5]}")
        saved_state = apply_fault_pattern(encoding, pattern)
        saved_details = inspect_transitions(encoding, pattern, saved_state)
        summary = summarize_fault_pattern(encoding, pattern)
        for key in ("eligible_cells", "actual_corrupted_cells", "decoded_weight_changes", "lower_boundary_transitions", "upper_boundary_transitions"):
            require(saved_details[key] == summary[key], f"Independent count differs: {key}")
        require(summary["decoded_weight_changes"] == summary["actual_corrupted_cells"], "A transition left a weight unchanged despite strictly increasing codebooks")
        write_json(results_dir / "counts.json", summary)
        passed("saved_pattern_and_checkpoint", {**saved_details, "pattern": str(pattern_path.relative_to(root)), "pattern_sha256": sha256_file(pattern_path), "identical_cells_to_step5_pattern": True})
        replay = apply_fault_pattern(encoding, pattern)
        require_state_equal(replay, saved_state, "Repeated application")
        require_state_equal(decode_state(encoding), baseline, "Baseline after replay")
        passed("replay_without_accumulation", {"two_applications_exactly_equal": True, "baseline_state_preserved": True})
        del replay, saved_state
        regenerated = generate_fault_pattern(encoding, config.audit_probability, seed=config.audit_seed)
        require(patterns_equal(regenerated, pattern), "Same seed did not reproduce saved transitions")
        passed("same_seed_reproducibility", {"seed": config.audit_seed, "all_positions_and_transitions_equal": True})
        alternate = generate_fault_pattern(encoding, config.audit_probability, seed=config.audit_seed + 1)
        require(not patterns_equal(alternate, pattern), "Different seed produced the identical pattern")
        alternate_details = inspect_transitions(encoding, alternate, apply_fault_pattern(encoding, alternate))
        passed("different_seed_valid_pattern", {"seed": config.audit_seed + 1, "differs_from_saved_pattern": True, **alternate_details})
        del alternate, regenerated
        full = generate_fault_pattern(encoding, 1, seed=config.audit_seed)
        full_details = inspect_transitions(encoding, full, apply_fault_pattern(encoding, full))
        require(full_details["actual_corrupted_cells"] == full_details["eligible_cells"], "Full selection omitted eligible cells")
        require(full_details["lower_boundary_transitions"] > 0 and full_details["upper_boundary_transitions"] > 0, "Full-model audit did not exercise both boundaries")
        passed("all_cell_selection_and_boundaries", full_details)
        del full
        require(encoding_fingerprint(encoding) == fingerprint, "In-memory encoding changed")
        require_state_equal(decode_state(encoding), baseline, "Final baseline preservation")
        require(all(sha256_file(path) == sources["hashes"][name] for name, path in sources["paths"].items()), "A source artifact changed")
        accepted = compare_source_hashes(root, source_code, "Source code changed during the audit")
        passed("source_preservation", {"in_memory_fingerprint_unchanged": True, "all_input_file_hashes_unchanged": True, "accepted_source_hash_changes": accepted})
        report = {"status": "passed", "run_id": config.run_id, "checks": checks, "encoding_fingerprint": fingerprint,
                  "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": sources["hashes"][name]} for name, path in sources["paths"].items()},
                  "performance_evaluated": False, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "report.json", report)
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "retrained_audit_complete", "passed_checks": len(checks), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        return report
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(results_dir / "report.json", {"status": status, "checks": checks, "error": repr(error)})
        write_json(results_dir / "status.json", {"status": status, "error": repr(error)})
        raise


@dataclass
class RetrainedSweepConfig:
    run_id: str = "resnet18-cifar10-retrained-fault-sweep-seed42"
    step7_results: str = "results/resnet18-cifar10-fault-sweep-seed42"
    finetune_results: str = "results/resnet18-cifar10-codebook-finetune-k16-seed42"
    audit_results: str = "results/resnet18-cifar10-retrained-fault-checks-seed42"
    baseline_variant: str = "codebook-bnstats"
    baseline_fingerprint: str = "18e55d67952361ebb6c47a4dc29887d17df2f367e2cf3fc22c116b72b9075220"
    ablation_variant: str = "codebook-bnaffine"
    ablation_fingerprint: str = "e088d98014a97c630f99bbee2d93a4766087955506a5ac0393e042ddd8c6f0ce"
    extra_probabilities: list = field(default_factory=lambda: [.2, .3, .5])
    extra_first_seed: int = 72
    patterns_per_rate: int = 5
    ablation_probabilities: list = field(default_factory=lambda: [0, .01, .05, .1, .2, .5])
    evaluation_batch_size: int = 256
    device: str = "mps"
    cpu_threads: int = 4
    num_workers: int = 2
    data_root: str = "data"
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        RetrainedAuditConfig(run_id=self.run_id, baseline_variant=self.baseline_variant, baseline_fingerprint=self.baseline_fingerprint, ablation_variant=self.ablation_variant,
                             ablation_fingerprint=self.ablation_fingerprint, cpu_threads=self.cpu_threads).validate()
        for name in ("extra_probabilities", "ablation_probabilities"):
            values = getattr(self, name)
            if not isinstance(values, list) or not values or any(isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1 for value in values) or values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted, unique probabilities")
        if self.ablation_probabilities[0] != 0:
            raise ValueError("ablation_probabilities must include the zero-fault reference")
        for name, minimum in (("extra_first_seed", 0), ("patterns_per_rate", 2), ("evaluation_batch_size", 1), ("num_workers", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def plot_retrained_sweep(aggregates, clean, output_dir, split="test"):
    """Accuracy versus actual fault rate for all three encodings: full range and 0-10% zoom."""
    colors = {"step4": "#eb6834", "retrained": "#2a78d6", "ablation": "#1baf7a"}
    markers = {"step4": "o", "retrained": "D", "ablation": "s"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    with tempfile.TemporaryDirectory(prefix="fault-lora-retrained-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor="#fcfcfb")
            for axis, limit in zip(axes, (None, .1)):
                axis.set_facecolor("#fcfcfb")
                axis.axhline(100 * clean[split]["accuracy"], color=muted, linestyle="--", linewidth=1.2, label=f"Clean model: {100 * clean[split]['accuracy']:.2f}%", zorder=2)
                for model in MODELS:
                    entries = [entry for entry in aggregates[model] if limit is None or entry["probability"] <= limit + 1e-12]
                    xs = [100 * entry["realized_fault_rate"]["mean"] for entry in entries]
                    values = [entry["splits"][split]["accuracy"] for entry in entries]
                    axis.axhline(100 * values[0]["mean"], color=colors[model], linestyle=":", linewidth=1, zorder=2)
                    axis.errorbar(xs, [100 * value["mean"] for value in values], yerr=[100 * (value["sample_std"] or 0) for value in values], color=colors[model], marker=markers[model],
                                  markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=1.5, linewidth=2, capsize=3, elinewidth=1.2,
                                  label=f"{LABELS[model]}; 0%: {100 * values[0]['mean']:.2f}%", zorder=4)
                axis.set_title(f"{split.capitalize()} accuracy, 0–{'50' if limit is None else '10'}% faults", fontsize=11, color=ink)
                axis.set_xlabel("Actual corrupted cells / eligible cells (%)", color=muted)
                axis.set_ylabel("Classification accuracy (%)", color=muted)
                axis.grid(color=grid, linewidth=.8)
                axis.set_axisbelow(True)
                axis.tick_params(colors=muted)
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    axis.spines[side].set_color(grid)
                axis.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=ink)
            fig.suptitle("ResNet-18 / CIFAR-10: identical adjacent-level faults on the Step 4 and codebook-retrained encodings", fontsize=12.5, color=ink)
            fig.text(.5, .015, "Mean ± 1 sample SD across 5 static patterns per nonzero rate; identical cells faulted in every encoding. Dotted lines: each encoding at 0% faults. No LoRA.",
                     ha="center", fontsize=8, color=muted)
            fig.tight_layout(rect=(0, .05, 1, .95))
            paths = [Path(output_dir) / f"retrained_fault_sweep.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def run_retrained_fault_sweep(config, project_root):
    config.validate()
    root = Path(project_root)
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Sweep directories already exist; choose a new run_id")
    audit = root / config.audit_results
    report = json.loads((audit / "report.json").read_text())
    require(json.loads((audit / "status.json").read_text())["status"] == "complete" and report["status"] == "passed" and len(report["checks"]) == 8
            and all(item["passed"] for item in report["checks"].values()) and report["encoding_fingerprint"] == config.baseline_fingerprint, "Retrained-encoding audit did not pass")
    sources = load_baselines(root, config)
    sources["paths"]["audit_report"], sources["hashes"]["audit_report"] = audit / "report.json", sha256_file(audit / "report.json")
    plan = extended_plan(sources["step7_plan"], config.extra_probabilities, config.extra_first_seed, config.patterns_per_rate)
    step7_records = {record["pattern_id"]: record for record in sources["step7_records"]}
    probabilities = sorted({entry["probability"] for entry in plan})
    require(set(config.ablation_probabilities) <= set(probabilities), "Ablation rates must be sweep rates")
    (results_dir / "patterns").mkdir(parents=True)
    (checkpoint_dir / "patterns").mkdir(parents=True)
    started, records = time.monotonic(), []
    write_json(results_dir / "plan.json", plan)
    write_json(results_dir / "status.json", {"status": "running", "stage": "patterns", "planned_patterns": len(plan), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        source_code, environment = write_common(root, results_dir, config, "scripts/run_retrained_fault_sweep.py", device)
        write_json(results_dir / "splits.json", sources["split"])
        # Stage 1: every pattern, and its identity across encodings, before any evaluation.
        identity = []
        for entry in plan:
            pattern = generate_fault_pattern(sources["retrained"], entry["probability"], seed=entry["seed"])
            path = checkpoint_dir / "patterns" / f"{entry['pattern_id']}.pt"
            save_torch(path, pattern)
            pattern = torch.load(path, map_location="cpu", weights_only=True)
            item = {"pattern_id": entry["pattern_id"], "probability": entry["probability"], "seed": entry["seed"], "pattern": str(path.relative_to(root)), "pattern_sha256": sha256_file(path)}
            if entry["pattern_id"] in step7_records:
                old = torch.load(root / step7_records[entry["pattern_id"]]["pattern"], map_location="cpu", weights_only=True)
                differences = transition_differences(pattern, old)
                item["compared_with"] = "saved Step 7 pattern"
            else:
                differences = transition_differences(pattern, generate_fault_pattern(sources["step4"], entry["probability"], seed=entry["seed"]))
                item["compared_with"] = "Step 4 encoding, same seed (regenerated)"
            if entry["probability"] in config.ablation_probabilities:
                differences += [{**difference, "encoding": "ablation"} for difference in transition_differences(pattern, generate_fault_pattern(sources["ablation"], entry["probability"], seed=entry["seed"]))]
                item["ablation_compared"] = True
            item["identical"] = not differences
            item["differences"] = differences
            item["counts"] = {key: value for key, value in summarize_fault_pattern(sources["retrained"], pattern).items() if key != "layers"}
            identity.append(item)
            write_json(results_dir / "pattern_identity.json", identity)
            print(json.dumps({"event": "pattern_identity", "pattern_id": entry["pattern_id"], "identical": item["identical"], "corrupted": item["counts"]["actual_corrupted_cells"]}), flush=True)
            del pattern
        mismatched = [item["pattern_id"] for item in identity if not item["identical"]]
        if mismatched:
            write_json(results_dir / "status.json", {"status": "stopped_pattern_mismatch", "mismatched_patterns": mismatched})
            print(json.dumps({"event": "pattern_mismatch", "patterns": mismatched}), flush=True)
            return {"status": "stopped_pattern_mismatch", "mismatched_patterns": mismatched}
        # Stage 2: identical faults on every encoding; validation and test for each.
        _, transform = cifar10_transforms(sources["metadata"]["preprocessing"]["input_size"][-1])
        train_data = CIFAR10(root / config.data_root, train=True, transform=transform, download=False)
        test_data = CIFAR10(root / config.data_root, train=False, transform=transform, download=False)
        require(train_data.classes == test_data.classes == sources["metadata"]["classes"] and len(train_data) == 50000 and len(test_data) == 10000, "CIFAR-10 class order or size changed")
        loaders = {"validation": DataLoader(Subset(train_data, sources["split"]["validation"]), batch_size=config.evaluation_batch_size, shuffle=False, num_workers=config.num_workers),
                   "test": DataLoader(Subset(test_data, sources["split"]["test"]["indices"]), batch_size=config.evaluation_batch_size, shuffle=False, num_workers=config.num_workers)}
        model = build_resnet18(sources["metadata"]["num_classes"], pretrained=False).to(device).eval().requires_grad_(False)

        def measure(state):
            model.load_state_dict(state, strict=True)
            return {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}

        clean = measure(sources["clean_state"])
        require(all(clean[split]["num_correct"] == sources["clean_metrics"][split]["num_correct"] for split in SPLITS), "Clean replay differs from Step 3")
        write_json(results_dir / "references.json", {"clean": clean})
        zero = {}
        for entry, item in zip(plan, identity):
            pattern = torch.load(root / item["pattern"], map_location="cpu", weights_only=True)
            require(sha256_file(root / item["pattern"]) == item["pattern_sha256"], f"Saved pattern changed: {entry['pattern_id']}")
            bound = {"retrained": pattern,
                     "step4": torch.load(root / step7_records[entry["pattern_id"]]["pattern"], map_location="cpu", weights_only=True) if entry["pattern_id"] in step7_records else rebind(pattern, sources["step4"])}
            if entry["probability"] in config.ablation_probabilities:
                bound["ablation"] = rebind(pattern, sources["ablation"])
            measurements = {name: measure(apply_fault_pattern(sources[name], bound[name])) for name in MODELS if name in bound}
            if entry["pattern_id"] in step7_records:
                require(all(measurements["step4"][split]["num_correct"] == step7_records[entry["pattern_id"]]["measurements"][split]["num_correct"] for split in SPLITS),
                        f"Step 4 replay differs from Step 7: {entry['pattern_id']}")
            if entry["probability"] == 0:
                zero = measurements
                for name in MODELS:
                    require(all(zero[name][split]["num_correct"] == sources["step11a_metrics"][name][split]["num_correct"] for split in SPLITS), f"Zero-fault {name} differs from Step 11a")
            record = {**entry, "counts": item["counts"], "pattern": item["pattern"], "pattern_sha256": item["pattern_sha256"], "models": measurements,
                      "fault_damage_percentage_points": {name: {split: 100 * (zero[name][split]["accuracy"] - value[split]["accuracy"]) for split in SPLITS} for name, value in measurements.items()}}
            records.append(record)
            write_json(results_dir / "patterns" / f"{entry['pattern_id']}.json", record)
            write_json(results_dir / "measurements.json", records)
            write_json(results_dir / "status.json", {"status": "running", "stage": "evaluation", "completed_patterns": len(records), "planned_patterns": len(plan)})
            print(json.dumps({"event": "pattern_evaluation_complete", "pattern_id": entry["pattern_id"], **{name: value["test"]["accuracy"] for name, value in measurements.items()}}), flush=True)
            del pattern, bound
        by_model = {"step4": probabilities, "retrained": probabilities, "ablation": config.ablation_probabilities}
        aggregates = aggregate(records, by_model)
        write_json(results_dir / "aggregate.json", {"models": aggregates, "side_by_side": side_by_side(aggregates),
                                                     "definitions": {"fault_damage": "That encoding's zero-fault accuracy minus its accuracy under the pattern, in percentage points; mean and sample SD (ddof=1) across patterns.",
                                                                     "identical_faults": "Every encoding receives the same cell positions and level transitions."}})
        with (results_dir / "measurements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows(records))
        figures = plot_retrained_sweep(aggregates, clean, results_dir)
        require(all(sha256_file(path) == sources["hashes"][name] for name, path in sources["paths"].items()), "A source artifact changed during the sweep")
        accepted = compare_source_hashes(root, source_code, "Source code changed during the sweep")
        manifest = {"run_id": config.run_id, "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": sources["hashes"][name]} for name, path in sources["paths"].items()},
                    "fingerprints": {name: encoding_fingerprint(sources[name]) for name in MODELS},
                    "pattern_policy": "Patterns saved against the retrained baseline; Step 7 patterns reused for the Step 4 baseline; other encodings regenerated with the same seed and required to have identical cells.",
                    "seed_rule": "Step 7 seeds 42-71 unchanged; new rates take consecutive seeds from extra_first_seed in ascending rate order.",
                    "selection_policy": "Rates, seeds, and encodings fixed in the configuration; no training or tuning.", "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures],
                    "accepted_source_hash_changes": accepted, "environment": environment, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "verification.json", verify_retrained_sweep(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "completed_patterns": len(records), "planned_patterns": len(plan), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "retrained_fault_sweep_complete", "run_id": config.run_id, "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return {"status": "complete", "aggregate": aggregates}
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_patterns": len(records), "error": repr(error)})
        raise


def verify_retrained_sweep(root, config):
    root = Path(root)
    results_dir = root / config.results_root / config.run_id
    plan = json.loads((results_dir / "plan.json").read_text())
    records = json.loads((results_dir / "measurements.json").read_text())
    identity = json.loads((results_dir / "pattern_identity.json").read_text())
    require([record["pattern_id"] for record in records] == [entry["pattern_id"] for entry in plan] == [item["pattern_id"] for item in identity], "Records are incomplete or reordered")
    require(all(item["identical"] and not item["differences"] for item in identity), "A pattern differed across encodings")
    for record in records:
        require(record == json.loads((results_dir / "patterns" / f"{record['pattern_id']}.json").read_text()), f"Individual and combined records differ: {record['pattern_id']}")
        require(sha256_file(root / record["pattern"]) == record["pattern_sha256"], f"Saved pattern changed: {record['pattern_id']}")
        require(set(record["models"]) == ({"step4", "retrained", "ablation"} if record["probability"] in config.ablation_probabilities else {"step4", "retrained"}), f"Wrong model set: {record['pattern_id']}")
    probabilities = sorted({entry["probability"] for entry in plan})
    saved = json.loads((results_dir / "aggregate.json").read_text())
    recomputed = aggregate(records, {"step4": probabilities, "retrained": probabilities, "ablation": config.ablation_probabilities})
    require(saved["models"] == json.loads(json.dumps(recomputed)) and saved["side_by_side"] == json.loads(json.dumps(side_by_side(recomputed))), "Aggregate differs from its recomputation")
    with (results_dir / "measurements.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == csv_rows(records), "CSV and JSON measurements disagree")
    manifest = json.loads((results_dir / "manifest.json").read_text())
    for name, item in manifest["source_artifacts"].items():
        require(sha256_file(root / item["path"]) == item["sha256"], f"Input artifact changed: {name}")
    for item in manifest["plots"]:
        require(sha256_file(root / item["path"]) == item["sha256"], "Plot changed")
    accepted = compare_source_hashes(root, json.loads((results_dir / "source_sha256.json").read_text()), "Source code changed after the sweep")
    return {"status": "passed", "patterns_verified": len(records), "all_patterns_identical_across_encodings": True,
            "step7_patterns_identical": sum(item["compared_with"] == "saved Step 7 pattern" for item in identity), "aggregate_recomputed": True, "csv_matches_json": True,
            "source_artifacts_verified": len(manifest["source_artifacts"]), "accepted_source_hash_changes": accepted}


def main_audit(project_root):
    parser = argparse.ArgumentParser(description="Step 11b: eight-check audit of the retrained encoding.")
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_retrained_fault_checks.json"))
    args = parser.parse_args()
    run_retrained_fault_checks(RetrainedAuditConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)


def main_sweep(project_root):
    parser = argparse.ArgumentParser(description="Step 11b: identical faults on the Step 4, retrained, and ablation encodings.")
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_retrained_fault_sweep.json"))
    args = parser.parse_args()
    run_retrained_fault_sweep(RetrainedSweepConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
