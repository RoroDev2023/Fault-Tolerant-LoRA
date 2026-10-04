"""Repeated static-pattern experiments against the audited clustered baseline."""

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
import statistics
import tempfile
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from fault_lora.data.cifar10 import cifar10_transforms
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import ClusteredConfig, save_torch, sha256_file, validate_source as validate_clean_source, write_json
from fault_lora.evaluation.fault_checks import require, require_state_equal
from fault_lora.evaluation.fault_simulation import FaultConfig, validate_source as validate_clustered_source
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern, validate_parameters
from fault_lora.models.resnet import build_resnet18


@dataclass
class SweepConfig:
    run_id: str = "resnet18-cifar10-fault-sweep-seed42"
    probabilities: list = field(default_factory=lambda: [0, .001, .005, .01, .02, .05, .1])
    patterns_per_rate: int = 5
    seed: int = 42
    device: str = "mps"
    cpu_threads: int = 4
    num_workers: int = 2
    source_audit_results: str = "results/resnet18-cifar10-fault-checks-seed42"
    data_root: str = "data"
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        validate_parameters(0, self.seed)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if not self.probabilities or any(isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or not 0 <= p <= 1 for p in self.probabilities):
            raise ValueError("probabilities must contain finite values in [0, 1]")
        if self.probabilities[0] != 0 or self.probabilities != sorted(set(self.probabilities)) or len(self.probabilities) < 2:
            raise ValueError("probabilities must be sorted, unique, and start with zero plus at least one nonzero rate")
        for name, minimum in (("patterns_per_rate", 2), ("cpu_threads", 1), ("num_workers", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (len(self.probabilities) - 1) * self.patterns_per_rate > 2**32:
            raise ValueError("Sweep exceeds the unique seed space")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def sweep_plan(config):
    config.validate()
    plan = [{"pattern_id": "rate00-zero", "probability": 0.0, "seed": config.seed, "repeat": 0}]
    for ordinal, probability in enumerate(config.probabilities[1:]):
        for repeat in range(config.patterns_per_rate):
            seed = (config.seed + ordinal * config.patterns_per_rate + repeat) % 2**32
            plan.append({"pattern_id": f"rate{ordinal + 1:02d}-seed{seed}", "probability": float(probability), "seed": seed, "repeat": repeat})
    return plan


def summarize_values(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def aggregate_measurements(records, config):
    """Reject incomplete/duplicate measurements before estimating variability."""
    plan = sweep_plan(config)
    expected = {entry["pattern_id"]: entry for entry in plan}
    require(len(records) == len(plan) and len({record["pattern_id"] for record in records}) == len(plan), "Sweep measurements are incomplete or duplicated")
    for record in records:
        entry = expected.get(record["pattern_id"])
        require(entry is not None and record["probability"] == entry["probability"] and record["seed"] == entry["seed"], "Measurement differs from the predeclared plan")
    rates = []
    for probability in config.probabilities:
        group = [record for record in records if record["probability"] == probability]
        require(len({record["counts"]["eligible_cells"] for record in group}) == 1, "Fault-rate denominators differ")
        require(all(set(record["measurements"]) == {"validation", "test"} for record in group), "Expected both validation and test measurements")
        rate = {"probability": float(probability), "num_patterns": len(group), "seeds": [record["seed"] for record in group],
                "eligible_cells": group[0]["counts"]["eligible_cells"],
                "actual_corrupted_cells": summarize_values([record["counts"]["actual_corrupted_cells"] for record in group]),
                "realized_fault_rate": summarize_values([record["counts"]["realized_fault_rate"] for record in group]), "splits": {}}
        for split in ("validation", "test"):
            require(len({record["measurements"][split]["num_examples"] for record in group}) == 1, "Evaluation example counts differ")
            rate["splits"][split] = {"num_examples": group[0]["measurements"][split]["num_examples"],
                **{key: summarize_values([record["measurements"][split][key] for record in group]) for key in ("accuracy", "cross_entropy_loss")},
                **{key: summarize_values([record["fault_loss"][split][key] for record in group]) for key in ("accuracy_drop_percentage_points", "cross_entropy_increase")}}
        rates.append(rate)
    return {"rates": rates, "variability": "Sample standard deviation across independent static patterns (ddof=1); not a confidence interval or variation over training seeds. Zero rate is one deterministic reference; sample_std is null."}


def plot_sweep(records, aggregate, references, output_dir):
    """Export a standalone scientific figure with actual fault-rate x values."""
    with tempfile.TemporaryDirectory(prefix="fault-lora-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
            visible = [100 * record["measurements"][split]["accuracy"] for record in records for split in ("validation", "test")]
            visible += [100 * references[split]["clean"]["accuracy"] for split in references]
            for rate in aggregate["rates"]:
                for split in ("validation", "test"):
                    values = rate["splits"][split]["accuracy"]
                    visible.extend([100 * (values["mean"] - (values["sample_std"] or 0)), 100 * (values["mean"] + (values["sample_std"] or 0))])
            lower, upper = max(0, 5 * math.floor((min(visible) - 3) / 5)), min(100, 5 * math.ceil((max(visible) + 3) / 5))
            for axis, split in zip(axes, ("validation", "test")):
                xs = [100 * record["counts"]["realized_fault_rate"] for record in records]
                ys = [100 * record["measurements"][split]["accuracy"] for record in records]
                axis.scatter(xs, ys, s=24, color="#527eaa", alpha=.55, label="Individual static patterns", zorder=3)
                xmean = [100 * rate["realized_fault_rate"]["mean"] for rate in aggregate["rates"]]
                ymean = [100 * rate["splits"][split]["accuracy"]["mean"] for rate in aggregate["rates"]]
                ystd = [100 * (rate["splits"][split]["accuracy"]["sample_std"] or 0) for rate in aggregate["rates"]]
                axis.errorbar(xmean, ymean, yerr=ystd, fmt="o-", color="#174c78", capsize=4, linewidth=1.6, label="Mean ± sample SD", zorder=4)
                axis.axhline(100 * references[split]["clean"]["accuracy"], color="#26734d", linestyle="--", linewidth=1.3, label="Original clean model")
                axis.axhline(100 * references[split]["clustered_no_faults"]["accuracy"], color="#a76a2d", linestyle=":", linewidth=1.5, label="Clustered zero-fault reference")
                axis.set_title(f"{split.capitalize()} ({references[split]['clean']['num_examples']:,} images)")
                axis.set_xlabel("Actual corrupted cells / eligible cells (%)")
                axis.set_ylim(lower, upper)
                axis.grid(alpha=.22)
                axis.legend(loc="lower left", fontsize=8)
            axes[0].set_ylabel("Classification accuracy (%)")
            fig.suptitle("ResNet-18 / CIFAR-10: adjacent-level faults, 16 clusters per tensor", fontsize=13)
            repeats = sorted({rate["num_patterns"] for rate in aggregate["rates"] if rate["probability"] > 0})
            fig.text(.5, .015, f"Nonzero rates: {repeats} patterns each; zero faults: one reference. Uniform selection; reliable codebooks and normalization state.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .055, 1, .94))
            paths = [Path(output_dir) / "accuracy_vs_fault_rate.png", Path(output_dir) / "accuracy_vs_fault_rate.pdf"]
            for path in paths:
                fig.savefig(path, dpi=200)
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def write_measurements_csv(path, records):
    fields = ["pattern_id", "probability", "seed", "eligible_cells", "actual_corrupted_cells", "realized_fault_rate", "split", "num_examples", "num_correct", "accuracy", "cross_entropy_loss", "accuracy_drop_percentage_points", "cross_entropy_increase"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for split, metrics in record["measurements"].items():
                writer.writerow({**{key: record[key] for key in ("pattern_id", "probability", "seed")},
                    **{key: record["counts"][key] for key in ("eligible_cells", "actual_corrupted_cells", "realized_fault_rate")},
                    "split": split, **metrics, **record["fault_loss"][split]})


def prepare_sources(root, config):
    audit_dir = root / config.source_audit_results
    require(json.loads((audit_dir / "status.json").read_text())["status"] == "complete", "Fault audit is incomplete")
    audit = json.loads((audit_dir / "report.json").read_text())
    required = {"source_integrity", "zero_fault_identity", "saved_pattern_and_checkpoint", "replay_without_accumulation", "same_seed_reproducibility", "different_seed_valid_pattern", "all_cell_selection_and_boundaries", "source_preservation"}
    require(audit["status"] == "passed" and required <= set(audit["checks"]) and all(audit["checks"][name]["passed"] for name in required), "Full-model fault audit did not pass")
    source_paths, source_hashes = {}, {}
    for name, record in audit["source_artifacts"].items():
        path = root / record["path"]
        require(sha256_file(path) == record["sha256"], f"Audited artifact changed: {name}")
        source_paths[name], source_hashes[name] = path, record["sha256"]
    fault_results = root / json.loads((audit_dir / "config.json").read_text())["source_fault_results"]
    fault_config = FaultConfig(**json.loads((fault_results / "config.json").read_text()))
    fault_config.validate()
    encoding, _, _ = validate_clustered_source(root, fault_config)
    require(encoding_fingerprint(encoding) == audit["checks"]["source_integrity"]["encoding_fingerprint"], "Encoding differs from audited storage")
    clean, metadata, split, clean_metrics, clean_hash = validate_clean_source(root, ClusteredConfig(**encoding["clustering_config"]))
    require(clean_hash == encoding["source_checkpoint_sha256"] and metadata["split_sha256"] == encoding["split_sha256"] and metadata["preprocessing"] == encoding["preprocessing"], "Clean and clustered provenance differ")
    original_clustered_metrics = json.loads((root / fault_config.source_results / "metrics.json").read_text())
    for name, path in {"audit_report": audit_dir / "report.json", "audit_status": audit_dir / "status.json", "audit_config": audit_dir / "config.json",
                       "splits": root / encoding["clustering_config"]["source_results"] / "splits.json", "clustered_metrics": root / fault_config.source_results / "metrics.json"}.items():
        source_paths[name], source_hashes[name] = path, sha256_file(path)
    return encoding, clean, metadata, split, clean_metrics, original_clustered_metrics, source_paths, source_hashes


def run_fault_sweep(config, project_root):
    config.validate()
    root = Path(project_root)
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    device_name = next(name for name in ("cuda", "mps", "cpu") if available[name]) if config.device == "auto" else config.device
    if not available[device_name]:
        raise RuntimeError(f"Requested {device_name} device is unavailable in this process")
    device = torch.device(device_name)
    torch.set_num_threads(config.cpu_threads)
    encoding, clean, metadata, split, clean_metrics, old_clustered, source_paths, source_hashes = prepare_sources(root, config)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Sweep directories already exist; choose a new run_id")
    (results_dir / "patterns").mkdir(parents=True)
    (checkpoint_dir / "patterns").mkdir(parents=True)
    started = time.monotonic()
    plan = sweep_plan(config)
    records = []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "plan.json", plan)
    write_json(results_dir / "status.json", {"status": "running", "completed_patterns": 0, "planned_patterns": len(plan), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_fault_sweep.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        write_json(results_dir / "splits.json", split)
        write_json(results_dir / "preprocessing.json", metadata["preprocessing"])
        environment = {"python": platform.python_version(), "platform": platform.platform(), "evaluation_device": device_name,
                       "fault_sampling_device": "cpu", "cpu_threads": config.cpu_threads,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "matplotlib")}}
        write_json(results_dir / "environment.json", environment)
        fingerprint = encoding_fingerprint(encoding)
        baseline_state = decode_state(encoding)
        _, transform = cifar10_transforms(metadata["preprocessing"]["input_size"][-1])
        train_data = CIFAR10(root / config.data_root, train=True, transform=transform, download=False)
        test_data = CIFAR10(root / config.data_root, train=False, transform=transform, download=False)
        require(train_data.classes == test_data.classes == metadata["classes"] and len(train_data) == 50000 and len(test_data) == 10000, "CIFAR-10 class order or size changed")
        datasets = {"validation": Subset(train_data, split["validation"]), "test": Subset(test_data, split["test"]["indices"])}
        loaders = {name: DataLoader(dataset, batch_size=metadata["config"]["evaluation_batch_size"], shuffle=False,
                   num_workers=config.num_workers, persistent_workers=config.num_workers > 0, pin_memory=device_name == "cuda",
                   generator=torch.Generator().manual_seed(config.seed)) for name, dataset in datasets.items()}
        references = {}
        clean.to(device).eval().requires_grad_(False)
        for name, loader in loaders.items():
            print(json.dumps({"event": "clean_reference_start", "split": name}), flush=True)
            current = evaluate_classifier(clean, loader, device)
            references[name] = {"clean": current, "clean_num_correct_difference_from_step3": current["num_correct"] - clean_metrics[name]["num_correct"]}
        del clean
        model = build_resnet18(10, pretrained=False).to(device).eval().requires_grad_(False)
        for entry in plan:
            print(json.dumps({"event": "pattern_evaluation_start", **entry, "completed_patterns": len(records), "planned_patterns": len(plan)}), flush=True)
            pattern_started = time.monotonic()
            pattern = generate_fault_pattern(encoding, entry["probability"], seed=entry["seed"])
            path = checkpoint_dir / "patterns" / f"{entry['pattern_id']}.pt"
            save_torch(path, pattern)
            saved_pattern = torch.load(path, map_location="cpu", weights_only=True)
            counts = summarize_fault_pattern(encoding, saved_pattern)
            state = apply_fault_pattern(encoding, saved_pattern)
            if entry["probability"] == 0:
                require_state_equal(state, baseline_state, "Sweep zero-fault identity")
            for name, value in encoding["untouched_state"].items():
                require(torch.equal(state[name], value), f"Reliable state changed: {name}")
            # Load the complete state from fresh baseline decoding for every run.
            model.load_state_dict(state, strict=True)
            measurements = {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}
            if entry["probability"] == 0:
                for name in loaders:
                    references[name]["clustered_no_faults"] = measurements[name]
                    references[name]["clustered_num_correct_difference_from_step4"] = measurements[name]["num_correct"] - old_clustered["measurements"][name]["clustered_no_faults"]["num_correct"]
                    references[name]["compression_accuracy_drop_percentage_points"] = 100 * (references[name]["clean"]["accuracy"] - measurements[name]["accuracy"])
                write_json(results_dir / "references.json", references)
            fault_loss = {name: {"accuracy_drop_percentage_points": 100 * (references[name]["clustered_no_faults"]["accuracy"] - measurements[name]["accuracy"]),
                                "cross_entropy_increase": measurements[name]["cross_entropy_loss"] - references[name]["clustered_no_faults"]["cross_entropy_loss"]} for name in loaders}
            record = {**entry, "pattern": str(path.relative_to(root)), "pattern_sha256": sha256_file(path), "counts": counts,
                      "measurements": measurements, "fault_loss": fault_loss, "elapsed_seconds": time.monotonic() - pattern_started}
            write_json(results_dir / "patterns" / f"{entry['pattern_id']}.json", record)
            records.append(record)
            write_json(results_dir / "measurements.json", records)
            write_json(results_dir / "status.json", {"status": "running", "completed_patterns": len(records), "planned_patterns": len(plan)})
            print(json.dumps({"event": "pattern_evaluation_complete", **entry, "actual_corrupted_cells": counts["actual_corrupted_cells"],
                              "realized_fault_rate": counts["realized_fault_rate"], "measurements": measurements, "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            del pattern, saved_pattern, state
        aggregate = aggregate_measurements(records, config)
        write_json(results_dir / "aggregate.json", aggregate)
        write_measurements_csv(results_dir / "measurements.csv", records)
        figures = plot_sweep(records, aggregate, references, results_dir)
        require(encoding_fingerprint(encoding) == fingerprint, "Source encoding changed during sweep")
        for name, path in source_paths.items():
            require(sha256_file(path) == source_hashes[name], f"Source artifact changed: {name}")
        manifest = {
            "run_id": config.run_id, "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": source_hashes[name]} for name, path in source_paths.items()},
            "encoding_fingerprint": fingerprint, "eligible_tensor_names": sorted(encoding["clustered_tensors"]), "eligible_cells": records[0]["counts"]["eligible_cells"],
            "preprocessing": metadata["preprocessing"], "split_sha256": metadata["split_sha256"], "evaluation_batch_size": metadata["config"]["evaluation_batch_size"],
            "seed_rule": "Nonzero rate ordinal * patterns_per_rate + repeat + base seed, modulo 2**32; unique seeds across all nonzero runs. Zero is one deterministic reference.",
            "fault_model": {key: value for key, value in torch.load(root / records[0]["pattern"], map_location="cpu", weights_only=True).items() if key not in ("tensors", "tensor_order", "seed", "fault_probability")},
            "selection_policy": "Rates and seed plan fixed before evaluation; no training, checkpoint selection, or tuning against validation/test results.",
            "comparisons": "Compression loss: original clean versus clustered zero faults. Fault loss: clustered zero faults versus each saved faulty pattern; signed drops are allowed.",
            "reset_policy": "Every state reconstructed from the uncorrupted encoding; full state loaded before each evaluation; saved pattern static for both splits.",
            "pattern_storage": "Exact sparse patterns saved; dense faulty checkpoints are reconstructed from the source encoding and pattern when needed.",
            "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures],
            "source_artifacts_preserved": True, "completed_patterns": len(records), "environment": environment,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "status.json", {"status": "complete", "completed_patterns": len(records), "planned_patterns": len(plan), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "fault_sweep_complete", "run_id": config.run_id, "completed_patterns": len(records), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return aggregate
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_patterns": len(records), "planned_patterns": len(plan), "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_fault_sweep.json"))
    args = parser.parse_args()
    run_fault_sweep(SweepConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
