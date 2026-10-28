"""Re-evaluate the Step 10a clean TinyBERT checkpoint on the selection split and both test sets."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

import torch

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.source_hashes import compare_source_hashes
from fault_lora.evaluation.tinybert_sources import evaluate_splits, make_loaders, prepare_tinybert_sources, test_set_sha256
from fault_lora.training.baseline import choose_device


@dataclass
class CleanEvalConfig:
    run_id: str = "tinybert-sst2-clean-twotest-seed42"
    source_results: str = "results/tinybert-sst2-clean-seed42"
    expected_original_test_examples: int | None = 1820
    device: str = "cpu"
    cpu_threads: int = 4
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int) or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")
        if self.expected_original_test_examples is not None and (isinstance(self.expected_original_test_examples, bool) or not isinstance(self.expected_original_test_examples, int) or self.expected_original_test_examples < 1):
            raise ValueError("expected_original_test_examples must be a positive integer or null")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def environment_record(device, threads, extra=()):
    packages = ("torch", "numpy", "transformers", "tokenizers", "datasets", "safetensors") + tuple(extra)
    return {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "cpu_threads": threads,
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def source_artifacts(root, paths, hashes):
    return {name: {"path": str(path.relative_to(root)), "sha256": hashes[name]} for name, path in paths.items()}


def verify_source_artifacts(root, manifest, label):
    for name, item in manifest["source_artifacts"].items():
        require(sha256_file(root / item["path"]) == item["sha256"], f"{label}: input artifact changed: {name}")


def verify_source_code(root, results_dir, label):
    """Recorded source hashes must match, or be an exactly listed change (shared rule); return accepted changes."""
    accepted = compare_source_hashes(root, json.loads((results_dir / "source_sha256.json").read_text()), f"{label}: source code changed")
    if accepted:
        print(json.dumps({"event": "accepted_source_hash_changes", "label": label, "paths": sorted({change["path"] for change in accepted})}), flush=True)
    return accepted


def verify_clean_eval(root, config):
    root = Path(root).resolve()
    results_dir = root / config.results_root / config.run_id
    read = lambda name: json.loads((results_dir / name).read_text())
    require(read("config.json") == asdict(config), "Saved configuration differs")
    manifest, metrics, test_sets = read("manifest.json"), read("metrics.json"), read("test_sets.json")
    verify_source_artifacts(root, manifest, "Clean evaluation")
    verify_source_code(root, results_dir, "Clean evaluation")
    sources = prepare_tinybert_sources(root, config.source_results, expected_original_test_examples=config.expected_original_test_examples)
    require(test_sets == sources.test_sets, "Recomputed test-set definitions differ from the saved ones")
    for name, (sentences, labels) in sources.datasets.items():
        require(test_set_sha256(sentences, labels) == test_sets[name]["sha256"], f"Test-set content hash differs: {name}")
        measurement = metrics["measurements"][name]
        require(measurement["num_examples"] == len(labels) and measurement["accuracy"] == measurement["num_correct"] / measurement["num_examples"], f"Inconsistent measurement: {name}")
    require(list(metrics["measurements"]) == list(sources.datasets), "Measured splits differ from the declared splits")
    for name, difference in metrics["num_correct_difference_from_step10a"].items():
        require(difference == 0, f"Clean replay differs from Step 10a: {name}")
    return {"status": "passed", "configuration_matches": True, "input_artifact_hashes_verified": len(manifest["source_artifacts"]), "source_code_unchanged": True,
            "test_sets_recomputed_and_equal": True, "measurement_counts_verified": list(sources.datasets), "step10a_counts_reproduced": True}


def run_clean_eval(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Run directory already exists. Choose a new run_id; existing research artifacts are preserved.")
    sources = prepare_tinybert_sources(root, config.source_results, expected_original_test_examples=config.expected_original_test_examples)
    results_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_clean_eval.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        write_json(results_dir / "environment.json", environment_record(device, config.cpu_threads))
        write_json(results_dir / "test_sets.json", sources.test_sets)
        removed = sources.test_sets[sources.test_names[1]]["removed"]
        print(json.dumps({"event": "test_sets_built", "sizes": {name: item["num_examples"] for name, item in sources.test_sets.items()}, "removed_sentences": [item["sentence"] for item in removed]}), flush=True)
        model = sources.model.to(device)
        measurements = evaluate_splits(model, make_loaders(sources), device)
        step10a = sources.metrics["measurements"]
        differences = {"selection": measurements["selection"]["num_correct"] - step10a["selection"]["num_correct"],
                       sources.test_names[0]: measurements[sources.test_names[0]]["num_correct"] - step10a["test"]["num_correct"]}
        metrics = {
            "run_id": config.run_id, "fine_tuning": "none", "checkpoint": sources.metrics["checkpoint"], "checkpoint_sha256": sources.hashes["clean_checkpoint"],
            "measurements": measurements, "num_correct_difference_from_step10a": differences,
            "reporting_policy": "Each test set is reported separately and never pooled. Selection accuracy is used only for selection; it is not a held-out estimate for this checkpoint, whose publisher fine-tuned on the full SST-2 train split.",
            "elapsed_seconds": time.monotonic() - started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        require(all(value == 0 for value in differences.values()), f"Clean replay differs from Step 10a: {differences}")
        write_json(results_dir / "metrics.json", metrics)
        write_json(results_dir / "manifest.json", {
            "run_id": config.run_id, "source_artifacts": source_artifacts(root, sources.paths, sources.hashes),
            "test_set_names": sources.test_names, "split_sha256": sources.manifest["split_sha256"],
            "evaluation": {"batch_size": sources.config.evaluation_batch_size, "max_length": sources.config.max_length, "training": "none"},
        })
        write_json(results_dir / "verification.json", verify_clean_eval(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": metrics["completed_at_utc"]})
        print(json.dumps({"event": "clean_eval_complete", **metrics}), flush=True)
        return metrics
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error": str(error), "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2_clean_twotest.json")
    args = parser.parse_args()
    run_clean_eval(CleanEvalConfig(**json.loads(args.config.read_text())), project_root)
