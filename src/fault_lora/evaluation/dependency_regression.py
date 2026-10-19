"""Evaluation-only proof that the dependency change did not alter ResNet-18 results.

Reloads saved adapters and the clean/clustered references, evaluates them on the
saved validation and test splits, and requires the exact saved correct counts. It
also re-runs the Step 9b verifications and source preparation under the shared
recorded-hash rule. Nothing is trained, selected, or written outside its run.
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from fault_lora.data.cifar10 import cifar10_transforms
from fault_lora.evaluation.bn_recalibration import BNRecalibrationConfig, verify_recalibration
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.fault_sweep import SweepConfig, prepare_sources as prepare_sweep_sources
from fault_lora.evaluation.lora_controls import ControlsConfig, build_analysis
from fault_lora.evaluation.source_hashes import (DEPENDENCY_CHANGES, DEPENDENCY_FILES, SOURCE_CHANGES, compare_source_hashes, load_accepted_changes, summarize_lockfile_change,
                                                 summarize_requirements_change, unique_changes)
from fault_lora.memory.clustering import decode_state
from fault_lora.models.lora import adapter_state_dict, load_adapter_state
from fault_lora.models.resnet import build_resnet18
from fault_lora.training import recovery
from fault_lora.training.baseline import choose_device
from fault_lora.training.recovery_matrix import MatrixConfig, verify_matrix, verify_run


SPLITS = ("validation", "test")
LOCKFILE_SUMMARY = "configs/accepted_dependency_changes_lockfile_diff.json"
# Provenance fields that a re-derivation reports afresh; everything else must equal the saved analysis.
ANALYSIS_PROVENANCE = ("source_files_changed_since_runs", "accepted_source_hash_changes")


@dataclass
class DependencyRegressionConfig:
    run_id: str = "resnet18-cifar10-dependency-regression-v1"
    adapter_runs: list = field(default_factory=lambda: ["resnet18-cifar10-lora-r4-p001-seed52-train42", "resnet18-cifar10-lora-clustered-none-r4-train42",
                                                        "resnet18-cifar10-lora-faulty-rate06-seed67-r4-train42"])
    source_sweep_results: str = "results/resnet18-cifar10-fault-sweep-seed42"
    clean_results: str = "results/resnet18-cifar10-clean-seed42"
    clustered_results: str = "results/resnet18-cifar10-clustered-k16-seed42"
    verified_matrices: list = field(default_factory=list)
    interrupted_matrix: str | None = None
    bn_results: str | None = None
    analysis_config: str = "configs/resnet18_cifar10_lora_controls.json"
    analysis_results: str = "results/resnet18-cifar10-lora-controls"
    old_dependency_commit: str = "159ec94"
    evaluation_batch_size: int = 256
    device: str = "mps"
    cpu_threads: int = 4
    num_workers: int = 2
    data_root: str = "data"
    results_root: str = "results"

    def validate(self):
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if not self.adapter_runs or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in self.adapter_runs) or len(set(self.adapter_runs)) != len(self.adapter_runs):
            raise ValueError("adapter_runs must be distinct run IDs")
        if not isinstance(self.verified_matrices, list) or len(set(self.verified_matrices)) != len(self.verified_matrices):
            raise ValueError("verified_matrices must be distinct result directories")
        if not re.fullmatch(r"[0-9a-f]{7,40}", self.old_dependency_commit):
            raise ValueError("old_dependency_commit must be a commit id")
        for name, minimum in (("evaluation_batch_size", 1), ("cpu_threads", 1), ("num_workers", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def split_sha256(split):
    return hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_counts(measured, saved):
    """Exact correct counts are required; cross-entropy differences are reported."""
    comparison = {split: {"measured_num_correct": measured[split]["num_correct"], "saved_num_correct": saved[split]["num_correct"],
                          "num_correct_difference": measured[split]["num_correct"] - saved[split]["num_correct"],
                          "measured_cross_entropy_loss": measured[split]["cross_entropy_loss"], "saved_cross_entropy_loss": saved[split]["cross_entropy_loss"],
                          "cross_entropy_difference": measured[split]["cross_entropy_loss"] - saved[split]["cross_entropy_loss"]} for split in SPLITS}
    return comparison, all(item["num_correct_difference"] == 0 and measured[split]["num_examples"] == saved[split]["num_examples"] for split, item in comparison.items())


def dependency_decision(root, config):
    """Re-derive the lockfile and requirements differences from git history and the current files."""
    show = lambda path: subprocess.run(["git", "show", f"{config.old_dependency_commit}:{path}"], cwd=root, capture_output=True, text=True, check=True).stdout
    old = {path: show(path) for path in DEPENDENCY_FILES}
    current = {path: (root / path).read_text() for path in DEPENDENCY_FILES}
    summary = {"old_commit": config.old_dependency_commit,
               "old_sha256": {path: hashlib.sha256(text.encode()).hexdigest() for path, text in old.items()},
               "new_sha256": {path: sha256_file(root / path) for path in DEPENDENCY_FILES},
               "requirements": summarize_requirements_change(old["requirements.txt"], current["requirements.txt"]),
               "lockfile": summarize_lockfile_change(old["requirements-macos-arm64.lock.txt"], current["requirements-macos-arm64.lock.txt"])}
    saved = json.loads((root / LOCKFILE_SUMMARY).read_text())
    require(saved == summary, "Saved dependency diff summary differs from its re-derivation from git history")
    accepted = load_accepted_changes(root)
    dependency = [entry for entry in accepted if entry["list"] == "dependency"]
    require({(entry["path"], entry["old_sha256"], entry["new_sha256"]) for entry in dependency} == {(path, summary["old_sha256"][path], summary["new_sha256"][path]) for path in DEPENDENCY_FILES},
            "Accepted dependency changes differ from the git-history and current hashes")
    lock = summary["lockfile"]
    claim = lock["old_packages"] == 28 and not lock["removed"] and lock["version_changed"] == {"fsspec": {"old": "2026.9.0", "new": "2026.7.0"}} and lock["same_version"] == 27 \
        and not lock["same_version_with_different_wheel_hashes"] and summary["requirements"]["added"] == ["transformers==5.19.0", "datasets==5.1.0"] and not summary["requirements"]["removed"]
    require(claim, "Recorded reason does not match the lockfile difference")
    return {"summary": summary, "accepted_changes": accepted, "claim_verified": True,
            "decision_files": {path: sha256_file(root / path) for path in (DEPENDENCY_CHANGES, SOURCE_CHANGES, LOCKFILE_SUMMARY) if (root / path).exists()}}


def attempt(function):
    """Run one re-verification; record success with its accepted changes, or the exact failure."""
    try:
        value = function()
        return {"passed": True, **value}
    except Exception as error:   # Report every failure rather than stopping at the first one.
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}


def run_dependency_regression(config, project_root):
    config.validate()
    root = Path(project_root)
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Regression directory already exists; choose a new run_id")
    decision = dependency_decision(root, config)
    results_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/check_dependency_regression.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "cpu_threads": config.cpu_threads,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "fsspec", "transformers", "datasets")}}
        write_json(results_dir / "environment.json", environment)
        write_json(results_dir / "dependency_decision.json", decision)
        # References: the exact Step 3 clean checkpoint and the decoded Step 4 encoding.
        sweep_config = SweepConfig(**json.loads((root / config.source_sweep_results / "config.json").read_text()))
        encoding, clean, metadata, split, clean_metrics, clustered_metrics, _, _ = prepare_sweep_sources(root, sweep_config)
        _, transform = cifar10_transforms(metadata["preprocessing"]["input_size"][-1])
        train_data = CIFAR10(root / config.data_root, train=True, transform=transform, download=False)
        test_data = CIFAR10(root / config.data_root, train=False, transform=transform, download=False)
        require(train_data.classes == test_data.classes == metadata["classes"] and len(train_data) == 50000 and len(test_data) == 10000, "CIFAR-10 class order or size changed")
        loaders = {"validation": DataLoader(Subset(train_data, split["validation"]), batch_size=config.evaluation_batch_size, shuffle=False, num_workers=config.num_workers),
                   "test": DataLoader(Subset(test_data, split["test"]["indices"]), batch_size=config.evaluation_batch_size, shuffle=False, num_workers=config.num_workers)}
        measure = lambda model: {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}
        evaluations = {}
        print(json.dumps({"event": "reference_evaluation_start", "reference": "clean"}), flush=True)
        clean.to(device).eval().requires_grad_(False)
        evaluations["clean_step3"] = dict(zip(("comparison", "passed"), compare_counts(measure(clean), clean_metrics)))
        del clean
        clustered = build_resnet18(metadata["num_classes"], pretrained=False)
        clustered.load_state_dict(decode_state(encoding), strict=True)
        clustered.to(device).eval().requires_grad_(False)
        print(json.dumps({"event": "reference_evaluation_start", "reference": "clustered"}), flush=True)
        saved_clustered = {name: clustered_metrics["measurements"][name]["clustered_no_faults"] for name in SPLITS}
        evaluations["clustered_step4"] = dict(zip(("comparison", "passed"), compare_counts(measure(clustered), saved_clustered)))
        del clustered
        accepted = []
        for run_id in config.adapter_runs:
            print(json.dumps({"event": "adapter_evaluation_start", "run_id": run_id}), flush=True)
            metrics = json.loads((root / config.results_root / run_id / "metrics.json").read_text())
            checkpoint = root / metrics["best_checkpoint"]
            require(sha256_file(checkpoint) == metrics["best_checkpoint_sha256"], f"Saved adapter changed: {run_id}")
            model, payload = recovery.load_recovered_model(checkpoint, root, device=device)
            require(payload["split_sha256"] == split_sha256(split) == metadata["split_sha256"], f"Adapter uses a different split: {run_id}")
            accepted += payload["accepted_source_hash_changes"]
            after = measure(model)
            trained = adapter_state_dict(model)
            load_adapter_state(model, {name: torch.zeros_like(value) if ".lora_B." in name else value for name, value in trained.items()})
            before = measure(model)
            after_comparison, after_passed = compare_counts(after, metrics["after"])
            before_comparison, before_passed = compare_counts(before, metrics["before"])
            evaluations[run_id] = {"base_kind": payload["config"].get("base_kind", "faulty"), "pattern_id": payload["config"]["pattern_id"], "selected_epoch": payload["epoch"],
                                   "checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": metrics["best_checkpoint_sha256"],
                                   "with_adapter": after_comparison, "adapter_output_zeroed": before_comparison, "passed": after_passed and before_passed}
            del model
            print(json.dumps({"event": "adapter_evaluation_complete", "run_id": run_id, "passed": evaluations[run_id]["passed"]}), flush=True)
        write_json(results_dir / "regression.json", {"evaluations": evaluations, "accepted_source_hash_changes": unique_changes(accepted)})
        # Re-verification under the shared rule; each result records its accepted changes.
        checks = {}
        for kind, pattern_id in (("clean", None), ("clustered", None), ("faulty", "rate03-seed52")):
            prepare = lambda kind=kind, pattern_id=pattern_id: {"accepted_source_hash_changes": unique_changes(
                recovery.prepare_sources(root, recovery.RecoveryConfig(base_kind=kind, pattern_id=pattern_id))[3]["accepted_source_hash_changes"])}
            checks[f"prepare_sources:{kind}"] = attempt(prepare)
        for directory in config.verified_matrices:
            matrix = MatrixConfig(**json.loads((root / directory / "config.json").read_text()))
            checks[f"verify_matrix:{directory}"] = attempt(lambda matrix=matrix: verify_matrix(root, matrix))
        if config.interrupted_matrix is not None:
            matrix = MatrixConfig(**json.loads((root / config.interrupted_matrix / "config.json").read_text()))
            rows = json.loads((root / config.interrupted_matrix / "summary.json").read_text())

            def finished_runs():
                results = [verify_run(root, matrix, run, row) for run, row in zip(matrix.runs, rows)]
                changes = [change for result in results for change in result["accepted_source_hash_changes"]]
                changes += compare_source_hashes(root, json.loads((root / config.interrupted_matrix / "source_sha256.json").read_text()), "Source code changed during the matrix")
                return {"runs_verified": len(results), "accepted_source_hash_changes": unique_changes(changes)}
            checks[f"verify_run:{config.interrupted_matrix} (finished runs)"] = attempt(finished_runs)
        if config.bn_results is not None:
            bn = BNRecalibrationConfig(**json.loads((root / config.bn_results / "config.json").read_text()))
            checks[f"verify_recalibration:{config.bn_results}"] = attempt(lambda: verify_recalibration(root, bn))

        def analysis_inputs():
            analysis, _, hashes = build_analysis(root, ControlsConfig(**json.loads((root / config.analysis_config).read_text())))
            saved = json.loads((root / config.analysis_results / "analysis.json").read_text())
            strip = lambda value: {key: item for key, item in json.loads(json.dumps(value)).items() if key not in ANALYSIS_PROVENANCE}
            require(strip(analysis) == strip(saved), "Re-derived Step 9b analysis differs from the saved analysis")
            return {"input_files_hashed": len(hashes), "rederived_analysis_equals_saved": True, "accepted_source_hash_changes": analysis["accepted_source_hash_changes"]}
        checks["lora_controls analysis input loading"] = attempt(analysis_inputs)
        write_json(results_dir / "reverification.json", checks)
        compare_source_hashes(root, source_code, "Source code changed during the regression run")
        regression_passed = all(item["passed"] for item in evaluations.values())
        reverification_passed = all(item["passed"] for item in checks.values())
        verification = {"status": "passed" if regression_passed and reverification_passed else "failed", "dependency_claim_verified_from_git_history": decision["claim_verified"],
                        "exact_counts": {name: item["passed"] for name, item in evaluations.items()}, "regression_passed": regression_passed,
                        "reverification": {name: item["passed"] for name, item in checks.items()}, "reverification_passed": reverification_passed,
                        "accepted_source_hash_changes": unique_changes(accepted + [change for item in checks.values() for change in item.get("accepted_source_hash_changes", [])]),
                        "training_performed": False, "saved_results_modified": False}
        write_json(results_dir / "verification.json", verification)
        manifest = {"run_id": config.run_id, "purpose": "Evaluation-only regression and re-verification after the dependency change; no training.",
                    "count_rule": "Correct counts must equal the saved values exactly on validation and test; cross-entropy differences are reported, not gated.",
                    "acceptance_rule": "Recorded source hashes must equal the current file or be an exactly listed (path, old, new) change in a decision file.",
                    "environment": environment, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "status.json", {"status": "complete" if verification["status"] == "passed" else "failed", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "dependency_regression_complete", "run_id": config.run_id, "verification": verification["status"], "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        require(verification["status"] == "passed", f"Regression or re-verification failed; see {results_dir / 'verification.json'}")
        return verification
    except BaseException as error:
        if json.loads((results_dir / "status.json").read_text())["status"] == "running":
            write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_dependency_regression.json"))
    args = parser.parse_args()
    run_dependency_regression(DependencyRegressionConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
