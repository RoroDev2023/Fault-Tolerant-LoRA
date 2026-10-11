"""Run a predeclared list of independent LoRA recovery runs and summarize them."""

import argparse
import csv
from dataclasses import asdict, dataclass, field
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
from fault_lora.training import recovery
from fault_lora.training.recovery import RecoveryConfig


RUN_FIELDS = ("base_kind", "pattern_id", "rank", "alpha", "seed")
SUMMARY_FIELDS = ["run_id", "base_kind", "pattern_id", "fault_probability", "actual_corrupted_cells", "realized_fault_rate", "rank", "alpha", "seed", "selected_epoch",
                  *(f"{split}_{metric}_{stage}" for split in ("validation", "test") for metric in ("accuracy", "cross_entropy_loss") for stage in ("before", "after")),
                  "validation_accuracy_gain_percentage_points", "test_accuracy_gain_percentage_points",
                  "trainable_adapter_parameters", "training_and_validation_seconds", "elapsed_seconds"]


@dataclass
class MatrixConfig:
    matrix_id: str = "resnet18-cifar10-lora-matrix"
    runs: list = field(default_factory=list)
    run_id_prefix: str = "resnet18-cifar10-lora"
    source_sweep_results: str = "results/resnet18-cifar10-fault-sweep-seed42"
    source_audit_results: str = "results/resnet18-cifar10-fault-checks-seed42"
    clean_checkpoint_sha256: str | None = None
    target_names: list | None = None
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

    def validate(self):
        for name in ("matrix_id", "run_id_prefix"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", getattr(self, name)):
                raise ValueError(f"{name} must be a simple name")
        if not isinstance(self.runs, list) or not self.runs:
            raise ValueError("runs must be a nonempty list")
        if any(not isinstance(run, dict) or set(run) != set(RUN_FIELDS) for run in self.runs):
            raise ValueError(f"Every run must declare exactly {RUN_FIELDS}")
        configs = [run_config(self, run) for run in self.runs]
        for config in configs:
            config.validate()
        run_ids = [config.run_id for config in configs]
        if len(set(run_ids)) != len(run_ids) or self.matrix_id in run_ids:
            raise ValueError("Runs must produce distinct run IDs that differ from matrix_id; alpha is not part of a run ID")


def run_config(config, run):
    """One matrix entry as a complete, independently reproducible recovery configuration."""
    shared = {name: getattr(config, name) for name in ("source_sweep_results", "source_audit_results", "clean_checkpoint_sha256", "target_names", "epochs", "batch_size",
              "evaluation_batch_size", "learning_rate", "weight_decay", "max_grad_norm", "device", "cpu_threads", "num_workers", "data_root", "results_root", "checkpoint_root")}
    run_id = f"{config.run_id_prefix}-{run['base_kind']}-{run['pattern_id'] or 'none'}-r{run['rank']}-train{run['seed']}"
    return RecoveryConfig(run_id=run_id, **{name: run[name] for name in RUN_FIELDS}, **shared)


def matrix_plan(config):
    config.validate()
    return [{"run_id": run_config(config, run).run_id, **{name: run[name] for name in RUN_FIELDS}} for run in config.runs]


def summary_row(metrics):
    row = {"run_id": metrics["run_id"], "base_kind": metrics["base_kind"], "pattern_id": metrics["pattern_id"], "fault_probability": metrics["fault_probability"],
           "actual_corrupted_cells": metrics["actual_corrupted_cells"], "realized_fault_rate": metrics["realized_fault_rate"],
           **{name: metrics["config"][name] for name in ("rank", "alpha", "seed")}, "selected_epoch": metrics["selected_epoch"]}
    for split in ("validation", "test"):
        for metric in ("accuracy", "cross_entropy_loss"):
            for stage in ("before", "after"):
                row[f"{split}_{metric}_{stage}"] = metrics[stage][split][metric]
    for split in ("validation", "test"):
        row[f"{split}_accuracy_gain_percentage_points"] = metrics["recovery"][split]["accuracy_gain_percentage_points"]
    row.update({"trainable_adapter_parameters": metrics["parameters"]["trainable_adapter"],
                "training_and_validation_seconds": metrics["training_and_validation_seconds"], "elapsed_seconds": metrics["elapsed_seconds"]})
    require(list(row) == SUMMARY_FIELDS, "Summary columns changed")
    return row


def csv_row(row):
    return {name: "" if value is None else str(value) for name, value in row.items()}


def write_summary(results_dir, rows):
    write_json(results_dir / "summary.json", rows)
    temporary = results_dir / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(csv_row(row) for row in rows)
    temporary.replace(results_dir / "summary.csv")


def verify_matrix(root, config):
    """Re-read every saved run and the summary; fail on any inconsistency."""
    root = Path(root)
    results_dir = root / config.results_root / config.matrix_id
    plan = matrix_plan(config)
    require(json.loads((results_dir / "plan.json").read_text()) == plan, "Saved matrix plan differs from its configuration")
    rows = json.loads((results_dir / "summary.json").read_text())
    require([row["run_id"] for row in rows] == [entry["run_id"] for entry in plan], "Summary is incomplete, duplicated, or reordered")
    with (results_dir / "summary.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == [csv_row(row) for row in rows], "CSV and JSON summaries disagree")
    source_artifacts = 0
    for run, row in zip(config.runs, rows):
        expected = run_config(config, run)
        directory = root / config.results_root / expected.run_id
        status = json.loads((directory / "status.json").read_text())
        require(status["status"] == "complete" and status["completed_epochs"] == config.epochs, f"Run is incomplete: {expected.run_id}")
        require(json.loads((directory / "config.json").read_text()) == asdict(expected), f"Saved run configuration differs from the matrix: {expected.run_id}")
        metrics = json.loads((directory / "metrics.json").read_text())
        require(metrics["config"] == asdict(expected) and summary_row(metrics) == row, f"Summary row differs from saved metrics: {expected.run_id}")
        require(metrics["before"] == json.loads((directory / "before.json").read_text()), f"Saved before-LoRA metrics disagree: {expected.run_id}")
        history = json.loads((directory / "history.json").read_text())
        require([item["epoch"] for item in history] == list(range(1, config.epochs + 1)) and all(item["frozen_base_verified"] for item in history), f"Epoch history is incomplete: {expected.run_id}")
        # Recompute selection from validation records only; strict improvement keeps the earlier epoch.
        selected, best_key = 0, (metrics["before"]["validation"]["accuracy"], -metrics["before"]["validation"]["cross_entropy_loss"])
        for item in history:
            key = (item["validation"]["accuracy"], -item["validation"]["cross_entropy_loss"])
            if key > best_key:
                selected, best_key = item["epoch"], key
        require(selected == metrics["selected_epoch"] == status["selected_epoch"], f"Selected epoch is not the validation optimum: {expected.run_id}")
        best = root / metrics["best_checkpoint"]
        require(sha256_file(best) == metrics["best_checkpoint_sha256"], f"Selected adapter checkpoint changed: {expected.run_id}")
        payload = torch.load(best, map_location="cpu", weights_only=True)
        require(payload["epoch"] == selected and payload["validation_metrics"] == metrics["after"]["validation"] and payload["config"] == asdict(expected),
                f"Selected adapter checkpoint differs from metrics: {expected.run_id}")
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, item in manifest["source_artifacts"].items():
            require(sha256_file(root / item["path"]) == item["sha256"], f"Input artifact changed after {expected.run_id}: {name}")
        source_artifacts += len(manifest["source_artifacts"])
        for relative, digest in json.loads((directory / "source_sha256.json").read_text()).items():
            require(sha256_file(root / relative) == digest, f"Source code changed after {expected.run_id}: {relative}")
    for relative, digest in json.loads((results_dir / "source_sha256.json").read_text()).items():
        require(sha256_file(root / relative) == digest, f"Source code changed during the matrix: {relative}")
    return {"status": "passed", "runs_verified": len(rows), "plan_and_summary_order_verified": True, "csv_matches_json": True, "run_configurations_match_matrix": True,
            "summary_rows_match_saved_metrics": True, "selection_verified_using_validation_only": True, "selected_checkpoint_hashes_verified": True,
            "base_frozen_every_epoch": True, "source_artifacts_verified": source_artifacts, "source_code_unchanged": True}


def run_matrix(config, project_root):
    plan = matrix_plan(config)
    root = Path(project_root)
    results_dir = root / config.results_root / config.matrix_id
    existing = [str(path.relative_to(root)) for entry in plan for path in (root / config.results_root / entry["run_id"], root / config.checkpoint_root / entry["run_id"]) if path.exists()]
    if results_dir.exists() or existing:
        raise FileExistsError(f"Matrix or run directories already exist; choose new IDs: {[str(results_dir.relative_to(root))] if results_dir.exists() else existing}")
    # Every pattern must be a saved nonzero Step 7 pattern before any run starts.
    sweep_plan_path = root / config.source_sweep_results / "plan.json"
    saved = {entry["pattern_id"]: entry for entry in json.loads(sweep_plan_path.read_text())}
    for entry in plan:
        require(entry["pattern_id"] is None or (entry["pattern_id"] in saved and saved[entry["pattern_id"]]["probability"] > 0), f"Not a saved nonzero sweep pattern: {entry['pattern_id']}")
    results_dir.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "plan.json", plan)
    write_json(results_dir / "status.json", {"status": "running", "completed_runs": 0, "planned_runs": len(plan), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    current = None
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_lora_matrix.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": config.device, "cpu_threads": config.cpu_threads,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "matplotlib")}}
        write_json(results_dir / "environment.json", environment)
        sweep_plan_hash = sha256_file(sweep_plan_path)
        for run, entry in zip(config.runs, plan):
            current = entry["run_id"]
            print(json.dumps({"event": "matrix_run_start", **entry, "completed_runs": len(rows), "planned_runs": len(plan)}), flush=True)
            rows.append(summary_row(recovery.run_recovery(run_config(config, run), root)))
            write_summary(results_dir, rows)
            write_json(results_dir / "status.json", {"status": "running", "completed_runs": len(rows), "planned_runs": len(plan)})
            print(json.dumps({"event": "matrix_run_complete", **rows[-1], "completed_runs": len(rows), "planned_runs": len(plan)}), flush=True)
        current = None
        require(sha256_file(sweep_plan_path) == sweep_plan_hash, "Sweep plan changed during the matrix")
        verification = verify_matrix(root, config)
        write_json(results_dir / "verification.json", verification)
        manifest = {"matrix_id": config.matrix_id, "runs": [{"run_id": row["run_id"], "results": f"{config.results_root}/{row['run_id']}", "checkpoints": f"{config.checkpoint_root}/{row['run_id']}"} for row in rows],
                    "source_artifacts": {"sweep_plan": {"path": str(sweep_plan_path.relative_to(root)), "sha256": sweep_plan_hash}},
                    "run_id_rule": "{run_id_prefix}-{base_kind}-{pattern_id or 'none'}-r{rank}-train{seed}",
                    "run_policy": "Every declared run is an independent recovery run with its own directories, provenance, and verification; existing directories are never reused; the first failure stops the matrix.",
                    "selection_rule": "Per run: highest validation accuracy, then lowest validation loss; exact ties retain the earlier epoch. Includes epoch-zero adapters.",
                    "test_policy": "Per run: test evaluated only after validation-only checkpoint selection. Runs, ranks, seeds, and patterns are fixed in the configuration before any run.",
                    "summary_fields": SUMMARY_FIELDS, "environment": environment, "completed_runs": len(rows), "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "status.json", {"status": "complete", "completed_runs": len(rows), "planned_runs": len(plan), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "lora_matrix_complete", "matrix_id": config.matrix_id, "completed_runs": len(rows), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return rows
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_runs": len(rows), "planned_runs": len(plan),
                                                  "failed_run_id": current, "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_matrix(MatrixConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
