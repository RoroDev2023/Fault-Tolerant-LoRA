"""Declared TinyBERT recovery matrix: independent recovery runs, then saved transfer evaluations.

Each run is an ordinary `run_tinybert_recovery` call with its own directories and
verification; run IDs are derived by `recovery_run_id`. Existing directories are
never reused, nothing is skipped or resumed, and the first failure stops the
matrix. Transfer evaluations load a clustered-trained adapter onto faulty bases
of the same condition and train nothing.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.training.tinybert_recovery import TinyBertRecoveryConfig, evaluate_tinybert_adapter_on_base, recovery_run_id, run_tinybert_recovery

RUN_FIELDS = ("base_kind", "condition", "pattern_id", "source_sweep_results", "seed")
RECIPE_FIELDS = tuple(item.name for item in fields(TinyBertRecoveryConfig) if item.name not in RUN_FIELDS + ("run_id", "results_root", "checkpoint_root"))


@dataclass
class TinyBertMatrixConfig:
    matrix_id: str = "tinybert-sst2-lora-matrix"
    recipe: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    transfers: list = field(default_factory=list)
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.matrix_id):
            raise ValueError("matrix_id must be a simple directory name")
        if not isinstance(self.recipe, dict) or set(self.recipe) - set(RECIPE_FIELDS):
            raise ValueError(f"recipe may only set {RECIPE_FIELDS}")
        if not isinstance(self.runs, list) or not self.runs or any(not isinstance(run, dict) or set(run) != set(RUN_FIELDS) for run in self.runs):
            raise ValueError(f"runs must be a nonempty list of objects with exactly {RUN_FIELDS}")
        for transfer in self.transfers:
            if not isinstance(transfer, dict) or not {"adapter_run", "base_kind", "pattern_id"} <= set(transfer) <= {"adapter_run", "base_kind", "pattern_id", "source_sweep_results"}:
                raise ValueError("Each transfer needs adapter_run, base_kind, pattern_id, and optionally source_sweep_results")
            index = transfer["adapter_run"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.runs) or self.runs[index]["base_kind"] != "clustered" or transfer["base_kind"] != "faulty":
                raise ValueError("Transfers apply a clustered-trained adapter from this matrix to a faulty base")


def matrix_plan(config):
    config.validate()
    plan = []
    for run in config.runs:
        recovery = TinyBertRecoveryConfig(**config.recipe, **run, run_id=recovery_run_id(run["base_kind"], run["condition"], run["pattern_id"], config.recipe.get("rank", 4), run["seed"]),
                                          results_root=config.results_root, checkpoint_root=config.checkpoint_root)
        recovery.validate()
        plan.append({"run_id": recovery.run_id, "config": asdict(recovery)})
    require(len({entry["run_id"] for entry in plan}) == len(plan), "Two runs share a run ID")
    return plan


SUMMARY_FIELDS = ["run_id", "base_kind", "condition", "pattern_id", "fault_probability", "seed", "selected_epoch", "split", "before_accuracy", "after_accuracy", "accuracy_gain_percentage_points", "elapsed_seconds"]


def summary_rows(rows):
    return [{key: str(row[key]) for key in SUMMARY_FIELDS} for row in rows]


def run_tinybert_matrix(config, project_root):
    root = Path(project_root).resolve()
    plan = matrix_plan(config)
    results_dir = root / config.results_root / config.matrix_id
    if results_dir.exists() or any((root / config.results_root / entry["run_id"]).exists() or (root / config.checkpoint_root / entry["run_id"]).exists() for entry in plan):
        raise FileExistsError("The matrix or one of its run directories already exists; declare a new matrix")
    results_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "plan.json", plan)
    rows, finished, transfers = [], [], []
    started = time.monotonic()
    write_json(results_dir / "status.json", {"status": "running", "completed_runs": 0, "planned_runs": len(plan), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    current = None
    try:
        for entry in plan:
            current = entry["run_id"]
            metrics = run_tinybert_recovery(TinyBertRecoveryConfig(**entry["config"]), root)
            finished.append(metrics)
            for split in metrics["after"]:
                rows.append({"run_id": metrics["run_id"], "base_kind": metrics["base_kind"], "condition": metrics["condition"], "pattern_id": metrics["pattern_id"],
                             "fault_probability": metrics["fault_probability"], "seed": entry["config"]["seed"], "selected_epoch": metrics["selected_epoch"], "split": split,
                             "before_accuracy": metrics["before"][split]["accuracy"], "after_accuracy": metrics["after"][split]["accuracy"],
                             "accuracy_gain_percentage_points": metrics["recovery"][split]["accuracy_gain_percentage_points"], "elapsed_seconds": metrics["elapsed_seconds"]})
            write_json(results_dir / "summary.json", rows)
            write_json(results_dir / "status.json", {"status": "running", "completed_runs": len(finished), "planned_runs": len(plan)})
        current = None
        for transfer in config.transfers:
            adapter = root / finished[transfer["adapter_run"]]["best_checkpoint"]
            result = evaluate_tinybert_adapter_on_base(adapter, transfer["base_kind"], transfer["pattern_id"], root, source_sweep_results=transfer.get("source_sweep_results"),
                                                       device=plan[transfer["adapter_run"]]["config"]["device"])
            transfers.append({"adapter_run_id": plan[transfer["adapter_run"]]["run_id"], **result})
            write_json(results_dir / "transfers.json", transfers)
        with (results_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows(rows))
        write_json(results_dir / "manifest.json", {"matrix_id": config.matrix_id, "run_policy": "Independent recovery runs in declared order; no skipping or resuming; first failure stops the matrix.",
                                                   "transfer_policy": "Clustered-trained adapters evaluated on faulty bases of the same encoding; evaluation only; not used for any selection.",
                                                   "reporting": "Each test set separately; never pooled; selection for checkpoint choice only.",
                                                   "run_checkpoints": {metrics["run_id"]: sha256_file(root / metrics["best_checkpoint"]) for metrics in finished},
                                                   "elapsed_seconds": time.monotonic() - started})
        write_json(results_dir / "status.json", {"status": "complete", "completed_runs": len(finished), "planned_runs": len(plan), "transfers": len(transfers),
                                                 "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"runs": finished, "transfers": transfers}
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_runs": len(finished),
                                                 "planned_runs": len(plan), "failed_run_id": current, "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_tinybert_matrix(TinyBertMatrixConfig(**json.loads(args.config.read_text())), project_root)
