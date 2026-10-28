"""Evaluation-only proof that the TinyBERT results reproduce in this checkout.

The TinyBERT inputs were copied here byte-identically, and their source-hash checks
now use the shared accepted-change rule. This run re-runs the saved TinyBERT
verifiers and reloads saved adapters on their own bases, requiring the exact saved
correct counts with the adapter and with its output zeroed (which reproduces the
clean, clustered, or faulty base). Nothing is trained, selected, or written outside
its own run directory.
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.source_hashes import compare_source_hashes, unique_changes
from fault_lora.evaluation.tinybert_baseline import TinyBertConfig, verify_tinybert_baseline
from fault_lora.evaluation.tinybert_clean_eval import CleanEvalConfig, verify_clean_eval
from fault_lora.evaluation.tinybert_condition_sweep import ConditionSweepConfig, verify_condition_sweep
from fault_lora.evaluation.tinybert_fault_sweep import TinyBertSweepConfig, verify_tinybert_sweep
from fault_lora.training.baseline import choose_device
from fault_lora.training.tinybert_recovery import TinyBertRecoveryConfig, evaluate_tinybert_adapter_on_base, load_tinybert_recovered_model, verify_tinybert_recovery


@dataclass
class TinyBertRegressionConfig:
    run_id: str = "tinybert-sst2-dependency-regression-v1"
    step10a_results: str = "results/tinybert-sst2-clean-seed42"
    clean_eval_results: str = "results/tinybert-sst2-clean-twotest-seed42"
    sweep_results: list = field(default_factory=lambda: ["results/tinybert-sst2-fault-sweep-seed42"])
    condition_sweep_results: list = field(default_factory=lambda: ["results/tinybert-sst2-fault-sweep-linear-high-seed72", "results/tinybert-sst2-fault-sweep-linear-emb-seed1000"])
    adapter_runs: list = field(default_factory=lambda: [
        "tinybert-sst2-lora-clean-none-none-r4-train42", "tinybert-sst2-lora-clustered-lin-none-r4-train42", "tinybert-sst2-lora-clustered-linemb-none-r4-train42",
        "tinybert-sst2-lora-faulty-lin-rate11-seed92-r4-train42", "tinybert-sst2-lora-faulty-linemb-rate07-seed1030-r4-train42-mps"])
    device: str = "mps"
    results_root: str = "results"

    def validate(self):
        names = [self.run_id] + self.adapter_runs
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names) or len(set(self.adapter_runs)) != len(self.adapter_runs):
            raise ValueError("run IDs must be distinct simple names")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def attempt(function):
    try:
        return {"passed": True, **(function() or {})}
    except Exception as error:   # Record every failure rather than stopping at the first.
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}


def compare(measured, saved):
    rows = {name: {"measured_num_correct": measured[name]["num_correct"], "saved_num_correct": saved[name]["num_correct"],
                   "cross_entropy_difference": measured[name]["cross_entropy_loss"] - saved[name]["cross_entropy_loss"]} for name in saved}
    return rows, all(item["measured_num_correct"] == item["saved_num_correct"] for item in rows.values())


def run_tinybert_regression(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Regression directory already exists; choose a new run_id")
    results_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/check_tinybert_dependency_regression.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device),
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "numpy")}}
        write_json(results_dir / "environment.json", environment)
        read = lambda directory: json.loads((root / directory / "config.json").read_text())
        checks = {"verify_tinybert_baseline:" + config.step10a_results: attempt(lambda: verify_tinybert_baseline(root, TinyBertConfig(**read(config.step10a_results)))),
                  "verify_clean_eval:" + config.clean_eval_results: attempt(lambda: verify_clean_eval(root, CleanEvalConfig(**read(config.clean_eval_results))))}
        for directory in config.sweep_results:
            checks["verify_tinybert_sweep:" + directory] = attempt(lambda directory=directory: verify_tinybert_sweep(root, TinyBertSweepConfig(**read(directory))))
        for directory in config.condition_sweep_results:
            checks["verify_condition_sweep:" + directory] = attempt(lambda directory=directory: verify_condition_sweep(root, ConditionSweepConfig(**read(directory))))
        evaluations = {}
        for run_id in config.adapter_runs:
            run_dir = root / config.results_root / run_id
            checks[f"verify_tinybert_recovery:{run_id}"] = attempt(lambda run_dir=run_dir: verify_tinybert_recovery(root, TinyBertRecoveryConfig(**json.loads((run_dir / "config.json").read_text()))))
            metrics = json.loads((run_dir / "metrics.json").read_text())
            checkpoint = root / metrics["best_checkpoint"]
            require(sha256_file(checkpoint) == metrics["best_checkpoint_sha256"], f"Saved adapter changed: {run_id}")
            print(json.dumps({"event": "tinybert_adapter_regression_start", "run_id": run_id}), flush=True)
            checks[f"load_tinybert_recovered_model:{run_id}"] = attempt(lambda checkpoint=checkpoint: {"selected_epoch": load_tinybert_recovered_model(checkpoint, root)[1]["epoch"]})
            measured = evaluate_tinybert_adapter_on_base(checkpoint, metrics["base_kind"], metrics["pattern_id"], root, device=str(device))
            tests = [name for name in metrics["after"] if name in measured["with_adapter"]]
            after, after_ok = compare(measured["with_adapter"], {name: metrics["after"][name] for name in tests})
            before, before_ok = compare(measured["without_adapter"], {name: metrics["before"][name] for name in tests})
            evaluations[run_id] = {"base_kind": metrics["base_kind"], "condition": metrics.get("condition"), "pattern_id": metrics["pattern_id"], "selected_epoch": metrics["selected_epoch"],
                                   "with_adapter": after, "adapter_output_zeroed": before, "passed": after_ok and before_ok}
            print(json.dumps({"event": "tinybert_adapter_regression_complete", "run_id": run_id, "passed": evaluations[run_id]["passed"]}), flush=True)
        write_json(results_dir / "regression.json", evaluations)
        write_json(results_dir / "reverification.json", checks)
        accepted = compare_source_hashes(root, source_code, "Source code changed during the regression run")
        all_changes = [change for item in checks.values() for change in item.get("accepted_source_hash_changes", [])] + accepted
        verification = {"status": "passed" if all(item["passed"] for item in checks.values()) and all(item["passed"] for item in evaluations.values()) else "failed",
                        "exact_counts": {name: item["passed"] for name, item in evaluations.items()}, "reverification": {name: item["passed"] for name, item in checks.items()},
                        "accepted_source_hash_changes_reported_by_checks": unique_changes(all_changes), "training_performed": False, "saved_results_modified": False}
        write_json(results_dir / "verification.json", verification)
        write_json(results_dir / "manifest.json", {"run_id": config.run_id, "purpose": "Evaluation-only TinyBERT regression after copying inputs and moving hash checks onto the shared rule.",
                                                   "count_rule": "Correct counts on the selection split and both test sets must equal the saved values exactly, with the adapter and with its output zeroed.",
                                                   "environment": environment, "elapsed_seconds": time.monotonic() - started})
        write_json(results_dir / "status.json", {"status": "complete" if verification["status"] == "passed" else "failed", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "tinybert_regression_complete", "verification": verification["status"], "elapsed_seconds": time.monotonic() - started}), flush=True)
        require(verification["status"] == "passed", f"TinyBERT regression failed; see {results_dir / 'verification.json'}")
        return verification
    except BaseException as error:
        if json.loads((results_dir / "status.json").read_text())["status"] == "running":
            write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tinybert_sst2_dependency_regression.json"))
    args = parser.parse_args()
    run_tinybert_regression(TinyBertRegressionConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
