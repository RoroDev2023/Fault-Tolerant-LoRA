"""Step 12c: TinyBERT LoRA recovery on the codebook-retrained baseline, with two adapter placements.

Each run is an ordinary `run_tinybert_recovery` run (same recipe, checks,
selection, and verification); this module supplies what Step 10d/10e lacked:

- base kinds "retrained-clustered" (the Step 12a encoding with zero faults) and
  "retrained-faulty" (that encoding read through a Step 12b retrained-bound
  pattern). For these, `source_sweep_results` names the Step 12b sweep; the
  Step 12b records, its audit, the pinned Step 12a encoding, the Step 10c source
  chain, and the selected pattern are verified before use;
- `adapter_placement`: "linear" (the 26 nn.Linear layers, exactly Step 10e) or
  "linear+emb" (the same, plus EmbeddingLoRALayer on the word, position, and
  token-type embeddings);
- the declared Step 12c matrix: independent runs, then clustered → faulty
  transfer evaluations of each placement's retrained-clustered adapters.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import torch

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.source_hashes import compare_source_hashes
from fault_lora.evaluation.tinybert_retrained_faults import TinyBertRetrainedSweepConfig, load_group
from fault_lora.memory.faults import encoding_fingerprint, summarize_fault_pattern
from fault_lora.training.tinybert_recovery import CONDITIONS, TinyBertRecoveryConfig, evaluate_tinybert_adapter_on_base, run_tinybert_recovery

RETRAINED_KINDS = ("retrained-clustered", "retrained-faulty")
PLACEMENTS = ("linear", "linear+emb")
PLACEMENT_TAGS = {"linear": "linear", "linear+emb": "linear-emb"}


def retrained_recovery_run_id(adapter_placement, base_kind, pattern_id, rank, seed):
    return f"tinybert-sst2-lora12c-{PLACEMENT_TAGS[adapter_placement]}-{base_kind}-{pattern_id or 'none'}-r{rank}-train{seed}"


@dataclass
class TinyBertRetrainedRecoveryConfig(TinyBertRecoveryConfig):
    adapter_placement: str = "linear"

    def validate(self):
        if self.adapter_placement not in PLACEMENTS:
            raise ValueError(f"adapter_placement must be one of {PLACEMENTS}")
        if self.base_kind in RETRAINED_KINDS:
            if self.condition not in CONDITIONS:
                raise ValueError(f"Retrained bases need a condition in {CONDITIONS}")
            # The remaining rules are the clustered/faulty ones.
            TinyBertRecoveryConfig.validate(replace(self, base_kind="faulty" if self.base_kind == "retrained-faulty" else "clustered"))
        else:
            TinyBertRecoveryConfig.validate(self)


def prepare_retrained_sources(root, config):
    """Same return layout as tinybert_recovery.prepare_recovery_sources, for the Step 12a encoding and Step 12b patterns."""
    directory = root / config.source_sweep_results
    read = lambda name: json.loads((directory / name).read_text())
    require(read("status.json")["status"] == "complete" and read("verification.json")["status"] == "passed", "Step 12b sweep is incomplete or unverified")
    sweep_config = TinyBertRetrainedSweepConfig(**read("config.json"))
    sweep_config.validate()
    groups = [group for group in sweep_config.groups if group["condition"] == config.condition]
    require(len(groups) == 1 and groups[0]["audit"], f"Step 12b has no single audited group for {config.condition}")
    group = groups[0]
    gid = group["group_id"]
    finetune_dir = root / sweep_config.finetune_results
    require(json.loads((finetune_dir / "status.json").read_text())["status"] == "complete" and json.loads((finetune_dir / "verification.json").read_text())["status"] == "passed",
            "Step 12a run is incomplete or unverified")
    data = load_group(root, sweep_config, group, json.loads((finetune_dir / "manifest.json").read_text()))
    encoding, sources = data["retrained"], data["sources"]
    fingerprint = encoding_fingerprint(encoding)
    require(read("manifest.json")["fingerprints"][gid]["retrained"] == fingerprint, "Step 12b swept another retrained encoding")
    require(read("audit.json")["status"] == "passed" and gid in read("audit.json")["checks"], "Step 12b audit did not pass for this group")
    records = [record for record in read("measurements.json") if record["group_id"] == gid]
    identity = [item for item in read("pattern_identity.json") if item["group_id"] == gid]
    require([record["pattern_id"] for record in records] == [item["pattern_id"] for item in identity] and records and records[0]["probability"] == 0,
            "Step 12b records are incomplete or reordered")
    require(all(item["identical"] and not item["differences"] for item in identity), "A Step 12b pattern differed from its source pattern")
    splits = list(sources.datasets)
    clean = read("aggregate.json")["clean"]
    require(all(clean[name]["num_correct"] == data["references"][name]["clean"]["num_correct"] for name in splits), "Step 12b clean replay differs from the source sweep")
    zero = records[0]["models"]["retrained"]
    references = {name: {"clean": data["references"][name]["clean"], "clustered_no_faults": zero[name]} for name in splits}
    accepted = compare_source_hashes(root, read("source_sha256.json"), "Previously used Step 12b sweep code changed")
    sources.accepted_source_hash_changes = list(getattr(sources, "accepted_source_hash_changes", [])) + accepted
    for key in ("retrained_encoding", "retrained_dense"):
        sources.paths[key], sources.hashes[key] = data["paths"][f"{gid}:{key}"], data["hashes"][f"{gid}:{key}"]
    files = ["config.json", "status.json", "verification.json", "manifest.json", "audit.json", "measurements.json", "pattern_identity.json", "aggregate.json", "source_sha256.json"]
    pattern = None
    if config.base_kind == "retrained-faulty":
        matches = [(record, item) for record, item in zip(records, identity) if record["pattern_id"] == config.pattern_id]
        require(len(matches) == 1 and matches[0][0]["probability"] > 0, "Recovery requires a saved nonzero Step 12b pattern")
        saved, item = matches[0]
        require(saved == read(f"patterns/{gid}-{config.pattern_id}.json") and saved["retrained_pattern"] == item["retrained_pattern"], "Individual and combined Step 12b records disagree")
        files.append(f"patterns/{gid}-{config.pattern_id}.json")
        sources.paths["selected_pattern"], sources.hashes["selected_pattern"] = root / item["retrained_pattern"], item["retrained_pattern_sha256"]
        require(sha256_file(sources.paths["selected_pattern"]) == item["retrained_pattern_sha256"], "Selected Step 12b pattern file changed")
        pattern = torch.load(sources.paths["selected_pattern"], map_location="cpu", weights_only=True)
        require(pattern["source_fingerprint"] == fingerprint and pattern["seed"] == saved["seed"] and pattern["fault_probability"] == saved["probability"],
                "Selected pattern is bound to another encoding, seed, or probability")
        require({key: value for key, value in summarize_fault_pattern(encoding, pattern).items() if key != "layers"} == saved["counts"], "Selected pattern counts differ from Step 12b")
        record = {"pattern_id": config.pattern_id, "seed": saved["seed"], "probability": saved["probability"], "counts": saved["counts"], "measurements": saved["models"]["retrained"],
                  "pattern": item["retrained_pattern"], "pattern_sha256": item["retrained_pattern_sha256"]}
    else:
        record = {"pattern_id": None, "seed": None, "probability": 0.0, "counts": {"actual_corrupted_cells": 0, "realized_fault_rate": 0.0, "eligible_cells": records[0]["counts"]["eligible_cells"]},
                  "measurements": zero}
    for name in files:
        sources.paths[f"retrained_sweep_{name}"], sources.hashes[f"retrained_sweep_{name}"] = directory / name, sha256_file(directory / name)
    for name in ("status.json", "verification.json", "manifest.json", "encodings.json"):
        sources.paths[f"retrained_finetune_{name}"], sources.hashes[f"retrained_finetune_{name}"] = finetune_dir / name, sha256_file(finetune_dir / name)
    return {"condition": config.condition, "sources": sources, "encoding": encoding, "record": record, "references": references, "pattern": pattern}


RUN_FIELDS = ("adapter_placement", "base_kind", "condition", "pattern_id", "source_sweep_results", "seed")
RECIPE_FIELDS = tuple(item.name for item in fields(TinyBertRetrainedRecoveryConfig) if item.name not in RUN_FIELDS + ("run_id", "results_root", "checkpoint_root"))
SUMMARY_FIELDS = ["run_id", "adapter_placement", "base_kind", "condition", "pattern_id", "fault_probability", "seed", "selected_epoch", "split", "before_accuracy", "after_accuracy",
                  "accuracy_gain_percentage_points", "elapsed_seconds"]


@dataclass
class TinyBertRetrainedMatrixConfig:
    matrix_id: str = "tinybert-sst2-lora12c-matrix"
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
            if not isinstance(transfer, dict) or set(transfer) != {"adapter_run", "pattern_id"}:
                raise ValueError("Each transfer needs exactly adapter_run and pattern_id")
            index = transfer["adapter_run"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.runs) or self.runs[index]["base_kind"] != "retrained-clustered":
                raise ValueError("Transfers apply a retrained-clustered adapter from this matrix to a retrained-faulty base")


def matrix_plan(config):
    config.validate()
    plan = []
    for run in config.runs:
        recovery = TinyBertRetrainedRecoveryConfig(**config.recipe, **run, results_root=config.results_root, checkpoint_root=config.checkpoint_root,
                                                   run_id=retrained_recovery_run_id(run["adapter_placement"], run["base_kind"], run["pattern_id"], config.recipe.get("rank", 4), run["seed"]))
        recovery.validate()
        plan.append({"run_id": recovery.run_id, "config": asdict(recovery)})
    require(len({entry["run_id"] for entry in plan}) == len(plan), "Two runs share a run ID")
    return plan


def run_retrained_matrix(config, project_root):
    """Declared runs in order, then transfers; existing directories are refused and the first failure stops the matrix."""
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
    write_json(results_dir / "status.json", {"status": "running", "completed_runs": 0, "planned_runs": len(plan), "planned_transfers": len(config.transfers),
                                             "started_at_utc": datetime.now(timezone.utc).isoformat()})
    current = None
    try:
        for entry in plan:
            current = entry["run_id"]
            metrics = run_tinybert_recovery(TinyBertRetrainedRecoveryConfig(**entry["config"]), root)
            finished.append(metrics)
            for split in metrics["after"]:
                rows.append({"run_id": metrics["run_id"], "adapter_placement": entry["config"]["adapter_placement"], "base_kind": metrics["base_kind"], "condition": metrics["condition"],
                             "pattern_id": metrics["pattern_id"], "fault_probability": metrics["fault_probability"], "seed": entry["config"]["seed"], "selected_epoch": metrics["selected_epoch"],
                             "split": split, "before_accuracy": metrics["before"][split]["accuracy"], "after_accuracy": metrics["after"][split]["accuracy"],
                             "accuracy_gain_percentage_points": metrics["recovery"][split]["accuracy_gain_percentage_points"], "elapsed_seconds": metrics["elapsed_seconds"]})
            write_json(results_dir / "summary.json", rows)
            write_json(results_dir / "status.json", {"status": "running", "completed_runs": len(finished), "planned_runs": len(plan), "planned_transfers": len(config.transfers),
                                                     "last_run_id": current})
        current = None
        for index, transfer in enumerate(config.transfers):
            adapter = root / finished[transfer["adapter_run"]]["best_checkpoint"]
            result = evaluate_tinybert_adapter_on_base(adapter, "retrained-faulty", transfer["pattern_id"], root, device=plan[transfer["adapter_run"]]["config"]["device"])
            transfers.append({"adapter_run_id": plan[transfer["adapter_run"]]["run_id"], "adapter_placement": plan[transfer["adapter_run"]]["config"]["adapter_placement"], **result})
            write_json(results_dir / "transfers.json", transfers)
            write_json(results_dir / "status.json", {"status": "running", "completed_runs": len(finished), "planned_runs": len(plan), "completed_transfers": index + 1,
                                                     "planned_transfers": len(config.transfers)})
            print(json.dumps({"event": "transfer_complete", "index": index + 1, "adapter_run_id": transfers[-1]["adapter_run_id"], "pattern_id": transfer["pattern_id"],
                              "gain": {name: round(value, 2) for name, value in result["accuracy_gain_percentage_points"].items()}}), flush=True)
        with (results_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows([{key: str(row[key]) for key in SUMMARY_FIELDS} for row in rows])
        write_json(results_dir / "manifest.json", {"matrix_id": config.matrix_id, "run_policy": "Independent recovery runs in declared order; no skipping or resuming; first failure stops the matrix.",
                                                   "transfer_policy": "Each placement's retrained-clustered adapters evaluated on retrained-faulty bases of the same encoding; evaluation only; not used for any selection.",
                                                   "reporting": "Each test set separately; never pooled; selection for checkpoint choice only.",
                                                   "run_checkpoints": {metrics["run_id"]: sha256_file(root / metrics["best_checkpoint"]) for metrics in finished},
                                                   "elapsed_seconds": time.monotonic() - started})
        write_json(results_dir / "status.json", {"status": "complete", "completed_runs": len(finished), "planned_runs": len(plan), "transfers": len(transfers),
                                                 "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {"runs": finished, "transfers": transfers}
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_runs": len(finished),
                                                 "planned_runs": len(plan), "completed_transfers": len(transfers), "failed_run_id": current, "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_retrained_matrix(TinyBertRetrainedMatrixConfig(**json.loads(args.config.read_text())), project_root)
