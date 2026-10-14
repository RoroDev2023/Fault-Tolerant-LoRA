"""Apply the predeclared Part 7 rate-selection rule to the TinyBERT fault sweeps.

Rule (declared before Part 7): within each condition, select every nonzero rate
whose mean SELECTION-split accuracy drop from that condition's clustered
zero-fault reference is at least the threshold, plus the next lower rate below
the lowest selected one. Only selection-split aggregates are read; test-set
entries are never passed to the rule. Nothing is trained here.
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require

SELECTION = "selection"


@dataclass
class ProposalConfig:
    run_id: str = "tinybert-sst2-lora-rate-proposal"
    threshold_percentage_points: float = 3.0
    conditions: dict = field(default_factory=lambda: {
        "linear-only": ["results/tinybert-sst2-fault-sweep-seed42", "results/tinybert-sst2-fault-sweep-linear-high-seed72"],
        "linear-and-embeddings": ["results/tinybert-sst2-fault-sweep-linear-emb-seed1000"],
    })
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if isinstance(self.threshold_percentage_points, bool) or not isinstance(self.threshold_percentage_points, (int, float)) or not math.isfinite(self.threshold_percentage_points) or self.threshold_percentage_points <= 0:
            raise ValueError("threshold_percentage_points must be positive")
        if not isinstance(self.conditions, dict) or not self.conditions or not all(isinstance(paths, list) and paths for paths in self.conditions.values()):
            raise ValueError("conditions must map names to nonempty lists of sweep result directories")


def selection_ladder(aggregates):
    """(probability, mean selection drop, pattern IDs, run) for every nonzero rate, ascending; selection only."""
    ladder = []
    for run, aggregate in aggregates:
        for rate in aggregate["rates"]:
            if rate["probability"] > 0:
                ladder.append({"probability": rate["probability"], "selection_mean_drop_percentage_points": rate["splits"][SELECTION]["accuracy_drop_percentage_points"]["mean"],
                               "pattern_ids": rate.get("pattern_ids"), "seeds": rate["seeds"], "source_run": run})
    ladder.sort(key=lambda item: item["probability"])
    require(len({item["probability"] for item in ladder}) == len(ladder), "A rate appears in more than one sweep of the same condition")
    return ladder


def apply_rule(ladder, threshold):
    qualifying = [index for index, item in enumerate(ladder) if item["selection_mean_drop_percentage_points"] >= threshold]
    if not qualifying:
        return [], None
    neighbor = qualifying[0] - 1 if qualifying[0] > 0 else None
    chosen = sorted(set(qualifying) | ({neighbor} if neighbor is not None else set()))
    return [{**ladder[index], "reason": "selection drop at or above threshold" if index in qualifying else "next lower rate below the lowest qualifying rate"} for index in chosen], neighbor


def run_proposal(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Proposal directory already exists; choose a new run_id")
    inputs, proposal = {}, {}
    for condition, directories in config.conditions.items():
        aggregates = []
        for directory in directories:
            path = root / directory
            require(json.loads((path / "status.json").read_text())["status"] == "complete" and json.loads((path / "verification.json").read_text())["status"] == "passed", f"Sweep is incomplete or unverified: {directory}")
            aggregate = json.loads((path / "aggregate.json").read_text())
            require(SELECTION in aggregate["splits"], f"Sweep has no selection split: {directory}")
            # Only the selection split is kept; test-set aggregates never reach the rule.
            aggregates.append((directory, {"rates": [{key: value for key, value in rate.items() if key != "splits"} | {"splits": {SELECTION: rate["splits"][SELECTION]}} for rate in aggregate["rates"]]}))
            for name in ("aggregate", "status", "verification"):
                inputs[f"{directory}/{name}.json"] = sha256_file(path / f"{name}.json")
        ladder = selection_ladder(aggregates)
        chosen, neighbor = apply_rule(ladder, config.threshold_percentage_points)
        proposal[condition] = {"ladder": ladder, "selected_rates": chosen, "num_qualifying": sum(item["reason"].startswith("selection drop") for item in chosen),
                               "neighbor_included": neighbor is not None,
                               "note": None if chosen else f"No rate reached a {config.threshold_percentage_points} pp mean selection drop; the rule selects nothing for this condition."}
    results_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "inputs_sha256.json", inputs)
    write_json(results_dir / "proposal.json", {
        "rule": f"Per condition: every nonzero rate with mean selection-split accuracy drop >= {config.threshold_percentage_points} pp from that condition's clustered zero-fault reference, plus the next lower rate below the lowest such rate.",
        "data_used": "selection-split aggregates only; no test-set value is read by the rule", "training_run": False, "conditions": proposal,
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(results_dir / "status.json", {"status": "complete"})
    return proposal


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2_lora_rate_proposal.json")
    args = parser.parse_args()
    print(json.dumps(run_proposal(ProposalConfig(**json.loads(args.config.read_text())), project_root), indent=2))
