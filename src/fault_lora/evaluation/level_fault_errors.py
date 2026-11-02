"""Step 13b diagnostic: decoded weight error of saved level-dependent patterns, per mapping (reads saved files only).

For every faulted cell, the weight change is codebook[after] - codebook[before]
through that encoding's level_to_index. It is reported in raw units and in
units of the tensor's median sorted-codebook neighbor gap, so tensors with
different scales can be pooled.
"""

import argparse
import json
from pathlib import Path

import torch

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.fault_sweep import summarize_values
from fault_lora.memory.level_faults import summarize_level_fault_pattern


def pattern_errors(encoding, pattern):
    """Totals over faulted cells: count, sum |dw|, sum dw^2, and the same in median-gap units; gap-size histogram."""
    summarize_level_fault_pattern(encoding, pattern)
    totals = {"faulted_cells": 0, "sum_abs_change": 0.0, "sum_squared_change": 0.0, "sum_abs_change_gap_units": 0.0, "sum_squared_change_gap_units": 0.0}
    jumps = {}
    for name, entry in pattern["tensors"].items():
        record = encoding["clustered_tensors"][name]
        codebook = record["codebook"].double()
        gap = (codebook[1:] - codebook[:-1]).median().item()
        old = record["level_to_index"][entry["before_levels"].long()]
        new = record["level_to_index"][entry["after_levels"].long()]
        change = codebook[new] - codebook[old]
        totals["faulted_cells"] += change.numel()
        totals["sum_abs_change"] += change.abs().sum().item()
        totals["sum_squared_change"] += change.pow(2).sum().item()
        totals["sum_abs_change_gap_units"] += (change.abs() / gap).sum().item()
        totals["sum_squared_change_gap_units"] += (change / gap).pow(2).sum().item()
        for distance, count in zip(*torch.unique((new - old).abs(), return_counts=True)):
            jumps[int(distance)] = jumps.get(int(distance), 0) + int(count)
    return {**totals, "cluster_distance_histogram": dict(sorted(jumps.items()))}


def run(project_root, sweep_results, run_id):
    root = Path(project_root)
    source = root / sweep_results
    require(json.loads((source / "verification.json").read_text())["status"] == "passed", "Step 13b sweep is not verified")
    output = root / "results" / run_id
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new run_id")
    manifest = json.loads((source / "manifest.json").read_text())
    encodings = {}
    for mapping, item in manifest["encodings"].items():
        require(sha256_file(root / item["path"]) == item["sha256"], f"{mapping} encoding changed")
        encodings[mapping] = torch.load(root / item["path"], map_location="cpu", weights_only=False)
    records = json.loads((source / "measurements.json").read_text())
    rows = []
    for record in records:
        require(sha256_file(root / record["pattern"]) == record["pattern_sha256"], f"Pattern changed: {record['record_id']}")
        errors = pattern_errors(encodings[record["mapping"]], torch.load(root / record["pattern"], map_location="cpu", weights_only=True))
        cells = record["counts"]["eligible_cells"]
        rows.append({key: record[key] for key in ("record_id", "sigma_over_window", "condition_id", "mapping", "level0_safe", "seed")}
                    | errors | {"eligible_cells": cells, "mean_squared_change_per_cell_gap_units": errors["sum_squared_change_gap_units"] / cells,
                               "mean_abs_change_per_fault_gap_units": errors["sum_abs_change_gap_units"] / errors["faulted_cells"],
                               "share_faults_beyond_one_cluster": 1 - errors["cluster_distance_histogram"].get(1, 0) / errors["faulted_cells"],
                               "test_accuracy_no_repair": record["no_repair"]["test"]["accuracy"]})
    groups = []
    for sigma in dict.fromkeys(row["sigma_over_window"] for row in rows):
        for condition in dict.fromkeys(row["condition_id"] for row in rows):
            group = [row for row in rows if row["sigma_over_window"] == sigma and row["condition_id"] == condition]
            groups.append({"sigma_over_window": sigma, "condition_id": condition, "num_patterns": len(group),
                           **{key: summarize_values([row[key] for row in group]) for key in (
                               "mean_squared_change_per_cell_gap_units", "mean_abs_change_per_fault_gap_units", "share_faults_beyond_one_cluster", "sum_squared_change")}})
    output.mkdir(parents=True)
    write_json(output / "pattern_errors.json", rows)
    write_json(output / "aggregate.json", {"groups": groups, "definitions": __doc__,
                                           "inputs": {"sweep_results": sweep_results, "measurements_sha256": sha256_file(source / "measurements.json"),
                                                      "source_sha256": {p: sha256_file(root / p) for p in ("src/fault_lora/evaluation/level_fault_errors.py", "scripts/analyze_level_fault_errors.py")}}})
    for group in groups:
        print(json.dumps({k: (round(v["mean"], 4) if isinstance(v, dict) else v) for k, v in group.items()}))
    return groups


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-results", default="results/resnet18-cifar10-level-faults-seed1300")
    parser.add_argument("--run-id", default="resnet18-cifar10-level-faults-seed1300-weight-errors")
    args = parser.parse_args()
    run(project_root, args.sweep_results, args.run_id)
