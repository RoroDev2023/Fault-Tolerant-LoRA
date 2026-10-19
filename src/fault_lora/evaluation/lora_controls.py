"""Controlled comparison of LoRA recovery on clean, clustered, and faulty bases; analysis only.

Reads the finished runs of one or more recovery matrices, the BatchNorm-recalibration
baseline, and the Step 7 sweep. Nothing is trained or evaluated here; every quantity is computed
from saved per-run measurements under a design declared in the configuration.
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
import statistics
import tempfile
import time

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.fault_sweep import summarize_values
from fault_lora.evaluation.source_hashes import compare_source_hashes, unique_changes
from fault_lora.training.recovery_matrix import MatrixConfig, matrix_plan, summary_row, verify_run


SPLITS = ("validation", "test")
CONDITION_FIELDS = ["split", "condition", "fault_probability", "num_runs", "num_patterns", "accuracy_before_mean", "accuracy_before_sample_std", "accuracy_after_mean", "accuracy_after_sample_std",
                    "accuracy_gain_percentage_points_mean", "accuracy_gain_percentage_points_sample_std", "normalized_recovery_mean", "normalized_recovery_sample_std",
                    "fault_specific_gain_percentage_points_mean", "fault_specific_gain_percentage_points_sample_std", "bn_recalibration_accuracy_mean", "bn_recalibration_accuracy_sample_std",
                    "lora_minus_bn_recalibration_percentage_points_mean", "lora_minus_bn_recalibration_percentage_points_sample_std"]
DEFINITIONS = {
    "variability": "Sample standard deviation (ddof=1) over the listed runs or patterns; null for one value. Not a confidence interval.",
    "condition": "accuracy_before/after pool every run of the condition: training seeds for clean and clustered bases, patterns x training seeds for a faulty rate.",
    "extra_training_effect": "LoRA(clean) minus the same clean base before LoRA, per training seed, in percentage points.",
    "normalized_recovery": "R = (after - before) / (clean - before) per run, with clean the Step 7 clean replay accuracy of the split. Undefined (null) when before >= clean, so always for a clean base.",
    "fault_specific_gain": "LoRA(faulty) accuracy minus the mean LoRA(clustered) accuracy over all control seeds, per faulty run, in percentage points. seed_matched subtracts the LoRA(clustered) run with the same training seed instead.",
    "bn_recalibration_vs_lora": "Per pattern: BatchNorm-recalibrated accuracy versus the mean LoRA accuracy over that pattern's training seeds. Statistics are across patterns.",
    "selection": "Design, seeds, patterns, and formulas fixed before the runs. Test accuracy is reported after per-run validation-only checkpoint selection and chooses nothing.",
}


@dataclass
class ControlsConfig:
    run_id: str = "resnet18-cifar10-lora-controls"
    source_matrices: list = field(default_factory=lambda: [{"results": "results/resnet18-cifar10-lora-controls-matrix", "status": "complete"}])
    source_bn_results: str = "results/resnet18-cifar10-bn-recalibration-seed42"
    source_sweep_results: str = "results/resnet18-cifar10-fault-sweep-seed42"
    fault_probabilities: list = field(default_factory=lambda: [.01, .05, .1])
    control_seeds: list = field(default_factory=lambda: [42, 43, 44])
    faulty_seeds: list = field(default_factory=lambda: [42, 43])
    rank: int = 4
    alpha: float = 4.0
    results_root: str = "results"
    sources_allowed_to_differ: list = field(default_factory=list)
    step8_results: str | None = None
    step8_rerun_id: str | None = None
    step8_tolerance_percentage_points: float = .3

    def validate(self):
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if not isinstance(self.source_matrices, list) or not self.source_matrices or len({item.get("results") for item in self.source_matrices if isinstance(item, dict)}) != len(self.source_matrices):
            raise ValueError("source_matrices must be a nonempty list of distinct matrices")
        for item in self.source_matrices:
            interrupted = item.get("status") == "interrupted"
            if set(item) != ({"results", "status", "completed_runs"} if interrupted else {"results", "status"}) or item["status"] not in ("complete", "interrupted") \
                    or not isinstance(item["results"], str) or "_aborted" in Path(item["results"]).parts:
                raise ValueError("Each source matrix declares results and status; an interrupted one also declares completed_runs; archived runs are never inputs")
            if interrupted and (isinstance(item["completed_runs"], bool) or not isinstance(item["completed_runs"], int) or item["completed_runs"] < 1):
                raise ValueError("completed_runs must be a positive integer")
        if not isinstance(self.sources_allowed_to_differ, list) or any(not isinstance(name, str) for name in self.sources_allowed_to_differ):
            raise ValueError("sources_allowed_to_differ must be a list of repository paths")
        if (self.step8_results is None) != (self.step8_rerun_id is None):
            raise ValueError("step8_results and step8_rerun_id must be declared together")
        if isinstance(self.step8_tolerance_percentage_points, bool) or not isinstance(self.step8_tolerance_percentage_points, Real) or not self.step8_tolerance_percentage_points >= 0:
            raise ValueError("Invalid Step 8 tolerance")
        if not self.fault_probabilities or any(isinstance(p, bool) or not isinstance(p, Real) or not 0 < p <= 1 for p in self.fault_probabilities) or self.fault_probabilities != sorted(set(self.fault_probabilities)):
            raise ValueError("fault_probabilities must be sorted, unique, nonzero probabilities")
        for name in ("control_seeds", "faulty_seeds"):
            seeds = getattr(self, name)
            if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds) or seeds != sorted(set(seeds)):
                raise ValueError(f"{name} must be sorted, unique integers")
        if len(self.control_seeds) < 2 or not set(self.faulty_seeds) <= set(self.control_seeds):
            raise ValueError("At least two control seeds are required and faulty seeds must also be control seeds")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1 or isinstance(self.alpha, bool) or not isinstance(self.alpha, Real) or self.alpha <= 0:
            raise ValueError("Invalid rank or alpha")


def expected_patterns(sweep_plan, config):
    patterns = {probability: [entry["pattern_id"] for entry in sweep_plan if entry["probability"] == probability] for probability in config.fault_probabilities}
    require(all(len(names) >= 2 for names in patterns.values()), "Every analyzed fault probability needs at least two saved sweep patterns")
    return patterns


def validate_design(rows, bn_records, sweep_records, sweep_plan, config):
    """The matrix and baseline must contain exactly the declared design, paired with Step 7."""
    config.validate()
    patterns = expected_patterns(sweep_plan, config)
    expected = [("clean", None, seed) for seed in config.control_seeds] + [("clustered", None, seed) for seed in config.control_seeds]
    expected += [("faulty", name, seed) for names in patterns.values() for name in names for seed in config.faulty_seeds]
    actual = [(row["base_kind"], row["pattern_id"], row["seed"]) for row in rows]
    require(len(set(actual)) == len(actual) and set(actual) == set(expected), "Matrix runs differ from the declared control design")
    require(all(row["rank"] == config.rank and row["alpha"] == config.alpha for row in rows), "Matrix runs use a different rank or alpha")
    saved = {record["pattern_id"]: record for record in sweep_records}
    for row in rows:
        if row["base_kind"] == "faulty":
            require(row["fault_probability"] == saved[row["pattern_id"]]["probability"] and all(
                row[f"{split}_accuracy_before"] == saved[row["pattern_id"]]["measurements"][split]["accuracy"] for split in SPLITS), f"Run is not paired with its Step 7 pattern: {row['run_id']}")
    wanted = ["clustered"] + [name for names in patterns.values() for name in names]
    require(sorted(record["condition_id"] for record in bn_records) == sorted(wanted), "BatchNorm baseline conditions differ from the declared control design")
    for record in bn_records:
        if record["base_kind"] == "faulty":
            require(all(record["before"][split]["accuracy"] == saved[record["pattern_id"]]["measurements"][split]["accuracy"] for split in SPLITS), f"BatchNorm baseline is not paired with Step 7: {record['condition_id']}")
    return patterns


def condition_summary(group, split, clean):
    runs = []
    for row in sorted(group, key=lambda row: (row["pattern_id"] or "", row["seed"])):
        before, after = row[f"{split}_accuracy_before"], row[f"{split}_accuracy_after"]
        runs.append({"run_id": row["run_id"], "pattern_id": row["pattern_id"], "seed": row["seed"], "accuracy_before": before, "accuracy_after": after,
                     "accuracy_gain_percentage_points": 100 * (after - before), "normalized_recovery": (after - before) / (clean - before) if clean > before else None})
    defined = [run["normalized_recovery"] for run in runs if run["normalized_recovery"] is not None]
    names = sorted({run["pattern_id"] for run in runs if run["pattern_id"] is not None})
    entry = {"num_runs": len(runs), "num_patterns": len(names), "seeds": sorted({run["seed"] for run in runs}),
             **{key: summarize_values([run[key] for run in runs]) for key in ("accuracy_before", "accuracy_after", "accuracy_gain_percentage_points")},
             "normalized_recovery": summarize_values(defined) if defined else None, "normalized_recovery_undefined_runs": len(runs) - len(defined)}
    if names:
        entry["accuracy_after_by_pattern"] = {name: statistics.mean(run["accuracy_after"] for run in runs if run["pattern_id"] == name) for name in names}
        entry["accuracy_after_pattern_means"] = summarize_values(list(entry["accuracy_after_by_pattern"].values()))
    entry["accuracy_after_by_seed"] = {str(seed): statistics.mean(run["accuracy_after"] for run in runs if run["seed"] == seed) for seed in entry["seeds"]}
    entry["runs"] = runs
    return entry


def analyze_split(rows, bn_records, sweep_aggregate, references, patterns, split):
    clean = references[split]["clean"]["accuracy"]
    clustered = references[split]["clustered_no_faults"]["accuracy"]
    groups = {"clean": [row for row in rows if row["base_kind"] == "clean"], "clustered": [row for row in rows if row["base_kind"] == "clustered"]}
    require(all(row[f"{split}_accuracy_before"] == clean for row in groups["clean"]) and all(row[f"{split}_accuracy_before"] == clustered for row in groups["clustered"]),
            "Control runs do not start from the Step 7 clean and clustered references")
    conditions = [{"condition": name, "fault_probability": None if name == "clean" else 0.0, **condition_summary(group, split, clean)} for name, group in groups.items()]
    lora_clean, lora_clustered = conditions[0], conditions[1]
    clustered_mean = lora_clustered["accuracy_after"]["mean"]
    clustered_by_seed = {run["seed"]: run["accuracy_after"] for run in lora_clustered["runs"]}
    bn = {record["condition_id"]: record for record in bn_records}
    fault_specific, comparison = [], [{"condition": "clustered", "fault_probability": 0.0, "num_patterns": 0,
        "bn_recalibration_accuracy": summarize_values([bn["clustered"]["after"][split]["accuracy"]]), "lora_accuracy": lora_clustered["accuracy_after"],
        "lora_minus_bn_recalibration_percentage_points": summarize_values([100 * (clustered_mean - bn["clustered"]["after"][split]["accuracy"])]),
        "bn_recalibration_normalized_recovery": summarize_values([(bn["clustered"]["after"][split]["accuracy"] - clustered) / (clean - clustered)]) if clean > clustered else None}]
    for probability, names in patterns.items():
        entry = {"condition": "faulty", "fault_probability": probability, **condition_summary([row for row in rows if row["pattern_id"] in names], split, clean)}
        conditions.append(entry)
        versus_mean = [100 * (run["accuracy_after"] - clustered_mean) for run in entry["runs"]]
        matched = [100 * (run["accuracy_after"] - clustered_by_seed[run["seed"]]) for run in entry["runs"]]
        fault_specific.append({"fault_probability": probability, "num_runs": len(versus_mean), "lora_faulty_accuracy": entry["accuracy_after"], "lora_clustered_accuracy_mean": clustered_mean,
                               "gain_percentage_points": summarize_values(versus_mean), "seed_matched_gain_percentage_points": summarize_values(matched)})
        per_pattern = []
        for name in names:
            before, after = bn[name]["before"][split]["accuracy"], bn[name]["after"][split]["accuracy"]
            lora = entry["accuracy_after_by_pattern"][name]
            per_pattern.append({"pattern_id": name, "accuracy_before": before, "bn_recalibration_accuracy": after, "lora_accuracy_mean_over_seeds": lora,
                                "lora_minus_bn_recalibration_percentage_points": 100 * (lora - after), "bn_recalibration_normalized_recovery": (after - before) / (clean - before) if clean > before else None})
        recoveries = [item["bn_recalibration_normalized_recovery"] for item in per_pattern if item["bn_recalibration_normalized_recovery"] is not None]
        comparison.append({"condition": "faulty", "fault_probability": probability, "num_patterns": len(names),
                           "accuracy_before": summarize_values([item["accuracy_before"] for item in per_pattern]),
                           "bn_recalibration_accuracy": summarize_values([item["bn_recalibration_accuracy"] for item in per_pattern]),
                           "lora_accuracy": entry["accuracy_after"], "lora_accuracy_pattern_means": entry["accuracy_after_pattern_means"],
                           "lora_minus_bn_recalibration_percentage_points": summarize_values([item["lora_minus_bn_recalibration_percentage_points"] for item in per_pattern]),
                           "bn_recalibration_normalized_recovery": summarize_values(recoveries) if recoveries else None, "per_pattern": per_pattern})
    step7 = [{"fault_probability": rate["probability"], "num_patterns": rate["num_patterns"], "realized_fault_rate_mean": rate["realized_fault_rate"]["mean"], "accuracy": rate["splits"][split]["accuracy"]}
             for rate in sweep_aggregate["rates"]]
    require({rate["fault_probability"] for rate in step7} >= set(patterns), "Step 7 aggregate lacks an analyzed fault probability")
    return {"references": {"clean_accuracy": clean, "clustered_zero_fault_accuracy": clustered}, "conditions": conditions,
            "extra_training_effect_percentage_points": {"per_seed": {str(run["seed"]): run["accuracy_gain_percentage_points"] for run in lora_clean["runs"]},
                                                        **lora_clean["accuracy_gain_percentage_points"]},
            "fault_specific_gain": fault_specific, "bn_recalibration_vs_lora": comparison, "step7_fault_rates": step7}


def analyze_controls(rows, bn_records, sweep_records, sweep_plan, sweep_aggregate, references, config):
    patterns = validate_design(rows, bn_records, sweep_records, sweep_plan, config)
    return {"run_id": config.run_id, "design": {"rank": config.rank, "alpha": config.alpha, "control_seeds": config.control_seeds, "faulty_seeds": config.faulty_seeds,
                                                 "patterns": {str(probability): names for probability, names in patterns.items()}, "num_runs": len(rows)},
            "definitions": DEFINITIONS, "splits": {split: analyze_split(rows, bn_records, sweep_aggregate, references, patterns, split) for split in SPLITS}}


def condition_table(analysis):
    """One flat row per split and condition for the CSV summary."""
    table = []
    for split, result in analysis["splits"].items():
        specific = {item["fault_probability"]: item["gain_percentage_points"] for item in result["fault_specific_gain"]}
        baseline = {(item["condition"], item["fault_probability"]): item for item in result["bn_recalibration_vs_lora"]}
        for entry in result["conditions"]:
            other = baseline.get((entry["condition"], entry["fault_probability"]))
            pairs = {"accuracy_before": entry["accuracy_before"], "accuracy_after": entry["accuracy_after"], "accuracy_gain_percentage_points": entry["accuracy_gain_percentage_points"],
                     "normalized_recovery": entry["normalized_recovery"], "fault_specific_gain_percentage_points": specific.get(entry["fault_probability"]) if entry["condition"] == "faulty" else None,
                     "bn_recalibration_accuracy": other and other["bn_recalibration_accuracy"], "lora_minus_bn_recalibration_percentage_points": other and other["lora_minus_bn_recalibration_percentage_points"]}
            row = {"split": split, "condition": entry["condition"], "fault_probability": entry["fault_probability"], "num_runs": entry["num_runs"], "num_patterns": entry["num_patterns"]}
            for name, values in pairs.items():
                row[f"{name}_mean"], row[f"{name}_sample_std"] = (values["mean"], values["sample_std"]) if values else (None, None)
            table.append({name: "" if row[name] is None else str(row[name]) for name in CONDITION_FIELDS})
    return table


def plot_controls(result, output_dir, *, split="test"):
    """Accuracy versus actual fault rate: full range, and a zoom on the LoRA results."""
    orange, aqua, blue, ink, muted, grid = "#eb6834", "#1baf7a", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    with tempfile.TemporaryDirectory(prefix="fault-lora-controls-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            rate = {item["fault_probability"]: 100 * item["realized_fault_rate_mean"] for item in result["step7_fault_rates"]}
            conditions = {(item["condition"], item["fault_probability"]): item for item in result["conditions"]}
            lora = [conditions["clustered", 0.0]] + [item for item in result["conditions"] if item["condition"] == "faulty"]
            series = {
                "faulty": ([rate[item["fault_probability"]] for item in result["step7_fault_rates"]], [item["accuracy"] for item in result["step7_fault_rates"]]),
                "bn": ([rate[item["fault_probability"]] for item in result["bn_recalibration_vs_lora"]], [item["bn_recalibration_accuracy"] for item in result["bn_recalibration_vs_lora"]]),
                "lora": ([rate[item["fault_probability"]] for item in lora], [item["accuracy_after"] for item in lora])}
            styles = {"faulty": (orange, "o", "Faulty, no repair (Step 7)"), "bn": (aqua, "s", "Faulty + BatchNorm recalibration"), "lora": (blue, "D", "Faulty + LoRA")}
            references = [("Clean", result["references"]["clean_accuracy"], muted, "--"), ("Clustered, zero faults", result["references"]["clustered_zero_fault_accuracy"], muted, ":"),
                          ("LoRA on clean", conditions["clean", None]["accuracy_after"]["mean"], blue, "--"), ("LoRA on clustered", conditions["clustered", 0.0]["accuracy_after"]["mean"], blue, ":")]
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor="#fcfcfb")
            top = [100 * item["accuracy_after"][key] for item in lora for key in ("min", "max")] + [100 * value for _, value, _, _ in references if value >= min(item["accuracy_after"]["min"] for item in lora)]
            limits = [None, (math.floor(min(top) - .5), math.ceil(max(top) + .5))]
            for axis, limit, names in zip(axes, limits, (("faulty", "bn", "lora"), ("lora",))):
                axis.set_facecolor("#fcfcfb")
                for label, value, color, style in references:
                    if limit is None or limit[0] <= 100 * value <= limit[1]:
                        axis.axhline(100 * value, color=color, linestyle=style, linewidth=1.2, label=f"{label}: {100 * value:.2f}%", zorder=2)
                for name in names:
                    color, marker, label = styles[name]
                    xs, values = series[name]
                    axis.errorbar(xs, [100 * value["mean"] for value in values], yerr=[100 * (value["sample_std"] or 0) for value in values], color=color, marker=marker,
                                  markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=1.5, linewidth=2, capsize=3, elinewidth=1.2, label=label, zorder=4)
                if limit is not None:
                    for item in lora:
                        axis.scatter([rate[item["fault_probability"]]] * item["num_runs"], [100 * run["accuracy_after"] for run in item["runs"]], s=14, color=blue, alpha=.35, linewidths=0, zorder=3)
                    for position, (item, value) in enumerate(zip(lora, series["lora"][1])):
                        # Alternate sides so the labels at 0% and 1% cannot collide.
                        axis.annotate(f"{100 * value['mean']:.2f}%", (rate[item["fault_probability"]], 100 * value["mean"]), textcoords="offset points",
                                      xytext=(-8, 12) if position % 2 == 0 else (8, -16), ha="right" if position % 2 == 0 else "left", fontsize=8, color=ink)
                    axis.set_ylim(*limit)
                    axis.set_title("Zoom: LoRA runs and references", fontsize=11, color=ink)
                else:
                    axis.set_title(f"{split.capitalize()} accuracy, full range", fontsize=11, color=ink)
                axis.set_xlabel("Actual corrupted cells / eligible cells (%)", color=muted)
                axis.set_ylabel("Classification accuracy (%)", color=muted)
                axis.set_xticks([0, 1, 2, 5, 10])
                axis.grid(color=grid, linewidth=.8)
                axis.set_axisbelow(True)
                axis.tick_params(colors=muted)
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    axis.spines[side].set_color(grid)
                axis.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=ink)
            runs = sorted({item["num_runs"] for item in lora[1:]})
            fig.suptitle("ResNet-18 / CIFAR-10: LoRA recovery controls, adjacent-level faults, 16 clusters per tensor", fontsize=13, color=ink)
            fig.text(.5, .015, f"Means ± 1 sample SD. LoRA at 0%: clustered base, {lora[0]['num_runs']} training seeds; nonzero rates: {'/'.join(map(str, runs))} runs each (patterns × training seeds). "
                               "Step 7 and BatchNorm recalibration: across patterns. Small dots: individual LoRA runs.", ha="center", fontsize=8, color=muted)
            fig.tight_layout(rect=(0, .05, 1, .95))
            paths = [Path(output_dir) / f"lora_controls.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def step8_regression(original, rerun, tolerance):
    """Compare the rerun of the Step 8 pattern and seed with the original Step 8 run."""
    recipe = ("base_kind", "pattern_id", "rank", "alpha", "target_names", "seed", "epochs", "batch_size", "evaluation_batch_size", "learning_rate", "weight_decay", "max_grad_norm")
    old_config = {"base_kind": "faulty", **original["config"]}
    require(all(old_config[name] == rerun["config"][name] for name in recipe), "Step 8 rerun uses a different pattern, seed, or recipe")
    splits = {}
    for split in SPLITS:
        entry = {}
        for stage in ("before", "after"):
            old, new = original[stage][split], rerun[stage][split]
            entry[stage] = {"step8_num_correct": old["num_correct"], "rerun_num_correct": new["num_correct"], "num_correct_difference": new["num_correct"] - old["num_correct"],
                            "accuracy_difference_percentage_points": 100 * (new["accuracy"] - old["accuracy"]),
                            "step8_cross_entropy_loss": old["cross_entropy_loss"], "rerun_cross_entropy_loss": new["cross_entropy_loss"],
                            "cross_entropy_difference": new["cross_entropy_loss"] - old["cross_entropy_loss"]}
        splits[split] = entry
    same_base = all(splits[split]["before"]["num_correct_difference"] == 0 for split in SPLITS)
    within = all(abs(splits[split]["after"]["accuracy_difference_percentage_points"]) <= tolerance for split in SPLITS)
    return {"step8_run_id": original["run_id"], "rerun_run_id": rerun["run_id"], "tolerance_percentage_points": tolerance,
            "step8_selected_epoch": original["selected_epoch"], "rerun_selected_epoch": rerun["selected_epoch"], "same_selected_epoch": original["selected_epoch"] == rerun["selected_epoch"],
            "splits": splits, "identical_before_lora_counts": same_base, "after_lora_accuracy_within_tolerance": within, "passed": same_base and within,
            "criterion": "Identical before-LoRA correct counts on both splits and after-LoRA accuracy within the tolerance on both splits. Selected epoch and cross-entropy are reported, not gated."}


def load_sources(root, config):
    """Read verified inputs and hash every file the analysis depends on.

    A complete matrix must have passed its own verification. An interrupted one
    contributes only its finished runs, in plan order. Every used run is verified
    again here; runs archived outside a matrix summary are never read.
    """
    baseline, sweep = root / config.source_bn_results, root / config.source_sweep_results
    paths = {f"{label}_{filename}": directory / filename for label, directory, filenames in (
        ("bn", baseline, ("status.json", "verification.json", "config.json", "measurements.json")),
        ("sweep", sweep, ("status.json", "plan.json", "aggregate.json", "references.json", "measurements.json"))) for filename in filenames}
    read = lambda name: json.loads(paths[name].read_text())
    require(read("bn_status.json")["status"] == "complete" and read("bn_verification.json")["status"] == "passed", "BatchNorm baseline is incomplete or unverified")
    require(read("sweep_status.json")["status"] == "complete", "Source sweep is incomplete")
    require(read("bn_config.json")["source_sweep_results"] == config.source_sweep_results, "Inputs were produced from different sweeps")
    rows, run_metrics, accepted, matrices = [], {}, [], []
    for item in config.source_matrices:
        directory = root / item["results"]
        label = f"matrix_{directory.name}"
        names = ("status.json", "config.json", "plan.json", "summary.json", "source_sha256.json") + (("verification.json",) if item["status"] == "complete" else ())
        paths.update({f"{label}_{name}": directory / name for name in names})
        matrix_config = MatrixConfig(**read(f"{label}_config.json"))
        require(matrix_config.source_sweep_results == config.source_sweep_results and (root / matrix_config.results_root / matrix_config.matrix_id).resolve() == directory.resolve(),
                f"Matrix configuration differs from its directory or sweep: {item['results']}")
        plan, status, finished = matrix_plan(matrix_config), read(f"{label}_status.json"), read(f"{label}_summary.json")
        require(read(f"{label}_plan.json") == plan and status["status"] == item["status"], f"Matrix plan or status differs from the declaration: {item['results']}")
        if item["status"] == "complete":
            require(read(f"{label}_verification.json")["status"] == "passed" and len(finished) == len(plan), f"Recovery matrix is unverified: {item['results']}")
        else:
            require(status["completed_runs"] == item["completed_runs"] == len(finished) < len(plan), f"Interrupted matrix finished a different number of runs: {item['results']}")
        require([row["run_id"] for row in finished] == [entry["run_id"] for entry in plan[:len(finished)]], f"Matrix summary is reordered: {item['results']}")
        for run, row in zip(matrix_config.runs, finished):
            run_directory = root / matrix_config.results_root / row["run_id"]
            # Recorded source hashes must match, or be a change listed in a committed decision file.
            accepted += verify_run(root, matrix_config, run, row)["accepted_source_hash_changes"]
            paths[f"run_{row['run_id']}"] = run_directory / "metrics.json"
            paths[f"run_source_{row['run_id']}"] = run_directory / "source_sha256.json"
            run_metrics[row["run_id"]] = json.loads(paths[f"run_{row['run_id']}"].read_text())
        accepted += compare_source_hashes(root, read(f"{label}_source_sha256.json"), f"Source code changed after matrix {item['results']}")
        rows += finished
        matrices.append({"results": item["results"], "status": item["status"], "planned_runs": len(plan), "used_runs": len(finished)})
    require(len({row["run_id"] for row in rows}) == len(rows), "A run appears in more than one matrix")
    regression = None
    if config.step8_results is not None:
        paths["step8_metrics.json"] = root / config.step8_results / "metrics.json"
        require(config.step8_rerun_id in run_metrics, "Step 8 rerun is not among the analyzed runs")
        regression = step8_regression(read("step8_metrics.json"), run_metrics[config.step8_rerun_id], config.step8_tolerance_percentage_points)
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    inputs = (rows, read("bn_measurements.json"), read("sweep_measurements.json"), read("sweep_plan.json"), read("sweep_aggregate.json"), read("sweep_references.json"))
    # sources_allowed_to_differ (Step 9b) is reported only; acceptance is decided by the shared rule alone.
    return inputs, {"matrices": matrices, "source_files_changed_since_runs": sorted({change["path"] for change in accepted}), "accepted_source_hash_changes": unique_changes(accepted),
                    "declared_sources_allowed_to_differ": config.sources_allowed_to_differ, "step8_regression": regression}, paths, hashes


def build_analysis(root, config):
    inputs, provenance, paths, hashes = load_sources(root, config)
    analysis = analyze_controls(*inputs, config)
    analysis["source_matrices"] = provenance["matrices"]
    analysis["source_files_changed_since_runs"] = provenance["source_files_changed_since_runs"]
    analysis["accepted_source_hash_changes"] = provenance["accepted_source_hash_changes"]
    analysis["step8_regression"] = provenance["step8_regression"]
    return analysis, paths, hashes


def run_controls_analysis(config, project_root):
    config.validate()
    root = Path(project_root)
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Analysis directory already exists; choose a new run_id")
    analysis, paths, hashes = build_analysis(root, config)
    regression = analysis["step8_regression"]
    results_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/analyze_lora_controls.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu; no model evaluation",
                       "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")}}
        write_json(results_dir / "environment.json", environment)
        write_json(results_dir / "analysis.json", analysis)
        if regression is not None:
            write_json(results_dir / "step8_regression.json", regression)
            require(regression["passed"], f"Step 8 regression check failed; see {results_dir / 'step8_regression.json'}")
        with (results_dir / "conditions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CONDITION_FIELDS)
            writer.writeheader()
            writer.writerows(condition_table(analysis))
        figures = plot_controls(analysis["splits"]["test"], results_dir)
        # Recompute from re-read inputs so the saved analysis cannot drift from its sources.
        again, _, hashes_again = build_analysis(root, config)
        require(hashes_again == hashes, "An input changed during the analysis")
        require(json.loads((results_dir / "analysis.json").read_text()) == json.loads(json.dumps(again)), "Saved analysis differs from its recomputation")
        with (results_dir / "conditions.csv").open(newline="") as stream:
            require(list(csv.DictReader(stream)) == condition_table(analysis), "CSV differs from the saved analysis")
        require(all(path.stat().st_size > 1000 for path in figures), "Figure export failed")
        compare_source_hashes(root, source_code, "Source code changed during the analysis")
        manifest = {"run_id": config.run_id, "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": hashes[name]} for name, path in paths.items()},
                    "source_matrices": analysis["source_matrices"], "definitions": DEFINITIONS,
                    "run_policy": "Only runs listed in a declared matrix summary are read; an interrupted matrix contributes its finished runs; archived runs are never inputs.",
                    "figure": "Test split only. Left: full range. Right: zoom on LoRA runs and the references inside that range.",
                    "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures], "environment": environment, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "verification.json", {"status": "passed", "design_matches_configuration": True, "runs_paired_with_step7": True, "runs_reverified": analysis["design"]["num_runs"],
                                                      "source_files_changed_since_runs": analysis["source_files_changed_since_runs"],
                                                      "accepted_source_hash_changes": analysis["accepted_source_hash_changes"], "step8_regression_passed": regression and regression["passed"],
                                                      "analysis_recomputed_from_reread_inputs": True, "csv_matches_analysis": True, "input_artifacts_unchanged": len(hashes),
                                                      "source_code_unchanged_during_analysis": True, "plot_hashes_recorded": True})
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "lora_controls_analysis_complete", "run_id": config.run_id, "figures": [str(path.relative_to(root)) for path in figures]}), flush=True)
        return analysis
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_lora_controls.json"))
    args = parser.parse_args()
    run_controls_analysis(ControlsConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
