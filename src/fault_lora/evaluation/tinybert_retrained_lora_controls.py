"""Step 12c analysis: LoRA on the retrained TinyBERT baseline, Linear vs Linear + embedding adapters; analysis only.

Reads the finished Step 12c matrix and its transfer evaluations, re-verifies
every run, and applies the Step 10e definitions per adapter placement. It adds
the placement comparison (Linear + embeddings minus Linear, paired by pattern
and training seed) and a side-by-side with Step 10e, whose Linear adapters were
trained on the old Condition B encoding with the same patterns and seeds.
Nothing is trained or evaluated here. Each test set is analyzed separately and
never pooled; the selection split chose checkpoints and is not a result.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.fault_sweep import summarize_values
from fault_lora.evaluation.tinybert_lora_controls import stats_or_none, summarize_runs
from fault_lora.evaluation.tinybert_retrained_faults import TinyBertRetrainedSweepConfig
from fault_lora.training.tinybert_recovery import verify_tinybert_recovery
from fault_lora.training.tinybert_retrained_recovery import PLACEMENTS, TinyBertRetrainedMatrixConfig, TinyBertRetrainedRecoveryConfig, matrix_plan

PLACEMENT_LABELS = {"linear": "Linear adapters (26 nn.Linear)", "linear+emb": "Linear + embedding adapters (26 nn.Linear + 3 nn.Embedding)"}
DEFINITIONS = {
    "variability": "Sample standard deviation (ddof=1) over the listed runs; null for one value. Not a confidence interval.",
    "pooling": "A faulty rate pools patterns x training seeds; clean and retrained-clustered controls pool training seeds. Test sets are never pooled.",
    "extra_training_effect": "LoRA(clean) minus the clean base before LoRA, per training seed, in percentage points, per placement.",
    "clustering_repair": "LoRA(retrained clustered) minus the retrained clustered base before LoRA, per training seed, per placement.",
    "normalized_recovery": "R = (after - before) / (clean - before) per run, with clean the clean model's accuracy on that test set; null when before >= clean.",
    "fault_damage_left": "LoRA(retrained faulty) minus the mean LoRA(retrained clustered) of the same placement over all control seeds, per faulty run (Step 10e's fault-specific gain); "
                         "seed_matched subtracts the same-seed LoRA(retrained clustered) run. Negative means LoRA on the faulty base ends below LoRA on the fault-free base.",
    "placement_difference": "Linear + embeddings minus Linear, paired by base, pattern, and training seed (by pattern and adapter seed for transfers), in percentage points.",
    "transfer": "A retrained-clustered adapter evaluated on a retrained-faulty pattern with no training, per placement. transfer_gain = with adapter minus the same faulty base; "
                "trained_minus_transfer = mean LoRA trained on that pattern (over seeds, same placement) minus the transfer accuracy.",
    "step10e_comparison": "Step 10e Linear adapters on the old Condition B encoding against Step 12c Linear adapters on the retrained encoding: same patterns (identical cells), "
                          "training seeds, recipe, and device; paired by pattern and seed.",
    "binomial_sd": "100 * sqrt(p (1 - p) / n) at the clean accuracy: the sampling SD of one run's accuracy on that test set, for scale only.",
    "selection": "Design, seeds, patterns, and formulas fixed before the runs. Test accuracy follows per-run selection-split checkpoint choice and chooses nothing.",
}


@dataclass
class TinyBertRetrainedControlsConfig:
    run_id: str = "tinybert-sst2-lora12c-controls"
    source_matrix_config: str = "configs/tinybert_sst2_lora12c_matrix.json"
    step10e_analysis: str | None = "results/tinybert-sst2-lora-controls"
    placements: list = field(default_factory=lambda: ["linear", "linear+emb"])
    rates: list = field(default_factory=lambda: [.1, .2, .5])
    control_seeds: list = field(default_factory=lambda: [42, 43, 44])
    faulty_seeds: list = field(default_factory=lambda: [42, 43])
    patterns_per_rate: int = 3
    device: str = "mps"
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if self.placements != [name for name in PLACEMENTS if name in self.placements] or len(self.placements) != 2:
            raise ValueError(f"placements must be {list(PLACEMENTS)}")
        if not self.rates or self.rates != sorted(set(self.rates)) or any(isinstance(rate, bool) or not 0 < rate <= 1 for rate in self.rates):
            raise ValueError("rates must be sorted, unique, and in (0, 1]")
        for name in ("control_seeds", "faulty_seeds"):
            seeds = getattr(self, name)
            if not seeds or seeds != sorted(set(seeds)) or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
                raise ValueError(f"{name} must be sorted, unique integers")
        if len(self.control_seeds) < 2 or not set(self.faulty_seeds) <= set(self.control_seeds) or self.patterns_per_rate < 1:
            raise ValueError("Need at least two control seeds, faulty seeds among them, and at least one pattern per rate")


def load_inputs(root, config):
    """Every planned run (complete, re-verified, on the declared device), the transfers, the Step 12b no-repair curve, and Step 10e's analysis."""
    matrix = TinyBertRetrainedMatrixConfig(**json.loads((root / config.source_matrix_config).read_text()))
    plan = matrix_plan(matrix)
    matrix_dir = root / matrix.results_root / matrix.matrix_id
    status = json.loads((matrix_dir / "status.json").read_text())
    require(status["status"] == "complete" and status["completed_runs"] == len(plan) and status["transfers"] == len(matrix.transfers), "Step 12c matrix did not complete")
    require(json.loads((matrix_dir / "plan.json").read_text()) == plan, "Saved matrix plan differs from its configuration")
    inputs, runs = {}, {}
    for entry in plan:
        directory = root / matrix.results_root / entry["run_id"]
        read = lambda name: json.loads((directory / name).read_text())
        require(read("status.json")["status"] == "complete" and read("verification.json")["status"] == "passed", f"Run is incomplete or unverified: {entry['run_id']}")
        require(verify_tinybert_recovery(root, TinyBertRetrainedRecoveryConfig(**entry["config"]))["status"] == "passed", f"Run does not re-verify: {entry['run_id']}")
        metrics = read("metrics.json")
        require(metrics["config"] == entry["config"] and metrics["config"]["device"] == config.device, f"Run configuration or device differs: {entry['run_id']}")
        key = (entry["config"]["adapter_placement"], entry["config"]["base_kind"], entry["config"]["pattern_id"], entry["config"]["seed"])
        require(key not in runs, f"Two runs fill the same design cell: {key}")
        runs[key] = {**metrics, "adapter_layout": read("adapter_layout.json")}
        for name in ("metrics", "status", "verification", "config", "adapter_layout"):
            inputs[str((directory / f"{name}.json").relative_to(root))] = sha256_file(directory / f"{name}.json")
    transfers = json.loads((matrix_dir / "transfers.json").read_text())
    require(len(transfers) == len(matrix.transfers), "Transfer evaluations are incomplete")
    for name in ("status", "plan", "transfers", "config", "manifest"):
        inputs[str((matrix_dir / f"{name}.json").relative_to(root))] = sha256_file(matrix_dir / f"{name}.json")
    sweeps = {metrics["config"]["source_sweep_results"] for (_, kind, _, _), metrics in runs.items() if kind.startswith("retrained-")}
    require(len(sweeps) == 1, "Runs use more than one Step 12b sweep")
    sweep_dir = root / sweeps.pop()
    sweep_config = TinyBertRetrainedSweepConfig(**json.loads((sweep_dir / "config.json").read_text()))
    conditions = {metrics["condition"] for (_, kind, _, _), metrics in runs.items() if kind.startswith("retrained-")}
    require(len(conditions) == 1, "Runs use more than one condition")
    condition = conditions.pop()
    group = next(item["group_id"] for item in sweep_config.groups if item["condition"] == condition)
    sweep = json.loads((sweep_dir / "aggregate.json").read_text())["groups"][group]
    for name in ("aggregate", "status", "verification"):
        inputs[str((sweep_dir / f"{name}.json").relative_to(root))] = sha256_file(sweep_dir / f"{name}.json")
    step10e = None
    if config.step10e_analysis is not None:
        step10e_dir = root / config.step10e_analysis
        require(json.loads((step10e_dir / "status.json").read_text())["status"] == "complete" and json.loads((step10e_dir / "verification.json").read_text())["status"] == "passed",
                "Step 10e analysis is incomplete or unverified")
        step10e = json.loads((step10e_dir / "analysis.json").read_text())
        inputs[str((step10e_dir / "analysis.json").relative_to(root))] = sha256_file(step10e_dir / "analysis.json")
    return {"runs": runs, "transfers": transfers, "sweep": sweep, "condition": condition, "step10e": step10e, "inputs": inputs}


def validate_design(runs, transfers, config):
    """Exactly the declared cells for both placements; transfers placement-matched and covering every pattern."""
    patterns = {}
    for rate in config.rates:
        names = sorted({metrics["pattern_id"] for (_, kind, _, _), metrics in runs.items() if kind == "retrained-faulty" and metrics["fault_probability"] == rate})
        require(len(names) == config.patterns_per_rate, f"Rate {rate}: expected {config.patterns_per_rate} patterns, found {names}")
        patterns[rate] = names
    expected = set()
    for placement in config.placements:
        expected |= {(placement, "clean", None, seed) for seed in config.control_seeds}
        expected |= {(placement, "retrained-clustered", None, seed) for seed in config.control_seeds}
        expected |= {(placement, "retrained-faulty", name, seed) for names in patterns.values() for name in names for seed in config.faulty_seeds}
    require(set(runs) == expected, f"Runs differ from the declared design: missing {sorted(map(str, expected - set(runs)))}, extra {sorted(map(str, set(runs) - expected))}")
    covered = {}
    for item in transfers:
        require(item["adapter_base"]["base_kind"] == "retrained-clustered" and item["evaluation_base"]["base_kind"] == "retrained-faulty", "Transfer is not retrained clustered → retrained faulty")
        placement, name = item["adapter_placement"], item["evaluation_base"]["pattern_id"]
        require(runs[(placement, "retrained-clustered", None, item["adapter_seed"])]["best_checkpoint"].split("/")[1] == item["adapter_run_id"], "Transfer adapter is not that placement's control run")
        require(item["without_adapter"] == runs[(placement, "retrained-faulty", name, config.faulty_seeds[0])]["before"], f"Transfer's zero-adapter base differs from the trained run's base: {name}")
        covered.setdefault((placement, name), []).append(item["adapter_seed"])
    wanted = {(placement, name) for placement in config.placements for names in patterns.values() for name in names}
    require(set(covered) == wanted and all(sorted(seeds) == config.control_seeds for seeds in covered.values()), "Transfers do not cover every pattern with every control adapter of each placement")
    for names in patterns.values():
        for name in names:
            require(len({str(runs[(placement, "retrained-faulty", name, seed)]["before"]) for placement in config.placements for seed in config.faulty_seeds}) == 1,
                    f"Runs on pattern {name} start from different bases")
    return patterns


def paired(values_a, values_b):
    """b - a over shared keys, in percentage points, with the per-pair values."""
    require(set(values_a) == set(values_b), "Paired comparison over different keys")
    differences = {key: 100 * (values_b[key] - values_a[key]) for key in sorted(values_a)}
    return {"pairs": [{"key": list(key) if isinstance(key, tuple) else key, "difference": value} for key, value in differences.items()], **summarize_values(list(differences.values()))}


def analyze_placement(runs, transfers, patterns, config, split, placement):
    select = lambda kind: [metrics for (pl, k, _, _), metrics in runs.items() if pl == placement and k == kind]
    clean_runs = select("clean")
    clean = clean_runs[0]["before"][split]["accuracy"]
    require(all(metrics["before"][split]["accuracy"] == clean for metrics in clean_runs), "Clean controls start from different accuracies")
    lora_clean = summarize_runs(clean_runs, split, clean, None)
    clustered_runs = select("retrained-clustered")
    clustered = clustered_runs[0]["before"][split]["accuracy"]
    require(all(metrics["before"][split]["accuracy"] == clustered for metrics in clustered_runs), "Retrained clustered controls start from different accuracies")
    lora_clustered = summarize_runs(clustered_runs, split, clean, None)
    clustered_mean = lora_clustered["accuracy_after"]["mean"]
    clustered_by_seed = {run["seed"]: run["accuracy_after"] for run in lora_clustered["runs"]}
    layout = clean_runs[0]["adapter_layout"]
    parameters = clean_runs[0]["parameters"]
    rate_entries = []
    for rate in config.rates:
        names = patterns[rate]
        group = [metrics for (pl, kind, name, _), metrics in runs.items() if pl == placement and kind == "retrained-faulty" and name in names]
        entry = {"fault_probability": rate, **summarize_runs(group, split, clean, clustered)}
        entry["fault_damage_left_percentage_points"] = summarize_values([100 * (run["accuracy_after"] - clustered_mean) for run in entry["runs"]])
        entry["fault_damage_left_seed_matched_percentage_points"] = summarize_values([100 * (run["accuracy_after"] - clustered_by_seed[run["seed"]]) for run in entry["runs"]])
        trained_by_pattern = {name: statistics.mean(run["accuracy_after"] for run in entry["runs"] if run["pattern_id"] == name) for name in names}
        items = sorted((item for item in transfers if item["adapter_placement"] == placement and item["evaluation_base"]["pattern_id"] in names),
                       key=lambda item: (item["evaluation_base"]["pattern_id"], item["adapter_seed"]))
        transfer_runs = [{"pattern_id": item["evaluation_base"]["pattern_id"], "adapter_seed": item["adapter_seed"], "accuracy_without_adapter": item["without_adapter"][split]["accuracy"],
                          "accuracy_with_adapter": item["with_adapter"][split]["accuracy"], "transfer_gain_percentage_points": item["accuracy_gain_percentage_points"][split],
                          "trained_minus_transfer_percentage_points": 100 * (trained_by_pattern[item["evaluation_base"]["pattern_id"]] - item["with_adapter"][split]["accuracy"])}
                         for item in items]
        entry["transfer"] = {"num_evaluations": len(transfer_runs),
                             **{key: summarize_values([run[key] for run in transfer_runs]) for key in ("accuracy_with_adapter", "transfer_gain_percentage_points", "trained_minus_transfer_percentage_points")},
                             "evaluations": transfer_runs}
        rate_entries.append(entry)
    return {"adapter": {"trainable_parameters": parameters["trainable_adapter"], "float32_bytes": parameters["adapter_tensor_bytes"],
                        "layers": {"Linear": sum(item["kind"] == "Linear" for item in layout), "Embedding": sum(item["kind"] == "Embedding" for item in layout)}},
            "clean_accuracy": clean, "lora_clean": lora_clean,
            "extra_training_effect_percentage_points": {"per_seed": {str(run["seed"]): run["accuracy_gain_percentage_points"] for run in lora_clean["runs"]}, **lora_clean["accuracy_gain_percentage_points"]},
            "retrained_clustered_accuracy": clustered, "lora_retrained_clustered": lora_clustered, "clustering_repair_percentage_points": lora_clustered["accuracy_gain_percentage_points"],
            "rates": rate_entries}


def compare_placements(result, config):
    """Linear + embeddings minus Linear, paired by seed (controls), by pattern and seed (faulty), and by pattern and adapter seed (transfer)."""
    first, second = (result["placements"][name] for name in config.placements)
    by_seed = lambda entry: {run["seed"]: run["accuracy_after"] for run in entry["runs"]}
    by_run = lambda entry: {(run["pattern_id"], run["seed"]): run["accuracy_after"] for run in entry["runs"]}
    by_transfer = lambda entry: {(run["pattern_id"], run["adapter_seed"]): run["accuracy_with_adapter"] for run in entry["transfer"]["evaluations"]}
    return {"lora_clean": paired(by_seed(first["lora_clean"]), by_seed(second["lora_clean"])),
            "lora_retrained_clustered": paired(by_seed(first["lora_retrained_clustered"]), by_seed(second["lora_retrained_clustered"])),
            "rates": [{"fault_probability": a["fault_probability"], "lora_after": paired(by_run(a), by_run(b)), "transfer": paired(by_transfer(a), by_transfer(b)),
                       "fault_damage_left_difference_of_means": b["fault_damage_left_percentage_points"]["mean"] - a["fault_damage_left_percentage_points"]["mean"]}
                      for a, b in zip(first["rates"], second["rates"])]}


def compare_step10e(result, step10e_split, condition, patterns):
    """Step 10e Linear adapters on the old encoding vs Step 12c Linear adapters on the retrained one, same patterns and seeds."""
    ours = result["placements"]["linear"]
    old = step10e_split["conditions"][condition]
    by_seed = lambda entry: {run["seed"]: run["accuracy_after"] for run in entry["runs"]}
    comparison = {"lora_clean_reproduction": paired(by_seed(step10e_split["lora_clean"]), by_seed(ours["lora_clean"])),
                  "clustered": {"step10e_before": old["clustered_accuracy"], "step12c_before": ours["retrained_clustered_accuracy"],
                                "step10e_after": old["lora_clustered"]["accuracy_after"], "step12c_after": ours["lora_retrained_clustered"]["accuracy_after"],
                                "after_paired": paired(by_seed(old["lora_clustered"]), by_seed(ours["lora_retrained_clustered"]))},
                  "rates": []}
    old_rates = {entry["fault_probability"]: entry for entry in old["rates"]}
    for entry in ours["rates"]:
        rate = entry["fault_probability"]
        require(rate in old_rates and sorted({run["pattern_id"] for run in old_rates[rate]["runs"]}) == patterns[rate], f"Step 10e used other patterns at {rate}")
        previous = old_rates[rate]
        by_run = lambda item, field: {(run["pattern_id"], run["seed"]): run[field] for run in item["runs"]}
        comparison["rates"].append({"fault_probability": rate, "step10e_before": previous["accuracy_before"], "step12c_before": entry["accuracy_before"],
                                    "before_paired": paired(by_run(previous, "accuracy_before"), by_run(entry, "accuracy_before")),
                                    "step10e_after": previous["accuracy_after"], "step12c_after": entry["accuracy_after"], "after_paired": paired(by_run(previous, "accuracy_after"), by_run(entry, "accuracy_after")),
                                    "step10e_fault_specific_gain": previous["fault_specific_gain_percentage_points"], "step12c_fault_damage_left": entry["fault_damage_left_percentage_points"],
                                    "step10e_transfer_accuracy": previous["transfer"]["accuracy_with_adapter"], "step12c_transfer_accuracy": entry["transfer"]["accuracy_with_adapter"]})
    return comparison


def analyze(data, config, test_names):
    runs, transfers = data["runs"], data["transfers"]
    patterns = validate_design(runs, transfers, config)
    test_sets = {}
    for split in test_names:
        result = {"placements": {placement: analyze_placement(runs, transfers, patterns, config, split, placement) for placement in config.placements}}
        clean = result["placements"][config.placements[0]]["clean_accuracy"]
        examples = next(iter(runs.values()))["after"][split]["num_examples"]
        result["num_examples"] = examples
        result["binomial_sd_percentage_points"] = 100 * math.sqrt(clean * (1 - clean) / examples)
        result["placement_difference"] = compare_placements(result, config)
        result["step10e_comparison"] = compare_step10e(result, data["step10e"]["test_sets"][split], data["condition"], patterns) if data["step10e"] else None
        test_sets[split] = result
    return {"run_id": config.run_id, "design": {"placements": config.placements, "control_seeds": config.control_seeds, "faulty_seeds": config.faulty_seeds,
                                                 "patterns": {str(rate): names for rate, names in patterns.items()}, "num_runs": len(runs), "num_transfer_evaluations": len(transfers),
                                                 "condition": data["condition"], "device": config.device},
            "definitions": DEFINITIONS, "test_sets": test_sets}


TABLE_FIELDS = ["test_set", "adapter_placement", "base", "fault_probability", "num_runs", "accuracy_before_mean", "accuracy_before_sample_std", "accuracy_after_mean", "accuracy_after_sample_std",
                "accuracy_gain_percentage_points_mean", "accuracy_gain_percentage_points_sample_std", "normalized_recovery_mean", "normalized_recovery_sample_std",
                "fault_damage_left_percentage_points_mean", "fault_damage_left_percentage_points_sample_std", "transfer_accuracy_mean", "transfer_accuracy_sample_std",
                "transfer_gain_percentage_points_mean", "transfer_gain_percentage_points_sample_std", "trained_minus_transfer_percentage_points_mean",
                "trained_minus_transfer_percentage_points_sample_std"]


def table_rows(analysis):
    rows = []

    def add(test_set, placement, base, probability, entry, damage=None, transfer=None):
        row = {"test_set": test_set, "adapter_placement": placement, "base": base, "fault_probability": probability, "num_runs": entry["num_runs"]}
        values = {"accuracy_before": entry["accuracy_before"], "accuracy_after": entry["accuracy_after"], "accuracy_gain_percentage_points": entry["accuracy_gain_percentage_points"],
                  "normalized_recovery": entry.get("normalized_recovery"), "fault_damage_left_percentage_points": damage, "transfer_accuracy": transfer and transfer["accuracy_with_adapter"],
                  "transfer_gain_percentage_points": transfer and transfer["transfer_gain_percentage_points"],
                  "trained_minus_transfer_percentage_points": transfer and transfer["trained_minus_transfer_percentage_points"]}
        for name, value in values.items():
            row[f"{name}_mean"], row[f"{name}_sample_std"] = (value["mean"], value["sample_std"]) if value else (None, None)
        rows.append({name: "" if row[name] is None else str(row[name]) for name in TABLE_FIELDS})

    for test_set, result in analysis["test_sets"].items():
        for placement, item in result["placements"].items():
            add(test_set, placement, "clean", "", item["lora_clean"])
            add(test_set, placement, "retrained-clustered", 0.0, item["lora_retrained_clustered"])
            for entry in item["rates"]:
                add(test_set, placement, "retrained-faulty", entry["fault_probability"], entry, entry["fault_damage_left_percentage_points"], entry["transfer"])
    return rows


def plot_controls(analysis, sweep, output_dir):
    """One panel per test set: no repair, LoRA per placement, transfer per placement, and Step 10e's Linear LoRA on the old encoding."""
    with tempfile.TemporaryDirectory(prefix="fault-lora-12c-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
            # Categorical slots in fixed order (validated: CVD and normal-vision separation pass; contrast relief via legend, markers, and line styles).
            colours = {"linear": "#2a78d6", "none": "#eb6834", "linear+emb": "#1baf7a", "step10e": "#eda100"}
            markers = {"linear": "D", "linear+emb": "s"}
            offsets = {"linear": -0.6, "linear+emb": 0.6}
            test_names = list(analysis["test_sets"])
            fig, axes = plt.subplots(1, len(test_names), figsize=(13.5, 5.6), squeeze=False)
            fig.patch.set_facecolor(surface)
            for axis, test_set in zip(axes[0], test_names):
                axis.set_facecolor(surface)
                result = analysis["test_sets"][test_set]
                points = [entry for entry in sweep if entry["num_patterns"] >= 1]
                axis.errorbar([100 * entry["probability"] for entry in points], [100 * entry["splits"][test_set]["accuracy"]["mean"] for entry in points],
                              yerr=[100 * (entry["splits"][test_set]["accuracy"]["sample_std"] or 0) for entry in points], color=colours["none"], marker="o", markersize=6, capsize=3,
                              linewidth=2, label="Retrained base, no repair (Step 12b, 5 patterns)", zorder=2)
                comparison = result["step10e_comparison"]
                xs = [0.0] + [100 * entry["fault_probability"] for entry in result["placements"]["linear"]["rates"]]
                if comparison:
                    old = [comparison["clustered"]["step10e_after"]] + [entry["step10e_after"] for entry in comparison["rates"]]
                    axis.plot(xs, [100 * value["mean"] for value in old], color=colours["step10e"], marker="^", markersize=7, linewidth=1.4,
                              label="Step 10e: Linear LoRA on the old Condition B base", zorder=3)
                for placement, item in result["placements"].items():
                    x = [value + offsets[placement] for value in xs]
                    lora = [item["lora_retrained_clustered"]["accuracy_after"]] + [entry["accuracy_after"] for entry in item["rates"]]
                    label = "Linear" if placement == "linear" else "Linear + embeddings"
                    axis.errorbar(x, [100 * value["mean"] for value in lora], yerr=[100 * (value["sample_std"] or 0) for value in lora], color=colours[placement], marker=markers[placement],
                                  markersize=7, markeredgecolor=surface, markeredgewidth=1.5, capsize=3, linewidth=2, label=f"LoRA, {label} adapters (trained on that base)", zorder=4)
                    transfer = [entry["transfer"]["accuracy_with_adapter"] for entry in item["rates"]]
                    axis.errorbar(x[1:], [100 * value["mean"] for value in transfer], yerr=[100 * (value["sample_std"] or 0) for value in transfer], color=colours[placement],
                                  marker=markers[placement], markersize=7, markerfacecolor=surface, capsize=3, linewidth=1.4, linestyle="--",
                                  label=f"Transfer, {label} (retrained-clustered adapter, no training)", zorder=3)
                clean = result["placements"]["linear"]["clean_accuracy"]
                axis.axhline(100 * clean, color=muted, linestyle="--", linewidth=1, label=f"Clean: {100 * clean:.2f}%", zorder=1)
                axis.axhline(50, color="#aaaaaa", linewidth=.8, zorder=1)
                axis.set_title(test_set, fontsize=11, color=ink)
                axis.set_xlabel("Requested fault probability (%)", color=muted)
                axis.set_ylabel("Accuracy (%)", color=muted)
                axis.set_xticks([0, 1, 5, 10, 20, 50])
                axis.set_ylim(50, 92)
                axis.grid(color=grid, linewidth=.8)
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
                axis.tick_params(colors=muted)
                axis.legend(loc="lower left", fontsize=7.5, frameon=False)
            fig.suptitle("TinyBERT / SST-2, Condition B: LoRA on the codebook-retrained base, Linear vs Linear + embedding adapters", fontsize=12.5, color=ink)
            fig.text(.5, .015, "Means ± 1 sample SD. LoRA at 0%: retrained clustered base, 3 seeds; faulty rates: 3 patterns × 2 seeds. Transfer: 3 adapters × 3 patterns. "
                               "Placements offset ±0.6 points on x for legibility. Test sets shown separately.", ha="center", fontsize=8, color=muted)
            fig.tight_layout(rect=(0, .035, 1, .95))
            paths = [Path(output_dir) / f"retrained_lora_controls.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=200, facecolor=surface)
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def run_retrained_controls(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Analysis directory already exists; choose a new run_id")
    data = load_inputs(root, config)
    test_names = [name for name in next(iter(data["runs"].values()))["after"] if name != "selection"]
    analysis = analyze(data, config, test_names)
    results_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/analyze_tinybert_retrained_lora_controls.py"]
    write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
    write_json(results_dir / "inputs_sha256.json", data["inputs"])
    write_json(results_dir / "analysis.json", analysis)
    rows = table_rows(analysis)
    with (results_dir / "conditions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    figures = plot_controls(analysis, data["sweep"]["retrained"], results_dir)
    # Verification: inputs unchanged and the analysis reproduces from the saved runs.
    again = load_inputs(root, config)
    require(again["inputs"] == data["inputs"], "Inputs changed during the analysis")
    require(json.loads(json.dumps(analyze(again, config, test_names))) == json.loads((results_dir / "analysis.json").read_text()), "Analysis does not reproduce")
    with (results_dir / "conditions.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == rows, "CSV and JSON tables disagree")
    write_json(results_dir / "verification.json", {"status": "passed", "runs_reverified": len(data["runs"]), "transfer_evaluations": len(data["transfers"]), "inputs_unchanged": True,
                                                   "analysis_reproduced": True, "csv_matches_json": True,
                                                   "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures]})
    write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return analysis


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2_lora12c_controls.json")
    args = parser.parse_args()
    run_retrained_controls(TinyBertRetrainedControlsConfig(**json.loads(args.config.read_text())), project_root)
