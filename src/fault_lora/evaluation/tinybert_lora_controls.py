"""Controlled comparison of TinyBERT LoRA recovery on clean, clustered, and faulty bases; analysis only.

Mirrors `evaluation/lora_controls.py` (ResNet). It reads the finished Part 7
matrix, its transfer evaluations, and the declared extra run that stands in for
the matrix cell already run as the pilot. Nothing is trained or evaluated here.
Each test set is analyzed separately and never pooled; the selection split
chose checkpoints and is not analyzed as a result. There is no BatchNorm
recalibration counterpart: TinyBERT's LayerNorm has no running statistics.
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
from fault_lora.training.tinybert_recovery import TinyBertRecoveryConfig, recovery_run_id
from fault_lora.training.tinybert_recovery_matrix import TinyBertMatrixConfig, matrix_plan

CONDITION_LABELS = {"linear-only": "Linear weights only", "linear-and-embeddings": "Linear weights + embeddings"}
DEFINITIONS = {
    "variability": "Sample standard deviation (ddof=1) over the listed runs; null for one value. Not a confidence interval.",
    "pooling": "A faulty rate pools patterns x training seeds; clean and clustered controls pool training seeds. Test sets are never pooled with each other.",
    "extra_training_effect": "LoRA(clean) minus the clean base before LoRA, per training seed, in percentage points.",
    "clustering_repair": "LoRA(clustered) minus the clustered base before LoRA, per training seed, for each condition.",
    "normalized_recovery": "R = (after - before) / (clean - before) per run, with clean the clean model's accuracy on that test set; null when before >= clean.",
    "normalized_recovery_to_clustered": "(after - before) / (clustered - before) per run, with clustered that condition's zero-fault reference; null when before >= clustered.",
    "fault_specific_gain": "LoRA(faulty) minus the mean LoRA(clustered) of the same condition over all control seeds, per faulty run; seed_matched subtracts the same-seed LoRA(clustered) run.",
    "transfer": "A clustered-trained adapter evaluated on a faulty pattern of the same condition with no training. transfer_gain = with adapter minus the same faulty base; trained_minus_transfer = mean LoRA trained on that pattern (over seeds) minus the transfer accuracy.",
    "selection": "Design, seeds, patterns, and formulas fixed before the runs. Test accuracy follows per-run selection-split checkpoint choice and chooses nothing.",
}


@dataclass
class TinyBertControlsConfig:
    run_id: str = "tinybert-sst2-lora-controls"
    source_matrix_config: str = "configs/tinybert_sst2_lora_matrix_part7_proposed.json"
    extra_runs: list = field(default_factory=lambda: ["results/tinybert-sst2-lora-faulty-linemb-rate07-seed1030-r4-train42-mps"])
    conditions: dict = field(default_factory=lambda: {"linear-only": [.3, .5, .75, 1.0], "linear-and-embeddings": [.1, .2, .5]})
    control_seeds: list = field(default_factory=lambda: [42, 43, 44])
    faulty_seeds: list = field(default_factory=lambda: [42, 43])
    patterns_per_rate: int = 3
    device: str = "mps"
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if not isinstance(self.extra_runs, list) or any(not isinstance(item, str) for item in self.extra_runs):
            raise ValueError("extra_runs must be a list of run result directories")
        if not isinstance(self.conditions, dict) or set(self.conditions) - set(CONDITION_LABELS) or not self.conditions:
            raise ValueError(f"conditions must map {sorted(CONDITION_LABELS)} to rate lists")
        for rates in self.conditions.values():
            if not rates or rates != sorted(set(rates)) or any(isinstance(rate, bool) or not 0 < rate <= 1 for rate in rates):
                raise ValueError("Each condition needs sorted, unique, nonzero rates")
        for name in ("control_seeds", "faulty_seeds"):
            seeds = getattr(self, name)
            if not seeds or seeds != sorted(set(seeds)) or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
                raise ValueError(f"{name} must be sorted, unique integers")
        if len(self.control_seeds) < 2 or not set(self.faulty_seeds) <= set(self.control_seeds) or self.patterns_per_rate < 1:
            raise ValueError("Need at least two control seeds, faulty seeds among them, and at least one pattern per rate")


def run_key(config):
    return (config["base_kind"], config["condition"], config["pattern_id"], config["seed"])


def load_runs(root, config):
    """Every planned matrix run plus the declared extra runs, each complete, verified, and on the declared device."""
    matrix = TinyBertMatrixConfig(**json.loads((root / config.source_matrix_config).read_text()))
    plan = matrix_plan(matrix)
    matrix_dir = root / matrix.results_root / matrix.matrix_id
    status = json.loads((matrix_dir / "status.json").read_text())
    require(status["status"] == "complete" and status["completed_runs"] == len(plan), "Part 7 matrix did not complete")
    require(json.loads((matrix_dir / "plan.json").read_text()) == plan, "Saved matrix plan differs from its configuration")
    inputs, runs = {}, {}
    directories = [root / matrix.results_root / entry["run_id"] for entry in plan] + [root / path for path in config.extra_runs]
    for directory in directories:
        read = lambda name: json.loads((directory / name).read_text())
        require(read("status.json")["status"] == "complete" and read("verification.json")["status"] == "passed", f"Run is incomplete or unverified: {directory.name}")
        metrics = read("metrics.json")
        run_config = metrics["config"]
        require(run_config["device"] == config.device, f"Run used another device: {directory.name}")
        key = run_key(run_config)
        require(key not in runs, f"Two runs fill the same design cell: {key}")
        runs[key] = metrics
        for name in ("metrics", "status", "verification", "config"):
            inputs[str((directory / f"{name}.json").relative_to(root))] = sha256_file(directory / f"{name}.json")
    for path in config.extra_runs:
        metrics = json.loads((root / path / "metrics.json").read_text())
        expected = {**metrics["config"], "run_id": recovery_run_id(*[metrics["config"][name] for name in ("base_kind", "condition", "pattern_id", "rank", "seed")])}
        derived = TinyBertRecoveryConfig(**{**matrix.recipe, **{name: metrics["config"][name] for name in ("base_kind", "condition", "pattern_id", "source_sweep_results", "seed")},
                                            "run_id": expected["run_id"], "results_root": matrix.results_root, "checkpoint_root": matrix.checkpoint_root})
        require(expected == asdict(derived), f"Extra run differs from the matrix recipe in more than its run ID: {path}")
        require(expected["run_id"] not in {entry["run_id"] for entry in plan}, f"Extra run duplicates a planned matrix run: {path}")
    transfers = json.loads((matrix_dir / "transfers.json").read_text())
    require(len(transfers) == len(matrix.transfers), "Transfer evaluations are incomplete")
    for name in ("status", "plan", "transfers", "config", "manifest"):
        inputs[str((matrix_dir / f"{name}.json").relative_to(root))] = sha256_file(matrix_dir / f"{name}.json")
    return matrix, runs, transfers, inputs


def validate_design(runs, transfers, config):
    """Exactly the declared cells; faulty runs paired with their sweeps; transfers condition-matched."""
    patterns = {}
    for condition, rates in config.conditions.items():
        for rate in rates:
            names = sorted({metrics["pattern_id"] for (kind, cond, _, _), metrics in runs.items() if kind == "faulty" and cond == condition and metrics["fault_probability"] == rate})
            require(len(names) == config.patterns_per_rate, f"{condition} at {rate}: expected {config.patterns_per_rate} patterns, found {names}")
            patterns[(condition, rate)] = names
    expected = {("clean", None, None, seed) for seed in config.control_seeds}
    expected |= {("clustered", condition, None, seed) for condition in config.conditions for seed in config.control_seeds}
    expected |= {("faulty", condition, name, seed) for (condition, _), names in patterns.items() for name in names for seed in config.faulty_seeds}
    require(set(runs) == expected, f"Runs differ from the declared design: missing {sorted(map(str, expected - set(runs)))}, extra {sorted(map(str, set(runs) - expected))}")
    clustered_adapters = {}
    for (kind, condition, _, seed), metrics in runs.items():
        if kind == "clustered":
            clustered_adapters[metrics["best_checkpoint"]] = (condition, seed)
    by_key = {(item["evaluation_base"]["condition"], item["evaluation_base"]["pattern_id"]): [] for item in transfers}
    for item in transfers:
        adapter = item["adapter_base"]
        require(adapter["base_kind"] == "clustered" and adapter["condition"] == item["evaluation_base"]["condition"] and item["evaluation_base"]["base_kind"] == "faulty", "Transfer is not clustered → faulty of one condition")
        by_key[(item["evaluation_base"]["condition"], item["evaluation_base"]["pattern_id"])].append(item)
    wanted = {(condition, name) for (condition, _), names in patterns.items() for name in names}
    require(set(by_key) == wanted and all(len(items) == len(config.control_seeds) for items in by_key.values()), "Transfers do not cover every faulty pattern with every clustered adapter")
    for (condition, name), items in by_key.items():
        before = {split: value for split, value in runs[("faulty", condition, name, config.faulty_seeds[0])]["before"].items()}
        for item in items:
            require(item["without_adapter"] == before, f"Transfer's zero-adapter base differs from the trained run's base: {condition} {name}")
            require({item["adapter_seed"]} <= set(config.control_seeds), "Transfer adapter seed is not a control seed")
    return patterns


def stats_or_none(values):
    return summarize_values(values) if values else None


def summarize_runs(group, split, clean, clustered):
    runs = []
    for metrics in sorted(group, key=lambda item: (item["pattern_id"] or "", item["config"]["seed"])):
        before, after = metrics["before"][split]["accuracy"], metrics["after"][split]["accuracy"]
        runs.append({"run_id": metrics["run_id"], "pattern_id": metrics["pattern_id"], "seed": metrics["config"]["seed"], "selected_epoch": metrics["selected_epoch"],
                     "realized_fault_rate": metrics["realized_fault_rate"], "accuracy_before": before, "accuracy_after": after,
                     "accuracy_gain_percentage_points": 100 * (after - before),
                     "normalized_recovery": (after - before) / (clean - before) if clean > before else None,
                     "normalized_recovery_to_clustered": (after - before) / (clustered - before) if clustered is not None and clustered > before else None})
    entry = {"num_runs": len(runs), "num_patterns": len({run["pattern_id"] for run in runs if run["pattern_id"]}), "seeds": sorted({run["seed"] for run in runs}),
             **{key: summarize_values([run[key] for run in runs]) for key in ("accuracy_before", "accuracy_after", "accuracy_gain_percentage_points")}}
    for key in ("normalized_recovery", "normalized_recovery_to_clustered"):
        defined = [run[key] for run in runs if run[key] is not None]
        entry[key], entry[f"{key}_undefined_runs"] = stats_or_none(defined), len(runs) - len(defined)
    rates = [run["realized_fault_rate"] for run in runs if run["realized_fault_rate"] is not None]
    entry["realized_fault_rate"] = stats_or_none(rates)
    entry["selected_epochs"] = sorted(run["selected_epoch"] for run in runs)
    entry["runs"] = runs
    return entry


def analyze_split(runs, transfers, patterns, config, split):
    clean_runs = [metrics for (kind, *_), metrics in runs.items() if kind == "clean"]
    clean = clean_runs[0]["before"][split]["accuracy"]
    require(all(metrics["before"][split]["accuracy"] == clean for metrics in clean_runs), "Clean controls start from different accuracies")
    lora_clean = summarize_runs(clean_runs, split, clean, None)
    result = {"clean_accuracy": clean, "lora_clean": lora_clean,
              "extra_training_effect_percentage_points": {"per_seed": {str(run["seed"]): run["accuracy_gain_percentage_points"] for run in lora_clean["runs"]}, **lora_clean["accuracy_gain_percentage_points"]},
              "conditions": {}}
    for condition, rates in config.conditions.items():
        clustered_runs = [metrics for (kind, cond, *_), metrics in runs.items() if kind == "clustered" and cond == condition]
        clustered = clustered_runs[0]["before"][split]["accuracy"]
        require(all(metrics["before"][split]["accuracy"] == clustered for metrics in clustered_runs), f"{condition} clustered controls start from different accuracies")
        lora_clustered = summarize_runs(clustered_runs, split, clean, None)
        clustered_mean = lora_clustered["accuracy_after"]["mean"]
        clustered_by_seed = {run["seed"]: run["accuracy_after"] for run in lora_clustered["runs"]}
        rate_entries = []
        for rate in rates:
            names = patterns[(condition, rate)]
            group = [metrics for (kind, cond, name, _), metrics in runs.items() if kind == "faulty" and cond == condition and name in names]
            entry = {"fault_probability": rate, **summarize_runs(group, split, clean, clustered)}
            entry["fault_specific_gain_percentage_points"] = summarize_values([100 * (run["accuracy_after"] - clustered_mean) for run in entry["runs"]])
            entry["fault_specific_gain_seed_matched_percentage_points"] = summarize_values([100 * (run["accuracy_after"] - clustered_by_seed[run["seed"]]) for run in entry["runs"]])
            trained_by_pattern = {name: statistics.mean(run["accuracy_after"] for run in entry["runs"] if run["pattern_id"] == name) for name in names}
            items = [item for item in transfers if item["evaluation_base"]["condition"] == condition and item["evaluation_base"]["pattern_id"] in names]
            transfer_runs = [{"pattern_id": item["evaluation_base"]["pattern_id"], "adapter_seed": item["adapter_seed"], "accuracy_without_adapter": item["without_adapter"][split]["accuracy"],
                              "accuracy_with_adapter": item["with_adapter"][split]["accuracy"], "transfer_gain_percentage_points": item["accuracy_gain_percentage_points"][split],
                              "trained_minus_transfer_percentage_points": 100 * (trained_by_pattern[item["evaluation_base"]["pattern_id"]] - item["with_adapter"][split]["accuracy"])}
                             for item in sorted(items, key=lambda item: (item["evaluation_base"]["pattern_id"], item["adapter_seed"]))]
            entry["transfer"] = {"num_evaluations": len(transfer_runs),
                                 **{key: summarize_values([run[key] for run in transfer_runs]) for key in ("accuracy_with_adapter", "transfer_gain_percentage_points", "trained_minus_transfer_percentage_points")},
                                 "evaluations": transfer_runs}
            rate_entries.append(entry)
        result["conditions"][condition] = {"clustered_accuracy": clustered, "lora_clustered": lora_clustered,
                                           "clustering_repair_percentage_points": lora_clustered["accuracy_gain_percentage_points"], "rates": rate_entries}
    return result


def analyze(runs, transfers, config, test_names):
    patterns = validate_design(runs, transfers, config)
    return {"run_id": config.run_id, "design": {"control_seeds": config.control_seeds, "faulty_seeds": config.faulty_seeds,
                                                 "patterns": {f"{condition} {rate}": names for (condition, rate), names in patterns.items()}, "num_runs": len(runs),
                                                 "num_transfer_evaluations": len(transfers), "device": config.device},
            "definitions": DEFINITIONS, "test_sets": {name: analyze_split(runs, transfers, patterns, config, name) for name in test_names}}


TABLE_FIELDS = ["test_set", "condition", "base", "fault_probability", "num_runs", "accuracy_before_mean", "accuracy_before_sample_std", "accuracy_after_mean", "accuracy_after_sample_std",
                "accuracy_gain_percentage_points_mean", "accuracy_gain_percentage_points_sample_std", "normalized_recovery_mean", "normalized_recovery_sample_std",
                "normalized_recovery_to_clustered_mean", "normalized_recovery_to_clustered_sample_std", "fault_specific_gain_percentage_points_mean", "fault_specific_gain_percentage_points_sample_std",
                "transfer_accuracy_mean", "transfer_accuracy_sample_std", "transfer_gain_percentage_points_mean", "transfer_gain_percentage_points_sample_std",
                "trained_minus_transfer_percentage_points_mean", "trained_minus_transfer_percentage_points_sample_std"]


def table_rows(analysis):
    rows = []

    def add(test_set, condition, base, probability, entry, specific=None, transfer=None):
        row = {"test_set": test_set, "condition": condition, "base": base, "fault_probability": probability, "num_runs": entry["num_runs"]}
        values = {"accuracy_before": entry["accuracy_before"], "accuracy_after": entry["accuracy_after"], "accuracy_gain_percentage_points": entry["accuracy_gain_percentage_points"],
                  "normalized_recovery": entry.get("normalized_recovery"), "normalized_recovery_to_clustered": entry.get("normalized_recovery_to_clustered"), "fault_specific_gain_percentage_points": specific,
                  "transfer_accuracy": transfer and transfer["accuracy_with_adapter"], "transfer_gain_percentage_points": transfer and transfer["transfer_gain_percentage_points"],
                  "trained_minus_transfer_percentage_points": transfer and transfer["trained_minus_transfer_percentage_points"]}
        for name, value in values.items():
            row[f"{name}_mean"], row[f"{name}_sample_std"] = (value["mean"], value["sample_std"]) if value else (None, None)
        rows.append({name: "" if row[name] is None else str(row[name]) for name in TABLE_FIELDS})

    for test_set, result in analysis["test_sets"].items():
        add(test_set, "", "clean", "", result["lora_clean"])
        for condition, item in result["conditions"].items():
            add(test_set, condition, "clustered", 0.0, item["lora_clustered"])
            for entry in item["rates"]:
                add(test_set, condition, "faulty", entry["fault_probability"], entry, entry["fault_specific_gain_percentage_points"], entry["transfer"])
    return rows


def sweep_curves(root, runs, config):
    """Five-pattern sweep means for the plotted 'no repair' curve, from each run's own source sweep."""
    curves = {}
    for condition, rates in config.conditions.items():
        sweeps = {metrics["config"]["source_sweep_results"] for (kind, cond, *_), metrics in runs.items() if kind == "faulty" and cond == condition}
        points = {}
        for sweep in sweeps:
            for rate in json.loads((root / sweep / "aggregate.json").read_text())["rates"]:
                if rate["probability"] > 0:
                    points[rate["probability"]] = rate
        curves[condition] = [points[rate] for rate in sorted(points)]
    return curves


def plot_controls(analysis, curves, output_dir):
    """Rows: conditions; columns: test sets. No repair, LoRA trained on the pattern, and clustered-adapter transfer."""
    with tempfile.TemporaryDirectory(prefix="fault-lora-controls-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            test_names = list(analysis["test_sets"])
            conditions = list(next(iter(analysis["test_sets"].values()))["conditions"])
            fig, axes = plt.subplots(len(conditions), len(test_names), figsize=(13, 5.2 * len(conditions)), squeeze=False)
            colours = {"none": "#eb6834", "lora": "#2a78d6", "transfer": "#1baf7a", "ink": "#0b0b0b", "muted": "#52514e"}
            for row, condition in enumerate(conditions):
                for column, test_set in enumerate(test_names):
                    axis = axes[row][column]
                    result = analysis["test_sets"][test_set]
                    item = result["conditions"][condition]
                    curve = curves[condition]
                    axis.errorbar([100 * rate["realized_fault_rate"]["mean"] for rate in curve], [100 * rate["splits"][test_set]["accuracy"]["mean"] for rate in curve],
                                  yerr=[100 * (rate["splits"][test_set]["accuracy"]["sample_std"] or 0) for rate in curve], color=colours["none"], marker="o", capsize=3, linewidth=1.6,
                                  label="Faulty, no repair (sweep, 5 patterns)")
                    xs = [0.0] + [100 * entry["realized_fault_rate"]["mean"] for entry in item["rates"]]
                    lora = [item["lora_clustered"]["accuracy_after"]] + [entry["accuracy_after"] for entry in item["rates"]]
                    axis.errorbar(xs, [100 * value["mean"] for value in lora], yerr=[100 * (value["sample_std"] or 0) for value in lora], color=colours["lora"], marker="D", capsize=3,
                                  linewidth=2, label="LoRA trained on that base (patterns × seeds)")
                    for entry in item["rates"]:
                        axis.scatter([100 * run["realized_fault_rate"] for run in entry["runs"]], [100 * run["accuracy_after"] for run in entry["runs"]], s=12, color=colours["lora"], alpha=.35, linewidths=0)
                    axis.errorbar(xs[1:], [100 * entry["transfer"]["accuracy_with_adapter"]["mean"] for entry in item["rates"]],
                                  yerr=[100 * (entry["transfer"]["accuracy_with_adapter"]["sample_std"] or 0) for entry in item["rates"]], color=colours["transfer"], marker="s", capsize=3,
                                  linewidth=1.6, linestyle="--", label="Clustered-trained adapter, transferred (no training)")
                    for label, value, style in (("Clean", result["clean_accuracy"], "--"), ("Clustered, zero faults", item["clustered_accuracy"], ":"),
                                                ("LoRA on clean", result["lora_clean"]["accuracy_after"]["mean"], "-.")):
                        axis.axhline(100 * value, color=colours["muted"], linestyle=style, linewidth=1.1, label=f"{label}: {100 * value:.2f}%")
                    axis.axhline(50, color="#aaaaaa", linewidth=.8)
                    axis.set_title(f"{CONDITION_LABELS[condition]} — {test_set}", fontsize=11)
                    axis.set_xlabel("Actual corrupted cells / eligible cells (%)")
                    axis.set_ylabel("Accuracy (%)")
                    axis.set_ylim(45, 95)
                    axis.grid(alpha=.25)
                    axis.legend(loc="lower left", fontsize=7.5, frameon=False)
            fig.suptitle("TinyBERT / SST-2: LoRA recovery controls (rank 4, all Linear layers, 16 clusters per tensor)", fontsize=13)
            fig.text(.5, .01, "Means ± 1 sample SD. LoRA at 0%: clustered base, 3 seeds; faulty rates: 3 patterns × 2 seeds. Transfer: 3 clustered adapters × 3 patterns. Small dots: individual LoRA runs. Test sets shown separately.",
                     ha="center", fontsize=8, color=colours["muted"])
            fig.tight_layout(rect=(0, .03, 1, .96))
            paths = [Path(output_dir) / f"lora_controls.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=200)
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def run_controls(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Analysis directory already exists; choose a new run_id")
    matrix, runs, transfers, inputs = load_runs(root, config)
    test_names = [name for name in next(iter(runs.values()))["after"] if name != "selection"]
    analysis = analyze(runs, transfers, config, test_names)
    curves = sweep_curves(root, runs, config)
    results_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/analyze_tinybert_lora_controls.py"]
    write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
    write_json(results_dir / "inputs_sha256.json", inputs)
    write_json(results_dir / "analysis.json", analysis)
    rows = table_rows(analysis)
    with (results_dir / "conditions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    figures = plot_controls(analysis, curves, results_dir)
    # Verification: inputs unchanged and the analysis reproduces from the saved runs.
    _, runs_again, transfers_again, inputs_again = load_runs(root, config)
    require(inputs_again == inputs, "Inputs changed during the analysis")
    require(json.loads(json.dumps(analyze(runs_again, transfers_again, config, test_names))) == json.loads((results_dir / "analysis.json").read_text()), "Analysis does not reproduce")
    with (results_dir / "conditions.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == rows, "CSV and JSON tables disagree")
    write_json(results_dir / "verification.json", {"status": "passed", "runs_verified": len(runs), "transfer_evaluations": len(transfers), "inputs_unchanged": True,
                                                   "analysis_reproduced": True, "csv_matches_json": True, "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures]})
    write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return analysis


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2_lora_controls.json")
    args = parser.parse_args()
    run_controls(TinyBertControlsConfig(**json.loads(args.config.read_text())), project_root)
