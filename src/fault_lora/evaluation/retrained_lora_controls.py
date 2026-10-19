"""Step 11c analysis: LoRA controls on the retrained baseline, side by side with Step 9b.

The Step 9b formulas (lora_controls.analyze_controls) are reused unchanged. Retrained
runs are mapped onto its vocabulary: "clustered" means the zero-fault retrained base,
"faulty" a retrained base under a Step 11b pattern, and "no repair" the Step 11b
retrained measurements. The Step 9b clean-base runs are reused as the extra-training
control after re-verification. Nothing is trained or evaluated here.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.lora_controls import ControlsConfig, analyze_controls, condition_table, CONDITION_FIELDS
from fault_lora.evaluation.source_hashes import compare_source_hashes, unique_changes
from fault_lora.training.recovery_matrix import MatrixConfig, matrix_plan, run_config, verify_run


SPLITS = ("validation", "test")
RENAMED = {"retrained-clustered": "clustered", "retrained-faulty": "faulty"}
SIDE_FIELDS = ["split", "fault_probability", "baseline", "no_repair_mean", "no_repair_sample_std", "bn_recalibration_mean", "bn_recalibration_sample_std", "lora_mean", "lora_sample_std",
               "lora_minus_bn_recalibration_pp_mean", "fault_specific_gain_pp_mean", "fault_specific_gain_pp_sample_std", "normalized_recovery_mean", "normalized_recovery_sample_std"]


@dataclass
class RetrainedControlsConfig:
    run_id: str = "resnet18-cifar10-retrained-lora-controls"
    source_matrices: list = field(default_factory=lambda: [f"results/resnet18-cifar10-lora-retrained-{name}" for name in ("controls", "p001", "p005", "p010", "p020")])
    clean_control_matrix: str = "results/resnet18-cifar10-lora-controls-matrix"
    clean_control_run_ids: list = field(default_factory=lambda: [f"resnet18-cifar10-lora-clean-none-r4-train{seed}" for seed in (42, 43, 44)])
    source_bn_results: str = "results/resnet18-cifar10-retrained-bn-recalibration-seed42"
    retrained_sweep_results: str = "results/resnet18-cifar10-retrained-fault-sweep-seed42"
    step9b_analysis_results: str = "results/resnet18-cifar10-lora-controls"
    fault_probabilities: list = field(default_factory=lambda: [.01, .05, .1, .2])
    control_seeds: list = field(default_factory=lambda: [42, 43, 44])
    faulty_seeds: list = field(default_factory=lambda: [42, 43])
    rank: int = 4
    alpha: float = 4.0
    results_root: str = "results"

    def validate(self):
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if not isinstance(self.source_matrices, list) or not self.source_matrices or len(set(self.source_matrices)) != len(self.source_matrices) \
                or any(not isinstance(item, str) or "_aborted" in Path(item).parts for item in self.source_matrices):
            raise ValueError("source_matrices must be distinct result directories outside _aborted")
        if not isinstance(self.clean_control_run_ids, list) or len(set(self.clean_control_run_ids)) != len(self.clean_control_run_ids) or not self.clean_control_run_ids:
            raise ValueError("clean_control_run_ids must be distinct")
        self.controls_config().validate()

    def controls_config(self):
        return ControlsConfig(run_id=self.run_id, fault_probabilities=self.fault_probabilities, control_seeds=self.control_seeds, faulty_seeds=self.faulty_seeds,
                              rank=self.rank, alpha=self.alpha)


def to_controls_vocabulary(rows, bn_records, sweep_records, sweep_aggregate, references):
    """Express retrained inputs in the Step 9b analysis vocabulary (see module docstring)."""
    mapped_rows = [{**row, "base_kind": RENAMED.get(row["base_kind"], row["base_kind"])} for row in rows]
    require({row["base_kind"] for row in mapped_rows} <= {"clean", "clustered", "faulty"}, "Unexpected base kind")
    mapped_bn = [{**record, "condition_id": "clustered" if record["condition_id"] == "retrained-clustered" else record["condition_id"],
                  "base_kind": RENAMED.get(record["base_kind"], record["base_kind"])} for record in bn_records]
    records = [{"pattern_id": record["pattern_id"], "probability": record["probability"], "seed": record["seed"], "measurements": record["models"]["retrained"]} for record in sweep_records]
    aggregate = {"rates": [{"probability": entry["probability"], "num_patterns": entry["num_patterns"], "realized_fault_rate": entry["realized_fault_rate"],
                            "splits": {split: {"accuracy": entry["splits"][split]["accuracy"]} for split in SPLITS}} for entry in sweep_aggregate["models"]["retrained"]]}
    zero = sweep_records[0]["models"]["retrained"]
    require(sweep_records[0]["probability"] == 0, "Step 11b records must start with the zero-fault pattern")
    mapped_references = {split: {"clean": references["clean"][split], "clustered_no_faults": zero[split]} for split in SPLITS}
    plan = [{key: record[key] for key in ("pattern_id", "probability", "seed", "repeat")} for record in sweep_records]
    return mapped_rows, mapped_bn, records, plan, aggregate, mapped_references


def side_by_side(retrained, step9b, step4_no_repair, rates):
    """Per split and declared rate: no repair, BN recalibration, LoRA, and derived gaps for both baselines."""
    rows = []
    for split in SPLITS:
        for label, result, no_repair in (("step4", step9b["splits"][split], step4_no_repair[split]), ("retrained", retrained["splits"][split], None)):
            conditions = {(item["condition"], item["fault_probability"]): item for item in result["conditions"]}
            baseline = {(item["condition"], item["fault_probability"]): item for item in result["bn_recalibration_vs_lora"]}
            specific = {item["fault_probability"]: item for item in result["fault_specific_gain"]}
            sweep = {item["fault_probability"]: item["accuracy"] for item in result["step7_fault_rates"]}
            for probability in rates:
                condition = conditions.get(("clustered" if probability == 0 else "faulty", probability))
                bn = baseline.get(("clustered" if probability == 0 else "faulty", probability))
                plain = (no_repair or {}).get(probability) or sweep.get(probability)
                gain = specific.get(probability)
                stats = lambda value: (value["mean"], value["sample_std"]) if value else (None, None)
                row = {"split": split, "fault_probability": probability, "baseline": label}
                row["no_repair_mean"], row["no_repair_sample_std"] = stats(plain)
                row["bn_recalibration_mean"], row["bn_recalibration_sample_std"] = stats(bn and bn["bn_recalibration_accuracy"])
                row["lora_mean"], row["lora_sample_std"] = stats(condition and condition["accuracy_after"])
                row["lora_minus_bn_recalibration_pp_mean"] = bn and bn["lora_minus_bn_recalibration_percentage_points"]["mean"]
                row["fault_specific_gain_pp_mean"], row["fault_specific_gain_pp_sample_std"] = stats(gain and gain["gain_percentage_points"])
                row["normalized_recovery_mean"], row["normalized_recovery_sample_std"] = stats(condition and condition["normalized_recovery"])
                rows.append(row)
    return rows


def load_inputs(root, config):
    """Verify every used run and input, then return the analysis inputs and their provenance."""
    paths, accepted, rows = {}, [], []
    for directory in config.source_matrices:
        matrix_dir = root / directory
        for name in ("status.json", "verification.json", "config.json", "plan.json", "summary.json", "source_sha256.json"):
            paths[f"matrix_{matrix_dir.name}_{name}"] = matrix_dir / name
        require(json.loads((matrix_dir / "status.json").read_text())["status"] == "complete" and json.loads((matrix_dir / "verification.json").read_text())["status"] == "passed",
                f"Matrix is incomplete or unverified: {directory}")
        matrix = MatrixConfig(**json.loads((matrix_dir / "config.json").read_text()))
        require(json.loads((matrix_dir / "plan.json").read_text()) == matrix_plan(matrix), f"Matrix plan changed: {directory}")
        summary = json.loads((matrix_dir / "summary.json").read_text())
        require(len(summary) == len(matrix.runs), f"Matrix summary incomplete: {directory}")
        for run, row in zip(matrix.runs, summary):
            accepted += verify_run(root, matrix, run, row)["accepted_source_hash_changes"]
            paths[f"run_{row['run_id']}"] = root / matrix.results_root / row["run_id"] / "metrics.json"
        accepted += compare_source_hashes(root, json.loads((matrix_dir / "source_sha256.json").read_text()), f"Source code changed after matrix {directory}")
        rows += summary
    control_dir = root / config.clean_control_matrix
    control = MatrixConfig(**json.loads((control_dir / "config.json").read_text()))
    control_rows = {row["run_id"]: row for row in json.loads((control_dir / "summary.json").read_text())}
    by_id = {run_config(control, run).run_id: run for run in control.runs}
    for run_id in config.clean_control_run_ids:
        require(run_id in control_rows and run_id in by_id and control_rows[run_id]["base_kind"] == "clean", f"Clean control run is not a finished clean run: {run_id}")
        accepted += verify_run(root, control, by_id[run_id], control_rows[run_id])["accepted_source_hash_changes"]
        rows.append(control_rows[run_id])
        paths[f"run_{run_id}"] = root / control.results_root / run_id / "metrics.json"
    paths["clean_control_summary.json"] = control_dir / "summary.json"
    bn = root / config.source_bn_results
    for name in ("status.json", "verification.json", "measurements.json", "config.json"):
        paths[f"bn_{name}"] = bn / name
    require(json.loads((bn / "status.json").read_text())["status"] == "complete" and json.loads((bn / "verification.json").read_text())["status"] == "passed", "BatchNorm baseline is incomplete")
    sweep = root / config.retrained_sweep_results
    for name in ("status.json", "verification.json", "measurements.json", "aggregate.json", "references.json"):
        paths[f"retrained_sweep_{name}"] = sweep / name
    require(json.loads((sweep / "status.json").read_text())["status"] == "complete" and json.loads((sweep / "verification.json").read_text())["status"] == "passed", "Step 11b sweep is incomplete")
    step9b = root / config.step9b_analysis_results
    for name in ("status.json", "verification.json", "analysis.json"):
        paths[f"step9b_{name}"] = step9b / name
    require(json.loads((step9b / "status.json").read_text())["status"] == "complete" and json.loads((step9b / "verification.json").read_text())["status"] == "passed", "Step 9b analysis is incomplete")
    read = lambda name: json.loads(paths[name].read_text())
    return {"rows": rows, "bn": read("bn_measurements.json"), "sweep_records": read("retrained_sweep_measurements.json"), "sweep_aggregate": read("retrained_sweep_aggregate.json"),
            "references": read("retrained_sweep_references.json"), "step9b": read("step9b_analysis.json"), "accepted": accepted,
            "paths": paths, "hashes": {name: sha256_file(path) for name, path in paths.items()}}


def build_analysis(inputs, config):
    mapped = to_controls_vocabulary(inputs["rows"], inputs["bn"], inputs["sweep_records"], inputs["sweep_aggregate"], inputs["references"])
    rows, bn, records, plan, aggregate, references = mapped
    analysis = analyze_controls(rows, bn, records, plan, aggregate, references, config.controls_config())
    step4 = {split: {entry["probability"]: entry["splits"][split]["accuracy"] for entry in inputs["sweep_aggregate"]["models"]["step4"]} for split in SPLITS}
    analysis["vocabulary"] = {"clustered": "zero-fault retrained base (Step 11a codebook+BNstats)", "faulty": "retrained base under a Step 11b pattern",
                              "no repair": "Step 11b retrained measurements", "clean": "Step 9b clean-base LoRA runs, reused and re-verified"}
    analysis["side_by_side_with_step9b"] = side_by_side(analysis, inputs["step9b"], step4, [0.0] + list(config.fault_probabilities))
    analysis["accepted_source_hash_changes"] = unique_changes(inputs["accepted"])
    return analysis


def plot_side_by_side(rows, references, output_dir, split="test"):
    """Two panels (Step 4 baseline / retrained baseline): no repair, BN recalibration, LoRA."""
    colors = {"no_repair": "#eb6834", "bn_recalibration": "#1baf7a", "lora": "#2a78d6"}
    markers = {"no_repair": "o", "bn_recalibration": "s", "lora": "D"}
    names = {"no_repair": "No repair", "bn_recalibration": "BatchNorm recalibration", "lora": "LoRA"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    with tempfile.TemporaryDirectory(prefix="fault-lora-11c-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, facecolor="#fcfcfb")
            for axis, baseline, title in zip(axes, ("step4", "retrained"), ("Step 4 clustered baseline (Steps 7, 9b, 11b)", "Retrained baseline, codebook+BNstats (Steps 11b, 11c)")):
                axis.set_facecolor("#fcfcfb")
                selected = [row for row in rows if row["split"] == split and row["baseline"] == baseline and row["fault_probability"] <= .2 + 1e-12]
                for label, value, style, color in references[baseline]:
                    axis.axhline(100 * value, color=color, linestyle=style, linewidth=1.1, label=f"{label}: {100 * value:.2f}%", zorder=2)
                for method in ("no_repair", "bn_recalibration", "lora"):
                    points = [(100 * row["fault_probability"], 100 * row[f"{method}_mean"], 100 * (row[f"{method}_sample_std"] or 0)) for row in selected if row[f"{method}_mean"] is not None]
                    if points:
                        axis.errorbar([p[0] for p in points], [p[1] for p in points], yerr=[p[2] for p in points], color=colors[method], marker=markers[method], markersize=8,
                                      markeredgecolor="#fcfcfb", markeredgewidth=1.5, linewidth=2, capsize=3, elinewidth=1.2, label=names[method], zorder=4)
                axis.set_title(title, fontsize=11, color=ink)
                axis.set_xlabel("Requested fault probability (%)", color=muted)
                axis.set_xticks([0, 1, 5, 10, 20])
                axis.grid(color=grid, linewidth=.8)
                axis.set_axisbelow(True)
                axis.tick_params(colors=muted)
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    axis.spines[side].set_color(grid)
                axis.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=ink)
            axes[0].set_ylabel(f"{split.capitalize()} accuracy (%)", color=muted)
            fig.suptitle("ResNet-18 / CIFAR-10: no repair, BatchNorm recalibration, and LoRA on two clustered baselines", fontsize=12.5, color=ink)
            fig.text(.5, .015, "Mean ± 1 sample SD. No repair and BN recalibration: across 5 patterns; LoRA: 5 patterns × 2 training seeds (3 seeds at 0%). "
                               "Step 4 LoRA and BN recalibration were run to 10% only.", ha="center", fontsize=8, color=muted)
            fig.tight_layout(rect=(0, .05, 1, .95))
            paths = [Path(output_dir) / f"retrained_lora_controls.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def figure_references(analysis, step9b, split="test"):
    def lora(result, condition, probability):
        return next(item["accuracy_after"]["mean"] for item in result["conditions"] if item["condition"] == condition and item["fault_probability"] == probability)
    muted, blue = "#52514e", "#2a78d6"
    clean = analysis["splits"][split]["references"]["clean_accuracy"]
    return {"step4": [("Clean", clean, "--", muted), ("LoRA on clean", lora(step9b["splits"][split], "clean", None), "--", blue),
                      ("LoRA on Step 4 clustered", lora(step9b["splits"][split], "clustered", 0.0), ":", blue)],
            "retrained": [("Clean", clean, "--", muted), ("LoRA on clean", lora(analysis["splits"][split], "clean", None), "--", blue),
                          ("LoRA on retrained clustered", lora(analysis["splits"][split], "clustered", 0.0), ":", blue)]}


def run_retrained_controls_analysis(config, project_root):
    config.validate()
    root = Path(project_root)
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Analysis directory already exists; choose a new run_id")
    inputs = load_inputs(root, config)
    analysis = build_analysis(inputs, config)
    results_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/analyze_retrained_lora_controls.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu; no model evaluation",
                       "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")}}
        write_json(results_dir / "environment.json", environment)
        write_json(results_dir / "analysis.json", analysis)
        for name, rows, fields in (("conditions.csv", condition_table(analysis), CONDITION_FIELDS),
                                   ("side_by_side.csv", [{key: "" if row[key] is None else str(row[key]) for key in SIDE_FIELDS} for row in analysis["side_by_side_with_step9b"]], SIDE_FIELDS)):
            with (results_dir / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        figures = plot_side_by_side(analysis["side_by_side_with_step9b"], figure_references(analysis, inputs["step9b"]), results_dir)
        again = load_inputs(root, config)
        require(again["hashes"] == inputs["hashes"], "An input changed during the analysis")
        require(json.loads((results_dir / "analysis.json").read_text()) == json.loads(json.dumps(build_analysis(again, config))), "Saved analysis differs from its recomputation")
        compare_source_hashes(root, source_code, "Source code changed during the analysis")
        manifest = {"run_id": config.run_id, "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": inputs["hashes"][name]} for name, path in inputs["paths"].items()},
                    "vocabulary": analysis["vocabulary"], "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures],
                    "environment": environment, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "verification.json", {"status": "passed", "runs_reverified": len(inputs["rows"]), "clean_control_runs_reverified": len(config.clean_control_run_ids),
                                                      "design_matches_configuration": True, "analysis_recomputed_from_reread_inputs": True, "input_artifacts_unchanged": len(inputs["hashes"]),
                                                      "accepted_source_hash_changes": analysis["accepted_source_hash_changes"]})
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "retrained_lora_controls_complete", "run_id": config.run_id}), flush=True)
        return analysis
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_retrained_lora_controls.json"))
    args = parser.parse_args()
    run_retrained_controls_analysis(RetrainedControlsConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
