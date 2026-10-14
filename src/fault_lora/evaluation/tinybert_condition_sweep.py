"""Step 10c fault sweeps: harsher TinyBERT conditions with the unchanged simulator.

- Condition A (`linear-high-rate`): the audited Step 10b Linear-only encoding at
  higher rates, continuing Step 10b's rate labels and seeds. Its clustered
  reference is replayed and must equal Step 10b's.
- Condition B (`linear-and-embeddings`): the audited Linear + embedding encoding,
  with its own zero-fault reference `rate00-zero`.

Pattern IDs follow Step 7: `rateNN-seedS`, with seeds allocated consecutively in
ascending rate order. Each split, including selection, is aggregated separately.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile
import time

import torch

from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import inspect_transitions, patterns_equal, require, require_state_equal
from fault_lora.evaluation.fault_sweep import summarize_values
from fault_lora.evaluation.tinybert_clean_eval import environment_record, source_artifacts, verify_source_artifacts, verify_source_code
from fault_lora.evaluation.tinybert_condition_audit import ConditionAuditConfig, load_embedding_clustered_run
from fault_lora.evaluation.tinybert_fault_checks import REQUIRED_CHECKS
from fault_lora.evaluation.tinybert_fault_sweep import LOSS_KEYS, TinyBertSweepConfig, csv_rows, prepare_sweep_sources, write_measurements_csv
from fault_lora.evaluation.tinybert_sources import build_tinybert, evaluate_splits, make_loaders
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern
from fault_lora.training.baseline import choose_device

CONDITIONS = {
    "linear-high-rate": "A: nn.Linear weights only (Step 10b encoding); embeddings, LayerNorm, and biases reliable",
    "linear-and-embeddings": "B: nn.Linear weights and the word, position, and token-type embedding tables; LayerNorm and biases reliable",
}


@dataclass
class ConditionSweepConfig:
    run_id: str = "tinybert-sst2-fault-sweep-linear-high-seed72"
    condition: str = "linear-high-rate"
    probabilities: list = None
    patterns_per_rate: int = 5
    first_seed: int = 72
    first_rate_label: int = 7
    source_audit_results: str = "results/tinybert-sst2-fault-checks-seed42"
    previous_sweep_results: str | None = "results/tinybert-sst2-fault-sweep-seed42"
    device: str = "cpu"
    cpu_threads: int = 4
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {sorted(CONDITIONS)}")
        rates = self.probabilities
        if not isinstance(rates, list) or not rates or any(isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or not 0 <= p <= 1 for p in rates):
            raise ValueError("probabilities must be a nonempty list of values in [0, 1]")
        if rates != sorted(set(rates)) or not any(p > 0 for p in rates):
            raise ValueError("probabilities must be sorted, unique, and include a nonzero rate")
        for name, minimum in (("patterns_per_rate", 2), ("first_seed", 0), ("first_rate_label", 1), ("cpu_threads", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        nonzero = [p for p in rates if p > 0]
        if self.first_seed + len(nonzero) * self.patterns_per_rate > 2**32 or self.first_rate_label + len(nonzero) > 100:
            raise ValueError("Seeds or rate labels exceed their ranges")
        if self.condition == "linear-high-rate" and (rates[0] == 0 or not self.previous_sweep_results):
            raise ValueError("Condition A extends Step 10b: no zero rate, and previous_sweep_results is required")
        if self.condition == "linear-and-embeddings" and (rates[0] != 0 or self.previous_sweep_results is not None):
            raise ValueError("Condition B needs its own zero-fault reference first and no previous sweep")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def condition_plan(config):
    """Step 7's allocation rule with a configurable first label and seed; a zero rate is `rate00-zero`."""
    config.validate()
    plan = []
    rates = list(config.probabilities)
    if rates[0] == 0:
        plan.append({"pattern_id": "rate00-zero", "probability": 0.0, "seed": config.first_seed, "repeat": 0})
        rates = rates[1:]
    for ordinal, probability in enumerate(rates):
        for repeat in range(config.patterns_per_rate):
            seed = config.first_seed + ordinal * config.patterns_per_rate + repeat
            plan.append({"pattern_id": f"rate{config.first_rate_label + ordinal:02d}-seed{seed}", "probability": float(probability), "seed": seed, "repeat": repeat})
    return plan


def aggregate_condition(records, plan, probabilities, split_names):
    """Per-rate means and sample SDs for every split; rejects incomplete or off-plan records."""
    expected = {entry["pattern_id"]: entry for entry in plan}
    require(len(records) == len(plan) and len({record["pattern_id"] for record in records}) == len(plan), "Sweep measurements are incomplete or duplicated")
    for record in records:
        entry = expected.get(record["pattern_id"])
        require(entry is not None and record["probability"] == entry["probability"] and record["seed"] == entry["seed"], "Measurement differs from the predeclared plan")
        require(list(record["measurements"]) == list(split_names), "Every pattern needs every split, in order")
    rates = []
    for probability in probabilities:
        group = [record for record in records if record["probability"] == probability]
        require(group and len({record["counts"]["eligible_cells"] for record in group}) == 1, "Missing rate or differing denominators")
        rate = {"probability": float(probability), "num_patterns": len(group), "pattern_ids": [record["pattern_id"] for record in group], "seeds": [record["seed"] for record in group],
                "eligible_cells": group[0]["counts"]["eligible_cells"],
                "actual_corrupted_cells": summarize_values([record["counts"]["actual_corrupted_cells"] for record in group]),
                "realized_fault_rate": summarize_values([record["counts"]["realized_fault_rate"] for record in group]), "splits": {}}
        for split in split_names:
            require(len({record["measurements"][split]["num_examples"] for record in group}) == 1, "Evaluation example counts differ")
            rate["splits"][split] = {"num_examples": group[0]["measurements"][split]["num_examples"],
                **{key: summarize_values([record["measurements"][split][key] for record in group]) for key in ("accuracy", "cross_entropy_loss")},
                **{key: summarize_values([record["fault_loss"][split][key] for record in group]) for key in LOSS_KEYS}}
        rates.append(rate)
    return {"splits": list(split_names), "rates": rates,
            "variability": "Sample standard deviation across independent static patterns (ddof=1); not a confidence interval or variation over training seeds. A single zero-fault reference has sample_std null.",
            "reporting": "Each split is aggregated separately; test sets are never pooled; the selection split is for selection only, not a held-out result."}


def plot_condition(series, references, test_names, title, footnote, output_dir):
    """One panel per test set; each series is (label, records, aggregate, colour, marker)."""
    with tempfile.TemporaryDirectory(prefix="fault-lora-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            titles = {test_names[0]: "SST-2 validation", test_names[1]: "SST-2 original test, overlap removed"}
            fig, axes = plt.subplots(1, len(test_names), figsize=(12.5, 5.6), sharey=True)
            visible = [100 * references[split][kind]["accuracy"] for split in test_names for kind in ("clean", "clustered_no_faults")]
            for _, records, aggregate, _, _ in series:
                visible += [100 * record["measurements"][split]["accuracy"] for record in records for split in test_names]
                for rate in aggregate["rates"]:
                    for split in test_names:
                        values = rate["splits"][split]["accuracy"]
                        visible += [100 * (values["mean"] - (values["sample_std"] or 0)), 100 * (values["mean"] + (values["sample_std"] or 0))]
            lower, upper = max(0, 5 * math.floor((min(visible) - 3) / 5)), min(100, 5 * math.ceil((max(visible) + 3) / 5))
            for axis, split in zip(axes, test_names):
                for label, records, aggregate, colour, marker in series:
                    axis.scatter([100 * record["counts"]["realized_fault_rate"] for record in records], [100 * record["measurements"][split]["accuracy"] for record in records],
                                 s=22, color=colour, alpha=.4, marker=marker, zorder=3)
                    axis.errorbar([100 * rate["realized_fault_rate"]["mean"] for rate in aggregate["rates"]], [100 * rate["splits"][split]["accuracy"]["mean"] for rate in aggregate["rates"]],
                                  yerr=[100 * (rate["splits"][split]["accuracy"]["sample_std"] or 0) for rate in aggregate["rates"]],
                                  fmt=f"{marker}-", color=colour, capsize=4, linewidth=1.6, markersize=6, label=f"{label}: mean ± sample SD (light: patterns)", zorder=4)
                axis.axhline(100 * references[split]["clean"]["accuracy"], color="#26734d", linestyle="--", linewidth=1.3, label="Original clean model")
                axis.axhline(100 * references[split]["clustered_no_faults"]["accuracy"], color="#a76a2d", linestyle=":", linewidth=1.5, label="Clustered zero-fault reference")
                axis.axhline(50, color="#888888", linestyle="-.", linewidth=1, label="Chance (two classes)")
                axis.set_title(f"{titles[split]} ({references[split]['clean']['num_examples']:,} sentences)")
                axis.set_xlabel("Actual corrupted cells / eligible cells (%)")
                axis.set_ylim(min(lower, 45), upper)
                axis.grid(alpha=.22)
                axis.legend(loc="lower left", fontsize=7.5)
            axes[0].set_ylabel("Classification accuracy (%)")
            fig.suptitle(title, fontsize=13)
            fig.text(.5, .015, footnote, ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .055, 1, .94))
            paths = [Path(output_dir) / "accuracy_vs_fault_rate.png", Path(output_dir) / "accuracy_vs_fault_rate.pdf"]
            for path in paths:
                fig.savefig(path, dpi=200)
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def load_previous_sweep(root, config, sources):
    """Condition A's Step 10b sweep: verified status, recorded hashes, and no label or seed collision."""
    directory = root / config.previous_sweep_results
    read = lambda name: json.loads((directory / name).read_text())
    require(read("status.json")["status"] == "complete" and read("verification.json")["status"] == "passed", "Step 10b sweep is incomplete or unverified")
    verify_source_artifacts(root, read("manifest.json"), "Step 10b sweep")
    for name in ("config", "plan", "measurements", "aggregate", "references", "manifest", "status", "verification"):
        sources.paths[f"previous_sweep_{name}"], sources.hashes[f"previous_sweep_{name}"] = directory / f"{name}.json", sha256_file(directory / f"{name}.json")
    previous_plan = read("plan.json")
    plan = condition_plan(config)
    require(not {entry["pattern_id"] for entry in plan} & {entry["pattern_id"] for entry in previous_plan}, "Pattern IDs collide with Step 10b")
    require(min(entry["seed"] for entry in plan) > max(entry["seed"] for entry in previous_plan), "Seeds must continue after Step 10b's")
    require(min(int(entry["pattern_id"][4:6]) for entry in plan) > max(int(entry["pattern_id"][4:6]) for entry in previous_plan), "Rate labels must continue after Step 10b's")
    require(min(config.probabilities) > max(read("config.json")["probabilities"]), "Condition A rates must exceed Step 10b's")
    return read("measurements.json"), read("aggregate.json"), read("references.json")


def prepare_condition_sources(root, config):
    """Sources, clean metrics, the clustering run's metrics, and the audited encoding for either condition."""
    if config.condition == "linear-high-rate":
        return prepare_sweep_sources(root, TinyBertSweepConfig(source_audit_results=config.source_audit_results))
    audit_dir = root / config.source_audit_results
    read = lambda name: json.loads((audit_dir / name).read_text())
    require(read("status.json")["status"] == "complete", "Condition B audit is incomplete")
    audit = read("report.json")
    require(audit["status"] == "passed" and all(audit["checks"].get(name, {}).get("passed") for name in REQUIRED_CHECKS), "Condition B audit did not pass")
    audit_config = ConditionAuditConfig(**read("config.json"))
    audit_config.validate()
    verify_source_artifacts(root, audit, "Condition B audit")
    verify_source_code(root, audit_dir, "Condition B audit")
    _, sources, clean_metrics, clustered_metrics, encoding, _ = load_embedding_clustered_run(root, audit_config.source_clustered_results)
    require(encoding_fingerprint(encoding) == audit["checks"]["source_integrity"]["encoding_fingerprint"], "Encoding differs from the audited storage")
    for name in ("report", "config", "status", "source_sha256"):
        sources.paths[f"audit_{name}"], sources.hashes[f"audit_{name}"] = audit_dir / f"{name}.json", sha256_file(audit_dir / f"{name}.json")
    return sources, clean_metrics, clustered_metrics, encoding


def figure_series(config, records, aggregate, previous):
    if config.condition == "linear-high-rate":
        return [("Step 10b sweep, 0–10%", previous[0], previous[1], "#527eaa", "o"), ("Step 10c condition A, 20–100%", records, aggregate, "#b2392f", "s")]
    return [("Step 10c condition B, 0–50%", records, aggregate, "#6a3d9a", "D")]


def figure_text(config, aggregate):
    repeats = sorted({rate["num_patterns"] for rate in aggregate["rates"] if rate["probability"] > 0})
    if config.condition == "linear-high-rate":
        return ("TinyBERT / SST-2, condition A: faults in Linear weights only, 0–100% (16 clusters per tensor)",
                f"Circles: Step 10b run (0–10%); squares: Step 10c condition A run (20–100%); {repeats} patterns per nonzero rate. Same encoding and clustered reference; embeddings, LayerNorm, biases reliable.")
    return ("TinyBERT / SST-2, condition B: faults in Linear weights and embeddings (16 clusters per tensor)",
            f"{repeats} patterns per nonzero rate; zero faults: one reference. Embedding tables clustered and faulted; LayerNorm, biases, codebooks reliable.")


def verify_condition_sweep(root, config):
    """Re-read every saved pattern and record; recompute counts, decoding, aggregates, and CSV."""
    root = Path(root).resolve()
    results_dir = root / config.results_root / config.run_id
    read = lambda name: json.loads((results_dir / name).read_text())
    require(read("config.json") == asdict(config), "Saved configuration differs")
    plan = condition_plan(config)
    require(read("plan.json") == plan, "Saved plan differs from the configuration")
    records, manifest, references = read("measurements.json"), read("manifest.json"), read("references.json")
    verify_source_artifacts(root, manifest, "Condition sweep")
    verify_source_code(root, results_dir, "Condition sweep")
    _, _, _, encoding = prepare_condition_sources(root, config)
    require(encoding_fingerprint(encoding) == manifest["encoding_fingerprint"], "Encoding fingerprint differs")
    splits = manifest["splits"]
    require([record["pattern_id"] for record in records] == [entry["pattern_id"] for entry in plan], "Measurements are incomplete or reordered")
    for record in records:
        require(read(f"patterns/{record['pattern_id']}.json") == record, f"Per-pattern record differs: {record['pattern_id']}")
        path = root / record["pattern"]
        require(sha256_file(path) == record["pattern_sha256"], f"Saved pattern changed: {record['pattern_id']}")
        pattern = torch.load(path, map_location="cpu", weights_only=True)
        require(summarize_fault_pattern(encoding, pattern) == record["counts"], f"Recomputed counts differ: {record['pattern_id']}")
        require(patterns_equal(generate_fault_pattern(encoding, record["probability"], seed=record["seed"]), pattern), f"Seed does not regenerate the saved pattern: {record['pattern_id']}")
        independent = inspect_transitions(encoding, pattern, apply_fault_pattern(encoding, pattern))
        require(independent["actual_corrupted_cells"] == record["counts"]["actual_corrupted_cells"], f"Independent decoding count differs: {record['pattern_id']}")
        del pattern
        for split in splits:
            measurement, clustered = record["measurements"][split], references[split]["clustered_no_faults"]
            require(measurement["accuracy"] == measurement["num_correct"] / measurement["num_examples"], f"Inconsistent accuracy: {record['pattern_id']} {split}")
            require(record["fault_loss"][split] == {"accuracy_drop_percentage_points": 100 * (clustered["accuracy"] - measurement["accuracy"]),
                                                    "cross_entropy_increase": measurement["cross_entropy_loss"] - clustered["cross_entropy_loss"]}, f"Fault loss differs: {record['pattern_id']} {split}")
    if plan[0]["probability"] == 0:
        require(records[0]["counts"]["actual_corrupted_cells"] == 0 and all(records[0]["measurements"][split] == references[split]["clustered_no_faults"] for split in splits), "Zero pattern is not the clustered reference")
    require(read("aggregate.json") == json.loads(json.dumps(aggregate_condition(records, plan, config.probabilities, splits))), "Aggregate differs from a recomputation")
    with (results_dir / "measurements.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == csv_rows(records), "CSV and JSON measurements disagree")
    for plot in manifest["plots"]:
        require(sha256_file(root / plot["path"]) == plot["sha256"], f"Plot changed: {plot['path']}")
    return {"status": "passed", "patterns_verified": len(records), "pattern_hashes_counts_and_seed_regeneration_verified": True, "independent_decoding_verified": True,
            "per_pattern_records_match_measurements": True, "fault_losses_recomputed": True, "aggregate_recomputed": True, "csv_matches_json": True,
            "plot_hashes_verified": len(manifest["plots"]), "input_artifact_hashes_verified": len(manifest["source_artifacts"]), "source_code_unchanged": True}


def run_condition_sweep(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    sources, clean_metrics, clustered_metrics, encoding = prepare_condition_sources(root, config)
    previous = load_previous_sweep(root, config, sources) if config.condition == "linear-high-rate" else None
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Sweep directories already exist; choose a new run_id")
    (results_dir / "patterns").mkdir(parents=True)
    (checkpoint_dir / "patterns").mkdir(parents=True)
    started = time.monotonic()
    plan = condition_plan(config)
    records = []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "plan.json", plan)
    write_json(results_dir / "status.json", {"status": "running", "completed_patterns": 0, "planned_patterns": len(plan), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_condition_sweep.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        environment = {**environment_record(device, config.cpu_threads, ("matplotlib",)), "fault_sampling_device": "cpu"}
        write_json(results_dir / "environment.json", environment)
        write_json(results_dir / "test_sets.json", sources.test_sets)
        fingerprint = encoding_fingerprint(encoding)
        baseline_state = decode_state(encoding)
        loaders = make_loaders(sources)
        splits = list(loaders)
        references = {}
        for split, metrics in evaluate_splits(sources.model.to(device), loaders, device).items():
            references[split] = {"clean": metrics, "clean_num_correct_difference_from_clean_eval": metrics["num_correct"] - clean_metrics["measurements"][split]["num_correct"]}
        require(all(item["clean_num_correct_difference_from_clean_eval"] == 0 for item in references.values()), "Clean replay differs from the two-test clean evaluation")
        model = build_tinybert(encoding["model_config"]).to(device)

        def set_clustered_reference(measurements):
            for split in splits:
                clustered = measurements[split]
                references[split].update({
                    "clustered_no_faults": clustered,
                    "clustered_num_correct_difference_from_clustering_run": clustered["num_correct"] - clustered_metrics["measurements"][split]["clustered_no_faults"]["num_correct"],
                    "clustering_accuracy_drop_percentage_points": 100 * (references[split]["clean"]["accuracy"] - clustered["accuracy"]),
                })
            require(all(item["clustered_num_correct_difference_from_clustering_run"] == 0 for item in references.values()), "Zero-fault replay differs from the clustering run")
            if previous is not None:
                for split in splits:
                    require(references[split]["clean"] == previous[2][split]["clean"] and references[split]["clustered_no_faults"] == previous[2][split]["clustered_no_faults"],
                            f"References differ from the Step 10b sweep: {split}")
            write_json(results_dir / "references.json", references)

        if plan[0]["probability"] > 0:
            # Condition A has no zero pattern of its own: replay the uncorrupted encoding as the reference.
            model.load_state_dict(baseline_state, strict=True)
            set_clustered_reference(evaluate_splits(model, loaders, device))
        for entry in plan:
            print(json.dumps({"event": "pattern_evaluation_start", **entry, "completed_patterns": len(records), "planned_patterns": len(plan)}), flush=True)
            pattern_started = time.monotonic()
            path = checkpoint_dir / "patterns" / f"{entry['pattern_id']}.pt"
            save_torch(path, generate_fault_pattern(encoding, entry["probability"], seed=entry["seed"]))
            saved_pattern = torch.load(path, map_location="cpu", weights_only=True)
            counts = summarize_fault_pattern(encoding, saved_pattern)
            state = apply_fault_pattern(encoding, saved_pattern)
            if entry["probability"] == 0:
                require_state_equal(state, baseline_state, "Sweep zero-fault identity")
            for name, value in encoding["untouched_state"].items():
                require(torch.equal(state[name], value), f"Reliable state changed: {name}")
            model.load_state_dict(state, strict=True)
            measurements = evaluate_splits(model, loaders, device)
            if entry["probability"] == 0:
                set_clustered_reference(measurements)
            fault_loss = {split: {"accuracy_drop_percentage_points": 100 * (references[split]["clustered_no_faults"]["accuracy"] - measurements[split]["accuracy"]),
                                  "cross_entropy_increase": measurements[split]["cross_entropy_loss"] - references[split]["clustered_no_faults"]["cross_entropy_loss"]} for split in splits}
            record = {**entry, "pattern": str(path.relative_to(root)), "pattern_sha256": sha256_file(path), "counts": counts,
                      "measurements": measurements, "fault_loss": fault_loss, "elapsed_seconds": time.monotonic() - pattern_started}
            write_json(results_dir / "patterns" / f"{entry['pattern_id']}.json", record)
            records.append(record)
            write_json(results_dir / "measurements.json", records)
            write_json(results_dir / "status.json", {"status": "running", "completed_patterns": len(records), "planned_patterns": len(plan)})
            print(json.dumps({"event": "pattern_evaluation_complete", **entry, "actual_corrupted_cells": counts["actual_corrupted_cells"], "realized_fault_rate": counts["realized_fault_rate"],
                              "accuracy": {split: measurements[split]["accuracy"] for split in splits}, "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            del saved_pattern, state
        aggregate = aggregate_condition(records, plan, config.probabilities, splits)
        write_json(results_dir / "aggregate.json", aggregate)
        write_measurements_csv(results_dir / "measurements.csv", records)
        title, footnote = figure_text(config, aggregate)
        figures = plot_condition(figure_series(config, records, aggregate, previous), references, sources.test_names, title, footnote, results_dir)
        require(encoding_fingerprint(encoding) == fingerprint, "Source encoding changed during the sweep")
        for name, path in sources.paths.items():
            require(sha256_file(path) == sources.hashes[name], f"Source artifact changed: {name}")
        first_nonzero = next(record for record in records if record["probability"] > 0)
        write_json(results_dir / "manifest.json", {
            "run_id": config.run_id, "condition": config.condition, "condition_description": CONDITIONS[config.condition],
            "source_artifacts": source_artifacts(root, sources.paths, sources.hashes), "encoding_fingerprint": fingerprint,
            "eligible_tensor_names": list(encoding["clustered_tensors"]), "eligible_cells": first_nonzero["counts"]["eligible_cells"],
            "splits": splits, "test_set_names": sources.test_names, "split_sha256": encoding["split_sha256"], "test_set_sha256": encoding["test_set_sha256"],
            "evaluation": {"batch_size": sources.config.evaluation_batch_size, "max_length": sources.config.max_length},
            "seed_rule": "Step 7 allocation: seed = first_seed + nonzero-rate ordinal * patterns_per_rate + repeat; label = first_rate_label + ordinal; a zero rate is rate00-zero with first_seed.",
            "fault_model": {key: value for key, value in torch.load(root / first_nonzero["pattern"], map_location="cpu", weights_only=True).items() if key not in ("tensors", "tensor_order", "seed", "fault_probability")},
            "selection_policy": "Rates and seed plan fixed before evaluation; no training, checkpoint selection, or tuning against any result.",
            "reporting": "Selection split and each test set reported separately; test sets never pooled; selection accuracy is not a held-out estimate for this checkpoint.",
            "comparisons": "Fault loss: this condition's clustered zero-fault reference versus each saved pattern; signed drops are allowed.",
            "reset_policy": "Every state reconstructed from the uncorrupted encoding; the full state is loaded before each evaluation; the saved pattern is static across all splits.",
            "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures],
            "completed_patterns": len(records), "environment": environment, "elapsed_seconds": time.monotonic() - started,
        })
        write_json(results_dir / "verification.json", verify_condition_sweep(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "completed_patterns": len(records), "planned_patterns": len(plan), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "condition_sweep_complete", "run_id": config.run_id, "completed_patterns": len(records), "elapsed_seconds": time.monotonic() - started}), flush=True)
        return aggregate
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_patterns": len(records), "planned_patterns": len(plan), "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_condition_sweep(ConditionSweepConfig(**json.loads(args.config.read_text())), project_root)
