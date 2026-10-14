"""Step 7's static-pattern fault sweep on the audited TinyBERT encoding.

Plan, seeds, pattern IDs, simulator, and reset policy are the Step 7 ones
(`fault_sweep.sweep_plan`, `memory/faults.py`). Measurements cover the
selection split and both test sets, each reported separately.
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
import tempfile
import time

import torch

from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import inspect_transitions, patterns_equal, require, require_state_equal
from fault_lora.evaluation.fault_sweep import SweepConfig, summarize_values, sweep_plan
from fault_lora.evaluation.tinybert_clean_eval import environment_record, source_artifacts, verify_source_artifacts, verify_source_code
from fault_lora.evaluation.tinybert_fault_checks import REQUIRED_CHECKS, TinyBertFaultCheckConfig, load_clustered_run
from fault_lora.evaluation.tinybert_sources import build_tinybert, evaluate_splits, make_loaders
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern
from fault_lora.training.baseline import choose_device

LOSS_KEYS = ("accuracy_drop_percentage_points", "cross_entropy_increase")
CSV_FIELDS = ["pattern_id", "probability", "seed", "eligible_cells", "actual_corrupted_cells", "realized_fault_rate", "split", "num_examples", "num_correct", "accuracy", "cross_entropy_loss", *LOSS_KEYS]


@dataclass
class TinyBertSweepConfig:
    run_id: str = "tinybert-sst2-fault-sweep-seed42"
    probabilities: list = field(default_factory=lambda: [0, .001, .005, .01, .02, .05, .1])
    patterns_per_rate: int = 5
    seed: int = 42
    device: str = "cpu"
    cpu_threads: int = 4
    source_audit_results: str = "results/tinybert-sst2-fault-checks-seed42"
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def step7_config(self):
        """The Step 7 configuration whose validation and seed plan are reused unchanged."""
        return SweepConfig(run_id=self.run_id, probabilities=self.probabilities, patterns_per_rate=self.patterns_per_rate, seed=self.seed,
                           device=self.device, cpu_threads=self.cpu_threads, num_workers=0)

    def validate(self):
        self.step7_config().validate()


def aggregate_tinybert(records, config, split_names):
    """Per-rate means and sample SDs for every split; rejects incomplete or off-plan records."""
    plan = sweep_plan(config.step7_config())
    expected = {entry["pattern_id"]: entry for entry in plan}
    require(len(records) == len(plan) and len({record["pattern_id"] for record in records}) == len(plan), "Sweep measurements are incomplete or duplicated")
    for record in records:
        entry = expected.get(record["pattern_id"])
        require(entry is not None and record["probability"] == entry["probability"] and record["seed"] == entry["seed"], "Measurement differs from the predeclared plan")
        require(list(record["measurements"]) == list(split_names), "Every pattern needs every split, in order")
    rates = []
    for probability in config.probabilities:
        group = [record for record in records if record["probability"] == probability]
        require(len({record["counts"]["eligible_cells"] for record in group}) == 1, "Fault-rate denominators differ")
        rate = {"probability": float(probability), "num_patterns": len(group), "seeds": [record["seed"] for record in group], "eligible_cells": group[0]["counts"]["eligible_cells"],
                "actual_corrupted_cells": summarize_values([record["counts"]["actual_corrupted_cells"] for record in group]),
                "realized_fault_rate": summarize_values([record["counts"]["realized_fault_rate"] for record in group]), "splits": {}}
        for split in split_names:
            require(len({record["measurements"][split]["num_examples"] for record in group}) == 1, "Evaluation example counts differ")
            rate["splits"][split] = {"num_examples": group[0]["measurements"][split]["num_examples"],
                **{key: summarize_values([record["measurements"][split][key] for record in group]) for key in ("accuracy", "cross_entropy_loss")},
                **{key: summarize_values([record["fault_loss"][split][key] for record in group]) for key in LOSS_KEYS}}
        rates.append(rate)
    return {"splits": list(split_names), "rates": rates,
            "variability": "Sample standard deviation across independent static patterns (ddof=1); not a confidence interval or variation over training seeds. Zero rate is one deterministic reference; sample_std is null.",
            "reporting": "Each split is aggregated separately; test sets are never pooled."}


def csv_rows(records):
    rows = []
    for record in records:
        for split, metrics in record["measurements"].items():
            row = {**{key: record[key] for key in ("pattern_id", "probability", "seed")},
                   **{key: record["counts"][key] for key in ("eligible_cells", "actual_corrupted_cells", "realized_fault_rate")},
                   "split": split, **metrics, **record["fault_loss"][split]}
            rows.append({key: str(row[key]) for key in CSV_FIELDS})
    return rows


def write_measurements_csv(path, records):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows(records))


def plot_tinybert_sweep(records, aggregate, references, test_names, output_dir):
    """One panel per test set: individual patterns, mean ± sample SD, clean and clustered references."""
    with tempfile.TemporaryDirectory(prefix="fault-lora-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            titles = {test_names[0]: "SST-2 validation", test_names[1]: "SST-2 original test, overlap removed"}
            fig, axes = plt.subplots(1, len(test_names), figsize=(12, 5.4), sharey=True)
            visible = [100 * record["measurements"][split]["accuracy"] for record in records for split in test_names]
            visible += [100 * references[split][kind]["accuracy"] for split in test_names for kind in ("clean", "clustered_no_faults")]
            for rate in aggregate["rates"]:
                for split in test_names:
                    values = rate["splits"][split]["accuracy"]
                    visible.extend([100 * (values["mean"] - (values["sample_std"] or 0)), 100 * (values["mean"] + (values["sample_std"] or 0))])
            lower, upper = max(0, 5 * math.floor((min(visible) - 3) / 5)), min(100, 5 * math.ceil((max(visible) + 3) / 5))
            for axis, split in zip(axes, test_names):
                axis.scatter([100 * record["counts"]["realized_fault_rate"] for record in records], [100 * record["measurements"][split]["accuracy"] for record in records],
                             s=24, color="#527eaa", alpha=.55, label="Individual static patterns", zorder=3)
                axis.errorbar([100 * rate["realized_fault_rate"]["mean"] for rate in aggregate["rates"]], [100 * rate["splits"][split]["accuracy"]["mean"] for rate in aggregate["rates"]],
                              yerr=[100 * (rate["splits"][split]["accuracy"]["sample_std"] or 0) for rate in aggregate["rates"]],
                              fmt="o-", color="#174c78", capsize=4, linewidth=1.6, label="Mean ± sample SD", zorder=4)
                axis.axhline(100 * references[split]["clean"]["accuracy"], color="#26734d", linestyle="--", linewidth=1.3, label="Original clean model")
                axis.axhline(100 * references[split]["clustered_no_faults"]["accuracy"], color="#a76a2d", linestyle=":", linewidth=1.5, label="Clustered zero-fault reference")
                axis.set_title(f"{titles[split]} ({references[split]['clean']['num_examples']:,} sentences)")
                axis.set_xlabel("Actual corrupted cells / eligible Linear-weight cells (%)")
                axis.set_ylim(lower, upper)
                axis.grid(alpha=.22)
                axis.legend(loc="lower left", fontsize=8)
            axes[0].set_ylabel("Classification accuracy (%)")
            fig.suptitle("TinyBERT / SST-2: adjacent-level faults in Linear weights, 16 clusters per tensor", fontsize=13)
            repeats = sorted({rate["num_patterns"] for rate in aggregate["rates"] if rate["probability"] > 0})
            fig.text(.5, .015, f"Nonzero rates: {repeats} patterns each; zero faults: one reference. Uniform selection; embeddings, LayerNorm, biases, codebooks reliable.", ha="center", fontsize=8)
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


def prepare_sweep_sources(root, config):
    audit_dir = root / config.source_audit_results
    read = lambda name: json.loads((audit_dir / name).read_text())
    require(read("status.json")["status"] == "complete", "TinyBERT fault audit is incomplete")
    audit = read("report.json")
    require(audit["status"] == "passed" and all(audit["checks"].get(name, {}).get("passed") for name in REQUIRED_CHECKS), "TinyBERT fault audit did not pass")
    audit_config = TinyBertFaultCheckConfig(**read("config.json"))
    audit_config.validate()
    verify_source_artifacts(root, audit, "TinyBERT fault audit")
    verify_source_code(root, audit_dir, "TinyBERT fault audit")
    clustered_config, sources, clean_metrics, clustered_metrics, encoding, _ = load_clustered_run(root, audit_config.source_clustered_results)
    require(encoding_fingerprint(encoding) == audit["checks"]["source_integrity"]["encoding_fingerprint"], "Encoding differs from the audited storage")
    for name in ("report", "config", "status", "source_sha256"):
        sources.paths[f"audit_{name}"], sources.hashes[f"audit_{name}"] = audit_dir / f"{name}.json", sha256_file(audit_dir / f"{name}.json")
    return sources, clean_metrics, clustered_metrics, encoding


def verify_tinybert_sweep(root, config):
    """Re-read every saved pattern and record; recompute counts, decoding, aggregates, and CSV."""
    root = Path(root).resolve()
    results_dir = root / config.results_root / config.run_id
    read = lambda name: json.loads((results_dir / name).read_text())
    require(read("config.json") == asdict(config), "Saved configuration differs")
    plan = sweep_plan(config.step7_config())
    require(read("plan.json") == plan, "Saved plan differs from the configuration")
    records, manifest, references = read("measurements.json"), read("manifest.json"), read("references.json")
    verify_source_artifacts(root, manifest, "TinyBERT sweep")
    verify_source_code(root, results_dir, "TinyBERT sweep")
    _, _, _, encoding = prepare_sweep_sources(root, config)
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
        for split in splits:
            measurement = record["measurements"][split]
            require(measurement["accuracy"] == measurement["num_correct"] / measurement["num_examples"], f"Inconsistent accuracy: {record['pattern_id']} {split}")
            clustered = references[split]["clustered_no_faults"]
            require(record["fault_loss"][split] == {"accuracy_drop_percentage_points": 100 * (clustered["accuracy"] - measurement["accuracy"]),
                                                    "cross_entropy_increase": measurement["cross_entropy_loss"] - clustered["cross_entropy_loss"]}, f"Fault loss differs: {record['pattern_id']} {split}")
    zero = records[0]
    require(zero["counts"]["actual_corrupted_cells"] == 0 and all(zero["measurements"][split] == references[split]["clustered_no_faults"] for split in splits), "Zero pattern is not the clustered reference")
    require(read("aggregate.json") == json.loads(json.dumps(aggregate_tinybert(records, config, splits))), "Aggregate differs from a recomputation")
    with (results_dir / "measurements.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == csv_rows(records), "CSV and JSON measurements disagree")
    for plot in manifest["plots"]:
        require(sha256_file(root / plot["path"]) == plot["sha256"], f"Plot changed: {plot['path']}")
    return {"status": "passed", "patterns_verified": len(records), "pattern_hashes_counts_and_seed_regeneration_verified": True, "independent_decoding_verified": True,
            "per_pattern_records_match_measurements": True, "fault_losses_recomputed": True, "zero_pattern_equals_clustered_reference": True,
            "aggregate_recomputed": True, "csv_matches_json": True, "plot_hashes_verified": len(manifest["plots"]),
            "input_artifact_hashes_verified": len(manifest["source_artifacts"]), "source_code_unchanged": True}


def run_tinybert_fault_sweep(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    sources, clean_metrics, clustered_metrics, encoding = prepare_sweep_sources(root, config)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Sweep directories already exist; choose a new run_id")
    (results_dir / "patterns").mkdir(parents=True)
    (checkpoint_dir / "patterns").mkdir(parents=True)
    started = time.monotonic()
    plan = sweep_plan(config.step7_config())
    records = []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "plan.json", plan)
    write_json(results_dir / "status.json", {"status": "running", "completed_patterns": 0, "planned_patterns": len(plan), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_fault_sweep.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        environment = {**environment_record(device, config.cpu_threads, ("matplotlib",)), "fault_sampling_device": "cpu"}
        write_json(results_dir / "environment.json", environment)
        write_json(results_dir / "test_sets.json", sources.test_sets)
        fingerprint = encoding_fingerprint(encoding)
        baseline_state = decode_state(encoding)
        loaders = make_loaders(sources)
        splits = list(loaders)
        references = {}
        clean = sources.model.to(device)
        for split, metrics in evaluate_splits(clean, loaders, device).items():
            references[split] = {"clean": metrics, "clean_num_correct_difference_from_clean_eval": metrics["num_correct"] - clean_metrics["measurements"][split]["num_correct"]}
        require(all(item["clean_num_correct_difference_from_clean_eval"] == 0 for item in references.values()), "Clean replay differs from the two-test clean evaluation")
        model = build_tinybert(encoding["model_config"]).to(device)
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
            # The complete state comes from fresh decoding of the uncorrupted encoding for every pattern.
            model.load_state_dict(state, strict=True)
            measurements = evaluate_splits(model, loaders, device)
            if entry["probability"] == 0:
                for split in splits:
                    clustered = measurements[split]
                    references[split].update({
                        "clustered_no_faults": clustered,
                        "clustered_num_correct_difference_from_clustering_run": clustered["num_correct"] - clustered_metrics["measurements"][split]["clustered_no_faults"]["num_correct"],
                        "clustering_accuracy_drop_percentage_points": 100 * (references[split]["clean"]["accuracy"] - clustered["accuracy"]),
                    })
                require(all(item["clustered_num_correct_difference_from_clustering_run"] == 0 for item in references.values()), "Zero-fault replay differs from the clustering run")
                write_json(results_dir / "references.json", references)
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
        aggregate = aggregate_tinybert(records, config, splits)
        write_json(results_dir / "aggregate.json", aggregate)
        write_measurements_csv(results_dir / "measurements.csv", records)
        figures = plot_tinybert_sweep(records, aggregate, references, sources.test_names, results_dir)
        require(encoding_fingerprint(encoding) == fingerprint, "Source encoding changed during the sweep")
        for name, path in sources.paths.items():
            require(sha256_file(path) == sources.hashes[name], f"Source artifact changed: {name}")
        first_pattern = torch.load(root / records[1]["pattern"], map_location="cpu", weights_only=True)
        write_json(results_dir / "manifest.json", {
            "run_id": config.run_id, "source_artifacts": source_artifacts(root, sources.paths, sources.hashes), "encoding_fingerprint": fingerprint,
            "eligible_tensor_names": sorted(encoding["clustered_tensors"]), "eligible_cells": records[0]["counts"]["eligible_cells"],
            "splits": splits, "test_set_names": sources.test_names, "split_sha256": encoding["split_sha256"], "test_set_sha256": encoding["test_set_sha256"],
            "evaluation": {"batch_size": sources.config.evaluation_batch_size, "max_length": sources.config.max_length},
            "seed_rule": "Step 7 sweep_plan: nonzero rate ordinal * patterns_per_rate + repeat + base seed, modulo 2**32; zero is one deterministic reference.",
            "fault_model": {key: value for key, value in first_pattern.items() if key not in ("tensors", "tensor_order", "seed", "fault_probability")},
            "denominator": "nn.Linear weight cells only; embedding tables, LayerNorm, biases, codebooks, and mappings are reliable and never faulted",
            "selection_policy": "Rates and seed plan fixed before evaluation; no training, checkpoint selection, or tuning against any result.",
            "reporting": "Selection split and each test set reported separately; test sets never pooled; selection accuracy is not a held-out estimate for this checkpoint.",
            "comparisons": "Clustering loss: clean versus clustered zero faults. Fault loss: clustered zero faults versus each saved pattern; signed drops are allowed.",
            "reset_policy": "Every state reconstructed from the uncorrupted encoding; the full state is loaded before each evaluation; the saved pattern is static across all splits.",
            "plots": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in figures],
            "completed_patterns": len(records), "environment": environment, "elapsed_seconds": time.monotonic() - started,
        })
        write_json(results_dir / "verification.json", verify_tinybert_sweep(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "completed_patterns": len(records), "planned_patterns": len(plan), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "fault_sweep_complete", "run_id": config.run_id, "completed_patterns": len(records), "elapsed_seconds": time.monotonic() - started}), flush=True)
        return aggregate
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_patterns": len(records), "planned_patterns": len(plan), "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2_fault_sweep.json")
    args = parser.parse_args()
    run_tinybert_fault_sweep(TinyBertSweepConfig(**json.loads(args.config.read_text())), project_root)
