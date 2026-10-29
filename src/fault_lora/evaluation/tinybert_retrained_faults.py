"""Step 12b: audit and fault sweep on the codebook-retrained TinyBERT encodings.

The retrained encodings (Step 12a) keep the clustered indices and mappings, so a
pattern regenerated with the same seed and probability must have exactly the cells
of the saved sweep pattern; that identity is checked for every pattern before any
evaluation. Identical faults are then applied to the old clustered baseline and the
retrained one. Both test sets are reported separately; the selection split is
measured but, as before, is not a result.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace

import torch

from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import inspect_transitions, patterns_equal, require, require_state_equal
from fault_lora.evaluation.fault_sweep import summarize_values
from fault_lora.evaluation.retrained_faults import transition_differences
from fault_lora.evaluation.source_hashes import compare_source_hashes, unique_changes
from fault_lora.evaluation.tinybert_clean_eval import environment_record, source_artifacts
from fault_lora.evaluation.tinybert_sources import build_tinybert, evaluate_splits, make_loaders, synthetic_token_batch
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern
from fault_lora.training.baseline import choose_device
from fault_lora.training.codebook_finetune import codebook_order, verify_same_cells
from fault_lora.training.tinybert_recovery import load_sweep


SELECTION = "selection"
MODELS = ("old", "retrained")
CSV_FIELDS = ["group_id", "pattern_id", "probability", "seed", "actual_corrupted_cells", "realized_fault_rate", "model", "split", "num_examples", "num_correct", "accuracy",
              "cross_entropy_loss", "fault_damage_percentage_points"]


@dataclass
class TinyBertRetrainedSweepConfig:
    run_id: str = "tinybert-sst2-retrained-fault-sweep-seed1000"
    finetune_results: str = "results/tinybert-sst2-codebook-finetune-monotone-k16-seed42"
    groups: list = field(default_factory=lambda: [
        {"group_id": "linemb", "variant_id": "linemb-codebook", "condition": "linear-and-embeddings", "sweeps": ["results/tinybert-sst2-fault-sweep-linear-emb-seed1000"], "audit": True},
        {"group_id": "lin", "variant_id": "lin-codebook", "condition": "linear-only",
         "sweeps": ["results/tinybert-sst2-fault-sweep-seed42", "results/tinybert-sst2-fault-sweep-linear-high-seed72"], "audit": False}])
    probabilities: list = field(default_factory=lambda: [0, .01, .05, .1, .2, .5])
    patterns_per_rate: int = 5
    audit_probability: float = .01
    audit_seed: int = 42
    device: str = "mps"
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        keys = {"group_id", "variant_id", "condition", "sweeps", "audit"}
        if not isinstance(self.groups, list) or not self.groups or any(not isinstance(item, dict) or set(item) != keys or not item["sweeps"] for item in self.groups):
            raise ValueError(f"Each group declares exactly {sorted(keys)} with at least one sweep")
        if len({item["group_id"] for item in self.groups}) != len(self.groups) or any(not re.fullmatch(r"[a-z0-9-]+", item["group_id"]) for item in self.groups):
            raise ValueError("group_id values must be distinct simple names")
        if not self.probabilities or self.probabilities[0] != 0 or self.probabilities != sorted(set(self.probabilities)) or any(not 0 <= p <= 1 for p in self.probabilities):
            raise ValueError("probabilities must be sorted, unique, start at 0, and lie in [0, 1]")
        if isinstance(self.patterns_per_rate, bool) or not isinstance(self.patterns_per_rate, int) or self.patterns_per_rate < 2:
            raise ValueError("patterns_per_rate must be an integer >= 2")
        if not 0 < self.audit_probability < 1 or not isinstance(self.audit_seed, int) or not 0 <= self.audit_seed < 2**32 - 1:
            raise ValueError("Invalid audit probability or seed")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def load_group(root, config, group, finetune):
    """Verified old sweeps, the declared patterns at each rate, and the pinned retrained encoding."""
    sweeps, old_encoding, sources, references, selected = [], None, None, None, []
    for directory in group["sweeps"]:
        condition, plan, records, refs, srcs, _, encoding = load_sweep(root, SimpleNamespace(source_sweep_results=directory))
        require(condition == group["condition"], f"Sweep {directory} has condition {condition}, not {group['condition']}")
        if old_encoding is None:
            old_encoding, sources, references = encoding, srcs, refs
        require(encoding_fingerprint(encoding) == encoding_fingerprint(old_encoding), "A group's sweeps use different encodings")
        sweeps.append({"directory": directory, "records": {record["pattern_id"]: record for record in records}, "sources": srcs})
    for probability in config.probabilities:
        found = [(sweep["directory"], record) for sweep in sweeps for record in sweep["records"].values() if record["probability"] == probability]
        if probability == 0:
            found = found[:1]
        require(len(found) == (1 if probability == 0 else config.patterns_per_rate), f"{group['group_id']}: expected {config.patterns_per_rate} saved patterns at {probability}, found {len(found)}")
        selected += [{"directory": directory, **record} for directory, record in sorted(found, key=lambda item: item[1]["seed"])]
    item = finetune["encodings"][group["variant_id"]]
    path = root / item["encoding"]
    require(sha256_file(path) == item["encoding_sha256"] and item["condition"] == group["condition"], f"Retrained encoding changed or has another condition: {group['variant_id']}")
    retrained = torch.load(path, map_location="cpu", weights_only=True)
    require(encoding_fingerprint(retrained) == item["fingerprint"], f"Retrained fingerprint differs: {group['variant_id']}")
    verify_same_cells(retrained, old_encoding)
    require(all(report["strictly_increasing"] for report in codebook_order({n: r["codebook"] for n, r in retrained["clustered_tensors"].items()}).values()), "Retrained codebooks are not strictly increasing")
    dense = torch.load(root / item["dense"], map_location="cpu", weights_only=True)
    require(sha256_file(root / item["dense"]) == item["dense_sha256"], "Retrained dense reconstruction changed")
    require_state_equal(decode_state(retrained), dense["model_state_dict"], f"Retrained reconstruction ({group['variant_id']})")
    paths = {f"{group['group_id']}:retrained_encoding": path, f"{group['group_id']}:retrained_dense": root / item["dense"]}
    hashes = {f"{group['group_id']}:retrained_encoding": item["encoding_sha256"], f"{group['group_id']}:retrained_dense": item["dense_sha256"]}
    for sweep in sweeps:
        paths.update({f"{group['group_id']}:{name}": value for name, value in sweep["sources"].paths.items()})
        hashes.update({f"{group['group_id']}:{name}": value for name, value in sweep["sources"].hashes.items()})
    return {"old": old_encoding, "retrained": retrained, "sources": sources, "references": references, "patterns": selected, "paths": paths, "hashes": hashes,
            "accepted": list(sources.accepted_source_hash_changes)}


def audit(root, config, group, data, checkpoint_dir):
    """The eight Step 6 checks on a retrained encoding."""
    encoding, old = data["retrained"], data["old"]
    checks = {}
    baseline = decode_state(encoding)
    checks["source_integrity"] = {"passed": True, "encoding_fingerprint": encoding_fingerprint(encoding), "cells_identical_to_old": len(encoding["clustered_tensors"]),
                                  "codebooks_strictly_increasing": True, "dense_reconstruction_exact": True}
    zero = generate_fault_pattern(encoding, 0, seed=config.audit_seed)
    zero_state = apply_fault_pattern(encoding, zero)
    require_state_equal(zero_state, baseline, "Zero-fault identity")
    model_config = data["sources"].checkpoint["model_config"]
    probe = synthetic_token_batch(model_config, seed=0)
    with torch.no_grad():
        same = torch.equal(build_tinybert(model_config, baseline)(**probe).logits, build_tinybert(model_config, zero_state)(**probe).logits)
    require(same, "Zero-fault logits differ")
    checks["zero_fault_identity"] = {"passed": True, **inspect_transitions(encoding, zero, zero_state), "logits_exactly_equal": True}
    path = checkpoint_dir / f"audit-{group['group_id']}-p{config.audit_probability}-seed{config.audit_seed}.pt"
    save_torch(path, generate_fault_pattern(encoding, config.audit_probability, seed=config.audit_seed))
    pattern = torch.load(path, map_location="cpu", weights_only=True)
    differences = transition_differences(pattern, generate_fault_pattern(old, config.audit_probability, seed=config.audit_seed))
    require(not differences, f"Audit pattern cells differ from the old encoding's: {differences[:5]}")
    state = apply_fault_pattern(encoding, pattern)
    details = inspect_transitions(encoding, pattern, state)
    summary = summarize_fault_pattern(encoding, pattern)
    for key in ("eligible_cells", "actual_corrupted_cells", "decoded_weight_changes", "lower_boundary_transitions", "upper_boundary_transitions"):
        require(details[key] == summary[key], f"Independent count differs: {key}")
    require(summary["decoded_weight_changes"] == summary["actual_corrupted_cells"], "A transition left a weight unchanged despite strictly increasing codebooks")
    checks["saved_pattern_and_checkpoint"] = {"passed": True, **{key: details[key] for key in ("eligible_cells", "actual_corrupted_cells", "decoded_weight_changes")},
                                              "pattern": str(path.relative_to(root)), "pattern_sha256": sha256_file(path), "identical_cells_to_old_encoding_pattern": True}
    require_state_equal(apply_fault_pattern(encoding, pattern), state, "Repeated application")
    require_state_equal(decode_state(encoding), baseline, "Baseline after replay")
    checks["replay_without_accumulation"] = {"passed": True}
    require(patterns_equal(generate_fault_pattern(encoding, config.audit_probability, seed=config.audit_seed), pattern), "Same seed did not reproduce the pattern")
    checks["same_seed_reproducibility"] = {"passed": True, "seed": config.audit_seed}
    alternate = generate_fault_pattern(encoding, config.audit_probability, seed=config.audit_seed + 1)
    require(not patterns_equal(alternate, pattern), "Different seed produced the same pattern")
    checks["different_seed_valid_pattern"] = {"passed": True, "seed": config.audit_seed + 1, **{k: v for k, v in inspect_transitions(encoding, alternate, apply_fault_pattern(encoding, alternate)).items() if k != "layers"}}
    full = generate_fault_pattern(encoding, 1, seed=config.audit_seed)
    full_details = inspect_transitions(encoding, full, apply_fault_pattern(encoding, full))
    require(full_details["actual_corrupted_cells"] == full_details["eligible_cells"] and full_details["lower_boundary_transitions"] > 0 and full_details["upper_boundary_transitions"] > 0,
            "All-cell selection or boundaries failed")
    checks["all_cell_selection_and_boundaries"] = {"passed": True, **{k: v for k, v in full_details.items() if k != "layers"}}
    require_state_equal(decode_state(encoding), baseline, "Final baseline preservation")
    require(all(sha256_file(p) == data["hashes"][name] for name, p in data["paths"].items()), "A source artifact changed during the audit")
    checks["source_preservation"] = {"passed": True}
    return checks


def aggregate(records, probabilities):
    result = {}
    for group in sorted({record["group_id"] for record in records}):
        rows = [record for record in records if record["group_id"] == group]
        zero = next(record for record in rows if record["probability"] == 0)["models"]
        result[group] = {}
        for model in MODELS:
            entries = []
            for probability in probabilities:
                chosen = [record for record in rows if record["probability"] == probability]
                splits = list(chosen[0]["models"][model])
                entries.append({"probability": probability, "num_patterns": len(chosen), "seeds": [record["seed"] for record in chosen],
                                "realized_fault_rate": summarize_values([record["counts"]["realized_fault_rate"] for record in chosen]),
                                "splits": {split: {"accuracy": summarize_values([record["models"][model][split]["accuracy"] for record in chosen]),
                                                   "fault_damage_percentage_points": summarize_values([100 * (zero[model][split]["accuracy"] - record["models"][model][split]["accuracy"]) for record in chosen])}
                                           for split in splits}})
            result[group][model] = entries
    return result


def csv_rows(records):
    rows = []
    for record in records:
        for model, measurements in record["models"].items():
            for split, values in measurements.items():
                row = {"group_id": record["group_id"], "pattern_id": record["pattern_id"], "probability": record["probability"], "seed": record["seed"],
                       "actual_corrupted_cells": record["counts"]["actual_corrupted_cells"], "realized_fault_rate": record["counts"]["realized_fault_rate"], "model": model, "split": split,
                       **{key: values[key] for key in ("num_examples", "num_correct", "accuracy", "cross_entropy_loss")},
                       "fault_damage_percentage_points": record["fault_damage_percentage_points"][model][split]}
                rows.append({key: str(row[key]) for key in CSV_FIELDS})
    return rows


def plot(aggregates, clean, test_names, output_dir):
    colors = {"old": "#eb6834", "retrained": "#2a78d6"}
    labels = {"linemb": "Linear + embeddings faulted", "lin": "Linear only faulted (ablation)"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    with tempfile.TemporaryDirectory(prefix="fault-lora-12b-plot-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            groups = list(aggregates)
            fig, axes = plt.subplots(len(groups), len(test_names), figsize=(12.5, 4.6 * len(groups)), squeeze=False, facecolor="#fcfcfb")
            for row, group in enumerate(groups):
                for column, split in enumerate(test_names):
                    axis = axes[row][column]
                    axis.set_facecolor("#fcfcfb")
                    axis.axhline(100 * clean[split]["accuracy"], color=muted, linestyle="--", linewidth=1.1, label=f"Clean: {100 * clean[split]['accuracy']:.2f}%")
                    for model, name in (("old", "Clustered baseline (Steps 10b/10c)"), ("retrained", "Codebook-retrained baseline (Step 12a)")):
                        entries = aggregates[group][model]
                        xs = [100 * entry["probability"] for entry in entries]
                        ys = [100 * entry["splits"][split]["accuracy"]["mean"] for entry in entries]
                        sd = [100 * (entry["splits"][split]["accuracy"]["sample_std"] or 0) for entry in entries]
                        axis.errorbar(xs, ys, yerr=sd, color=colors[model], marker="o" if model == "old" else "D", markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.4,
                                      linewidth=2, capsize=3, label=f"{name}; 0%: {ys[0]:.2f}%")
                    axis.set_title(f"{labels.get(group, group)} — {split}", fontsize=10.5, color=ink)
                    axis.set_xlabel("Requested fault probability (%)", color=muted)
                    axis.set_ylabel("Accuracy (%)", color=muted)
                    axis.set_xticks([100 * entry["probability"] for entry in aggregates[group]["old"]])
                    axis.grid(color=grid, linewidth=.8)
                    axis.set_axisbelow(True)
                    axis.tick_params(colors=muted)
                    for side in ("top", "right"):
                        axis.spines[side].set_visible(False)
                    axis.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=ink)
            fig.suptitle("TinyBERT / SST-2: identical faults on the clustered and codebook-retrained encodings", fontsize=12.5, color=ink)
            fig.text(.5, .01, "Mean ± 1 sample SD across 5 static patterns per nonzero rate; identical cells faulted in both encodings. Test sets reported separately. No LoRA.",
                     ha="center", fontsize=8, color=muted)
            fig.tight_layout(rect=(0, .03, 1, .96))
            paths = [Path(output_dir) / f"retrained_fault_sweep.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
            plt.close(fig)
            return paths
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def run_tinybert_retrained_sweep(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Sweep directories already exist; choose a new run_id")
    finetune_dir = root / config.finetune_results
    require(json.loads((finetune_dir / "status.json").read_text())["status"] == "complete" and json.loads((finetune_dir / "verification.json").read_text())["status"] == "passed",
            "Step 12a run is incomplete or unverified")
    finetune = json.loads((finetune_dir / "manifest.json").read_text())
    step12a = json.loads((finetune_dir / "paired_evaluation.json").read_text())["models"]
    data = {group["group_id"]: load_group(root, config, group, finetune) for group in config.groups}
    (checkpoint_dir / "patterns").mkdir(parents=True)
    (results_dir / "patterns").mkdir(parents=True)
    started, records = time.monotonic(), []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "stage": "audit", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_retrained_fault_sweep.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        environment = environment_record(device, 4, ("matplotlib",))
        write_json(results_dir / "environment.json", environment)
        audits = {group["group_id"]: audit(root, config, group, data[group["group_id"]], checkpoint_dir) for group in config.groups if group["audit"]}
        write_json(results_dir / "audit.json", {"status": "passed", "checks": audits})
        print(json.dumps({"event": "retrained_audit_passed", "groups": list(audits), "checks": {key: len(value) for key, value in audits.items()}}), flush=True)
        # Stage 1: identical cells for every declared pattern, before any evaluation.
        identity = []
        for group in config.groups:
            item = data[group["group_id"]]
            for saved in item["patterns"]:
                old_pattern = torch.load(root / saved["pattern"], map_location="cpu", weights_only=True)
                require(sha256_file(root / saved["pattern"]) == saved["pattern_sha256"], f"Saved pattern changed: {saved['pattern_id']}")
                new = generate_fault_pattern(item["retrained"], saved["probability"], seed=saved["seed"])
                differences = transition_differences(new, old_pattern)
                path = checkpoint_dir / "patterns" / f"{group['group_id']}-{saved['pattern_id']}.pt"
                save_torch(path, new)
                identity.append({"group_id": group["group_id"], "pattern_id": saved["pattern_id"], "probability": saved["probability"], "seed": saved["seed"], "source_sweep": saved["directory"],
                                 "saved_pattern": saved["pattern"], "retrained_pattern": str(path.relative_to(root)), "retrained_pattern_sha256": sha256_file(path),
                                 "identical": not differences, "differences": differences})
                print(json.dumps({"event": "pattern_identity", "group": group["group_id"], "pattern_id": saved["pattern_id"], "identical": not differences}), flush=True)
        write_json(results_dir / "pattern_identity.json", identity)
        mismatched = [f"{item['group_id']}:{item['pattern_id']}" for item in identity if not item["identical"]]
        if mismatched:
            write_json(results_dir / "status.json", {"status": "stopped_pattern_mismatch", "mismatched_patterns": mismatched})
            return {"status": "stopped_pattern_mismatch", "mismatched_patterns": mismatched}
        # Stage 2: identical faults on the old and retrained encodings.
        any_sources = next(iter(data.values()))["sources"]
        loaders = make_loaders(any_sources)
        model_config = any_sources.checkpoint["model_config"]
        model = build_tinybert(model_config).to(device)

        def measure(state):
            model.load_state_dict(state, strict=True)
            return evaluate_splits(model.eval(), loaders, device)

        clean = measure({name: value.detach().clone() for name, value in any_sources.model.state_dict().items()})
        for group in config.groups:
            item, zero = data[group["group_id"]], None
            for saved, ident in zip(item["patterns"], [entry for entry in identity if entry["group_id"] == group["group_id"]]):
                old_pattern = torch.load(root / saved["pattern"], map_location="cpu", weights_only=True)
                new_pattern = torch.load(root / ident["retrained_pattern"], map_location="cpu", weights_only=True)
                measurements = {"old": measure(apply_fault_pattern(item["old"], old_pattern)), "retrained": measure(apply_fault_pattern(item["retrained"], new_pattern))}
                require(all(measurements["old"][split]["num_correct"] == saved["measurements"][split]["num_correct"] for split in measurements["old"]),
                        f"Old-baseline replay differs from its sweep record: {saved['pattern_id']}")
                if saved["probability"] == 0:
                    zero = measurements
                    require(all(zero["retrained"][split]["num_correct"] == step12a[group["variant_id"]][split]["num_correct"] for split in zero["retrained"]),
                            f"Retrained zero-fault result differs from Step 12a: {group['variant_id']}")
                counts = {key: value for key, value in summarize_fault_pattern(item["retrained"], new_pattern).items() if key != "layers"}
                record = {"group_id": group["group_id"], "pattern_id": saved["pattern_id"], "probability": saved["probability"], "seed": saved["seed"], "counts": counts,
                          "saved_pattern": saved["pattern"], "retrained_pattern": ident["retrained_pattern"], "models": measurements,
                          "fault_damage_percentage_points": {name: {split: 100 * (zero[name][split]["accuracy"] - value[split]["accuracy"]) for split in value} for name, value in measurements.items()}}
                records.append(record)
                write_json(results_dir / "patterns" / f"{group['group_id']}-{saved['pattern_id']}.json", record)
                write_json(results_dir / "measurements.json", records)
                write_json(results_dir / "status.json", {"status": "running", "stage": "evaluation", "completed_patterns": len(records), "planned_patterns": len(identity)})
                print(json.dumps({"event": "pattern_evaluation_complete", "group": group["group_id"], "pattern_id": saved["pattern_id"],
                                  **{f"{name}_{split}": round(value[split]["accuracy"], 4) for name, value in measurements.items() for split in any_sources.test_names}}), flush=True)
        aggregates = aggregate(records, config.probabilities)
        write_json(results_dir / "aggregate.json", {"groups": aggregates, "clean": clean, "test_names": any_sources.test_names,
                                                     "definitions": {"fault_damage": "That encoding's own zero-fault accuracy minus its accuracy under the pattern, in percentage points; mean and sample SD across patterns.",
                                                                     "selection": "The selection split was part of the checkpoint's fine-tuning data; it is measured but is not a result."}})
        with (results_dir / "measurements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows(records))
        figures = plot(aggregates, clean, any_sources.test_names, results_dir)
        paths = {name: p for item in data.values() for name, p in item["paths"].items()}
        hashes = {name: h for item in data.values() for name, h in item["hashes"].items()}
        require(all(sha256_file(p) == hashes[name] for name, p in paths.items()), "An input artifact changed during the sweep")
        accepted = [change for item in data.values() for change in item["accepted"]] + compare_source_hashes(root, source_code, "Source code changed during the sweep")
        write_json(results_dir / "manifest.json", {"run_id": config.run_id, "source_artifacts": source_artifacts(root, paths, hashes),
                                                   "fingerprints": {group: {"old": encoding_fingerprint(item["old"]), "retrained": encoding_fingerprint(item["retrained"])} for group, item in data.items()},
                                                   "plots": [{"path": str(p.relative_to(root)), "sha256": sha256_file(p)} for p in figures], "accepted_source_hash_changes": unique_changes(accepted),
                                                   "environment": environment, "elapsed_seconds": time.monotonic() - started})
        write_json(results_dir / "verification.json", verify_tinybert_retrained_sweep(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "completed_patterns": len(records), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "tinybert_retrained_sweep_complete", "elapsed_seconds": time.monotonic() - started}), flush=True)
        return {"status": "complete", "aggregate": aggregates, "clean": clean}
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_patterns": len(records), "error": repr(error)})
        raise


def verify_tinybert_retrained_sweep(root, config):
    root = Path(root).resolve()
    results_dir = root / config.results_root / config.run_id
    identity = json.loads((results_dir / "pattern_identity.json").read_text())
    records = json.loads((results_dir / "measurements.json").read_text())
    require([(item["group_id"], item["pattern_id"]) for item in identity] == [(record["group_id"], record["pattern_id"]) for record in records], "Records are incomplete or reordered")
    require(all(item["identical"] and not item["differences"] for item in identity), "A pattern differed")
    for item, record in zip(identity, records):
        require(sha256_file(root / item["retrained_pattern"]) == item["retrained_pattern_sha256"], f"Retrained pattern changed: {item['pattern_id']}")
        require(record == json.loads((results_dir / "patterns" / f"{record['group_id']}-{record['pattern_id']}.json").read_text()), f"Records disagree: {record['pattern_id']}")
    saved = json.loads((results_dir / "aggregate.json").read_text())
    require(saved["groups"] == json.loads(json.dumps(aggregate(records, config.probabilities))), "Aggregate differs from its recomputation")
    with (results_dir / "measurements.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == csv_rows(records), "CSV and JSON disagree")
    manifest = json.loads((results_dir / "manifest.json").read_text())
    for name, item in manifest["source_artifacts"].items():
        require(sha256_file(root / item["path"]) == item["sha256"], f"Input artifact changed: {name}")
    accepted = compare_source_hashes(root, json.loads((results_dir / "source_sha256.json").read_text()), "Source code changed after the sweep")
    return {"status": "passed", "patterns_verified": len(records), "all_patterns_identical": True, "audit_passed": json.loads((results_dir / "audit.json").read_text())["status"] == "passed",
            "aggregate_recomputed": True, "csv_matches_json": True, "source_artifacts_verified": len(manifest["source_artifacts"]), "accepted_source_hash_changes": unique_changes(accepted)}


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tinybert_sst2_retrained_fault_sweep.json"))
    args = parser.parse_args()
    run_tinybert_retrained_sweep(TinyBertRetrainedSweepConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
