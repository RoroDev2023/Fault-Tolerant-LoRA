"""Step 13b: level-dependent faults on the retrained ResNet-18 baseline, two cluster-to-level mappings.

Evaluation only. For each declared sigma_over_window x condition (mapping, level-0
protection) x pattern seed: generate and audit a static level-dependent pattern,
then measure accuracy with no repair and after BatchNorm recalibration (the
Step 9b method, unchanged). No LoRA, no training of weights.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from fault_lora.data.cifar10 import cifar10_transforms
from fault_lora.evaluation.bn_recalibration import InputsOnly, batchnorm_buffer_names, calibration_indices, recalibrate_batchnorm
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import inspect_transitions, patterns_equal, require, require_state_equal
from fault_lora.evaluation.fault_sweep import summarize_values
from fault_lora.evaluation.source_hashes import compare_source_hashes, unique_changes
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import encoding_fingerprint, validate_parameters
from fault_lora.memory.level_faults import (KIND, MAPPINGS, apply_level_fault_pattern, gaussian_level_probabilities, generate_level_fault_pattern, mapping_report,
                                            per_level_z_scores, remap_encoding, summarize_level_fault_pattern)
from fault_lora.models.resnet import build_resnet18
from fault_lora.training.baseline import choose_device
from fault_lora.training.recovery import RecoveryConfig
from fault_lora.training.retrained_bases import prepare_retrained_sources


SPLITS = ("validation", "test")
CSV_FIELDS = ["record_id", "sigma_over_window", "condition_id", "mapping", "level0_safe", "seed", "requested_average_fault_probability", "realized_fault_rate",
              "actual_corrupted_cells", "decoded_weight_changes", "split", "accuracy_no_repair", "accuracy_bn_recalibrated", "bn_gain_percentage_points"]


@dataclass
class LevelFaultSweepConfig:
    run_id: str = "resnet18-cifar10-level-faults-seed1300"
    source_sweep_results: str = "results/resnet18-cifar10-retrained-fault-sweep-seed42"
    source_audit_results: str = "results/resnet18-cifar10-retrained-fault-checks-seed42"
    step11c_bn_results: str = "results/resnet18-cifar10-retrained-bn-recalibration-seed42"
    clean_checkpoint_sha256: str | None = None
    sigmas: list = field(default_factory=lambda: [0.013, 0.020, 0.026])
    conditions: list = field(default_factory=lambda: [{"condition_id": "sorted", "mapping": "sorted", "level0_safe": False},
                                                       {"condition_id": "sorted-l0safe", "mapping": "sorted", "level0_safe": True},
                                                       {"condition_id": "frequent-l0safe", "mapping": "frequent-at-0", "level0_safe": True}])
    seeds: list = field(default_factory=lambda: [1300, 1301, 1302, 1303, 1304])
    z_threshold: float = 5.0
    calibration_examples: int = 5000
    calibration_batch_size: int = 250
    calibration_seed: int = 42
    evaluation_batch_size: int = 256
    device: str = "mps"
    cpu_threads: int = 4
    num_workers: int = 2
    data_root: str = "data"
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if not self.sigmas or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not 0 < s < 1 for s in self.sigmas) or self.sigmas != sorted(set(self.sigmas)):
            raise ValueError("sigmas must be increasing, distinct, and in (0, 1)")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        for seed in self.seeds + [self.calibration_seed]:
            validate_parameters(0, seed)
        identifiers = [item.get("condition_id") for item in self.conditions]
        if not self.conditions or len(set(identifiers)) != len(identifiers) or any(
                set(item) != {"condition_id", "mapping", "level0_safe"} or item["mapping"] not in MAPPINGS or not isinstance(item["level0_safe"], bool)
                or not re.fullmatch(r"[a-z0-9-]+", item["condition_id"]) for item in self.conditions):
            raise ValueError("conditions must be distinct {condition_id, mapping, level0_safe} entries")
        if not self.z_threshold > 0:
            raise ValueError("z_threshold must be positive")
        if self.calibration_examples % self.calibration_batch_size:
            raise ValueError("calibration_examples must be a multiple of calibration_batch_size")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def sigma_tag(sigma):
    return "sw" + f"{sigma:g}".replace(".", "p")


def record_id(sigma, condition, seed):
    return f"{sigma_tag(sigma)}-{condition['condition_id']}-seed{seed}"


def plan(config):
    """Declared order: sigma, then seed, then condition, so conditions on one seed are adjacent."""
    return [{"record_id": record_id(sigma, condition, seed), "sigma_over_window": sigma, **condition, "seed": seed}
            for sigma in config.sigmas for seed in config.seeds for condition in config.conditions]


def audit_pattern(encoding, other_encoding, pattern, saved, state, config):
    """Step 6-style checks for the level-dependent kind; raises on any failure."""
    checks = {}
    regenerated = generate_level_fault_pattern(encoding, {"down": pattern["level_down_probabilities"], "up": pattern["level_up_probabilities"],
                                                          **pattern["probability_model"]}, seed=pattern["seed"])
    require(patterns_equal(pattern, regenerated), "Seeded regeneration differs")
    checks["seeded_regeneration_identical"] = True
    require(patterns_equal(pattern, saved) and {k: v for k, v in pattern.items() if k != "tensors"} == {k: v for k, v in saved.items() if k != "tensors"},
            "Saved pattern differs from the generated one")
    checks["saved_round_trip_identical"] = True
    summary = summarize_level_fault_pattern(encoding, pattern)
    oracle = inspect_transitions(encoding, pattern, state)
    require(oracle["actual_corrupted_cells"] == summary["actual_corrupted_cells"] and oracle["decoded_weight_changes"] == summary["decoded_weight_changes"],
            "Independent decoding counts differ")
    checks["independent_decoding_oracle"] = {"passed": True, "only_plus_minus_one_transitions": True, "boundaries_move_inward": True,
                                             "unselected_and_reliable_state_unchanged": True, "state_tensors_checked": oracle["all_state_tensors_checked"]}
    level0 = sum(int((entry["before_levels"] == 0).sum().item()) for entry in pattern["tensors"].values())
    if pattern["level_down_probabilities"][0] == 0 and pattern["level_up_probabilities"][0] == 0:
        require(level0 == 0, "A protected level-0 cell faulted")
        checks["protected_level0_never_faults"] = {"passed": True, "level0_cells": summary["per_level"]["cells"][0]}
    else:
        checks["level0_transitions"] = level0
    rows = per_level_z_scores(summary)
    require(not any(row["impossible_event_occurred"] for row in rows), "A zero-probability transition occurred")
    largest = max((abs(row["z"]) for row in rows if row["z"] is not None), default=0.0)
    require(largest <= config.z_threshold, f"Per-level realized rate outside sampling error (|z| = {largest:.2f})")
    total_variance = sum(c * (d + u) * (1 - d - u) for c, d, u in zip(summary["per_level"]["cells"], summary["per_level"]["requested_down"], summary["per_level"]["requested_up"]))
    total_z = (summary["actual_corrupted_cells"] - summary["expected_corrupted_cells"]) / math.sqrt(total_variance)
    require(abs(total_z) <= config.z_threshold, "Total realized count outside sampling error")
    checks["per_level_rates_within_sampling_error"] = {"passed": True, "z_threshold": config.z_threshold, "largest_abs_z": largest, "total_z": total_z, "per_level": rows}
    try:
        summarize_level_fault_pattern(other_encoding, pattern)
        rejected = False
    except ValueError:
        rejected = True
    require(rejected, "Pattern was accepted by the other mapping's encoding")
    checks["bound_to_its_mapping"] = True
    return summary, checks


def summary_counts(summary):
    return {key: value for key, value in summary.items() if key not in ("layers",)}


def aggregate(records, reference, config):
    """Per sigma x condition: mean and sample SD over seeds; paired differences against the sorted condition."""
    groups = []
    baseline = config.conditions[0]["condition_id"]
    for sigma in config.sigmas:
        for condition in config.conditions:
            group = [r for r in records if r["sigma_over_window"] == sigma and r["condition_id"] == condition["condition_id"]]
            require([r["seed"] for r in group] == config.seeds, "Aggregate group is incomplete")
            entry = {"sigma_over_window": sigma, **condition, "num_patterns": len(group),
                     "realized_fault_rate": summarize_values([r["counts"]["realized_fault_rate"] for r in group]),
                     "requested_average_fault_probability": summarize_values([r["counts"]["requested_average_fault_probability"] for r in group]),
                     "splits": {}}
            for split in SPLITS:
                no_repair = [r["no_repair"][split]["accuracy"] for r in group]
                bn = [r["bn_recalibrated"][split]["accuracy"] for r in group]
                entry["splits"][split] = {
                    "accuracy_no_repair": summarize_values(no_repair), "accuracy_bn_recalibrated": summarize_values(bn),
                    "damage_no_repair_percentage_points": summarize_values([100 * (reference["no_repair"][split]["accuracy"] - a) for a in no_repair]),
                    "damage_bn_recalibrated_percentage_points": summarize_values([100 * (reference["bn_recalibrated"][split]["accuracy"] - a) for a in bn]),
                    "bn_gain_percentage_points": summarize_values([100 * (b - a) for a, b in zip(no_repair, bn)])}
                if condition["condition_id"] != baseline:
                    paired = {r["seed"]: r for r in records if r["sigma_over_window"] == sigma and r["condition_id"] == baseline}
                    entry["splits"][split]["minus_" + baseline] = {
                        mode: summarize_values([100 * (r[key][split]["accuracy"] - paired[r["seed"]][key][split]["accuracy"]) for r in group])
                        for mode, key in (("no_repair_percentage_points", "no_repair"), ("bn_recalibrated_percentage_points", "bn_recalibrated"))}
            groups.append(entry)
    return {"groups": groups, "reference": {k: reference[k] for k in ("no_repair", "bn_recalibrated")},
            "variability": "Sample standard deviation across the declared pattern seeds (ddof=1). Not a confidence interval.",
            "pairing": f"Differences against '{baseline}' pair the same sigma and seed; with one uniform draw per cell position, '{baseline}' and its level-0-safe variant differ only on level-0 cells."}


def csv_rows(records):
    rows = []
    for r in records:
        for split in SPLITS:
            rows.append({name: "" if value is None else str(value) for name, value in {
                "record_id": r["record_id"], "sigma_over_window": r["sigma_over_window"], "condition_id": r["condition_id"], "mapping": r["mapping"],
                "level0_safe": r["level0_safe"], "seed": r["seed"], "requested_average_fault_probability": r["counts"]["requested_average_fault_probability"],
                "realized_fault_rate": r["counts"]["realized_fault_rate"], "actual_corrupted_cells": r["counts"]["actual_corrupted_cells"],
                "decoded_weight_changes": r["counts"]["decoded_weight_changes"], "split": split,
                "accuracy_no_repair": r["no_repair"][split]["accuracy"], "accuracy_bn_recalibrated": r["bn_recalibrated"][split]["accuracy"],
                "bn_gain_percentage_points": 100 * (r["bn_recalibrated"][split]["accuracy"] - r["no_repair"][split]["accuracy"])}.items()})
    return rows


COLORS = {"sorted": ("#2a78d6", "o", "-"), "sorted-l0safe": ("#eb6834", "s", "--"), "frequent-l0safe": ("#1baf7a", "D", "-.")}
LABELS = {"sorted": "Sorted mapping", "sorted-l0safe": "Sorted + level 0 protected", "frequent-l0safe": "Most frequent cluster on protected level 0"}


def plot(summary, output_dir, split="test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, reference = summary["groups"], summary["reference"]
    sigmas = sorted({g["sigma_over_window"] for g in groups})
    conditions = list(dict.fromkeys(g["condition_id"] for g in groups))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    offsets = {c: (i - (len(conditions) - 1) / 2) * 0.0004 for i, c in enumerate(conditions)}
    panels = [("realized_fault_rate", "Realized cell fault rate (%)", None), ("accuracy_no_repair", f"Test accuracy, no repair (%)", "no_repair"),
              ("accuracy_bn_recalibrated", "Test accuracy after BN recalibration (%)", "bn_recalibrated")]
    for axis, (key, title, ref_key) in zip(axes, panels):
        for condition in conditions:
            color, marker, style = COLORS.get(condition, ("#555555", "o", "-"))
            rows = [next(g for g in groups if g["sigma_over_window"] == s and g["condition_id"] == condition) for s in sigmas]
            stats = [row[key] if key == "realized_fault_rate" else row["splits"][split][key] for row in rows]
            axis.errorbar([s + offsets[condition] for s in sigmas], [100 * x["mean"] for x in stats], yerr=[100 * (x["sample_std"] or 0) for x in stats],
                          color=color, marker=marker, linestyle=style, linewidth=2, markersize=8, capsize=3, label=LABELS.get(condition, condition))
        if ref_key:
            axis.axhline(100 * reference[ref_key][split]["accuracy"], color="#555555", linestyle=":", linewidth=1.5,
                         label=f"Fault-free {'BN-recalibrated ' if ref_key == 'bn_recalibrated' else ''}reference: {100 * reference[ref_key][split]['accuracy']:.2f}%")
        else:
            axis.set_yscale("log")
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("Read noise sigma / read window")
        axis.set_xticks(sigmas)
        axis.grid(color="#dddddd", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8.5, frameon=False, loc="lower left" if ref_key else "upper left")
    figure.suptitle("ResNet-18 / CIFAR-10, retrained baseline: level-dependent faults (16-level cells), sorted vs most-frequent-cluster-on-level-0 mapping", fontsize=13)
    figure.text(0.5, 0.005, "Means ± 1 sample SD over 5 pattern seeds; conditions offset slightly on x. Test split (10,000 images).", ha="center", fontsize=9, color="#444444")
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"level_faults.{extension}", dpi=150)
    plt.close(figure)


def verify(root, config):
    root = Path(root)
    results_dir = root / config.results_root / config.run_id
    records = json.loads((results_dir / "measurements.json").read_text())
    require([r["record_id"] for r in records] == [entry["record_id"] for entry in plan(config)], "Measurements incomplete or reordered")
    manifest = json.loads((results_dir / "manifest.json").read_text())
    encodings = {}
    for mapping, item in manifest["encodings"].items():
        require(sha256_file(root / item["path"]) == item["sha256"], f"Saved {mapping} encoding changed")
        encodings[mapping] = torch.load(root / item["path"], map_location="cpu", weights_only=False)
        require(encoding_fingerprint(encodings[mapping]) == item["fingerprint"], f"{mapping} fingerprint differs")
    require_state_equal(decode_state(encodings["frequent-at-0"]), decode_state(encodings["sorted"]), "Fault-free decoding under the two mappings")
    for record in records:
        require(record == json.loads((results_dir / "conditions" / f"{record['record_id']}.json").read_text()), f"Individual and combined records disagree: {record['record_id']}")
        path = root / record["pattern"]
        require(sha256_file(path) == record["pattern_sha256"], f"Pattern file changed: {record['record_id']}")
        pattern = torch.load(path, map_location="cpu", weights_only=True)
        require(pattern["seed"] == record["seed"] and pattern["probability_model"]["sigma_over_window"] == record["sigma_over_window"]
                and pattern["probability_model"]["level0_safe"] == record["level0_safe"], f"Pattern settings differ: {record['record_id']}")
        require(summary_counts(summarize_level_fault_pattern(encodings[record["mapping"]], pattern)) == record["counts"], f"Pattern counts differ: {record['record_id']}")
        require(json.loads((results_dir / "audit" / f"{record['record_id']}.json").read_text())["passed"], f"Audit not passed: {record['record_id']}")
        require(record["non_batchnorm_state_preserved"], f"BN recalibration changed other state: {record['record_id']}")
    reference = json.loads((results_dir / "reference.json").read_text())
    require(json.loads((results_dir / "aggregate.json").read_text()) == json.loads(json.dumps(aggregate(records, reference, config))), "Aggregate differs from records")
    with (results_dir / "measurements.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == csv_rows(records), "CSV and JSON measurements disagree")
    for name, item in manifest["source_artifacts"].items():
        require(sha256_file(root / item["path"]) == item["sha256"], f"Input artifact changed: {name}")
    accepted = compare_source_hashes(root, json.loads((results_dir / "source_sha256.json").read_text()), "Source code changed during the run")
    return {"status": "passed", "records_verified": len(records), "patterns_rehashed_and_resummarized": len(records), "audits_passed": len(records),
            "fault_free_decoding_identical_under_both_mappings": True, "aggregate_recomputed": True, "csv_matches_json": True,
            "source_artifacts_verified": len(manifest["source_artifacts"]), "source_code_unchanged": not accepted, "accepted_source_hash_changes": unique_changes(accepted)}


def run_level_fault_sweep(config, project_root):
    config.validate()
    root = Path(project_root)
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run directories already exist; choose a new run_id")
    for directory in (results_dir / "conditions", results_dir / "audit", checkpoint_dir / "patterns", checkpoint_dir / "encodings", checkpoint_dir / "batchnorm"):
        directory.mkdir(parents=True)
    started = time.monotonic()
    entries, records = plan(config), []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "plan.json", entries)
    write_json(results_dir / "status.json", {"status": "running", "completed": 0, "planned": len(entries), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_level_fault_sweep.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "cpu_threads": config.cpu_threads,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy")}}
        write_json(results_dir / "environment.json", environment)
        # Verified Step 11b sources: the retrained (codebook+BNstats) encoding, its split, and its zero-fault record.
        source = RecoveryConfig(run_id="level-fault-source-check", base_kind="retrained-clustered", pattern_id=None, source_sweep_results=config.source_sweep_results,
                                source_audit_results=config.source_audit_results, clean_checkpoint_sha256=config.clean_checkpoint_sha256)
        encoding, metadata, split, zero_record, _, source_paths, source_hashes = prepare_retrained_sources(root, source)
        accepted = list(zero_record.get("accepted_source_hash_changes", []))
        encodings = {mapping: remap_encoding(encoding, mapping) for mapping in MAPPINGS}
        require(encoding_fingerprint(encodings["sorted"]) == encoding_fingerprint(encoding), "The saved retrained encoding is not the identity (sorted) mapping")
        reference_state = decode_state(encoding)
        require_state_equal(decode_state(encodings["frequent-at-0"]), reference_state, "Fault-free decoding under the frequent-at-0 mapping")
        mapping_rows = mapping_report(encoding, encodings["frequent-at-0"])
        write_json(results_dir / "mapping_report.json", mapping_rows)
        with (results_dir / "mapping_report.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0]))
            writer.writeheader()
            writer.writerows(mapping_rows)
        encoding_items = {}
        for mapping in MAPPINGS:
            path = checkpoint_dir / "encodings" / f"{mapping}.pt"
            save_torch(path, encodings[mapping])
            encoding_items[mapping] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "fingerprint": encoding_fingerprint(encodings[mapping])}
        probabilities = {sigma_tag(sigma): {c["condition_id"]: gaussian_level_probabilities(sigma, 16, level0_safe=c["level0_safe"]) for c in config.conditions}
                         for sigma in config.sigmas}
        write_json(results_dir / "probabilities.json", probabilities)
        # Data, exactly as Step 9b.
        indices = calibration_indices(split["train"], config.calibration_examples, config.calibration_seed)
        write_json(results_dir / "splits.json", split)
        write_json(results_dir / "calibration_indices.json", indices)
        _, transform = cifar10_transforms(metadata["preprocessing"]["input_size"][-1])
        train_data = CIFAR10(root / config.data_root, train=True, transform=transform, download=False)
        test_data = CIFAR10(root / config.data_root, train=False, transform=transform, download=False)
        require(train_data.classes == test_data.classes == metadata["classes"], "CIFAR-10 class order changed")
        options = {"shuffle": False, "num_workers": config.num_workers, "persistent_workers": config.num_workers > 0, "pin_memory": device.type == "cuda"}
        calibration_loader = DataLoader(InputsOnly(Subset(train_data, indices)), batch_size=config.calibration_batch_size, **options)
        loaders = {"validation": DataLoader(Subset(train_data, split["validation"]), batch_size=config.evaluation_batch_size, **options),
                   "test": DataLoader(Subset(test_data, split["test"]["indices"]), batch_size=config.evaluation_batch_size, **options)}

        def measure(state, label):
            model = build_resnet18(metadata["num_classes"], pretrained=False)
            model.load_state_dict(state, strict=True)
            model.to(device).eval().requires_grad_(False)
            no_repair = {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}
            information = recalibrate_batchnorm(model, calibration_loader, device)
            after_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            buffers = batchnorm_buffer_names(model)
            require(all(torch.equal(after_state[name], state[name]) for name in after_state if name not in buffers), f"BN recalibration changed other state: {label}")
            recalibrated = {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}
            path = checkpoint_dir / "batchnorm" / f"{label}.pt"
            save_torch(path, {"format_version": 1, "kind": "recalibrated_batchnorm_statistics", "label": label, "statistics": {name: after_state[name] for name in buffers}})
            return no_repair, recalibrated, information, path

        no_repair, recalibrated, information, path = measure(reference_state, "fault-free")
        require(all(no_repair[s]["num_correct"] == zero_record["measurements"][s]["num_correct"] for s in SPLITS), "Fault-free accuracy differs from the Step 11b record")
        step11c = json.loads((root / config.step11c_bn_results / "measurements.json").read_text())
        previous = next(r for r in step11c if r["condition_id"] == "retrained-clustered")
        reference = {"no_repair": no_repair, "bn_recalibrated": recalibrated, "recalibration": information, "batchnorm_statistics": str(path.relative_to(root)),
                     "no_repair_matches_step11b": True,
                     "bn_recalibrated_num_correct_difference_from_step11c": {s: recalibrated[s]["num_correct"] - previous["after"][s]["num_correct"] for s in SPLITS}}
        write_json(results_dir / "reference.json", reference)
        source_paths["step11c_bn_measurements"], source_hashes["step11c_bn_measurements"] = root / config.step11c_bn_results / "measurements.json", sha256_file(root / config.step11c_bn_results / "measurements.json")
        print(json.dumps({"event": "reference_complete", "no_repair": no_repair, "bn_recalibrated": recalibrated,
                          "bn_difference_from_step11c": reference["bn_recalibrated_num_correct_difference_from_step11c"]}), flush=True)
        other = {"sorted": "frequent-at-0", "frequent-at-0": "sorted"}
        for entry in entries:
            entry_started = time.monotonic()
            current = encodings[entry["mapping"]]
            probabilities_entry = probabilities[sigma_tag(entry["sigma_over_window"])][entry["condition_id"]]
            pattern = generate_level_fault_pattern(current, probabilities_entry, seed=entry["seed"])
            pattern_path = checkpoint_dir / "patterns" / f"{entry['record_id']}.pt"
            save_torch(pattern_path, pattern)
            saved = torch.load(pattern_path, map_location="cpu", weights_only=True)
            state = apply_level_fault_pattern(current, saved)
            summary, checks = audit_pattern(current, encodings[other[entry["mapping"]]], pattern, saved, state, config)
            write_json(results_dir / "audit" / f"{entry['record_id']}.json", {"record_id": entry["record_id"], "kind": KIND, "passed": True, "checks": checks})
            faulty_no_repair, faulty_bn, faulty_information, bn_path = measure(state, entry["record_id"])
            records.append({**entry, "counts": summary_counts(summary), "pattern": str(pattern_path.relative_to(root)), "pattern_sha256": sha256_file(pattern_path),
                            "no_repair": faulty_no_repair, "bn_recalibrated": faulty_bn, "recalibration": faulty_information, "non_batchnorm_state_preserved": True,
                            "batchnorm_statistics": str(bn_path.relative_to(root)), "batchnorm_statistics_sha256": sha256_file(bn_path),
                            "audit_largest_abs_z": checks["per_level_rates_within_sampling_error"]["largest_abs_z"], "elapsed_seconds": time.monotonic() - entry_started})
            write_json(results_dir / "conditions" / f"{entry['record_id']}.json", records[-1])
            write_json(results_dir / "measurements.json", records)
            write_json(results_dir / "status.json", {"status": "running", "completed": len(records), "planned": len(entries)})
            print(json.dumps({"event": "condition_complete", "record_id": entry["record_id"], "realized_fault_rate": summary["realized_fault_rate"],
                              "test_no_repair": faulty_no_repair["test"]["accuracy"], "test_bn": faulty_bn["test"]["accuracy"],
                              "completed": len(records), "planned": len(entries), "elapsed_seconds": records[-1]["elapsed_seconds"]}), flush=True)
            del state, pattern, saved
        summary = aggregate(records, reference, config)
        write_json(results_dir / "aggregate.json", summary)
        with (results_dir / "measurements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows(records))
        plot(summary, results_dir)
        manifest = {"run_id": config.run_id, "encodings": encoding_items,
                    "source_artifacts": {name: {"path": str(Path(path).relative_to(root)), "sha256": source_hashes[name]} for name, path in source_paths.items()},
                    "pattern_kind": KIND, "method": "Level-dependent static adjacent-level patterns from the Step 13a Gaussian model (L = 16); no repair, then Step 9b BatchNorm recalibration.",
                    "plots": {f"level_faults.{e}": sha256_file(results_dir / f"level_faults.{e}") for e in ("png", "pdf")},
                    "accepted_source_hash_changes": unique_changes(accepted), "environment": environment, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "verification.json", verify(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "completed": len(records), "planned": len(entries), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "level_fault_sweep_complete", "run_id": config.run_id, "completed": len(records), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return records
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed": len(records),
                                                  "planned": len(entries), "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_level_faults.json"))
    args = parser.parse_args()
    run_level_fault_sweep(LevelFaultSweepConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
