"""No-training baseline: re-estimate BatchNorm running statistics from unlabeled training images.

Weights, biases, and BatchNorm affine parameters are never changed and no
gradient or label is used. Only running means/variances and batch counters are
replaced, by a cumulative average over one fixed, seeded training subset.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from fault_lora.data.cifar10 import cifar10_transforms
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.fault_sweep import summarize_values
from fault_lora.evaluation.source_hashes import compare_source_hashes, unique_changes
from fault_lora.memory.faults import validate_parameters
from fault_lora.models.resnet import build_resnet18
from fault_lora.training import recovery
from fault_lora.training.baseline import choose_device
from fault_lora.training.recovery import RecoveryConfig


BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")
CSV_FIELDS = ["condition_id", "base_kind", "pattern_id", "fault_probability", "actual_corrupted_cells", "realized_fault_rate", "split", "num_examples",
              "accuracy_before", "accuracy_after", "cross_entropy_loss_before", "cross_entropy_loss_after", "accuracy_gain_percentage_points"]


@dataclass
class BNRecalibrationConfig:
    run_id: str = "resnet18-cifar10-bn-recalibration-seed42"
    conditions: list = field(default_factory=list)
    calibration_examples: int = 5000
    calibration_batch_size: int = 250
    seed: int = 42
    evaluation_batch_size: int = 256
    source_sweep_results: str = "results/resnet18-cifar10-fault-sweep-seed42"
    source_audit_results: str = "results/resnet18-cifar10-fault-checks-seed42"
    clean_checkpoint_sha256: str | None = None
    device: str = "mps"
    cpu_threads: int = 4
    num_workers: int = 2
    data_root: str = "data"
    results_root: str = "results"
    checkpoint_root: str = "checkpoints"

    def validate(self):
        validate_parameters(0, self.seed)
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        for name, minimum in (("calibration_examples", 1), ("calibration_batch_size", 2), ("evaluation_batch_size", 1), ("cpu_threads", 1), ("num_workers", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.calibration_examples % self.calibration_batch_size:
            raise ValueError("calibration_examples must be a multiple of calibration_batch_size so every batch has equal weight")
        if not isinstance(self.conditions, list) or not self.conditions or any(not isinstance(item, dict) or set(item) != {"base_kind", "pattern_id"} for item in self.conditions):
            raise ValueError("conditions must be a nonempty list declaring exactly base_kind and pattern_id")
        for item in self.conditions:
            condition_config(self, item).validate()
        identifiers = [condition_id(item) for item in self.conditions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("conditions must be distinct")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def condition_id(condition):
    return condition["pattern_id"] or condition["base_kind"]


def condition_config(config, condition):
    """Reuse the recovery source audit and base construction for one condition."""
    return RecoveryConfig(base_kind=condition["base_kind"], pattern_id=condition["pattern_id"], source_sweep_results=config.source_sweep_results,
                          source_audit_results=config.source_audit_results, clean_checkpoint_sha256=config.clean_checkpoint_sha256)


def calibration_indices(train_indices, num_examples, seed):
    """Fixed seeded subset of the saved training split, in the order it is presented."""
    require(0 < num_examples <= len(train_indices), "Calibration subset exceeds the training split")
    order = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(seed))[:num_examples]
    return [train_indices[position] for position in order.tolist()]


class InputsOnly(Dataset):
    """Drop labels before they can reach the recalibration loop."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0]


def recalibrate_batchnorm(model, batches, device):
    """Reset running statistics, then average batch statistics over label-free inputs.

    momentum=None makes PyTorch keep the cumulative (equal-weight) average of the
    per-batch statistics. The model is returned to eval() with momenta restored.
    """
    layers = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    require(layers and all(layer.track_running_stats for layer in layers), "Recalibration requires BatchNorm layers with running statistics")
    momenta = [layer.momentum for layer in layers]
    count = batch_count = 0
    try:
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
        model.train()
        with torch.no_grad():
            for inputs in batches:
                require(isinstance(inputs, torch.Tensor), "Calibration batches must contain inputs only")
                model(inputs.to(device))
                count += len(inputs)
                batch_count += 1
    finally:
        for layer, momentum in zip(layers, momenta):
            layer.momentum = momentum
        model.eval()
    require(batch_count > 0 and all(layer.num_batches_tracked.item() == batch_count for layer in layers), "BatchNorm layers saw a different number of calibration batches")
    require(all(torch.isfinite(layer.running_mean).all() and torch.isfinite(layer.running_var).all() for layer in layers), "Nonfinite recalibrated statistics")
    return {"calibration_examples": count, "calibration_batches": batch_count, "batchnorm_layers": len(layers)}


def batchnorm_buffer_names(model):
    return sorted(f"{name}.{key}" for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm) for key in BN_BUFFERS)


def aggregate_recalibration(records):
    """Per fault probability: mean and sample SD across static patterns."""
    groups = []
    for probability in sorted({record["fault_probability"] for record in records}, key=lambda value: (value is not None, value or 0)):
        group = [record for record in records if record["fault_probability"] == probability]
        require(len({record["base_kind"] for record in group}) == 1, "A fault probability mixes base kinds")
        entry = {"base_kind": group[0]["base_kind"], "fault_probability": probability, "num_conditions": len(group), "condition_ids": [record["condition_id"] for record in group], "splits": {}}
        for split in ("validation", "test"):
            entry["splits"][split] = {"accuracy_before": summarize_values([record["before"][split]["accuracy"] for record in group]),
                                      "accuracy_after": summarize_values([record["after"][split]["accuracy"] for record in group]),
                                      "accuracy_gain_percentage_points": summarize_values([record["accuracy_gain_percentage_points"][split] for record in group])}
        groups.append(entry)
    return {"groups": groups, "variability": "Sample standard deviation across static patterns (ddof=1); null for a single condition. Not a confidence interval."}


def csv_rows(records):
    rows = []
    for record in records:
        for split in ("validation", "test"):
            row = {**{name: record[name] for name in CSV_FIELDS[:6]}, "split": split, "num_examples": record["after"][split]["num_examples"],
                   "accuracy_before": record["before"][split]["accuracy"], "accuracy_after": record["after"][split]["accuracy"],
                   "cross_entropy_loss_before": record["before"][split]["cross_entropy_loss"], "cross_entropy_loss_after": record["after"][split]["cross_entropy_loss"],
                   "accuracy_gain_percentage_points": record["accuracy_gain_percentage_points"][split]}
            rows.append({name: "" if value is None else str(value) for name, value in row.items()})
    return rows


def verify_recalibration(root, config):
    """Re-read every saved record; fail on any inconsistency."""
    root = Path(root)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    records = json.loads((results_dir / "measurements.json").read_text())
    require([record["condition_id"] for record in records] == [condition_id(item) for item in config.conditions], "Measurements are incomplete, duplicated, or reordered")
    split = json.loads((results_dir / "splits.json").read_text())
    indices = json.loads((results_dir / "calibration_indices.json").read_text())
    require(indices == calibration_indices(split["train"], config.calibration_examples, config.seed), "Calibration subset differs from its seeded definition")
    require(len(set(indices)) == config.calibration_examples and set(indices) <= set(split["train"]) and not set(indices) & set(split["validation"]),
            "Calibration subset must be distinct training images outside validation")
    batches = config.calibration_examples // config.calibration_batch_size
    for record in records:
        require(record == json.loads((results_dir / "conditions" / f"{record['condition_id']}.json").read_text()), f"Individual and combined records disagree: {record['condition_id']}")
        require(record["recalibration"]["calibration_examples"] == config.calibration_examples and record["recalibration"]["calibration_batches"] == batches, f"Calibration size differs: {record['condition_id']}")
        require(record["non_batchnorm_state_preserved"] and all(value == 0 for value in record["before_num_correct_difference_from_step7"].values()), f"Base check failed: {record['condition_id']}")
        path = root / record["batchnorm_statistics"]
        require(path.parent == checkpoint_dir / "batchnorm" and sha256_file(path) == record["batchnorm_statistics_sha256"], f"Saved statistics changed: {record['condition_id']}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(len(saved["statistics"]) == 3 * record["recalibration"]["batchnorm_layers"] and all(torch.isfinite(value).all() for value in saved["statistics"].values()), f"Saved statistics are malformed: {record['condition_id']}")
        for name in ("validation", "test"):
            require(record["accuracy_gain_percentage_points"][name] == 100 * (record["after"][name]["accuracy"] - record["before"][name]["accuracy"]), f"Gain differs from its measurements: {record['condition_id']}")
    require(json.loads((results_dir / "aggregate.json").read_text()) == json.loads(json.dumps(aggregate_recalibration(records))), "Saved aggregate differs from the individual records")
    with (results_dir / "measurements.csv").open(newline="") as stream:
        require(list(csv.DictReader(stream)) == csv_rows(records), "CSV and JSON measurements disagree")
    manifest = json.loads((results_dir / "manifest.json").read_text())
    for name, item in manifest["source_artifacts"].items():
        require(sha256_file(root / item["path"]) == item["sha256"], f"Input artifact changed: {name}")
    accepted = compare_source_hashes(root, json.loads((results_dir / "source_sha256.json").read_text()), "Source code changed during the run")
    return {"status": "passed", "conditions_verified": len(records), "calibration_subset_reproduced_and_disjoint_from_validation": True, "paired_baselines_match_step7": True,
            "only_batchnorm_buffers_changed": True, "saved_statistics_hashes_verified": True, "aggregate_recomputed": True, "csv_matches_json": True,
            "source_artifacts_verified": len(manifest["source_artifacts"]), "source_code_unchanged": not accepted, "accepted_source_hash_changes": unique_changes(accepted)}


def run_bn_recalibration(config, project_root):
    config.validate()
    root = Path(project_root)
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Recalibration directories already exist; choose a new run_id")
    (results_dir / "conditions").mkdir(parents=True)
    (checkpoint_dir / "batchnorm").mkdir(parents=True)
    started = time.monotonic()
    records = []
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "completed_conditions": 0, "planned_conditions": len(config.conditions), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_bn_recalibration.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "cpu_threads": config.cpu_threads,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy")}}
        write_json(results_dir / "environment.json", environment)
        loaders, shared, source_paths, source_hashes, accepted = None, None, {}, {}, []
        for condition in config.conditions:
            identifier = condition_id(condition)
            print(json.dumps({"event": "recalibration_start", "condition_id": identifier, "completed_conditions": len(records), "planned_conditions": len(config.conditions)}), flush=True)
            condition_started = time.monotonic()
            source = condition_config(config, condition)
            encoding, metadata, split, record, _, paths, hashes = recovery.prepare_sources(root, source)
            accepted += record.get("accepted_source_hash_changes", [])
            require(all(source_hashes.get(name, digest) == digest for name, digest in hashes.items()), "Conditions disagree about a shared input artifact")
            source_paths.update({name: path for name, path in paths.items() if name != "selected_pattern"})
            source_hashes.update({name: digest for name, digest in hashes.items() if name != "selected_pattern"})
            if "selected_pattern" in paths:
                source_paths[f"pattern_{identifier}"], source_hashes[f"pattern_{identifier}"] = paths["selected_pattern"], hashes["selected_pattern"]
            if loaders is None:
                shared = (metadata["split_sha256"], metadata["preprocessing"], metadata["classes"])
                indices = calibration_indices(split["train"], config.calibration_examples, config.seed)
                write_json(results_dir / "splits.json", split)
                write_json(results_dir / "preprocessing.json", metadata["preprocessing"])
                write_json(results_dir / "calibration_indices.json", indices)
                _, transform = cifar10_transforms(metadata["preprocessing"]["input_size"][-1])
                train_data = CIFAR10(root / config.data_root, train=True, transform=transform, download=False)
                test_data = CIFAR10(root / config.data_root, train=False, transform=transform, download=False)
                require(train_data.classes == test_data.classes == metadata["classes"] and len(train_data) == 50000 and len(test_data) == 10000, "CIFAR-10 class order or size changed")
                options = {"shuffle": False, "num_workers": config.num_workers, "persistent_workers": config.num_workers > 0, "pin_memory": device.type == "cuda"}
                # Deterministic evaluation preprocessing, fixed order, inputs only.
                calibration_loader = DataLoader(InputsOnly(Subset(train_data, indices)), batch_size=config.calibration_batch_size, **options)
                loaders = {"validation": DataLoader(Subset(train_data, split["validation"]), batch_size=config.evaluation_batch_size, **options),
                           "test": DataLoader(Subset(test_data, split["test"]["indices"]), batch_size=config.evaluation_batch_size, **options)}
            require(shared == (metadata["split_sha256"], metadata["preprocessing"], metadata["classes"]), "Conditions use different splits or preprocessing")
            original = recovery.build_base_state(source, encoding, paths)
            model = build_resnet18(metadata["num_classes"], pretrained=False)
            model.load_state_dict(original, strict=True)
            model.to(device).eval().requires_grad_(False)
            before = {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}
            differences = {name: before[name]["num_correct"] - record["measurements"][name]["num_correct"] for name in loaders}
            require(all(value == 0 for value in differences.values()), f"Paired baseline differs from the Step 7 measurement: {identifier}")
            information = recalibrate_batchnorm(model, calibration_loader, device)
            require(information["calibration_examples"] == config.calibration_examples, "Calibration loader returned a different number of images")
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            buffers = batchnorm_buffer_names(model)
            require(set(state) == set(original) and set(buffers) <= set(state), "Recalibration changed the state layout")
            require(all(torch.equal(state[name], original[name]) for name in state if name not in buffers), f"Recalibration changed a tensor outside BatchNorm statistics: {identifier}")
            after = {name: evaluate_classifier(model, loader, device) for name, loader in loaders.items()}
            path = checkpoint_dir / "batchnorm" / f"{identifier}.pt"
            save_torch(path, {"format_version": 1, "kind": "recalibrated_batchnorm_statistics", "condition": condition, "config": asdict(config),
                              "split_sha256": metadata["split_sha256"], "statistics": {name: state[name] for name in buffers}})
            records.append({"condition_id": identifier, **condition, "fault_seed": record["seed"], "fault_probability": record["probability"],
                            "actual_corrupted_cells": record["counts"]["actual_corrupted_cells"], "realized_fault_rate": record["counts"]["realized_fault_rate"],
                            "before": before, "after": after, "before_num_correct_difference_from_step7": differences,
                            "accuracy_gain_percentage_points": {name: 100 * (after[name]["accuracy"] - before[name]["accuracy"]) for name in loaders},
                            "recalibration": information, "non_batchnorm_state_preserved": True,
                            "largest_running_mean_change": max((state[name] - original[name]).abs().max().item() for name in buffers if name.endswith("running_mean")),
                            "largest_running_variance_change": max((state[name] - original[name]).abs().max().item() for name in buffers if name.endswith("running_var")),
                            "batchnorm_statistics": str(path.relative_to(root)), "batchnorm_statistics_sha256": sha256_file(path), "elapsed_seconds": time.monotonic() - condition_started})
            write_json(results_dir / "conditions" / f"{identifier}.json", records[-1])
            write_json(results_dir / "measurements.json", records)
            write_json(results_dir / "status.json", {"status": "running", "completed_conditions": len(records), "planned_conditions": len(config.conditions)})
            print(json.dumps({"event": "recalibration_complete", "condition_id": identifier, "before": before, "after": after, "elapsed_seconds": records[-1]["elapsed_seconds"]}), flush=True)
            del model, state, original
        write_json(results_dir / "aggregate.json", aggregate_recalibration(records))
        temporary = results_dir / "measurements.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows(records))
        temporary.replace(results_dir / "measurements.csv")
        manifest = {"run_id": config.run_id, "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": source_hashes[name]} for name, path in source_paths.items()},
                    "split_sha256": shared[0], "preprocessing": shared[1],
                    "method": "Reset every BatchNorm running mean/variance and counter; forward the calibration subset once in train() mode under torch.no_grad() with momentum=None (cumulative average of per-batch statistics); evaluate in eval() mode.",
                    "calibration_data": "Seeded subset of the saved 45,000-image training split in fixed order; deterministic evaluation preprocessing; labels dropped before the loop; equal-size batches.",
                    "unchanged_state": "All weights, biases, and BatchNorm affine parameters are exactly preserved; only running_mean, running_var, and num_batches_tracked change.",
                    "selection_policy": "Conditions, subset size, batch size, and seed fixed in the configuration; no training, tuning, or checkpoint selection. Validation and test only characterize the declared conditions.",
                    "relation_to_lora": "Separate run and directories; LoRA recovery runs keep their own frozen original statistics and equality checks.",
                    "accepted_source_hash_changes": unique_changes(accepted), "environment": environment, "completed_conditions": len(records), "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "verification.json", verify_recalibration(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "completed_conditions": len(records), "planned_conditions": len(config.conditions), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "bn_recalibration_complete", "run_id": config.run_id, "completed_conditions": len(records), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return records
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "completed_conditions": len(records),
                                                  "planned_conditions": len(config.conditions), "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_bn_recalibration.json"))
    args = parser.parse_args()
    run_bn_recalibration(BNRecalibrationConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
