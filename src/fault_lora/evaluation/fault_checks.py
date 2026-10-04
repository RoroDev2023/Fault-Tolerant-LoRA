"""Audit saved ResNet fault artifacts without evaluating a dataset."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import time

import torch

from fault_lora.evaluation.clustered_baseline import sha256_file, write_json
from fault_lora.evaluation.fault_simulation import FaultConfig, validate_source
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern
from fault_lora.models.resnet import build_resnet18


@dataclass
class FaultCheckConfig:
    run_id: str = "resnet18-cifar10-fault-checks-seed42"
    source_fault_results: str = "results/resnet18-cifar10-faults-p001-seed42"
    results_root: str = "results"
    cpu_threads: int = 4

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int) or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def require_state_equal(actual, expected, label):
    require(set(actual) == set(expected), f"{label}: state tensor names differ")
    for name, value in expected.items():
        other = actual[name]
        require(other.dtype == value.dtype and other.shape == value.shape and torch.equal(other, value),
                f"{label}: tensor differs: {name}")


def inspect_transitions(encoding, pattern, actual_state):
    """Independent sparse decoding oracle; does not use simulator validators.

    Derive original physical levels directly from indices, check every saved
    transition, and build expected weights by substituting only selected entries.
    Comparing the full state also verifies all unselected and reliable entries.
    """
    records = encoding["clustered_tensors"]
    names = pattern["tensor_order"]
    require(names == sorted(set(names)) and set(names) == set(pattern["tensors"]) and set(names) <= set(records), "Invalid tensor scope")
    expected = {name: value.clone() for name, value in encoding["untouched_state"].items()}
    totals = {key: 0 for key in ("eligible_cells", "actual_corrupted_cells", "decoded_weight_changes", "lower_boundary_transitions", "upper_boundary_transitions")}
    layers = {}
    for name, record in records.items():
        indices = record["indices"].reshape(-1).long()
        weights = record["codebook"][indices].clone()
        baseline_weights = weights.clone()
        if name in pattern["tensors"]:
            entry = pattern["tensors"][name]
            positions, before, after = entry["positions"], entry["before_levels"], entry["after_levels"]
            k = record["codebook"].numel()
            require(entry["shape"] == record["shape"] and entry["num_levels"] == k, f"Layout differs: {name}")
            require(positions.ndim == before.ndim == after.ndim == 1 and positions.shape == before.shape == after.shape, f"Sparse shapes differ: {name}")
            require(positions.dtype == torch.int64 and before.dtype == after.dtype == torch.uint8, f"Sparse dtypes differ: {name}")
            if positions.numel():
                require(positions[0].item() >= 0 and positions[-1].item() < indices.numel() and bool(torch.all(positions[1:] > positions[:-1])), f"Invalid positions: {name}")
                require(k > 1 and before.max().item() < k and after.max().item() < k, f"Levels out of bounds: {name}")
                require(torch.equal(before.long(), record["index_to_level"][indices[positions]]), f"Original levels differ: {name}")
                delta = after.to(torch.int16) - before.to(torch.int16)
                require(bool(torch.all(delta.abs() == 1)), f"Non-neighbor transition: {name}")
                require(bool(torch.all(after[before == 0] == 1)), f"Lower boundary failed: {name}")
                require(bool(torch.all(after[before == k - 1] == k - 2)), f"Upper boundary failed: {name}")
                weights[positions] = record["codebook"][record["level_to_index"][after.long()]]
            details = {
                "eligible_cells": indices.numel() if k > 1 else 0,
                "actual_corrupted_cells": positions.numel(),
                "decoded_weight_changes": int((weights != baseline_weights).sum().item()),
                "lower_boundary_transitions": int((before == 0).sum().item()),
                "upper_boundary_transitions": int((before == k - 1).sum().item()),
            }
            layers[name] = details
            for key in totals:
                totals[key] += details[key]
        expected[name] = weights.reshape(record["shape"])
    require_state_equal(actual_state, expected, "Independent transition decoding")
    return {**totals, "layers": layers, "all_state_tensors_checked": len(expected),
            "unselected_weights_and_reliable_state_match": True}


def patterns_equal(first, second):
    if first["tensor_order"] != second["tensor_order"]:
        return False
    for name in first["tensor_order"]:
        for key in ("positions", "before_levels", "after_levels"):
            if not torch.equal(first["tensors"][name][key], second["tensors"][name][key]):
                return False
    return True


def run_fault_checks(config, project_root):
    config.validate()
    root = Path(project_root)
    torch.set_num_threads(config.cpu_threads)
    fault_results = root / config.source_fault_results
    require(json.loads((fault_results / "status.json").read_text())["status"] == "complete", "Source fault run is incomplete")
    fault_config = FaultConfig(**json.loads((fault_results / "config.json").read_text()))
    fault_config.validate()
    encoding, source_paths, source_hashes = validate_source(root, fault_config)
    manifest = json.loads((fault_results / "manifest.json").read_text())
    source_paths.update({"pattern": root / manifest["fault_pattern"], "faulty_dense": root / manifest["faulty_dense"]})
    for name, key in (("pattern", "fault_pattern_sha256"), ("faulty_dense", "faulty_dense_sha256")):
        source_hashes[name] = sha256_file(source_paths[name])
        require(source_hashes[name] == manifest[key], f"Saved {name} hash differs")
    pattern = torch.load(source_paths["pattern"], map_location="cpu", weights_only=True)
    faulty_checkpoint = torch.load(source_paths["faulty_dense"], map_location="cpu", weights_only=True)
    require(pattern["source_fingerprint"] == manifest["source_encoding_fingerprint"], "Pattern fingerprint differs from manifest")
    require(pattern["seed"] == fault_config.seed and pattern["fault_probability"] == fault_config.fault_probability, "Sampling choices differ from saved configuration")
    require(pattern["torch_version"] == str(torch.__version__), "Exact seed-regeneration audit requires the recorded PyTorch version; saved-pattern replay is a separate check")
    require(set(pattern["tensor_order"]) == set(encoding["clustered_tensors"]), "Initial saved pattern must cover every clustered tensor")
    for name in ("config", "manifest", "counts", "status", "source_sha256"):
        source_paths[f"fault_{name}"] = fault_results / f"{name}.json"
        source_hashes[f"fault_{name}"] = sha256_file(source_paths[f"fault_{name}"])
    # Existing experiment code is immutable for this audit.
    for filename, digest in json.loads((fault_results / "source_sha256.json").read_text()).items():
        require(sha256_file(root / filename) == digest, f"Step 5 source changed: {filename}")
    results_dir = root / config.results_root / config.run_id
    if results_dir.exists():
        raise FileExistsError("Audit directory already exists; choose a new run_id")
    results_dir.mkdir(parents=True)
    started = time.monotonic()
    checks = {}
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})

    def passed(name, details):
        checks[name] = {"passed": True, **details}
        print(json.dumps({"event": "fault_check_passed", "check": name}), flush=True)

    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/check_fault_injection.py", root / "configs/resnet18_cifar10_fault_checks.json", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        write_json(results_dir / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__), "device": "cpu", "cpu_threads": config.cpu_threads})
        fingerprint = encoding_fingerprint(encoding)
        baseline = decode_state(encoding)
        passed("source_integrity", {"encoding_fingerprint": fingerprint, "clustered_tensor_count": len(encoding["clustered_tensors"]), "state_tensor_count": len(baseline)})

        zero = generate_fault_pattern(encoding, 0, seed=pattern["seed"])
        zero_state = apply_fault_pattern(encoding, zero)
        require_state_equal(zero_state, baseline, "Zero-fault identity")
        zero_details = inspect_transitions(encoding, zero, zero_state)
        require(zero_details["actual_corrupted_cells"] == 0, "Zero pattern selected cells")
        model = build_resnet18(num_classes=10, pretrained=False).eval().requires_grad_(False)
        inputs = torch.randn(2, *encoding["preprocessing"]["input_size"], generator=torch.Generator().manual_seed(0))
        model.load_state_dict(baseline, strict=True)
        with torch.inference_mode():
            baseline_logits = model(inputs)
        model.load_state_dict(zero_state, strict=True)
        with torch.inference_mode():
            zero_logits = model(inputs)
        require(torch.equal(zero_logits, baseline_logits) and bool(torch.isfinite(zero_logits).all()), "Zero-fault logits differ")
        passed("zero_fault_identity", {**zero_details, "synthetic_input_seed": 0, "logits_shape": list(zero_logits.shape), "logits_exactly_equal": True})
        del zero, zero_state, model

        saved_state = apply_fault_pattern(encoding, pattern)
        saved_details = inspect_transitions(encoding, pattern, saved_state)
        summary = summarize_fault_pattern(encoding, pattern)
        require(summary == json.loads((fault_results / "counts.json").read_text()), "Saved counts differ from recomputed summary")
        for key in ("eligible_cells", "actual_corrupted_cells", "decoded_weight_changes", "lower_boundary_transitions", "upper_boundary_transitions"):
            require(saved_details[key] == summary[key], f"Independent count differs: {key}")
        require_state_equal(saved_state, faulty_checkpoint["model_state_dict"], "Saved faulty checkpoint")
        require(summary["actual_corrupted_cells"] == manifest["faults_injected"] == faulty_checkpoint["faults_injected"], "Recorded fault counts differ")
        passed("saved_pattern_and_checkpoint", saved_details)
        replay = apply_fault_pattern(encoding, pattern)
        require_state_equal(replay, saved_state, "Repeated application")
        require_state_equal(decode_state(encoding), baseline, "Baseline after replay")
        passed("replay_without_accumulation", {"two_applications_exactly_equal": True, "baseline_state_preserved": True})
        del replay, saved_state, faulty_checkpoint

        regenerated = generate_fault_pattern(encoding, pattern["fault_probability"], seed=pattern["seed"])
        require(patterns_equal(regenerated, pattern), "Same seed did not reproduce saved transitions")
        passed("same_seed_reproducibility", {"seed": pattern["seed"], "all_positions_and_transitions_equal": True})
        del regenerated
        alternate_seed = (pattern["seed"] + 1) % 2**32
        alternate = generate_fault_pattern(encoding, pattern["fault_probability"], seed=alternate_seed)
        variation_expected = 0 < pattern["fault_probability"] < 1 or (pattern["fault_probability"] == 1 and any(
            bool(torch.any((entry["before_levels"] > 0) & (entry["before_levels"] < entry["num_levels"] - 1)))
            for entry in pattern["tensors"].values()))
        different = not patterns_equal(alternate, pattern)
        if variation_expected:
            require(different, "Different seed produced the identical full pattern")
        alternate_state = apply_fault_pattern(encoding, alternate)
        alternate_details = inspect_transitions(encoding, alternate, alternate_state)
        passed("different_seed_valid_pattern", {"seed": alternate_seed, "variation_expected": variation_expected, "differs_from_saved_pattern": different, **alternate_details})
        del alternate, alternate_state

        full = generate_fault_pattern(encoding, 1, seed=pattern["seed"])
        full_state = apply_fault_pattern(encoding, full)
        full_details = inspect_transitions(encoding, full, full_state)
        require(full_details["actual_corrupted_cells"] == full_details["eligible_cells"], "Full selection omitted eligible cells")
        require(full_details["lower_boundary_transitions"] > 0 and full_details["upper_boundary_transitions"] > 0, "Full-model audit did not exercise both boundaries")
        passed("all_cell_selection_and_boundaries", full_details)
        del full, full_state

        require(encoding_fingerprint(encoding) == fingerprint, "In-memory source encoding changed")
        require_state_equal(decode_state(encoding), baseline, "Final baseline preservation")
        for name, path in source_paths.items():
            require(sha256_file(path) == source_hashes[name], f"Source artifact changed: {name}")
        passed("source_preservation", {"in_memory_fingerprint_unchanged": True, "all_input_file_hashes_unchanged": True})
        report = {
            "status": "passed", "run_id": config.run_id, "checks": checks,
            "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": source_hashes[name]} for name, path in source_paths.items()},
            "performance_evaluated": False, "fault_model": pattern["kind"],
            "note": "Correctness audit on the saved ResNet model. Temporary p=0, p=1, and alternate-seed patterns are not performance experiments or hardware calibration.",
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(results_dir / "report.json", report)
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "fault_audit_complete", "passed_checks": len(checks), "report": str((results_dir / "report.json").relative_to(root)), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        return report
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(results_dir / "report.json", {"status": status, "checks": checks, "error": repr(error)})
        write_json(results_dir / "status.json", {"status": status, "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_fault_checks.json"))
    args = parser.parse_args()
    config = FaultCheckConfig(**json.loads((Path(project_root) / args.config).read_text()))
    run_fault_checks(config, project_root)
