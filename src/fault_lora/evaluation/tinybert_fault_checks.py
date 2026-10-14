"""Audit the unchanged fault simulator on the saved TinyBERT encoding, without dataset evaluation.

The checks mirror Step 6 and reuse its independent decoding oracle
(`fault_checks.inspect_transitions`). TinyBERT has no separately saved Step 5
pattern, so a reference pattern is generated here, saved to a temporary file,
reloaded, and audited; it is not a performance experiment.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import time

import torch

from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import inspect_transitions, patterns_equal, require, require_state_equal
from fault_lora.evaluation.tinybert_clean_eval import environment_record, source_artifacts, verify_source_code
from fault_lora.evaluation.tinybert_clustered import REPRESENTATION, TinyBertClusteredConfig, prepare_clustered_sources
from fault_lora.evaluation.tinybert_sources import build_tinybert, linear_weight_names, synthetic_token_batch
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern, validate_parameters

REQUIRED_CHECKS = ("source_integrity", "zero_fault_identity", "reference_pattern_independent_decoding", "replay_without_accumulation",
                   "same_seed_reproducibility", "different_seed_valid_pattern", "all_cell_selection_and_boundaries", "source_preservation")


@dataclass
class TinyBertFaultCheckConfig:
    run_id: str = "tinybert-sst2-fault-checks-seed42"
    source_clustered_results: str = "results/tinybert-sst2-clustered-k16-seed42"
    reference_probability: float = 0.01
    reference_seed: int = 42
    cpu_threads: int = 4
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        validate_parameters(self.reference_probability, self.reference_seed)
        if not 0 < self.reference_probability < 1:
            raise ValueError("reference_probability must be strictly between 0 and 1")
        if isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int) or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")


def load_clustered_run(root, clustered_results):
    """Verify a completed TinyBERT clustering run and return its encoding, dense state, and sources."""
    directory = root / clustered_results
    read = lambda name: json.loads((directory / name).read_text())
    require(read("status.json")["status"] == "complete", "TinyBERT clustering run is incomplete")
    config = TinyBertClusteredConfig(**read("config.json"))
    config.validate()
    sources, clean_metrics = prepare_clustered_sources(root, config)
    verify_source_code(root, directory, "TinyBERT clustering")
    metrics, manifest = read("metrics.json"), read("manifest.json")
    checkpoint_dir = root / config.checkpoint_root / config.run_id
    for name, key in (("encoding", "encoding_sha256"), ("clustered_dense", "clustered_dense_sha256")):
        path = checkpoint_dir / f"{name}.pt"
        sources.paths[name], sources.hashes[name] = path, sha256_file(path)
        require(sources.hashes[name] == metrics[key], f"Saved {name} differs from its recorded hash")
    for name in ("config", "manifest", "metrics", "status", "source_sha256", "scope", "layers", "test_sets"):
        sources.paths[f"clustered_{name}"], sources.hashes[f"clustered_{name}"] = directory / f"{name}.json", sha256_file(directory / f"{name}.json")
    encoding = torch.load(sources.paths["encoding"], map_location="cpu", weights_only=True)
    dense = torch.load(sources.paths["clustered_dense"], map_location="cpu", weights_only=True)
    require(encoding["representation"] == REPRESENTATION and encoding["source_checkpoint_sha256"] == sources.hashes["clean_checkpoint"], "Encoding provenance differs")
    require(encoding_fingerprint(encoding) == metrics["encoding_fingerprint"], "Encoding fingerprint differs from the recorded one")
    require(sorted(encoding["clustered_tensors"]) == sorted(linear_weight_names(sources.model)) == sorted(manifest["eligible_tensor_names"]), "Clustered tensors are not exactly the Linear weights")
    clean_state = sources.model.state_dict()
    require_state_equal(encoding["untouched_state"], {name: value for name, value in clean_state.items() if name not in encoding["clustered_tensors"]}, "Reliable state versus clean checkpoint")
    require_state_equal(decode_state(encoding), dense["model_state_dict"], "Encoding versus saved dense reconstruction")
    return config, sources, clean_metrics, metrics, encoding, dense


def run_tinybert_fault_checks(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    torch.set_num_threads(config.cpu_threads)
    clustered_config, sources, _, clustered_metrics, encoding, dense = load_clustered_run(root, config.source_clustered_results)
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
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/check_tinybert_faults.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        write_json(results_dir / "environment.json", {**environment_record("cpu", config.cpu_threads), "torch": str(torch.__version__)})
        fingerprint = encoding_fingerprint(encoding)
        baseline = decode_state(encoding)
        passed("source_integrity", {
            "encoding_fingerprint": fingerprint, "clean_checkpoint_sha256": sources.hashes["clean_checkpoint"], "encoding_sha256": sources.hashes["encoding"],
            "clustered_dense_sha256": sources.hashes["clustered_dense"], "clustered_tensor_count": len(encoding["clustered_tensors"]), "state_tensor_count": len(baseline),
            "encoding_reconstructs_saved_dense_exactly": True, "reliable_state_equals_clean_checkpoint": True, "clustered_tensors_are_exactly_linear_weights": True,
        })

        model = build_tinybert(encoding["model_config"])
        inputs = synthetic_token_batch(encoding["model_config"], seed=0)
        zero = generate_fault_pattern(encoding, 0, seed=config.reference_seed)
        zero_state = apply_fault_pattern(encoding, zero)
        require_state_equal(zero_state, baseline, "Zero-fault identity")
        zero_details = inspect_transitions(encoding, zero, zero_state)
        require(zero_details["actual_corrupted_cells"] == 0, "Zero pattern selected cells")
        model.load_state_dict(dense["model_state_dict"], strict=True)
        with torch.inference_mode():
            clustered_logits = model(**inputs).logits
        model.load_state_dict(zero_state, strict=True)
        with torch.inference_mode():
            zero_logits = model(**inputs).logits
        require(torch.equal(zero_logits, clustered_logits) and bool(torch.isfinite(zero_logits).all()), "Zero-fault logits differ from the clustered model")
        passed("zero_fault_identity", {**zero_details, "synthetic_token_seed": 0, "logits_shape": list(zero_logits.shape), "logits_exactly_equal_to_clustered_model": True})
        del zero, zero_state

        with tempfile.TemporaryDirectory(prefix="tinybert-audit-") as directory:
            path = Path(directory) / "reference.pt"
            save_torch(path, generate_fault_pattern(encoding, config.reference_probability, seed=config.reference_seed))
            pattern = torch.load(path, map_location="cpu", weights_only=True)
        require(pattern["torch_version"] == str(torch.__version__), "Seed regeneration requires the recorded PyTorch version")
        reference_state = apply_fault_pattern(encoding, pattern)
        reference_details = inspect_transitions(encoding, pattern, reference_state)
        summary = summarize_fault_pattern(encoding, pattern)
        for key in ("eligible_cells", "actual_corrupted_cells", "decoded_weight_changes", "lower_boundary_transitions", "upper_boundary_transitions"):
            require(reference_details[key] == summary[key], f"Independent count differs: {key}")
        require(reference_details["actual_corrupted_cells"] > 0, "Reference pattern selected no cells")
        require(summary["upward_transitions"] + summary["downward_transitions"] == summary["actual_corrupted_cells"], "Transition directions do not account for every cell")
        model.load_state_dict(reference_state, strict=True)
        with torch.inference_mode():
            require(bool(torch.isfinite(model(**inputs).logits).all()), "Faulty model produced non-finite logits")
        passed("reference_pattern_independent_decoding", {"probability": config.reference_probability, "seed": config.reference_seed, "saved_and_reloaded": True,
                                                          "realized_fault_rate": summary["realized_fault_rate"], "upward_transitions": summary["upward_transitions"],
                                                          "downward_transitions": summary["downward_transitions"], "finite_logits": True, **reference_details})
        replay = apply_fault_pattern(encoding, pattern)
        require_state_equal(replay, reference_state, "Repeated application")
        require_state_equal(decode_state(encoding), baseline, "Baseline after replay")
        passed("replay_without_accumulation", {"two_applications_exactly_equal": True, "baseline_state_preserved": True})
        del replay, reference_state

        regenerated = generate_fault_pattern(encoding, pattern["fault_probability"], seed=pattern["seed"])
        require(patterns_equal(regenerated, pattern), "Same seed did not reproduce the saved transitions")
        passed("same_seed_reproducibility", {"seed": pattern["seed"], "all_positions_and_transitions_equal": True})
        del regenerated
        alternate_seed = (pattern["seed"] + 1) % 2**32
        alternate = generate_fault_pattern(encoding, pattern["fault_probability"], seed=alternate_seed)
        require(not patterns_equal(alternate, pattern), "Different seed produced the identical full pattern")
        alternate_state = apply_fault_pattern(encoding, alternate)
        alternate_details = inspect_transitions(encoding, alternate, alternate_state)
        passed("different_seed_valid_pattern", {"seed": alternate_seed, "differs_from_reference_pattern": True, **alternate_details})
        del alternate, alternate_state

        full = generate_fault_pattern(encoding, 1, seed=pattern["seed"])
        full_state = apply_fault_pattern(encoding, full)
        full_details = inspect_transitions(encoding, full, full_state)
        require(full_details["actual_corrupted_cells"] == full_details["eligible_cells"], "Full selection omitted eligible cells")
        require(full_details["lower_boundary_transitions"] > 0 and full_details["upper_boundary_transitions"] > 0, "Full-model audit did not exercise both boundaries")
        for name, value in encoding["untouched_state"].items():
            require(torch.equal(full_state[name], value), f"Reliable state changed under p=1: {name}")
        passed("all_cell_selection_and_boundaries", {**full_details, "reliable_state_unchanged": True})
        del full, full_state

        require(encoding_fingerprint(encoding) == fingerprint, "In-memory source encoding changed")
        require_state_equal(decode_state(encoding), baseline, "Final baseline preservation")
        for name, path in sources.paths.items():
            require(sha256_file(path) == sources.hashes[name], f"Source artifact changed: {name}")
        passed("source_preservation", {"in_memory_fingerprint_unchanged": True, "all_input_file_hashes_unchanged": len(sources.paths)})
        report = {
            "status": "passed", "run_id": config.run_id, "checks": checks, "source_artifacts": source_artifacts(root, sources.paths, sources.hashes),
            "performance_evaluated": False, "fault_model": pattern["kind"],
            "note": "Correctness audit on the saved TinyBERT encoding. The temporary p=0, reference, alternate-seed, and p=1 patterns are not performance experiments or hardware calibration.",
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(results_dir / "report.json", report)
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "fault_audit_complete", "passed_checks": len(checks), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        return report
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(results_dir / "report.json", {"status": status, "checks": checks, "error": repr(error)})
        write_json(results_dir / "status.json", {"status": status, "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2_fault_checks.json")
    args = parser.parse_args()
    run_tinybert_fault_checks(TinyBertFaultCheckConfig(**json.loads(args.config.read_text())), project_root)
