"""Generate one static fault artifact; no dataset evaluation or training."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

import torch

from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.memory.clustering import decode_state, eligible_weight_names
from fault_lora.memory.faults import apply_fault_pattern, generate_fault_pattern, summarize_fault_pattern, validate_parameters
from fault_lora.models.resnet import build_resnet18, load_clean_checkpoint


@dataclass
class FaultConfig:
    run_id: str = "resnet18-cifar10-faults-p001-seed42"
    fault_probability: float = 0.01
    seed: int = 42
    cpu_threads: int = 4
    source_encoding: str = "checkpoints/resnet18-cifar10-clustered-k16-seed42/encoding.pt"
    source_clustered_dense: str = "checkpoints/resnet18-cifar10-clustered-k16-seed42/clustered_dense.pt"
    source_results: str = "results/resnet18-cifar10-clustered-k16-seed42"
    checkpoint_root: str = "checkpoints"
    results_root: str = "results"

    def validate(self):
        validate_parameters(self.fault_probability, self.seed)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int) or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")


def validate_source(root, config):
    results = root / config.source_results
    if json.loads((results / "status.json").read_text())["status"] != "complete":
        raise ValueError("Source clustered baseline did not complete")
    metrics = json.loads((results / "metrics.json").read_text())
    paths = {"encoding": root / config.source_encoding, "clustered_dense": root / config.source_clustered_dense}
    hashes = {}
    for name, path in paths.items():
        hashes[name] = sha256_file(path)
        if hashes[name] != metrics[f"{name}_sha256"]:
            raise ValueError(f"Source {name} differs from the recorded clustered baseline")
    encoding = torch.load(paths["encoding"], map_location="cpu", weights_only=True)
    dense = torch.load(paths["clustered_dense"], map_location="cpu", weights_only=True)
    if encoding["faults_injected"] != 0 or dense["faults_injected"] != 0 or metrics["faults_injected"] != 0:
        raise ValueError("Step 5 requires the uncorrupted clustered baseline")
    if encoding["architecture"] != "resnet18" or encoding["dataset"] != "CIFAR-10" or encoding["num_classes"] != 10:
        raise ValueError("This runner requires the saved CIFAR-10 ResNet-18 encoding")
    model = build_resnet18(num_classes=10, pretrained=False)
    if set(eligible_weight_names(model)) != set(encoding["clustered_tensors"]):
        raise ValueError("Saved scope differs from all ResNet-18 Conv2d/Linear weights")
    decoded = decode_state(encoding)
    if set(decoded) != set(dense["model_state_dict"]) or any(not torch.equal(value, dense["model_state_dict"][name]) for name, value in decoded.items()):
        raise ValueError("Saved encoding does not reconstruct the recorded clustered dense state")
    paths["clean"] = root / encoding["source_checkpoint"]
    hashes["clean"] = sha256_file(paths["clean"])
    if hashes["clean"] != encoding["source_checkpoint_sha256"] or hashes["clean"] != metrics["source_checkpoint_sha256"]:
        raise ValueError("Original clean checkpoint differs from clustered provenance")
    return encoding, paths, hashes


def run_fault_simulation(config, project_root):
    config.validate()
    root = Path(project_root)
    torch.set_num_threads(config.cpu_threads)
    encoding, source_paths, source_hashes = validate_source(root, config)
    results_dir = root / config.results_root / config.run_id
    checkpoint_dir = root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run directories already exist; choose a new run_id to preserve artifacts")
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        source_files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/generate_fault_pattern.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in source_files})
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "cpu_threads": config.cpu_threads,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy")}}
        write_json(results_dir / "environment.json", environment)
        pattern = generate_fault_pattern(encoding, config.fault_probability, seed=config.seed)
        pattern_path = checkpoint_dir / "pattern.pt"
        save_torch(pattern_path, pattern)
        saved_pattern = torch.load(pattern_path, map_location="cpu", weights_only=True)
        summary = summarize_fault_pattern(encoding, saved_pattern)
        faulty_state = apply_fault_pattern(encoding, saved_pattern)
        baseline = decode_state(encoding)
        actual_weight_changes = sum(int((baseline[name] != faulty_state[name]).sum().item()) for name in encoding["clustered_tensors"])
        if actual_weight_changes != summary["decoded_weight_changes"]:
            raise RuntimeError("Dense changed-weight count differs from saved transitions")
        for name in encoding["untouched_state"]:
            if not torch.equal(faulty_state[name], baseline[name]):
                raise RuntimeError(f"Fault application changed reliable state: {name}")
        pattern_hash = sha256_file(pattern_path)
        metadata = {key: encoding[key] for key in ("architecture", "num_classes", "dataset", "classes", "preprocessing", "split_sha256", "source_checkpoint", "source_checkpoint_sha256", "clustering_config")}
        metadata.update({
            "representation": "dense reconstruction of clustered weights under a saved static adjacent-physical-level pattern; no adapters",
            "source_encoding": config.source_encoding, "source_encoding_sha256": source_hashes["encoding"],
            "source_encoding_fingerprint": saved_pattern["source_fingerprint"],
            "fault_pattern": str(pattern_path.relative_to(root)), "fault_pattern_sha256": pattern_hash,
            "fault_config": asdict(config), "faults_injected": summary["actual_corrupted_cells"],
            "fault_model": {key: value for key, value in saved_pattern.items() if key not in ("tensors", "tensor_order")},
            "environment": environment,
        })
        dense_path = checkpoint_dir / "faulty_dense.pt"
        save_torch(dense_path, {"model_state_dict": faulty_state, **metadata})
        # Check restoration and inference without touching any dataset split.
        restored, restored_metadata = load_clean_checkpoint(dense_path)
        restored.requires_grad_(False)
        for name, value in restored.state_dict().items():
            if not torch.equal(value, faulty_state[name]):
                raise RuntimeError(f"Saved faulty checkpoint round trip failed: {name}")
        image_size = restored_metadata["preprocessing"]["input_size"][-1]
        with torch.inference_mode():
            logits = restored(torch.zeros(2, 3, image_size, image_size))
        if tuple(logits.shape) != (2, 10) or not torch.isfinite(logits).all():
            raise RuntimeError("Restored faulty ResNet failed the CPU forward smoke check")
        for name, path in source_paths.items():
            if sha256_file(path) != source_hashes[name]:
                raise RuntimeError(f"Source artifact changed during generation: {name}")
        write_json(results_dir / "counts.json", summary)
        manifest = {
            **metadata, "eligible_tensor_names": saved_pattern["tensor_order"],
            "preserved_tensor_names": sorted(encoding["untouched_state"]),
            "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": source_hashes[name]} for name, path in source_paths.items()},
            "faulty_dense": str(dense_path.relative_to(root)), "faulty_dense_sha256": sha256_file(dense_path),
            "pattern_file_bytes": pattern_path.stat().st_size, "faulty_dense_file_bytes": dense_path.stat().st_size,
            "checks": {"saved_pattern_validated": True, "dense_changed_weight_count_matches": True,
                       "reliable_state_preserved": True, "source_artifact_hashes_preserved": True,
                       "saved_dense_round_trip_exact": True, "cpu_forward_shape": [2, 10], "cpu_forward_finite": True},
            "performance_evaluated": False,
            "note": "One static pattern only. No test/validation measurements, rate sweep, hardware calibration, or adapter training. Fault application begins from the source encoding each time.",
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "static_fault_pattern_saved", "run_id": config.run_id,
                          **{key: value for key, value in summary.items() if key != "layers"},
                          "pattern": str(pattern_path.relative_to(root)), "faulty_dense": str(dense_path.relative_to(root)),
                          "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return summary
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_faults.json"))
    args = parser.parse_args()
    config_path = Path(project_root) / args.config
    run_fault_simulation(FaultConfig(**json.loads(config_path.read_text())), project_root)
