"""Measure clustering-only loss against a saved, immutable clean baseline."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
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

from fault_lora.data.cifar10 import cifar10_transforms, preprocessing_metadata
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.memory.clustering import cluster_model, decode_layer, decode_state, physical_levels, storage_estimates
from fault_lora.models.resnet import build_resnet18, load_clean_checkpoint


@dataclass
class ClusteredConfig:
    run_id: str = "resnet18-cifar10-clustered-k16-seed42"
    num_clusters: int = 16
    seed: int = 42
    n_init: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    clustering_threads: int = 4
    device: str = "mps"
    source_checkpoint: str = "checkpoints/resnet18-cifar10-clean-seed42/best.pt"
    source_results: str = "results/resnet18-cifar10-clean-seed42"
    data_root: str = "data"
    checkpoint_root: str = "checkpoints"
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        for name in ("num_clusters", "seed", "n_init", "max_iter", "clustering_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.num_clusters <= 256 or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid cluster count or seed")
        if min(self.n_init, self.max_iter, self.clustering_threads) < 1 or not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError("Invalid KMeans configuration")
        if self.device not in ("mps", "cpu", "cuda", "auto"):
            raise ValueError("Invalid device")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_torch(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_source(root, config):
    source = root / config.source_checkpoint
    results = root / config.source_results
    recorded_metrics = json.loads((results / "metrics.json").read_text())
    if json.loads((results / "status.json").read_text())["status"] != "complete":
        raise ValueError("Source baseline did not complete")
    digest = sha256_file(source)
    if digest != recorded_metrics["checkpoint_sha256"]:
        raise ValueError("Clean checkpoint differs from the recorded baseline")
    if (root / recorded_metrics["checkpoint"]).resolve() != source.resolve():
        raise ValueError("Source results refer to a different checkpoint")
    clean, checkpoint = load_clean_checkpoint(source)
    if checkpoint["architecture"] != "resnet18" or checkpoint["dataset"] != "CIFAR-10" or checkpoint["num_classes"] != 10:
        raise ValueError("This workflow requires a clean CIFAR-10 ResNet-18 baseline")
    if checkpoint["representation"] != "original dense weights; no clustering, faults, or adapters":
        raise ValueError("Source checkpoint is not an original dense baseline")
    split = json.loads((results / "splits.json").read_text())
    split_digest = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not split_digest == checkpoint["split_sha256"] == recorded_metrics["split_sha256"]:
        raise ValueError("Saved split hash differs from baseline provenance")
    train, validation = set(split["train"]), set(split["validation"])
    if len(train) != len(split["train"]) or len(validation) != len(split["validation"]) or train & validation or train | validation != set(range(50000)):
        raise ValueError("Invalid saved training/validation partition")
    if split["test"]["source"] != "official test set" or split["test"]["indices"] != list(range(10000)):
        raise ValueError("Unexpected saved test split")
    image_size = checkpoint["preprocessing"]["input_size"][-1]
    if preprocessing_metadata(image_size) != checkpoint["preprocessing"]:
        raise ValueError("Current transforms do not reproduce the saved preprocessing")
    return clean, checkpoint, split, recorded_metrics, digest


def run_clustered_baseline(config, project_root):
    config.validate()
    root = Path(project_root)
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    device_name = config.device
    if device_name == "auto":
        device_name = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available[device_name]:
        raise RuntimeError(f"Requested {device_name} device is unavailable in this process")
    device = torch.device(device_name)
    torch.set_num_threads(config.clustering_threads)
    clean, source_metadata, split, original_metrics, source_hash = validate_source(root, config)
    results_dir = root / config.results_root / config.run_id
    checkpoint_dir = root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run directories already exist; choose a new run_id to preserve artifacts")
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    try:
        source_files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_clustered_baseline.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in source_files})
        write_json(results_dir / "splits.json", split)
        write_json(results_dir / "preprocessing.json", source_metadata["preprocessing"])
        environment = {"python": platform.python_version(), "platform": platform.platform(), "evaluation_device": device_name, "clustering_device": "cpu", "clustering_threads": config.clustering_threads, "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "scikit-learn", "threadpoolctl")}}
        write_json(results_dir / "environment.json", environment)

        def progress(name, statistics):
            print(json.dumps({"event": "layer_clustered", "tensor": name, **statistics}), flush=True)

        encoding, layers = cluster_model(clean, config.num_clusters, seed=config.seed,
                                        n_init=config.n_init, max_iter=config.max_iter, tol=config.tol,
                                        threads=config.clustering_threads, progress=progress)
        for name, value in clean.state_dict().items():
            if not torch.equal(value.cpu(), source_metadata["model_state_dict"][name]):
                raise RuntimeError(f"Clustering modified the clean model: {name}")
        metadata = {
            "architecture": "resnet18", "num_classes": 10, "dataset": "CIFAR-10",
            "classes": source_metadata["classes"], "preprocessing": source_metadata["preprocessing"],
            "split_sha256": source_metadata["split_sha256"], "source_checkpoint_sha256": source_hash,
            "source_checkpoint": config.source_checkpoint, "clustering_config": asdict(config),
            "representation": "per-tensor scalar codebooks and indices; no faults or adapters",
            "scope": "all Conv2d/Linear weight tensors, including stem, projections, and classifier; biases and normalization state unchanged",
            "physical_mapping": "sorted codebook; identity index-to-level; one cell per index; number of levels equals actual codebook size",
            "codebook_storage_assumption": "reliable codebooks and mapping metadata",
            "faults_injected": 0, "environment": environment,
        }
        encoding.update(metadata)
        encoded_path = checkpoint_dir / "encoding.pt"
        save_torch(encoded_path, encoding)
        # Evaluate weights reconstructed from the saved artifact, not fit-time labels.
        saved_encoding = torch.load(encoded_path, map_location="cpu", weights_only=True)
        decoded = decode_state(saved_encoding)
        for name, record in saved_encoding["clustered_tensors"].items():
            if not torch.equal(decoded[name], decode_layer(record, levels=physical_levels(record))):
                raise RuntimeError(f"Physical-level and index decoding disagree: {name}")
            layers[name].update({"codebook": record["codebook"].tolist(), "index_to_level": record["index_to_level"].tolist(), "level_to_index": record["level_to_index"].tolist()})
        for name, value in saved_encoding["untouched_state"].items():
            if not torch.equal(value, source_metadata["model_state_dict"][name]):
                raise RuntimeError(f"Untouched state changed: {name}")
        dense_path = checkpoint_dir / "clustered_dense.pt"
        save_torch(dense_path, {"model_state_dict": decoded, **metadata})
        write_json(results_dir / "layers.json", layers)
        storage = storage_estimates(saved_encoding)
        storage.update({"serialized_encoding_bytes": encoded_path.stat().st_size, "serialized_dense_checkpoint_bytes": dense_path.stat().st_size, "serialized_clean_checkpoint_bytes": (root / config.source_checkpoint).stat().st_size})
        write_json(results_dir / "storage.json", storage)
        write_json(results_dir / "manifest.json", {
            **metadata, "eligible_tensors": list(layers), "eligible_scalar_weights": sum(layer["num_weights"] for layer in layers.values()),
            "clustering": {"algorithm": "full-data scalar KMeans / Lloyd", "initialization": "k-means++", "n_init": config.n_init, "max_iter": config.max_iter, "tol": config.tol, "seed_rule": "(seed + tensor ordinal) modulo 2**32", "sampling": "none; fit every eligible scalar weight", "selection": "fixed cluster count; no tuning against validation/test performance"},
            "evaluation": {"splits": ["validation", "test"], "preprocessing": "exact saved baseline transforms", "training": "none", "batch_size": source_metadata["config"]["evaluation_batch_size"]},
            "original_clean_metrics": original_metrics,
        })
        image_size = source_metadata["preprocessing"]["input_size"][-1]
        _, transform = cifar10_transforms(image_size)
        train_data = CIFAR10(root / config.data_root, train=True, transform=transform, download=False)
        test_data = CIFAR10(root / config.data_root, train=False, transform=transform, download=False)
        if train_data.classes != source_metadata["classes"] or len(train_data) != 50000 or len(test_data) != 10000:
            raise ValueError("Dataset class order or size differs from the clean baseline")
        datasets = {"validation": Subset(train_data, split["validation"]), "test": Subset(test_data, split["test"]["indices"])}
        loaders = {name: DataLoader(dataset, batch_size=source_metadata["config"]["evaluation_batch_size"], shuffle=False, num_workers=source_metadata["config"]["num_workers"], pin_memory=device_name == "cuda", generator=torch.Generator().manual_seed(config.seed)) for name, dataset in datasets.items()}
        clustered = build_resnet18(10, pretrained=False)
        clustered.load_state_dict(decoded, strict=True)
        clustered.to(device).eval().requires_grad_(False)
        clean.to(device).eval().requires_grad_(False)
        measurements = {}
        for name, loader in loaders.items():
            print(json.dumps({"event": "evaluation_start", "split": name}), flush=True)
            clean_metrics = evaluate_classifier(clean, loader, device)
            clustered_metrics = evaluate_classifier(clustered, loader, device)
            measurements[name] = {
                "clean": clean_metrics, "clustered_no_faults": clustered_metrics,
                "accuracy_drop_percentage_points": 100 * (clean_metrics["accuracy"] - clustered_metrics["accuracy"]),
                "cross_entropy_increase": clustered_metrics["cross_entropy_loss"] - clean_metrics["cross_entropy_loss"],
                "clean_num_correct_difference_from_step3": clean_metrics["num_correct"] - original_metrics[name]["num_correct"],
            }
            print(json.dumps({"event": "evaluation_complete", "split": name, **measurements[name]}), flush=True)
        if sha256_file(root / config.source_checkpoint) != source_hash:
            raise RuntimeError("Clean checkpoint changed during the experiment")
        metrics = {"run_id": config.run_id, "num_clusters": config.num_clusters, "faults_injected": 0, "measurements": measurements,
                   "source_checkpoint_sha256": source_hash, "encoding_sha256": sha256_file(encoded_path),
                   "clustered_dense_sha256": sha256_file(dense_path), "elapsed_seconds": time.monotonic() - started,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(results_dir / "metrics.json", metrics)
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": metrics["completed_at_utc"]})
        print(json.dumps({"event": "clustered_baseline_complete", **metrics}), flush=True)
        return metrics
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/resnet18_cifar10_clustered.json")
    args = parser.parse_args()
    run_clustered_baseline(ClusteredConfig(**json.loads(args.config.read_text())), project_root)
