"""A bounded, reproducible clean CIFAR-10 fine-tuning run."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from fault_lora.data.cifar10 import cifar10_transforms, preprocessing_metadata, stratified_split
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.models.resnet import build_resnet18, load_clean_checkpoint


@dataclass
class BaselineConfig:
    run_id: str = "resnet18-cifar10-clean-seed42"
    seed: int = 42
    epochs: int = 5
    image_size: int = 96
    batch_size: int = 128
    evaluation_batch_size: int = 256
    validation_per_class: int = 500
    num_workers: int = 2
    backbone_learning_rate: float = 0.001
    classifier_learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    device: str = "mps"
    data_root: str = "data"
    checkpoint_root: str = "checkpoints"
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        if self.epochs < 1 or min(self.batch_size, self.evaluation_batch_size) < 1:
            raise ValueError("Epoch count and batch sizes must be positive")
        if self.image_size < 32 or not 1 <= self.validation_per_class < 5000:
            raise ValueError("Invalid image size or CIFAR-10 validation size")
        if self.num_workers < 0 or self.seed < 0:
            raise ValueError("Worker count and seed must be nonnegative")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")
        if min(self.backbone_learning_rate, self.classifier_learning_rate) <= 0:
            raise ValueError("Learning rates must be positive")
        if not 0 <= self.momentum < 1 or self.weight_decay < 0:
            raise ValueError("Invalid SGD configuration")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def choose_device(requested):
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available[requested]:
        raise RuntimeError(f"Requested {requested} device is unavailable; use an explicit available device or allow GPU access.")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    count, correct, loss_sum = 0, 0, 0.0
    started = time.monotonic()
    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if not math.isfinite(loss.item()):
            raise RuntimeError("Non-finite training loss; stopping the run")
        loss.backward()
        optimizer.step()
        count += targets.numel()
        correct += (logits.detach().argmax(1) == targets).sum().item()
        loss_sum += loss.item() * targets.numel()
        if step % 50 == 0 or step == len(loader):
            print(json.dumps({"event": "training_progress", "batch": step, "batches": len(loader), "examples": count, "loss": loss_sum / count, "accuracy": correct / count, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    return {"num_examples": count, "num_correct": correct, "accuracy": correct / count, "cross_entropy_loss": loss_sum / count}


def save_checkpoint(path, model, metadata):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {"model_state_dict": state, **metadata}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_baseline(config, project_root, *, download=False):
    config.validate()
    project_root = Path(project_root)
    device = choose_device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    results_dir = project_root / config.results_root / config.run_id
    checkpoint_dir = project_root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run directories already exist. Choose a new run_id; existing research artifacts are preserved.")
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})

    started = time.monotonic()
    try:
        train_transform, eval_transform = cifar10_transforms(config.image_size)
        data_root = project_root / config.data_root
        train_data = CIFAR10(data_root, train=True, transform=train_transform, download=download)
        validation_data = CIFAR10(data_root, train=True, transform=eval_transform, download=False)
        split = stratified_split(train_data.targets, config.validation_per_class, config.seed)
        split.update({"seed": config.seed, "dataset": "CIFAR-10", "index_space": "official training set", "test": {"source": "official test set", "indices": list(range(10000))}})
        split_bytes = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
        split_hash = hashlib.sha256(split_bytes).hexdigest()
        write_json(results_dir / "splits.json", split)
        preprocessing = preprocessing_metadata(config.image_size)
        write_json(results_dir / "preprocessing.json", preprocessing)

        loader_options = {"num_workers": config.num_workers, "worker_init_fn": seed_worker, "pin_memory": device.type == "cuda"}
        train_generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(Subset(train_data, split["train"]), batch_size=config.batch_size, shuffle=True, generator=train_generator, **loader_options)
        validation_loader = DataLoader(Subset(validation_data, split["validation"]), batch_size=config.evaluation_batch_size, shuffle=False, generator=torch.Generator().manual_seed(config.seed + 1), **loader_options)

        cache_dir = project_root / config.checkpoint_root / "torch-cache"
        model = build_resnet18(10, pretrained=True, cache_dir=cache_dir).to(device)
        pretrained_file = cache_dir / "checkpoints" / "resnet18-f37072fd.pth"
        pretrained_hash = hashlib.sha256(pretrained_file.read_bytes()).hexdigest()
        environment = {
            "python": platform.python_version(), "platform": platform.platform(),
            "device": str(device), "torch": str(torch.__version__),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy")},
            "seed": config.seed, "reproducibility": "Seeded RNGs and saved splits; bitwise reproducibility across devices or library versions is not guaranteed.",
        }
        write_json(results_dir / "environment.json", environment)
        metadata = {
            "architecture": "resnet18", "num_classes": 10, "dataset": "CIFAR-10",
            "classes": train_data.classes, "config": asdict(config), "preprocessing": preprocessing,
            "split_sha256": split_hash, "pretrained_weights": "IMAGENET1K_V1",
            "pretrained_sha256": pretrained_hash, "environment": environment,
            "representation": "original dense weights; no clustering, faults, or adapters",
        }
        write_json(results_dir / "manifest.json", {
            **metadata,
            "dataset_integrity": {"archive_md5": train_data.tgz_md5, "training_files": train_data.train_list, "test_files": train_data.test_list},
            "selection_rule": "highest validation accuracy, then lowest validation loss; exact ties retain earlier epoch",
            "split_sizes": {"train": len(split["train"]), "validation": len(split["validation"]), "test": 10000},
            "test_policy": "Official test set is evaluated once after validation-only checkpoint selection.",
        })
        backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
        optimizer = torch.optim.SGD([
            {"params": backbone, "lr": config.backbone_learning_rate},
            {"params": model.fc.parameters(), "lr": config.classifier_learning_rate},
        ], momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        history, best_key, best_epoch = [], None, None
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.monotonic()
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            print(json.dumps({"event": "epoch_start", "epoch": epoch, "epochs": config.epochs, "learning_rates": learning_rates}), flush=True)
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            validation_metrics = evaluate_classifier(model, validation_loader, device)
            key = (validation_metrics["accuracy"], -validation_metrics["cross_entropy_loss"])
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                save_checkpoint(checkpoint_dir / "best.pt", model, {**metadata, "epoch": epoch, "validation_metrics": validation_metrics})
            history.append({"epoch": epoch, "learning_rates": learning_rates, "training": train_metrics, "validation": validation_metrics, "seconds": time.monotonic() - epoch_started})
            scheduler.step()
            save_checkpoint(checkpoint_dir / "last.pt", model, {**metadata, "epoch": epoch, "validation_metrics": validation_metrics})
            write_json(results_dir / "history.json", history)
            print(json.dumps({"event": "epoch_complete", **history[-1], "best_epoch": best_epoch}), flush=True)

        # The test set enters the workflow only after the final checkpoint is selected.
        test_data = CIFAR10(data_root, train=False, transform=eval_transform, download=False)
        test_loader = DataLoader(test_data, batch_size=config.evaluation_batch_size, shuffle=False, generator=torch.Generator().manual_seed(config.seed + 2), **loader_options)
        selected, selected_checkpoint = load_clean_checkpoint(checkpoint_dir / "best.pt", device=device)
        test_metrics = evaluate_classifier(selected, test_loader, device)
        metrics = {
            "run_id": config.run_id, "selected_epoch": best_epoch,
            "validation": selected_checkpoint["validation_metrics"], "test": test_metrics,
            "checkpoint": str((checkpoint_dir / "best.pt").relative_to(project_root)),
            "checkpoint_sha256": hashlib.sha256((checkpoint_dir / "best.pt").read_bytes()).hexdigest(),
            "split_sha256": split_hash, "elapsed_seconds": time.monotonic() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(results_dir / "metrics.json", metrics)
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": metrics["completed_at_utc"]})
        print(json.dumps({"event": "baseline_complete", **metrics}), flush=True)
        return metrics
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error": str(error), "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/resnet18_cifar10.json")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if it is missing; pretrained weights may also be downloaded.")
    args = parser.parse_args()
    config = BaselineConfig(**json.loads(args.config.read_text()))
    run_baseline(config, project_root, download=args.download)
