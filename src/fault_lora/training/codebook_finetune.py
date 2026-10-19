"""Fine-tune only codebook values of the clustered ResNet-18; indices stay fixed.

Each eligible Conv2d/Linear weight is computed as codebook[indices] in the forward
pass, so a codebook entry's gradient is the sum of its member weights' gradients
(weight sharing as in Deep Compression, Han et al. 2016). BatchNorm layers train in
train() mode so their running statistics are re-estimated for the clustered weights.
Indices and mappings never change; the selected model is saved in the Step 4
encoding format with retrained codebooks and the new BatchNorm state.
"""

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from fault_lora.data.cifar10 import cifar10_transforms
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import require, require_state_equal
from fault_lora.evaluation.fault_sweep import SweepConfig, prepare_sources as prepare_sweep_sources
from fault_lora.evaluation.source_hashes import compare_source_hashes
from fault_lora.memory.clustering import decode_state, eligible_weight_names, validate_record
from fault_lora.memory.faults import encoding_fingerprint, validate_parameters
from fault_lora.models.resnet import build_resnet18
from fault_lora.training.baseline import choose_device, seed_worker


SPLITS = ("validation", "test")
BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")
EVALUATION_FIELDS = ["model", "split", "num_examples", "num_correct", "accuracy", "cross_entropy_loss", "gap_to_clean_percentage_points"]


class CodebookLayer(nn.Module):
    """A Conv2d/Linear whose weight exists only as codebook[indices]."""

    def __init__(self, base, record):
        super().__init__()
        validate_record(record)
        require(isinstance(base, (nn.Conv2d, nn.Linear)) and list(base.weight.shape) == record["shape"], "Codebook record does not fit this layer")
        require(not isinstance(base, nn.Conv2d) or base.groups == 1, "Grouped convolutions are not supported")
        del base._parameters["weight"]
        self.base = base
        self.register_buffer("indices", record["indices"].detach().clone())
        self.codebook = nn.Parameter(record["codebook"].detach().clone())

    def weight(self):
        return self.codebook[self.indices.long()]

    def forward(self, inputs):
        if isinstance(self.base, nn.Conv2d):
            return self.base._conv_forward(inputs, self.weight(), self.base.bias)
        return F.linear(inputs, self.weight(), self.base.bias)


def attach_codebooks(model, encoding):
    """Replace every clustered layer by a shared-weight layer built from the encoding."""
    names = eligible_weight_names(model)
    require(set(names) == set(encoding["clustered_tensors"]), "Encoding does not cover exactly the model's Conv2d/Linear weights")
    layers = {}
    for name in names:
        module_name = name.removesuffix(".weight")
        parent_name, _, child = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = CodebookLayer(getattr(parent, child), encoding["clustered_tensors"][name])
        setattr(parent, child, layer)
        layers[module_name] = layer
    return layers


def model_key(layers, name):
    """Map an original state name (e.g. fc.bias) to the codebook model's name (fc.base.bias)."""
    module_name, _, leaf = name.rpartition(".")
    return f"{module_name}.base.{leaf}" if module_name in layers else name


def effective_state(model, layers):
    """Dense state under the original names: weights as codebook[indices], everything else as is."""
    state = {}
    for name, value in model.state_dict().items():
        module_name, _, leaf = name.rpartition(".")
        if module_name in layers and leaf in ("indices", "codebook"):
            continue
        owner = module_name.removesuffix(".base") if module_name.endswith(".base") and module_name.removesuffix(".base") in layers else None
        state[f"{owner}.{leaf}" if owner else name] = value.detach().cpu().clone()
    with torch.no_grad():
        for module_name, layer in layers.items():
            state[f"{module_name}.weight"] = layer.weight().detach().cpu().clone()
    return state


def set_trainable(model, layers, variant):
    """Freeze everything, then enable exactly the declared parameters; return their names."""
    model.requires_grad_(False)
    names = []
    for module_name, layer in layers.items():
        layer.codebook.requires_grad_(True)
        names.append(f"{module_name}.codebook")
    if variant["train_batchnorm_affine"]:
        for module_name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for leaf in ("weight", "bias"):
                    getattr(module, leaf).requires_grad_(True)
                    names.append(f"{module_name}.{leaf}")
    if variant["train_fc_bias"]:
        for module_name, layer in layers.items():
            if isinstance(layer.base, nn.Linear) and layer.base.bias is not None:
                layer.base.bias.requires_grad_(True)
                names.append(f"{module_name}.base.bias")
    require({name for name, parameter in model.named_parameters() if parameter.requires_grad} == set(names), "Trainable set differs from the declaration")
    return sorted(names)


def frozen_snapshot(model):
    """Every tensor that must not change: frozen parameters, indices, and non-BatchNorm buffers."""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name not in trainable and not name.endswith(BN_BUFFERS)}


def assert_frozen(model, snapshot, trainable):
    current = {name: value.detach().cpu() for name, value in model.state_dict().items() if name in snapshot}
    require_state_equal(current, snapshot, "Codebook fine-tuning frozen state")
    require({name for name, parameter in model.named_parameters() if parameter.requires_grad} == set(trainable), "Trainable set changed")
    require(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in trainable), "A frozen parameter received a gradient")


def train_epoch(model, loader, optimizer, device, max_grad_norm):
    # train(): BatchNorm uses batch statistics and updates its running estimates.
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    count, correct, total = 0, 0, 0.0
    started = time.monotonic()
    for batch, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        require(math.isfinite(loss.item()), "Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        count += targets.numel()
        correct += (logits.detach().argmax(1) == targets).sum().item()
        total += loss.item() * targets.numel()
        if batch % 100 == 0 or batch == len(loader):
            print(json.dumps({"event": "codebook_training_progress", "batch": batch, "batches": len(loader), "accuracy": correct / count,
                              "cross_entropy_loss": total / count, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    model.eval()
    require(count > 0, "Empty training split")
    return {"num_examples": count, "num_correct": correct, "accuracy": correct / count, "cross_entropy_loss": total / count}


def codebook_order(codebooks):
    """Strict ordering per tensor, with adjacent violations and all inverted pairs counted."""
    report = {}
    for name, codebook in codebooks.items():
        values = codebook.detach().cpu().double()
        gaps = values[1:] - values[:-1]
        inverted = int((values[:, None] >= values[None, :]).triu(1).sum().item())
        report[name] = {"strictly_increasing": bool((gaps > 0).all().item()), "adjacent_violations": int((gaps <= 0).sum().item()),
                        "inverted_pairs": inverted, "min_adjacent_gap": gaps.min().item() if gaps.numel() else None}
    return report


def build_encoding(source, codebooks, untouched_state, metadata):
    """Step 4 format with identical indices and mappings, new codebooks and untouched state."""
    order = codebook_order(codebooks)
    violations = sorted(name for name, item in order.items() if not item["strictly_increasing"])
    require(not violations, f"Codebooks lost strict order; not saving an encoding: {violations}")
    require(set(codebooks) == set(source["clustered_tensors"]) and set(untouched_state) == set(source["untouched_state"]), "Encoding layout differs from the source")
    records = {}
    for name, record in source["clustered_tensors"].items():
        codebook = codebooks[name].detach().cpu().contiguous().clone()
        require(codebook.dtype == record["codebook"].dtype and codebook.shape == record["codebook"].shape and torch.isfinite(codebook).all(), f"Invalid retrained codebook: {name}")
        records[name] = {**{key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) for key, value in record.items()}, "codebook": codebook}
    for name, value in untouched_state.items():
        require(value.dtype == source["untouched_state"][name].dtype and value.shape == source["untouched_state"][name].shape, f"Untouched tensor layout differs: {name}")
    metadata_keys = {key: copy.deepcopy(value) for key, value in source.items() if key not in ("clustered_tensors", "untouched_state")}
    return {**metadata_keys, **metadata, "format_version": 1, "clustered_tensors": records,
            "untouched_state": {name: value.detach().cpu().clone() for name, value in untouched_state.items()}}


def verify_same_cells(encoding, source):
    """Indices, mappings, shapes, and dtypes must equal the source bit for bit."""
    require(set(encoding["clustered_tensors"]) == set(source["clustered_tensors"]) and set(encoding["untouched_state"]) == set(source["untouched_state"]), "Tensor sets differ")
    for name, record in source["clustered_tensors"].items():
        other = encoding["clustered_tensors"][name]
        validate_record(other)
        require(other["shape"] == record["shape"] and other["dtype"] == record["dtype"], f"Layout differs: {name}")
        for key in ("indices", "index_to_level", "level_to_index"):
            require(other[key].dtype == record[key].dtype and torch.equal(other[key], record[key]), f"Stored cells differ: {name}.{key}")
    return len(source["clustered_tensors"])


def codebook_changes(encoding, source):
    changes = {}
    for name, record in source["clustered_tensors"].items():
        old, new = record["codebook"].double(), encoding["clustered_tensors"][name]["codebook"].double()
        counts = torch.bincount(record["indices"].reshape(-1).long(), minlength=old.numel()).double()
        weight_change = ((new - old).square() * counts).sum().sqrt() / (old.square() * counts).sum().sqrt()
        changes[name] = {"codebook_size": old.numel(), "old_codebook": old.tolist(), "new_codebook": new.tolist(), "max_abs_change": (new - old).abs().max().item(),
                         "mean_abs_change": (new - old).abs().mean().item(), "relative_codebook_change": ((new - old).norm() / old.norm()).item(),
                         "relative_weight_change": weight_change.item(), "old_min_adjacent_gap": (old[1:] - old[:-1]).min().item(), "new_min_adjacent_gap": (new[1:] - new[:-1]).min().item()}
    return changes


@dataclass
class CodebookFinetuneConfig:
    run_id: str = "resnet18-cifar10-codebook-finetune-k16-seed42"
    variants: list = field(default_factory=lambda: [
        {"variant_id": "codebook-bnstats", "label": "codebook+BNstats", "train_batchnorm_affine": False, "train_fc_bias": False},
        {"variant_id": "codebook-bnaffine", "label": "codebook+BNaffine", "train_batchnorm_affine": True, "train_fc_bias": True}])
    source_sweep_results: str = "results/resnet18-cifar10-fault-sweep-seed42"
    source_bn_results: str = "results/resnet18-cifar10-bn-recalibration-seed42"
    expected_clean_sha256: str = "7a9d200cbce14c01ddadb68c3a94ce8a9a7990a4cfcdfb7d9f5ae4bdfac47439"
    expected_encoding_sha256: str = "67f9169b1d379e3b4317b72902985efcc2c849df80cd59a8bfad4b57f2793477"
    epochs: int = 5
    batch_size: int = 128
    evaluation_batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
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
        if not isinstance(self.variants, list) or not self.variants or any(not isinstance(item, dict) or set(item) != {"variant_id", "label", "train_batchnorm_affine", "train_fc_bias"} for item in self.variants):
            raise ValueError("Each variant declares exactly variant_id, label, train_batchnorm_affine, and train_fc_bias")
        if len({item["variant_id"] for item in self.variants}) != len(self.variants) or any(not re.fullmatch(r"[a-z0-9][a-z0-9-]*", str(item["variant_id"])) for item in self.variants):
            raise ValueError("variant_id values must be distinct simple names")
        if any(not isinstance(item[key], bool) for item in self.variants for key in ("train_batchnorm_affine", "train_fc_bias")):
            raise ValueError("Variant switches must be booleans")
        for name in ("expected_clean_sha256", "expected_encoding_sha256"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 digest")
        for name, minimum in (("epochs", 1), ("batch_size", 1), ("evaluation_batch_size", 1), ("cpu_threads", 1), ("num_workers", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, positive in (("learning_rate", True), ("weight_decay", False), ("max_grad_norm", True)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"Invalid {name}")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def prepare_codebook_sources(root, config):
    """Audited Step 3/4 sources (pinned hashes) and the Step 9b clustered BatchNorm statistics."""
    sweep_config = SweepConfig(**json.loads((root / config.source_sweep_results / "config.json").read_text()))
    encoding, clean, metadata, split, clean_metrics, clustered_metrics, paths, hashes = prepare_sweep_sources(root, sweep_config)
    require(hashes["clean"] == config.expected_clean_sha256 and hashes["encoding"] == config.expected_encoding_sha256, "Clean checkpoint or Step 4 encoding differs from the pinned hash")
    order = codebook_order({name: record["codebook"] for name, record in encoding["clustered_tensors"].items()})
    require(all(item["strictly_increasing"] for item in order.values()), "Step 4 codebooks are not strictly increasing")
    bn = root / config.source_bn_results
    require(json.loads((bn / "status.json").read_text())["status"] == "complete" and json.loads((bn / "verification.json").read_text())["status"] == "passed", "BatchNorm baseline is incomplete")
    record = json.loads((bn / "conditions" / "clustered.json").read_text())
    paths["bn_clustered_record"], hashes["bn_clustered_record"] = bn / "conditions" / "clustered.json", sha256_file(bn / "conditions" / "clustered.json")
    paths["bn_clustered_statistics"] = root / record["batchnorm_statistics"]
    hashes["bn_clustered_statistics"] = sha256_file(paths["bn_clustered_statistics"])
    require(hashes["bn_clustered_statistics"] == record["batchnorm_statistics_sha256"], "Saved BatchNorm statistics changed")
    statistics = torch.load(paths["bn_clustered_statistics"], map_location="cpu", weights_only=True)["statistics"]
    return {"encoding": encoding, "clean_state": {name: value.detach().cpu().clone() for name, value in clean.state_dict().items()}, "metadata": metadata, "split": split,
            "clean_metrics": {name: clean_metrics[name] for name in SPLITS}, "clustered_metrics": {name: clustered_metrics["measurements"][name]["clustered_no_faults"] for name in SPLITS},
            "bn_statistics": statistics, "bn_metrics": record["after"], "paths": paths, "hashes": hashes}


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def variant_payload(layers, model, variant, epoch, validation, config, source_info):
    state = effective_state(model, layers)
    return {"format_version": 1, "kind": "codebook_finetune_state", "variant": variant, "epoch": epoch, "validation_metrics": validation, "config": asdict(config), **source_info,
            "codebooks": {f"{name}.weight": layer.codebook.detach().cpu().clone() for name, layer in layers.items()},
            "untouched_state": {name: value for name, value in state.items() if name not in {f"{module}.weight" for module in layers}}}


def load_variant_payload(model, layers, payload):
    with torch.no_grad():
        for name, value in payload["codebooks"].items():
            layers[name.removesuffix(".weight")].codebook.copy_(value)
        named = dict(model.named_parameters()) | dict(model.named_buffers())
        for name, value in payload["untouched_state"].items():
            named[model_key(layers, name)].copy_(value)


def train_variant(config, variant, sources, loaders, device, checkpoint_dir, source_info):
    """Train one variant from the Step 4 model; select on validation only; return the selected payload."""
    seed_everything(config.seed, device)
    model = build_resnet18(sources["metadata"]["num_classes"], pretrained=False)
    model.load_state_dict(decode_state(sources["encoding"]), strict=True)
    layers = attach_codebooks(model, sources["encoding"])
    trainable = set_trainable(model, layers, variant)
    model.to(device).eval()
    snapshot = frozen_snapshot(model)
    original_indices = {name: layer.indices.detach().cpu().clone() for name, layer in layers.items()}
    counts = {"trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "trainable_tensors": len(trainable),
              "codebook_values": sum(layer.codebook.numel() for layer in layers.values())}
    directory = checkpoint_dir / variant["variant_id"]
    directory.mkdir(parents=True)
    initial = evaluate_classifier(model, loaders["validation"], device)
    require(initial["num_correct"] == sources["clustered_metrics"]["validation"]["num_correct"], "Epoch-0 model differs from the Step 4 clustered model")
    best_key, best_epoch = (initial["accuracy"], -initial["cross_entropy_loss"]), 0
    save_torch(directory / "best.pt", variant_payload(layers, model, variant, 0, initial, config, source_info))
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history = [{"epoch": 0, "validation": initial, "note": "Step 4 model with its original BatchNorm statistics"}]
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        learning_rate = optimizer.param_groups[0]["lr"]
        print(json.dumps({"event": "codebook_epoch_start", "variant": variant["variant_id"], "epoch": epoch, "learning_rate": learning_rate}), flush=True)
        training = train_epoch(model, loaders["train"], optimizer, device, config.max_grad_norm)
        optimizer.zero_grad(set_to_none=True)
        assert_frozen(model, snapshot, trainable)
        require(all(torch.equal(layer.indices.detach().cpu(), original_indices[name]) for name, layer in layers.items()), "Indices changed during training")
        validation = evaluate_classifier(model, loaders["validation"], device)
        key = (validation["accuracy"], -validation["cross_entropy_loss"])
        payload = variant_payload(layers, model, variant, epoch, validation, config, source_info)
        if key > best_key:
            best_key, best_epoch = key, epoch
            save_torch(directory / "best.pt", payload)
        save_torch(directory / "last.pt", payload)
        order = codebook_order({name: layer.codebook for name, layer in layers.items()})
        history.append({"epoch": epoch, "learning_rate": learning_rate, "training": training, "validation": validation, "frozen_state_verified": True, "indices_unchanged": True,
                        "codebooks_strictly_increasing": all(item["strictly_increasing"] for item in order.values()), "seconds": time.monotonic() - started})
        scheduler.step()
        print(json.dumps({"event": "codebook_epoch_complete", "variant": variant["variant_id"], **history[-1], "selected_epoch": best_epoch}), flush=True)
    selected = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    require(selected["epoch"] == best_epoch, "Selected checkpoint differs from the selection")
    return {"variant": variant, "selected_epoch": best_epoch, "selected": selected, "history": history, "trainable": trainable, "counts": counts}


def evaluate_state(state, sources, loaders, device):
    model = build_resnet18(sources["metadata"]["num_classes"], pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return {name: evaluate_classifier(model, loaders[name], device) for name in SPLITS}


def run_codebook_finetune(config, project_root):
    config.validate()
    root = Path(project_root)
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    results_dir, checkpoint_dir = root / config.results_root / config.run_id, root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run directories already exist; choose a new run_id")
    sources = prepare_codebook_sources(root, config)
    encoding, metadata, split = sources["encoding"], sources["metadata"], sources["split"]
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_codebook_finetune.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        source_code = {str(path.relative_to(root)): sha256_file(path) for path in files}
        write_json(results_dir / "source_sha256.json", source_code)
        write_json(results_dir / "splits.json", split)
        write_json(results_dir / "preprocessing.json", metadata["preprocessing"])
        environment = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "cpu_threads": config.cpu_threads,
                       "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy")},
                       "reproducibility": "Seeded data order and augmentation; codebook gradients are scatter-add sums whose float order may differ on GPU backends."}
        write_json(results_dir / "environment.json", environment)
        source_info = {"source_encoding_sha256": sources["hashes"]["encoding"], "source_encoding_fingerprint": encoding_fingerprint(encoding),
                       "source_clean_sha256": sources["hashes"]["clean"], "split_sha256": metadata["split_sha256"]}
        train_transform, eval_transform = cifar10_transforms(metadata["preprocessing"]["input_size"][-1])
        training_data = CIFAR10(root / config.data_root, train=True, transform=train_transform, download=False)
        evaluation_data = CIFAR10(root / config.data_root, train=True, transform=eval_transform, download=False)
        require(training_data.classes == evaluation_data.classes == metadata["classes"] and len(training_data) == 50000, "CIFAR-10 class order or size changed")
        options = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0, "worker_init_fn": seed_worker, "pin_memory": device.type == "cuda"}
        loaders = {"train": DataLoader(Subset(training_data, split["train"]), batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed), **options),
                   "validation": DataLoader(Subset(evaluation_data, split["validation"]), batch_size=config.evaluation_batch_size, shuffle=False, **options)}
        results = {}
        for variant in config.variants:
            result = train_variant(config, variant, sources, loaders, device, checkpoint_dir, source_info)
            results[variant["variant_id"]] = result
            write_json(results_dir / f"history_{variant['variant_id']}.json", result["history"])
            write_json(results_dir / "status.json", {"status": "running", "completed_variants": list(results)})
        write_json(results_dir / "trainable.json", {key: {"label": item["variant"]["label"], **item["counts"], "trainable_names": item["trainable"], "selected_epoch": item["selected_epoch"]}
                                                    for key, item in results.items()})
        order = {key: codebook_order(item["selected"]["codebooks"]) for key, item in results.items()}
        violations = {key: {name: item for name, item in report.items() if not item["strictly_increasing"]} for key, report in order.items()}
        write_json(results_dir / "codebook_order.json", {"selected": order, "violations": violations})
        if any(violations.values()):
            # Keep the run and its trained states; never re-sort or remap cells, and save no baseline encoding.
            summary = {key: {name: {k: item[k] for k in ("adjacent_violations", "inverted_pairs")} for name, item in found.items()} for key, found in violations.items() if found}
            write_json(results_dir / "status.json", {"status": "stopped_codebook_order_violation", "violations": summary, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
            print(json.dumps({"event": "codebook_order_violation", "violations": summary}), flush=True)
            return {"status": "stopped_codebook_order_violation", "violations": summary}
        encodings = {}
        for key, item in results.items():
            selected = item["selected"]
            metadata_update = {"representation": "per-tensor scalar codebooks and indices; codebooks fine-tuned with indices fixed; no faults or adapters",
                               "scope": "all Conv2d/Linear weight tensors; indices and mappings identical to the Step 4 encoding; BatchNorm statistics re-estimated during fine-tuning",
                               "source_encoding": str(sources["paths"]["encoding"].relative_to(root)), **source_info,
                               "codebook_finetune": {"run_id": config.run_id, "variant": item["variant"], "selected_epoch": item["selected_epoch"], "config": asdict(config),
                                                     "trainable_parameters": item["counts"]["trainable_parameters"]},
                               "faults_injected": 0, "environment": environment}
            new = build_encoding(encoding, selected["codebooks"], selected["untouched_state"], metadata_update)
            path = checkpoint_dir / key / "encoding.pt"
            save_torch(path, new)
            reloaded = torch.load(path, map_location="cpu", weights_only=True)
            cells = verify_same_cells(reloaded, encoding)
            decoded = decode_state(reloaded)
            # The saved encoding must reconstruct exactly the weights the selected model used.
            model = build_resnet18(metadata["num_classes"], pretrained=False)
            model.load_state_dict(decode_state(encoding), strict=True)
            layers = attach_codebooks(model, encoding)
            load_variant_payload(model, layers, selected)
            require_state_equal(decoded, effective_state(model, layers), f"Saved encoding reconstruction ({key})")
            for name, value in encoding["untouched_state"].items():
                if not name.endswith(BN_BUFFERS) and model_key(layers, name) not in item["trainable"]:
                    require(torch.equal(reloaded["untouched_state"][name], value), f"Frozen state changed: {name}")
            dense_path = checkpoint_dir / key / "dense.pt"
            save_torch(dense_path, {"model_state_dict": decoded, **{k: v for k, v in reloaded.items() if k not in ("clustered_tensors", "untouched_state", "format_version")},
                                    "source_codebook_encoding": str(path.relative_to(root))})
            changes = codebook_changes(reloaded, encoding)
            write_json(results_dir / f"codebook_changes_{key}.json", changes)
            encodings[key] = {"encoding": str(path.relative_to(root)), "encoding_sha256": sha256_file(path), "fingerprint": encoding_fingerprint(reloaded),
                              "dense": str(dense_path.relative_to(root)), "dense_sha256": sha256_file(dense_path), "cells_identical_to_step4": cells,
                              "untouched_state_changed": sorted(name for name in encoding["untouched_state"] if not torch.equal(reloaded["untouched_state"][name], encoding["untouched_state"][name])),
                              "max_relative_weight_change": max(item["relative_weight_change"] for item in changes.values()),
                              "max_abs_codebook_change": max(item["max_abs_change"] for item in changes.values())}
        write_json(results_dir / "encodings.json", encodings)
        # Test data is loaded only after every variant's selection and encoding are final.
        test_data = CIFAR10(root / config.data_root, train=False, transform=eval_transform, download=False)
        require(test_data.classes == metadata["classes"] and len(test_data) == 10000, "CIFAR-10 test class order or size changed")
        loaders["test"] = DataLoader(Subset(test_data, split["test"]["indices"]), batch_size=config.evaluation_batch_size, shuffle=False, num_workers=config.num_workers)
        clustered_state = decode_state(encoding)
        recalibrated = {**clustered_state, **{name: value for name, value in sources["bn_statistics"].items()}}
        require(set(recalibrated) == set(clustered_state), "Saved BatchNorm statistics do not fit the clustered model")
        models = {"clean_step3": sources["clean_state"], "clustered_step4": clustered_state, "clustered_bn_recalibration_step9b": recalibrated}
        models.update({key: decode_state(torch.load(root / item["encoding"], map_location="cpu", weights_only=True)) for key, item in encodings.items()})
        evaluation = {}
        for name, state in models.items():
            print(json.dumps({"event": "paired_evaluation_start", "model": name}), flush=True)
            evaluation[name] = evaluate_state(state, sources, loaders, device)
        saved = {"clean_step3": sources["clean_metrics"], "clustered_step4": sources["clustered_metrics"], "clustered_bn_recalibration_step9b": sources["bn_metrics"]}
        for name, reference in saved.items():
            require(all(evaluation[name][split]["num_correct"] == reference[split]["num_correct"] for split in SPLITS), f"Re-evaluated reference differs from its saved counts: {name}")
        clean = evaluation["clean_step3"]
        rows = [{"model": name, "split": split, **values[split], "gap_to_clean_percentage_points": 100 * (clean[split]["accuracy"] - values[split]["accuracy"])}
                for name, values in evaluation.items() for split in SPLITS]
        write_json(results_dir / "paired_evaluation.json", {"models": evaluation, "rows": rows, "selected_epochs": {key: item["selected_epoch"] for key, item in results.items()}})
        with (results_dir / "paired_evaluation.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVALUATION_FIELDS)
            writer.writeheader()
            writer.writerows({name: row[name] for name in EVALUATION_FIELDS} for row in rows)
        accepted = compare_source_hashes(root, source_code, "Source code changed during the run")
        require(all(sha256_file(path) == sources["hashes"][name] for name, path in sources["paths"].items()), "An input artifact changed during the run")
        manifest = {"run_id": config.run_id, "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": sources["hashes"][name]} for name, path in sources["paths"].items()},
                    **source_info, "method": "weight = codebook[indices] in the forward pass; only declared parameters train; BatchNorm in train() mode re-estimates running statistics (momentum 0.1).",
                    "selection_rule": "Highest validation accuracy, then lowest validation cross-entropy; earlier epoch on exact ties; epoch 0 (Step 4 model, original BatchNorm statistics) competes.",
                    "test_policy": "Test evaluated once per model after both variants were selected and saved.",
                    "ordering": "Codebooks checked for strict order after selection; never re-sorted or remapped.", "encodings": encodings,
                    "accepted_source_hash_changes": accepted, "environment": environment, "elapsed_seconds": time.monotonic() - started}
        write_json(results_dir / "manifest.json", manifest)
        write_json(results_dir / "verification.json", verify_codebook_finetune(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "selected_epochs": {key: item["selected_epoch"] for key, item in results.items()},
                                                  "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"event": "codebook_finetune_complete", "run_id": config.run_id, "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
        return {"status": "complete", "evaluation": evaluation, "selected_epochs": {key: item["selected_epoch"] for key, item in results.items()}, "encodings": encodings}
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": repr(error)})
        raise


def verify_codebook_finetune(root, config):
    """Re-read the saved run: selections, ordering, cell identity, reconstruction, and records."""
    root = Path(root)
    results_dir = root / config.results_root / config.run_id
    manifest = json.loads((results_dir / "manifest.json").read_text())
    source = torch.load(root / manifest["source_artifacts"]["encoding"]["path"], map_location="cpu", weights_only=True)
    require(sha256_file(root / manifest["source_artifacts"]["encoding"]["path"]) == config.expected_encoding_sha256, "Source encoding changed")
    paired = json.loads((results_dir / "paired_evaluation.json").read_text())
    checks = {}
    for variant in config.variants:
        key = variant["variant_id"]
        history = json.loads((results_dir / f"history_{key}.json").read_text())
        selected, best = 0, (history[0]["validation"]["accuracy"], -history[0]["validation"]["cross_entropy_loss"])
        for item in history[1:]:
            candidate = (item["validation"]["accuracy"], -item["validation"]["cross_entropy_loss"])
            if candidate > best:
                selected, best = item["epoch"], candidate
        require([item["epoch"] for item in history] == list(range(config.epochs + 1)) and all(item["frozen_state_verified"] and item["indices_unchanged"] for item in history[1:]), f"History incomplete: {key}")
        require(selected == paired["selected_epochs"][key], f"Selected epoch is not the validation optimum: {key}")
        payload = torch.load(root / config.checkpoint_root / config.run_id / key / "best.pt", map_location="cpu", weights_only=True)
        require(payload["epoch"] == selected and payload["validation_metrics"] == history[selected]["validation"], f"Selected checkpoint differs: {key}")
        item = manifest["encodings"][key]
        require(sha256_file(root / item["encoding"]) == item["encoding_sha256"] and sha256_file(root / item["dense"]) == item["dense_sha256"], f"Saved artifacts changed: {key}")
        encoding = torch.load(root / item["encoding"], map_location="cpu", weights_only=True)
        verify_same_cells(encoding, source)
        require(encoding_fingerprint(encoding) == item["fingerprint"], f"Fingerprint differs: {key}")
        require(all(report["strictly_increasing"] for report in codebook_order({name: record["codebook"] for name, record in encoding["clustered_tensors"].items()}).values()), f"Saved codebooks unsorted: {key}")
        require(all(torch.equal(encoding["clustered_tensors"][name]["codebook"], value) for name, value in payload["codebooks"].items()),
                f"Saved encoding codebooks differ from the selected checkpoint: {key}")
        require_state_equal(torch.load(root / item["dense"], map_location="cpu", weights_only=True)["model_state_dict"], decode_state(encoding), f"Dense reconstruction ({key})")
        checks[key] = {"selected_epoch": selected, "cells_identical_to_step4": True, "strictly_sorted": True, "fingerprint": item["fingerprint"], "dense_matches_encoding": True}
    for name, artifact in manifest["source_artifacts"].items():
        require(sha256_file(root / artifact["path"]) == artifact["sha256"], f"Input artifact changed: {name}")
    with (results_dir / "paired_evaluation.csv").open(newline="") as stream:
        require(len(list(csv.DictReader(stream))) == len(paired["rows"]), "CSV and JSON evaluations disagree")
    accepted = compare_source_hashes(root, json.loads((results_dir / "source_sha256.json").read_text()), "Source code changed after the run")
    return {"status": "passed", "variants": checks, "selection_verified_using_validation_only": True, "reference_counts_match_saved": True,
            "source_artifacts_verified": len(manifest["source_artifacts"]), "accepted_source_hash_changes": accepted}


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_cifar10_codebook_finetune.json"))
    args = parser.parse_args()
    run_codebook_finetune(CodebookFinetuneConfig(**json.loads((Path(project_root) / args.config).read_text())), project_root)
