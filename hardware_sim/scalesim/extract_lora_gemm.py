#!/usr/bin/env python3
"""Export the configured ResNet LoRA adapter operations as a SCALE-Sim topology."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.models.lora import LoRALayer, attach_lora
from fault_lora.models.resnet import build_resnet18


GEMM_HEADER = ("Layer name", "M", "N", "K", "Sparsity")


def collect_lora_gemm_layers(
    model: nn.Module,
    *,
    image_size: int,
    rank: int,
    target_names: list[str] | None = None,
    batch_size: int = 1,
) -> list[dict[str, int | str]]:
    """Attach LoRA and record the A and B GEMMs for each configured target."""
    if image_size < 1 or batch_size < 1:
        raise ValueError("image_size and batch_size must be positive")

    layout = attach_lora(model, rank=rank, alpha=rank, target_names=target_names)
    layers: list[dict[str, int | str]] = []
    handles = []

    def record(module_name: str, module: nn.Module):
        def hook(_module, inputs, output):
            input_tensor = inputs[0]
            if isinstance(module, nn.Conv2d):
                if module.groups != 1:
                    raise ValueError(f"SCALE-Sim GEMM export requires groups=1: {module_name}")
                m = int(output.shape[0] * output.shape[2] * output.shape[3])
                n = module.out_channels
                k = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                operation = "conv2d"
            else:
                if input_tensor.shape[-1] != module.in_features:
                    raise ValueError(f"Unexpected Linear input shape for {module_name}: {tuple(input_tensor.shape)}")
                m = int(input_tensor.numel() // module.in_features)
                n = module.out_features
                k = module.in_features
                operation = "linear"

            if min(m, n, k) <= 0:
                raise ValueError(f"Invalid GEMM dimensions for {module_name}: M={m}, N={n}, K={k}")
            layers.append({
                "name": module_name,
                "operation": operation,
                "M": m,
                "N": n,
                "K": k,
                "macs": m * n * k,
            })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            for part in ("lora_A", "lora_B"):
                adapter = getattr(module, part)
                handles.append(adapter.register_forward_hook(record(f"{name}.{part}", adapter)))

    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(batch_size, 3, image_size, image_size))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    expected_count = 2 * len(layout)
    if len(layers) != expected_count:
        raise ValueError(f"Expected {expected_count} adapter GEMMs from {len(layout)} targets, recorded {len(layers)}")
    return layers


def write_gemm_topology(layers: list[dict[str, int | str]], output_path: Path) -> None:
    """Write SCALE-Sim v3 GEMM CSV format, including its trailing empty column."""
    if not layers:
        raise ValueError("Cannot write an empty GEMM topology")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow((*GEMM_HEADER, ""))
        for layer in layers:
            writer.writerow((layer["name"], layer["M"], layer["N"], layer["K"], "1:1", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs/resnet18_cifar10.json",
                        help="baseline model config; supplies the input resolution")
    parser.add_argument("--lora-config", type=Path, default=PROJECT_ROOT / "configs/resnet18_cifar10_lora.json",
                        help="LoRA config; supplies rank and optional target layer names")
    parser.add_argument("--image-size", type=int, help="override input resolution from --model-config")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="inference batch size represented in the GEMM dimensions")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "hardware_sim/scalesim/workloads/lora_gemm.csv",
                        help="output SCALE-Sim adapter GEMM topology CSV")
    args = parser.parse_args()

    model_config_path = args.model_config if args.model_config.is_absolute() else PROJECT_ROOT / args.model_config
    lora_config_path = args.lora_config if args.lora_config.is_absolute() else PROJECT_ROOT / args.lora_config
    model_config = json.loads(model_config_path.read_text())
    lora_config = json.loads(lora_config_path.read_text())
    image_size = args.image_size if args.image_size is not None else int(model_config["image_size"])
    rank = int(lora_config["rank"])
    target_names = lora_config.get("target_names")
    if target_names is not None and not isinstance(target_names, list):
        parser.error("target_names in the LoRA config must be null or a list of layer names")

    model = build_resnet18(num_classes=10, pretrained=False)
    layers = collect_lora_gemm_layers(
        model,
        image_size=image_size,
        rank=rank,
        target_names=target_names,
        batch_size=args.batch_size,
    )
    output_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    write_gemm_topology(layers, output_path)
    macs = sum(int(layer["macs"]) for layer in layers)
    manifest_path = output_path.with_name("lora_manifest.json")
    manifest = {
        "model": "torchvision ResNet-18 LoRA adapters",
        "dataset": "CIFAR-10",
        "input_shape": [args.batch_size, 3, image_size, image_size],
        "batch_size": args.batch_size,
        "rank": rank,
        "target_names": target_names,
        "adapter_target_count": len(layers) // 2,
        "gemm_layer_count": len(layers),
        "total_multiply_accumulates_per_inference_batch": macs,
        "topology_format": "SCALE-Sim v3 GEMM: layer_name,M,N,K,sparsity with trailing comma",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(layers)} adapter GEMMs ({macs:,} MACs) to {output_path}")
    print(f"Wrote workload metadata to {manifest_path}")


if __name__ == "__main__":
    main()
