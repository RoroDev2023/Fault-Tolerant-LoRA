#!/usr/bin/env python3
"""Export the project's ResNet-18 inference path as a SCALE-Sim GEMM topology."""

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

from fault_lora.models.resnet import build_resnet18, load_clean_checkpoint


GEMM_HEADER = ("Layer name", "M", "N", "K", "Sparsity")


def collect_gemm_layers(model: nn.Module, *, image_size: int, batch_size: int = 1) -> list[dict[str, int | str]]:
    """Run a shape-only forward pass and map each Conv2d/Linear to GEMM M,N,K."""
    if image_size < 1 or batch_size < 1:
        raise ValueError("image_size and batch_size must be positive")

    layers: list[dict[str, int | str]] = []
    handles = []

    def record(module_name: str, module: nn.Module):
        def hook(_module, inputs, output):
            input_tensor = inputs[0]
            if isinstance(module, nn.Conv2d):
                if module.groups != 1:
                    raise ValueError(f"SCALE-Sim GEMM export currently requires groups=1: {module_name}")
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
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(record(name, module)))

    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(batch_size, 3, image_size, image_size))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    if not layers:
        raise ValueError("Model forward pass contained no Conv2d or Linear layers")
    return layers


def write_gemm_topology(layers: list[dict[str, int | str]], output_path: Path) -> None:
    """Write SCALE-Sim v3's GEMM CSV format (name,M,N,K,sparsity,trailing comma)."""
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
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/resnet18_cifar10.json",
                        help="project baseline config; supplies the model input resolution")
    parser.add_argument("--image-size", type=int, help="override input resolution from --config")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="inference batch size represented in the GEMM dimensions")
    parser.add_argument("--checkpoint", type=Path,
                        help="optional clean ResNet checkpoint; its saved preprocessing resolution is used")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "hardware_sim/scalesim/workloads/original_gemm.csv",
                        help="output SCALE-Sim GEMM topology CSV")
    args = parser.parse_args()

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        model, metadata = load_clean_checkpoint(checkpoint_path)
        checkpoint_size = int(metadata["preprocessing"]["input_size"][-1])
        if args.image_size is not None and args.image_size != checkpoint_size:
            parser.error(f"--image-size {args.image_size} differs from checkpoint preprocessing size {checkpoint_size}")
        image_size = checkpoint_size
        architecture = metadata.get("architecture")
        if architecture != "resnet18":
            parser.error(f"Expected a ResNet-18 checkpoint, got {architecture!r}")
    else:
        config_path = args.config
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config = json.loads(config_path.read_text())
        image_size = args.image_size if args.image_size is not None else int(config["image_size"])
        model = build_resnet18(num_classes=10, pretrained=False)

    layers = collect_gemm_layers(model, image_size=image_size, batch_size=args.batch_size)
    output_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    write_gemm_topology(layers, output_path)
    macs = sum(int(layer["macs"]) for layer in layers)
    manifest_path = output_path.with_name("manifest.json")
    manifest = {
        "model": "torchvision ResNet-18",
        "dataset": "CIFAR-10",
        "input_shape": [args.batch_size, 3, image_size, image_size],
        "batch_size": args.batch_size,
        "topology_format": "SCALE-Sim v3 GEMM: layer_name,M,N,K,sparsity with trailing comma",
        "source_checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "layers": layers,
        "gemm_layer_count": len(layers),
        "total_multiply_accumulates_per_inference_batch": macs,
        "excluded_operations": ["batch normalization", "activation", "pooling", "residual addition"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(layers)} GEMMs ({macs:,} MACs) to {output_path}")
    print(f"Wrote workload metadata to {manifest_path}")


if __name__ == "__main__":
    main()
