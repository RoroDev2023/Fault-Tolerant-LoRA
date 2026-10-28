#!/usr/bin/env python3
"""Export GEMM approximations for the actual project ResNet18 + unmerged LoRA.

This imports the research project's existing model and LoRA code. It requires no
checkpoint, dataset, downloads, clustering, or training. It is specifically an
execution-shape extractor for one image of shape (1, 3, 96, 96).

Reports are ONLY GEMM-equivalent convolutions (im2col assumption). Non-GEMM
operators, codebook decoding, memory faults, reconstruction and branch scheduling
are not represented in these files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def export(outdir: Path, image_size: int = 96, rank: int = 4, alpha: float = 4.0):
    import torch
    from torch import nn
    from fault_lora.models.resnet import build_resnet18
    from fault_lora.models.lora import LoRALayer, attach_lora

    if image_size <= 0 or rank <= 0:
        raise ValueError("image_size and rank must be positive")

    # No pretrained weights download. We need real tensor geometry, not values.
    model = build_resnet18(num_classes=10, pretrained=False).cpu().eval()
    selected = attach_lora(model, rank=rank, alpha=alpha)
    # For the current branch this MUST cover all 21 Conv2d/Linear layers.
    expected = {x["name"] for x in selected}
    if len(expected) != 21:
        raise RuntimeError(f"Expected 21 LoRA sites, found {len(expected)}; inspect model changes")

    observed = {}
    events = []
    handles = []

    def make_hook(parent_name, role, layer):
        def hook(_module, inp, output):
            if parent_name not in expected:
                raise RuntimeError("Unrecognized layer: " + parent_name)
            if role in observed.get(parent_name, {}):
                raise RuntimeError("Repeated layer execution not supported: " + parent_name)
            x = inp[0]
            if isinstance(layer, nn.Conv2d):
                if layer.groups != 1:
                    raise RuntimeError("Grouped convolution not supported: " + parent_name)
                b, cin, _, _ = x.shape
                batch, cout, hout, wout = output.shape
                assert b == batch and cin == layer.in_channels and cout == layer.out_channels
                h, w = layer.kernel_size
                M, N, K = int(batch * hout * wout), int(cout), int(cin * h * w)
                meta = {"op": "conv2d", "input_shape": list(x.shape),
                        "output_shape": list(output.shape), "kernel": list(layer.kernel_size),
                        "stride": list(layer.stride), "padding": list(layer.padding),
                        "dilation": list(layer.dilation), "groups": layer.groups}
            elif isinstance(layer, nn.Linear):
                M = int(math.prod(x.shape[:-1]))
                K, N = int(layer.in_features), int(layer.out_features)
                meta = {"op": "linear", "input_shape": list(x.shape),
                        "output_shape": list(output.shape)}
            else:
                raise RuntimeError("Unsupported operation")
            observed.setdefault(parent_name, {})[role] = {
                "M": M, "N": N, "K": K, "MACs": M * N * K,
                "weight_shape": list(layer.weight.shape), **meta,
            }
            events.append((parent_name, role))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            for role in ("base", "lora_A", "lora_B"):
                sub = getattr(module, role)
                handles.append(sub.register_forward_hook(make_hook(name, role, sub)))

    with torch.inference_mode():
        model(torch.zeros(1, 3, image_size, image_size))
    for handle in handles:
        handle.remove()

    base_order = [name for name, role in events if role == "base"]
    if set(base_order) != expected or len(base_order) != len(expected):
        raise RuntimeError("Incomplete/duplicated base execution; review ResNet branches")
    if any(set(observed[n]) != {"base", "lora_A", "lora_B"} for n in base_order):
        raise RuntimeError("Some LoRA operations weren't observed")
    for name in base_order:
        base, a, b = (observed[name][key] for key in ("base", "lora_A", "lora_B"))
        if b["M"] != a["M"] or b["K"] != rank or a["N"] != rank:
            raise RuntimeError("Unexpected LoRA GEMM shapes: " + name)
        if base["M"] != a["M"] or base["N"] != b["N"] or base["K"] != a["K"]:
            raise RuntimeError("LoRA/base geometry mismatch: " + name)

    outdir.mkdir(parents=True, exist_ok=True)

    def write_gemm(filename, items):
        with (outdir / filename).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Layer Name", " M", " N", " K", ""])
            writer.writerows([[name, dims["M"], dims["N"], dims["K"], ""] for name, dims in items])

    write_gemm("original_gemm.csv", [(name, observed[name]["base"]) for name in base_order])
    write_gemm("lora_gemm.csv", [
        (name + "__" + role, observed[name][role])
        for name in base_order for role in ("lora_A", "lora_B")
    ])
    manifest = {
        "model": "RoroDev Fault-Tolerant-LoRA ResNet18, native ImageNet stem, CIFAR10 classifier",
        "batch_size": 1, "image_size": image_size, "rank": rank, "alpha": alpha,
        "weight_representation_for_future_modeling": "k=16 scalar cluster indices, 1 uint8 index cell/weight in current software",
        "layers": [{"name": name, "base": observed[name]["base"],
                    "lora_A": observed[name]["lora_A"],
                    "lora_B": observed[name]["lora_B"]} for name in base_order],
        "execution_events": [dict(layer=name, role=role) for name, role in events],
        "limitations": ["Conv2d mapped to im2col GEMM, not true convolution memory trace",
                        "Does not simulate BN, ReLU, pooling, elementwise additions or residual branch overlap",
                        "Does not model codebook lookup/reconstruction or faults", 
                        "LoRA sequential cycle summation is not exact end-to-end inference timing"]
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(base_order)} base ops and {len(base_order) * 2} adapter ops -> {outdir}")
    print("First ops:", [(name, observed[name]['base']['M'],observed[name]['base']['N'],observed[name]['base']['K']) for name in base_order[:3]])
    print("Total base MACs:", f"{sum(observed[n]['base']['MACs'] for n in base_order):,}")
    print("Total LoRA MACs:", f"{sum(observed[n][r]['MACs'] for n in base_order for r in ('lora_A','lora_B')):,}")
    print("NOTE: GEMM-only approximation; not compressed-memory or fault timing.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("hardware_sim/scalesim/workloads"))
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=4.)
    args = p.parse_args()
    export(args.out, args.image_size, args.rank, args.alpha)


if __name__ == "__main__":
    main()
