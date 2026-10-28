#!/usr/bin/env python3
"""Analyze ResNet compressed-weight storage from the *actual exported* model shapes.

This is an analytic storage / minimum-transfer report, NOT a SCALE-Sim memory
extension and NOT a latency model. It should not be reported as cycle counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def positive_integer(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def compute(manifest: dict, k: int, index_bits: int, weight_bytes: int, centroid_bytes: int):
    layers = manifest["layers"]
    if not layers:
        raise ValueError("Manifest has no layers")
    results = []
    for layer in layers:
        name = layer["name"]
        base, a, b = layer["base"], layer["lora_A"], layer["lora_B"]
        base_elems = int(base["N"]) * int(base["K"])
        shape = base.get("weight_shape")
        if shape and math.prod(shape) != base_elems:
            raise ValueError(f"Weight shape mismatch in {name}: {shape} vs GEMM {base_elems}")
        a_elems = int(a["N"]) * int(a["K"])
        b_elems = int(b["N"]) * int(b["K"])
        if not (base_elems > 0 and a_elems > 0 and b_elems > 0):
            raise ValueError(f"Invalid dimensions for {name}")
        if k > (1 << index_bits):
            raise ValueError(f"k={k} cannot be encoded in {index_bits} index bits")
        original_bytes = base_elems * weight_bytes
        indices_bytes = math.ceil(base_elems * index_bits / 8)
        codebook_bytes = k * centroid_bytes
        compressed_bytes = indices_bytes + codebook_bytes
        lora_bytes = (a_elems + b_elems) * weight_bytes
        results.append({
            "layer": name,
            "num_base_weights": base_elems,
            "num_lora_weights": a_elems + b_elems,
            "original_base_bytes": original_bytes,
            "index_bytes": indices_bytes,
            "codebook_bytes": codebook_bytes,
            "compressed_base_bytes": compressed_bytes,
            "lora_bytes": lora_bytes,
            "clustered_plus_lora_bytes": compressed_bytes + lora_bytes,
            "base_compression_ratio": round(original_bytes / compressed_bytes, 5),
        })
    keys = ["num_base_weights", "num_lora_weights", "original_base_bytes", "index_bytes", "codebook_bytes", "compressed_base_bytes", "lora_bytes", "clustered_plus_lora_bytes"]
    total = {x: sum(r[x] for r in results) for x in keys}
    total["base_compression_ratio"] = round(total["original_base_bytes"] / total["compressed_base_bytes"], 5)
    total["layer_count"] = len(results)
    total["input_image_size"] = manifest.get("image_size")
    total["assumptions"] = {
        "k": k,
        "index_bits": index_bits,
        "weight_bytes": weight_bytes,
        "centroid_bytes": centroid_bytes,
        "description": "Storage footprint, assuming one codebook per modeled Conv/Linear weight tensor, no biases, no BN, no activations, no metadata/alignment overhead. Values are NOT measured DRAM accesses or cycles.",
    }
    return results, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("hardware_sim/scalesim/workloads/manifest.json"))
    p.add_argument("--out", type=Path, default=Path("results/hardware_sim/scalesim"))
    p.add_argument("--clusters", type=positive_integer, default=16)
    p.add_argument("--index-bits", type=positive_integer, default=8, help="8 for software uint8 indices; 4 for an ideal packed 16-cluster design")
    p.add_argument("--weight-bytes", type=positive_integer, default=4, help="Assumed dense/LoRA weight bytes; 4=FP32")
    p.add_argument("--centroid-bytes", type=positive_integer, default=4, help="Assumed codebook centroid bytes; 4=FP32")
    p.add_argument("--bandwidth-bytes-per-cycle", type=positive_integer, default=None,
                   help="Optional transfer lower bound, NOT a SCALE-Sim timing model")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows, summary = compute(manifest, args.clusters, args.index_bits, args.weight_bytes, args.centroid_bytes)
    if args.bandwidth_bytes_per_cycle:
        bw = args.bandwidth_bytes_per_cycle
        summary["transfer_lower_bound_cycles"] = {
            "bandwidth_bytes_per_cycle": bw,
            "dense_base_only": math.ceil(summary["original_base_bytes"] / bw),
            "compressed_base_plus_lora": math.ceil(summary["clustered_plus_lora_bytes"] / bw),
            "warning": "Arithmetic bandwidth lower bounds only; not modeled execution cycles or measured DRAM traffic.",
        }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "cluster_storage_by_layer.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "cluster_storage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Layers analyzed: {summary['layer_count']}")
    print(f"Original base weights: {summary['num_base_weights']:,} weights / {summary['original_base_bytes']:,} bytes")
    print(f"Clustered base: {summary['index_bytes']:,} index bytes + {summary['codebook_bytes']:,} codebook bytes")
    print(f"LoRA: {summary['num_lora_weights']:,} weights / {summary['lora_bytes']:,} bytes")
    print(f"Clustered+LoRA storage: {summary['clustered_plus_lora_bytes']:,} bytes")
    print(f"Base-only compression ratio: {summary['base_compression_ratio']:.3f}x")
    print(f"Wrote {args.out / 'cluster_storage_by_layer.csv'}")
    print(f"Wrote {args.out / 'cluster_storage_summary.json'}")
    print("NOT cycle-accurate: these numbers exclude caching, reuse, decoding, and physical memory timing.")


if __name__ == "__main__":
    main()
