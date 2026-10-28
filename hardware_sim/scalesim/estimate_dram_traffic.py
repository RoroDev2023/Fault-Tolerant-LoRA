#!/usr/bin/env python3
"""SCALE-Sim DRAM access postprocessor for the ResNet compressed-weight study.

Uses ACTUAL SCALE-Sim DETAILED_ACCESS_REPORT.csv element counts, and estimates
hypothetical byte traffic for different weight representations. This script DOES
NOT replace SCALE-Sim's buffers, change cycle counts, or replay memory timing.

DRAM filter read references are assumed to request a scalar weight address.
A compressed, lookup-aware accelerator is assumed to read an index instead,
fetch each layer's codebook once, and decode without modeling delay. This is
an explicit *what-if*, not the current software execution behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def find_report(where: Path) -> Path:
    if where.is_file():
        return where
    matches = list(where.rglob('DETAILED_ACCESS_REPORT.csv'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one DETAILED_ACCESS_REPORT.csv in {where}; got {matches}')
    return matches[0]


def read_counts(path: Path, expected: int):
    with path.open(newline='') as f:
        rows = [{k.strip(): (v or '').strip() for k, v in row.items() if k is not None}
                for row in csv.DictReader(f)]
    if len(rows) != expected:
        raise ValueError(f'{path} has {len(rows)} layers, expected {expected}. Ensure the correct report.')
    output = []
    for i, r in enumerate(rows):
        if int(r['LayerID']) != i:
            raise ValueError(f'Layer ID mismatch: report row {i}: {r["LayerID"]}')
        counts = {}
        for name, key in [('ifmap', 'DRAM IFMAP Reads'),
                          ('filter', 'DRAM Filter Reads'),
                          ('ofmap', 'DRAM OFMAP Writes')]:
            if key not in r:
                raise ValueError(f'Missing column {key} in {path}. Have: {list(r)}')
            count = float(r[key])
            if not count.is_integer() or count < 0:
                raise ValueError(f'Invalid {key}: {r[key]}')
            counts[name] = int(count)
        output.append(counts)
    return output


def analyze(manifest: dict, base: list, adapter: list, k: int,
            index_bits: int, dense_bytes: int, centroid_bytes: int):
    layers = manifest['layers']
    if k > 2 ** index_bits:
        raise ValueError('index bits insufficient for K clusters')
    if len(base) != len(layers) or len(adapter) != 2 * len(layers):
        raise ValueError('Manifest and report layer counts differ')
    per_layer = []
    for i, layer in enumerate(layers):
        name = layer['name']
        weights = int(layer['base']['N']) * int(layer['base']['K'])
        reads = base[i]
        a, b = adapter[2 * i], adapter[2 * i + 1]
        raw_base_filter = reads['filter'] * dense_bytes
        compressed_index = math.ceil(reads['filter'] * index_bits / 8)
        # Codebook preload once per layer; no repeated codebook DRAM misses.
        codebook = k * centroid_bytes
        compressed_filter = compressed_index + codebook
        # Add unchanged input/output traffic plus LoRA traffic under serial-GEMM assumption.
        activation_output = (reads['ifmap'] + reads['ofmap']) * dense_bytes
        adapters = sum((r['ifmap'] + r['filter'] + r['ofmap']) * dense_bytes for r in (a, b))
        total_dense = raw_base_filter + activation_output
        total_compressed_lora = compressed_filter + activation_output + adapters
        per_layer.append(dict(layer=name, base_weights=weights,
            base_filter_dram_reads=reads['filter'],
            original_filter_bytes=raw_base_filter,
            compressed_index_bytes_if_same_access_count=compressed_index,
            codebook_load_bytes=codebook,
            compressed_filter_bytes=compressed_filter,
            base_activation_and_output_bytes=activation_output,
            lora_all_dram_bytes=adapters,
            original_all_gemm_dram_bytes=total_dense,
            compressed_plus_lora_all_gemm_dram_bytes=total_compressed_lora,
            filter_payload_savings_percent=round(100*(1-compressed_filter/max(1,raw_base_filter)), 4)
        ))
    numeric = [key for key in per_layer[0] if key != 'layer' and key != 'filter_payload_savings_percent']
    totals = {key:sum(row[key] for row in per_layer) for key in numeric}
    totals['filter_payload_savings_percent'] = round(100*(1-totals['compressed_filter_bytes']/max(1,totals['original_filter_bytes'])),4)
    totals['all_gemm_traffic_change_percent_vs_original'] = round(100*(totals['compressed_plus_lora_all_gemm_dram_bytes']/max(1,totals['original_all_gemm_dram_bytes'])-1),4)
    totals['assumptions'] = dict(k=k, index_bits=index_bits, dense_value_bytes=dense_bytes,
        codebook_centroid_bytes=centroid_bytes, codebook_loaded_once_per_layer=True,
        filter_read_count_is_retained_from_dense_scale_sim=True,
        lora_branches_have_separately_counted_gemm_traffic=True,
        limitation='Postprocessed traffic hypotheses. Not simulated clustered memory, stall cycles, lookup latency, memory coalescing, parallel scheduling, or end-to-end latency.')
    return per_layer, totals


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=Path('hardware_sim/scalesim/workloads/manifest.json'))
    ap.add_argument('--base-report',type=Path,default=Path('results/hardware_sim/scalesim/base'))
    ap.add_argument('--lora-report',type=Path,default=Path('results/hardware_sim/scalesim/adapters'))
    ap.add_argument('--out',type=Path,default=Path('results/hardware_sim/scalesim/traffic_8bit'))
    ap.add_argument('--clusters',type=int,default=16)
    ap.add_argument('--index-bits',type=int,choices=[4,8],default=8)
    ap.add_argument('--dense-bytes',type=int,default=4)
    ap.add_argument('--centroid-bytes',type=int,default=4)
    args=ap.parse_args()
    if min(args.clusters,args.index_bits,args.dense_bytes,args.centroid_bytes)<1:
        ap.error('Sizes must be positive')
    manifest=json.loads(args.manifest.read_text())
    base=read_counts(find_report(args.base_report),len(manifest['layers']))
    adapter=read_counts(find_report(args.lora_report),2*len(manifest['layers']))
    rows, totals=analyze(manifest,base,adapter,args.clusters,args.index_bits,args.dense_bytes,args.centroid_bytes)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'dram_traffic_by_layer.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (args.out/'dram_traffic_summary.json').write_text(json.dumps(totals,indent=2)+'\n')
    print('Based on SCALE-Sim DETAILED_ACCESS_REPORT DRAM element counts:')
    print(f'  Original base-filter DRAM payload: {totals["original_filter_bytes"]:,} bytes')
    print(f'  Hypothetical {args.index_bits}-bit index + codebook payload: {totals["compressed_filter_bytes"]:,} bytes')
    print(f'  Filter payload savings: {totals["filter_payload_savings_percent"]:.2f}%')
    print(f'  Baseline GEMM DRAM payload incl. IFMAP/OFMAP: {totals["original_all_gemm_dram_bytes"]:,} bytes')
    print(f'  Hypothetical clustered+LoRA serial-GEMM DRAM payload: {totals["compressed_plus_lora_all_gemm_dram_bytes"]:,} bytes')
    print('  WARNING: byte-traffic counterfactual only, not cycle counts. See summary assumptions.')
    print('Saved',args.out/'dram_traffic_summary.json')

if __name__=='__main__': main()
