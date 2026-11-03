#!/usr/bin/env python3
"""Validate exported ResNet-18 GEMM+LoRA workloads against the manifest.

Uses only the Python standard library; fails before an expensive SCALE-Sim run.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def parse_workload(path: Path) -> list[dict]:
    with path.open(newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty workload: {path}')
    header = [x.strip().lower() for x in rows[0]]
    if header != ['layer name', 'm', 'n', 'k', 'sparsity', '']:
        raise ValueError(f'{path}: wrong GEMM header {header!r}; re-export with the fixed extractor')
    parsed = []
    for line, fields in enumerate(rows[1:], 2):
        if len(fields) != 6 or fields[-1].strip():
            raise ValueError(f'{path}:{line}: expected six CSV columns with trailing empty field')
        name, m, n, k, sparsity, _ = [v.strip() for v in fields]
        if not name or sparsity != '1:1':
            raise ValueError(f'{path}:{line}: expected nonempty name and dense sparsity 1:1')
        try:
            dims = tuple(int(x) for x in (m, n, k))
        except ValueError as exc:
            raise ValueError(f'{path}:{line}: non-integer GEMM dimensions') from exc
        if min(dims) <= 0:
            raise ValueError(f'{path}:{line}: nonpositive GEMM dimension')
        parsed.append({'name': name, 'M': dims[0], 'N': dims[1], 'K': dims[2]})
    if not parsed or len({r['name'] for r in parsed}) != len(parsed):
        raise ValueError(f'{path}: no layers or duplicate layer names')
    return parsed


def validate(workloads: Path) -> dict:
    manifest = json.loads((workloads / 'manifest.json').read_text())
    layers = manifest['layers']
    original = parse_workload(workloads / 'original_gemm.csv')
    adapter = parse_workload(workloads / 'lora_gemm.csv')
    if not (len(layers) == len(original) == 21 and len(adapter) == 42):
        raise ValueError('Expected 21 original and 42 LoRA GEMM layers')
    if manifest.get('image_size') != 96 or manifest.get('batch_size') != 1:
        raise ValueError('Expected batch-1, 96x96 input; confirm model preprocessing')
    for i, layer in enumerate(layers):
        name = layer['name']
        for role, row in [('base', original[i]),
                          ('lora_A', adapter[2*i]),
                          ('lora_B', adapter[2*i+1])]:
            expected_name = name if role == 'base' else f'{name}__{role}'
            if row['name'] != expected_name:
                raise ValueError(f'{role} row {i}: expected {expected_name}, got {row["name"]}')
            expected = layer[role]
            if any(row[d] != int(expected[d]) for d in ('M','N','K')):
                raise ValueError(f'{role} {name}: CSV geometry does not match manifest')
            if row['M'] * row['N'] * row['K'] != int(expected['MACs']):
                raise ValueError(f'{role} {name}: MAC count does not match manifest')
    return {'base_layers': len(original), 'lora_layers': len(adapter),
            'base_macs': sum(r['M']*r['N']*r['K'] for r in original)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workloads', type=Path, default=Path('hardware_sim/scalesim/workloads'))
    args=p.parse_args()
    r=validate(args.workloads)
    print('PASS:',r)

if __name__=='__main__':
    main()
