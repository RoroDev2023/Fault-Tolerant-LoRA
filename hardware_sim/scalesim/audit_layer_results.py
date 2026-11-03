#!/usr/bin/env python3
"""Check existing per-layer SCALE-Sim results without rerunning the simulator.

Only total cycles are summed; utilization and mapping are *cycle-weighted*
layer-level metrics. Detect missing/mismatched configs, suspicious stall ratios,
and changed bandwidth/array/dataflow across completed layer results.
"""
from __future__ import annotations
import argparse
import configparser
import json
from pathlib import Path


def audit(root: Path, *, size: int, bandwidth: int, flow: str, sram_kb: int):
    found = []
    missing = []
    for i in range(21):
        folder = root / f'layer_{i:02d}'
        result = folder / 'result.json'
        cfg_path = folder / 'config.cfg'
        if not result.exists():
            missing.append(i)
            continue
        if not cfg_path.exists():
            raise ValueError(f'{folder}: result exists without configuration')
        cfg = configparser.ConfigParser()
        if not cfg.read(cfg_path):
            raise ValueError(f'Cannot read {cfg_path}')
        arch = cfg['architecture_presets']
        for key, expected in [('arrayheight',size),('arraywidth',size),
                              ('bandwidth',bandwidth),('ifmapsramszkb',sram_kb),
                              ('filtersramszkb',sram_kb),('ofmapsramszkb',sram_kb)]:
            if int(arch[key]) != expected:
                raise ValueError(f'{cfg_path}: {key}={arch[key]}, expected {expected}')
        if arch['dataflow'].lower() != flow:
            raise ValueError(f'{cfg_path}: dataflow {arch["dataflow"]} != {flow}')
        if cfg['run_presets']['interfacebandwidth'].upper() != 'USER':
            raise ValueError(f'{cfg_path}: not USER mode')
        item = json.loads(result.read_text())
        if int(item['array_size']) != size or int(item['layer_index']) != i:
            raise ValueError(f'{result}: array/layer mismatch')
        if int(item['cycles']) <= 0:
            raise ValueError(f'{result}: invalid cycles')
        for key in ('utilization_pct','mapping_efficiency_pct'):
            if not 0 <= float(item[key]) <= 100:
                raise ValueError(f'{result}: invalid {key}')
        found.append(item)
    count = len(found)
    cycles = sum(int(x['cycles']) for x in found)
    return {'completed_layers':count,'missing_layers':missing,
            'serial_sum_cycles':cycles if count==21 else None,
            'partial_cycles':cycles,
            'cycle_weighted_overall_util_pct':
              sum(x['cycles']*x['utilization_pct'] for x in found)/cycles if cycles else None,
            'cycle_weighted_mapping_eff_pct':
              sum(x['cycles']*x['mapping_efficiency_pct'] for x in found)/cycles if cycles else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path,help='directory containing layer_00 .. layer_20')
    p.add_argument('--size',required=True,type=int)
    p.add_argument('--bandwidth',required=True,type=int)
    p.add_argument('--dataflow',choices=('ws','os','is'),required=True)
    p.add_argument('--sram-kb',required=True,type=int)
    args=p.parse_args()
    if min(args.size,args.bandwidth,args.sram_kb)<=0:
        p.error('dimensions, bandwidth, and SRAM must be positive')
    print(json.dumps(audit(args.root,size=args.size,bandwidth=args.bandwidth,
                           flow=args.dataflow,sram_kb=args.sram_kb),indent=2))

if __name__=='__main__': main()
