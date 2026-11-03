#!/usr/bin/env python3
"""Create a USER-bandwidth copy of a SCALE-Sim v3 config without altering original."""
import argparse
import configparser
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=Path('../scale-sim-v3/configs/scale.cfg'))
    p.add_argument('--out', type=Path, default=Path('hardware_sim/scalesim/scale_32x32_user_bw.cfg'))
    p.add_argument('--bandwidth',type=int,default=10,help='SCALE-Sim bandwidth in data elements per cycle (not automatically bytes/cycle)')
    p.add_argument('--array-height', type=int, default=None)
    p.add_argument('--array-width', type=int, default=None)
    p.add_argument('--dataflow', choices=('ws', 'os', 'is'), default=None)
    p.add_argument('--ifmap-sram-kb', type=int, default=None)
    p.add_argument('--filter-sram-kb', type=int, default=None)
    p.add_argument('--ofmap-sram-kb', type=int, default=None)
    args = p.parse_args()
    requested = {
        'Bandwidth': args.bandwidth,
        'ArrayHeight': args.array_height,
        'ArrayWidth': args.array_width,
        'IfmapSramSzkB': args.ifmap_sram_kb,
        'FilterSramSzkB': args.filter_sram_kb,
        'OfmapSramSzkB': args.ofmap_sram_kb,
    }
    for key, value in requested.items():
        if value is not None and value <= 0:
            p.error(f'{key} must be positive')
    if args.base.resolve() == args.out.resolve():
        p.error('Refusing to overwrite the base SCALE-Sim config')
    c = configparser.ConfigParser()
    if not c.read(args.base):
        p.error(f'Cannot read {args.base}')
    if 'run_presets' not in c or 'architecture_presets' not in c:
        p.error('Unexpected SCALE-Sim config sections')
    c['run_presets']['InterfaceBandwidth'] = 'USER'
    arch = c['architecture_presets']
    for key, value in requested.items():
        if value is not None:
            arch[key] = str(value)
    if args.dataflow:
        arch['Dataflow'] = args.dataflow
    height, width = int(arch['ArrayHeight']), int(arch['ArrayWidth'])
    flow = arch['Dataflow'].strip().lower()
    if flow not in ('ws', 'os', 'is'):
        p.error(f'Unexpected dataflow in base config: {flow!r}')
    c['general']['run_name'] = f'resnet_{height}x{width}_{flow}_bw{args.bandwidth}'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        c.write(f)
    print('Wrote', args.out, 'array', f'{height}x{width}', 'dataflow', flow,
          'USER bandwidth', args.bandwidth, 'elements/cycle')

if __name__=='__main__':main()
