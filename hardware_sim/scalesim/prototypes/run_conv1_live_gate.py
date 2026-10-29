#!/usr/bin/env python3
"""Run dense vs experimental decoder-gated conv1 through live SCALE-Sim USER mode.

NOT a true compressed-DRAM execution model: native DRAM/filter prefetch is
unchanged and the original dense transfer remains in the timing/report.
Decoder readiness is precomputed from an existing conv1 DRAM trace.
"""
from __future__ import annotations

import argparse
import csv
import configparser
import json
import tempfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from live_filter_gate import GateStats, patch_live_filter_reads
from trace_cluster_replay import Config, extract_order, parse_trace, simulate_supply


def extract_single_layer(original_gemm: Path, target: Path, name='conv1'):
    """Extract a real GEMM topology row. Refuse accidentally running all layers."""
    with original_gemm.open(newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError('No GEMM rows in ' + str(original_gemm))
    matching = [row for row in rows[1:] if row and row[0].strip() == name]
    if len(matching) != 1:
        raise ValueError(f'Expected exactly one {name!r} in {original_gemm}; found {len(matching)}')
    with target.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(rows[0])
        writer.writerow(matching[0])
    row = matching[0]
    M, N, K = [int(row[i].strip()) for i in (1, 2, 3)]
    return {'layer': name, 'M': M, 'N': N, 'K': K, 'weights': N * K}


def load_cycle_report(folder: Path):
    paths = list(folder.rglob('COMPUTE_REPORT.csv'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected one COMPUTE_REPORT.csv under {folder}, found {len(paths)}')
    with paths[0].open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f'Expected one conv1 in report, got {len(rows)}')
    row = {k.strip().lower(): v.strip() for k, v in rows[0].items() if k}
    def numeric(key):
        return int(float(row[key])) if key in row else None
    return {
        'total_cycles_native': numeric('total cycles'),
        'compute_cycles_native': numeric('compute cycles'),
        'stall_cycles_native': numeric('stall cycles'),
        'prefetch_inclusive_cycles_native': next((int(float(v)) for k, v in row.items()
                                      if 'incl. prefetch' in k), None),
        'original_report': str(paths[0]),
        'all_report_fields': row,
    }


def run_one(config: Path, layout: Path, topo: Path, destination: Path,
            ready: dict[int, int] | None):
    from scalesim.scale_sim import scalesim
    destination.mkdir(parents=True, exist_ok=True)
    stats = GateStats()
    context = patch_live_filter_reads(ready, stats) if ready is not None else nullcontext()
    with context:
        sim = scalesim(save_disk_space=True, verbose=False,
                       config=str(config), topology=str(topo),
                       layout=str(layout), input_type_gemm=True)
        sim.run_scale(top_path=str(destination))
    return load_cycle_report(destination), stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload', type=Path, default=Path('hardware_sim/scalesim/workloads/original_gemm.csv'))
    p.add_argument('--config', type=Path, default=Path('hardware_sim/scalesim/scale_32x32_user_bw.cfg'))
    p.add_argument('--layout', type=Path, default=Path('../scale-sim-v3/layouts/conv_nets/test.csv'))
    p.add_argument('--trace-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results/hardware_sim/conv1_live_decoder_gate'))
    p.add_argument('--bytes-per-cycle', type=int, default=40)
    p.add_argument('--decoder-width', type=int, default=32)
    p.add_argument('--queue-capacity', type=int, default=256)
    p.add_argument('--index-bits', type=int, default=8)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--lookup-latency', type=int, default=1)
    p.add_argument('--prefetch-start', type=int, default=None,
                   help='optional decoder time origin; default earliest original filter DRAM cycle')
    args = p.parse_args()
    for path in (args.workload, args.config, args.layout,
                 args.trace_dir / 'FILTER_DRAM_TRACE.csv'):
        if not path.is_file():
            p.error(f'Required file does not exist: {path}')
    parser = configparser.ConfigParser()
    parser.read(args.config)
    bandwidth_mode = parser.get('run_presets', 'InterfaceBandwidth', fallback='').strip().upper()
    if bandwidth_mode != 'USER':
        p.error(f'This prototype supports only InterfaceBandwidth: USER (found {bandwidth_mode!r})')
    cfg = Config(bytes_per_cycle=args.bytes_per_cycle, decoder_width=args.decoder_width,
                 queue_capacity=args.queue_capacity, index_bits=args.index_bits,
                 k=args.k, lookup_latency=args.lookup_latency,
                 prefetch_start_cycle=args.prefetch_start)
    dram_rows = list(parse_trace(args.trace_dir / 'FILTER_DRAM_TRACE.csv'))
    start, addresses = extract_order(dram_rows)
    if cfg.prefetch_start_cycle is not None:
        start = cfg.prefetch_start_cycle
    ready, supply_stats = simulate_supply(addresses, start, cfg)
    with tempfile.TemporaryDirectory(prefix='ftlora_conv1_') as temporary:
        topo = Path(temporary) / 'conv1_only_gemm.csv'
        dims = extract_single_layer(args.workload, topo)
        if dims['weights'] != len(addresses):
            p.error(f'conv1 GEMM weights={dims["weights"]}, old DRAM trace={len(addresses)}; wrong trace?')
        # Same single-layer topology, same config, same outputs — only the
        # decoder readiness gate differs between the two runs.
        dense, dense_stats = run_one(args.config, args.layout, topo, args.out / 'dense', None)
        gated, gate_stats = run_one(args.config, args.layout, topo, args.out / 'clustered_gate', ready)
        if gate_stats.rows_seen == 0 or gate_stats.accessed_addresses != set(ready):
            raise RuntimeError('Decoder gate did not observe exactly all the expected conv1 filter addresses; '
                               f'got {len(gate_stats.accessed_addresses)}/{len(ready)}. '
                               'Check the config, layout, native simulator version and trace source.')
    result = {
        'scope': 'LIVE decoder timing gate over original dense SCALE-Sim USER-mode filter buffer; '
                 'NOT a compressed-DRAM memory simulation, not a true clustered inference speedup',
        'layer': dims,
        'source_trace_dir': str(args.trace_dir),
        'decoder_config': asdict(cfg),
        'counterfactual_decoder_supply': supply_stats,
        'dense': dense,
        'clustered_gate_with_native_dense_dram': gated,
        'gate_diagnostics': {k: (len(v) if isinstance(v, set) else v) for k, v in vars(gate_stats).items()},
        'comparison': {
            'native_report_total_cycles_delta': (gated['total_cycles_native'] - dense['total_cycles_native'])
                                                if gated['total_cycles_native'] is not None and dense['total_cycles_native'] is not None else None,
            'native_report_stall_cycles_delta': (gated['stall_cycles_native'] - dense['stall_cycles_native'])
                                                if gated['stall_cycles_native'] is not None and dense['stall_cycles_native'] is not None else None,
        },
        'limitations': [
            'Original SCALE-Sim dense FILTER_DRAM_TRACE and prefetch cost remain unchanged',
            'Decoder availability originates from previous conv1 DRAM trace (not generated online)',
            'Input/output and memory-port contention not resimulated for compressed transfers',
            'No actual KMeans index values, codebook values, faults or LoRA computations modeled',
            'Only conv1, no SRAM evictions, USER mode; not general to other layers',
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'comparison_decoder_gate.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('\nCONV1 LIVE FILTER DECODER GATE — experimental integration')
    print('  Dense native total/stalls:', dense['total_cycles_native'], dense['stall_cycles_native'])
    print('  Gated native total/stalls:', gated['total_cycles_native'], gated['stall_cycles_native'])
    print('  Gated rows seen:', gate_stats.rows_seen, 'rows delayed:', gate_stats.rows_delayed)
    print('  Native total delta:', result['comparison']['native_report_total_cycles_delta'])
    print('  Native stalls delta:', result['comparison']['native_report_stall_cycles_delta'])
    print('  Saved:', path)
    print('  WARNING: dense filter prefetch is still modeled in both runs; not full compressed-memory cycles.')


if __name__ == '__main__':
    main()
