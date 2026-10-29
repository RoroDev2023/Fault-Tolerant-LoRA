#!/usr/bin/env python3
"""Dense vs compressed-prefetch-only conv1 (SCALE-Sim USER mode).

Experimental: physical filter DRAM byte transfers replace the native dense
filter backing transfer and decoded availability gates native SRAM reads.
A distinct >=37,632-B decoded-weight staging buffer is assumed, separately
from filter SRAM. A validated full accelerator model would need storage,
port/bank, energy and cross-operand contention modeling, plus all layers/LoRA.
"""
from __future__ import annotations

import argparse
import configparser
import json
import tempfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from compressed_filter_prefetch import PrefetchStats, patch_compressed_conv1_prefetch
from run_conv1_live_gate import extract_single_layer, load_cycle_report
from trace_cluster_replay import Config


def run_one(config, layout, topo, destination, context):
    from scalesim.scale_sim import scalesim
    destination.mkdir(parents=True, exist_ok=True)
    with context:
        sim = scalesim(save_disk_space=True, verbose=False,
                       config=str(config), topology=str(topo),
                       layout=str(layout), input_type_gemm=True)
        sim.run_scale(top_path=str(destination))
    return load_cycle_report(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload', type=Path, default=Path('hardware_sim/scalesim/workloads/original_gemm.csv'))
    p.add_argument('--config', type=Path, default=Path('hardware_sim/scalesim/scale_32x32_user_bw.cfg'))
    p.add_argument('--layout', type=Path, default=Path('../scale-sim-v3/layouts/conv_nets/test.csv'))
    p.add_argument('--out', type=Path, default=Path('results/hardware_sim/conv1_compressed_prefetch_v3'))
    p.add_argument('--bytes-per-cycle', type=int, default=40)
    p.add_argument('--decoder-width', type=int, default=32)
    p.add_argument('--queue-capacity', type=int, default=256)
    p.add_argument('--index-bits', type=int, default=8)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--lookup-latency', type=int, default=1)
    p.add_argument('--decoded-staging-bytes', type=int, default=37632,
                   help='separate physical staging capacity for reconstructed FP32 conv1 weights')
    args = p.parse_args()
    for file in (args.workload, args.config, args.layout):
        if not file.is_file():
            p.error(f'Missing input file: {file}')
    config = configparser.ConfigParser()
    config.read(args.config)
    bandwidth_mode = config.get('run_presets', 'InterfaceBandwidth', fallback='').strip().upper()
    if bandwidth_mode != 'USER':
        p.error(f'v3 requires InterfaceBandwidth: USER, got {bandwidth_mode!r}')

    cfg = Config(bytes_per_cycle=args.bytes_per_cycle, decoder_width=args.decoder_width,
                 queue_capacity=args.queue_capacity, index_bits=args.index_bits,
                 k=args.k, lookup_latency=args.lookup_latency)
    cfg.validate()
    with tempfile.TemporaryDirectory(prefix='ftlora_conv1_prefetch_v3_') as temporary:
        topo = Path(temporary) / 'conv1_only_gemm.csv'
        dims = extract_single_layer(args.workload, topo)
        if dims['weights'] != 9408:
            p.error('v3 currently requires exactly 9408 conv1 filter weights')
        dense = run_one(args.config, args.layout, topo, args.out / 'dense', nullcontext())
        stats = PrefetchStats()
        cm = patch_compressed_conv1_prefetch(cfg, stats, args.decoded_staging_bytes)
        compressed = run_one(args.config, args.layout, topo, args.out / 'compressed_prefetch', cm)

    def diff(k):
        a, b = dense[k], compressed[k]
        return (b - a) if a is not None and b is not None else None

    output = {
        'scope': 'conv1 experimental compressed filter DRAM prefetch + decoded readiness gate; '
                 'not a full physical hardware, end-to-end inference or LoRA/fault cycle model',
        'assumptions': {
            'physical_filter_link_bytes_per_cycle': cfg.bytes_per_cycle,
            'decoder_weights_per_cycle': cfg.decoder_width,
            'decoder_lookup_latency_cycles': cfg.lookup_latency,
            'index_bits': cfg.index_bits,
            'centroids': cfg.k,
            'separate_decoded_fp32_staging_capacity_bytes': args.decoded_staging_bytes,
            'filter_link_is_dedicated': True,
            'dense_prefetch_lookahead_cycles': stats.decoder.get('ideal_dense_transfer_cycles_at_same_physical_bandwidth'),
        },
        'layer': dims,
        'native_dense': dense,
        'compressed_prefetch_experimental': compressed,
        'physical_filter_bytes': {
            'dense': stats.dense_bytes,
            'compressed_indices_and_codebook': stats.phys_bytes,
        },
        'filter_supply': stats.decoder,
        'filter_gate': {
            'rows_seen': stats.gate.rows_seen,
            'rows_delayed': stats.gate.rows_delayed,
            'unique_accessed_addresses': len(stats.gate.accessed_addresses),
        },
        'filter_physical_dram_link': {
            'first_cycle': stats.compressed_fetch_start_cycle,
            'last_cycle': stats.compressed_fetch_last_cycle,
            'last_decoded_ready': stats.decoded_ready_last_cycle,
            'note': 'FILTER_DRAM_TRACE output in this experiment, if written, uses synthetic '
                    'physical BYTE addresses, not native logical weight addresses',
        },
        'comparison': {
            'native_total_cycle_delta': diff('total_cycles_native'),
            'native_stall_cycle_delta': diff('stall_cycles_native'),
            'native_prefetch_inclusive_delta': diff('prefetch_inclusive_cycles_native'),
        },
        'limitations': [
            'Strictly first 9408 unique conv1 filter weights, no filter SRAM refills, no other layers',
            'Separate decoded-weight staging capacity is assumed in addition to native filter SRAM',
            'Native filter SRAM addressing and abstract one-word/element hit modeling are unchanged',
            'Native filter bandwidth/energy report fields may mix element units and physical byte units',
            'Native IFMAP/OFMAP streams are unchanged; physical shared-link contention is not modeled',
            'No actual centroid values, index data, faults, LoRA execution, or decoding energy',
            'All decoded weights are staged prior to use when prefetch schedule permits',
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / 'comparison_compressed_prefetch_v3.json'
    target.write_text(json.dumps(output, indent=2) + '\n')
    print('\nCONV1 COMPRESSED FILTER PREFETCH v3 — experimental')
    print('  Dense total / stalls:', dense['total_cycles_native'], dense['stall_cycles_native'])
    print('  Compressed total / stalls:', compressed['total_cycles_native'], compressed['stall_cycles_native'])
    print('  Delta total / stalls:', output['comparison']['native_total_cycle_delta'],
          output['comparison']['native_stall_cycle_delta'])
    print('  Physical filter bytes dense / compressed:', stats.dense_bytes, stats.phys_bytes)
    print('  Decoder readiness last cycle:', stats.decoded_ready_last_cycle)
    print('  Gate rows delayed:', stats.gate.rows_delayed)
    print('  Staged decoded bytes (separate storage):', stats.decoded_staging_bytes)
    print('  Saved:', target)
    print('  WARNING: narrow simulated filter prefetch, not full hardware or LoRA/fault inference.')


if __name__ == '__main__':
    main()
