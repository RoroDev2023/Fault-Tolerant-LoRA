#!/usr/bin/env python3
"""An isolated, inspectable *weight-supply* cycle model for one ResNet layer.

This is NOT a SCALE-Sim extension or a ResNet inference timing estimator.
It models an off-chip transfer link and a pipelined on-chip centroid decoder.
The codebook is fetched once, then index transfers and decoding can overlap.
SCALE-Sim demand traces, IFMAP/OFMAP, filter-buffer refill/reuse and PE
scheduling are deliberately NOT represented. Do not add these results to
SCALE-Sim COMPUTE_REPORT.csv cycles.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Assumptions:
    n_weights: int = 9408  # Conv1 = [64,3,7,7]
    k: int = 16
    index_bits: int = 8
    weight_bytes: int = 4
    centroid_bytes: int = 4
    bandwidth_words_per_cycle: int = 10
    word_bytes: int = 4  # assumed bus word; not verified from SCALE-Sim config
    decoder_weights_per_cycle: int = 32
    lookup_latency_cycles: int = 1
    queue_capacity_indices: int = 256
    codebook_cached: bool = False


def positive_int_fields(a: Assumptions) -> None:
    for field in (
        'n_weights', 'k', 'index_bits', 'weight_bytes',
        'centroid_bytes', 'bandwidth_words_per_cycle', 'word_bytes',
        'decoder_weights_per_cycle', 'queue_capacity_indices'
    ):
        if getattr(a, field) <= 0:
            raise ValueError(f'{field} must be positive')
    if a.lookup_latency_cycles < 0:
        raise ValueError('lookup_latency_cycles must be nonnegative')
    if a.index_bits < (a.k - 1).bit_length():
        raise ValueError(f'{a.k} codebook entries need at least {(a.k - 1).bit_length()} index bits')


def simulate(a: Assumptions) -> dict:
    positive_int_fields(a)
    bus_bytes = a.bandwidth_words_per_cycle * a.word_bytes
    bw_bits = 8 * bus_bytes
    dense_bytes = a.n_weights * a.weight_bytes
    index_bits_total = a.n_weights * a.index_bits
    index_bytes = math.ceil(index_bits_total / 8)
    codebook_bytes = a.k * a.centroid_bytes
    dense_transfer = math.ceil(dense_bytes / bus_bytes)
    codebook_transfer = 0 if a.codebook_cached else math.ceil(codebook_bytes / bus_bytes)

    # Cycle-by-cycle *streaming* model: DRAM supplies packed index bits into
    # a finite on-chip queue; a separate decoder consumes complete indices.
    # A one-time codebook transfer comes first. No overlapping of codebook
    # transfer with index transfer, but index transfer may overlap decoding.
    transferred_bits = 0
    decoded = 0
    buffered = 0
    max_queue = 0
    cycles = 0
    index_transfer_active_cycles = 0
    decode_active_cycles = 0
    transfer_blocked_cycles = 0
    while decoded < a.n_weights:
        cycles += 1
        missing_bits = index_bits_total - transferred_bits
        spare_slots = a.queue_capacity_indices - buffered
        available_bits = max(0, min(bw_bits, missing_bits, spare_slots * a.index_bits))
        # The total bits are divisible by index_bits, so this is integral indices.
        incoming = available_bits // a.index_bits
        if incoming:
            index_transfer_active_cycles += 1
        elif missing_bits > 0 and spare_slots == 0:
            transfer_blocked_cycles += 1
        transferred_bits += incoming * a.index_bits
        buffered += incoming
        just_decoded = min(buffered, a.decoder_weights_per_cycle,
                           a.n_weights - decoded)
        if just_decoded:
            decode_active_cycles += 1
        decoded += just_decoded
        buffered -= just_decoded
        max_queue = max(max_queue, buffered)
        if cycles > a.n_weights + 100000:
            raise RuntimeError('unexpected nonterminating simulation')

    # Pipeline output valid after final decode plus fixed lookup latency.
    clustered_supply_cycles = codebook_transfer + cycles + a.lookup_latency_cycles
    return {
        'layer': 'conv1',
        'weight_positions': a.n_weights,
        'dense_weight_bytes': dense_bytes,
        'cluster_index_bytes': index_bytes,
        'codebook_bytes': codebook_bytes,
        'cluster_total_bytes': index_bytes + codebook_bytes,
        'bus_bytes_per_cycle_assumed': bus_bytes,
        'dense_weight_supply_cycles': dense_transfer,
        'codebook_preload_cycles': codebook_transfer,
        'index_transfer_active_cycles': index_transfer_active_cycles,
        'decoder_active_cycles': decode_active_cycles,
        'decoder_pipeline_cycles': cycles,
        'decoder_lookup_latency_cycles': a.lookup_latency_cycles,
        'clustered_weight_supply_cycles': clustered_supply_cycles,
        'faulty_clustered_weight_supply_cycles': clustered_supply_cycles,
        'max_queued_indices': max_queue,
        'memory_backpressure_cycles': transfer_blocked_cycles,
        'fault_timing_model': 'persistent silent index corruption; same indices accessed; no detection/correction or retry',
        'scope': 'ISOLATED weight supply microbenchmark ONLY; NOT SCALE-Sim cycles or whole-layer inference',
        'assumptions': asdict(a),
    }


def count_weights_from_manifest(path: Path, layer: str) -> int:
    payload = json.loads(path.read_text())
    matching = [v for v in payload['layers'] if v['name'] == layer]
    if len(matching) != 1:
        raise ValueError(f'Expected exactly one layer {layer!r} in manifest {path}')
    shape = matching[0]['base']['weight_shape']
    return math.prod(int(x) for x in shape)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, help='Existing extractor manifest.json; reads conv1 actual weight_shape')
    p.add_argument('--layer', default='conv1')
    p.add_argument('--bandwidth-words', type=int, default=10)
    p.add_argument('--word-bytes', type=int, default=4)
    p.add_argument('--decoder-width', type=int, default=32)
    p.add_argument('--lookup-latency', type=int, default=1)
    p.add_argument('--queue-capacity', type=int, default=256)
    p.add_argument('--index-bits', type=int, default=8)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--codebook-cached', action='store_true')
    p.add_argument('--output', type=Path, default=Path('results/hardware_sim/conv1_supply_microbenchmark.json'))
    args = p.parse_args()
    if args.layer != 'conv1':
        raise ValueError('prototype currently labeled for conv1 only; extend after validating workload semantics')
    n_weights = count_weights_from_manifest(args.manifest, args.layer) if args.manifest else 9408
    a = Assumptions(n_weights=n_weights, k=args.k, index_bits=args.index_bits,
                    bandwidth_words_per_cycle=args.bandwidth_words, word_bytes=args.word_bytes,
                    decoder_weights_per_cycle=args.decoder_width,
                    lookup_latency_cycles=args.lookup_latency,
                    queue_capacity_indices=args.queue_capacity,
                    codebook_cached=args.codebook_cached)
    result = simulate(a)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Conv1 isolated weight-supply microbenchmark (not SCALE-Sim inference cycles)')
    print(f"  Weight positions: {result['weight_positions']:,}")
    print(f"  Dense DRAM bytes: {result['dense_weight_bytes']:,}")
    print(f"  Compressed DRAM bytes: {result['cluster_total_bytes']:,}")
    print(f"  Dense weight-supply cycles: {result['dense_weight_supply_cycles']:,}")
    print(f"  Clustered weight-supply cycles: {result['clustered_weight_supply_cycles']:,}")
    print(f"  Faulty-clustered weight-supply cycles: {result['faulty_clustered_weight_supply_cycles']:,}")
    print(f"  Codebook preload: {result['codebook_preload_cycles']} cycles; decoder pipeline: {result['decoder_pipeline_cycles']} cycles")
    print('  WARNING: no SRAM/PE demand traces, IFMAP/OFMAP traffic, compute, LoRA or overlap with compute.')
    print('  Saved:', args.output)


if __name__ == '__main__':
    main()
