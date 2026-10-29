#!/usr/bin/env python3
"""First trace-driven integration for clustered conv1 weight supply.

Reads actual SCALE-Sim FILTER_DRAM_TRACE and FILTER_SRAM_TRACE, constructs
counterfactual arrivals of KMeans index bytes/codebook/decoded FP32 weights,
then checks whether the original SRAM read schedule would have been satisfied.

This is NOT a modification of SCALE-Sim's event loop. The output's replay
stalls are a diagnostic with fixed native timing, NOT recomputed layer cycles.
The original prefetch scheduling and IFMAP/OFMAP stalls are not recomputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    bytes_per_cycle: int = 40   # 10 native SCALE-Sim elements/cycle * 4 FP32 bytes/element
    decoder_width: int = 32    # number of 8-bit indices decoded per cycle
    queue_capacity: int = 256
    index_bits: int = 8
    k: int = 16
    weight_bytes: int = 4
    centroid_bytes: int = 4
    lookup_latency: int = 1
    prefetch_start_cycle: int | None = None  # defaults to earliest FILTER_DRAM_TRACE timestamp

    def validate(self):
        for field in ('bytes_per_cycle', 'decoder_width', 'queue_capacity',
                      'index_bits', 'k', 'weight_bytes', 'centroid_bytes'):
            if getattr(self, field) <= 0:
                raise ValueError(f'{field} must be positive')
        if self.index_bits < (self.k - 1).bit_length():
            raise ValueError('not enough bits to represent codebook entries')
        if self.lookup_latency < 0:
            raise ValueError('lookup_latency must be >=0')


def parse_trace(path: Path):
    """Yield (integer trace_cycle, tuple[integer addresses]); -1 indicates inactive."""
    with path.open(newline='') as file:
        for lineno, row in enumerate(csv.reader(file), 1):
            if not row:
                continue
            try:
                cycle = int(float(row[0]))
                addresses = tuple(int(float(v)) for v in row[1:]
                                  if v.strip() and float(v) >= 0)
            except ValueError as exc:
                raise ValueError(f'Invalid trace value in {path}:{lineno}: {row[:4]}') from exc
            yield cycle, addresses


def extract_order(dram_rows):
    """This prototype is intentionally restricted to all-once cold-prefetch layers."""
    first = min(c for c, _ in dram_rows)
    # Stable chronological order: transfers from same trace cycle retain CSV order.
    chronological = sorted(enumerate(dram_rows), key=lambda t: (t[1][0], t[0]))
    addresses = [a for _, (_, arr) in chronological for a in arr]
    if not addresses:
        raise ValueError('DRAM trace has no filter addresses')
    if len(addresses) != len(set(addresses)):
        raise ValueError('Repeated DRAM weight fetches: prototype supports only a single cold prefetch; do not use for other layers yet')
    return first, addresses


def simulate_supply(addresses, start_cycle: int, cfg: Config):
    """Return decoder-ready cycle per logical weight address and pipeline stats.

    Codebook is fetched serially first. Index bits stream into a bounded FIFO;
    decoder runs in parallel with the transfer (same-cycle consumption permitted).
    This deliberately assumes a dedicated filter link and whole compressed tensor
    can be reconstructed into filter SRAM without evictions.
    """
    cfg.validate()
    if len(addresses) != len(set(addresses)):
        raise ValueError('addresses must be unique')
    total = len(addresses)
    codebook_bytes = cfg.k * cfg.centroid_bytes
    codebook_cycles = math.ceil(codebook_bytes / cfg.bytes_per_cycle)
    ready = {}
    fifo = deque()
    next_index = 0
    bit_reservoir = 0
    cycle = start_cycle + codebook_cycles
    blocked_cycles = 0
    decode_cycles = 0
    index_link_cycles = 0
    peak_queue = 0
    # Guard loop to avoid silently hanging on impossible configurations.
    max_cycles = total * (cfg.index_bits + 10) + 100000
    active_cycles = 0
    while len(ready) < total:
        active_cycles += 1
        if active_cycles > max_cycles:
            raise RuntimeError('pipeline failed to terminate')
        # Index transfer; enough complete bits must be available for decoding.
        # FIFO slots cap the number of complete indices in queue; permit
        # same-cycle transfer+decode even with queue_capacity=1.
        space = cfg.queue_capacity - len(fifo)
        incoming = 0
        if next_index < total and space > 0:
            max_new_bits = cfg.bytes_per_cycle * 8
            remaining_bits = (total - next_index) * cfg.index_bits - bit_reservoir
            transported_bits = min(max_new_bits, remaining_bits,
                                   max(0, space * cfg.index_bits - bit_reservoir))
            if transported_bits:
                index_link_cycles += 1
                bit_reservoir += transported_bits
            incoming = min(space, total - next_index, bit_reservoir // cfg.index_bits)
            for i in range(incoming):
                fifo.append(addresses[next_index + i])
            next_index += incoming
            bit_reservoir -= incoming * cfg.index_bits
        elif next_index < total:
            blocked_cycles += 1
        # Consume indices from FIFO; result available at cycle + latency.
        decoded_this_cycle = min(len(fifo), cfg.decoder_width)
        if decoded_this_cycle:
            decode_cycles += 1
        for _ in range(decoded_this_cycle):
            ready[fifo.popleft()] = cycle + cfg.lookup_latency
        peak_queue = max(peak_queue, len(fifo))
        cycle += 1
    stats = {
        'weight_positions': total,
        'dense_bytes': total * cfg.weight_bytes,
        'compressed_index_bytes': math.ceil(total * cfg.index_bits / 8),
        'codebook_bytes': codebook_bytes,
        'compressed_total_bytes': math.ceil(total * cfg.index_bits / 8) + codebook_bytes,
        'codebook_transfer_cycles': codebook_cycles,
        'index_link_active_cycles': index_link_cycles,
        'decoder_active_cycles': decode_cycles,
        'pipeline_cycles_after_codebook': active_cycles,
        'peak_queue_after_decode': peak_queue,
        'backpressure_cycles': blocked_cycles,
        'first_ready_cycle': min(ready.values()),
        'last_ready_cycle': max(ready.values()),
        'prefetch_start_cycle': start_cycle,
        'total_prefetch_span_cycles': max(ready.values()) - start_cycle + 1,
        'ideal_dense_transfer_cycles_at_same_physical_bandwidth':
            math.ceil(total * cfg.weight_bytes / cfg.bytes_per_cycle),
        'ideal_compressed_transfer_cycles_excluding_decode':
            math.ceil((math.ceil(total * cfg.index_bits / 8) + codebook_bytes) / cfg.bytes_per_cycle),
    }
    return ready, stats


def replay_sram(sram_rows, ready):
    """Check time-of-availability against original *serviced* SRAM trace cycles.

    An optional shifted-clock diagnostic propagates extra delays through later
    rows, but does not update IFMAP/OFMAP, prefetch, or original stall timelines.
    Therefore DO NOT add 'replay_extra_delay_cycles' to COMPUTE_REPORT.csv.
    """
    total_accesses = 0
    distinct = set()
    original_late_rows = 0
    max_original_lateness = 0
    replay_extra_delay = 0
    delayed_rows = 0
    first_active = None
    last_active = None
    rows = 0
    for baseline_cycle, addresses in sram_rows:
        rows += 1
        if not addresses:
            continue
        if first_active is None:
            first_active = baseline_cycle
        last_active = baseline_cycle
        for addr in addresses:
            if addr not in ready:
                raise ValueError(f'SRAM address {addr} not found in DRAM trace')
        total_accesses += len(addresses)
        distinct.update(addresses)
        latest_available = max(ready[a] for a in addresses)
        original_lateness = max(0, latest_available - baseline_cycle)
        if original_lateness:
            original_late_rows += 1
            max_original_lateness = max(max_original_lateness, original_lateness)
        counterfactual_cycle = baseline_cycle + replay_extra_delay
        add_delay = max(0, latest_available - counterfactual_cycle)
        if add_delay:
            replay_extra_delay += add_delay
            delayed_rows += 1
    if not distinct:
        raise ValueError('SRAM trace has no filter reads')
    return {
        'sram_trace_rows': rows,
        'sram_read_requests': total_accesses,
        'unique_sram_addresses': len(distinct),
        'first_active_sram_cycle': first_active,
        'last_active_sram_cycle': last_active,
        'rows_missing_original_deadline': original_late_rows,
        'max_original_deadline_miss_cycles': max_original_lateness,
        'shifted_clock_delay_diagnostic_cycles': replay_extra_delay,
        'shifted_clock_delayed_rows': delayed_rows,
    }


def analyze(dram_rows, sram_rows, cfg: Config):
    cfg.validate()
    if not dram_rows:
        raise ValueError('empty filter DRAM trace')
    start_cycle, addr_order = extract_order(dram_rows)
    if cfg.prefetch_start_cycle is not None:
        start_cycle = cfg.prefetch_start_cycle
    ready, supply = simulate_supply(addr_order, start_cycle, cfg)
    demand = replay_sram(sram_rows, ready)
    return {
        'scope': 'trace-driven clustered filter supply; NOT recomputed SCALE-Sim inference/stalls',
        'configuration': asdict(cfg),
        'filter_dram': {
            'trace_rows': len(dram_rows),
            'first_recorded_cycle': min(c for c, _ in dram_rows),
            'last_recorded_cycle': max(c for c, _ in dram_rows),
            'unique_addresses': len(addr_order),
            'repeated_addresses': 0,
        },
        'clustered_supply_model': supply,
        'filter_sram_deadline_check': demand,
        'interpretation': (
            'Original SRAM trace reports serviced times after native stalls. '
            'The checker asks whether compressed weights could be ready by those times. '
            'It does not remove native dense prefetch cycles, recompute IFMAP/OFMAP overlap, '
            'or change SCALE-Sim total cycles. Shifted-clock delays are diagnostics only.'
        ),
        'fault_timing': 'silent index corruption with no ECC, retry or fault-dependent control flow: unchanged supply timing',
    }


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', required=True, type=Path,
                        help='folder containing FILTER_DRAM_TRACE.csv and FILTER_SRAM_TRACE.csv')
    parser.add_argument('--out', type=Path, default=Path('results/hardware_sim/conv1_cluster_trace_replay.json'))
    parser.add_argument('--bytes-per-cycle', type=int, default=40)
    parser.add_argument('--decoder-width', type=int, default=32)
    parser.add_argument('--queue-capacity', type=int, default=256)
    parser.add_argument('--index-bits', type=int, default=8)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--lookup-latency', type=int, default=1)
    parser.add_argument('--prefetch-start', type=int, default=None,
                        help='optional time origin for counterfactual; default earliest original DRAM timestamp')
    parser.add_argument('--expected-weights', type=int, default=9408)
    args = parser.parse_args()
    cfg = Config(bytes_per_cycle=args.bytes_per_cycle,
                 decoder_width=args.decoder_width, queue_capacity=args.queue_capacity,
                 index_bits=args.index_bits, k=args.k,
                 lookup_latency=args.lookup_latency,
                 prefetch_start_cycle=args.prefetch_start)
    dram_path = args.trace_dir / 'FILTER_DRAM_TRACE.csv'
    sram_path = args.trace_dir / 'FILTER_SRAM_TRACE.csv'
    for path in (dram_path, sram_path):
        if not path.is_file():
            parser.error(f'File not found: {path}')
    report = analyze(list(parse_trace(dram_path)), parse_trace(sram_path), cfg)
    if report['clustered_supply_model']['weight_positions'] != args.expected_weights:
        parser.error(f'Expected {args.expected_weights} filter weights, observed '
                     f"{report['clustered_supply_model']['weight_positions']}; "
                     'prototype is for conv1 only')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    m = report['clustered_supply_model']; r = report['filter_sram_deadline_check']
    print('CONV1 TRACE-DRIVEN CLUSTERED FILTER REPLAY (not full inference cycles)')
    print(f"  DRAM addresses: {m['weight_positions']:,}; SRAM accesses: {r['sram_read_requests']:,}")
    print(f"  Dense bytes: {m['dense_bytes']:,}; compressed bytes: {m['compressed_total_bytes']:,}")
    print(f"  Ideal dense transfer: {m['ideal_dense_transfer_cycles_at_same_physical_bandwidth']:,} cycles")
    print(f"  Clustered prefetch+decode: {m['total_prefetch_span_cycles']:,} modeled cycles")
    print(f"  Counterfactual decoder ready: {m['first_ready_cycle']} to {m['last_ready_cycle']}")
    print(f"  Original SRAM serviced: {r['first_active_sram_cycle']} to {r['last_active_sram_cycle']}")
    print(f"  Rows missing decoder deadline: {r['rows_missing_original_deadline']:,}")
    print(f"  Shifted-clock extra delay diagnostic: {r['shifted_clock_delay_diagnostic_cycles']:,}")
    print('  WARNING: This is trace replay, not modified SCALE-Sim inference or cycles.')
    print('  Saved:', args.out)


if __name__ == '__main__':
    cli()
