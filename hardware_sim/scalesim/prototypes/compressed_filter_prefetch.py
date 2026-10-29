"""Experimental compressed filter prefetch for one all-once ResNet conv1 GEMM.

Intercepts *only* SCALE-Sim's filter prefetch inside the USER-mode driver.
The original dense backing-port transfer is NOT invoked; instead, a 40-B/cycle
byte link transfers codebook bytes followed by index bytes. A bounded decoder
reconstructs logical weights into a separately budgeted staging buffer.
The native SRAM demand and IFMAP/OFMAP scheduler remain unchanged.

This is a narrow *timing model*, not a modeled physical SRAM organization:
- the native filter SRAM still uses one simulated element per logical address
- reconstructed FP32 weights require a separate explicitly budgeted staging area
- the software simulates no actual centroid arithmetic or faults
- all filter addresses must occur exactly once during a single initial prefetch
- subsequent filter SRAM misses/prefetches cause a loud error
- no bank, arbitration, port, decoder energy, or contention modeling
Do not use unmodified native bandwidth/energy fields as physical-byte metrics.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from live_filter_gate import gate_reads, GateStats
from trace_cluster_replay import Config, simulate_supply

INDEX_BYTE_BASE = 200_000_000  # synthetic physical byte IDs; NOT KMeans labels
CODEBOOK_BYTE_BASE = 300_000_000


def physical_transfer_trace(start: int, weight_positions: int, cfg: Config):
    """2D [cycle, physical-byte-addresses...] for compressed physical DRAM link."""
    count_index = math.ceil(weight_positions * cfg.index_bits / 8)
    count_codebook = cfg.k * cfg.centroid_bytes
    addresses = ([CODEBOOK_BYTE_BASE + i for i in range(count_codebook)] +
                 [INDEX_BYTE_BASE + i for i in range(count_index)])
    bw = cfg.bytes_per_cycle
    rows = math.ceil(len(addresses) / bw)
    result = np.full((rows, bw + 1), -1, dtype=np.int64)
    for r in range(rows):
        result[r, 0] = start + r
        items = addresses[r * bw:(r + 1) * bw]
        result[r, 1:len(items) + 1] = items
    return result


def logical_filter_addresses(matrix):
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError('Expected 2D native filter prefetch matrix')
    result = [int(x) for x in arr.flat if float(x) >= 0]
    if not result or len(result) != len(set(result)):
        raise ValueError('This v3 supports exactly one prefetch of unique logical weights')
    return result


@dataclass
class PrefetchStats:
    prefetch_calls: int = 0
    phys_bytes: int = 0
    decoded_staging_bytes: int = 0
    dense_bytes: int = 0
    compressed_fetch_start_cycle: int | None = None
    compressed_fetch_last_cycle: int | None = None
    decoded_ready_last_cycle: int | None = None
    filter_addresses: int = 0
    decoder: dict = field(default_factory=dict)
    gate: GateStats = field(default_factory=GateStats)


@contextmanager
def patch_compressed_conv1_prefetch(
    cfg: Config, stats: PrefetchStats, decoded_staging_bytes: int
):
    """Patch only memory-driver context, restore it even after failures.

    Must surround sim.run_scale() and requires USER mode. Not concurrency-safe.
    """
    cfg.validate()
    if decoded_staging_bytes <= 0:
        raise ValueError('decoded_staging_bytes must be positive')
    from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad
    cls = double_buffered_scratchpad
    native_driver = cls.service_memory_requests

    def experimental_driver(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        if self.estimate_bandwidth_mode:
            raise RuntimeError('Compressed filter prefetch v3 requires USER mode')
        buf = self.filter_buf
        old_prefetch = buf.prefetch_active_buffer
        old_reads = buf.service_reads
        old_new = buf.new_prefetch
        backing = buf.backing_buffer
        old_port_reads = backing.service_reads
        ready = {}

        def refuse_next_prefetch():
            raise RuntimeError('v3 supports one all-once conv1 prefetch only. '
                               'A filter cache refill was requested; model cannot handle this case.')

        def override_prefetch(start_cycle):
            if stats.prefetch_calls != 0:
                raise RuntimeError('v3 prefetch called more than once')
            # Construct the schedule from the live logical prefetch matrix,
            # never a stale previous trace.
            addresses = logical_filter_addresses(buf.fetch_matrix)
            n = len(addresses)
            if n != 9408:
                raise RuntimeError(f'v3 supports only conv1 9,408 weights; got {n}')
            if getattr(buf, 'use_ramulator_trace', False):
                raise RuntimeError('v3 does not support Ramulator')
            if getattr(buf, 'enable_layout_evaluation', False):
                raise RuntimeError('v3 has not validated layout/bank conflict modeling')
            # SCALE-Sim native read_buffer represents each logical address as
            # one element; ensure it can hold those logical addresses.
            if buf.active_buf_size < n:
                raise RuntimeError('Native active buffer cannot hold all logical conv1 addresses')
            # The compressed indices plus the (separate!) reconstructed FP32
            # staging space are accounted for explicitly, not assumed free.
            required = n * cfg.weight_bytes
            if decoded_staging_bytes < required:
                raise RuntimeError(f'Decoded staging requires {required} bytes; '
                                   f'configured {decoded_staging_bytes}')
            # Same look-ahead duration as the dense prefetch using the SAME
            # physical byte-bandwidth. Compression reduces occupied link cycles,
            # not the artificial time at which the prefetch is initiated.
            start = int(start_cycle) - math.ceil(n * cfg.weight_bytes / cfg.bytes_per_cycle)
            ready_map, simstats = simulate_supply(addresses, start, cfg)
            ready.update(ready_map)
            stats.prefetch_calls += 1
            stats.decoder = simstats
            stats.decoded_staging_bytes = required
            stats.dense_bytes = n * cfg.weight_bytes
            stats.filter_addresses = n
            phys = physical_transfer_trace(start, n, cfg)
            stats.phys_bytes = simstats['compressed_total_bytes']
            stats.compressed_fetch_start_cycle = int(phys[0, 0])
            stats.compressed_fetch_last_cycle = int(phys[-1, 0])
            stats.decoded_ready_last_cycle = max(ready.values())

            def decoded_response(incoming_cycles_arr, incoming_requests_arr_np):
                requests = np.asarray(incoming_requests_arr_np)
                if requests.ndim != 2 or requests.shape[1] != buf.req_gen_bandwidth:
                    raise RuntimeError('Unexpected native prefetch request shape')
                seen = [int(a) for a in requests.flat if float(a) >= 0]
                if seen != addresses:
                    raise RuntimeError('Unexpected native prefetch address order; refusing to approximate')
                return np.asarray([
                    [max((ready[int(a)] for a in row if float(a) >= 0), default=start)]
                    for row in requests
                ], dtype=float)

            # Run the upstream prefetch *state machine* to populate its active
            # logical weight-address cache, but replace the dense DRAM port call
            # so 9,408 native full-weight DRAM requests never execute.
            backing.service_reads = decoded_response
            try:
                old_prefetch(start_cycle)
            finally:
                backing.service_reads = old_port_reads
            # Externally visible filter DRAM trace is now a PHYSICAL-BYTE
            # transfer log: codebook byte addresses followed by index bytes.
            buf.trace_matrix = phys
            buf.num_access = stats.phys_bytes
            return 0

        def compressed_reads(incoming_requests_arr_np, incoming_cycles_arr):
            base = old_reads(incoming_requests_arr_np=incoming_requests_arr_np,
                             incoming_cycles_arr=incoming_cycles_arr)
            # old_reads triggers our compressed prefetch on first invocation.
            if not ready:
                raise RuntimeError('Compressed prefetch did not initialize before first read')
            # We already called old_reads, so avoid calling it twice via gate_reads.
            return gate_reads(lambda **kwargs: base, incoming_requests_arr_np,
                              incoming_cycles_arr, ready, buf.get_hit_latency(), stats.gate)

        buf.prefetch_active_buffer = override_prefetch
        buf.service_reads = compressed_reads
        buf.new_prefetch = refuse_next_prefetch
        try:
            result = native_driver(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat)
            if stats.prefetch_calls != 1 or stats.gate.accessed_addresses != set(ready):
                raise RuntimeError('v3 did not service every conv1 filter address')
            return result
        finally:
            buf.prefetch_active_buffer = old_prefetch
            buf.service_reads = old_reads
            buf.new_prefetch = old_new
            backing.service_reads = old_port_reads

    cls.service_memory_requests = experimental_driver
    try:
        yield
    finally:
        cls.service_memory_requests = native_driver
