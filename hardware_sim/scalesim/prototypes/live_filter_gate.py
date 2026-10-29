"""Experimental SCALE-Sim v3 filter-decoder availability gate.

IMPORTANT: This is an *availability-gating integration*, NOT a replacement for
SCALE-Sim's dense filter DRAM model. The native filter buffer still schedules
and reports dense filter transfers. Do NOT describe these cycles as a true
compressed-memory or energy result. Prototype restricted to conv1, USER mode.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class GateStats:
    rows_seen: int = 0
    rows_with_weights: int = 0
    rows_delayed: int = 0
    largest_extra_wait_cycles: int = 0
    total_added_read_wait_cycles: int = 0  # sum of per-row floor adjustments, NOT inferred system stalls
    accessed_addresses: set[int] = field(default_factory=set)


def gate_reads(original_service, incoming_requests_arr_np, incoming_cycles_arr,
               ready_by_address: Mapping[int, int], hit_latency: int, stats: GateStats):
    """Return native SRAM completion cycles, floored by decoder availability.

    Preserves original buffer behavior and its internal bookkeeping, changing
    only the *returned* row completion cycles used by the outer memory scheduler.
    Designed for original USER-mode batch size one, but handles multiple rows.
    """
    baseline = np.asarray(original_service(incoming_requests_arr_np=incoming_requests_arr_np,
                                           incoming_cycles_arr=incoming_cycles_arr))
    demands = np.asarray(incoming_requests_arr_np)
    if demands.ndim != 2 or baseline.shape != (demands.shape[0], 1):
        raise ValueError(f'Unexpected SCALE-Sim shapes demand={demands.shape}, output={baseline.shape}')
    result = baseline.copy()
    for row_num, row in enumerate(demands):
        stats.rows_seen += 1
        addresses = [int(x) for x in row if float(x) >= 0]
        if not addresses:
            continue
        stats.rows_with_weights += 1
        stats.accessed_addresses.update(addresses)
        try:
            decoded_ready = max(ready_by_address[a] for a in addresses)
        except KeyError as exc:
            raise ValueError(f'Filter address {exc.args[0]} absent from conv1 decoder readiness map; '
                             'this prototype cannot be reused on other layers') from exc
        # Native SRAM cache-hit latency is still paid after a reconstructed
        # weight becomes available. Only delay; never speed up an original read.
        service_floor = decoded_ready + hit_latency
        native_time = float(result[row_num, 0])
        delta = max(0, service_floor - native_time)
        if delta > 0:
            stats.rows_delayed += 1
            stats.total_added_read_wait_cycles += int(round(delta))
            stats.largest_extra_wait_cycles = max(stats.largest_extra_wait_cycles, int(round(delta)))
            result[row_num, 0] = service_floor
    return result


@contextmanager
def patch_live_filter_reads(ready_by_address: Mapping[int, int], stats: GateStats):
    """Patch SCALE-Sim's USER-mode memory driver only inside this context.

    The native filter cache/DRAM model remains intact; the decoder merely
    adds read-availability constraints to the serviced timestamps. Dense mode
    should be run with this context NOT active.
    """
    from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad
    cls = double_buffered_scratchpad
    original_driver = cls.service_memory_requests
    if not ready_by_address:
        raise ValueError('Decoder availability map must not be empty')

    def patched_driver(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        if self.estimate_bandwidth_mode:
            raise RuntimeError('Decoder gating prototype only supports USER fixed-bandwidth mode')
        buf = self.filter_buf
        original_filter_reads = buf.service_reads
        hit_latency = int(buf.get_hit_latency())

        def gated_filter_reads(incoming_requests_arr_np, incoming_cycles_arr):
            return gate_reads(original_filter_reads, incoming_requests_arr_np,
                              incoming_cycles_arr, ready_by_address, hit_latency, stats)

        buf.service_reads = gated_filter_reads
        try:
            return original_driver(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat)
        finally:
            buf.service_reads = original_filter_reads

    cls.service_memory_requests = patched_driver
    try:
        yield
    finally:
        cls.service_memory_requests = original_driver
