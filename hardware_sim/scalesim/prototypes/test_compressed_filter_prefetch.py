import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from compressed_filter_prefetch import (
    PrefetchStats, logical_filter_addresses, patch_compressed_conv1_prefetch,
    physical_transfer_trace,
)
from trace_cluster_replay import Config


class MockPort:
    def __init__(self):
        self.calls = 0
    def service_reads(self, incoming_cycles_arr, incoming_requests_arr_np):
        self.calls += 1
        raise AssertionError('native dense DRAM backing service must not be called')
    def get_latency(self):
        return 0


class MockFilterBuffer:
    def __init__(self):
        self.backing_buffer = MockPort()
        self.req_gen_bandwidth = 10
        self.active_buf_size = 20000
        self.word_size = 1
        self.use_ramulator_trace = False
        self.enable_layout_evaluation = False
        self.fetch_matrix = np.full((941, 10), -1., dtype=float)
        self.fetch_matrix.flat[:9408] = np.arange(9408) + 10_000_000
        self.active_buf_full_flag = False
        self.num_access = 0
        self.trace_matrix = None
        self.prefetch_calls = 0
        self.new_calls = 0

    def get_hit_latency(self):
        return 1

    def prefetch_active_buffer(self, start_cycle):
        self.prefetch_calls += 1
        reqs = self.fetch_matrix
        cycles = np.arange(reqs.shape[0]).reshape(-1, 1)
        ready = self.backing_buffer.service_reads(
            incoming_cycles_arr=cycles, incoming_requests_arr_np=reqs)
        self.last_prefetch_cycle = int(np.max(ready))
        self.trace_matrix = np.column_stack((ready, reqs))
        self.num_access = len([a for a in reqs.flat if a >= 0])
        self.active_buf_full_flag = True
        return 0

    def new_prefetch(self):
        self.new_calls += 1
        raise AssertionError('unexpected extra prefetch')

    def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
        if not self.active_buf_full_flag:
            self.prefetch_active_buffer(incoming_cycles_arr[0][0])
        return np.asarray(incoming_cycles_arr) + 1


class MockMemory:
    def __init__(self):
        self.filter_buf = MockFilterBuffer()
        self.estimate_bandwidth_mode = False
    def service_memory_requests(self, inputs, filters, outputs):
        return self.filter_buf.service_reads(filters, inputs)


class PrefetchTests(unittest.TestCase):
    def test_physical_trace_counts_bytes_and_order(self):
        cfg = Config(bytes_per_cycle=40, decoder_width=32)
        trace = physical_transfer_trace(-941, 9408, cfg)
        self.assertEqual(trace.shape, (237, 41))
        addresses = [int(x) for x in trace[:, 1:].flat if x >= 0]
        self.assertEqual(len(addresses), 9472)
        self.assertEqual(len(set(addresses)), 9472)
        self.assertGreater(addresses[0], addresses[64])
        self.assertEqual(int(trace[0,0]), -941)
        self.assertEqual(int(trace[-1,0]), -705)

    def test_rejects_duplicate_logical_address(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            logical_filter_addresses(np.asarray([[1,1]]))

    def test_rejects_insufficient_staging_space(self):
        with self.assertRaises(ValueError):
            with patch_compressed_conv1_prefetch(Config(), PrefetchStats(), 0):
                pass

    def test_replaces_native_transfer_and_restores_hooks(self):
        module = types.ModuleType('scalesim.memory.double_buffered_scratchpad_mem')
        module.double_buffered_scratchpad = MockMemory
        saved_driver = MockMemory.service_memory_requests
        stats = PrefetchStats()
        modules = {'scalesim': types.ModuleType('scalesim'),
                   'scalesim.memory': types.ModuleType('scalesim.memory'),
                   'scalesim.memory.double_buffered_scratchpad_mem': module}
        with patch.dict(sys.modules, modules):
            with patch_compressed_conv1_prefetch(Config(), stats, 37632):
                mem = MockMemory()
                inputs = np.arange(9408).reshape(-1, 1)
                filters = (np.arange(9408) + 10_000_000).reshape(-1, 1)
                out = mem.service_memory_requests(inputs, filters, None)
            self.assertIs(MockMemory.service_memory_requests, saved_driver)
            self.assertEqual(mem.filter_buf.backing_buffer.calls, 0)
            self.assertEqual(mem.filter_buf.prefetch_calls, 1)
            self.assertEqual(mem.filter_buf.new_calls, 0)
            self.assertEqual(stats.prefetch_calls, 1)
            self.assertEqual(stats.phys_bytes, 9472)
            self.assertEqual(stats.dense_bytes, 37632)
            self.assertEqual(stats.decoded_staging_bytes, 37632)
            self.assertEqual(stats.gate.rows_delayed, 0)
            self.assertEqual(stats.gate.rows_seen, 9408)
            self.assertEqual(out.shape, (9408, 1))
            self.assertEqual(mem.filter_buf.trace_matrix.shape, (237, 41))
            # Ensures the native instance's method was restored even after patch.
            self.assertEqual(mem.filter_buf.prefetch_active_buffer.__func__,
                             MockFilterBuffer.prefetch_active_buffer)


if __name__ == '__main__':
    unittest.main()
