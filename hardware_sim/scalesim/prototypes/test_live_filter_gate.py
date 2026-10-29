import unittest
import sys
import types
from unittest.mock import patch
import numpy as np

from live_filter_gate import GateStats, gate_reads, patch_live_filter_reads
from run_conv1_live_gate import extract_single_layer, load_cycle_report
from pathlib import Path
from tempfile import TemporaryDirectory


class FakeFilterBuffer:
    def __init__(self, cycle_add=1):
        self.cycle_add = cycle_add
        self.called = 0

    def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
        self.called += 1
        return np.asarray(incoming_cycles_arr) + self.cycle_add


class LiveGateTests(unittest.TestCase):
    def test_dense_ready_preserves_native_cycles(self):
        buf = FakeFilterBuffer()
        stats = GateStats()
        output = gate_reads(buf.service_reads,
                            np.asarray([[7, 8, -1], [9, -1, -1]]),
                            np.asarray([[100], [101]]),
                            {7: -8, 8: 20, 9: 50}, 1, stats)
        np.testing.assert_array_equal(output, [[101], [102]])
        self.assertEqual(stats.rows_delayed, 0)
        self.assertEqual(stats.accessed_addresses, {7, 8, 9})

    def test_slow_decoder_causes_one_delayed_read(self):
        buf = FakeFilterBuffer()
        stats = GateStats()
        output = gate_reads(buf.service_reads,
                            np.asarray([[7, 8, -1], [9, -1, -1]]),
                            np.asarray([[100], [101]]),
                            {7: 50, 8: 110, 9: 90}, 1, stats)
        np.testing.assert_array_equal(output, [[111], [102]])
        self.assertEqual(stats.rows_delayed, 1)
        self.assertEqual(stats.largest_extra_wait_cycles, 10)

    def test_missing_address_fails_loudly(self):
        buf = FakeFilterBuffer()
        with self.assertRaisesRegex(ValueError, 'absent from conv1'):
            gate_reads(buf.service_reads, np.asarray([[99]]), np.asarray([[0]]),
                       {7: 0}, 1, GateStats())

    def test_empty_demand_preserves_native_cycle(self):
        buf = FakeFilterBuffer()
        s = GateStats()
        out = gate_reads(buf.service_reads, np.asarray([[-1, -1]]), np.asarray([[5]]),
                         {7: 0}, 1, s)
        np.testing.assert_array_equal(out, [[6]])
        self.assertEqual(s.rows_with_weights, 0)

    def test_live_driver_patch_reverts_and_gates(self):
        module = types.ModuleType('scalesim.memory.double_buffered_scratchpad_mem')
        class FakeMemory:
            def __init__(self):
                self.filter_buf = FakeFilterBuffer()
                self.filter_buf.get_hit_latency = lambda: 1
                self.estimate_bandwidth_mode = False
            def service_memory_requests(self, a, b, c):
                return self.filter_buf.service_reads(b, a)
        module.double_buffered_scratchpad = FakeMemory
        original_driver = FakeMemory.service_memory_requests
        stats = GateStats()
        with patch.dict(sys.modules, {'scalesim': types.ModuleType('scalesim'),
                                      'scalesim.memory': types.ModuleType('scalesim.memory'),
                                      'scalesim.memory.double_buffered_scratchpad_mem': module}):
            with patch_live_filter_reads({7: 30}, stats):
                mem = FakeMemory()
                answer = mem.service_memory_requests(np.asarray([[2]]), np.asarray([[7]]), None)
                np.testing.assert_array_equal(answer, [[31]])
            self.assertIs(FakeMemory.service_memory_requests, original_driver)
            np.testing.assert_array_equal(mem.service_memory_requests(np.asarray([[2]]), np.asarray([[7]]), None), [[3]])
        self.assertEqual(stats.rows_delayed, 1)

    def test_conv1_extraction_only_one_layer(self):
        with TemporaryDirectory() as d:
            d = Path(d)
            csv = d/'input.csv'
            csv.write_text('Layer Name, M, N, K,\nconv1,2304,64,147,\nnext,576,64,576,\n')
            output = d/'conv1.csv'
            info = extract_single_layer(csv, output)
            self.assertEqual(info['weights'], 9408)
            self.assertEqual(len(output.read_text().splitlines()), 2)

    def test_report_reader_fields(self):
        with TemporaryDirectory() as d:
            d = Path(d)
            csv = d/'COMPUTE_REPORT.csv'
            csv.write_text('Layer, Total Cycles, Stall Cycles, Total Cycles incl. prefetch\nconv1,113860,81691,114801\n')
            report = load_cycle_report(d)
            self.assertEqual(report['total_cycles_native'], 113860)
            self.assertEqual(report['stall_cycles_native'], 81691)
            self.assertEqual(report['prefetch_inclusive_cycles_native'], 114801)


if __name__ == '__main__':
    unittest.main()
