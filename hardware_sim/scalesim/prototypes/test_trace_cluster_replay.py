import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_cluster_replay import Config, analyze, extract_order, parse_trace, simulate_supply


class TraceReplayTests(unittest.TestCase):
    def test_conv1_40_byte_bus(self):
        ready, metrics = simulate_supply(list(range(9408)), -941, Config())
        self.assertEqual(metrics['weight_positions'], 9408)
        self.assertEqual(metrics['dense_bytes'], 37632)
        self.assertEqual(metrics['compressed_total_bytes'], 9472)
        self.assertEqual(metrics['ideal_dense_transfer_cycles_at_same_physical_bandwidth'], 941)
        self.assertEqual(metrics['codebook_transfer_cycles'], 2)
        self.assertEqual(metrics['pipeline_cycles_after_codebook'], 294)
        self.assertEqual(metrics['total_prefetch_span_cycles'], 297)
        self.assertEqual(len(ready), 9408)

    def test_prefetch_ready_before_later_reads(self):
        dram = [(-5, (100, 101)), (-4, (102, 103))]
        sram = [(1, (100, 102)), (2, (101, 103)), (3, ())]
        r = analyze(dram, sram, Config(bytes_per_cycle=8, decoder_width=2, k=2))
        self.assertEqual(r['filter_sram_deadline_check']['rows_missing_original_deadline'], 0)
        self.assertEqual(r['filter_sram_deadline_check']['unique_sram_addresses'], 4)

    def test_slow_decoder_misses_deadline_and_is_not_claimed_total_cycles(self):
        dram = [(-3, (100, 101, 102, 103))]
        sram = [(-1, (100, 101, 102, 103))]
        r = analyze(dram, sram, Config(bytes_per_cycle=40, decoder_width=1, k=2))
        self.assertGreater(r['filter_sram_deadline_check']['rows_missing_original_deadline'], 0)
        self.assertGreater(r['filter_sram_deadline_check']['shifted_clock_delay_diagnostic_cycles'], 0)
        self.assertNotIn('total_inference_cycles', r)

    def test_repeated_dram_address_not_supported(self):
        with self.assertRaisesRegex(ValueError, 'Repeated DRAM'):
            extract_order([(0, (1, 2)), (1, (1,))])

    def test_sram_address_missing_from_dram(self):
        with self.assertRaisesRegex(ValueError, 'not found'):
            analyze([(0, (1,))], [(1, (2,))], Config())

    def test_decoder_bottleneck_and_queue(self):
        addrs = list(range(100))
        _, fast = simulate_supply(addrs, 0, Config(decoder_width=32, queue_capacity=1))
        _, slow = simulate_supply(addrs, 0, Config(decoder_width=1, queue_capacity=1))
        self.assertGreaterEqual(slow['total_prefetch_span_cycles'], fast['total_prefetch_span_cycles'])
        self.assertLessEqual(fast['peak_queue_after_decode'], 1)

    def test_trace_reader_handles_minus_one_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.csv'
            path.write_text('1,10,-1,11,\n2,-1,,\n3,12\n')
            self.assertEqual(list(parse_trace(path)), [(1, (10, 11)), (2, ()), (3, (12,))])

    def test_preserves_chronological_order(self):
        first, addrs = extract_order([(-2, (3,)), (-4, (1, 2)), (-1, (4,))])
        self.assertEqual((first, addrs), (-4, [1, 2, 3, 4]))

    def test_negative_prefetch_start_preserved(self):
        _, metrics = simulate_supply([100, 200], -941, Config())
        self.assertEqual(metrics['prefetch_start_cycle'], -941)
        self.assertLess(metrics['last_ready_cycle'], 0)


if __name__ == '__main__':
    unittest.main()
