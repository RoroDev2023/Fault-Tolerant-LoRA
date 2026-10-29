import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conv1_cluster_supply import Assumptions, simulate


class ClusterSupplyTests(unittest.TestCase):
    def test_storage_conv1(self):
        r = simulate(Assumptions())
        self.assertEqual(r['weight_positions'], 9408)
        self.assertEqual(r['dense_weight_bytes'], 37632)
        self.assertEqual(r['cluster_index_bytes'], 9408)
        self.assertEqual(r['codebook_bytes'], 64)

    def test_same_cycles_for_silent_faults(self):
        r = simulate(Assumptions())
        self.assertEqual(r['clustered_weight_supply_cycles'], r['faulty_clustered_weight_supply_cycles'])

    def test_decoder_bottleneck(self):
        a = Assumptions()
        slow = simulate(replace(a, decoder_weights_per_cycle=4))
        fast = simulate(replace(a, decoder_weights_per_cycle=32))
        self.assertGreater(slow['clustered_weight_supply_cycles'], fast['clustered_weight_supply_cycles'])

    def test_bandwidth_bottleneck(self):
        a = Assumptions(decoder_weights_per_cycle=512)
        slow = simulate(replace(a, bandwidth_words_per_cycle=2))
        fast = simulate(replace(a, bandwidth_words_per_cycle=10))
        self.assertGreater(slow['clustered_weight_supply_cycles'], fast['clustered_weight_supply_cycles'])
        self.assertGreater(slow['dense_weight_supply_cycles'], fast['dense_weight_supply_cycles'])

    def test_cached_codebook_removes_initial_transfer_only(self):
        a = Assumptions()
        cold = simulate(a)
        warm = simulate(replace(a, codebook_cached=True))
        self.assertEqual(cold['clustered_weight_supply_cycles'] - warm['clustered_weight_supply_cycles'],
                         cold['codebook_preload_cycles'])

    def test_4bit_fewer_index_bytes(self):
        a = Assumptions()
        packed = simulate(replace(a, index_bits=4))
        standard = simulate(a)
        self.assertEqual(packed['cluster_index_bytes'], 4704)
        self.assertLess(packed['cluster_index_bytes'], standard['cluster_index_bytes'])

    def test_queue_bounds_and_completion(self):
        for cap in (1, 16, 256):
            r = simulate(Assumptions(queue_capacity_indices=cap))
            self.assertLessEqual(r['max_queued_indices'], cap)

    def test_validation(self):
        with self.assertRaises(ValueError):
            simulate(Assumptions(index_bits=2, k=16))


if __name__ == '__main__':
    unittest.main()
