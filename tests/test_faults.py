"""Offline unit checks for the Step 5 simulator and saved-pattern interface."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.memory.clustering import cluster_tensor, decode_state, physical_levels
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern
from fault_lora.evaluation.fault_simulation import FaultConfig


def small_encoding(k=4, repeats=1):
    record, _ = cluster_tensor(torch.arange(k, dtype=torch.float32).repeat(repeats), k)
    return {"format_version": 1, "clustered_tensors": {"weight": record}, "untouched_state": {"bias": torch.tensor([3.])}}


class FaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_faults_preserve_dense_state_and_source(self):
        encoding = small_encoding()
        fingerprint = encoding_fingerprint(encoding)
        pattern = generate_fault_pattern(encoding, 0)
        state = apply_fault_pattern(encoding, pattern)
        for name, value in decode_state(encoding).items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)
        self.assertEqual(summarize_fault_pattern(encoding, pattern)["actual_corrupted_cells"], 0)
        self.assertEqual(encoding_fingerprint(encoding), fingerprint)
        state["bias"][0] = 100
        self.assertEqual(encoding["untouched_state"]["bias"].item(), 3)

    def test_full_selection_has_valid_neighbors_and_inward_boundaries(self):
        for k in (2, 4, 256):
            with self.subTest(k=k):
                encoding = small_encoding(k, repeats=8)
                pattern = generate_fault_pattern(encoding, 1)
                entry = pattern["tensors"]["weight"]
                before, after = entry["before_levels"], entry["after_levels"]
                self.assertTrue(torch.all((after.to(torch.int16) - before.to(torch.int16)).abs() == 1))
                self.assertTrue(torch.all(after[before == 0] == 1))
                self.assertTrue(torch.all(after[before == k - 1] == k - 2))
                summary = summarize_fault_pattern(encoding, pattern)
                self.assertEqual(summary["actual_corrupted_cells"], k * 8)
                self.assertEqual(summary["upward_transitions"] + summary["downward_transitions"], k * 8)

    def test_faults_follow_physical_mapping_instead_of_index_number(self):
        encoding = small_encoding()
        record = encoding["clustered_tensors"]["weight"]
        record["index_to_level"] = torch.tensor([2, 0, 3, 1])
        record["level_to_index"] = torch.argsort(record["index_to_level"])
        pattern = generate_fault_pattern(encoding, 1)
        new_levels = pattern["tensors"]["weight"]["after_levels"]
        expected = record["codebook"][record["level_to_index"][new_levels.long()]]
        torch.testing.assert_close(apply_fault_pattern(encoding, pattern)["weight"], expected, rtol=0, atol=0)
        # Weight at index 1 occupies physical level 0; its neighbor is index 3.
        self.assertEqual(expected[1].item(), 3)

    def test_seeded_saved_pattern_replays_without_accumulation(self):
        encoding = small_encoding(repeats=1000)
        before = encoding_fingerprint(encoding)
        first = generate_fault_pattern(encoding, .2, seed=7)
        torch.manual_seed(999)  # Global RNG does not control this simulator.
        second = generate_fault_pattern(encoding, .2, seed=7)
        different = generate_fault_pattern(encoding, .2, seed=8)
        for key in ("positions", "before_levels", "after_levels"):
            torch.testing.assert_close(first["tensors"]["weight"][key], second["tensors"]["weight"][key], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["tensors"]["weight"]["positions"], different["tensors"]["weight"]["positions"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pattern.pt"
            torch.save(first, path)
            loaded = torch.load(path, weights_only=True, map_location="cpu")
            for name, value in apply_fault_pattern(encoding, first).items():
                torch.testing.assert_close(value, apply_fault_pattern(encoding, loaded)[name], rtol=0, atol=0)
        self.assertEqual(encoding_fingerprint(encoding), before)

    def test_single_level_exclusion_scope_and_equal_representative_counts(self):
        encoding = small_encoding()
        constant, _ = cluster_tensor(torch.ones(5), 16)
        encoding["clustered_tensors"]["constant"] = constant
        summary = summarize_fault_pattern(encoding, generate_fault_pattern(encoding, 1))
        self.assertEqual(summary["eligible_cells"], 4)
        self.assertEqual(summary["layers"]["constant"]["actual_corrupted_cells"], 0)
        only_constant = generate_fault_pattern(encoding, 1, tensor_names=["constant"])
        self.assertEqual(summarize_fault_pattern(encoding, only_constant)["eligible_cells"], 0)
        # A physical state substitution need not change a decoded value if representatives tie.
        encoding["clustered_tensors"]["weight"]["codebook"] = torch.zeros(4)
        summary = summarize_fault_pattern(encoding, generate_fault_pattern(encoding, 1))
        self.assertEqual(summary["actual_corrupted_cells"], 4)
        self.assertEqual(summary["decoded_weight_changes"], 0)

    def test_mismatched_baseline_and_malformed_transitions_are_rejected(self):
        encoding = small_encoding()
        pattern = generate_fault_pattern(encoding, 1)
        altered = copy.deepcopy(encoding)
        altered["clustered_tensors"]["weight"]["indices"][0] = 1
        with self.assertRaisesRegex(ValueError, "different uncorrupted"):
            apply_fault_pattern(altered, pattern)
        for mutation in ("duplicate", "out_of_range", "not_adjacent", "wrong_before"):
            invalid = copy.deepcopy(pattern)
            entry = invalid["tensors"]["weight"]
            if mutation == "duplicate":
                entry["positions"][1] = entry["positions"][0]
            elif mutation == "out_of_range":
                entry["after_levels"][0] = 4
            elif mutation == "not_adjacent":
                entry["after_levels"][0] = 2
            else:
                entry["before_levels"][0] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                apply_fault_pattern(encoding, invalid)

    def test_invalid_probability_seed_config_and_tensor_scope_fail(self):
        encoding = small_encoding()
        for probability in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                generate_fault_pattern(encoding, probability)
        for seed in (-1, 2**32, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_fault_pattern(encoding, .1, seed=seed)
        for names in ([], ["unknown"], ["weight", "weight"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                generate_fault_pattern(encoding, .1, tensor_names=names)
        for invalid in ({"run_id": "../other"}, {"cpu_threads": 0}, {"fault_probability": 2}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FaultConfig(**invalid).validate()


if __name__ == "__main__":
    unittest.main()
