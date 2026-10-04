"""Check that the independent Step 6 oracle catches corrupted output states."""

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.evaluation.fault_checks import FaultCheckConfig, inspect_transitions, patterns_equal, require_state_equal
from fault_lora.memory.clustering import cluster_tensor, decode_state
from fault_lora.memory.faults import apply_fault_pattern, generate_fault_pattern


class FaultAuditTests(unittest.TestCase):
    def setUp(self):
        record, _ = cluster_tensor(torch.arange(4, dtype=torch.float32), 4)
        self.encoding = {"format_version": 1, "clustered_tensors": {"weight": record}, "untouched_state": {"bias": torch.ones(1)}}

    def test_independent_oracle_accepts_nonidentity_mapping_and_scope(self):
        record = self.encoding["clustered_tensors"]["weight"]
        record["index_to_level"] = torch.tensor([2, 0, 3, 1])
        record["level_to_index"] = torch.argsort(record["index_to_level"])
        for p in (0, .3, 1):
            with self.subTest(p=p):
                pattern = generate_fault_pattern(self.encoding, p)
                result = inspect_transitions(self.encoding, pattern, apply_fault_pattern(self.encoding, pattern))
                self.assertEqual(result["eligible_cells"], 4)
                self.assertTrue(result["unselected_weights_and_reliable_state_match"])
        record2, _ = cluster_tensor(torch.ones(2), 2)
        self.encoding["clustered_tensors"]["constant"] = record2
        selected = generate_fault_pattern(self.encoding, 1, tensor_names=["weight"])
        result = inspect_transitions(self.encoding, selected, apply_fault_pattern(self.encoding, selected))
        self.assertEqual(result["all_state_tensors_checked"], 3)
        self.assertEqual(result["actual_corrupted_cells"], 4)

    def test_oracle_rejects_unselected_and_reliable_output_changes(self):
        pattern = generate_fault_pattern(self.encoding, 0)
        for name in ("weight", "bias"):
            invalid = decode_state(self.encoding)
            invalid[name].reshape(-1)[0] += 1
            with self.subTest(name=name), self.assertRaisesRegex(AssertionError, "tensor differs"):
                inspect_transitions(self.encoding, pattern, invalid)

    def test_oracle_rejects_wrong_selected_output(self):
        pattern = generate_fault_pattern(self.encoding, 1)
        invalid = apply_fault_pattern(self.encoding, pattern)
        invalid["weight"][0] = 100
        with self.assertRaisesRegex(AssertionError, "tensor differs"):
            inspect_transitions(self.encoding, pattern, invalid)

    def test_oracle_rejects_non_neighbor_even_with_matching_dense_values(self):
        pattern = generate_fault_pattern(self.encoding, 1)
        pattern["tensors"]["weight"]["after_levels"][0] = 2
        invalid = decode_state(self.encoding)
        invalid["weight"] = self.encoding["clustered_tensors"]["weight"]["codebook"][pattern["tensors"]["weight"]["after_levels"].long()]
        with self.assertRaisesRegex(AssertionError, "Non-neighbor"):
            inspect_transitions(self.encoding, pattern, invalid)

    def test_state_equality_rejects_missing_tensor_and_dtype_changes(self):
        baseline = decode_state(self.encoding)
        invalid = copy.deepcopy(baseline)
        invalid.pop("bias")
        with self.assertRaisesRegex(AssertionError, "names differ"):
            require_state_equal(invalid, baseline, "test")
        invalid = copy.deepcopy(baseline)
        invalid["weight"] = invalid["weight"].double()
        with self.assertRaisesRegex(AssertionError, "tensor differs"):
            require_state_equal(invalid, baseline, "test")

    def test_pattern_comparison_checks_destinations_not_only_positions(self):
        first = generate_fault_pattern(self.encoding, 1)
        second = copy.deepcopy(first)
        self.assertTrue(patterns_equal(first, second))
        second["tensors"]["weight"]["after_levels"][1] = 1
        self.assertFalse(patterns_equal(first, second))

    def test_invalid_audit_config_rejected(self):
        for kwargs in ({"cpu_threads": 0}, {"cpu_threads": True}, {"run_id": "../other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FaultCheckConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
