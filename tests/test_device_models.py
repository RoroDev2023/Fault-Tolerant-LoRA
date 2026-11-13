"""Offline checks for level-dependent device fault models and their saved patterns."""

from pathlib import Path
import statistics
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.evaluation.fault_simulation import FaultConfig
from fault_lora.memory.clustering import cluster_tensor, decode_state, most_common_at_level0, physical_levels, with_level_order
from fault_lora.memory.device_models import LevelFaultModel, ctt_levels, device_model
from fault_lora.memory.faults import apply_fault_pattern, generate_device_fault_pattern, generate_fault_pattern, summarize_fault_pattern


def encoding_for(k, repeats):
    record, _ = cluster_tensor(torch.arange(k, dtype=torch.float32).repeat(repeats), k)
    return {"format_version": 1, "clustered_tensors": {"weight": record}, "untouched_state": {"bias": torch.tensor([3.])}}


def toy_model(k, down=0.1, up=0.2):
    p_down = tuple([0.0] + [down] * (k - 1))
    p_up = tuple([up] * (k - 1) + [0.0])
    return LevelFaultModel(f"toy_{k}", p_down, p_up, {"note": "test"})


class DeviceModelTests(unittest.TestCase):
    def test_reference_rates_match_the_documented_analysis(self):
        mean = lambda model: statistics.mean(model.fault_probability)
        self.assertAlmostEqual(mean(device_model("ctt_maxnvm2019", 16)), 0.0118, delta=0.0002)
        self.assertAlmostEqual(mean(device_model("ctt_dac2018", 16)), 0.0646, delta=0.0002)
        self.assertAlmostEqual(mean(device_model("ctt_maxnvm2019", 16, l0_gap_ua=30.0)), 0.00213, delta=0.00005)
        self.assertAlmostEqual(mean(device_model("rram_msxfi", 8)), 0.00381, delta=0.00002)
        self.assertAlmostEqual(mean(device_model("fefet_msxfi", 4)), 0.0110, delta=0.0002)
        self.assertAlmostEqual(mean(device_model("fefet_msxfi", 8)), 0.0814, delta=0.0002)
        self.assertLess(max(device_model("ctt_maxnvm2019", 8).fault_probability), 1e-6)

    def test_ctt_structure_protects_l0_and_has_one_sided_edges(self):
        model = device_model("ctt_maxnvm2019", 16)
        middle = model.fault_probability[8]
        self.assertLess(model.fault_probability[0], 1e-9)
        self.assertLess(model.p_down[1], 1e-9)                 # L1 cannot realistically reach L0
        self.assertAlmostEqual(model.p_up[1], middle / 2, delta=1e-6)
        self.assertAlmostEqual(model.p_down[15], middle / 2, delta=1e-6)
        self.assertAlmostEqual(model.p_down[8], model.p_up[8], delta=1e-12)
        self.assertEqual((model.p_down[0], model.p_up[15]), (0.0, 0.0))

    def test_read_noise_raises_fault_rates(self):
        rates = [statistics.mean(device_model("ctt_maxnvm2019", 16, read_noise_ua=n).fault_probability) for n in (0.0, 0.92, 1.29, 1.76)]
        self.assertEqual(rates, sorted(rates))
        self.assertAlmostEqual(rates[1], 0.0386, delta=0.0003)
        self.assertAlmostEqual(rates[3], 0.125, delta=0.001)

    def test_measured_levels_used_as_is_and_rebuilt_otherwise(self):
        self.assertEqual(ctt_levels(8)[3], (64.21, 1.39))
        rebuilt = ctt_levels(16)
        self.assertEqual(rebuilt[0], (144.70, 5.80))
        self.assertAlmostEqual(rebuilt[1][0], 94.39)
        self.assertAlmostEqual(rebuilt[-1][0], 4.91)
        self.assertAlmostEqual(rebuilt[5][1], (1.10 + 1.25 + 1.39 + 1.40 + 1.43 + 1.19) / 6)

    def test_invalid_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            device_model("rram_msxfi", 16)
        with self.assertRaises(ValueError):
            device_model("rram_msxfi", 8, read_noise_ua=1.0)
        with self.assertRaises(ValueError):
            device_model("sram", 16)
        with self.assertRaises(ValueError):
            device_model("ctt_maxnvm2019", 16, read_noise_ua=-1.0)
        with self.assertRaises(ValueError):
            LevelFaultModel("bad", (0.1, 0.0), (0.0, 0.0)).validate()
        with self.assertRaises(ValueError):
            LevelFaultModel("bad", (0.0, 0.6, 0.1), (0.2, 0.5, 0.0)).validate()   # level 1 totals 1.1

    def test_model_round_trips_through_a_dict(self):
        model = device_model("ctt_maxnvm2019", 16, read_noise_ua=0.5)
        self.assertEqual(LevelFaultModel.from_dict(model.to_dict()), model)


class DevicePatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampled_rates_and_directions_follow_the_model(self):
        k, repeats = 8, 20000
        encoding = encoding_for(k, repeats)
        pattern = generate_device_fault_pattern(encoding, toy_model(k), seed=3)
        entry = pattern["tensors"]["weight"]
        before, after = entry["before_levels"].long(), entry["after_levels"].long()
        for level in range(k):
            moved = before == level
            down = int((moved & (after < before)).sum())
            up = int((moved & (after > before)).sum())
            self.assertAlmostEqual(down / repeats, 0.0 if level == 0 else 0.1, delta=0.01)
            self.assertAlmostEqual(up / repeats, 0.0 if level == k - 1 else 0.2, delta=0.01)
        summary = summarize_fault_pattern(encoding, pattern)
        self.assertEqual(summary["fault_model"], "toy_8")
        self.assertAlmostEqual(summary["expected_corrupted_cells"], repeats * (0.2 + 0.3 * (k - 2) + 0.1))
        self.assertAlmostEqual(summary["realized_fault_rate"], summary["requested_fault_probability"], delta=0.005)

    def test_saved_pattern_replays_and_rejects_tampering(self):
        encoding = encoding_for(4, 500)
        first = generate_device_fault_pattern(encoding, toy_model(4), seed=11)
        second = generate_device_fault_pattern(encoding, toy_model(4), seed=11)
        for key in ("positions", "before_levels", "after_levels"):
            torch.testing.assert_close(first["tensors"]["weight"][key], second["tensors"]["weight"][key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pattern.pt"
            torch.save(first, path)
            loaded = torch.load(path, weights_only=True, map_location="cpu")
        self.assertEqual(summarize_fault_pattern(encoding, loaded), summarize_fault_pattern(encoding, first))
        state = apply_fault_pattern(encoding, loaded)
        self.assertTrue(torch.equal(state["bias"], encoding["untouched_state"]["bias"]))
        # A transition the model gives zero probability is rejected.
        never_up = LevelFaultModel("down_only", (0.0, 0.3, 0.3, 0.3), (0.0, 0.0, 0.0, 0.0))
        tampered = dict(first, fault_model=never_up.to_dict())
        if bool((first["tensors"]["weight"]["after_levels"] > first["tensors"]["weight"]["before_levels"]).any()):
            with self.assertRaises(ValueError):
                summarize_fault_pattern(encoding, tampered)

    def test_level_count_must_match_the_model(self):
        with self.assertRaises(ValueError):
            generate_device_fault_pattern(encoding_for(4, 10), toy_model(8))

    def test_uniform_summary_is_unchanged(self):
        encoding = encoding_for(4, 100)
        summary = summarize_fault_pattern(encoding, generate_fault_pattern(encoding, 0.2, seed=5))
        self.assertNotIn("fault_model", summary)
        self.assertNotIn("expected_corrupted_cells", summary["layers"]["weight"])
        self.assertEqual(summary["expected_corrupted_cells"], summary["eligible_cells"] * 0.2)


class LevelOrderTests(unittest.TestCase):
    def test_most_common_index_moves_to_level0_without_changing_weights(self):
        values = torch.tensor([0.0] * 5 + [1.0] * 5 + [2.0] * 50 + [3.0] * 5)
        record, _ = cluster_tensor(values, 4)
        order = most_common_at_level0(record)
        self.assertEqual(order, [2, 0, 1, 3])
        remapped = with_level_order(record, order)
        self.assertEqual(int(remapped["level_to_index"][0]), 2)
        self.assertTrue(torch.equal(physical_levels(remapped)[values == 2.0], torch.zeros(50, dtype=torch.uint8)))
        before = decode_state({"format_version": 1, "clustered_tensors": {"w": record}, "untouched_state": {}})
        after = decode_state({"format_version": 1, "clustered_tensors": {"w": remapped}, "untouched_state": {}})
        self.assertTrue(torch.equal(before["w"], after["w"]))
        with self.assertRaises(ValueError):
            with_level_order(record, [0, 0, 1, 2])


class ConfigTests(unittest.TestCase):
    def test_device_config_validates_and_old_configs_still_load(self):
        FaultConfig(fault_model="device", device="ctt_maxnvm2019", read_noise_ua=0.9).validate()
        FaultConfig(fault_probability=0.05).validate()
        with self.assertRaises(ValueError):
            FaultConfig(fault_model="device", device="sram").validate()
        with self.assertRaises(ValueError):
            FaultConfig(fault_model="other").validate()


if __name__ == "__main__":
    unittest.main()
