"""Offline checks for Step 13a: budget -> cells -> cell fault probability."""

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.memory.fault_rate_estimate import (BITS_PER_MB, bitstream_occupancy, budget_rows, cells_needed, gaussian_level_errors, index_bits, main,
                                                   q_function, sigma_for_target, storage_rows, table_level_errors)

ROOT = Path(__file__).resolve().parents[1]


def tiny_encoding(levels_by_name, k=16):
    records = {}
    for name, levels in levels_by_name.items():
        indices = torch.tensor(levels, dtype=torch.uint8).reshape(2, -1)
        records[name] = {"codebook": torch.linspace(-1, 1, k), "indices": indices, "dtype": "torch.float32", "shape": list(indices.shape),
                         "index_to_level": torch.arange(k), "level_to_index": torch.arange(k)}
    return {"format_version": 1, "clustered_tensors": records, "untouched_state": {}}


class FaultRateMathTests(unittest.TestCase):
    def test_q_function_values(self):
        self.assertAlmostEqual(q_function(0), 0.5, places=15)
        self.assertAlmostEqual(q_function(1), 0.15865525393145707, places=12)
        self.assertAlmostEqual(q_function(3), 0.0013498980316301, places=12)
        self.assertAlmostEqual(q_function(-1), 1 - q_function(1), places=15)
        self.assertGreater(q_function(30), 0)  # erfc keeps tiny tails

    def test_spacing_edges_and_level0_safe(self):
        sigma = 0.02
        errors = gaussian_level_errors(4, sigma)
        self.assertEqual(errors["levels"], 16)
        self.assertAlmostEqual(errors["spacing_over_window"], 1 / 15)
        one = q_function((1 / 15) / (2 * sigma))
        self.assertEqual((errors["down"][0], errors["up"][15]), (0.0, 0.0))
        self.assertAlmostEqual(errors["up"][0], one)
        self.assertAlmostEqual(errors["down"][15], one)
        for level in range(1, 15):
            self.assertAlmostEqual(errors["fault"][level], 2 * one)
        self.assertAlmostEqual(errors["two_level_jump"][1], q_function(3 * (1 / 15) / (2 * sigma)))
        self.assertAlmostEqual(errors["two_level_jump"][0], errors["two_level_jump"][1])  # level 0 and 1 can only jump two up
        self.assertAlmostEqual(errors["two_level_jump"][5], 2 * q_function(3 * (1 / 15) / (2 * sigma)))
        single = gaussian_level_errors(1, sigma)
        self.assertEqual(single["fault"], [single["up"][0], single["down"][1]])
        self.assertAlmostEqual(single["fault"][0], q_function(1 / (2 * sigma)))
        safe = gaussian_level_errors(4, sigma, level0_safe=True)
        self.assertEqual((safe["down"][0], safe["up"][0], safe["fault"][0]), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(safe["down"][1], one)  # level 1 can still be read as level 0
        self.assertEqual(safe["fault"][1:], errors["fault"][1:])
        # More levels in the same window -> closer spacing -> more faults.
        rates = [gaussian_level_errors(b, sigma)["fault"][1 if b > 1 else 0] for b in (1, 2, 3, 4)]
        self.assertEqual(rates, sorted(rates))
        with self.assertRaises(ValueError):
            gaussian_level_errors(4, 0)

    def test_explicit_table(self):
        table = {"down": [0, 0.01, 0.02, 0.03], "up": [0.04, 0.05, 0.06, 0]}
        errors = table_level_errors(table, 2)
        self.assertTrue(np.allclose(errors["fault"], [0.04, 0.06, 0.08, 0.03]))
        self.assertEqual(table_level_errors(table, 2, level0_safe=True)["fault"][0], 0.0)
        for bad in ({"down": [0.1, 0, 0, 0], "up": [0, 0, 0, 0]}, {"down": [0, 0, 0], "up": [0, 0, 0]}, {"down": [0, 0.7, 0, 0], "up": [0, 0.7, 0, 0]}):
            with self.assertRaises(ValueError):
                table_level_errors(bad, 2)

    def test_cells_fit_and_headroom(self):
        self.assertEqual([index_bits(k) for k in (2, 8, 9, 16, 17)], [1, 3, 4, 4, 5])
        self.assertEqual([cells_needed(10, 16, b) for b in (1, 2, 3, 4)], [40, 20, 14, 10])
        storage = storage_rows(10, 16)
        self.assertEqual([item["runnable_now"] for item in storage], [False, False, False, True])
        self.assertEqual(storage[2]["stored_bits"], 42)
        # A budget of exactly the 2-bit cell count: under slc-area b >= 2 fits; under capacity, every b needs >= 40 bits.
        budget = 20 / BITS_PER_MB
        rows = {(r["budget_meaning"], r["bits_per_cell"]): r for r in budget_rows(storage, budget)}
        self.assertEqual([rows[("slc-area", b)]["fits"] for b in (1, 2, 3, 4)], [False, True, True, True])
        self.assertAlmostEqual(rows[("slc-area", 2)]["headroom"], 1.0)
        self.assertAlmostEqual(rows[("slc-area", 4)]["headroom"], 2.0)
        self.assertEqual(rows[("slc-area", 1)]["fewest_levels_that_fit"], 2)
        self.assertEqual([rows[("capacity", b)]["fits"] for b in (1, 2, 3, 4)], [False] * 4)
        self.assertIsNone(rows[("capacity", 1)]["fewest_levels_that_fit"])
        rows = {(r["budget_meaning"], r["bits_per_cell"]): r for r in budget_rows(storage, 41 / BITS_PER_MB)}
        self.assertEqual([rows[("capacity", b)]["fits"] for b in (1, 2, 3, 4)], [True, True, False, True])  # b = 3 pads to 42 bits

    def test_bitstream_occupancy_and_sigma_inversion(self):
        levels = [np.array([0, 15, 5, 10], dtype=np.uint8), np.array([3], dtype=np.uint8)]
        self.assertEqual(bitstream_occupancy(levels, 4, 4), [1 / 5 if v in (0, 15, 5, 10, 3) else 0 for v in range(16)])
        # 0000 1111 0101 1010 0011 -> 2-bit cells 00 00 11 11 01 01 10 10 00 11
        self.assertEqual(bitstream_occupancy(levels, 4, 2), [0.3, 0.2, 0.2, 0.3])
        self.assertAlmostEqual(sum(bitstream_occupancy(levels, 4, 3)), 1.0)  # 20 bits -> 7 cells, last padded
        for b in (2, 4):
            sigma = sigma_for_target(b, 0.01)
            rate = np.mean(gaussian_level_errors(b, sigma)["fault"])
            self.assertTrue(math.isclose(rate, 0.01, rel_tol=1e-9))
        self.assertIsNone(sigma_for_target(4, 0.99))


class FaultRateReportTests(unittest.TestCase):
    def run_main(self, root, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(root, ["--config", "configs/fault_rate_estimate.json", *args])

    def test_budget_changes_only_budget_outputs_and_runs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "scripts").mkdir()
            (root / "src/fault_lora/memory").mkdir(parents=True)
            for name in ("scripts/estimate_fault_rate.py", "src/fault_lora/memory/fault_rate_estimate.py"):
                (root / name).write_text((ROOT / name).read_text())
            levels = {"b.weight": [0, 1, 2, 3, 7, 7], "a.weight": [15, 14, 8, 8, 8, 9, 1, 2]}
            torch.save(tiny_encoding(levels), root / "enc.pt")
            config = json.loads((ROOT / "configs/fault_rate_estimate.json").read_text())
            config["encodings"] = {name: "enc.pt" for name in config["encodings"]}
            (root / "configs/fault_rate_estimate.json").write_text(json.dumps(config))
            first = self.run_main(root, "--model", "resnet18", "--budget-mb", "128")
            second = self.run_main(root, "--model", "resnet18", "--budget-mb", "0.000001", "--run-id", "second")
            self.assertEqual(first["budget_independent"]["num_weights"], 14)
            self.assertEqual(first["budget_independent"]["encoding"]["index_to_level"], "identity")
            # Sorted tensor order: a.weight first.
            expected = np.bincount(np.array(levels["a.weight"] + levels["b.weight"]), minlength=16) / 14
            self.assertTrue(np.allclose(first["budget_independent"]["fault_probabilities"][3]["occupancy"], expected))
            self.assertEqual(first["budget_independent"], second["budget_independent"])
            self.assertNotEqual(first["budgets"], second["budgets"])
            self.assertIn("1e-06", second["budgets"])
            self.assertFalse(any(row["fits"] for row in second["budgets"]["1e-06"]))
            self.assertEqual(set(first["budgets"]), {"8", "32", "128", "512", "4096"})
            results = root / "results/fault-rate-estimate-resnet18-128mb"
            for name in ("report.json", "budgets.csv", "fault_probabilities.csv", "report.txt"):
                self.assertTrue((results / name).stat().st_size > 0)
            self.assertEqual(json.loads((results / "report.json").read_text())["inputs"]["resolved"]["budget_mb"], 128)
            with self.assertRaises(FileExistsError):
                self.run_main(root, "--model", "resnet18", "--budget-mb", "128")
            custom = self.run_main(root, "--model", "custom", "--num-weights", "7000000000", "--sigma-over-window", "0.01")
            self.assertIsNone(custom["budget_independent"]["fault_probabilities"][3]["average_fault_occupancy"])
            self.assertEqual(len(custom["budget_independent"]["fault_probabilities"]), 4)
            by = {(r["budget_meaning"], r["bits_per_cell"]): r["fits"] for r in custom["budgets"]["512"]}
            self.assertFalse(any(by.values()))
            self.assertTrue(all(r["fits"] for r in custom["budgets"]["4096"]))
            safe = self.run_main(root, "--model", "resnet18", "--level0-safe", "--sigma-over-window", "0.02", "--run-id", "safe")
            self.assertEqual(safe["budget_independent"]["fault_probabilities"][3]["fault"][0], 0.0)
            for bad in (["--model", "custom"], ["--model", "resnet18", "--num-weights", "5"], ["--model", "resnet18", "--clusters", "8"],
                        ["--model", "resnet18", "--budget-mb", "0"]):
                with self.assertRaises(ValueError):
                    self.run_main(root, *bad, "--run-id", "bad")


if __name__ == "__main__":
    unittest.main()
