"""Offline checks for Step 13b: level-dependent fault patterns, the frequent-at-0 mapping, and the evaluation sweep."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fault_lora.evaluation.fault_checks import inspect_transitions, patterns_equal, require_state_equal
from fault_lora.evaluation.level_fault_sweep import LevelFaultSweepConfig, aggregate, audit_pattern, plan, run_level_fault_sweep, verify
from fault_lora.memory.clustering import cluster_model, decode_state, physical_levels
from fault_lora.memory.fault_rate_estimate import gaussian_level_errors
from fault_lora.memory.faults import encoding_fingerprint
from fault_lora.memory.level_faults import (apply_level_fault_pattern, frequent_at_zero_mapping, gaussian_level_probabilities, generate_level_fault_pattern,
                                            mapping_report, per_level_z_scores, remap_encoding, summarize_level_fault_pattern)
from test_lora_matrix import TinyCIFAR


def synthetic_encoding(cells=64000, seed=0):
    """One 16-level tensor with a peaked level histogram (most frequent index 9)."""
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.ones(16) / 16 + torch.nn.functional.one_hot(torch.tensor(9), 16) * .3
    indices = torch.multinomial(probabilities / probabilities.sum(), cells, replacement=True, generator=generator).to(torch.uint8).reshape(-1, 40)
    record = {"codebook": torch.linspace(-1.5, 1.5, 16), "indices": indices, "dtype": "torch.float32", "shape": list(indices.shape),
              "index_to_level": torch.arange(16), "level_to_index": torch.arange(16)}
    return {"format_version": 1, "clustered_tensors": {"layer.weight": record}, "untouched_state": {"layer.bias": torch.zeros(3)}}


class LevelFaultTests(unittest.TestCase):
    def test_probabilities_follow_the_step13a_model(self):
        p = gaussian_level_probabilities(.02)
        errors = gaussian_level_errors(4, .02)
        self.assertEqual((p["down"], p["up"]), (errors["down"], errors["up"]))
        safe = gaussian_level_probabilities(.02, level0_safe=True)
        self.assertEqual((safe["down"][0], safe["up"][0]), (0.0, 0.0))
        self.assertEqual(safe["down"][1], p["down"][1])
        with self.assertRaises(ValueError):
            gaussian_level_probabilities(.02, 12)

    def test_patterns_are_adjacent_seeded_level_dependent_and_protect_level0(self):
        encoding = synthetic_encoding()
        sorted_p, safe_p = gaussian_level_probabilities(.04), gaussian_level_probabilities(.04, level0_safe=True)
        pattern = generate_level_fault_pattern(encoding, sorted_p, seed=7)
        self.assertTrue(patterns_equal(pattern, generate_level_fault_pattern(encoding, sorted_p, seed=7)))
        self.assertFalse(patterns_equal(pattern, generate_level_fault_pattern(encoding, sorted_p, seed=8)))
        entry = pattern["tensors"]["layer.weight"]
        self.assertTrue(torch.all((entry["after_levels"].long() - entry["before_levels"].long()).abs() == 1))
        self.assertTrue(torch.all(entry["after_levels"][entry["before_levels"] == 0] == 1))
        self.assertTrue(torch.all(entry["after_levels"][entry["before_levels"] == 15] == 14))
        summary = summarize_level_fault_pattern(encoding, pattern)
        rows = per_level_z_scores(summary)
        self.assertLess(max(abs(row["z"]) for row in rows if row["z"] is not None), 5)
        self.assertEqual([row["z"] for row in rows if row["level"] == 0 and row["direction"] == "down"], [None])
        self.assertGreater(summary["per_level"]["down_transitions"][0] + summary["per_level"]["up_transitions"][0], 0)
        safe = generate_level_fault_pattern(encoding, safe_p, seed=7)
        safe_entry = safe["tensors"]["layer.weight"]
        self.assertEqual(int((safe_entry["before_levels"] == 0).sum()), 0)
        # One uniform per cell: the protected pattern is the unprotected one without its level-0 cells.
        keep = entry["before_levels"] != 0
        for key in ("positions", "before_levels", "after_levels"):
            self.assertTrue(torch.equal(safe_entry[key], entry[key][keep]))
        state = apply_level_fault_pattern(encoding, pattern)
        oracle = inspect_transitions(encoding, pattern, state)
        self.assertEqual(oracle["actual_corrupted_cells"], summary["actual_corrupted_cells"])
        levels = physical_levels(encoding["clustered_tensors"]["layer.weight"]).reshape(-1)
        self.assertAlmostEqual(summary["requested_average_fault_probability"],
                               sum((sorted_p["down"][l] + sorted_p["up"][l]) for l in levels.tolist()) / levels.numel(), places=12)

    def test_tampered_or_mismatched_patterns_are_rejected(self):
        encoding = synthetic_encoding(4000)
        pattern = generate_level_fault_pattern(encoding, gaussian_level_probabilities(.05, level0_safe=True), seed=3)
        entry = pattern["tensors"]["layer.weight"]
        for name, change in (("two-level jump", lambda p: p["tensors"]["layer.weight"]["after_levels"].__setitem__(0, (int(p["tensors"]["layer.weight"]["before_levels"][0]) + 2) % 16)),
                             ("wrong original", lambda p: p["tensors"]["layer.weight"]["before_levels"].__setitem__(0, (int(p["tensors"]["layer.weight"]["before_levels"][0]) + 1) % 15)),
                             ("wrong kind", lambda p: p.__setitem__("kind", "static_adjacent_physical_levels_v1"))):
            tampered = copy.deepcopy(pattern)
            change(tampered)
            with self.subTest(name), self.assertRaises(ValueError):
                summarize_level_fault_pattern(encoding, tampered)
        # A level-0 transition is impossible under protection.
        level0 = (physical_levels(encoding["clustered_tensors"]["layer.weight"]).reshape(-1) == 0).nonzero().flatten()[0]
        tampered = copy.deepcopy(pattern)
        order = torch.argsort(torch.cat([entry["positions"], level0.view(1)]))
        for key, value in (("positions", level0.view(1)), ("before_levels", torch.tensor([0], dtype=torch.uint8)), ("after_levels", torch.tensor([1], dtype=torch.uint8))):
            tampered["tensors"]["layer.weight"][key] = torch.cat([entry[key], value])[order]
        with self.assertRaises(ValueError):
            summarize_level_fault_pattern(encoding, tampered)
        with self.assertRaises(ValueError):
            generate_level_fault_pattern(encoding, {"down": [0.1] + [0.0] * 15, "up": [0.0] * 16}, seed=1)

    def test_frequent_at_zero_mapping_keeps_decoding_and_rebinds_patterns(self):
        encoding = synthetic_encoding()
        frequent = remap_encoding(encoding, "frequent-at-0")
        record = frequent["clustered_tensors"]["layer.weight"]
        self.assertEqual(record["level_to_index"].tolist(), [9] + [i for i in range(16) if i != 9])
        self.assertTrue(torch.equal(record["level_to_index"][record["index_to_level"]], torch.arange(16)))
        self.assertTrue(torch.equal(record["indices"], encoding["clustered_tensors"]["layer.weight"]["indices"]))
        require_state_equal(decode_state(frequent), decode_state(encoding), "fault-free decoding")
        self.assertEqual(encoding_fingerprint(remap_encoding(encoding, "sorted")), encoding_fingerprint(encoding))
        self.assertNotEqual(encoding_fingerprint(frequent), encoding_fingerprint(encoding))
        counts = torch.bincount(encoding["clustered_tensors"]["layer.weight"]["indices"].reshape(-1).long(), minlength=16)
        self.assertEqual(frequent_at_zero_mapping(encoding["clustered_tensors"]["layer.weight"])[2], int(counts.argmax()))
        row = mapping_report(encoding, frequent)[0]
        self.assertAlmostEqual(row["frequent_level0_share"], counts[9].item() / counts.sum().item())
        self.assertAlmostEqual(row["sorted_level0_share"], counts[0].item() / counts.sum().item())
        self.assertAlmostEqual(row["jump_level1_to_0_frequent"], 9 * 0.2, places=5)
        self.assertAlmostEqual(row["sorted_median_neighbor_gap"], 0.2, places=5)
        pattern = generate_level_fault_pattern(frequent, gaussian_level_probabilities(.04, level0_safe=True), seed=1)
        self.assertEqual(int((pattern["tensors"]["layer.weight"]["before_levels"] == 0).sum()), 0)
        self.assertEqual(int((physical_levels(record).reshape(-1) == 0).sum()), counts[9].item())
        with self.assertRaises(ValueError):
            summarize_level_fault_pattern(encoding, pattern)
        # A level-1 -> 0 read now decodes to the frequent cluster's value.
        state = apply_level_fault_pattern(frequent, pattern)
        entry = pattern["tensors"]["layer.weight"]
        downs = entry["positions"][(entry["before_levels"] == 1) & (entry["after_levels"] == 0)]
        self.assertGreater(downs.numel(), 0)
        self.assertTrue(torch.all(state["layer.weight"].reshape(-1)[downs] == record["codebook"][9]))
        with self.assertRaises(ValueError):
            remap_encoding(encoding, "gray")


def tiny_project(root):
    torch.manual_seed(0)
    template = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))
    encoding, _ = cluster_model(template, 16, threads=1)
    split = {"train": list(range(20)), "validation": list(range(20, 30)), "test": {"indices": list(range(20))}}
    metadata = {"num_classes": 10, "classes": TinyCIFAR.classes, "preprocessing": {"input_size": [3, 32, 32]}, "split_sha256": "synthetic-test-only"}
    (root / "src/fault_lora").mkdir(parents=True)
    (root / "scripts").mkdir()
    for filename in ("scripts/run_level_fault_sweep.py", "requirements.txt", "requirements-macos-arm64.lock.txt"):
        (root / filename).write_text("synthetic test input\n")
    torch.save(encoding, root / "encoding.pt")
    from fault_lora.evaluation.classification import evaluate_classifier
    from torch.utils.data import DataLoader, Subset
    dataset = TinyCIFAR("unused", True, None, False)

    def measure(state, name):
        model = copy.deepcopy(template)
        model.load_state_dict(state)
        indices = split["validation"] if name == "validation" else split["test"]["indices"]
        return evaluate_classifier(model, DataLoader(Subset(dataset if name == "validation" else TinyCIFAR("unused", False, None, False), indices), batch_size=10), "cpu")

    zero = {name: measure(decode_state(encoding), name) for name in ("validation", "test")}
    (root / "results/tiny-11c").mkdir(parents=True)
    (root / "results/tiny-11c/measurements.json").write_text(json.dumps([{"condition_id": "retrained-clustered", "after": zero}]))

    def prepare(project_root, config):
        record = {"pattern_id": None, "seed": None, "probability": 0.0, "counts": {"actual_corrupted_cells": 0, "realized_fault_rate": 0.0}, "measurements": zero}
        from fault_lora.evaluation.clustered_baseline import sha256_file
        return encoding, metadata, split, record, {}, {"encoding": root / "encoding.pt"}, {"encoding": sha256_file(root / "encoding.pt")}

    return template, encoding, prepare


class LevelFaultSweepTests(unittest.TestCase):
    def test_offline_sweep_is_audited_verified_and_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, encoding, prepare = tiny_project(root)
            options = dict(sigmas=[0.03, 0.05], seeds=[5, 6], z_threshold=8.0, step11c_bn_results="results/tiny-11c", calibration_examples=20, calibration_batch_size=10,
                           evaluation_batch_size=10, device="cpu", num_workers=0, cpu_threads=1)
            config = LevelFaultSweepConfig(run_id="tiny-13b", **options)
            with patch("fault_lora.evaluation.level_fault_sweep.prepare_retrained_sources", side_effect=prepare), \
                    patch("fault_lora.evaluation.level_fault_sweep.CIFAR10", TinyCIFAR), \
                    patch("fault_lora.evaluation.level_fault_sweep.build_resnet18", side_effect=lambda *a, **k: copy.deepcopy(template)):
                records = run_level_fault_sweep(config, root)
                with self.assertRaises(FileExistsError):
                    run_level_fault_sweep(config, root)
            results = root / "results/tiny-13b"
            self.assertEqual(json.loads((results / "status.json").read_text())["status"], "complete")
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            self.assertEqual([r["record_id"] for r in records], [e["record_id"] for e in plan(config)])
            self.assertEqual(len(records), 2 * 2 * 3)
            reference = json.loads((results / "reference.json").read_text())
            self.assertEqual(reference["bn_recalibrated_num_correct_difference_from_step11c"]["validation"],
                             reference["bn_recalibrated"]["validation"]["num_correct"] - reference["no_repair"]["validation"]["num_correct"])
            by_id = {r["record_id"]: r for r in records}
            for record in records:
                audit = json.loads((results / "audit" / f"{record['record_id']}.json").read_text())
                self.assertTrue(audit["passed"])
                if record["level0_safe"]:
                    self.assertTrue(audit["checks"]["protected_level0_never_faults"]["passed"])
                pattern = torch.load(root / record["pattern"], weights_only=True)
                self.assertEqual(pattern["kind"], "static_level_dependent_adjacent_levels_v1")
            summary = aggregate(records, reference, config)
            group = next(g for g in summary["groups"] if g["sigma_over_window"] == 0.05 and g["condition_id"] == "frequent-l0safe")
            pairs = [100 * (by_id[f"sw0p05-frequent-l0safe-seed{s}"]["no_repair"]["test"]["accuracy"] - by_id[f"sw0p05-sorted-seed{s}"]["no_repair"]["test"]["accuracy"]) for s in (5, 6)]
            self.assertAlmostEqual(group["splits"]["test"]["minus_sorted"]["no_repair_percentage_points"]["mean"], sum(pairs) / 2)
            self.assertTrue((results / "level_faults.png").stat().st_size > 1000)
            tampered = json.loads((results / "measurements.json").read_text())
            tampered[0]["no_repair"]["test"]["accuracy"] += .1
            (results / "measurements.json").write_text(json.dumps(tampered))
            with self.assertRaises(AssertionError):
                verify(root, config)

    def test_audit_rejects_a_saved_copy_that_differs(self):
        encoding = synthetic_encoding(4000)
        other = remap_encoding(encoding, "frequent-at-0")
        pattern = generate_level_fault_pattern(encoding, gaussian_level_probabilities(.05), seed=2)
        state = apply_level_fault_pattern(encoding, pattern)
        audit_pattern(encoding, other, pattern, copy.deepcopy(pattern), state, LevelFaultSweepConfig())
        changed = copy.deepcopy(pattern)
        changed["seed"] = 3
        with self.assertRaises(AssertionError):
            audit_pattern(encoding, other, pattern, changed, state, LevelFaultSweepConfig())
        with self.assertRaises(AssertionError):
            audit_pattern(encoding, other, pattern, pattern, decode_state(encoding), LevelFaultSweepConfig())

    def test_invalid_configuration_rejected(self):
        LevelFaultSweepConfig().validate()
        base = LevelFaultSweepConfig()
        for change in ({"sigmas": [0.02, 0.013]}, {"sigmas": []}, {"seeds": [1, 1]}, {"conditions": base.conditions + [base.conditions[0]]},
                       {"conditions": [{"condition_id": "x", "mapping": "gray", "level0_safe": False}]}, {"calibration_batch_size": 300}, {"run_id": "../x"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LevelFaultSweepConfig(**{**base.__dict__, **change}).validate()


class WeightErrorDiagnosticTests(unittest.TestCase):
    def test_frequent_mapping_makes_the_frequent_clusters_neighbors_two_clusters_apart(self):
        from fault_lora.evaluation.level_fault_errors import pattern_errors
        encoding = synthetic_encoding()
        frequent = remap_encoding(encoding, "frequent-at-0")
        probabilities = gaussian_level_probabilities(.04, level0_safe=True)
        sorted_errors = pattern_errors(encoding, generate_level_fault_pattern(encoding, probabilities, seed=4))
        self.assertEqual(set(sorted_errors["cluster_distance_histogram"]), {1})
        self.assertAlmostEqual(sorted_errors["sum_abs_change_gap_units"], sorted_errors["faulted_cells"], delta=1e-3)  # equal gaps of 0.2 (float32)
        pattern = generate_level_fault_pattern(frequent, probabilities, seed=4)
        errors = pattern_errors(frequent, pattern)
        entry = pattern["tensors"]["layer.weight"]
        # Index 8 sits on level 9 and index 10 on level 10: a fault between them skips cluster 9.
        between = int((((entry["before_levels"] == 9) & (entry["after_levels"] == 10)) | ((entry["before_levels"] == 10) & (entry["after_levels"] == 9))).sum())
        level1_down = int(((entry["before_levels"] == 1) & (entry["after_levels"] == 0)).sum())
        self.assertEqual(errors["cluster_distance_histogram"].get(2, 0), between)
        self.assertEqual(errors["cluster_distance_histogram"].get(9, 0), level1_down)
        self.assertGreater(between, 0)


if __name__ == "__main__":
    unittest.main()
