"""Offline checks for Step 11b: pattern identity across encodings, extended plan, damage, and runs."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.evaluation.fault_sweep import SweepConfig, sweep_plan
from fault_lora.evaluation.retrained_faults import (RetrainedAuditConfig, RetrainedSweepConfig, aggregate, extended_plan, rebind, run_retrained_fault_checks,
                                                    run_retrained_fault_sweep, side_by_side, transition_differences)
from fault_lora.memory.clustering import cluster_model, decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern
from test_lora_matrix import TinyCIFAR


def tiny_template():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))


def retrained_copy(encoding, scale, shift):
    """Same cells, different (still strictly increasing) codebooks: a stand-in for Step 11a."""
    other = copy.deepcopy(encoding)
    for record in other["clustered_tensors"].values():
        record["codebook"] = record["codebook"] * scale + shift
    return other


class PatternIdentityTests(unittest.TestCase):
    def setUp(self):
        self.encoding, _ = cluster_model(tiny_template(), 4, threads=1)
        self.retrained = retrained_copy(self.encoding, 1.1, .01)

    def test_same_seed_gives_identical_cells_on_an_encoding_with_the_same_indices(self):
        self.assertNotEqual(encoding_fingerprint(self.encoding), encoding_fingerprint(self.retrained))
        for probability, seed in ((0, 42), (.1, 52), (.5, 86), (1, 7)):
            first = generate_fault_pattern(self.encoding, probability, seed=seed)
            second = rebind(first, self.retrained)
            self.assertEqual(transition_differences(first, second), [])
            self.assertNotEqual(first["source_fingerprint"], second["source_fingerprint"])
            # Identical cells give each encoding's own codebook values at the same positions.
            self.assertEqual({name for name, value in apply_fault_pattern(self.retrained, second).items() if not torch.equal(value, decode_state(self.retrained)[name])},
                             {name for name, value in apply_fault_pattern(self.encoding, first).items() if not torch.equal(value, decode_state(self.encoding)[name])})

    def test_different_indices_are_reported_and_rejected(self):
        changed = copy.deepcopy(self.retrained)
        indices = changed["clustered_tensors"]["0.weight"]["indices"].view(-1)
        indices[:] = (indices.long() + 1).clamp(max=3).to(torch.uint8)
        first = generate_fault_pattern(self.encoding, .5, seed=3)
        differences = transition_differences(first, generate_fault_pattern(changed, .5, seed=3))
        self.assertEqual({(item["tensor"], item["field"]) for item in differences} & {("0.weight", "before_levels")}, {("0.weight", "before_levels")})
        self.assertFalse(any(item["tensor"] == "5.weight" for item in differences))
        with self.assertRaisesRegex(AssertionError, "differ across encodings"):
            rebind(first, changed)
        self.assertTrue(transition_differences(first, generate_fault_pattern(self.encoding, .5, seed=4)))

    def test_extended_plan_keeps_step7_and_adds_consecutive_new_seeds(self):
        step7 = sweep_plan(SweepConfig(probabilities=[0, .001, .005, .01, .02, .05, .1], patterns_per_rate=5, seed=42))
        plan = extended_plan(step7, [.2, .3, .5], 72, 5)
        self.assertEqual(plan[:31], step7)
        self.assertEqual([entry["seed"] for entry in plan[31:]], list(range(72, 87)))
        self.assertEqual([plan[i]["pattern_id"] for i in (31, 36, 41, 45)], ["rate07-seed72", "rate08-seed77", "rate09-seed82", "rate09-seed86"])
        for probabilities, seed in (([.2, .3], 70), ([.05, .2], 72), ([.3, .2], 72)):
            with self.subTest(probabilities=probabilities, seed=seed), self.assertRaises(AssertionError):
                extended_plan(step7, probabilities, seed, 5)

    def test_damage_is_each_encodings_own_zero_minus_rate(self):
        def record(probability, seed, accuracies):
            return {"probability": probability, "seed": seed, "counts": {"realized_fault_rate": probability},
                    "models": {name: {split: {"accuracy": value, "cross_entropy_loss": 1 - value} for split in ("validation", "test")} for name, value in accuracies.items()}}
        records = [record(0, 1, {"step4": .83, "retrained": .91}), record(.1, 2, {"step4": .60, "retrained": .89}), record(.1, 3, {"step4": .70, "retrained": .90})]
        result = aggregate(records, {"step4": [0, .1], "retrained": [0, .1]})
        self.assertAlmostEqual(result["step4"][1]["splits"]["test"]["fault_damage_percentage_points"]["mean"], 18.0)
        self.assertAlmostEqual(result["retrained"][1]["splits"]["test"]["fault_damage_percentage_points"]["mean"], 1.5)
        self.assertIsNone(result["retrained"][0]["splits"]["test"]["fault_damage_percentage_points"]["sample_std"])
        row = side_by_side(result)[1]
        self.assertAlmostEqual(row["test_fault_damage_percentage_points"]["step4_minus_retrained_mean"], 16.5)
        self.assertAlmostEqual(row["test_accuracy"]["retrained"]["mean"], .895)


class OfflineRunTests(unittest.TestCase):
    def build(self, root, *, break_indices=False):
        template = tiny_template()
        step4, _ = cluster_model(template, 4, threads=1)
        retrained, ablation = retrained_copy(step4, 1.1, .01), retrained_copy(step4, 1.05, -.02)
        if break_indices:
            retrained["clustered_tensors"]["5.weight"]["indices"].view(-1)[0] ^= 1
        dataset = TinyCIFAR("unused", True, None, False)
        split = {"train": list(range(20)), "validation": list(range(20, 30)), "test": {"indices": list(range(20))}}

        def measure(state):
            model = copy.deepcopy(template)
            model.load_state_dict(state)
            return {name: evaluate_classifier(model, DataLoader(Subset(dataset, indices), batch_size=10), "cpu") for name, indices in (("validation", split["validation"]), ("test", split["test"]["indices"]))}

        for name in ("scripts/check_retrained_faults.py", "scripts/run_retrained_fault_sweep.py", "requirements.txt", "requirements-macos-arm64.lock.txt"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("synthetic test input\n")
        (root / "src/fault_lora").mkdir(parents=True)
        (root / "step7").mkdir()
        plan = sweep_plan(SweepConfig(probabilities=[0, .1, .3], patterns_per_rate=2, seed=42))
        records, paths = [], {"pattern": root / "step5.pt"}
        torch.save(generate_fault_pattern(step4, .3, seed=42), paths["pattern"])
        for entry in plan:
            path = root / "step7" / f"{entry['pattern_id']}.pt"
            pattern = generate_fault_pattern(step4, entry["probability"], seed=entry["seed"])
            torch.save(pattern, path)
            records.append({**entry, "pattern": str(path.relative_to(root)), "pattern_sha256": sha256_file(path), "measurements": measure(apply_fault_pattern(step4, pattern))})
        sources = {"step4": step4, "retrained": retrained, "ablation": ablation, "clean_state": template.state_dict(), "split": split,
                   "metadata": {"num_classes": 10, "classes": TinyCIFAR.classes, "preprocessing": {"input_size": [3, 32, 32]}, "split_sha256": "synthetic"},
                   "clean_metrics": measure(template.state_dict()), "step7_plan": plan, "step7_records": records,
                   "step11a_metrics": {name: measure(decode_state(sources_encoding)) for name, sources_encoding in (("step4", step4), ("retrained", retrained), ("ablation", ablation))},
                   "paths": paths, "hashes": {name: sha256_file(path) for name, path in paths.items()}}
        return template, sources

    def patched(self, template, sources):
        return [patch("fault_lora.evaluation.retrained_faults.load_baselines", return_value=sources), patch("fault_lora.evaluation.retrained_faults.CIFAR10", TinyCIFAR),
                patch("fault_lora.evaluation.retrained_faults.build_resnet18", side_effect=lambda *args, **kwargs: copy.deepcopy(template))]

    def configs(self, sources):
        fingerprint = encoding_fingerprint(sources["retrained"])
        audit = RetrainedAuditConfig(run_id="tiny-audit", audit_probability=.3, audit_seed=42, baseline_fingerprint=fingerprint)
        sweep = RetrainedSweepConfig(run_id="tiny-sweep", audit_results="results/tiny-audit", baseline_fingerprint=fingerprint, extra_probabilities=[.5, .8], extra_first_seed=46,
                                     patterns_per_rate=2, ablation_probabilities=[0, .3, .8], evaluation_batch_size=10, device="cpu", num_workers=0)
        return audit, sweep

    def run_both(self, root, template, sources):
        audit, sweep = self.configs(sources)
        patches = self.patched(template, sources)
        for item in patches:
            item.start()
        try:
            report = run_retrained_fault_checks(audit, root)
            return report, run_retrained_fault_sweep(sweep, root)
        finally:
            for item in patches:
                item.stop()

    def test_audit_and_sweep_apply_identical_faults_and_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, sources = self.build(root)
            report, result = self.run_both(root, template, sources)
            self.assertEqual((report["status"], len(report["checks"])), ("passed", 8))
            self.assertTrue(report["checks"]["saved_pattern_and_checkpoint"]["identical_cells_to_step5_pattern"])
            self.assertEqual(result["status"], "complete")
            results = root / "results/tiny-sweep"
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            identity = json.loads((results / "pattern_identity.json").read_text())
            self.assertEqual(len(identity), 9)
            self.assertTrue(all(item["identical"] for item in identity))
            self.assertEqual(sum(item["compared_with"] == "saved Step 7 pattern" for item in identity), 5)
            records = json.loads((results / "measurements.json").read_text())
            self.assertEqual([sorted(record["models"]) for record in records if record["probability"] in (0, .3, .8)][0], ["ablation", "retrained", "step4"])
            self.assertEqual([record["models"]["step4"] for record in records[:5]], [record["measurements"] for record in sources["step7_records"]])
            aggregate_file = json.loads((results / "aggregate.json").read_text())
            self.assertEqual([entry["probability"] for entry in aggregate_file["models"]["ablation"]], [0, .3, .8])
            self.assertEqual(len(aggregate_file["side_by_side"]), 5)
            for extension in ("png", "pdf"):
                self.assertGreater((results / f"retrained_fault_sweep.{extension}").stat().st_size, 1000)
            self.assertTrue(all(sha256_file(path) == sources["hashes"][name] for name, path in sources["paths"].items()))

    def test_sweep_stops_before_evaluation_when_cells_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, sources = self.build(root, break_indices=True)
            _, result = self.run_both(root, template, sources)
            self.assertEqual(result["status"], "stopped_pattern_mismatch")
            self.assertTrue(result["mismatched_patterns"])
            results = root / "results/tiny-sweep"
            self.assertEqual(json.loads((results / "status.json").read_text())["status"], "stopped_pattern_mismatch")
            self.assertFalse((results / "measurements.json").exists())
            identity = json.loads((results / "pattern_identity.json").read_text())
            self.assertTrue(any(difference["tensor"] == "5.weight" for item in identity for difference in item["differences"]))

    def test_invalid_configuration_rejected(self):
        RetrainedSweepConfig().validate()
        for kwargs in ({"extra_probabilities": [.3, .2]}, {"ablation_probabilities": [.1, .2]}, {"patterns_per_rate": 1}, {"baseline_fingerprint": "18e55d67"}, {"run_id": "../x"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RetrainedSweepConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
