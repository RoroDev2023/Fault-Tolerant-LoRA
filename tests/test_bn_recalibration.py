"""Offline checks for label-free BatchNorm recalibration as a no-training baseline."""

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

from fault_lora.evaluation.bn_recalibration import (BNRecalibrationConfig, InputsOnly, aggregate_recalibration, calibration_indices, recalibrate_batchnorm,
                                                    run_bn_recalibration, verify_recalibration)
from fault_lora.evaluation.clustered_baseline import sha256_file
from test_lora_matrix import TinyCIFAR, TinyProject


CONDITIONS = [{"base_kind": "clustered", "pattern_id": None}, {"base_kind": "faulty", "pattern_id": "rate01-seed52"}, {"base_kind": "faulty", "pattern_id": "rate01-seed53"}]


class RecalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_statistics_are_the_cumulative_average_of_batch_statistics(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, dtype=torch.float64), nn.BatchNorm2d(4, dtype=torch.float64), nn.ReLU())
        with torch.no_grad():
            model[1].weight.normal_()
            model[1].bias.normal_()
            model[1].running_mean.fill_(50)   # Stale statistics must not influence the result.
            model[1].running_var.fill_(9)
            model[1].num_batches_tracked.fill_(7)
        original = copy.deepcopy(model.state_dict())
        batches = [torch.randn(6, 3, 5, 5, dtype=torch.float64) * scale + scale for scale in (1, 2, 3)]
        with torch.no_grad():
            features = [model[0](batch) for batch in batches]
        mean = torch.stack([value.mean((0, 2, 3)) for value in features]).mean(0)
        variance = torch.stack([value.transpose(0, 1).flatten(1).var(1, unbiased=True) for value in features]).mean(0)
        information = recalibrate_batchnorm(model, batches, torch.device("cpu"))
        self.assertEqual(information, {"calibration_examples": 18, "calibration_batches": 3, "batchnorm_layers": 1})
        torch.testing.assert_close(model[1].running_mean, mean, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(model[1].running_var, variance, atol=1e-12, rtol=1e-12)
        self.assertEqual(model[1].num_batches_tracked.item(), 3)
        self.assertEqual(model[1].momentum, .1)
        self.assertFalse(model.training or model[1].training)
        changed = {name for name, value in model.state_dict().items() if not torch.equal(value, original[name])}
        self.assertEqual(changed, {"1.running_mean", "1.running_var", "1.num_batches_tracked"})
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        # Evaluation after recalibration uses the stored statistics, not the batch.
        with torch.no_grad():
            torch.testing.assert_close(model[:2](batches[0]), nn.functional.batch_norm(features[0], mean, variance, model[1].weight, model[1].bias, False, 0., model[1].eps))

    def test_labels_missing_layers_and_empty_data_rejected_and_modes_restored(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        for batches in ([(torch.randn(4, 3), torch.zeros(4))], []):
            with self.assertRaises(AssertionError):
                recalibrate_batchnorm(model, batches, torch.device("cpu"))
            self.assertEqual(model[1].momentum, .1)
            self.assertFalse(model.training)
        with self.assertRaises(AssertionError):
            recalibrate_batchnorm(nn.Sequential(nn.Linear(3, 4)), [torch.randn(4, 3)], torch.device("cpu"))
        with self.assertRaises(AssertionError):
            recalibrate_batchnorm(nn.Sequential(nn.BatchNorm1d(3, track_running_stats=False)), [torch.randn(4, 3)], torch.device("cpu"))

    def test_calibration_subset_is_seeded_distinct_training_images_without_labels(self):
        train = list(range(100, 400, 3))
        first = calibration_indices(train, 20, 42)
        self.assertEqual(first, calibration_indices(train, 20, 42))
        self.assertNotEqual(first, calibration_indices(train, 20, 43))
        self.assertEqual(len(set(first)), 20)
        self.assertTrue(set(first) <= set(train))
        self.assertNotEqual(first, sorted(first))
        for count in (0, 101):
            with self.assertRaises(AssertionError):
                calibration_indices(train, count, 42)
        dataset = InputsOnly(TinyCIFAR("unused", True, None, False))
        self.assertIsInstance(dataset[3], torch.Tensor)
        self.assertEqual(len(dataset), 50000)

    def test_invalid_configuration_rejected(self):
        BNRecalibrationConfig(conditions=CONDITIONS).validate()
        for kwargs in ({"conditions": []}, {"conditions": [CONDITIONS[0], CONDITIONS[0]]}, {"conditions": [{"base_kind": "faulty", "pattern_id": None}]},
                       {"conditions": [{"base_kind": "clustered"}]}, {"conditions": [{"base_kind": "clustered", "pattern_id": "rate01-seed52"}]},
                       {"conditions": CONDITIONS, "calibration_examples": 5000, "calibration_batch_size": 128}, {"conditions": CONDITIONS, "calibration_batch_size": 1},
                       {"conditions": CONDITIONS, "run_id": "../x"}, {"conditions": CONDITIONS, "seed": -1}, {"conditions": CONDITIONS, "device": "gpu"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BNRecalibrationConfig(**kwargs).validate()

    def test_offline_run_changes_only_statistics_and_is_verified_and_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = TinyProject(root)
            (root / "scripts/run_bn_recalibration.py").write_text("synthetic test input\n")
            options = {"conditions": CONDITIONS, "calibration_examples": 20, "calibration_batch_size": 10, "evaluation_batch_size": 10, "device": "cpu", "num_workers": 0,
                       "source_sweep_results": "results/tiny-sweep"}
            config = BNRecalibrationConfig(run_id="tiny-bn", **options)
            with patch("fault_lora.training.recovery.prepare_sources", side_effect=project.prepare), patch("fault_lora.evaluation.bn_recalibration.CIFAR10", TinyCIFAR), \
                    patch("fault_lora.evaluation.bn_recalibration.build_resnet18", side_effect=lambda *args, **kwargs: copy.deepcopy(project.template)):
                records = run_bn_recalibration(config, root)
                with self.assertRaises(FileExistsError):
                    run_bn_recalibration(config, root)
                repeated = run_bn_recalibration(BNRecalibrationConfig(run_id="tiny-bn-again", **options), root)
            results = root / "results/tiny-bn"
            self.assertEqual(json.loads((results / "status.json").read_text())["status"], "complete")
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            self.assertEqual([record["condition_id"] for record in records], ["clustered", "rate01-seed52", "rate01-seed53"])
            indices = json.loads((results / "calibration_indices.json").read_text())
            self.assertEqual(sorted(indices), project.split["train"])
            self.assertFalse(set(indices) & set(project.split["validation"]))
            for record, again, condition in zip(records, repeated, CONDITIONS):
                base = project.state(condition["base_kind"], condition["pattern_id"])
                self.assertEqual(record["before"], {name: project.measure(base, name) for name in ("validation", "test")})
                saved = torch.load(root / record["batchnorm_statistics"], weights_only=True)["statistics"]
                self.assertEqual(set(saved), {"1.running_mean", "1.running_var", "1.num_batches_tracked"})
                self.assertFalse(torch.equal(saved["1.running_mean"], base["1.running_mean"]))
                self.assertEqual(saved["1.num_batches_tracked"].item(), 2)
                self.assertEqual(record["after"], {name: project.measure({**base, **saved}, name) for name in ("validation", "test")})
                self.assertEqual((record["after"], record["recalibration"]), (again["after"], again["recalibration"]))
                other = torch.load(root / again["batchnorm_statistics"], weights_only=True)["statistics"]
                self.assertTrue(all(torch.equal(saved[name], other[name]) for name in saved))
            self.assertEqual((records[0]["fault_probability"], records[0]["actual_corrupted_cells"], records[1]["fault_probability"]), (0.0, 0, .2))
            aggregate = aggregate_recalibration(records)
            self.assertEqual([(group["fault_probability"], group["num_conditions"]) for group in aggregate["groups"]], [(0.0, 1), (.2, 2)])
            self.assertIsNone(aggregate["groups"][0]["splits"]["test"]["accuracy_after"]["sample_std"])
            self.assertTrue(all(sha256_file(path) == project.hashes[name] for name, path in project.paths.items()))
            tampered = json.loads((results / "measurements.json").read_text())
            tampered[1]["after"]["test"]["accuracy"] += .1
            (results / "measurements.json").write_text(json.dumps(tampered))
            with self.assertRaises(AssertionError):
                verify_recalibration(root, config)


if __name__ == "__main__":
    unittest.main()
