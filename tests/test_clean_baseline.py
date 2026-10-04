"""Offline checks for baseline data separation, metrics, and checkpoint integrity."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.data.cifar10 import cifar10_transforms, stratified_split
from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.models.resnet import build_resnet18, load_clean_checkpoint
from fault_lora.training.baseline import BaselineConfig, save_checkpoint


class CleanBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stratified_split_is_disjoint_complete_and_reproducible(self):
        targets = np.repeat(np.arange(10), 20)
        split = stratified_split(targets, 3, 42)
        self.assertEqual(split, stratified_split(targets, 3, 42))
        self.assertNotEqual(split, stratified_split(targets, 3, 43))
        train, validation = set(split["train"]), set(split["validation"])
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(range(len(targets))))
        np.testing.assert_array_equal(np.bincount(targets[split["validation"]]), np.full(10, 3))
        with self.assertRaises(ValueError):
            stratified_split(targets, 20, 42)

    def test_evaluation_transform_is_fixed_and_normalized(self):
        _, transform = cifar10_transforms(96)
        image = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
        output = transform(image)
        self.assertEqual(tuple(output.shape), (3, 96, 96))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, transform(image))
        expected = torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225])
        torch.testing.assert_close(output[:, 0, 0], expected)

    def test_metrics_weight_short_batches_by_example_count(self):
        logits = torch.tensor([[4, 0], [0, 4], [0, 4]], dtype=torch.float32)
        targets = torch.tensor([0, 1, 0])
        model = nn.Identity()
        model.train()
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        metrics = evaluate_classifier(model, loader, "cpu")
        self.assertEqual(metrics["num_examples"], 3)
        self.assertEqual(metrics["num_correct"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        expected_loss = nn.functional.cross_entropy(logits, targets).item()
        self.assertAlmostEqual(metrics["cross_entropy_loss"], expected_loss, places=6)
        self.assertTrue(model.training)

    def test_evaluation_preserves_batch_norm_and_mixed_training_modes(self):
        model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
        model.train()
        model[1].eval()
        before = copy.deepcopy(model.state_dict())
        loader = DataLoader(TensorDataset(torch.randn(3, 2), torch.tensor([0, 1, 0])), batch_size=2)
        evaluate_classifier(model, loader, "cpu")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(model.training)
        self.assertTrue(model[0].training)
        self.assertFalse(model[1].training)

    def test_empty_evaluation_fails(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
        with self.assertRaises(ValueError):
            evaluate_classifier(nn.Identity(), loader, "cpu")

    def test_checkpoint_round_trip_is_exact_and_does_not_download(self):
        model = build_resnet18(10, pretrained=False).eval()
        inputs = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            expected = model(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, {"num_classes": 10, "epoch": 1, "preprocessing": {"input_size": [3, 32, 32]}})
            restored, metadata = load_clean_checkpoint(path)
            self.assertEqual(metadata["epoch"], 1)
            self.assertFalse(restored.training)
            self.assertEqual(restored.fc.out_features, 10)
            with torch.inference_mode():
                torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)

    def test_invalid_run_configuration_fails(self):
        for invalid in ({"epochs": 0}, {"run_id": "../other"}, {"validation_per_class": 5000}, {"device": "bogus"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BaselineConfig(**invalid).validate()


if __name__ == "__main__":
    unittest.main()
