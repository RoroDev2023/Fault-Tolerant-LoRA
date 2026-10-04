"""Offline checks for low-rank geometry, frozen state and validation-only recovery."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.memory.clustering import cluster_model, decode_state
from fault_lora.memory.faults import apply_fault_pattern, generate_fault_pattern, summarize_fault_pattern
from fault_lora.models.lora import LoRALayer, adapter_state_dict, attach_lora, base_state_dict, load_adapter_state
from fault_lora.models.resnet import build_resnet18
from fault_lora.training.recovery import RecoveryConfig, assert_frozen_base, load_recovered_model, run_recovery, train_adapter_epoch


class TinyCIFAR(Dataset):
    classes = [str(i) for i in range(10)]
    test_loaded = False
    selected_checkpoint = None

    def __init__(self, root, train, transform, download):
        self.train = train
        if not train:
            if self.selected_checkpoint is not None:
                assert self.selected_checkpoint.exists(), "Test loaded before checkpoint selection"
                last = torch.load(self.selected_checkpoint.parent / "last.pt", weights_only=True)
                assert last["epoch"] == last["config"]["epochs"], "Test loaded before training finished"
            TinyCIFAR.test_loaded = True

    def __len__(self):
        return 50000 if self.train else 10000

    def __getitem__(self, index):
        return torch.full((3, 32, 32), (index % 10) / 10), index % 10


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_convolution_residual_equals_flattened_rank_update_with_geometry(self):
        for stride, dilation, mode in ((1, 1, "zeros"), (2, 1, "zeros"), (2, 2, "reflect")):
            with self.subTest(stride=stride, dilation=dilation, mode=mode):
                base = nn.Conv2d(3, 5, 3, stride=stride, padding=dilation, dilation=dilation, padding_mode=mode, dtype=torch.float64)
                layer = LoRALayer(base, 2, 6)
                nn.init.normal_(layer.lora_B.weight)
                delta = (layer.lora_B.weight[:, :, 0, 0] @ layer.lora_A.weight.flatten(1)).reshape_as(base.weight)
                inputs = torch.randn(2, 3, 15, 17, dtype=torch.float64)
                padding = dilation
                if mode == "reflect":
                    inputs_for_dense = F.pad(inputs, base._reversed_padding_repeated_twice, mode="reflect")
                    padding = 0
                else:
                    inputs_for_dense = inputs
                expected = F.conv2d(inputs_for_dense, base.weight + layer.scale * delta, base.bias, stride=stride, padding=padding, dilation=dilation)
                torch.testing.assert_close(layer(inputs), expected, atol=1e-10, rtol=1e-10)

    def test_linear_scaling_and_zero_output_identity(self):
        base = nn.Linear(5, 3, dtype=torch.float64)
        inputs = torch.randn(4, 5, dtype=torch.float64)
        original = base(inputs).detach()
        layer = LoRALayer(base, 2, 8)
        self.assertTrue(torch.equal(layer(inputs), original))
        nn.init.normal_(layer.lora_B.weight)
        expected = F.linear(inputs, base.weight + 4 * layer.lora_B.weight @ layer.lora_A.weight, base.bias)
        torch.testing.assert_close(layer(inputs), expected)
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))

    def test_training_changes_adapters_only_and_preserves_batchnorm_buffers(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
        original = copy.deepcopy(model.state_dict())
        attach_lora(model, rank=2, alpha=2)
        initial = adapter_state_dict(model)
        loader = DataLoader(TensorDataset(torch.randn(8, 3, 8, 8) + 2, torch.arange(8) % 2), batch_size=2)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.01)
        model.train()  # The training loop must explicitly undo this for BatchNorm.
        train_adapter_epoch(model, loader, optimizer, torch.device("cpu"), 1)
        assert_frozen_base(model, original)
        self.assertFalse(model[1].training)
        self.assertTrue(all(not torch.equal(initial[name], value) for name, value in adapter_state_dict(model).items()))
        self.assertTrue(all(torch.equal(original[name], value) for name, value in base_state_dict(model).items()))

    def test_full_resnet_zero_logits_and_all_21_layers(self):
        model = build_resnet18(pretrained=False).eval()
        inputs = torch.randn(2, 3, 32, 32)
        original = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            expected = model(inputs)
        layout = attach_lora(model, 4, 4)
        self.assertEqual(len(layout), 21)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected, model(inputs)))
        assert_frozen_base(model, original)

    def test_adapter_serialization_and_malformed_state_rejection(self):
        model = nn.Sequential(nn.Linear(4, 3))
        attach_lora(model, 2, 2)
        nn.init.normal_(model[0].lora_B.weight)
        inputs = torch.randn(3, 4)
        expected = model(inputs).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(adapter_state_dict(model), path)
            saved = torch.load(path, weights_only=True)
        nn.init.zeros_(model[0].lora_B.weight)
        load_adapter_state(model, saved)
        self.assertTrue(torch.equal(expected, model(inputs)))
        for state in ({}, {**saved, "unexpected": torch.ones(1)}, {**saved, "0.lora_A.weight": torch.ones(1)}, {**saved, "0.lora_A.weight": saved["0.lora_A.weight"].double()}, {**saved, "0.lora_A.weight": torch.full_like(saved["0.lora_A.weight"], float("nan"))}):
            with self.assertRaises(ValueError):
                load_adapter_state(model, state)

    def test_invalid_rank_groups_targets_and_configuration_preserve_base(self):
        for kwargs in ({"rank": 0}, {"alpha": float("nan")}, {"learning_rate": 0}, {"epochs": True}, {"target_names": []}, {"run_id": "../oops"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RecoveryConfig(**kwargs).validate()
        model = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 2))
        for rank, targets in ((4, None), (2, ["missing"]), (2, ["0", "0"])):
            with self.assertRaises(ValueError):
                attach_lora(model, rank, 2, targets)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            self.assertFalse(any(isinstance(module, LoRALayer) for module in model.modules()))
        with self.assertRaises(ValueError):
            LoRALayer(nn.Conv2d(4, 4, 3, groups=2), 2, 2)
        attach_lora(model, 2, 2, ["1"])
        self.assertIsInstance(model[0], nn.Linear)
        self.assertIsInstance(model[1], LoRALayer)
        self.assertTrue(all(not parameter.requires_grad for parameter in model[0].parameters()))
        with self.assertRaises(ValueError):
            attach_lora(model, 2, 2)

    def test_offline_recovery_selects_validation_reloads_and_preserves_sources(self):
        # All synthetic artifacts remain inside a temporary test directory.
        config = RecoveryConfig(run_id="tiny-recovery", rank=2, alpha=2, epochs=2, batch_size=10, evaluation_batch_size=10, device="cpu", num_workers=0)
        template = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))
        encoding, _ = cluster_model(template, 4, threads=1)
        pattern = generate_fault_pattern(encoding, .2, seed=52)
        faulty_state = apply_fault_pattern(encoding, pattern)
        faulty = copy.deepcopy(template)
        faulty.load_state_dict(faulty_state)
        dataset = TinyCIFAR("unused", True, None, False)
        subsets = {"validation": torch.utils.data.Subset(dataset, list(range(20, 30))), "test": torch.utils.data.Subset(dataset, list(range(20)))}
        measurements = {name: evaluate_classifier(faulty, DataLoader(subset, batch_size=10), "cpu") for name, subset in subsets.items()}
        references = {name: {"clean": value, "clustered_no_faults": value} for name, value in measurements.items()}
        metadata = {"num_classes": 10, "classes": TinyCIFAR.classes, "preprocessing": {"input_size": [3, 32, 32]}, "split_sha256": "synthetic-test-only"}
        split = {"train": list(range(20)), "validation": list(range(20, 30)), "test": {"indices": list(range(20))}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/fault_lora").mkdir(parents=True)
            (root / "scripts").mkdir()
            for filename in ("scripts/run_lora_recovery.py", "requirements.txt", "requirements-macos-arm64.lock.txt"):
                (root / filename).write_text("synthetic test input\n")
            paths = {"encoding": root / "encoding.pt", "selected_pattern": root / "pattern.pt"}
            torch.save(encoding, paths["encoding"])
            torch.save(pattern, paths["selected_pattern"])
            hashes = {name: sha256_file(path) for name, path in paths.items()}
            record = {"pattern_id": config.pattern_id, "seed": 52, "probability": .2, "counts": summarize_fault_pattern(encoding, pattern), "measurements": measurements}
            prepared = (encoding, metadata, split, record, references, paths, hashes)
            TinyCIFAR.test_loaded = False
            TinyCIFAR.selected_checkpoint = root / "checkpoints/tiny-recovery/best.pt"
            with patch("fault_lora.training.recovery.prepare_sources", return_value=prepared), patch("fault_lora.training.recovery.CIFAR10", TinyCIFAR), patch("fault_lora.training.recovery.build_resnet18", side_effect=lambda *args, **kwargs: copy.deepcopy(template)):
                result = run_recovery(config, root)
                restored, payload = load_recovered_model(root / result["best_checkpoint"], root)
                restored_metrics = evaluate_classifier(restored, DataLoader(subsets["test"], batch_size=10), "cpu")
                self.assertEqual(restored_metrics, result["after"]["test"])
                self.assertEqual(payload["epoch"], result["selected_epoch"])
                with self.assertRaises(FileExistsError):
                    run_recovery(config, root)
                modified = torch.load(root / result["best_checkpoint"], weights_only=True)
                modified["split_sha256"] = "wrong-split"
                bad_checkpoint = root / "bad.pt"
                torch.save(modified, bad_checkpoint)
                with self.assertRaises(AssertionError):
                    load_recovered_model(bad_checkpoint, root)
            TinyCIFAR.selected_checkpoint = None
            self.assertTrue(TinyCIFAR.test_loaded)
            self.assertEqual(result["before"], measurements)
            self.assertGreater(result["parameters"]["trainable_adapter"], 0)
            self.assertEqual(json.loads((root / "results/tiny-recovery/status.json").read_text())["status"], "complete")
            self.assertTrue(all(sha256_file(path) == hashes[name] for name, path in paths.items()))
            for name, value in base_state_dict(restored).items():
                self.assertTrue(torch.equal(value, faulty_state[name]))


if __name__ == "__main__":
    unittest.main()
