"""Offline checks for codebook-only fine-tuning with fixed indices (Step 11a)."""

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
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.memory.clustering import cluster_model, decode_state
from fault_lora.memory.faults import encoding_fingerprint
from fault_lora.training import codebook_finetune as finetune
from fault_lora.training.codebook_finetune import (CodebookFinetuneConfig, attach_codebooks, build_encoding, codebook_order, effective_state, set_trainable,
                                                   train_epoch, variant_payload, verify_same_cells)
from test_lora_matrix import TinyCIFAR


MAIN = {"variant_id": "codebook-bnstats", "label": "codebook+BNstats", "train_batchnorm_affine": False, "train_fc_bias": False}
ABLATION = {"variant_id": "codebook-bnaffine", "label": "codebook+BNaffine", "train_batchnorm_affine": True, "train_fc_bias": True}


def tiny_template():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))


def codebook_model(template, encoding):
    model = copy.deepcopy(template)
    model.load_state_dict(decode_state(encoding))
    return model, attach_codebooks(model, encoding)


class CodebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.template = tiny_template()
        self.encoding, _ = cluster_model(self.template, 4, threads=1)
        self.loader = DataLoader(TensorDataset(torch.randn(16, 3, 8, 8) + 1, torch.arange(16) % 10), batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))

    def test_indices_and_frozen_state_never_change_while_codebooks_and_bn_statistics_do(self):
        for variant in (MAIN, ABLATION):
            with self.subTest(variant=variant["variant_id"]):
                model, layers = codebook_model(self.template, self.encoding)
                trainable = set_trainable(model, layers, variant)
                snapshot = finetune.frozen_snapshot(model)
                before = effective_state(model, layers)
                optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.05)
                for _ in range(2):
                    train_epoch(model, self.loader, optimizer, torch.device("cpu"), 1.0)
                optimizer.zero_grad(set_to_none=True)
                finetune.assert_frozen(model, snapshot, trainable)
                self.assertFalse(model.training)
                for name, layer in layers.items():
                    self.assertTrue(torch.equal(layer.indices, self.encoding["clustered_tensors"][f"{name}.weight"]["indices"]))
                    self.assertEqual(layer.indices.dtype, torch.uint8)
                    self.assertFalse(torch.equal(layer.codebook.detach(), self.encoding["clustered_tensors"][f"{name}.weight"]["codebook"]))
                after = effective_state(model, layers)
                self.assertFalse(torch.equal(after["1.running_mean"], before["1.running_mean"]))
                self.assertEqual(torch.equal(after["1.weight"], before["1.weight"]), not variant["train_batchnorm_affine"])
                self.assertEqual(torch.equal(after["5.bias"], before["5.bias"]), not variant["train_fc_bias"])
                self.assertEqual(set(after), set(self.template.state_dict()))

    def test_gradients_reach_only_declared_parameters(self):
        expected = {"codebook-bnstats": {"0.codebook", "5.codebook"}, "codebook-bnaffine": {"0.codebook", "5.codebook", "1.weight", "1.bias", "5.base.bias"}}
        for variant in (MAIN, ABLATION):
            model, layers = codebook_model(self.template, self.encoding)
            self.assertEqual(set(set_trainable(model, layers, variant)), expected[variant["variant_id"]])
            model.train()
            F.cross_entropy(model(torch.randn(4, 3, 8, 8)), torch.arange(4)).backward()
            with_grad = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
            self.assertEqual(with_grad, expected[variant["variant_id"]])
            self.assertFalse(hasattr(layers["0"].base, "weight") and isinstance(getattr(layers["0"].base, "weight", None), nn.Parameter))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), 4 + 4 + 2 * 4 + 10)

    def test_codebook_gradient_is_the_sum_of_member_weight_gradients(self):
        template = tiny_template().double()
        encoding, _ = cluster_model(template, 4, threads=1)
        model, layers = codebook_model(template, encoding)
        set_trainable(model, layers, MAIN)
        model.eval()
        inputs, targets = torch.randn(5, 3, 8, 8, dtype=torch.float64), torch.arange(5)
        F.cross_entropy(model(inputs), targets).backward()
        dense = copy.deepcopy(template)
        dense.load_state_dict(decode_state(encoding))
        dense.eval()
        F.cross_entropy(dense(inputs), targets).backward()
        for name, layer in layers.items():
            member = getattr(dense, name).weight.grad.reshape(-1)
            summed = torch.zeros_like(layer.codebook).scatter_add_(0, layer.indices.reshape(-1).long(), member)
            torch.testing.assert_close(layer.codebook.grad, summed, rtol=1e-12, atol=1e-12)

    def test_saved_encoding_reconstructs_trained_weights_exactly_and_keeps_cells(self):
        model, layers = codebook_model(self.template, self.encoding)
        set_trainable(model, layers, ABLATION)
        train_epoch(model, self.loader, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01), torch.device("cpu"), 1.0)
        payload = variant_payload(layers, model, ABLATION, 1, {}, CodebookFinetuneConfig(), {})
        encoding = build_encoding(self.encoding, payload["codebooks"], payload["untouched_state"], {"faults_injected": 0})
        with tempfile.TemporaryDirectory() as directory:
            torch.save(encoding, Path(directory) / "encoding.pt")
            saved = torch.load(Path(directory) / "encoding.pt", weights_only=True)
        self.assertEqual(verify_same_cells(saved, self.encoding), 2)
        reconstructed, trained = decode_state(saved), effective_state(model, layers)
        self.assertEqual(set(reconstructed), set(trained))
        self.assertTrue(all(torch.equal(reconstructed[name], value) for name, value in trained.items()))
        self.assertNotEqual(encoding_fingerprint(saved), encoding_fingerprint(self.encoding))
        tampered = copy.deepcopy(saved)
        tampered["clustered_tensors"]["0.weight"]["indices"].view(-1)[0] ^= 1
        with self.assertRaises(AssertionError):
            verify_same_cells(tampered, self.encoding)

    def test_order_violation_is_detected_and_blocks_encoding(self):
        codebooks = {name: record["codebook"].clone() for name, record in self.encoding["clustered_tensors"].items()}
        self.assertTrue(all(item["strictly_increasing"] for item in codebook_order(codebooks).values()))
        codebooks["5.weight"][0] = codebooks["5.weight"][2] + 1   # 0 now exceeds entries 1, 2 (and maybe 3)
        codebooks["0.weight"][2] = codebooks["0.weight"][1]        # a tie also breaks strict order
        report = codebook_order(codebooks)
        self.assertFalse(report["5.weight"]["strictly_increasing"])
        self.assertEqual(report["5.weight"]["adjacent_violations"], 1)
        self.assertGreaterEqual(report["5.weight"]["inverted_pairs"], 2)
        self.assertEqual((report["0.weight"]["adjacent_violations"], report["0.weight"]["inverted_pairs"]), (1, 1))
        with self.assertRaisesRegex(AssertionError, "lost strict order"):
            build_encoding(self.encoding, codebooks, self.encoding["untouched_state"], {})

    def test_invalid_configuration_rejected(self):
        CodebookFinetuneConfig().validate()
        for kwargs in ({"variants": []}, {"variants": [MAIN, MAIN]}, {"variants": [{**MAIN, "train_fc_bias": 1}]}, {"variants": [{**MAIN, "extra": 1}]},
                       {"learning_rate": 0}, {"weight_decay": -1}, {"epochs": 0}, {"expected_clean_sha256": "7a9d"}, {"run_id": "../x"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CodebookFinetuneConfig(**kwargs).validate()


class OfflineRunTests(unittest.TestCase):
    def build(self, root):
        template = tiny_template()
        encoding, _ = cluster_model(template, 4, threads=1)
        dataset = TinyCIFAR("unused", True, None, False)
        split = {"train": list(range(20)), "validation": list(range(20, 30)), "test": {"indices": list(range(20))}}
        measure = lambda state: {name: self.evaluate(template, state, Subset(dataset, indices)) for name, indices in (("validation", split["validation"]), ("test", split["test"]["indices"]))}
        clustered = decode_state(encoding)
        statistics = {name: value + (0.1 if name.endswith("running_mean") else 0) for name, value in clustered.items() if name.startswith("1.") and name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
        for name in ("scripts/run_codebook_finetune.py", "requirements.txt", "requirements-macos-arm64.lock.txt"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("synthetic test input\n")
        (root / "src/fault_lora").mkdir(parents=True)
        paths = {"encoding": root / "encoding.pt", "clean": root / "clean.pt"}
        torch.save(encoding, paths["encoding"])
        torch.save({"model_state_dict": template.state_dict()}, paths["clean"])
        sources = {"encoding": encoding, "clean_state": template.state_dict(), "split": split,
                   "metadata": {"num_classes": 10, "classes": TinyCIFAR.classes, "preprocessing": {"input_size": [3, 32, 32]}, "split_sha256": "synthetic"},
                   "clean_metrics": measure(template.state_dict()), "clustered_metrics": measure(clustered), "bn_statistics": statistics, "bn_metrics": measure({**clustered, **statistics}),
                   "paths": paths, "hashes": {name: sha256_file(path) for name, path in paths.items()}}
        return template, sources

    @staticmethod
    def evaluate(template, state, subset):
        model = copy.deepcopy(template)
        model.load_state_dict(state)
        return evaluate_classifier(model, DataLoader(subset, batch_size=10), "cpu")

    def run_patched(self, root, template, sources, config, *extra):
        patches = [patch("fault_lora.training.codebook_finetune.prepare_codebook_sources", return_value=sources), patch("fault_lora.training.codebook_finetune.CIFAR10", TinyCIFAR),
                   patch("fault_lora.training.codebook_finetune.build_resnet18", side_effect=lambda *args, **kwargs: copy.deepcopy(template)), *extra]
        for item in patches:
            item.start()
        try:
            return finetune.run_codebook_finetune(config, root)
        finally:
            for item in patches:
                item.stop()

    def test_offline_run_saves_verified_encodings_and_paired_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, sources = self.build(root)
            config = CodebookFinetuneConfig(run_id="tiny-codebook", epochs=2, batch_size=5, evaluation_batch_size=10, learning_rate=.01, device="cpu", num_workers=0,
                                            expected_encoding_sha256=sources["hashes"]["encoding"])
            result = self.run_patched(root, template, sources, config)
            results = root / "results/tiny-codebook"
            self.assertEqual(json.loads((results / "status.json").read_text())["status"], "complete")
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            self.assertEqual(set(result["evaluation"]), {"clean_step3", "clustered_step4", "clustered_bn_recalibration_step9b", "codebook-bnstats", "codebook-bnaffine"})
            self.assertEqual(result["evaluation"]["clustered_step4"], sources["clustered_metrics"])
            for key in ("codebook-bnstats", "codebook-bnaffine"):
                saved = torch.load(root / result["encodings"][key]["encoding"], weights_only=True)
                verify_same_cells(saved, sources["encoding"])
                dense = torch.load(root / result["encodings"][key]["dense"], weights_only=True)["model_state_dict"]
                self.assertEqual(result["evaluation"][key]["test"], self.evaluate(template, dense, Subset(TinyCIFAR("unused", False, None, False), list(range(20)))))
            trainable = json.loads((results / "trainable.json").read_text())
            self.assertEqual((trainable["codebook-bnstats"]["trainable_parameters"], trainable["codebook-bnaffine"]["trainable_parameters"]), (8, 26))
            with self.assertRaises(FileExistsError):
                self.run_patched(root, template, sources, config)

    def test_order_violation_keeps_the_run_and_saves_no_encoding_or_test_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template, sources = self.build(root)
            real = finetune.train_variant

            def unsorted(*args, **kwargs):
                result = real(*args, **kwargs)
                result["selected"]["codebooks"]["5.weight"][0] = result["selected"]["codebooks"]["5.weight"][-1] + 1
                return result

            config = CodebookFinetuneConfig(run_id="tiny-unsorted", epochs=1, batch_size=5, evaluation_batch_size=10, device="cpu", num_workers=0)
            result = self.run_patched(root, template, sources, config, patch("fault_lora.training.codebook_finetune.train_variant", side_effect=unsorted))
            self.assertEqual(result["status"], "stopped_codebook_order_violation")
            self.assertEqual(result["violations"]["codebook-bnstats"]["5.weight"], {"adjacent_violations": 1, "inverted_pairs": 3})
            status = json.loads((root / "results/tiny-unsorted/status.json").read_text())
            self.assertEqual(status["status"], "stopped_codebook_order_violation")
            self.assertTrue((root / "checkpoints/tiny-unsorted/codebook-bnstats/best.pt").exists())
            self.assertFalse(list((root / "checkpoints/tiny-unsorted").rglob("encoding.pt")))
            self.assertFalse((root / "results/tiny-unsorted/paired_evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
