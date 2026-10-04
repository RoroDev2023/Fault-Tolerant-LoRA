"""Offline checks for clustered storage, decoding, and clean-state preservation."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.memory.clustering import cluster_model, cluster_tensor, decode_layer, decode_state, eligible_weight_names, physical_levels, storage_estimates
from fault_lora.evaluation.clustered_baseline import ClusteredConfig, validate_source
from fault_lora.models.resnet import build_resnet18


class ClusteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sorted_centers_and_remapped_indices_match_nearest_representatives(self):
        weights = torch.linspace(-3, 4, 101).reshape(1, 101)
        record, stats = cluster_tensor(weights, 4, seed=42, threads=1)
        self.assertEqual(record["indices"].dtype, torch.uint8)
        self.assertTrue(torch.all(record["codebook"][1:] >= record["codebook"][:-1]))
        distances = (weights.reshape(-1, 1) - record["codebook"]).abs()
        torch.testing.assert_close(record["indices"].reshape(-1).long(), distances.argmin(1))
        self.assertEqual(sum(stats["cluster_occupancy"]), 101)
        expected_mse = (decode_layer(record).double() - weights.double()).square().mean().item()
        self.assertAlmostEqual(stats["mean_squared_weight_error"], expected_mse)

    def test_same_seed_reproduces_encoding_and_input_is_unchanged(self):
        weights = torch.randn(50, generator=torch.Generator().manual_seed(10))
        original = weights.clone()
        first, _ = cluster_tensor(weights, 4, seed=7, threads=1)
        second, _ = cluster_tensor(weights, 4, seed=7, threads=1)
        torch.testing.assert_close(weights, original, rtol=0, atol=0)
        for name in ("codebook", "indices", "index_to_level", "level_to_index"):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_small_distinct_sets_and_constant_tensors_are_exact(self):
        for weights in (torch.ones(3, 4), torch.tensor([-2., 0., 5., -2.], dtype=torch.float64)):
            with self.subTest(weights=weights):
                record, stats = cluster_tensor(weights, 16)
                torch.testing.assert_close(decode_layer(record), weights, rtol=0, atol=0)
                self.assertEqual(decode_layer(record).dtype, weights.dtype)
                self.assertEqual(stats["mean_squared_weight_error"], 0)
                self.assertEqual(stats["actual_clusters"], weights.unique().numel())

    def test_nonidentity_physical_mapping_round_trip(self):
        record, _ = cluster_tensor(torch.tensor([-1., 0., 1., 2.]), 4)
        record["index_to_level"] = torch.tensor([2, 0, 3, 1])
        record["level_to_index"] = torch.argsort(record["index_to_level"])
        levels = physical_levels(record)
        torch.testing.assert_close(levels, torch.tensor([2, 0, 3, 1], dtype=torch.uint8))
        torch.testing.assert_close(decode_layer(record, levels=levels), decode_layer(record), rtol=0, atol=0)

    def test_invalid_indices_levels_and_mappings_fail(self):
        record, _ = cluster_tensor(torch.tensor([-1., 0., 1., 2.]), 4)
        bad = copy.deepcopy(record)
        bad["indices"][0] = 4
        with self.assertRaises(ValueError):
            decode_layer(bad)
        with self.assertRaises(ValueError):
            decode_layer(record, levels=torch.tensor([-1, 0, 1, 2]))
        with self.assertRaises(ValueError):
            decode_layer(record, levels=torch.tensor([0., 1., 2., 3.]))
        bad = copy.deepcopy(record)
        bad["level_to_index"] = torch.tensor([0, 0, 2, 3])
        with self.assertRaises(ValueError):
            physical_levels(bad)
        bad = copy.deepcopy(record)
        bad["shape"] = [2, 2]
        with self.assertRaises(ValueError):
            decode_layer(bad)

    def test_only_conv_linear_weights_change_and_model_is_preserved(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.Flatten(), nn.Linear(8, 2))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        self.assertEqual(eligible_weight_names(model), ["0.weight", "3.weight"])
        encoding, _ = cluster_model(model, 4, threads=1)
        restored_state = decode_state(encoding)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            if name not in encoding["clustered_tensors"]:
                torch.testing.assert_close(restored_state[name], before[name], rtol=0, atol=0)
        restored = copy.deepcopy(model).eval()
        restored.load_state_dict(restored_state, strict=True)
        with torch.inference_mode():
            self.assertEqual(tuple(restored(torch.zeros(2, 1, 4, 4)).shape), (2, 2))

    def test_saved_representation_round_trip_does_not_need_original(self):
        model = nn.Linear(4, 2)
        encoding, _ = cluster_model(model, 3, threads=1)
        expected = decode_state(encoding)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoding.pt"
            torch.save(encoding, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            restored = decode_state(loaded)
        for name in expected:
            torch.testing.assert_close(restored[name], expected[name], rtol=0, atol=0)

    def test_storage_counts_include_codebooks_mappings_and_untouched_state(self):
        record, _ = cluster_tensor(torch.arange(16, dtype=torch.float32), 16)
        encoding = {"format_version": 1, "clustered_tensors": {"weight": record}, "untouched_state": {"bias": torch.ones(2)}}
        storage = storage_estimates(encoding)
        self.assertEqual(storage["original_dense_state_tensor_bytes"], 72)
        self.assertEqual(storage["stored_index_tensor_bytes"], 16)
        self.assertEqual(storage["ideal_bitpacked_index_bytes"], 8)
        self.assertEqual(storage["codebook_tensor_bytes"], 64)
        self.assertEqual(storage["mapping_tensor_bytes"], 256)
        self.assertEqual(storage["untouched_state_tensor_bytes"], 8)
        self.assertEqual(storage["stored_representation_tensor_bytes"], 344)

    def test_invalid_clustering_inputs_and_configuration_fail(self):
        for weights, k in ((torch.tensor([float("nan")]), 2), (torch.empty(0), 2), (torch.ones(2, dtype=torch.int64), 2), (torch.ones(2), 0), (torch.ones(2), 257)):
            with self.subTest(weights=weights, k=k), self.assertRaises(ValueError):
                cluster_tensor(weights, k)
        for invalid in ({"num_clusters": 257}, {"seed": -1}, {"tol": float("nan")}, {"run_id": "../other"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ClusteredConfig(**invalid).validate()

    def test_tampered_source_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            source = root / "clean.pt"
            source.write_bytes(b"changed checkpoint")
            (results / "status.json").write_text(json.dumps({"status": "complete"}))
            (results / "metrics.json").write_text(json.dumps({"checkpoint_sha256": hashlib.sha256(b"original checkpoint").hexdigest()}))
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_source(root, ClusteredConfig(source_checkpoint="clean.pt", source_results="results"))


if __name__ == "__main__":
    unittest.main()
