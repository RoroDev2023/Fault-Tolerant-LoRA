"""Offline checks for TinyBERT clustering, fault audit, and fault sweep (Step 10b).

The full pipeline runs on the tiny randomly initialized BERT and hand-built
local cache from `test_tinybert_baseline`; nothing is downloaded.
"""

import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_tinybert_baseline as fixture
from fault_lora.evaluation import tinybert_baseline
from fault_lora.evaluation.fault_checks import inspect_transitions
from fault_lora.evaluation.tinybert_baseline import run_tinybert_baseline
from fault_lora.evaluation.tinybert_clean_eval import CleanEvalConfig, run_clean_eval
from fault_lora.evaluation.tinybert_clustered import TinyBertClusteredConfig, run_tinybert_clustered
from fault_lora.evaluation.tinybert_fault_checks import REQUIRED_CHECKS, TinyBertFaultCheckConfig, run_tinybert_fault_checks
from fault_lora.evaluation.tinybert_fault_sweep import TinyBertSweepConfig, aggregate_tinybert, run_tinybert_fault_sweep
from fault_lora.evaluation.tinybert_sources import build_tinybert, filter_second_test, linear_weight_names, protected_state, synthetic_token_batch
from fault_lora.memory.clustering import cluster_model, decode_state, eligible_weight_names
from fault_lora.memory.faults import apply_fault_pattern, generate_fault_pattern

SCRIPTS = ("run_tinybert_clean_eval.py", "run_tinybert_clustered.py", "check_tinybert_faults.py", "run_tinybert_fault_sweep.py")


class TinyBertScopeTests(unittest.TestCase):
    def test_cluster_model_selects_exactly_the_linear_weights(self):
        model = fixture.tiny_model()
        eligible = eligible_weight_names(model)
        self.assertEqual(eligible, linear_weight_names(model))
        self.assertEqual(len(eligible), 8)
        self.assertFalse([name for name in eligible if "embeddings" in name or "LayerNorm" in name or name.endswith(".bias")])
        protected = protected_state(model, set(eligible))
        self.assertEqual((protected["embedding_tables"]["num_tensors"], protected["other"]["num_tensors"]), (3, 0))
        state = model.state_dict()
        self.assertEqual(sum(group["num_values"] for group in protected.values()) + sum(state[name].numel() for name in eligible),
                         sum(value.numel() for value in state.values()))

    def test_unchanged_clustering_and_faults_work_on_bert_and_protect_reliable_state(self):
        model = fixture.tiny_model()
        encoding, layers = cluster_model(model, 16, seed=42, threads=1)
        self.assertEqual(sorted(encoding["clustered_tensors"]), sorted(linear_weight_names(model)))
        self.assertTrue(all(layer["actual_clusters"] == 16 for layer in layers.values()))
        clustered = build_tinybert(json.loads(json.dumps(model.config.to_dict())), decode_state(encoding))
        inputs = synthetic_token_batch(clustered.config.to_dict(), length=8)
        pattern = generate_fault_pattern(encoding, 1, seed=1)
        state = apply_fault_pattern(encoding, pattern)
        inspect_transitions(encoding, pattern, state)
        for name, value in model.state_dict().items():
            if name not in encoding["clustered_tensors"]:
                self.assertTrue(torch.equal(state[name], value), name)
        zero = apply_fault_pattern(encoding, generate_fault_pattern(encoding, 0, seed=1))
        with torch.inference_mode():
            before = clustered(**inputs).logits
            clustered.load_state_dict(zero)
            self.assertTrue(torch.equal(clustered(**inputs).logits, before))

    def test_second_test_filter_removes_and_records_overlap(self):
        names = ["negative", "positive"]
        sst2 = {"train": {"sentences": ["what 's next ? ", "good plot "], "labels": [1, 1]}, "validation": {"sentences": ["funny film "], "labels": [1]}}
        candidate = {"test": {"sentences": ["what 's next ?", "a dull mess", "funny  film"], "labels": [0, 0, 1], "label_texts": ["negative", "negative", "positive"]}}
        result = filter_second_test(candidate, sst2, names)
        self.assertEqual((result["indices"], result["sentences"], result["labels"]), ([1], ["a dull mess"], [0]))
        self.assertEqual([item["setfit_test_index"] for item in result["removed"]], [0, 2])
        self.assertEqual(result["removed"][0]["matches"], [{"split": "train", "index": 0, "sentence": "what 's next ? ", "label": 1}])
        self.assertIn("validation", result["removed"][1]["reason"])
        hidden = {"test": {**candidate["test"], "labels": [-1, -1, -1]}}
        with self.assertRaises(AssertionError):
            filter_second_test(hidden, sst2, names)

    def test_aggregate_rejects_incomplete_or_off_plan_records(self):
        config = TinyBertSweepConfig(probabilities=[0, .5], patterns_per_rate=2)
        counts = {"eligible_cells": 10, "actual_corrupted_cells": 0, "realized_fault_rate": 0.0}
        metric = {"num_examples": 4, "num_correct": 2, "accuracy": .5, "cross_entropy_loss": .7}
        loss = {"accuracy_drop_percentage_points": 0.0, "cross_entropy_increase": 0.0}
        record = lambda pattern_id, probability, seed: {"pattern_id": pattern_id, "probability": probability, "seed": seed, "counts": counts,
                                                        "measurements": {"selection": metric, "test-a": metric}, "fault_loss": {"selection": loss, "test-a": loss}}
        records = [record("rate00-zero", 0.0, 42), record("rate01-seed42", .5, 42), record("rate01-seed43", .5, 43)]
        aggregate = aggregate_tinybert(records, config, ["selection", "test-a"])
        self.assertEqual([rate["num_patterns"] for rate in aggregate["rates"]], [1, 2])
        self.assertIsNone(aggregate["rates"][0]["splits"]["test-a"]["accuracy"]["sample_std"])
        for broken in (records[:2], records[:2] + [record("rate01-seed44", .5, 44)]):
            with self.subTest(broken=[item["pattern_id"] for item in broken]), self.assertRaises(AssertionError):
                aggregate_tinybert(broken, config, ["selection", "test-a"])
        with self.assertRaises(AssertionError):
            aggregate_tinybert(records, config, ["test-a", "selection"])

    def test_invalid_configurations_fail(self):
        for config in (TinyBertClusteredConfig(clean_checkpoint_sha256="abc"), TinyBertClusteredConfig(num_clusters=0), CleanEvalConfig(expected_original_test_examples=0),
                       TinyBertFaultCheckConfig(reference_probability=1), TinyBertSweepConfig(probabilities=[.1, .2]), TinyBertSweepConfig(device="bogus")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()


class TinyBertPipelineTests(unittest.TestCase):
    """Step 10a fixture → two-test clean evaluation → clustering → audit → sweep, all offline."""

    def test_complete_pipeline_is_verified_and_reports_test_sets_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            step10a = fixture.TinyBertBaselineRunTests.build(None, root, leak=True)
            for name in SCRIPTS:
                (root / "scripts" / name).write_text("placeholder\n")
            with mock.patch.object(tinybert_baseline, "PROBES", {}):
                run_tinybert_baseline(step10a, root)
            source = "results/tiny-run"
            clean = CleanEvalConfig(run_id="tiny-clean", source_results=source, expected_original_test_examples=25)
            clean_metrics = run_clean_eval(clean, root)
            test_names = ["sst2-validation-30", "sst2-original-test-25"]
            self.assertEqual(list(clean_metrics["measurements"]), ["selection", *test_names])
            test_sets = json.loads((root / "results/tiny-clean/test_sets.json").read_text())
            removed = test_sets["sst2-original-test-25"]["removed"]
            self.assertEqual(len(removed), 1)
            self.assertEqual((removed[0]["setfit_test_index"], removed[0]["matches"][0]["split"]), (25, "train"))

            checkpoint_hash = clean_metrics["checkpoint_sha256"]
            clustered = TinyBertClusteredConfig(run_id="tiny-clustered", clean_checkpoint_sha256=checkpoint_hash, source_clean_eval_results="results/tiny-clean", clustering_threads=1)
            with self.assertRaises(AssertionError):
                run_tinybert_clustered(dataclasses.replace(clustered, run_id="tiny-wrong-hash", clean_checkpoint_sha256="0" * 64), root)
            metrics = run_tinybert_clustered(clustered, root)
            self.assertEqual(list(metrics["measurements"]), ["selection", *test_names])
            for name in test_names:
                item = metrics["measurements"][name]
                self.assertEqual(item["clean"], clean_metrics["measurements"][name])
                self.assertAlmostEqual(item["clustering_accuracy_drop_percentage_points"], 100 * (item["clean"]["accuracy"] - item["clustered_no_faults"]["accuracy"]))
            scope = json.loads((root / "results/tiny-clustered/scope.json").read_text())
            self.assertEqual((scope["num_eligible_tensors"], scope["protected"]["embedding_tables"]["num_tensors"]), (8, 3))

            audit = TinyBertFaultCheckConfig(run_id="tiny-audit", source_clustered_results="results/tiny-clustered", reference_probability=.05, cpu_threads=1)
            sweep = TinyBertSweepConfig(run_id="tiny-sweep", probabilities=[0, .05, .5], patterns_per_rate=2, cpu_threads=1, source_audit_results="results/tiny-audit")
            with self.assertRaises(FileNotFoundError):
                run_tinybert_fault_sweep(sweep, root)
            report = run_tinybert_fault_checks(audit, root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(sorted(report["checks"]), sorted(REQUIRED_CHECKS))
            aggregate = run_tinybert_fault_sweep(sweep, root)
            results = root / "results/tiny-sweep"
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            self.assertEqual([entry["pattern_id"] for entry in json.loads((results / "plan.json").read_text())],
                             ["rate00-zero", "rate01-seed42", "rate01-seed43", "rate02-seed44", "rate02-seed45"])
            self.assertEqual(aggregate["splits"], ["selection", *test_names])
            references = json.loads((results / "references.json").read_text())
            for name in test_names:
                self.assertEqual(aggregate["rates"][0]["splits"][name]["accuracy"]["mean"], references[name]["clustered_no_faults"]["accuracy"])
                self.assertEqual(references[name]["clustered_no_faults"], metrics["measurements"][name]["clustered_no_faults"])
            rows = (results / "measurements.csv").read_text().splitlines()
            self.assertEqual(len(rows), 1 + 5 * 3)
            self.assertTrue((results / "accuracy_vs_fault_rate.png").stat().st_size > 0 and (results / "accuracy_vs_fault_rate.pdf").stat().st_size > 0)
            with self.assertRaises(FileExistsError):
                run_tinybert_fault_sweep(sweep, root)
            # A changed input artifact is detected before any new run starts.
            (root / "results/tiny-clustered/layers.json").write_text("{}\n")
            with self.assertRaises(AssertionError):
                run_tinybert_fault_sweep(dataclasses.replace(sweep, run_id="tiny-sweep-2"), root)


if __name__ == "__main__":
    unittest.main()
