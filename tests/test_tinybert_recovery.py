"""Offline checks for TinyBERT LoRA recovery (Part 7 code) on the tiny randomly initialized BERT fixture."""

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
from fault_lora.evaluation.tinybert_baseline import run_tinybert_baseline
from fault_lora.evaluation.tinybert_clean_eval import CleanEvalConfig, run_clean_eval
from fault_lora.evaluation.tinybert_clustered import TinyBertClusteredConfig, run_tinybert_clustered
from fault_lora.evaluation.tinybert_fault_checks import TinyBertFaultCheckConfig, run_tinybert_fault_checks
from fault_lora.evaluation.tinybert_fault_sweep import TinyBertSweepConfig, run_tinybert_fault_sweep
from fault_lora.evaluation.tinybert_sources import build_tinybert, linear_weight_names, synthetic_token_batch
from fault_lora.models.lora import LoRALayer, adapter_state_dict, attach_lora, base_state_dict
from fault_lora.models.tinybert_lora import CappedRankLoRALayer, attach_linear_lora
from fault_lora.training.tinybert_recovery import (TinyBertRecoveryConfig, assert_frozen_base, evaluate_tinybert_adapter_on_base, load_tinybert_recovered_model,
                                                   recovery_run_id, run_tinybert_recovery)
from fault_lora.training.tinybert_recovery_matrix import TinyBertMatrixConfig, matrix_plan, run_tinybert_matrix

SCRIPTS = ("run_tinybert_clean_eval.py", "run_tinybert_clustered.py", "check_tinybert_faults.py", "run_tinybert_fault_sweep.py", "run_tinybert_recovery.py", "run_tinybert_recovery_matrix.py")


class AdapterOnBertTests(unittest.TestCase):
    def test_existing_attach_lora_covers_every_linear_and_keeps_bert_frozen_in_eval(self):
        model = fixture.tiny_model()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        probe = synthetic_token_batch(model.config.to_dict(), length=8)
        with torch.no_grad():
            before = model(**probe).logits
        # The existing attach_lora rejects rank 4 on the 16 -> 2 classifier; the sibling caps only that layer.
        with self.assertRaises(ValueError):
            attach_lora(fixture.tiny_model(), 4, 4.0)
        torch.manual_seed(0)
        layout = attach_linear_lora(model, 4, 4.0)
        self.assertEqual([item["name"] + ".weight" for item in layout], linear_weight_names(fixture.tiny_model()))
        self.assertEqual([item["name"] for item in layout if item["adapter_class"] == "CappedRankLoRALayer"], ["classifier"])
        self.assertEqual(layout[-1]["effective_rank_bound"], 2)
        self.assertEqual((layout[-1]["A_shape"], layout[-1]["B_shape"]), ([4, 16], [2, 4]))
        self.assertTrue(all(isinstance(model.get_submodule(item["name"]), LoRALayer) for item in layout))
        self.assertIsInstance(model.classifier, CappedRankLoRALayer)
        # Layers where rank fits use the original class and match attach_lora's seeded initialization exactly.
        reference = fixture.tiny_model()
        torch.manual_seed(0)
        attach_lora(reference, 4, 4.0, target_names=[item["name"] for item in layout[:-1]])
        for name in [item["name"] for item in layout[:-1]]:
            self.assertTrue(torch.equal(model.get_submodule(name).lora_A.weight, reference.get_submodule(name).lora_A.weight))
        self.assertEqual(sum(item["adapter_parameters"] for item in layout), sum(value.numel() for value in adapter_state_dict(model).values()))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(**probe).logits, before))
        self.assertFalse(any(module.training for module in model.modules()))
        loss = model(**probe).logits.sum()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for name, parameter in model.named_parameters() if "lora_A" in name or "lora_B" in name))
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters() if "lora_" not in name))
        for parameter in model.parameters():
            parameter.grad = None
        assert_frozen_base(model, state)
        self.assertEqual(sorted(base_state_dict(model)), sorted(state))
        # Dropout stays off in eval mode: repeated forwards are identical.
        with torch.no_grad():
            self.assertTrue(torch.equal(model(**probe).logits, model(**probe).logits))
        model.train()
        with self.assertRaises(AssertionError):
            assert_frozen_base(model, state)

    def test_configuration_and_run_ids(self):
        self.assertEqual(recovery_run_id("faulty", "linear-and-embeddings", "rate07-seed1030", 4, 42), "tinybert-sst2-lora-faulty-linemb-rate07-seed1030-r4-train42")
        self.assertEqual(recovery_run_id("clean", None, None, 4, 43), "tinybert-sst2-lora-clean-none-none-r4-train43")
        TinyBertRecoveryConfig().validate()
        base = TinyBertRecoveryConfig()
        for change in ({"base_kind": "clean"}, {"base_kind": "clustered"}, {"condition": None}, {"condition": "other"}, {"pattern_id": None}, {"learning_rate": 0},
                       {"schedule": "cosine-per-epoch"}, {"batch_size": 0}, {"evaluations_per_epoch": 0}, {"device": "bogus"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(base, **change).validate()
        dataclasses.replace(base, base_kind="clean", condition=None, pattern_id=None).validate()
        dataclasses.replace(base, base_kind="clustered", pattern_id=None).validate()


class RecoveryPipelineTests(unittest.TestCase):
    def test_recovery_runs_verify_reload_transfer_and_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            step10a = fixture.TinyBertBaselineRunTests.build(None, root)
            for name in SCRIPTS:
                (root / "scripts" / name).write_text("placeholder\n")
            with mock.patch.object(tinybert_baseline, "PROBES", {}):
                run_tinybert_baseline(step10a, root)
            clean = run_clean_eval(CleanEvalConfig(run_id="tiny-clean", source_results="results/tiny-run", expected_original_test_examples=25), root)
            run_tinybert_clustered(TinyBertClusteredConfig(run_id="tiny-clustered", clean_checkpoint_sha256=clean["checkpoint_sha256"], source_clean_eval_results="results/tiny-clean", clustering_threads=1), root)
            run_tinybert_fault_checks(TinyBertFaultCheckConfig(run_id="tiny-audit", source_clustered_results="results/tiny-clustered", reference_probability=.05, cpu_threads=1), root)
            run_tinybert_fault_sweep(TinyBertSweepConfig(run_id="tiny-sweep", probabilities=[0, .3, .9], patterns_per_rate=2, cpu_threads=1, source_audit_results="results/tiny-audit"), root)

            common = dict(source_sweep_results="results/tiny-sweep", epochs=2, evaluations_per_epoch=2, batch_size=16, expected_training_examples=80, loss_window_steps=2, cpu_threads=1)
            faulty = TinyBertRecoveryConfig(run_id=recovery_run_id("faulty", "linear-only", "rate02-seed44", 4, 42), base_kind="faulty", condition="linear-only", pattern_id="rate02-seed44", **common)
            metrics = run_tinybert_recovery(faulty, root)
            results = root / "results" / faulty.run_id
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            history = json.loads((results / "history.json").read_text())
            self.assertEqual([item["epoch"] for item in history], [.5, 1.0, 1.5, 2.0])
            self.assertEqual(sorted(metrics["after"]), ["selection", "sst2-original-test-25", "sst2-validation-30"])
            self.assertIn(metrics["selected_epoch"], [0.0, .5, 1.0, 1.5, 2.0])
            self.assertTrue(metrics["training_loss"]["all_finite"])
            sweep_record = json.loads((root / "results/tiny-sweep/patterns/rate02-seed44.json").read_text())
            for name, value in metrics["before"].items():
                self.assertEqual(value, sweep_record["measurements"][name])
            reloaded, payload = load_tinybert_recovered_model(root / metrics["best_checkpoint"], root)
            self.assertEqual(payload["epoch"], metrics["selected_epoch"])
            self.assertFalse(any(module.training for module in reloaded.modules()))
            with self.assertRaises(FileExistsError):
                run_tinybert_recovery(faulty, root)

            for kind in ("clean", "clustered"):
                config = dataclasses.replace(faulty, run_id=recovery_run_id(kind, None if kind == "clean" else "linear-only", None, 4, 42), base_kind=kind,
                                             condition=None if kind == "clean" else "linear-only", pattern_id=None, epochs=1)
                result = run_tinybert_recovery(config, root)
                self.assertEqual(json.loads((root / "results" / config.run_id / "verification.json").read_text())["status"], "passed")
                if kind == "clustered":
                    clustered_checkpoint = root / result["best_checkpoint"]
            # Transfer: the clustered-trained adapter on a faulty pattern; zero-adapter numbers must equal the sweep record.
            transfer = evaluate_tinybert_adapter_on_base(clustered_checkpoint, "faulty", "rate02-seed45", root)
            record = json.loads((root / "results/tiny-sweep/patterns/rate02-seed45.json").read_text())
            self.assertEqual(transfer["without_adapter"], record["measurements"])
            self.assertFalse(transfer["same_base"])
            with self.assertRaises(AssertionError):
                run_tinybert_recovery(dataclasses.replace(faulty, run_id="tiny-wrong-condition", condition="linear-and-embeddings"), root)

            matrix = TinyBertMatrixConfig(matrix_id="tiny-matrix", recipe={key: value for key, value in common.items() if key != "source_sweep_results"} | {"epochs": 1},
                                          runs=[{"base_kind": "clustered", "condition": "linear-only", "pattern_id": None, "source_sweep_results": "results/tiny-sweep", "seed": 43},
                                                {"base_kind": "faulty", "condition": "linear-only", "pattern_id": "rate01-seed42", "source_sweep_results": "results/tiny-sweep", "seed": 43}],
                                          transfers=[{"adapter_run": 0, "base_kind": "faulty", "pattern_id": "rate01-seed43"}])
            self.assertEqual([entry["run_id"] for entry in matrix_plan(matrix)], [recovery_run_id("clustered", "linear-only", None, 4, 43), recovery_run_id("faulty", "linear-only", "rate01-seed42", 4, 43)])
            summary = run_tinybert_matrix(matrix, root)
            self.assertEqual(len(summary["runs"]), 2)
            self.assertEqual(len(summary["transfers"]), 1)
            self.assertEqual(json.loads((root / "results/tiny-matrix/status.json").read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                run_tinybert_matrix(matrix, root)


if __name__ == "__main__":
    unittest.main()
