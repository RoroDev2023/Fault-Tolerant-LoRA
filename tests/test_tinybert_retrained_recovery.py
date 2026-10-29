"""Offline checks for Step 12c: LoRA on the retrained TinyBERT baseline with Linear or Linear + embedding adapters."""

import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_tinybert_codebook as codebook_tests
from fault_lora.evaluation.tinybert_retrained_faults import TinyBertRetrainedSweepConfig, run_tinybert_retrained_sweep
from fault_lora.training.tinybert_codebook_finetune import TinyBertCodebookConfig, run_tinybert_codebook_finetune
from fault_lora.training.tinybert_recovery import TinyBertRecoveryConfig, load_tinybert_recovered_model, recovery_config, run_tinybert_recovery
from fault_lora.training.tinybert_retrained_recovery import (TinyBertRetrainedMatrixConfig, TinyBertRetrainedRecoveryConfig, matrix_plan, retrained_recovery_run_id,
                                                             run_retrained_matrix)

EMBEDDINGS = ["bert.embeddings.word_embeddings", "bert.embeddings.position_embeddings", "bert.embeddings.token_type_embeddings"]


def build_retrained_chain(root):
    """Tiny Steps 10a-10b, a monotone Step 12a retraining of the Linear-only encoding, and its audited Step 12b sweep."""
    codebook_tests.CodebookPipelineTests.build_chain(None, root)
    for name in ("run_tinybert_retrained_fault_sweep.py", "run_tinybert_recovery.py"):
        (root / "scripts" / name).write_text("placeholder\n")
    run_tinybert_codebook_finetune(TinyBertCodebookConfig(run_id="tiny-monotone", variants=[{**codebook_tests.CODEBOOK_ONLY, "variant_id": "lin-codebook"}], parameterization="monotone",
                                                          source_sweeps={"linear-only": "results/tiny-sweep"}, epochs=1, evaluations_per_epoch=1, batch_size=16, learning_rate=1e-3,
                                                          expected_training_examples=80, device="cpu", cpu_threads=1), root)
    group = {"group_id": "lin", "variant_id": "lin-codebook", "condition": "linear-only", "sweeps": ["results/tiny-sweep"], "audit": True}
    run_tinybert_retrained_sweep(TinyBertRetrainedSweepConfig(run_id="tiny-retrained-sweep", finetune_results="results/tiny-monotone", groups=[group], probabilities=[0, .3],
                                                              patterns_per_rate=2, audit_probability=.05, device="cpu"), root)
    return [record for record in json.loads((root / "results/tiny-retrained-sweep/measurements.json").read_text())]


class RetrainedRecoveryConfigTests(unittest.TestCase):
    def test_placements_retrained_kinds_and_run_ids(self):
        self.assertEqual(retrained_recovery_run_id("linear+emb", "retrained-faulty", "rate07-seed1030", 4, 42), "tinybert-sst2-lora12c-linear-emb-retrained-faulty-rate07-seed1030-r4-train42")
        self.assertEqual(retrained_recovery_run_id("linear", "clean", None, 4, 43), "tinybert-sst2-lora12c-linear-clean-none-r4-train43")
        base = TinyBertRetrainedRecoveryConfig(base_kind="retrained-faulty", pattern_id="rate07-seed1030", adapter_placement="linear+emb")
        base.validate()
        dataclasses.replace(base, base_kind="retrained-clustered", pattern_id=None).validate()
        dataclasses.replace(base, base_kind="clean", condition=None, pattern_id=None).validate()
        for change in ({"adapter_placement": "embeddings"}, {"condition": None}, {"pattern_id": None}, {"base_kind": "retrained-clustered"}, {"base_kind": "retrained"},
                       {"learning_rate": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(base, **change).validate()
        # Saved configurations load as their own class; earlier ones are unchanged.
        self.assertIs(type(recovery_config(dataclasses.asdict(base))), TinyBertRetrainedRecoveryConfig)
        self.assertIs(type(recovery_config(dataclasses.asdict(TinyBertRecoveryConfig()))), TinyBertRecoveryConfig)
        self.assertNotIn("adapter_placement", dataclasses.asdict(TinyBertRecoveryConfig()))
        with self.assertRaises(ValueError):
            TinyBertRecoveryConfig(base_kind="retrained-clustered", pattern_id=None).validate()


class RetrainedRecoveryPipelineTests(unittest.TestCase):
    def test_both_placements_train_verify_reload_and_transfer_on_retrained_bases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            records = build_retrained_chain(root)
            faulty = [record["pattern_id"] for record in records if record["probability"] > 0]
            self.assertEqual(len(faulty), 2)
            recipe = dict(epochs=1, evaluations_per_epoch=2, batch_size=16, expected_training_examples=80, loss_window_steps=2, cpu_threads=1, device="cpu")
            sweep = "results/tiny-retrained-sweep"
            runs = [{"adapter_placement": placement, "base_kind": kind, "condition": None if kind == "clean" else "linear-only", "pattern_id": pattern,
                     "source_sweep_results": "results/tiny-sweep" if kind == "clean" else sweep, "seed": 42}
                    for placement in ("linear", "linear+emb") for kind, pattern in (("clean", None), ("retrained-clustered", None), ("retrained-faulty", faulty[0]))]
            matrix = TinyBertRetrainedMatrixConfig(matrix_id="tiny-12c", recipe=recipe, runs=runs,
                                                   transfers=[{"adapter_run": index, "pattern_id": name} for index in (1, 4) for name in faulty])
            plan = matrix_plan(matrix)
            self.assertEqual(plan[5]["run_id"], retrained_recovery_run_id("linear+emb", "retrained-faulty", faulty[0], 4, 42))
            summary = run_retrained_matrix(matrix, root)
            self.assertEqual(json.loads((root / "results/tiny-12c/status.json").read_text())["status"], "complete")
            self.assertEqual((len(summary["runs"]), len(summary["transfers"])), (6, 4))
            zero = records[0]["models"]["retrained"]
            pattern_record = next(record for record in records if record["pattern_id"] == faulty[0])
            for entry, metrics in zip(plan, summary["runs"]):
                results = root / "results" / entry["run_id"]
                self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
                layout = json.loads((results / "adapter_layout.json").read_text())
                embedded = [item["name"] for item in layout if item["kind"] == "Embedding"]
                self.assertEqual(embedded, EMBEDDINGS if entry["config"]["adapter_placement"] == "linear+emb" else [])
                self.assertEqual(metrics["parameters"]["trainable_adapter"], sum(item["adapter_parameters"] for item in layout))
                self.assertEqual(metrics["parameters"]["adapter_tensor_bytes"], 4 * metrics["parameters"]["trainable_adapter"])
                # Before LoRA is exactly the Step 12b retrained record of that base.
                if metrics["base_kind"] == "retrained-clustered":
                    self.assertEqual(metrics["before"], zero)
                if metrics["base_kind"] == "retrained-faulty":
                    self.assertEqual(metrics["before"], pattern_record["models"]["retrained"])
                    self.assertEqual(json.loads((results / "fault_counts.json").read_text()), pattern_record["counts"])
            linear, linear_emb = summary["runs"][1], summary["runs"][4]
            self.assertGreater(linear_emb["parameters"]["trainable_adapter"], linear["parameters"]["trainable_adapter"])
            model, payload = load_tinybert_recovered_model(root / linear_emb["best_checkpoint"], root)
            self.assertEqual(payload["config"]["adapter_placement"], "linear+emb")
            self.assertFalse(any(module.training for module in model.modules()))
            for item in summary["transfers"]:
                record = next(record for record in records if record["pattern_id"] == item["evaluation_base"]["pattern_id"])
                self.assertEqual(item["without_adapter"], record["models"]["retrained"])
                self.assertEqual(item["evaluation_base"]["base_kind"], "retrained-faulty")
            self.assertEqual([item["adapter_placement"] for item in summary["transfers"]], ["linear", "linear", "linear+emb", "linear+emb"])
            with self.assertRaises(FileExistsError):
                run_retrained_matrix(matrix, root)
            # No Step 12b group for this condition, and a pattern Step 12b never saved, are refused.
            wrong = TinyBertRetrainedRecoveryConfig(**{**plan[2]["config"], "run_id": "tiny-wrong-condition", "condition": "linear-and-embeddings"})
            with self.assertRaises(AssertionError):
                run_tinybert_recovery(wrong, root)
            with self.assertRaises(AssertionError):
                run_tinybert_recovery(TinyBertRetrainedRecoveryConfig(**{**plan[2]["config"], "run_id": "tiny-wrong-pattern", "pattern_id": "rate09-seed99"}), root)
            for bad in ({"transfers": [{"adapter_run": 2, "pattern_id": faulty[0]}]}, {"recipe": {"adapter_placement": "linear"}}, {"runs": [{**runs[0], "extra": 1}]}):
                with self.assertRaises(ValueError):
                    TinyBertRetrainedMatrixConfig(**{**dataclasses.asdict(matrix), **bad}).validate()


if __name__ == "__main__":
    unittest.main()
