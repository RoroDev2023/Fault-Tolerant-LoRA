"""Offline checks for the TinyBERT LoRA controls analysis (Part 7)."""

import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_tinybert_baseline as fixture
from fault_lora.evaluation import tinybert_baseline
from fault_lora.evaluation.tinybert_baseline import run_tinybert_baseline
from fault_lora.evaluation.tinybert_clean_eval import CleanEvalConfig, run_clean_eval
from fault_lora.evaluation.tinybert_clustered import TinyBertClusteredConfig, run_tinybert_clustered
from fault_lora.evaluation.tinybert_fault_checks import TinyBertFaultCheckConfig, run_tinybert_fault_checks
from fault_lora.evaluation.tinybert_fault_sweep import TinyBertSweepConfig, run_tinybert_fault_sweep
from fault_lora.evaluation.tinybert_lora_controls import TinyBertControlsConfig, analyze, run_controls, table_rows
from fault_lora.training.tinybert_recovery import TinyBertRecoveryConfig, recovery_run_id, run_tinybert_recovery
from fault_lora.training.tinybert_recovery_matrix import TinyBertMatrixConfig, run_tinybert_matrix

SCRIPTS = ("run_tinybert_clean_eval.py", "run_tinybert_clustered.py", "check_tinybert_faults.py", "run_tinybert_fault_sweep.py", "run_tinybert_recovery.py",
           "run_tinybert_recovery_matrix.py", "analyze_tinybert_lora_controls.py")
TEST = "test-a"


def metrics(kind, condition, pattern, seed, before, after, rate=None):
    split = lambda accuracy: {"num_examples": 100, "num_correct": round(100 * accuracy), "accuracy": accuracy, "cross_entropy_loss": .5}
    config = {"base_kind": kind, "condition": condition, "pattern_id": pattern, "seed": seed, "device": "mps", "source_sweep_results": "s"}
    return {"run_id": recovery_run_id(kind, condition, pattern, 4, seed), "config": config, "pattern_id": pattern, "fault_probability": rate, "selected_epoch": 1.0,
            "realized_fault_rate": rate, "before": {"selection": split(before), TEST: split(before)}, "after": {"selection": split(after), TEST: split(after)},
            "best_checkpoint": f"checkpoints/{kind}-{condition}-{seed}/best.pt"}


class ControlsArithmeticTests(unittest.TestCase):
    def design(self):
        config = TinyBertControlsConfig(conditions={"linear-only": [.5]}, control_seeds=[42, 43], faulty_seeds=[42], patterns_per_rate=2)
        runs = {}
        for seed, after in ((42, .90), (43, .92)):
            runs[("clean", None, None, seed)] = metrics("clean", None, None, seed, .88, after)
        for seed, after in ((42, .89), (43, .91)):
            runs[("clustered", "linear-only", None, seed)] = metrics("clustered", "linear-only", None, seed, .86, after)
        for pattern, before, after in (("p1", .70, .85), ("p2", .60, .88)):
            runs[("faulty", "linear-only", pattern, 42)] = metrics("faulty", "linear-only", pattern, 42, before, after, .5)
        transfers = []
        for pattern, before in (("p1", .70), ("p2", .60)):
            for seed, accuracy in ((42, .80), (43, .82)):
                transfers.append({"adapter_base": {"base_kind": "clustered", "condition": "linear-only", "pattern_id": None}, "adapter_seed": seed,
                                  "evaluation_base": {"base_kind": "faulty", "condition": "linear-only", "pattern_id": pattern},
                                  "with_adapter": {TEST: {"accuracy": accuracy}}, "without_adapter": runs[("faulty", "linear-only", pattern, 42)]["before"],
                                  "accuracy_gain_percentage_points": {TEST: 100 * (accuracy - before)}})
        return config, runs, transfers

    def test_definitions_match_hand_calculations(self):
        config, runs, transfers = self.design()
        result = analyze(runs, transfers, config, [TEST])["test_sets"][TEST]
        self.assertAlmostEqual(result["extra_training_effect_percentage_points"]["mean"], 3.0)
        condition = result["conditions"]["linear-only"]
        self.assertAlmostEqual(condition["clustering_repair_percentage_points"]["mean"], 4.0)
        rate = condition["rates"][0]
        self.assertAlmostEqual(rate["accuracy_gain_percentage_points"]["mean"], (15 + 28) / 2)
        # R = (after - before) / (clean - before), clean = 0.88.
        self.assertAlmostEqual(rate["normalized_recovery"]["mean"], ((.85 - .70) / (.88 - .70) + (.88 - .60) / (.88 - .60)) / 2)
        self.assertAlmostEqual(rate["normalized_recovery_to_clustered"]["mean"], ((.85 - .70) / (.86 - .70) + (.88 - .60) / (.86 - .60)) / 2)
        # Fault-specific gain: LoRA(faulty) minus mean LoRA(clustered) = 0.90.
        self.assertAlmostEqual(rate["fault_specific_gain_percentage_points"]["mean"], ((.85 - .90) + (.88 - .90)) / 2 * 100)
        self.assertAlmostEqual(rate["fault_specific_gain_seed_matched_percentage_points"]["mean"], ((.85 - .89) + (.88 - .89)) / 2 * 100)
        transfer = rate["transfer"]
        self.assertEqual(transfer["num_evaluations"], 4)
        self.assertAlmostEqual(transfer["accuracy_with_adapter"]["mean"], .81)
        self.assertAlmostEqual(transfer["trained_minus_transfer_percentage_points"]["mean"], ((.85 - .80) + (.85 - .82) + (.88 - .80) + (.88 - .82)) / 4 * 100)
        rows = table_rows({"test_sets": {TEST: result}})
        self.assertEqual([row["base"] for row in rows], ["clean", "clustered", "faulty"])

    def test_design_violations_are_rejected(self):
        config, runs, transfers = self.design()
        for broken in ({key: value for key, value in runs.items() if key != ("clustered", "linear-only", None, 43)},
                       {**runs, ("faulty", "linear-only", "p1", 43): metrics("faulty", "linear-only", "p1", 43, .70, .8, .5)}):
            with self.subTest(), self.assertRaises(AssertionError):
                analyze(broken, transfers, config, [TEST])
        with self.assertRaises(AssertionError):
            analyze(runs, transfers[:-1], config, [TEST])
        changed = [dict(item) for item in transfers]
        changed[0] = {**changed[0], "without_adapter": {**changed[0]["without_adapter"], TEST: {"accuracy": .1}}}
        with self.assertRaises(AssertionError):
            analyze(runs, changed, config, [TEST])
        for invalid in ({"conditions": {"other": [.1]}}, {"control_seeds": [42]}, {"faulty_seeds": [99]}, {"run_id": "../x"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                dataclasses.replace(TinyBertControlsConfig(), **invalid).validate()


class ControlsPipelineTests(unittest.TestCase):
    def test_analysis_of_a_tiny_matrix_with_an_extra_run(self):
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
            run_tinybert_fault_sweep(TinyBertSweepConfig(run_id="tiny-sweep", probabilities=[0, .5], patterns_per_rate=2, cpu_threads=1, source_audit_results="results/tiny-audit"), root)
            recipe = {"epochs": 1, "evaluations_per_epoch": 1, "batch_size": 16, "expected_training_examples": 80, "loss_window_steps": 2, "cpu_threads": 1, "device": "cpu"}
            sweep = "results/tiny-sweep"
            runs = [{"base_kind": "clean", "condition": None, "pattern_id": None, "source_sweep_results": sweep, "seed": seed} for seed in (42, 43)]
            runs += [{"base_kind": "clustered", "condition": "linear-only", "pattern_id": None, "source_sweep_results": sweep, "seed": seed} for seed in (42, 43)]
            runs += [{"base_kind": "faulty", "condition": "linear-only", "pattern_id": "rate01-seed43", "source_sweep_results": sweep, "seed": 42}]
            transfers = [{"adapter_run": index, "base_kind": "faulty", "pattern_id": pattern} for index in (2, 3) for pattern in ("rate01-seed42", "rate01-seed43")]
            matrix = {"matrix_id": "tiny-matrix", "recipe": recipe, "runs": runs, "transfers": transfers, "results_root": "results", "checkpoint_root": "checkpoints"}
            (root / "configs").mkdir()
            (root / "configs/matrix.json").write_text(json.dumps(matrix))
            # The pilot-style extra run fills the one cell the matrix omits.
            extra = TinyBertRecoveryConfig(**recipe, run_id=recovery_run_id("faulty", "linear-only", "rate01-seed42", 4, 42) + "-extra", base_kind="faulty", condition="linear-only",
                                           pattern_id="rate01-seed42", source_sweep_results=sweep, seed=42)
            run_tinybert_recovery(extra, root)
            run_tinybert_matrix(TinyBertMatrixConfig(**matrix), root)
            config = TinyBertControlsConfig(run_id="tiny-controls", source_matrix_config="configs/matrix.json", extra_runs=[f"results/{extra.run_id}"],
                                            conditions={"linear-only": [.5]}, control_seeds=[42, 43], faulty_seeds=[42], patterns_per_rate=2, device="cpu")
            analysis = run_controls(config, root)
            results = root / "results/tiny-controls"
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            self.assertEqual(sorted(analysis["test_sets"]), ["sst2-original-test-25", "sst2-validation-30"])
            rate = analysis["test_sets"]["sst2-validation-30"]["conditions"]["linear-only"]["rates"][0]
            self.assertEqual((rate["num_runs"], rate["transfer"]["num_evaluations"]), (2, 4))
            self.assertTrue((results / "lora_controls.png").stat().st_size > 0)
            self.assertNotIn("selection", (results / "conditions.csv").read_text())
            with self.assertRaises(FileExistsError):
                run_controls(config, root)
            # A run on another device is rejected.
            with self.assertRaises(AssertionError):
                run_controls(dataclasses.replace(config, run_id="tiny-controls-mps", device="mps"), root)


if __name__ == "__main__":
    unittest.main()
