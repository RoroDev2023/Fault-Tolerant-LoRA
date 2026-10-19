"""Offline checks for the control-matrix analysis formulas, design checks and saved records."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fault_lora.evaluation.bn_recalibration import BNRecalibrationConfig, run_bn_recalibration
from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.evaluation.fault_sweep import summarize_values
from fault_lora.evaluation.lora_controls import CONDITION_FIELDS, ControlsConfig, analyze_controls, condition_table, run_controls_analysis, step8_regression
from fault_lora.training import recovery
from fault_lora.training.recovery_matrix import MatrixConfig, run_matrix, summary_row
from test_lora_matrix import PATTERNS, SHARED, TinyCIFAR, TinyProject


SPLITS = ("validation", "test")
CONFIG = {"fault_probabilities": [.1], "control_seeds": [42, 43, 44], "faulty_seeds": [42, 43], "rank": 2, "alpha": 2.0}
BEFORE = {"clean": .9, "clustered": .8, "A": .7, "B": .6}
AFTER = {("clean", 42): .91, ("clean", 43): .92, ("clean", 44): .93, ("clustered", 42): .88, ("clustered", 43): .89, ("clustered", 44): .90,
         ("A", 42): .85, ("A", 43): .86, ("B", 42): .87, ("B", 43): .88}
BN_AFTER = {"clustered": .82, "A": .75, "B": .65}


def synthetic_inputs():
    measure = lambda accuracy: {split: {"accuracy": accuracy, "cross_entropy_loss": 1 - accuracy} for split in SPLITS}
    metrics = []
    for (name, seed), after in AFTER.items():
        faulty = name in ("A", "B")
        metrics.append({"run_id": f"lora-{name}-{seed}", "base_kind": "faulty" if faulty else name, "pattern_id": name if faulty else None,
                        "fault_probability": .1 if faulty else (0.0 if name == "clustered" else None), "actual_corrupted_cells": 10 if faulty else (0 if name == "clustered" else None),
                        "realized_fault_rate": .1 if faulty else (0.0 if name == "clustered" else None), "config": {"rank": 2, "alpha": 2.0, "seed": seed}, "selected_epoch": 1,
                        "before": measure(BEFORE[name]), "after": measure(after), "recovery": {split: {"accuracy_gain_percentage_points": 100 * (after - BEFORE[name])} for split in SPLITS},
                        "parameters": {"trainable_adapter": 8}, "training_and_validation_seconds": 1.0, "elapsed_seconds": 2.0})
    bn = [{"condition_id": name, "base_kind": "faulty" if name != "clustered" else name, "pattern_id": name if name != "clustered" else None,
           "before": measure(BEFORE[name]), "after": measure(after)} for name, after in BN_AFTER.items()]
    plan = [{"pattern_id": "zero", "probability": 0.0, "seed": 1, "repeat": 0}, {"pattern_id": "A", "probability": .1, "seed": 2, "repeat": 0}, {"pattern_id": "B", "probability": .1, "seed": 3, "repeat": 1}]
    sweep = [{**entry, "measurements": measure(BEFORE[{"zero": "clustered"}.get(entry["pattern_id"], entry["pattern_id"])])} for entry in plan]
    aggregate = {"rates": [{"probability": p, "num_patterns": len(values), "realized_fault_rate": summarize_values([p] * len(values)),
                            "splits": {split: {"accuracy": summarize_values(values)} for split in SPLITS}} for p, values in ((0.0, [.8]), (.1, [.7, .6]))]}
    references = {split: {"clean": measure(.9)[split], "clustered_no_faults": measure(.8)[split]} for split in SPLITS}
    return metrics, bn, sweep, plan, aggregate, references


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.metrics, self.bn, self.sweep, self.plan, self.aggregate, self.references = synthetic_inputs()
        self.rows = [summary_row(item) for item in self.metrics]
        self.config = ControlsConfig(**CONFIG)

    def analyze(self, rows=None, bn=None, config=None):
        return analyze_controls(self.rows if rows is None else rows, self.bn if bn is None else bn, self.sweep, self.plan, self.aggregate, self.references, config or self.config)

    def test_formulas_match_hand_computed_values(self):
        result = self.analyze()["splits"]["test"]
        clean, clustered, faulty = result["conditions"]
        self.assertEqual([(item["condition"], item["fault_probability"], item["num_runs"], item["num_patterns"]) for item in result["conditions"]],
                         [("clean", None, 3, 0), ("clustered", 0.0, 3, 0), ("faulty", .1, 4, 2)])
        self.assertAlmostEqual(result["extra_training_effect_percentage_points"]["mean"], 2.0)
        self.assertAlmostEqual(result["extra_training_effect_percentage_points"]["sample_std"], 1.0)
        self.assertAlmostEqual(result["extra_training_effect_percentage_points"]["per_seed"]["44"], 3.0)
        # A clean base has no gap to recover, so R is undefined rather than infinite.
        self.assertIsNone(clean["normalized_recovery"])
        self.assertEqual(clean["normalized_recovery_undefined_runs"], 3)
        self.assertAlmostEqual(clustered["normalized_recovery"]["mean"], .9)
        self.assertAlmostEqual(clustered["runs"][0]["normalized_recovery"], .8)
        self.assertAlmostEqual(faulty["accuracy_after"]["mean"], .865)
        self.assertAlmostEqual(faulty["accuracy_after"]["sample_std"], summarize_values([.85, .86, .87, .88])["sample_std"])
        self.assertAlmostEqual(faulty["accuracy_before"]["mean"], .65)
        self.assertAlmostEqual(faulty["runs"][0]["normalized_recovery"], .75)
        self.assertAlmostEqual(faulty["normalized_recovery"]["mean"], (.75 + .8 + .9 + 28 / 30) / 4)
        self.assertAlmostEqual(faulty["accuracy_after_by_pattern"]["B"], .875)
        self.assertAlmostEqual(faulty["accuracy_after_by_seed"]["43"], .87)
        specific = result["fault_specific_gain"][0]
        self.assertAlmostEqual(specific["lora_clustered_accuracy_mean"], .89)
        self.assertAlmostEqual(specific["gain_percentage_points"]["mean"], -2.5)
        self.assertAlmostEqual(specific["gain_percentage_points"]["min"], -4.0)
        self.assertAlmostEqual(specific["seed_matched_gain_percentage_points"]["mean"], -2.0)
        baseline_clustered, baseline_faulty = result["bn_recalibration_vs_lora"]
        self.assertAlmostEqual(baseline_clustered["lora_minus_bn_recalibration_percentage_points"]["mean"], 7.0)
        self.assertAlmostEqual(baseline_clustered["bn_recalibration_normalized_recovery"]["mean"], .2)
        self.assertAlmostEqual(baseline_faulty["bn_recalibration_accuracy"]["mean"], .70)
        self.assertAlmostEqual(baseline_faulty["lora_minus_bn_recalibration_percentage_points"]["mean"], 16.5)
        self.assertAlmostEqual(baseline_faulty["per_pattern"][0]["bn_recalibration_normalized_recovery"], .25)
        self.assertEqual([item["fault_probability"] for item in result["step7_fault_rates"]], [0.0, .1])
        table = condition_table(self.analyze())
        self.assertEqual(len(table), 6)
        self.assertEqual(list(table[0]), CONDITION_FIELDS)
        row = next(row for row in table if row["split"] == "test" and row["condition"] == "faulty")
        self.assertAlmostEqual(float(row["fault_specific_gain_percentage_points_mean"]), -2.5)
        self.assertEqual(table[0]["normalized_recovery_mean"], "")

    def test_design_must_match_the_declared_runs_and_step7_pairing(self):
        variants = [self.rows[:-1], self.rows + [{**self.rows[-1], "seed": 44}], [{**row, "rank": 4} for row in self.rows]]
        changed = copy.deepcopy(self.rows)
        changed[-1]["test_accuracy_before"] = .61
        variants.append(changed)
        changed = copy.deepcopy(self.rows)
        changed[0]["test_accuracy_before"] = .89
        variants.append(changed)
        for i, rows in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(AssertionError):
                self.analyze(rows=rows)
        unpaired = copy.deepcopy(self.bn)
        unpaired[1]["before"]["test"]["accuracy"] = .71
        for bn in (self.bn[:-1], self.bn + [{**self.bn[-1], "condition_id": "C"}], unpaired):
            with self.assertRaises(AssertionError):
                self.analyze(bn=bn)
        with self.assertRaises(AssertionError):
            self.analyze(config=ControlsConfig(**{**CONFIG, "fault_probabilities": [.2]}))
        matrix = {"results": "results/m", "status": "complete"}
        ControlsConfig(**CONFIG, source_matrices=[matrix, {"results": "results/n", "status": "interrupted", "completed_runs": 4}], step8_results="results/s", step8_rerun_id="r").validate()
        for kwargs in ({"fault_probabilities": [0, .1]}, {"fault_probabilities": [.1, .05]}, {"control_seeds": [42]}, {"faulty_seeds": [45]}, {"rank": 0}, {"run_id": "../x"},
                       {"source_matrices": []}, {"source_matrices": [matrix, matrix]}, {"source_matrices": [{**matrix, "status": "interrupted"}]}, {"source_matrices": [{**matrix, "completed_runs": 4}]},
                       {"source_matrices": [{**matrix, "status": "running"}]}, {"source_matrices": [{"results": "results/_aborted/m", "status": "complete"}]},
                       {"step8_results": "results/s"}, {"step8_tolerance_percentage_points": -1}, {"sources_allowed_to_differ": "src"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ControlsConfig(**{**CONFIG, **kwargs}).validate()

    def test_step8_regression_gates_base_counts_and_accuracy_tolerance(self):
        measure = lambda correct: {split: {"num_correct": correct, "accuracy": correct / 10000, "cross_entropy_loss": 1 - correct / 10000} for split in SPLITS}
        recipe = {"pattern_id": "A", "rank": 4, "alpha": 4.0, "target_names": None, "seed": 42, "epochs": 5, "batch_size": 128, "evaluation_batch_size": 256,
                  "learning_rate": .001, "weight_decay": .0001, "max_grad_norm": 1.0}
        original = {"run_id": "step8", "config": recipe, "selected_epoch": 3, "before": measure(7991), "after": measure(9176)}
        rerun = lambda before, after, epoch=3, **changes: {"run_id": "rerun", "config": {**recipe, "base_kind": "faulty", **changes}, "selected_epoch": epoch, "before": measure(before), "after": measure(after)}
        result = step8_regression(original, rerun(7991, 9150, epoch=5), .3)
        self.assertTrue(result["passed"])
        self.assertFalse(result["same_selected_epoch"])
        self.assertEqual(result["splits"]["test"]["after"]["num_correct_difference"], -26)
        self.assertAlmostEqual(result["splits"]["test"]["after"]["accuracy_difference_percentage_points"], -.26)
        self.assertAlmostEqual(result["splits"]["test"]["after"]["cross_entropy_difference"], .0026)
        self.assertFalse(step8_regression(original, rerun(7991, 9145), .3)["passed"])
        self.assertFalse(step8_regression(original, rerun(7990, 9176), .3)["passed"])
        for changes in ({"seed": 43}, {"pattern_id": "B"}, {"rank": 8}, {"base_kind": "clustered"}):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                step8_regression(original, rerun(7991, 9176, **changes), .3)

    def test_saved_analysis_reads_complete_and_interrupted_matrices_and_ignores_archived_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = TinyProject(root)
            for filename in ("scripts/analyze_lora_controls.py", "scripts/run_bn_recalibration.py"):
                (root / filename).write_text("synthetic test input\n")
            run = lambda kind, pattern_id, seed: {"base_kind": kind, "pattern_id": pattern_id, "rank": 2, "alpha": 2.0, "seed": seed}
            first = [run("clean", None, 42), run("clean", None, 43), run("clustered", None, 42), run("clustered", None, 43)]
            second = [run("clustered", None, 42), run("clustered", None, 43)] + [run("faulty", name, 42) for name in PATTERNS]
            real, calls = recovery.run_recovery, []

            def interrupt_third(config, project_root):
                calls.append(config.run_id)
                if len(calls) == 3:
                    real(config, project_root)   # Leaves finished-looking directories that the matrix never recorded.
                    raise KeyboardInterrupt
                return real(config, project_root)

            template = lambda *args, **kwargs: copy.deepcopy(project.template)
            with patch("fault_lora.training.recovery.prepare_sources", side_effect=project.prepare), patch("fault_lora.training.recovery.CIFAR10", TinyCIFAR), \
                    patch("fault_lora.training.recovery.build_resnet18", side_effect=template), patch("fault_lora.evaluation.bn_recalibration.CIFAR10", TinyCIFAR), \
                    patch("fault_lora.evaluation.bn_recalibration.build_resnet18", side_effect=template):
                with patch("fault_lora.training.recovery.run_recovery", side_effect=interrupt_third), self.assertRaises(KeyboardInterrupt):
                    run_matrix(MatrixConfig(matrix_id="first", run_id_prefix="tiny", **SHARED, runs=first), root)
                # Archive the unrecorded run intact, as an interrupted run would be, then rerun it elsewhere.
                for kind in ("results", "checkpoints"):
                    (root / kind / "_aborted").mkdir()
                    (root / kind / "tiny-clustered-none-r2-train42").rename(root / kind / "_aborted" / "tiny-clustered-none-r2-train42")
                run_matrix(MatrixConfig(matrix_id="second", run_id_prefix="tiny", **SHARED, runs=second), root)
                run_bn_recalibration(BNRecalibrationConfig(run_id="bn", conditions=[{"base_kind": "clustered", "pattern_id": None}] + [{"base_kind": "faulty", "pattern_id": name} for name in PATTERNS],
                                                           calibration_examples=20, calibration_batch_size=10, evaluation_batch_size=10, device="cpu", num_workers=0, source_sweep_results="results/tiny-sweep"), root)
            sweep = root / "results/tiny-sweep"
            plan = json.loads((sweep / "plan.json").read_text())
            state = lambda entry: project.state("clustered" if entry["probability"] == 0 else "faulty", entry["pattern_id"])
            records = [{**entry, "measurements": {split: project.measure(state(entry), split) for split in SPLITS}} for entry in plan]
            aggregate = {"rates": [{"probability": p, "num_patterns": len(group), "realized_fault_rate": summarize_values([p] * len(group)),
                                    "splits": {split: {"accuracy": summarize_values([record["measurements"][split]["accuracy"] for record in group])} for split in SPLITS}}
                                   for p in (0.0, .2) for group in [[record for record in records if record["probability"] == p]]]}
            for name, value in (("status.json", {"status": "complete"}), ("measurements.json", records), ("aggregate.json", aggregate), ("references.json", project.references)):
                (sweep / name).write_text(json.dumps(value))
            rerun = json.loads((root / "results/tiny-faulty-rate01-seed52-r2-train42/metrics.json").read_text())
            (root / "results/step8").mkdir()
            (root / "results/step8/metrics.json").write_text(json.dumps({**rerun, "run_id": "step8", "config": {name: value for name, value in rerun["config"].items() if name != "base_kind"}}))
            options = {"fault_probabilities": [.2], "control_seeds": [42, 43], "faulty_seeds": [42], "rank": 2, "alpha": 2.0, "source_bn_results": "results/bn", "source_sweep_results": "results/tiny-sweep",
                       "source_matrices": [{"results": "results/first", "status": "interrupted", "completed_runs": 2}, {"results": "results/second", "status": "complete"}],
                       "step8_results": "results/step8", "step8_rerun_id": "tiny-faulty-rate01-seed52-r2-train42"}
            analysis = run_controls_analysis(ControlsConfig(run_id="controls", **options), root)
            results = root / "results/controls"
            self.assertEqual(json.loads((results / "analysis.json").read_text()), json.loads(json.dumps(analysis)))
            self.assertEqual(json.loads((results / "status.json").read_text())["status"], "complete")
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            self.assertEqual(analysis["source_matrices"], [{"results": "results/first", "status": "interrupted", "planned_runs": 4, "used_runs": 2}, {"results": "results/second", "status": "complete", "planned_runs": 4, "used_runs": 4}])
            self.assertEqual(analysis["design"]["num_runs"], 6)
            self.assertTrue(analysis["step8_regression"]["passed"] and analysis["step8_regression"]["same_selected_epoch"])
            used = [item["path"] for item in json.loads((results / "manifest.json").read_text())["source_artifacts"].values()]
            self.assertFalse(any("_aborted" in path for path in used))
            self.assertIn("results/tiny-clustered-none-r2-train42/metrics.json", used)
            test = analysis["splits"]["test"]
            self.assertEqual([run["run_id"] for run in test["conditions"][1]["runs"]], ["tiny-clustered-none-r2-train42", "tiny-clustered-none-r2-train43"])
            self.assertEqual(test["conditions"][2]["accuracy_before"]["mean"], sum(project.measure(project.state("faulty", name), "test")["accuracy"] for name in PATTERNS) / 2)
            for extension in ("png", "pdf"):
                self.assertGreater((results / f"lora_controls.{extension}").stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                run_controls_analysis(ControlsConfig(run_id="controls", **options), root)
            attempt = lambda **changes: run_controls_analysis(ControlsConfig(run_id="controls-2", **{**options, **changes}), root)
            # An interrupted matrix cannot be passed off as complete, nor with another finished count.
            for matrices in ([{"results": "results/first", "status": "complete"}, options["source_matrices"][1]], [{**options["source_matrices"][0], "completed_runs": 3}, options["source_matrices"][1]],
                             options["source_matrices"][1:], options["source_matrices"] + [{"results": "results/second/../second", "status": "complete"}]):
                with self.assertRaises(AssertionError):
                    attempt(source_matrices=matrices)
            # Run source code may differ only where declared.
            (root / "src/fault_lora/changed.py").write_text("x = 1\n")
            source = root / "results/tiny-clean-none-r2-train42/source_sha256.json"
            saved = source.read_text()
            source.write_text(json.dumps({**json.loads(saved), "src/fault_lora/changed.py": "0" * 64}))
            with self.assertRaises(AssertionError):
                attempt()
            # A declared Step 9b exemption alone no longer accepts it; only a listed exact change does.
            with self.assertRaises(AssertionError):
                attempt(sources_allowed_to_differ=["src/fault_lora/changed.py"])
            (root / "configs").mkdir(exist_ok=True)
            current = sha256_file(root / "src/fault_lora/changed.py")
            (root / "configs/accepted_source_changes.json").write_text(json.dumps([{"path": "src/fault_lora/changed.py", "old_sha256": "0" * 64, "new_sha256": current,
                                                                                  "reason": "synthetic test change", "date": "2026-10-18", "commit": None}]))
            analysis = run_controls_analysis(ControlsConfig(run_id="controls-3", **options), root)
            self.assertEqual(analysis["source_files_changed_since_runs"], ["src/fault_lora/changed.py"])
            self.assertEqual([(change["path"], change["list"], change["records"]) for change in analysis["accepted_source_hash_changes"]], [("src/fault_lora/changed.py", "source", 1)])
            self.assertEqual(json.loads((root / "results/controls-3/verification.json").read_text())["accepted_source_hash_changes"][0]["current_sha256"], current)
            (root / "configs/accepted_source_changes.json").unlink()
            source.write_text(saved)
            original = json.loads((root / "results/step8/metrics.json").read_text())
            original["after"]["test"]["accuracy"] += .01
            (root / "results/step8/metrics.json").write_text(json.dumps(original))
            with self.assertRaises(AssertionError):
                attempt()
            self.assertEqual(json.loads((root / "results/controls-2/status.json").read_text())["status"], "failed")
            self.assertFalse(json.loads((root / "results/controls-2/step8_regression.json").read_text())["passed"])
            self.assertFalse((root / "results/controls-2/analysis.json").exists() and (root / "results/controls-2/verification.json").exists())


if __name__ == "__main__":
    unittest.main()
