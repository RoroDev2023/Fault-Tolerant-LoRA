"""Offline checks for Step 11c: retrained recovery bases, the controls analysis, and decision-file currency."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.evaluation.fault_sweep import SweepConfig, summarize_values, sweep_plan
from fault_lora.evaluation.lora_controls import ControlsConfig, analyze_controls
from fault_lora.evaluation.retrained_faults import RetrainedSweepConfig, extended_plan
from fault_lora.evaluation.retrained_lora_controls import RetrainedControlsConfig, build_analysis
from fault_lora.evaluation.source_hashes import load_accepted_changes
from fault_lora.memory.clustering import cluster_model, decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, generate_fault_pattern, summarize_fault_pattern
from fault_lora.training import recovery
from fault_lora.training.recovery import RecoveryConfig, build_base_state, load_recovered_model
from fault_lora.training.recovery_matrix import MatrixConfig, run_matrix, summary_row
from fault_lora.training.retrained_bases import prepare_retrained_sources
from test_lora_controls import CONFIG as CONTROLS_CONFIG, synthetic_inputs
from test_lora_matrix import TinyCIFAR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("validation", "test")


class TinyRetrained:
    """A synthetic Step 11b directory for a tiny model whose retrained codebooks keep the Step 4 cells."""

    def __init__(self, root):
        torch.manual_seed(0)
        self.template = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))
        self.step4, _ = cluster_model(self.template, 4, threads=1)
        self.retrained = copy.deepcopy(self.step4)
        for record in self.retrained["clustered_tensors"].values():
            record["codebook"] = record["codebook"] * 1.1 + .01
        self.split = {"train": list(range(20)), "validation": list(range(20, 30)), "test": {"indices": list(range(20))}}
        dataset = TinyCIFAR("unused", True, None, False)
        self.subsets = {"validation": Subset(dataset, self.split["validation"]), "test": Subset(dataset, self.split["test"]["indices"])}
        for name in ("scripts/run_lora_recovery.py", "scripts/run_lora_matrix.py", "requirements.txt", "requirements-macos-arm64.lock.txt"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("synthetic test input\n")
        (root / "src/fault_lora").mkdir(parents=True)
        step7 = sweep_plan(SweepConfig(probabilities=[0, .2], patterns_per_rate=2, seed=52))
        self.sweep_config = RetrainedSweepConfig(audit_results="results/audit", extra_probabilities=[.4], extra_first_seed=60, patterns_per_rate=2, ablation_probabilities=[0, .4],
                                                 baseline_fingerprint=encoding_fingerprint(self.retrained))
        self.plan = extended_plan(step7, [.4], 60, 2)
        sweep = root / "results/sweep11b"
        (sweep / "patterns").mkdir(parents=True)
        (root / "checkpoints/sweep11b/patterns").mkdir(parents=True)
        (root / "results/audit").mkdir(parents=True)
        records = []
        for entry in self.plan:
            pattern = generate_fault_pattern(self.retrained, entry["probability"], seed=entry["seed"])
            path = root / "checkpoints/sweep11b/patterns" / f"{entry['pattern_id']}.pt"
            torch.save(pattern, path)
            counts = {key: value for key, value in summarize_fault_pattern(self.retrained, pattern).items() if key != "layers"}
            record = {**entry, "counts": counts, "pattern": str(path.relative_to(root)), "pattern_sha256": sha256_file(path),
                      "models": {"retrained": self.measure(apply_fault_pattern(self.retrained, pattern))}}
            records.append(record)
            (sweep / "patterns" / f"{entry['pattern_id']}.json").write_text(json.dumps(record))
        clean = self.measure(self.template.state_dict())
        files = {"status.json": {"status": "complete"}, "verification.json": {"status": "passed"}, "config.json": asdict(self.sweep_config), "plan.json": self.plan,
                 "measurements.json": records, "references.json": {"clean": clean}, "manifest.json": {}, "source_sha256.json": {}}
        for name, value in files.items():
            (sweep / name).write_text(json.dumps(value))
        for name in ("report.json", "status.json"):
            (root / "results/audit" / name).write_text(json.dumps({"status": "passed"}))
        (root / "clean.pt").write_text("synthetic")
        self.records = records
        self.sources = {"retrained": self.retrained, "step7_plan": step7, "clean_metrics": clean, "split": self.split,
                        "metadata": {"num_classes": 10, "classes": TinyCIFAR.classes, "preprocessing": {"input_size": [3, 32, 32]}, "split_sha256": "synthetic", "model_state_dict": {}},
                        "paths": {"clean": root / "clean.pt"}, "hashes": {"clean": sha256_file(root / "clean.pt")}}

    def measure(self, state):
        model = copy.deepcopy(self.template)
        model.load_state_dict(state)
        return {name: evaluate_classifier(model, DataLoader(subset, batch_size=10), "cpu") for name, subset in self.subsets.items()}

    def config(self, **kwargs):
        return RecoveryConfig(source_sweep_results="results/sweep11b", source_audit_results="results/audit", **kwargs)

    def patches(self):
        return [patch("fault_lora.training.retrained_bases.load_baselines", return_value=self.sources), patch("fault_lora.training.recovery.CIFAR10", TinyCIFAR),
                patch("fault_lora.training.recovery.build_resnet18", side_effect=lambda *args, **kwargs: copy.deepcopy(self.template))]


class RetrainedBaseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.tiny = TinyRetrained(self.root)
        for item in self.tiny.patches():
            item.start()
            self.addCleanup(item.stop)

    def test_configuration_requires_a_pattern_exactly_for_retrained_faulty(self):
        RecoveryConfig(base_kind="retrained-clustered", pattern_id=None).validate()
        RecoveryConfig(base_kind="retrained-faulty", pattern_id="rate03-seed52").validate()
        for kwargs in ({"base_kind": "retrained-clustered"}, {"base_kind": "retrained-faulty", "pattern_id": None}, {"base_kind": "retrained"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RecoveryConfig(**kwargs).validate()

    def test_sources_and_states_come_from_the_retrained_encoding_and_step11b_patterns(self):
        faulty_id = self.tiny.plan[1]["pattern_id"]
        encoding, metadata, split, record, references, paths, hashes = recovery.prepare_sources(self.root, self.tiny.config(base_kind="retrained-faulty", pattern_id=faulty_id))
        self.assertEqual(encoding_fingerprint(encoding), encoding_fingerprint(self.tiny.retrained))
        self.assertNotIn("model_state_dict", metadata)
        self.assertEqual((record["pattern_id"], record["probability"], record["accepted_source_hash_changes"]), (faulty_id, .2, []))
        self.assertEqual(record["measurements"], self.tiny.records[1]["models"]["retrained"])
        self.assertEqual(references["test"]["clustered_no_faults"], self.tiny.records[0]["models"]["retrained"]["test"])
        self.assertIn("selected_pattern", paths)
        self.assertTrue(all(sha256_file(path) == hashes[name] for name, path in paths.items()))
        config = self.tiny.config(base_kind="retrained-faulty", pattern_id=faulty_id)
        state = build_base_state(config, encoding, paths)
        expected = apply_fault_pattern(self.tiny.retrained, generate_fault_pattern(self.tiny.retrained, .2, seed=self.tiny.plan[1]["seed"]))
        self.assertTrue(all(torch.equal(state[name], value) for name, value in expected.items()))
        clustered = self.tiny.config(base_kind="retrained-clustered", pattern_id=None)
        encoding, _, _, record, _, paths, _ = recovery.prepare_sources(self.root, clustered)
        self.assertEqual((record["pattern_id"], record["probability"], record["counts"]["actual_corrupted_cells"]), (None, 0.0, 0))
        self.assertNotIn("selected_pattern", paths)
        state = build_base_state(clustered, encoding, paths)
        self.assertTrue(all(torch.equal(state[name], value) for name, value in decode_state(self.tiny.retrained).items()))

    def test_invalid_patterns_and_incomplete_sweeps_rejected(self):
        for pattern_id in ("rate00-zero", "rate09-seed99"):
            with self.subTest(pattern_id=pattern_id), self.assertRaises(AssertionError):
                prepare_retrained_sources(self.root, self.tiny.config(base_kind="retrained-faulty", pattern_id=pattern_id))
        # A pattern bound to another encoding (the Step 4 one) must not be accepted.
        record = self.tiny.records[1]
        torch.save(generate_fault_pattern(self.tiny.step4, record["probability"], seed=record["seed"]), self.root / record["pattern"])
        with self.assertRaises(AssertionError):
            prepare_retrained_sources(self.root, self.tiny.config(base_kind="retrained-faulty", pattern_id=record["pattern_id"]))
        (self.root / "results/sweep11b/verification.json").write_text(json.dumps({"status": "failed"}))
        with self.assertRaises(AssertionError):
            prepare_retrained_sources(self.root, self.tiny.config(base_kind="retrained-clustered", pattern_id=None))

    def test_matrix_runs_and_verifies_retrained_bases_and_adapters_reload(self):
        shared = {"source_sweep_results": "results/sweep11b", "source_audit_results": "results/audit", "epochs": 2, "batch_size": 10, "evaluation_batch_size": 10, "device": "cpu", "num_workers": 0}
        faulty_id = self.tiny.plan[-1]["pattern_id"]
        config = MatrixConfig(matrix_id="tiny-retrained", run_id_prefix="tiny", **shared, runs=[
            {"base_kind": "retrained-clustered", "pattern_id": None, "rank": 2, "alpha": 2.0, "seed": 42},
            {"base_kind": "retrained-faulty", "pattern_id": faulty_id, "rank": 2, "alpha": 2.0, "seed": 43}])
        rows = run_matrix(config, self.root)
        self.assertEqual([row["run_id"] for row in rows], ["tiny-retrained-clustered-none-r2-train42", f"tiny-retrained-faulty-{faulty_id}-r2-train43"])
        self.assertEqual(json.loads((self.root / "results/tiny-retrained/verification.json").read_text())["status"], "passed")
        self.assertEqual((rows[1]["fault_probability"], rows[1]["test_accuracy_before"]), (.4, self.tiny.records[-1]["models"]["retrained"]["test"]["accuracy"]))
        self.assertTrue((self.root / f"results/{rows[1]['run_id']}/fault_counts.json").exists())
        self.assertFalse((self.root / f"results/{rows[0]['run_id']}/fault_counts.json").exists())
        manifest = json.loads((self.root / f"results/{rows[1]['run_id']}/manifest.json").read_text())
        self.assertIn("Step 11b", manifest["base_selection"])
        model, payload = load_recovered_model(self.root / f"checkpoints/{rows[1]['run_id']}/best.pt", self.root)
        self.assertEqual(payload["encoding_fingerprint"], encoding_fingerprint(self.tiny.retrained))
        self.assertEqual(evaluate_classifier(model, DataLoader(self.tiny.subsets["test"], batch_size=10), "cpu"), json.loads((self.root / f"results/{rows[1]['run_id']}/metrics.json").read_text())["after"]["test"])
        # Patterns must come from the Step 11b plan.
        with self.assertRaises(AssertionError):
            run_matrix(MatrixConfig(matrix_id="tiny-unknown", run_id_prefix="tiny", **shared, runs=[{"base_kind": "retrained-faulty", "pattern_id": "rate05-seed99", "rank": 2, "alpha": 2.0, "seed": 42}]), self.root)


class RetrainedAnalysisTests(unittest.TestCase):
    def test_retrained_inputs_give_exactly_the_step9b_formulas_and_side_by_side(self):
        metrics, bn, sweep, plan, aggregate, references = synthetic_inputs()
        controls = ControlsConfig(**CONTROLS_CONFIG)
        step9b = json.loads(json.dumps(analyze_controls([summary_row(item) for item in metrics], bn, sweep, plan, aggregate, references, controls)))
        rename = {"clustered": "retrained-clustered", "faulty": "retrained-faulty"}
        rows = [{**summary_row(item), "base_kind": rename.get(item["base_kind"], item["base_kind"])} for item in metrics]
        retrained_bn = [{**record, "condition_id": "retrained-clustered" if record["condition_id"] == "clustered" else record["condition_id"], "base_kind": rename.get(record["base_kind"], record["base_kind"])}
                        for record in bn]
        records = [{**{key: entry[key] for key in ("pattern_id", "probability", "seed", "repeat")}, "models": {"retrained": entry["measurements"]}} for entry in sweep]
        models = [{"probability": rate["probability"], "num_patterns": rate["num_patterns"], "realized_fault_rate": rate["realized_fault_rate"],
                   "splits": {split: {"accuracy": rate["splits"][split]["accuracy"]} for split in SPLITS}} for rate in aggregate["rates"]]
        inputs = {"rows": rows, "bn": retrained_bn, "sweep_records": records, "sweep_aggregate": {"models": {"retrained": models, "step4": models}},
                  "references": {"clean": {split: references[split]["clean"] for split in SPLITS}}, "step9b": step9b, "accepted": []}
        config = RetrainedControlsConfig(**{key: CONTROLS_CONFIG[key] for key in ("fault_probabilities", "control_seeds", "faulty_seeds", "rank", "alpha")})
        analysis = json.loads(json.dumps(build_analysis(inputs, config)))
        self.assertEqual(analysis["splits"], step9b["splits"])
        table = {(row["split"], row["baseline"], row["fault_probability"]): row for row in analysis["side_by_side_with_step9b"]}
        self.assertEqual(len(table), 2 * 2 * 2)
        for baseline in ("step4", "retrained"):
            row = table["test", baseline, .1]
            self.assertAlmostEqual(row["lora_mean"], .865)
            self.assertAlmostEqual(row["no_repair_mean"], .65)
            self.assertAlmostEqual(row["bn_recalibration_mean"], .70)
            self.assertAlmostEqual(row["fault_specific_gain_pp_mean"], -2.5)
            self.assertAlmostEqual(table["test", baseline, 0.0]["lora_mean"], .89)
            self.assertIsNone(table["test", baseline, 0.0]["fault_specific_gain_pp_mean"])


class DecisionFileTests(unittest.TestCase):
    def test_every_accepted_change_names_the_current_file(self):
        # An edit to a listed file without updating its entry would make every check that reads the record stop.
        accepted = load_accepted_changes(PROJECT_ROOT)
        self.assertTrue(accepted)
        for entry in accepted:
            with self.subTest(path=entry["path"], old=entry["old_sha256"][:12]):
                self.assertEqual(sha256_file(PROJECT_ROOT / entry["path"]), entry["new_sha256"])


if __name__ == "__main__":
    unittest.main()
