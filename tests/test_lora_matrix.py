"""Offline checks for base selection, the recovery matrix runner and adapter transfer."""

import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.evaluation.classification import evaluate_classifier
from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.memory.clustering import cluster_model, decode_state
from fault_lora.memory.faults import apply_fault_pattern, generate_fault_pattern, summarize_fault_pattern
from fault_lora.models.lora import adapter_state_dict, base_state_dict
from fault_lora.training import recovery
from fault_lora.training.recovery import RecoveryConfig, build_base_state, evaluate_adapter_on_base, load_adapter_on_base, load_recovered_model, run_recovery
from fault_lora.training.recovery_matrix import SUMMARY_FIELDS, MatrixConfig, matrix_plan, run_matrix, verify_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {"rate01-seed52": 52, "rate01-seed53": 53}
SHARED = {"source_sweep_results": "results/tiny-sweep", "epochs": 2, "batch_size": 10, "evaluation_batch_size": 10, "device": "cpu", "num_workers": 0}


class TinyCIFAR(Dataset):
    classes = [str(i) for i in range(10)]

    def __init__(self, root, train, transform, download):
        self.train = train

    def __len__(self):
        return 50000 if self.train else 10000

    def __getitem__(self, index):
        return torch.full((3, 32, 32), (index % 10) / 10), index % 10


class TinyProject:
    """Synthetic clean checkpoint, encoding and saved patterns inside a temporary root."""

    def __init__(self, root):
        self.root = root
        torch.manual_seed(0)
        self.template = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))
        self.encoding, _ = cluster_model(self.template, 4, threads=1)
        self.patterns = {name: generate_fault_pattern(self.encoding, .2, seed=seed) for name, seed in PATTERNS.items()}
        (root / "src/fault_lora").mkdir(parents=True)
        (root / "scripts").mkdir()
        (root / "results/tiny-sweep").mkdir(parents=True)
        for filename in ("scripts/run_lora_recovery.py", "scripts/run_lora_matrix.py", "requirements.txt", "requirements-macos-arm64.lock.txt"):
            (root / filename).write_text("synthetic test input\n")
        plan = [{"pattern_id": "rate00-zero", "probability": 0.0, "seed": 42, "repeat": 0}]
        plan += [{"pattern_id": name, "probability": .2, "seed": seed, "repeat": i} for i, (name, seed) in enumerate(PATTERNS.items())]
        (root / "results/tiny-sweep/plan.json").write_text(json.dumps(plan))
        self.paths = {"clean": root / "clean.pt", "encoding": root / "encoding.pt", **{name: root / f"{name}.pt" for name in PATTERNS}}
        torch.save({"model_state_dict": self.template.state_dict()}, self.paths["clean"])
        torch.save(self.encoding, self.paths["encoding"])
        for name, pattern in self.patterns.items():
            torch.save(pattern, self.paths[name])
        self.hashes = {name: sha256_file(path) for name, path in self.paths.items()}
        self.split = {"train": list(range(20)), "validation": list(range(20, 30)), "test": {"indices": list(range(20))}}
        self.metadata = {"num_classes": 10, "classes": TinyCIFAR.classes, "preprocessing": {"input_size": [3, 32, 32]}, "split_sha256": "synthetic-test-only"}
        dataset = TinyCIFAR("unused", True, None, False)
        self.subsets = {"validation": Subset(dataset, self.split["validation"]), "test": Subset(dataset, self.split["test"]["indices"])}
        self.references = {name: {"clean": self.measure(self.template.state_dict(), name), "clustered_no_faults": self.measure(decode_state(self.encoding), name)} for name in self.subsets}

    def measure(self, state, split):
        model = copy.deepcopy(self.template)
        model.load_state_dict(state)
        return evaluate_classifier(model, DataLoader(self.subsets[split], batch_size=10), "cpu")

    def state(self, base_kind, pattern_id):
        if base_kind == "clean":
            return self.template.state_dict()
        return decode_state(self.encoding) if base_kind == "clustered" else apply_fault_pattern(self.encoding, self.patterns[pattern_id])

    def prepare(self, root, config):
        """Stand-in for prepare_sources with the same return layout per base kind."""
        names = ["clean", "encoding"]
        if config.base_kind == "faulty":
            counts = summarize_fault_pattern(self.encoding, self.patterns[config.pattern_id])
            record = {"pattern_id": config.pattern_id, "seed": PATTERNS[config.pattern_id], "probability": .2, "counts": counts}
            names.append(config.pattern_id)
        else:
            zero = config.base_kind == "clustered"
            record = {"pattern_id": None, "seed": None, "probability": 0.0 if zero else None,
                      "counts": {"actual_corrupted_cells": 0 if zero else None, "realized_fault_rate": 0.0 if zero else None}}
        record["measurements"] = {name: self.measure(self.state(config.base_kind, config.pattern_id), name) for name in self.subsets}
        keys = {name: "selected_pattern" if name in PATTERNS else name for name in names}
        return (self.encoding, self.metadata, self.split, record, self.references,
                {keys[name]: self.paths[name] for name in names}, {keys[name]: self.hashes[name] for name in names})

    def patched(self):
        patches = [patch("fault_lora.training.recovery.prepare_sources", side_effect=self.prepare), patch("fault_lora.training.recovery.CIFAR10", TinyCIFAR),
                   patch("fault_lora.training.recovery.build_resnet18", side_effect=lambda *args, **kwargs: copy.deepcopy(self.template))]
        return patches


class MatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.project = TinyProject(self.root)
        for item in self.project.patched():
            item.start()
            self.addCleanup(item.stop)

    def test_pattern_is_required_exactly_for_a_faulty_base(self):
        RecoveryConfig().validate()
        for kind in ("clean", "clustered"):
            RecoveryConfig(base_kind=kind, pattern_id=None).validate()
            with self.assertRaises(ValueError):
                RecoveryConfig(base_kind=kind).validate()
            with self.assertRaises(ValueError):
                RecoveryConfig(base_kind=kind, pattern_id="rate03-seed52").validate()
        for kwargs in ({"pattern_id": None}, {"base_kind": "faulty", "pattern_id": None}, {"base_kind": "other"}, {"pattern_id": 3}, {"pattern_id": "../x"},
                       {"clean_checkpoint_sha256": "7a9d200c"}, {"clean_checkpoint_sha256": 7}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RecoveryConfig(**kwargs).validate()
        RecoveryConfig(base_kind="clean", pattern_id=None, clean_checkpoint_sha256="a" * 64).validate()

    def test_step8_configuration_still_loads_as_the_faulty_path(self):
        saved = json.loads((PROJECT_ROOT / "configs/resnet18_cifar10_lora.json").read_text())
        self.assertNotIn("base_kind", saved)
        config = RecoveryConfig(**saved)
        config.validate()
        self.assertEqual((config.base_kind, config.pattern_id, config.clean_checkpoint_sha256), ("faulty", "rate03-seed52", None))
        self.assertEqual({name: value for name, value in asdict(config).items() if name in saved}, saved)
        self.assertEqual(set(asdict(config)) - set(saved), {"base_kind", "clean_checkpoint_sha256"})

    def test_each_base_kind_builds_its_own_state_without_touching_sources(self):
        project = self.project
        states = {}
        for kind, pattern_id in (("clean", None), ("clustered", None), ("faulty", "rate01-seed52"), ("faulty", "rate01-seed53")):
            config = RecoveryConfig(base_kind=kind, pattern_id=pattern_id)
            _, _, _, _, _, paths, _ = project.prepare(self.root, config)
            states[kind, pattern_id] = state = build_base_state(config, project.encoding, paths)
            expected = project.state(kind, pattern_id)
            self.assertEqual(set(state), set(expected))
            self.assertTrue(all(torch.equal(state[name], value) and state[name].dtype == value.dtype for name, value in expected.items()))
        clustered = set(project.encoding["clustered_tensors"])
        differs = lambda a, b: {name for name in states[a] if not torch.equal(states[a][name], states[b][name])}
        for pair in ((("clean", None), ("clustered", None)), (("clustered", None), ("faulty", "rate01-seed52")), (("faulty", "rate01-seed52"), ("faulty", "rate01-seed53"))):
            changed = differs(*pair)
            self.assertTrue(changed and changed <= clustered, pair)
        # Mutating a returned state must not reach the saved sources or a second build.
        states["clean", None]["0.weight"].zero_()
        self.assertFalse(torch.equal(build_base_state(RecoveryConfig(base_kind="clean", pattern_id=None), project.encoding, project.paths)["0.weight"], states["clean", None]["0.weight"]))
        self.assertTrue(all(sha256_file(path) == project.hashes[name] for name, path in project.paths.items()))
        other = copy.deepcopy(project.template.state_dict())
        other["5.bias"] += 1
        torch.save({"model_state_dict": other}, self.root / "other.pt")
        with self.assertRaises(AssertionError):
            build_base_state(RecoveryConfig(base_kind="clean", pattern_id=None), project.encoding, {"clean": self.root / "other.pt"})

    def test_invalid_matrix_configurations_rejected(self):
        run = {"base_kind": "clean", "pattern_id": None, "rank": 2, "alpha": 2.0, "seed": 42}
        MatrixConfig(runs=[run]).validate()
        for runs in ([], [{**run, "alpha": 2.0}, {**run, "alpha": 4.0}], [{name: run[name] for name in run if name != "seed"}], [{**run, "epochs": 1}],
                     [{**run, "base_kind": "faulty"}], [{**run, "pattern_id": "rate01-seed52"}], [{**run, "rank": 0}], [{**run, "base_kind": "merged"}], [list(run.values())]):
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                MatrixConfig(runs=runs).validate()
        with self.assertRaises(ValueError):
            MatrixConfig(matrix_id="../oops", runs=[run]).validate()
        with self.assertRaises(ValueError):
            MatrixConfig(matrix_id="resnet18-cifar10-lora-clean-none-r2-train42", runs=[run]).validate()

    def test_two_run_matrix_saves_independent_runs_summary_and_verification(self):
        root = self.root
        config = MatrixConfig(matrix_id="tiny-matrix", run_id_prefix="tiny-lora", **SHARED, runs=[
            {"base_kind": "clean", "pattern_id": None, "rank": 2, "alpha": 2.0, "seed": 42},
            {"base_kind": "faulty", "pattern_id": "rate01-seed52", "rank": 1, "alpha": 2.0, "seed": 43}])
        run_ids = ["tiny-lora-clean-none-r2-train42", "tiny-lora-faulty-rate01-seed52-r1-train43"]
        self.assertEqual([entry["run_id"] for entry in matrix_plan(config)], run_ids)
        rows = run_matrix(config, root)
        results = root / "results/tiny-matrix"
        self.assertEqual(json.loads((results / "status.json").read_text())["status"], "complete")
        self.assertEqual(json.loads((results / "status.json").read_text())["completed_runs"], 2)
        self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
        self.assertEqual(json.loads((results / "summary.json").read_text()), rows)
        with (results / "summary.csv").open(newline="") as stream:
            table = list(csv.DictReader(stream))
        self.assertEqual([row["run_id"] for row in table], run_ids)
        self.assertEqual(list(table[0]), SUMMARY_FIELDS)
        self.assertEqual((table[0]["pattern_id"], table[0]["fault_probability"], table[1]["fault_probability"]), ("", "", "0.2"))
        for row, run, kind in zip(rows, config.runs, ("clean", "faulty")):
            directory = root / "results" / row["run_id"]
            metrics = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(json.loads((directory / "status.json").read_text())["status"], "complete")
            self.assertEqual((metrics["base_kind"], metrics["config"]["rank"], metrics["config"]["seed"]), (kind, run["rank"], run["seed"]))
            self.assertEqual(row["test_accuracy_before"], self.project.measure(self.project.state(kind, run["pattern_id"]), "test")["accuracy"])
            self.assertEqual((row["test_accuracy_after"], row["validation_cross_entropy_loss_after"]), (metrics["after"]["test"]["accuracy"], metrics["after"]["validation"]["cross_entropy_loss"]))
            self.assertEqual(row["trainable_adapter_parameters"], sum(value.numel() for value in torch.load(root / metrics["best_checkpoint"], weights_only=True)["adapter_state_dict"].values()))
            self.assertEqual((directory / "fault_counts.json").exists(), kind == "faulty")
            restored, _ = load_recovered_model(root / "checkpoints" / row["run_id"] / "best.pt", root)
            expected = self.project.state(kind, run["pattern_id"])
            self.assertTrue(all(torch.equal(value, expected[name]) for name, value in base_state_dict(restored).items()))
        self.assertEqual((rows[0]["realized_fault_rate"], rows[0]["actual_corrupted_cells"]), (None, None))
        self.assertGreater(rows[1]["realized_fault_rate"], 0)
        self.assertTrue(all(sha256_file(path) == self.project.hashes[name] for name, path in self.project.paths.items()))
        # Existing directories are never reused, even when only one declared run exists.
        with self.assertRaises(FileExistsError):
            run_matrix(config, root)
        other = MatrixConfig(matrix_id="tiny-matrix-2", run_id_prefix="tiny-lora", **SHARED, runs=[{"base_kind": "clustered", "pattern_id": None, "rank": 2, "alpha": 2.0, "seed": 42}, config.runs[0]])
        with self.assertRaises(FileExistsError):
            run_matrix(other, root)
        self.assertFalse((root / "results/tiny-matrix-2").exists() or (root / "results/tiny-lora-clustered-none-r2-train42").exists())
        tampered = json.loads((results / "summary.json").read_text())
        tampered[1]["test_accuracy_after"] += .1
        (results / "summary.json").write_text(json.dumps(tampered))
        with self.assertRaises(AssertionError):
            verify_matrix(root, config)

    def test_matrix_rejects_unknown_patterns_first_and_stops_at_the_first_failure(self):
        root = self.root
        runs = [{"base_kind": "clustered", "pattern_id": None, "rank": 2, "alpha": 2.0, "seed": 42}, {"base_kind": "faulty", "pattern_id": "rate09-seed99", "rank": 2, "alpha": 2.0, "seed": 42}]
        for pattern_id in ("rate09-seed99", "rate00-zero"):
            with self.assertRaises(AssertionError):
                run_matrix(MatrixConfig(matrix_id="tiny-unknown", run_id_prefix="tiny-lora", **SHARED, runs=[runs[0], {**runs[1], "pattern_id": pattern_id}]), root)
        self.assertEqual(sorted(path.name for path in (root / "results").iterdir()), ["tiny-sweep"])
        self.assertFalse((root / "checkpoints").exists())
        config = MatrixConfig(matrix_id="tiny-failing", run_id_prefix="tiny-lora", **SHARED, runs=[runs[0], {**runs[1], "pattern_id": "rate01-seed53"}, {**runs[0], "seed": 44}])
        real, calls = recovery.run_recovery, []

        def fail_second(run, project_root):
            calls.append(run.run_id)
            if len(calls) == 2:
                raise RuntimeError("synthetic failure")
            return real(run, project_root)

        with patch("fault_lora.training.recovery.run_recovery", side_effect=fail_second), self.assertRaises(RuntimeError):
            run_matrix(config, root)
        status = json.loads((root / "results/tiny-failing/status.json").read_text())
        self.assertEqual((status["status"], status["completed_runs"], status["planned_runs"], status["failed_run_id"]), ("failed", 1, 3, "tiny-lora-faulty-rate01-seed53-r2-train42"))
        self.assertEqual(len(calls), 2)
        self.assertEqual([row["run_id"] for row in json.loads((root / "results/tiny-failing/summary.json").read_text())], ["tiny-lora-clustered-none-r2-train42"])
        self.assertFalse((root / "results/tiny-failing/verification.json").exists())

    def test_adapter_transfer_is_exact_on_its_own_base_and_runs_on_other_bases(self):
        root = self.root
        config = RecoveryConfig(run_id="tiny-clustered", base_kind="clustered", pattern_id=None, rank=2, alpha=2, **SHARED)
        result = run_recovery(config, root)
        # last.pt always holds trained (nonzero-output) adapters, unlike a possible epoch-zero selection.
        checkpoint = root / "checkpoints/tiny-clustered/last.pt"
        history = json.loads((root / "results/tiny-clustered/history.json").read_text())
        before = sorted(path for path in root.rglob("*") if path.is_file())
        probe = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(1))
        original, payload = load_recovered_model(checkpoint, root)
        self.assertTrue(all(value.abs().sum() > 0 for name, value in payload["adapter_state_dict"].items() if ".lora_B." in name))
        same, _, _ = load_adapter_on_base(checkpoint, "clustered", None, root)
        with torch.no_grad():
            expected = original(probe)
            self.assertTrue(torch.equal(expected, same(probe)))
        measured = evaluate_adapter_on_base(checkpoint, "clustered", None, root)
        self.assertTrue(measured["same_base"])
        self.assertEqual(measured["with_adapter"]["validation"], history[-1]["validation"])
        self.assertEqual(measured["without_adapter"], result["before"])
        self.assertEqual(measured["base_num_correct_difference_from_step7"], {"validation": 0, "test": 0})
        self.assertEqual(evaluate_adapter_on_base(root / result["best_checkpoint"], "clustered", None, root)["with_adapter"], result["after"])
        for kind, pattern_id in (("clean", None), ("faulty", "rate01-seed52"), ("faulty", "rate01-seed53")):
            with self.subTest(kind=kind, pattern_id=pattern_id):
                moved, _, _ = load_adapter_on_base(checkpoint, kind, pattern_id, root)
                target = self.project.state(kind, pattern_id)
                self.assertTrue(all(torch.equal(value, target[name]) for name, value in base_state_dict(moved).items()))
                self.assertTrue(all(torch.equal(value, payload["adapter_state_dict"][name]) for name, value in adapter_state_dict(moved).items()))
                with torch.no_grad():
                    self.assertFalse(torch.equal(expected, moved(probe)))
                measured = evaluate_adapter_on_base(checkpoint, kind, pattern_id, root)
                self.assertFalse(measured["same_base"])
                self.assertEqual(measured["adapter_base"], {"base_kind": "clustered", "pattern_id": None})
                self.assertEqual((measured["evaluation_base"]["base_kind"], measured["evaluation_base"]["pattern_id"]), (kind, pattern_id))
                self.assertEqual(measured["without_adapter"], {name: self.project.measure(target, name) for name in ("validation", "test")})
                self.assertEqual({name: value["num_examples"] for name, value in measured["with_adapter"].items()}, {"validation": 10, "test": 20})
        for kind, pattern_id in (("faulty", None), ("clean", "rate01-seed52"), ("merged", None)):
            with self.assertRaises(ValueError):
                evaluate_adapter_on_base(checkpoint, kind, pattern_id, root)
        # A checkpoint for another layout or storage must not be loaded onto any base.
        for key, value in (("adapter_layout", payload["adapter_layout"][:1]), ("encoding_fingerprint", "0" * 64), ("architecture", "resnet34")):
            torch.save({**payload, key: value}, root / "bad.pt")
            with self.subTest(key=key), self.assertRaises(AssertionError):
                load_adapter_on_base(root / "bad.pt", "clean", None, root)
        (root / "bad.pt").unlink()
        self.assertEqual(sorted(path for path in root.rglob("*") if path.is_file()), before)
        self.assertTrue(all(sha256_file(path) == self.project.hashes[name] for name, path in self.project.paths.items()))


if __name__ == "__main__":
    unittest.main()
