"""Offline checks for independent sweep scheduling and statistical reporting."""

import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.evaluation.fault_sweep import SweepConfig, aggregate_measurements, plot_sweep, run_fault_sweep, summarize_values, sweep_plan, write_measurements_csv
from fault_lora.memory.clustering import cluster_model


def synthetic_records(config):
    records = []
    for i, entry in enumerate(sweep_plan(config)):
        accuracy = .9 if i == 0 else (.7 if entry["repeat"] == 0 else .8)
        measurements = {split: {"num_examples": 10, "num_correct": round(10 * accuracy), "accuracy": accuracy, "cross_entropy_loss": .4} for split in ("validation", "test")}
        records.append({**entry, "counts": {"eligible_cells": 100, "actual_corrupted_cells": 0 if i == 0 else 10, "realized_fault_rate": 0 if i == 0 else .1},
                        "measurements": measurements, "fault_loss": {split: {"accuracy_drop_percentage_points": 100 * (.9 - accuracy), "cross_entropy_increase": .2 if i else 0} for split in measurements}})
    return records


class TinyDataset(Dataset):
    classes = [str(i) for i in range(10)]

    def __init__(self, root, train, transform, download):
        self.train = train

    def __len__(self):
        return 50000 if self.train else 10000

    def __getitem__(self, index):
        return torch.tensor([float(index % 2), 1.]), index % 10


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_seed_plan_has_one_zero_and_unique_nonzero_seeds_even_at_wrap(self):
        config = SweepConfig(probabilities=[0, .01, .1], patterns_per_rate=3, seed=2**32 - 1)
        plan = sweep_plan(config)
        self.assertEqual(len(plan), 7)
        self.assertEqual(sum(entry["probability"] == 0 for entry in plan), 1)
        seeds = [entry["seed"] for entry in plan[1:]]
        self.assertEqual(seeds, [2**32 - 1, 0, 1, 2, 3, 4])

    def test_mean_and_sample_sd_use_pattern_variation_not_standard_error(self):
        config = SweepConfig(probabilities=[0, .1], patterns_per_rate=2)
        result = aggregate_measurements(synthetic_records(config), config)
        zero, faulty = result["rates"]
        self.assertIsNone(zero["splits"]["test"]["accuracy"]["sample_std"])
        self.assertAlmostEqual(faulty["splits"]["test"]["accuracy"]["mean"], .75)
        self.assertAlmostEqual(faulty["splits"]["test"]["accuracy"]["sample_std"], .1 / 2**.5)
        self.assertAlmostEqual(faulty["splits"]["test"]["accuracy_drop_percentage_points"]["mean"], 15)

    def test_incomplete_duplicate_wrong_seed_and_changed_denominator_rejected(self):
        config = SweepConfig(probabilities=[0, .1], patterns_per_rate=2)
        records = synthetic_records(config)
        variants = [records[:-1], [records[0], records[1], records[1]]]
        changed = copy.deepcopy(records)
        changed[1]["seed"] += 100
        variants.append(changed)
        changed = copy.deepcopy(records)
        changed[1]["counts"]["eligible_cells"] = 200
        variants.append(changed)
        for i, invalid in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(AssertionError):
                aggregate_measurements(invalid, config)

    def test_invalid_configuration_rejected(self):
        for kwargs in ({"probabilities": [0]}, {"probabilities": [.1, .2]}, {"probabilities": [0, .1, .1]}, {"probabilities": [0, float("nan")]}, {"probabilities": [0, True]}, {"patterns_per_rate": 1}, {"run_id": "../other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SweepConfig(**kwargs).validate()

    def test_csv_and_plot_include_both_splits_and_actual_rates(self):
        config = SweepConfig(probabilities=[0, .1], patterns_per_rate=2)
        records = synthetic_records(config)
        aggregate = aggregate_measurements(records, config)
        references = {split: {"clean": records[0]["measurements"][split], "clustered_no_faults": records[0]["measurements"][split]} for split in ("validation", "test")}
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "measurements.csv"
            write_measurements_csv(csv_path, records)
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["split"] for row in rows}, {"validation", "test"})
            self.assertEqual(float(rows[-1]["realized_fault_rate"]), .1)
            for path in plot_sweep(records, aggregate, references, directory):
                self.assertGreater(path.stat().st_size, 1000)

    def test_offline_sweep_saves_complete_static_patterns_and_preserves_source(self):
        config = SweepConfig(run_id="tiny-sweep", probabilities=[0, .5], patterns_per_rate=2, device="cpu", num_workers=0)
        clean = nn.Linear(2, 10)
        encoding, _ = cluster_model(clean, 4, threads=1)
        metadata = {"num_classes": 10, "classes": TinyDataset.classes, "preprocessing": {"input_size": [3, 32, 32]}, "config": {"evaluation_batch_size": 2}, "split_sha256": "synthetic"}
        split = {"validation": [0, 1, 2], "test": {"indices": [0, 1, 2, 3]}}
        clean_metrics = {name: {"num_correct": 0} for name in ("validation", "test")}
        clustered_metrics = {"measurements": {name: {"clustered_no_faults": {"num_correct": 0}} for name in clean_metrics}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/fault_lora").mkdir(parents=True)
            (root / "scripts").mkdir()
            for filename in ("scripts/run_fault_sweep.py", "requirements.txt", "requirements-macos-arm64.lock.txt"):
                (root / filename).write_text("synthetic input\n")
            source = root / "source.pt"
            torch.save(encoding, source)
            digest = sha256_file(source)
            prepared = (encoding, clean, metadata, split, clean_metrics, clustered_metrics, {"encoding": source}, {"encoding": digest})
            with patch("fault_lora.evaluation.fault_sweep.prepare_sources", return_value=prepared), patch("fault_lora.evaluation.fault_sweep.CIFAR10", TinyDataset), patch("fault_lora.evaluation.fault_sweep.build_resnet18", side_effect=lambda *args, **kwargs: nn.Linear(2, 10)):
                aggregate = run_fault_sweep(config, root)
                with self.assertRaises(FileExistsError):
                    run_fault_sweep(config, root)
            self.assertEqual(aggregate["rates"][1]["num_patterns"], 2)
            self.assertEqual(len(list((root / "checkpoints/tiny-sweep/patterns").glob("*.pt"))), 3)
            self.assertEqual(sha256_file(source), digest)


if __name__ == "__main__":
    unittest.main()
