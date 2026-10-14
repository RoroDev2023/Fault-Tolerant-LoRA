"""Offline checks for the Step 10c harsher TinyBERT fault conditions and the Part 7 rate rule.

Uses the tiny randomly initialized BERT and local cache from `test_tinybert_baseline`.
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
from fault_lora.evaluation.fault_sweep import SweepConfig, sweep_plan
from fault_lora.evaluation.tinybert_baseline import run_tinybert_baseline
from fault_lora.evaluation.tinybert_clean_eval import CleanEvalConfig, run_clean_eval
from fault_lora.evaluation.tinybert_clustered import TinyBertClusteredConfig, run_tinybert_clustered
from fault_lora.evaluation.tinybert_condition_audit import ConditionAuditConfig, run_condition_audit
from fault_lora.evaluation.tinybert_condition_sweep import ConditionSweepConfig, aggregate_condition, condition_plan, run_condition_sweep
from fault_lora.evaluation.tinybert_embedding_clustered import EmbeddingClusteredConfig, condition_b_names
from fault_lora.evaluation.tinybert_embedding_clustered_reuse import run_embedding_clustered_reusing_linear
from fault_lora.evaluation.tinybert_fault_checks import TinyBertFaultCheckConfig, run_tinybert_fault_checks
from fault_lora.evaluation.tinybert_fault_sweep import TinyBertSweepConfig, run_tinybert_fault_sweep
from fault_lora.evaluation.tinybert_lora_proposal import ProposalConfig, apply_rule, run_proposal
from fault_lora.evaluation.tinybert_sources import linear_weight_names
from fault_lora.memory.clustering import cluster_model, decode_state
from fault_lora.memory.faults import apply_fault_pattern, generate_fault_pattern
from fault_lora.memory.selected_clustering import cluster_named_tensors

SCRIPTS = ("run_tinybert_clean_eval.py", "run_tinybert_clustered.py", "check_tinybert_faults.py", "run_tinybert_fault_sweep.py",
           "run_tinybert_embedding_clustered.py", "run_tinybert_embedding_clustered_reuse.py", "check_tinybert_condition_faults.py", "run_tinybert_condition_sweep.py")


def equal_encodings(first, second, names):
    return all(torch.equal(first["clustered_tensors"][name][key], second["clustered_tensors"][name][key]) for name in names for key in ("codebook", "indices", "index_to_level", "level_to_index"))


class SelectedClusteringTests(unittest.TestCase):
    def test_wrapper_reproduces_cluster_model_and_extends_it_to_embeddings(self):
        model = fixture.tiny_model()
        reference, _ = cluster_model(model, 16, seed=42, threads=1)
        linear = linear_weight_names(model)
        same, _ = cluster_named_tensors(model, linear, 16, seed=42, threads=1)
        self.assertTrue(equal_encodings(same, reference, linear))
        self.assertEqual(sorted(same["untouched_state"]), sorted(reference["untouched_state"]))
        names = condition_b_names(model)
        self.assertEqual(names[:len(linear)], linear)
        self.assertEqual([name.split(".")[2] for name in names[len(linear):]], ["word_embeddings", "position_embeddings", "token_type_embeddings"])
        extended, _ = cluster_named_tensors(model, names, 16, seed=42, threads=1)
        self.assertTrue(equal_encodings(extended, reference, linear))
        self.assertFalse(set(names) & set(extended["untouched_state"]))
        self.assertTrue(all("LayerNorm" in name or name.endswith(".bias") for name in extended["untouched_state"]))
        # The unchanged fault simulator applies to embedding cells and leaves reliable state alone.
        state = apply_fault_pattern(extended, generate_fault_pattern(extended, 1, seed=3))
        self.assertFalse(torch.equal(state["bert.embeddings.word_embeddings.weight"], decode_state(extended)["bert.embeddings.word_embeddings.weight"]))
        for name, value in extended["untouched_state"].items():
            self.assertTrue(torch.equal(state[name], value))
        for invalid in ([], linear[:1] * 2, ["missing.weight"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                cluster_named_tensors(model, invalid, 16)


class ConditionPlanTests(unittest.TestCase):
    def test_plans_follow_step7_allocation(self):
        b = ConditionSweepConfig(run_id="b", condition="linear-and-embeddings", probabilities=[0, .001, .5], first_seed=1000, first_rate_label=1, previous_sweep_results=None)
        self.assertEqual(condition_plan(b), sweep_plan(SweepConfig(probabilities=[0, .001, .5], seed=1000, device="cpu")))
        a = ConditionSweepConfig(probabilities=[.2, .3, .5, .75, 1.0])
        plan = condition_plan(a)
        self.assertEqual(len(plan), 25)
        self.assertEqual((plan[0]["pattern_id"], plan[-1]["pattern_id"]), ("rate07-seed72", "rate11-seed96"))
        self.assertEqual([entry["seed"] for entry in plan], list(range(72, 97)))

    def test_invalid_condition_configurations_fail(self):
        base = ConditionSweepConfig(probabilities=[.2, 1.0])
        for change in ({"condition": "other"}, {"probabilities": [0, .2]}, {"probabilities": [.3, .2]}, {"probabilities": [1.5]}, {"previous_sweep_results": None},
                       {"condition": "linear-and-embeddings"}, {"patterns_per_rate": 1}, {"first_rate_label": 0}, {"device": "bogus"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(base, **change).validate()
        with self.assertRaises(ValueError):
            ConditionAuditConfig(reference_probability=0).validate()
        with self.assertRaises(ValueError):
            EmbeddingClusteredConfig(clean_checkpoint_sha256="x").validate()

    def test_aggregate_rejects_incomplete_records(self):
        config = ConditionSweepConfig(probabilities=[.2, .5], patterns_per_rate=2)
        plan = condition_plan(config)
        metric = {"num_examples": 4, "num_correct": 2, "accuracy": .5, "cross_entropy_loss": .7}
        loss = {"accuracy_drop_percentage_points": 1.0, "cross_entropy_increase": .1}
        records = [{**entry, "counts": {"eligible_cells": 10, "actual_corrupted_cells": 3, "realized_fault_rate": .3}, "measurements": {"selection": metric}, "fault_loss": {"selection": loss}} for entry in plan]
        aggregate = aggregate_condition(records, plan, config.probabilities, ["selection"])
        self.assertEqual(aggregate["rates"][1]["pattern_ids"], ["rate08-seed74", "rate08-seed75"])
        with self.assertRaises(AssertionError):
            aggregate_condition(records[:-1], plan, config.probabilities, ["selection"])


class ProposalRuleTests(unittest.TestCase):
    def ladder(self, drops):
        return [{"probability": probability, "selection_mean_drop_percentage_points": drop} for probability, drop in drops]

    def test_rule_selects_qualifying_rates_and_the_neighbor_below(self):
        chosen, neighbor = apply_rule(self.ladder([(.1, .5), (.2, 1.0), (.5, 3.0), (1.0, 20.0)]), 3.0)
        self.assertEqual([item["probability"] for item in chosen], [.2, .5, 1.0])
        self.assertEqual(neighbor, 1)
        self.assertEqual(chosen[0]["reason"], "next lower rate below the lowest qualifying rate")
        # Non-monotonic drops: every qualifying rate is kept; only the lowest gets a neighbor.
        chosen, _ = apply_rule(self.ladder([(.1, .2), (.2, 4.0), (.5, 1.0), (1.0, 5.0)]), 3.0)
        self.assertEqual([item["probability"] for item in chosen], [.1, .2, 1.0])
        self.assertEqual(apply_rule(self.ladder([(.1, 1.0), (.2, 2.9)]), 3.0), ([], None))
        chosen, neighbor = apply_rule(self.ladder([(.1, 3.5), (.2, 9.0)]), 3.0)
        self.assertEqual(([item["probability"] for item in chosen], neighbor), ([.1, .2], None))

    def test_proposal_reads_only_the_selection_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep = root / "results/sweep"
            sweep.mkdir(parents=True)
            rates = []
            for probability, selection_drop in ((0, 0), (.1, 1.0), (.2, 4.0)):
                split = lambda drop: {"accuracy_drop_percentage_points": {"mean": drop}}
                # The test set's huge drops must not influence the rule.
                rates.append({"probability": probability, "seeds": [1], "pattern_ids": ["x"], "splits": {"selection": split(selection_drop), "sst2-validation-872": split(40.0)}})
            (sweep / "aggregate.json").write_text(json.dumps({"splits": ["selection", "sst2-validation-872"], "rates": rates}))
            (sweep / "status.json").write_text(json.dumps({"status": "complete"}))
            (sweep / "verification.json").write_text(json.dumps({"status": "passed"}))
            proposal = run_proposal(ProposalConfig(run_id="p", conditions={"c": ["results/sweep"]}), root)
            self.assertEqual([item["probability"] for item in proposal["c"]["selected_rates"]], [.1, .2])
            saved = (root / "results/p/proposal.json").read_text()
            self.assertNotIn("sst2-validation-872", saved)
            with self.assertRaises(FileExistsError):
                run_proposal(ProposalConfig(run_id="p", conditions={"c": ["results/sweep"]}), root)


class ConditionPipelineTests(unittest.TestCase):
    def test_both_conditions_run_offline_and_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            step10a = fixture.TinyBertBaselineRunTests.build(None, root, leak=True)
            for name in SCRIPTS:
                (root / "scripts" / name).write_text("placeholder\n")
            with mock.patch.object(tinybert_baseline, "PROBES", {}):
                run_tinybert_baseline(step10a, root)
            clean = run_clean_eval(CleanEvalConfig(run_id="tiny-clean", source_results="results/tiny-run", expected_original_test_examples=25), root)
            linear_config = TinyBertClusteredConfig(run_id="tiny-clustered", clean_checkpoint_sha256=clean["checkpoint_sha256"], source_clean_eval_results="results/tiny-clean", clustering_threads=1)
            linear_metrics = run_tinybert_clustered(linear_config, root)
            run_tinybert_fault_checks(TinyBertFaultCheckConfig(run_id="tiny-audit", source_clustered_results="results/tiny-clustered", reference_probability=.05, cpu_threads=1), root)
            run_tinybert_fault_sweep(TinyBertSweepConfig(run_id="tiny-sweep", probabilities=[0, .05, .1], patterns_per_rate=2, cpu_threads=1, source_audit_results="results/tiny-audit"), root)

            a = ConditionSweepConfig(run_id="tiny-a", probabilities=[.3, 1.0], patterns_per_rate=2, first_seed=46, first_rate_label=3, cpu_threads=1,
                                     source_audit_results="results/tiny-audit", previous_sweep_results="results/tiny-sweep")
            with self.assertRaises(AssertionError):
                run_condition_sweep(dataclasses.replace(a, run_id="tiny-a-collide", first_seed=44), root)
            self.assertFalse((root / "results/tiny-a-collide").exists())
            aggregate_a = run_condition_sweep(a, root)
            results_a = root / "results/tiny-a"
            self.assertEqual(json.loads((results_a / "verification.json").read_text())["status"], "passed")
            self.assertEqual([rate["probability"] for rate in aggregate_a["rates"]], [.3, 1.0])
            self.assertEqual(aggregate_a["rates"][1]["actual_corrupted_cells"]["mean"], aggregate_a["rates"][1]["eligible_cells"])
            references_a = json.loads((results_a / "references.json").read_text())
            for name in ("sst2-validation-30", "sst2-original-test-25"):
                self.assertEqual(references_a[name]["clustered_no_faults"], linear_metrics["measurements"][name]["clustered_no_faults"])
            self.assertTrue((results_a / "accuracy_vs_fault_rate.png").stat().st_size > 0)

            b_clustered = EmbeddingClusteredConfig(run_id="tiny-emb", clean_checkpoint_sha256=clean["checkpoint_sha256"], source_clean_eval_results="results/tiny-clean",
                                                   step10b_clustered_results="results/tiny-clustered", clustering_threads=1)
            b_metrics = run_embedding_clustered_reusing_linear(b_clustered, root)
            linear_encoding = torch.load(root / "checkpoints/tiny-clustered/encoding.pt", weights_only=True)
            b_encoding = torch.load(root / "checkpoints/tiny-emb/encoding.pt", weights_only=True)
            self.assertTrue(equal_encodings(b_encoding, linear_encoding, list(linear_encoding["clustered_tensors"])))
            self.assertEqual(list(b_encoding["clustered_tensors"])[-3:], [name for name in b_encoding["clustered_tensors"] if "embeddings" in name])
            with self.assertRaises(FileExistsError):
                run_embedding_clustered_reusing_linear(b_clustered, root)
            scope = json.loads((root / "results/tiny-emb/scope.json").read_text())
            self.assertEqual((scope["num_eligible_tensors"], scope["num_embedding_tensors"]), (11, 3))
            self.assertEqual(scope["eligible_weights"], scope["linear_weights"] + scope["embedding_weights"])
            self.assertIn("additional_drop_from_embedding_clustering_percentage_points", b_metrics["measurements"]["sst2-validation-30"])
            report = run_condition_audit(ConditionAuditConfig(run_id="tiny-emb-audit", source_clustered_results="results/tiny-emb", reference_probability=.05, cpu_threads=1), root)
            self.assertEqual(report["status"], "passed")
            b = ConditionSweepConfig(run_id="tiny-b", condition="linear-and-embeddings", probabilities=[0, .1, .5], patterns_per_rate=2, first_seed=1000, first_rate_label=1,
                                     cpu_threads=1, source_audit_results="results/tiny-emb-audit", previous_sweep_results=None)
            aggregate_b = run_condition_sweep(b, root)
            self.assertEqual(json.loads((root / "results/tiny-b/verification.json").read_text())["status"], "passed")
            self.assertEqual(aggregate_b["rates"][0]["eligible_cells"], scope["eligible_weights"])
            self.assertEqual(aggregate_b["rates"][0]["splits"]["selection"]["accuracy_drop_percentage_points"]["mean"], 0)

            proposal = run_proposal(ProposalConfig(run_id="tiny-proposal", threshold_percentage_points=.01,
                                                   conditions={"linear-only": ["results/tiny-sweep", "results/tiny-a"], "linear-and-embeddings": ["results/tiny-b"]}), root)
            self.assertEqual([item["probability"] for item in proposal["linear-only"]["ladder"]], [.05, .1, .3, 1.0])
            for condition in proposal.values():
                for item in condition["selected_rates"]:
                    self.assertTrue(item["selection_mean_drop_percentage_points"] >= .01 or item["reason"].startswith("next lower"))


if __name__ == "__main__":
    unittest.main()
