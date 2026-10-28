"""Offline checks for Step 12a: embedding LoRA adapters and TinyBERT codebook-only retraining."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_tinybert_baseline as fixture
from fault_lora.evaluation import tinybert_baseline
from fault_lora.evaluation.tinybert_baseline import run_tinybert_baseline
from fault_lora.evaluation.tinybert_clean_eval import CleanEvalConfig, run_clean_eval
from fault_lora.evaluation.tinybert_clustered import TinyBertClusteredConfig, run_tinybert_clustered
from fault_lora.evaluation.tinybert_fault_checks import TinyBertFaultCheckConfig, run_tinybert_fault_checks
from fault_lora.evaluation.tinybert_fault_sweep import TinyBertSweepConfig, run_tinybert_fault_sweep
from fault_lora.evaluation.tinybert_sources import synthetic_token_batch
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.selected_clustering import cluster_named_tensors
from fault_lora.models.lora import adapter_state_dict, base_state_dict, load_adapter_state
from fault_lora.models.tinybert_embedding_lora import EmbeddingLoRALayer, attach_linear_embedding_lora
from fault_lora.models.tinybert_lora import attach_linear_lora
from fault_lora.training import tinybert_codebook_finetune as finetune
from fault_lora.training.codebook_finetune import build_encoding, codebook_order, effective_state, verify_same_cells
from fault_lora.training.tinybert_codebook_finetune import (CodebookEmbedding, TinyBertCodebookConfig, attach_tinybert_codebooks, run_tinybert_codebook_finetune,
                                                            set_tinybert_trainable)

EMBEDDINGS = ["bert.embeddings.word_embeddings", "bert.embeddings.position_embeddings", "bert.embeddings.token_type_embeddings"]
CODEBOOK_ONLY = {"variant_id": "codebook", "label": "codebooks only", "condition": "linear-only", "train_layernorm": False, "train_biases": False}
WITH_LN = {"variant_id": "codebook-lnbias", "label": "codebooks + LayerNorm + biases", "condition": "linear-only", "train_layernorm": True, "train_biases": True}


def linear_and_embedding_encoding(model):
    names = [f"{name}.weight" for name, module in model.named_modules() if isinstance(module, (nn.Linear, nn.Embedding))]
    return cluster_named_tensors(model, names, 4, threads=1)[0]


class EmbeddingLoRATests(unittest.TestCase):
    def test_zero_output_low_rank_delta_and_unchanged_linear_initialization(self):
        model = fixture.tiny_model()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        probe = synthetic_token_batch(model.config.to_dict(), length=8)
        with torch.no_grad():
            before = model(**probe).logits
        torch.manual_seed(0)
        layout = attach_linear_embedding_lora(model, 4, 4.0)
        reference = fixture.tiny_model()
        torch.manual_seed(0)
        linear = attach_linear_lora(reference, 4, 4.0)
        # Linear adapters are exactly attach_linear_lora's, including the seeded draws; embeddings follow.
        self.assertEqual(layout[:len(linear)], linear)
        self.assertEqual([item["name"] for item in layout[len(linear):]], EMBEDDINGS)
        for item in linear:
            self.assertTrue(torch.equal(model.get_submodule(item["name"]).lora_A.weight, reference.get_submodule(item["name"]).lora_A.weight))
        token_type = layout[-1]
        self.assertEqual((token_type["effective_rank_bound"], token_type["A_shape"], token_type["B_shape"]), (2, [2, 4], [16, 4]))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(**probe).logits, before))
        self.assertEqual(base_state_dict(model).keys(), state.keys())
        self.assertTrue(all(torch.equal(base_state_dict(model)[name], value) for name, value in state.items()))
        word = model.get_submodule(EMBEDDINGS[0])
        nn.init.normal_(word.lora_B.weight)
        ids = torch.tensor([[1, 2, 3, 2]])
        with torch.no_grad():
            torch.testing.assert_close(word(ids), F.embedding(ids, word.base.weight + word.delta_weight(), word.base.padding_idx))
            self.assertLessEqual(torch.linalg.matrix_rank(word.delta_weight()).item(), 4)
        model.train(False)
        F.cross_entropy(model(**probe).logits, torch.tensor([0, 1])).backward()
        trainable = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
        self.assertEqual(trainable, set(adapter_state_dict(model)))
        self.assertTrue(any(name.startswith(EMBEDDINGS[0]) for name in trainable))
        saved = adapter_state_dict(model)
        fresh = fixture.tiny_model()
        attach_linear_embedding_lora(fresh, 4, 4.0)
        load_adapter_state(fresh, saved)
        with torch.no_grad():
            self.assertTrue(torch.equal(fresh(**probe).logits, model(**probe).logits))

    def test_embeddings_can_be_excluded_and_invalid_layers_rejected(self):
        model = fixture.tiny_model()
        layout = attach_linear_embedding_lora(model, 2, 2.0, include_embeddings=False)
        self.assertTrue(all(item["kind"] == "Linear" for item in layout))
        with self.assertRaises(ValueError):
            attach_linear_embedding_lora(model, 2, 2.0)
        for args in ((nn.Linear(3, 3), 2, 2.0), (nn.Embedding(5, 3), 0, 2.0), (nn.Embedding(5, 3), 2, float("nan")), (nn.Embedding(5, 3, max_norm=1.0), 2, 2.0)):
            with self.assertRaises((TypeError, ValueError)):
                EmbeddingLoRALayer(*args)


class CodebookEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.model = fixture.tiny_model()
        self.encoding = linear_and_embedding_encoding(self.model)
        self.probe = synthetic_token_batch(self.model.config.to_dict(), length=8)

    def wrapped(self, variant):
        model = fixture.tiny_model()
        model.load_state_dict(decode_state(self.encoding))
        layers = attach_tinybert_codebooks(model, self.encoding)
        trainable = set_tinybert_trainable(model, layers, variant)
        return model.eval(), layers, trainable

    def test_wrapped_model_reproduces_the_decoded_model_and_declares_its_parameters(self):
        dense = fixture.tiny_model()
        dense.load_state_dict(decode_state(self.encoding))
        model, layers, trainable = self.wrapped({**CODEBOOK_ONLY, "condition": "linear-and-embeddings"})
        self.assertEqual(sum(isinstance(layer, CodebookEmbedding) for layer in layers.values()), 3)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(**self.probe).logits, dense.eval()(**self.probe).logits))
        self.assertEqual(trainable, sorted(f"{name}.codebook" for name in layers))
        _, _, with_ln = self.wrapped({**WITH_LN, "condition": "linear-and-embeddings"})
        extra = set(with_ln) - set(trainable)
        self.assertTrue(extra and all(name.endswith((".weight", ".bias")) for name in extra))
        self.assertTrue(any("LayerNorm" in name for name in extra) and any(name.endswith("base.bias") for name in extra))
        self.assertFalse(any(name.endswith("codebook") for name in extra))
        self.assertEqual(set(effective_state(model, layers)), set(dense.state_dict()))

    def test_embedding_codebook_gradient_is_the_summed_member_gradient(self):
        model, layers, _ = self.wrapped({**CODEBOOK_ONLY, "condition": "linear-and-embeddings"})
        dense = fixture.tiny_model()
        dense.load_state_dict(decode_state(self.encoding))
        dense.eval()
        targets = torch.tensor([0, 1])
        F.cross_entropy(model(**self.probe).logits, targets).backward()
        F.cross_entropy(dense(**self.probe).logits, targets).backward()
        for name in EMBEDDINGS:
            layer = layers[name]
            member = dense.get_submodule(name).weight.grad.reshape(-1)
            summed = torch.zeros_like(layer.codebook).scatter_add_(0, layer.indices.reshape(-1).long(), member)
            torch.testing.assert_close(layer.codebook.grad, summed, rtol=1e-5, atol=1e-6)

    def test_training_keeps_cells_and_the_saved_encoding_reconstructs_exactly(self):
        model, layers, trainable = self.wrapped({**WITH_LN, "condition": "linear-and-embeddings"})
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            F.cross_entropy(model(**self.probe).logits, torch.tensor([0, 1])).backward()
            optimizer.step()
        for name, layer in layers.items():
            self.assertTrue(torch.equal(layer.indices, self.encoding["clustered_tensors"][f"{name}.weight"]["indices"]))
        codebooks = {f"{name}.weight": layer.codebook.detach().clone() for name, layer in layers.items()}
        state = effective_state(model, layers)
        untouched = {name: value for name, value in state.items() if name not in codebooks}
        self.assertTrue(all(report["strictly_increasing"] for report in codebook_order(codebooks).values()))
        encoding = build_encoding(self.encoding, codebooks, untouched, {})
        self.assertEqual(verify_same_cells(encoding, self.encoding), len(layers))
        decoded = decode_state(encoding)
        self.assertTrue(all(torch.equal(decoded[name], value) for name, value in state.items()))
        broken = dict(codebooks)
        broken[EMBEDDINGS[0] + ".weight"] = broken[EMBEDDINGS[0] + ".weight"].flip(0)
        with self.assertRaisesRegex(AssertionError, "strict order"):
            build_encoding(self.encoding, broken, untouched, {})


class CodebookPipelineTests(unittest.TestCase):
    def build_chain(self, root):
        step10a = fixture.TinyBertBaselineRunTests.build(None, root)
        for name in ("run_tinybert_clean_eval.py", "run_tinybert_clustered.py", "check_tinybert_faults.py", "run_tinybert_fault_sweep.py", "run_tinybert_codebook_finetune.py"):
            (root / "scripts" / name).write_text("placeholder\n")
        with mock.patch.object(tinybert_baseline, "PROBES", {}):
            run_tinybert_baseline(step10a, root)
        clean = run_clean_eval(CleanEvalConfig(run_id="tiny-clean", source_results="results/tiny-run", expected_original_test_examples=25), root)
        run_tinybert_clustered(TinyBertClusteredConfig(run_id="tiny-clustered", clean_checkpoint_sha256=clean["checkpoint_sha256"], source_clean_eval_results="results/tiny-clean", clustering_threads=1), root)
        run_tinybert_fault_checks(TinyBertFaultCheckConfig(run_id="tiny-audit", source_clustered_results="results/tiny-clustered", reference_probability=.05, cpu_threads=1), root)
        run_tinybert_fault_sweep(TinyBertSweepConfig(run_id="tiny-sweep", probabilities=[0, .3], patterns_per_rate=2, cpu_threads=1, source_audit_results="results/tiny-audit"), root)

    def test_offline_retraining_saves_verified_encodings_and_stops_on_order_violations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.build_chain(root)
            options = dict(source_sweeps={"linear-only": "results/tiny-sweep"}, epochs=1, evaluations_per_epoch=2, batch_size=16, learning_rate=1e-5, expected_training_examples=80,
                           device="cpu", cpu_threads=1)
            config = TinyBertCodebookConfig(run_id="tiny-codebook", variants=[CODEBOOK_ONLY, WITH_LN], **options)
            result = run_tinybert_codebook_finetune(config, root)
            results = root / "results/tiny-codebook"
            self.assertEqual(result["status"], "complete")
            self.assertEqual(json.loads((results / "verification.json").read_text())["status"], "passed")
            self.assertEqual(set(result["evaluation"]), {"clean", "clustered-linear-only", "codebook", "codebook-lnbias"})
            sweep_zero = json.loads((root / "results/tiny-sweep/patterns/rate00-zero.json").read_text())["measurements"]
            self.assertEqual(result["evaluation"]["clustered-linear-only"], sweep_zero)
            history = json.loads((results / "history_codebook.json").read_text())
            self.assertEqual([item["epoch"] for item in history], [0.0, .5, 1.0])
            trainable = json.loads((results / "trainable.json").read_text())
            self.assertLess(trainable["codebook"]["trainable_parameters"], trainable["codebook-lnbias"]["trainable_parameters"])
            source = torch.load(root / "checkpoints/tiny-clustered/encoding.pt", weights_only=True)
            for key in ("codebook", "codebook-lnbias"):
                verify_same_cells(torch.load(root / result["encodings"][key]["encoding"], weights_only=True), source)
            self.assertEqual(result["encodings"]["codebook"]["untouched_state_changed"], [])
            with self.assertRaises(FileExistsError):
                run_tinybert_codebook_finetune(config, root)
            real = finetune.train_variant

            def unsorted(*args, **kwargs):
                trained = real(*args, **kwargs)
                name = sorted(trained["selected"]["codebooks"])[0]
                trained["selected"]["codebooks"][name] = trained["selected"]["codebooks"][name].flip(0)
                return trained

            with mock.patch.object(finetune, "train_variant", side_effect=unsorted):
                stopped = run_tinybert_codebook_finetune(TinyBertCodebookConfig(run_id="tiny-unsorted", variants=[CODEBOOK_ONLY], **options), root)
            self.assertEqual(stopped["status"], "stopped_codebook_order_violation")
            self.assertFalse(list((root / "checkpoints/tiny-unsorted").rglob("encoding.pt")))
            self.assertFalse((root / "results/tiny-unsorted/paired_evaluation.json").exists())

    def test_invalid_configuration_rejected(self):
        TinyBertCodebookConfig().validate()
        for kwargs in ({"variants": []}, {"variants": [CODEBOOK_ONLY, CODEBOOK_ONLY]}, {"variants": [{**CODEBOOK_ONLY, "condition": "other"}]}, {"variants": [{**CODEBOOK_ONLY, "train_biases": 1}]},
                       {"learning_rate": 0}, {"model_mode": "train"}, {"schedule": "cosine-per-epoch"}, {"run_id": "../x"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TinyBertCodebookConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
