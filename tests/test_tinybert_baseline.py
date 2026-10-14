"""Offline checks for dict-batch evaluation, SST-2 splits, tokenization, and the TinyBERT baseline run.

Everything uses a tiny randomly initialized BertForSequenceClassification and a
local vocabulary; nothing is downloaded.
"""

import copy
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fault_lora.evaluation import tinybert_baseline
from fault_lora.data.sst2 import SentenceDataset, make_collate, proportional_stratified_split, sentence_overlap, truncation_statistics
from fault_lora.evaluation.sequence_classification import evaluate_sequence_classifier
from fault_lora.evaluation.tinybert_baseline import SECOND_TEST_NAME, TinyBertConfig, check_second_test, run_tinybert_baseline
from fault_lora.models.tinybert import fetch_snapshot, linear_embedding_layout, load_tinybert_checkpoint

WORDS = ["a", "dull", "lifeless", "and", "painfully", "boring", "mess", "wonderful", "moving", "very", "funny", "film", ",", ".", "good", "bad", "plot", "cast"]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
MODEL_REVISION, DATA_REVISION, SECOND_REVISION = "a" * 40, "b" * 40, "c" * 40


def tiny_model(seed=0):
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(seed)
    config = BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=24, num_labels=2)
    return BertForSequenceClassification(config).eval()


def tiny_tokenizer(directory):
    from transformers import BertTokenizerFast

    vocab = Path(directory) / "vocab.txt"
    vocab.write_text("\n".join(VOCAB) + "\n")
    return BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)


def sentences(count, seed):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=rng.integers(2, 9))) + f" {WORDS[index % len(WORDS)]} " for index in range(count)]


class SequenceEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.tokenizer = tiny_tokenizer(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def loader(self, texts, labels, batch_size, max_length=16):
        return DataLoader(SentenceDataset(texts, labels), batch_size=batch_size, collate_fn=make_collate(self.tokenizer, max_length))

    def test_dict_batches_match_a_direct_calculation_and_weight_short_batches(self):
        model = tiny_model()
        texts, labels = sentences(7, 1), [0, 1, 1, 0, 1, 0, 0]
        metrics = evaluate_sequence_classifier(model, self.loader(texts, labels, 3), "cpu")
        with torch.inference_mode():
            logits = torch.cat([model(**self.tokenizer([text], return_tensors="pt")).logits for text in texts])
        targets = torch.tensor(labels)
        self.assertEqual(metrics["num_examples"], 7)
        self.assertEqual(metrics["num_correct"], (logits.argmax(1) == targets).sum().item())
        self.assertEqual(metrics["accuracy"], metrics["num_correct"] / 7)
        self.assertAlmostEqual(metrics["cross_entropy_loss"], nn.functional.cross_entropy(logits, targets).item(), places=5)
        # Padding to the longest sentence in a batch must not change the measurements.
        regrouped = evaluate_sequence_classifier(model, self.loader(texts, labels, 7), "cpu")
        self.assertEqual(regrouped["num_correct"], metrics["num_correct"])
        self.assertAlmostEqual(regrouped["cross_entropy_loss"], metrics["cross_entropy_loss"], places=5)

    def test_evaluation_restores_modes_and_leaves_the_model_unchanged(self):
        model = tiny_model()
        model.train()
        model.classifier.eval()
        before = copy.deepcopy(model.state_dict())
        loader = self.loader(sentences(5, 2), [0, 1, 0, 1, 1], 2)
        first = evaluate_sequence_classifier(model, loader, "cpu")
        # Dropout is disabled during evaluation, so a repeat is identical.
        self.assertEqual(first, evaluate_sequence_classifier(model, loader, "cpu"))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(model.training and model.bert.training)
        self.assertFalse(model.classifier.training)

    def test_plain_tensor_outputs_and_custom_label_keys_are_supported(self):
        class Wrapper(nn.Module):
            def forward(self, values):
                return values

        logits = torch.tensor([[4, 0], [0, 4], [0, 4]], dtype=torch.float32)
        batches = [{"values": logits[:2], "target": torch.tensor([0, 1])}, {"values": logits[2:], "target": torch.tensor([0])}]
        metrics = evaluate_sequence_classifier(Wrapper(), batches, "cpu", label_key="target")
        self.assertEqual((metrics["num_examples"], metrics["num_correct"]), (3, 2))
        self.assertAlmostEqual(metrics["cross_entropy_loss"], nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).item(), places=6)

    def test_invalid_batches_fail(self):
        model = tiny_model()
        with self.assertRaises(ValueError):
            evaluate_sequence_classifier(model, [], "cpu")
        with self.assertRaises(ValueError):
            evaluate_sequence_classifier(model, self.loader(sentences(2, 3), [0, 1], 2), "cpu", label_key="target")
        # GLUE's hidden test labels are -1 and must never be scored.
        for labels in ([0, -1], [0, 2]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_sequence_classifier(model, self.loader(sentences(2, 3), labels, 2), "cpu")

    def test_collate_pads_to_the_longest_sentence_and_truncates(self):
        texts = ["good", "a very funny film .", " ".join(["plot"] * 30)]
        batch = make_collate(self.tokenizer, 8)([(text, 1) for text in texts[:2]])
        self.assertEqual(sorted(batch), ["attention_mask", "input_ids", "labels", "token_type_ids"])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 7))
        self.assertEqual(batch["attention_mask"].sum(1).tolist(), [3, 7])
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(tuple(make_collate(self.tokenizer, 8)([(texts[2], 0)])["input_ids"].shape), (1, 8))
        statistics = truncation_statistics(self.tokenizer, texts, 8)
        self.assertEqual((statistics["num_sentences"], statistics["num_truncated"], statistics["max_tokens"]), (3, 1, 32))
        self.assertAlmostEqual(statistics["fraction_truncated"], 1 / 3)
        self.assertEqual(truncation_statistics(self.tokenizer, texts, 32)["num_truncated"], 0)


class Sst2DataTests(unittest.TestCase):
    def test_selection_split_is_proportional_disjoint_complete_and_reproducible(self):
        labels = np.array([0] * 297 + [1] * 376)
        np.random.default_rng(0).shuffle(labels)
        split = proportional_stratified_split(labels, 50, 42)
        self.assertEqual(split, proportional_stratified_split(labels, 50, 42))
        self.assertNotEqual(split, proportional_stratified_split(labels, 50, 43))
        selection, train = set(split["selection"]), set(split["train"])
        self.assertEqual(len(split["selection"]), 50)
        self.assertFalse(selection & train)
        self.assertEqual(selection | train, set(range(len(labels))))
        # 50 * 297 / 673 = 22.07 and 50 * 376 / 673 = 27.93.
        np.testing.assert_array_equal(np.bincount(labels[split["selection"]]), [22, 28])
        self.assertEqual(split["selection"], sorted(split["selection"]))
        for invalid in (0, len(labels)):
            with self.subTest(size=invalid), self.assertRaises(ValueError):
                proportional_stratified_split(labels, invalid, 42)
        with self.assertRaises(ValueError):
            proportional_stratified_split([0] * 99 + [1], 10, 42)

    def test_sentence_overlap_separates_raw_whitespace_and_case_matches(self):
        references = ["what 's next ? ", "A Fine Film", "unrelated"]
        overlap = sentence_overlap(["what 's next ?", "a fine film", "unrelated", "new sentence"], references)
        self.assertEqual((overlap["exact_raw"], overlap["exact"], overlap["casefolded"]), (1, 2, 3))
        self.assertEqual(overlap["exact_matches"], ["unrelated", "what 's next ?"])

    def test_second_test_set_needs_real_labels_and_no_shared_sentences(self):
        sst2 = {"train": {"sentences": ["a dull mess ", "good plot "]}, "validation": {"sentences": ["funny film "]}, "test": {"sentences": ["bad cast "]}}
        names = ["negative", "positive"]
        test = lambda texts, labels: {"test": {"sentences": texts, "labels": labels, "label_texts": [names[label] if label >= 0 else "unlabeled" for label in labels]}}
        usable = check_second_test(test(["bad cast", "very good"], [0, 1]), sst2, names)
        self.assertTrue(usable["usable"])
        self.assertEqual(usable["overlap_with_sst2"]["test"]["exact"], 1)
        self.assertFalse(check_second_test(test(["bad cast", "very good"], [-1, -1]), sst2, names)["usable"])
        leaked = check_second_test(test(["a dull mess", "very good"], [0, 1]), sst2, names)
        self.assertFalse(leaked["usable"])
        self.assertIn("train", leaked["reason"])
        self.assertFalse(check_second_test(test(["funny film", "very bad"], [1, 0]), sst2, names)["usable"])
        swapped = test(["bad cast", "very good"], [0, 1])
        swapped["test"]["label_texts"] = ["positive", "negative"]
        self.assertFalse(check_second_test(swapped, sst2, names)["usable"])


class TinyBertModelTests(unittest.TestCase):
    def test_layout_lists_every_linear_and_embedding(self):
        model = tiny_model()
        layout = linear_embedding_layout(model)
        names = [layer["name"] for layer in layout]
        self.assertEqual(names[:3], ["bert.embeddings.word_embeddings", "bert.embeddings.position_embeddings", "bert.embeddings.token_type_embeddings"])
        self.assertEqual(names[-1], "classifier")
        self.assertEqual(sum(layer["type"] == "Embedding" for layer in layout), 3)
        # One encoder layer has query, key, value, attention output, intermediate, and output; then pooler and classifier.
        self.assertEqual(sum(layer["type"] == "Linear" for layer in layout), 8)
        self.assertEqual(layout[0]["weight_shape"], [len(VOCAB), 16])
        self.assertIsNone(layout[0]["bias_shape"])
        self.assertEqual((layout[-1]["weight_shape"], layout[-1]["bias_shape"]), ([2, 16], [2]))
        state = model.state_dict()
        for layer in layout:
            self.assertEqual(list(state[layer["weight_name"]].shape), layer["weight_shape"])
        layer_norm = sum(module.weight.numel() + module.bias.numel() for module in model.modules() if isinstance(module, nn.LayerNorm))
        self.assertEqual(sum(layer["num_weights"] + layer["num_bias"] for layer in layout) + layer_norm, sum(value.numel() for value in model.parameters()))

    def test_revisions_must_be_full_commit_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            for revision in ("main", "5aa94d5", "A" * 40):
                with self.subTest(revision=revision), self.assertRaises(ValueError):
                    fetch_snapshot("tiny/model", revision, directory, ("config.json",))

    def test_invalid_run_configuration_fails(self):
        for invalid in ({"run_id": "../other"}, {"model_revision": "main"}, {"selection_examples": 67349}, {"max_length": 0}, {"device": "bogus"},
                        {"second_test_revision": None}, {"label_names": ["negative", "negative"]}, {"seed": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TinyBertConfig(**invalid).validate()
        TinyBertConfig().validate()
        TinyBertConfig(second_test_repo=None, second_test_revision=None).validate()


class TinyBertBaselineRunTests(unittest.TestCase):
    """A complete run against a hand-built local cache laid out like the Hugging Face hub cache."""

    def build(self, root, *, leak=False, agreement=True):
        from datasets import ClassLabel, Dataset, Features, Value
        import pyarrow as pa
        import pyarrow.parquet as pq

        snapshot = root / "checkpoints/hf-cache/hub/models--tiny--bert/snapshots" / MODEL_REVISION
        snapshot.mkdir(parents=True)
        model = tiny_model()
        model.save_pretrained(snapshot)
        tiny_tokenizer(snapshot).save_pretrained(snapshot)
        (snapshot / "special_tokens_map.json").write_text(json.dumps({name + "_token": f"[{name.upper()}]" for name in ("unk", "sep", "pad", "cls", "mask")}))
        features = Features({"idx": Value("int32"), "sentence": Value("string"), "label": ClassLabel(names=["negative", "positive"])})
        tokenizer = tiny_tokenizer(snapshot)

        def predicted(texts):
            with torch.inference_mode():
                return model(**tokenizer(texts, padding="longest", return_tensors="pt")).logits.argmax(1).tolist()

        def labelled(texts, seed):
            # Labels agree with the random model on 80% of sentences (20% when `agreement` is false): a known accuracy.
            flips = np.random.default_rng(seed).permutation(len(texts)) < len(texts) // 5
            return [1 - label if flip == agreement else label for label, flip in zip(predicted(texts), flips)]

        data = root / "data/hf-cache/hub/datasets--tiny--sst2/snapshots" / DATA_REVISION / "data"
        data.mkdir(parents=True)
        train, validation, hidden = sentences(120, 10), sentences(30, 11), sentences(20, 12)
        for name, texts, labels in (("train", train, labelled(train, 20)), ("validation", validation, labelled(validation, 21))):
            Dataset.from_dict({"idx": list(range(len(texts))), "sentence": texts, "label": labels}, features=features).to_parquet(str(data / f"{name}-00000-of-00001.parquet"))
        # GLUE's test file keeps the label names in its metadata while every stored label is -1.
        metadata = pq.read_schema(data / "train-00000-of-00001.parquet").metadata
        hidden_table = pa.table({"idx": pa.array(range(len(hidden)), pa.int32()), "sentence": hidden, "label": pa.array([-1] * len(hidden), pa.int64())})
        pq.write_table(hidden_table.replace_schema_metadata(metadata), data / "test-00000-of-00001.parquet")
        second = root / "data/hf-cache/hub/datasets--tiny--original/snapshots" / SECOND_REVISION
        second.mkdir(parents=True)
        names = ["negative", "positive"]
        extra = [text.strip() for text in sentences(25, 13)] + ([train[0].strip()] if leak else [])
        rows = lambda texts, labels: "".join(json.dumps({"text": text, "label": label, "label_text": names[label]}) + "\n" for text, label in zip(texts, labels))
        (second / "train.jsonl").write_text(rows(extra[:3], [0, 1, 0]))
        (second / "dev.jsonl").write_text(rows(extra[:3], [0, 1, 0]))
        (second / "test.jsonl").write_text(rows(extra, labelled(extra, 22)))
        for path in ("src/fault_lora", "scripts"):
            (root / path).mkdir(parents=True)
        for name in ("scripts/run_tinybert_baseline.py", "requirements.txt", "requirements-macos-arm64.lock.txt", "src/fault_lora/module.py"):
            (root / name).write_text("placeholder\n")
        return TinyBertConfig(run_id="tiny-run", model_repo="tiny/bert", model_revision=MODEL_REVISION, dataset_repo="tiny/sst2", dataset_revision=DATA_REVISION,
                              second_test_repo="tiny/original", second_test_revision=SECOND_REVISION, selection_examples=40, expected_train_examples=120,
                              expected_test_examples=30, model_card_accuracy=None, max_length=8, evaluation_batch_size=16, cpu_threads=2)

    def run_offline(self, config, root):
        # A random model has no opinion on the two real probe sentences; the other label checks stay active.
        with mock.patch.object(tinybert_baseline, "PROBES", {}):
            return run_tinybert_baseline(config, root)

    def test_complete_offline_run_saves_verified_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.build(root)
            metrics = self.run_offline(config, root)
            results = root / "results/tiny-run"
            read = lambda name: json.loads((results / name).read_text())
            self.assertEqual(read("status.json")["status"], "complete")
            self.assertEqual(read("verification.json")["status"], "passed")
            self.assertEqual(sorted(metrics["measurements"]), ["selection", SECOND_TEST_NAME, "test"])
            self.assertEqual(metrics["measurements"]["selection"]["num_examples"], 40)
            self.assertEqual(metrics["measurements"]["test"]["num_correct"], 24)
            self.assertEqual(metrics["measurements"][SECOND_TEST_NAME]["num_correct"], 20)
            split = read("splits.json")
            self.assertEqual((len(split["selection"]), len(split["train"])), (40, 80))
            official_test = read("manifest.json")["splits"]["official_test_split"]
            self.assertEqual((official_test["used"], official_test["label_counts"]), (False, {"-1": 20}))
            self.assertEqual(read("truncation.json")["splits"]["train"]["num_sentences"], 80)
            self.assertGreater(read("truncation.json")["splits"]["all"]["num_truncated"], 0)
            self.assertTrue(read("label_mapping.json")["confirmed"])
            model, checkpoint = load_tinybert_checkpoint(root / metrics["checkpoint"])
            self.assertEqual(checkpoint["split_sha256"], metrics["split_sha256"])
            self.assertEqual(read("layers.json")["total_parameters"], sum(value.numel() for value in model.parameters()))
            with self.assertRaises(FileExistsError):
                self.run_offline(config, root)

    def test_shared_sentence_excludes_the_second_test_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = self.run_offline(self.build(root, leak=True), root)
            self.assertEqual(sorted(metrics["measurements"]), ["selection", "test"])
            second = json.loads((root / "results/tiny-run/second_test_set.json").read_text())
            self.assertFalse(second["usable"])
            self.assertEqual(second["overlap_with_sst2"]["train"]["exact"], 1)
            self.assertEqual(json.loads((root / "results/tiny-run/verification.json").read_text())["status"], "passed")

    def test_reversed_labels_and_missing_files_stop_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.build(root, agreement=False)
            # Selection accuracy of 20% contradicts the declared label order; no test set may be evaluated.
            with self.assertRaises(AssertionError):
                self.run_offline(config, root)
            results = root / "results/tiny-run"
            self.assertEqual(json.loads((results / "status.json").read_text())["status"], "failed")
            self.assertFalse(json.loads((results / "label_mapping.json").read_text())["confirmed"])
            self.assertFalse((results / "metrics.json").exists())
            missing = dataclasses.replace(config, run_id="tiny-missing", dataset_revision="d" * 40)
            with self.assertRaises(Exception):
                self.run_offline(missing, root)
            self.assertEqual(json.loads((root / "results/tiny-missing/status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
