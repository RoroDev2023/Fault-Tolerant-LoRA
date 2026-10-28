"""Hash-verified TinyBERT / SST-2 inputs shared by Step 10b onward.

Everything is rebuilt from the completed Step 10a run: its clean checkpoint,
the tokenizer and dataset files it downloaded, its saved selection split, and
two labelled test sets that are always reported separately:

- `sst2-validation-<n>`: the official SST-2 validation split;
- `sst2-original-test-<n>`: SetFit/sst2's test split with every sentence that
  also occurs in SST-2 train or validation removed.
"""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from fault_lora.data.sst2 import SentenceDataset, make_collate, normalize_sentence, read_setfit_sst2, read_sst2
from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.sequence_classification import evaluate_sequence_classifier
from fault_lora.evaluation.source_hashes import compare_source_hashes
from fault_lora.evaluation.tinybert_baseline import TinyBertConfig, split_sha256
from fault_lora.models.tinybert import linear_embedding_layout, load_tinybert_checkpoint

VALIDATION_TEST_PREFIX = "sst2-validation"
ORIGINAL_TEST_PREFIX = "sst2-original-test"
REMOVAL_RULE = "remove every SetFit/sst2 test sentence equal, after whitespace normalization, to any SST-2 train or validation sentence"


def filter_second_test(candidate, sst2, label_names):
    """Keep SetFit/sst2 test sentences absent from SST-2 train and validation; record each removal."""
    test = candidate["test"]
    require(set(test["labels"]) == set(range(len(label_names))), f"Second test labels are {sorted(set(test['labels']))}, not the class indices")
    require(all(label_names[label] == text for label, text in zip(test["labels"], test["label_texts"])), "Second test label indices disagree with their names")
    lookup = {}
    for split in ("train", "validation"):
        for index, sentence in enumerate(sst2[split]["sentences"]):
            lookup.setdefault(normalize_sentence(sentence), []).append((split, index))
    kept, removed = [], []
    for index, (sentence, label, text) in enumerate(zip(test["sentences"], test["labels"], test["label_texts"])):
        matches = lookup.get(normalize_sentence(sentence), [])
        if not matches:
            kept.append(index)
            continue
        removed.append({
            "setfit_test_index": index, "sentence": sentence, "label": label, "label_text": text,
            "matches": [{"split": split, "index": position, "sentence": sst2[split]["sentences"][position], "label": sst2[split]["labels"][position]} for split, position in matches],
            "reason": f"also occurs in the SST-2 {' and '.join(sorted({split for split, _ in matches}))} split (whitespace-normalized exact match)",
        })
    return {"indices": kept, "sentences": [test["sentences"][index] for index in kept], "labels": [test["labels"][index] for index in kept],
            "num_candidates": len(test["labels"]), "removed": removed, "rule": REMOVAL_RULE}


def test_set_sha256(sentences, labels):
    return hashlib.sha256(json.dumps([list(sentences), list(labels)], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def prepare_tinybert_sources(root, step10a_results, *, expected_original_test_examples=None):
    """Verify and load the Step 10a artifacts; fail on any changed file."""
    from transformers import AutoTokenizer

    root = Path(root).resolve()
    results = root / step10a_results
    read = lambda name: json.loads((results / name).read_text())
    require(read("status.json")["status"] == "complete" and read("verification.json")["status"] == "passed", "Step 10a run is incomplete or unverified")
    config = TinyBertConfig(**read("config.json"))
    config.validate()
    manifest, metrics, split = read("manifest.json"), read("metrics.json"), read("splits.json")
    require(split_sha256(split) == manifest["split_sha256"] == metrics["split_sha256"], "Saved Step 10a split hash differs")
    paths, hashes = {}, {}
    checkpoint_path = root / metrics["checkpoint"]
    paths["clean_checkpoint"], hashes["clean_checkpoint"] = checkpoint_path, sha256_file(checkpoint_path)
    require(hashes["clean_checkpoint"] == metrics["checkpoint_sha256"], "Step 10a clean checkpoint differs from its recorded hash")
    for group, item in manifest["source_artifacts"].items():
        for name, record in item["files"].items():
            path = root / record["path"]
            digest = sha256_file(path)
            require(digest == record["sha256"], f"Downloaded file changed: {group}/{name}")
            paths[f"{group}/{name}"], hashes[f"{group}/{name}"] = path, digest
    accepted = compare_source_hashes(root, read("source_sha256.json"), "Step 10a source code changed")
    for name in ("config", "manifest", "metrics", "splits", "status", "verification", "source_sha256", "label_mapping"):
        path = results / f"{name}.json"
        paths[f"step10a_{name}"], hashes[f"step10a_{name}"] = path, sha256_file(path)
    require("second_test" in manifest["source_artifacts"], "Step 10a did not record the SetFit/sst2 files")

    model, checkpoint = load_tinybert_checkpoint(checkpoint_path)
    model.requires_grad_(False)
    require(checkpoint["split_sha256"] == manifest["split_sha256"], "Checkpoint split hash differs")
    snapshot = lambda group, name: root / manifest["source_artifacts"][group]["files"][name]["path"][: -len(name)]
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot("model", "tokenizer.json")), local_files_only=True)
    sst2, label_names = read_sst2(snapshot("dataset", "data/train-00000-of-00001.parquet"), root / config.data_cache / "datasets")
    require(label_names == config.label_names == checkpoint["classes"], "Label names differ from Step 10a")
    second = filter_second_test(read_setfit_sst2(snapshot("second_test", "test.jsonl")), sst2, config.label_names)
    if expected_original_test_examples is not None:
        require(len(second["labels"]) == expected_original_test_examples, f"Filtered second test has {len(second['labels'])} examples, expected {expected_original_test_examples}")
    validation = sst2["validation"]
    require(len(validation["labels"]) == config.expected_test_examples, "Official validation size changed")
    validation_name = f"{VALIDATION_TEST_PREFIX}-{len(validation['labels'])}"
    original_name = f"{ORIGINAL_TEST_PREFIX}-{len(second['labels'])}"
    datasets = {
        "selection": ([sst2["train"]["sentences"][index] for index in split["selection"]], [sst2["train"]["labels"][index] for index in split["selection"]]),
        validation_name: (validation["sentences"], validation["labels"]),
        original_name: (second["sentences"], second["labels"]),
    }
    test_sets = {
        "selection": {"role": "validation-only selection; not a held-out estimate for this checkpoint", "source": f"{config.dataset_repo} train rows listed in Step 10a splits.json"},
        validation_name: {"role": "test", "source": f"{config.dataset_repo}@{config.dataset_revision} official validation split, all rows"},
        original_name: {"role": "test", "source": f"{config.second_test_repo}@{config.second_test_revision} test split, overlap removed", "rule": REMOVAL_RULE,
                        "num_candidates": second["num_candidates"], "num_removed": len(second["removed"]), "removed": second["removed"], "kept_setfit_test_indices": second["indices"]},
    }
    for name, (sentences, labels) in datasets.items():
        test_sets[name].update({"num_examples": len(labels), "label_counts": {str(label): count for label, count in sorted(Counter(labels).items())}, "sha256": test_set_sha256(sentences, labels)})
    return SimpleNamespace(
        root=root, results=results, config=config, step10a_config=asdict(config), manifest=manifest, metrics=metrics, split=split,
        model=model, checkpoint=checkpoint, tokenizer=tokenizer, datasets=datasets, test_sets=test_sets,
        test_names=[validation_name, original_name], paths=paths, hashes=hashes, accepted_source_hash_changes=accepted,
    )


def make_loaders(sources):
    collate = make_collate(sources.tokenizer, sources.config.max_length)
    return {name: DataLoader(SentenceDataset(sentences, labels), batch_size=sources.config.evaluation_batch_size, shuffle=False, collate_fn=collate)
            for name, (sentences, labels) in sources.datasets.items()}


def evaluate_splits(model, loaders, device):
    """Measure every split separately; test sets are never pooled."""
    return {name: evaluate_sequence_classifier(model, loader, device) for name, loader in loaders.items()}


def build_tinybert(model_config, state=None):
    """An evaluation-mode BertForSequenceClassification, optionally loaded with a complete state."""
    from transformers import BertConfig, BertForSequenceClassification

    model = BertForSequenceClassification(BertConfig.from_dict(model_config))
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def linear_weight_names(model):
    return [layer["weight_name"] for layer in linear_embedding_layout(model) if layer["type"] == "Linear"]


def protected_state(model, eligible):
    """Reliable (never clustered or faulted) parameters, grouped by kind."""
    groups = {"embedding_tables": [], "layer_norm": [], "biases": [], "other": []}
    embedding_weights = {layer["weight_name"] for layer in linear_embedding_layout(model) if layer["type"] == "Embedding"}
    for name, value in model.state_dict().items():
        if name in eligible:
            continue
        group = "embedding_tables" if name in embedding_weights else "layer_norm" if "LayerNorm" in name else "biases" if name.endswith(".bias") else "other"
        groups[group].append({"name": name, "shape": list(value.shape), "num_values": value.numel()})
    return {group: {"num_tensors": len(items), "num_values": sum(item["num_values"] for item in items), "tensors": items} for group, items in groups.items()}


def synthetic_token_batch(model_config, *, seed=0, batch_size=2, length=16):
    """Fixed random token IDs for exact logit comparisons; not dataset inputs."""
    generator = torch.Generator().manual_seed(seed)
    length = min(length, model_config["max_position_embeddings"])
    return {"input_ids": torch.randint(0, model_config["vocab_size"], (batch_size, length), generator=generator),
            "token_type_ids": torch.zeros(batch_size, length, dtype=torch.long), "attention_mask": torch.ones(batch_size, length, dtype=torch.long)}
