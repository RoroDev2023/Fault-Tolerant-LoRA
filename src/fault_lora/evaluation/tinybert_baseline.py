"""Evaluate a pinned, downloaded TinyBERT SST-2 checkpoint with no fine-tuning."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from fault_lora.data.sst2 import (SETFIT_FILES, SST2_FILES, SentenceDataset, make_collate, normalize_sentence, proportional_stratified_split,
                                  read_setfit_sst2, read_sst2, sentence_overlap, tokenization_metadata, truncation_statistics)
from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.sequence_classification import evaluate_sequence_classifier
from fault_lora.models.tinybert import MODEL_FILES, WEIGHT_FILES, fetch_snapshot, linear_embedding_layout, load_sequence_classifier, load_tinybert_checkpoint
from fault_lora.training.baseline import choose_device

SECOND_TEST_NAME = "sst2-original-test"
REPRESENTATION = "original dense weights as downloaded; no fine-tuning, clustering, faults, or adapters"
# Used only as a recorded sanity check of the label order; never for selection.
PROBES = {"negative": "a dull , lifeless and painfully boring mess .", "positive": "a wonderful , moving and very funny film ."}


@dataclass
class TinyBertConfig:
    run_id: str = "tinybert-sst2-clean-seed42"
    seed: int = 42
    model_repo: str = "Vishnou/TinyBERT_SST2"
    model_revision: str = "5aa94d561e626f70a4380e2a51c786e64f3acad7"
    dataset_repo: str = "stanfordnlp/sst2"
    dataset_revision: str = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
    second_test_repo: str | None = "SetFit/sst2"
    second_test_revision: str | None = "00ea8ccb7a54b4e3780a3e51aa3f80361ff849c0"
    selection_examples: int = 5000
    expected_train_examples: int = 67349
    expected_test_examples: int = 872
    label_names: list = field(default_factory=lambda: ["negative", "positive"])
    model_card_accuracy: float | None = 0.8864678899082569
    max_length: int = 128
    evaluation_batch_size: int = 256
    device: str = "cpu"
    cpu_threads: int = 4
    model_cache: str = "checkpoints/hf-cache"
    data_cache: str = "data/hf-cache"
    checkpoint_root: str = "checkpoints"
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        for name in ("seed", "selection_examples", "expected_train_examples", "expected_test_examples", "max_length", "evaluation_batch_size", "cpu_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**32 or min(self.max_length, self.evaluation_batch_size, self.cpu_threads, self.expected_test_examples) < 1:
            raise ValueError("Invalid seed, length, batch size, thread count, or test size")
        if not 1 <= self.selection_examples < self.expected_train_examples:
            raise ValueError("The selection split must leave training examples")
        revisions = [self.model_revision, self.dataset_revision] + ([self.second_test_revision] if self.second_test_repo is not None else [])
        if (self.second_test_repo is None) != (self.second_test_revision is None) or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) for value in revisions):
            raise ValueError("Every repository needs a full 40-character commit hash")
        if not isinstance(self.label_names, list) or len(self.label_names) < 2 or len(set(self.label_names)) != len(self.label_names):
            raise ValueError("label_names must list distinct class names in index order")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def split_sha256(split):
    return hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_records(root, snapshot, names):
    """Path, size, and SHA-256 of each downloaded file (snapshot entries are links into the cache)."""
    return {name: {"path": str((snapshot / name).relative_to(root)), "bytes": (snapshot / name).stat().st_size, "sha256": sha256_file(snapshot / name)} for name in names}


def check_second_test(candidate, sst2, label_names):
    """Decide, before any evaluation, whether a labelled test set may be reported.

    It is usable only if every label is a real class index that agrees with its
    text name, and no sentence equals an SST-2 train or validation sentence once
    whitespace differences are ignored.
    """
    test = candidate["test"]
    labels_real = set(test["labels"]) == set(range(len(label_names)))
    names_agree = labels_real and all(label_names[label] == text for label, text in zip(test["labels"], test["label_texts"]))
    overlap = {name: sentence_overlap(test["sentences"], sst2[name]["sentences"]) for name in ("train", "validation", "test")}
    leaked = {name: overlap[name]["exact"] for name in ("train", "validation")}
    reasons = []
    if not labels_real:
        reasons.append(f"labels are {sorted(set(test['labels']))}, not exactly the class indices")
    elif not names_agree:
        reasons.append("label indices disagree with the label names")
    for name, count in leaked.items():
        if count:
            reasons.append(f"{count} of {len(test['labels'])} test sentences also occur in the SST-2 {name} split")
    return {
        "name": SECOND_TEST_NAME, "usable": not reasons, "reason": "; ".join(reasons) or "real labels and no sentence shared with SST-2 train or validation",
        "rule": "labels are real class indices matching their names, and zero exact sentence matches (whitespace-normalized) with SST-2 train and validation",
        "num_examples": len(test["labels"]), "label_counts": {str(label): count for label, count in sorted(Counter(test["labels"]).items())},
        "labels_real": labels_real, "label_names_agree": names_agree,
        "overlap_with_sst2": overlap, "note": "Overlap with the SST-2 test split is informational: those GLUE labels are hidden.",
    }


def environment_record(device, threads):
    packages = ("torch", "numpy", "transformers", "datasets", "tokenizers", "huggingface_hub", "safetensors", "pyarrow")
    return {"python": platform.python_version(), "platform": platform.platform(), "device": str(device), "cpu_threads": threads,
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "reproducibility": "Fixed weights, saved splits, and deterministic evaluation; bitwise equality across devices, thread counts, batch sizes, or library versions is not guaranteed."}


def verify_tinybert_baseline(root, config):
    """Re-read every saved record and artifact; fail on any inconsistency."""
    from safetensors.torch import load_file

    root = Path(root).resolve()
    results_dir = root / config.results_root / config.run_id
    read = lambda name: json.loads((results_dir / name).read_text())
    require(read("config.json") == asdict(config), "Saved configuration differs")
    split, manifest, metrics, layers = read("splits.json"), read("manifest.json"), read("metrics.json"), read("layers.json")
    selection, train = set(split["selection"]), set(split["train"])
    require(len(selection) == len(split["selection"]) == config.selection_examples and len(train) == len(split["train"]), "Split sizes or duplicates are wrong")
    require(not selection & train and selection | train == set(range(config.expected_train_examples)), "Selection and training indices are not a partition of the train split")
    require(split["test"]["indices"] == list(range(config.expected_test_examples)) and split["test"]["source"] == "official validation split", "Unexpected test split")
    digest = split_sha256(split)
    checkpoint_path = root / metrics["checkpoint"]
    model, checkpoint = load_tinybert_checkpoint(checkpoint_path)
    require(digest == manifest["split_sha256"] == metrics["split_sha256"] == checkpoint["split_sha256"], "Split hash differs between records")
    require(sha256_file(checkpoint_path) == metrics["checkpoint_sha256"], "Saved checkpoint differs from metrics")
    for group in manifest["source_artifacts"].values():
        for name, item in group["files"].items():
            require(sha256_file(root / item["path"]) == item["sha256"], f"Downloaded file changed: {name}")
    for relative, expected in read("source_sha256.json").items():
        require(sha256_file(root / relative) == expected, f"Source code changed during the run: {relative}")
    # No fine-tuning: every saved tensor equals the downloaded weight file.
    downloaded = load_file(root / manifest["source_artifacts"]["model"]["files"]["model.safetensors"]["path"])
    state = checkpoint["model_state_dict"]
    for name, value in state.items():
        require(name in downloaded and torch.equal(value, downloaded[name]), f"Saved weights differ from the downloaded checkpoint: {name}")
    require(checkpoint["representation"] == REPRESENTATION, "Unexpected checkpoint representation")
    for layer in layers["layers"]:
        require(list(state[layer["weight_name"]].shape) == layer["weight_shape"], f"Recorded layer shape differs: {layer['name']}")
    require(layers["layers"] == linear_embedding_layout(model), "Recorded layout differs from the restored model")
    require(sum(value.numel() for value in model.parameters()) == layers["total_parameters"], "Parameter count differs")
    sizes = {"selection": config.selection_examples, "test": config.expected_test_examples}
    second = read("second_test_set.json")
    if second["usable"]:
        sizes[SECOND_TEST_NAME] = second["num_examples"]
    require(set(metrics["measurements"]) == set(sizes), "Measured splits differ from the declared splits")
    for name, measurement in metrics["measurements"].items():
        require(measurement["num_examples"] == sizes[name] and measurement["accuracy"] == measurement["num_correct"] / measurement["num_examples"], f"Inconsistent {name} measurement")
    mapping = read("label_mapping.json")
    require(mapping["confirmed"] and mapping["mapping"] == {str(index): name for index, name in enumerate(config.label_names)}, "Label mapping was not confirmed")
    return {"status": "passed", "configuration_matches": True, "split_is_partition_with_matching_hashes": True, "downloaded_file_hashes_verified": True,
            "saved_weights_equal_downloaded_weights": True, "state_tensors_compared": len(state), "downloaded_tensors_not_in_model": sorted(set(downloaded) - set(state)),
            "layout_matches_restored_model": True, "measurement_counts_verified": sorted(sizes), "label_mapping_confirmed": True, "source_code_unchanged": True}


def run_tinybert_baseline(config, project_root, *, download=False):
    config.validate()
    # Resolved so that downloaded snapshot paths can be recorded relative to the project.
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ.setdefault("HF_HOME", str(root / config.model_cache))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    results_dir = root / config.results_root / config.run_id
    checkpoint_dir = root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run directories already exist. Choose a new run_id; existing research artifacts are preserved.")
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_baseline.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})

        model_snapshot = fetch_snapshot(config.model_repo, config.model_revision, root / config.model_cache / "hub", MODEL_FILES, download=download)
        data_snapshot = fetch_snapshot(config.dataset_repo, config.dataset_revision, root / config.data_cache / "hub", SST2_FILES.values(), repo_type="dataset", download=download)
        sources = {
            "model": {"repo": config.model_repo, "revision": config.model_revision, "role": "model weights, configuration, and tokenizer", "files": file_records(root, model_snapshot, MODEL_FILES)},
            "dataset": {"repo": config.dataset_repo, "revision": config.dataset_revision, "role": "SST-2 train, validation, and unlabelled test", "files": file_records(root, data_snapshot, SST2_FILES.values())},
        }
        model, tokenizer = load_sequence_classifier(model_snapshot)
        model.to(device).requires_grad_(False)
        environment = environment_record(device, config.cpu_threads)
        write_json(results_dir / "environment.json", environment)

        sst2, dataset_label_names = read_sst2(data_snapshot, root / config.data_cache / "datasets")
        train, official_validation = sst2["train"], sst2["validation"]
        require(len(train["labels"]) == config.expected_train_examples and len(official_validation["labels"]) == config.expected_test_examples, "SST-2 split sizes differ from the configuration")
        classes = set(range(len(config.label_names)))
        require(set(train["labels"]) == classes and set(official_validation["labels"]) == classes, "Train or validation labels are not the declared classes")
        hidden_test_labels = {str(label): count for label, count in sorted(Counter(sst2["test"]["labels"]).items())}

        split = proportional_stratified_split(train["labels"], config.selection_examples, config.seed)
        split.update({"seed": config.seed, "dataset": config.dataset_repo, "dataset_revision": config.dataset_revision, "index_space": "row position in the official train split (equal to its idx column)",
                      "rule": "label-proportional with largest-remainder rounding; per-label shuffle by one numpy default_rng(seed) in ascending label order",
                      "test": {"source": "official validation split", "indices": list(range(config.expected_test_examples))}})
        split_hash = split_sha256(split)
        write_json(results_dir / "splits.json", split)
        take = lambda name: ([train["sentences"][index] for index in split[name]], [train["labels"][index] for index in split[name]])
        splits = {"selection": take("selection"), "train": take("train"), "test": (official_validation["sentences"], official_validation["labels"])}
        train_sentences = {normalize_sentence(sentence) for sentence in splits["train"][0]}
        split_summary = {
            "sizes": {name: len(labels) for name, (_, labels) in splits.items()},
            "label_counts": {name: {str(label): count for label, count in sorted(Counter(labels).items())} for name, (_, labels) in splits.items()},
            "official_test_split": {"num_examples": len(sst2["test"]["labels"]), "label_counts": hidden_test_labels, "used": False, "reason": "GLUE test labels are hidden"},
            # The official train split repeats some sentences, so an index partition is not a sentence partition.
            "selection_sentences_also_in_training_split": sum(normalize_sentence(sentence) in train_sentences for sentence in splits["selection"][0]),
            "test_sentences_in_official_train_split": sentence_overlap(official_validation["sentences"], train["sentences"])["exact"],
        }

        if config.second_test_repo is None:
            second = {"name": SECOND_TEST_NAME, "usable": False, "reason": "no second test repository configured"}
        else:
            second_snapshot = fetch_snapshot(config.second_test_repo, config.second_test_revision, root / config.data_cache / "hub", SETFIT_FILES.values(), repo_type="dataset", download=download)
            sources["second_test"] = {"repo": config.second_test_repo, "revision": config.second_test_revision, "role": "candidate labelled test set", "files": file_records(root, second_snapshot, SETFIT_FILES.values())}
            candidate = read_setfit_sst2(second_snapshot)
            second = {"repo": config.second_test_repo, "revision": config.second_test_revision, **check_second_test(candidate, sst2, config.label_names)}
            if second["usable"]:
                splits[SECOND_TEST_NAME] = (candidate["test"]["sentences"], candidate["test"]["labels"])
        write_json(results_dir / "second_test_set.json", second)
        print(json.dumps({"event": "second_test_set_checked", "usable": second["usable"], "reason": second["reason"]}), flush=True)

        preprocessing = {**tokenization_metadata(config.max_length), "tokenizer_class": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size, "model_inputs": list(tokenizer.model_input_names)}
        write_json(results_dir / "preprocessing.json", preprocessing)
        truncation = {name: truncation_statistics(tokenizer, sentences, config.max_length) for name, (sentences, _) in splits.items()}
        truncation["all"] = {"num_sentences": sum(item["num_sentences"] for item in truncation.values()), "num_truncated": sum(item["num_truncated"] for item in truncation.values()), "max_tokens": max(item["max_tokens"] for item in truncation.values())}
        truncation["all"]["fraction_truncated"] = truncation["all"]["num_truncated"] / truncation["all"]["num_sentences"]
        write_json(results_dir / "truncation.json", {"max_length": config.max_length, "splits": truncation})

        layout = linear_embedding_layout(model)
        total_parameters = sum(value.numel() for value in model.parameters())
        write_json(results_dir / "layers.json", {
            "total_parameters": total_parameters, "num_layers": len(layout),
            "num_linear": sum(layer["type"] == "Linear" for layer in layout), "num_embedding": sum(layer["type"] == "Embedding" for layer in layout),
            "linear_embedding_weights": sum(layer["num_weights"] for layer in layout), "linear_bias": sum(layer["num_bias"] for layer in layout),
            "other_parameters": total_parameters - sum(layer["num_weights"] + layer["num_bias"] for layer in layout), "layers": layout})

        model_config = json.loads(json.dumps(model.config.to_dict()))
        metadata = {
            "architecture": type(model).__name__, "model_repo": config.model_repo, "model_revision": config.model_revision,
            "num_classes": len(config.label_names), "classes": config.label_names, "dataset": config.dataset_repo, "dataset_revision": config.dataset_revision,
            "model_config": model_config, "config": asdict(config), "preprocessing": preprocessing, "split_sha256": split_hash,
            "weight_file_sha256": {name: sources["model"]["files"][name]["sha256"] for name in WEIGHT_FILES},
            "environment": environment, "representation": REPRESENTATION,
        }
        write_json(results_dir / "manifest.json", {
            **{name: value for name, value in metadata.items() if name != "model_config"}, "source_artifacts": sources, "splits": split_summary,
            "selection_rule": "none applied in this run: there is no training, and one fixed checkpoint is evaluated",
            "split_roles": {"selection": "validation-only choices in later steps", "train": "LoRA training in later steps; not evaluated here", "test": "official validation split; reported only, never used for any selection"},
            "test_policy": "The selection split is evaluated first and alone confirms the label order; test sets are evaluated once afterwards.",
        })

        loaders = {name: DataLoader(SentenceDataset(sentences, labels), batch_size=config.evaluation_batch_size, shuffle=False, collate_fn=make_collate(tokenizer, config.max_length))
                   for name, (sentences, labels) in splits.items() if name != "train"}
        measurements = {}
        print(json.dumps({"event": "evaluation_start", "split": "selection"}), flush=True)
        measurements["selection"] = evaluate_sequence_classifier(model, loaders["selection"], device)
        print(json.dumps({"event": "evaluation_complete", "split": "selection", **measurements["selection"]}), flush=True)

        # Label order is confirmed before any test set is read by the model.
        raw_id2label = json.loads((model_snapshot / "config.json").read_text()).get("id2label")
        resolved = {str(index): name for index, name in model.config.id2label.items()}
        expected_mapping = {str(index): name for index, name in enumerate(config.label_names)}
        probe_predictions = []
        if PROBES:
            with torch.inference_mode():
                probe_batch = {name: value.to(device) for name, value in tokenizer(list(PROBES.values()), padding="longest", return_tensors="pt").items()}
                probe_predictions = model(**probe_batch).logits.argmax(dim=1).tolist()
        probes = {name: {"sentence": sentence, "predicted_index": prediction, "expected_index": config.label_names.index(name)} for (name, sentence), prediction in zip(PROBES.items(), probe_predictions)}
        checks = {
            "dataset_label_names_match": dataset_label_names == config.label_names,
            "config_id2label_names_match": None if raw_id2label is None else {key: value.lower() for key, value in resolved.items()} == expected_mapping,
            "selection_accuracy_above_flipped": measurements["selection"]["accuracy"] > 1 - measurements["selection"]["accuracy"] if len(config.label_names) == 2 else None,
            "probe_sentences_match": all(item["predicted_index"] == item["expected_index"] for item in probes.values()),
        }
        mapping = {
            "mapping": expected_mapping, "confirmed": all(value is not False for value in checks.values()), "checks": checks,
            "model_config_id2label_in_file": raw_id2label, "model_config_id2label_resolved": resolved, "dataset_label_names": dataset_label_names,
            "selection_accuracy": measurements["selection"]["accuracy"], "selection_accuracy_with_flipped_labels": 1 - measurements["selection"]["accuracy"] if len(config.label_names) == 2 else None,
            "probes": probes,
            "basis": ("model configuration names" if raw_id2label is not None else "the model configuration stores no id2label, so transformers supplies generic names; the order is confirmed by the dataset's label names together with selection-split accuracy and two probe sentences"),
        }
        write_json(results_dir / "label_mapping.json", mapping)
        require(mapping["confirmed"], f"Label mapping could not be confirmed: {checks}")

        checkpoint_path = checkpoint_dir / "clean.pt"
        save_torch(checkpoint_path, {"model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, **metadata, "selection_metrics": measurements["selection"]})
        restored, _ = load_tinybert_checkpoint(checkpoint_path, device=device)
        first = {name: value.to(device) for name, value in next(iter(loaders["selection"])).items() if name != "labels"}
        with torch.inference_mode():
            require(torch.equal(restored(**first).logits, model(**first).logits), "Restored checkpoint logits differ")
        del restored

        for name in loaders:
            if name == "selection":
                continue
            print(json.dumps({"event": "evaluation_start", "split": name}), flush=True)
            measurements[name] = evaluate_sequence_classifier(model, loaders[name], device)
            print(json.dumps({"event": "evaluation_complete", "split": name, **measurements[name]}), flush=True)

        reference = None
        if config.model_card_accuracy is not None:
            reference = {"accuracy": config.model_card_accuracy, "split": "official validation split, which is this run's test split", "source": "model card; not measured here",
                         "accuracy_difference_percentage_points": 100 * (measurements["test"]["accuracy"] - config.model_card_accuracy)}
        metrics = {
            "run_id": config.run_id, "fine_tuning": "none", "measurements": measurements, "model_card_reference": reference,
            "checkpoint": str(checkpoint_path.relative_to(root)), "checkpoint_sha256": sha256_file(checkpoint_path),
            "split_sha256": split_hash, "elapsed_seconds": time.monotonic() - started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(results_dir / "metrics.json", metrics)
        write_json(results_dir / "verification.json", verify_tinybert_baseline(root, config))
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": metrics["completed_at_utc"]})
        print(json.dumps({"event": "tinybert_baseline_complete", **metrics}), flush=True)
        return metrics
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error": str(error), "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2.json")
    parser.add_argument("--download", action="store_true", help="Download the pinned model and dataset files if they are missing from the local caches.")
    args = parser.parse_args()
    run_tinybert_baseline(TinyBertConfig(**json.loads(args.config.read_text())), project_root, download=args.download)
