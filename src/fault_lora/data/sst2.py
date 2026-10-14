"""SST-2 files, a saved label-stratified selection split, and batch tokenization."""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

SST2_FILES = {name: f"data/{name}-00000-of-00001.parquet" for name in ("train", "validation", "test")}
SETFIT_FILES = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}


def read_sst2(snapshot, cache_dir):
    """Read the pinned GLUE SST-2 parquet files; returns splits and the label names."""
    from datasets import load_dataset

    data = load_dataset("parquet", data_files={name: str(snapshot / path) for name, path in SST2_FILES.items()}, cache_dir=str(cache_dir))
    names = {tuple(split.features["label"].names) for split in data.values()}
    if len(names) != 1:
        raise ValueError("SST-2 splits disagree on label names")
    splits = {name: {"sentences": list(split["sentence"]), "labels": list(split["label"])} for name, split in data.items()}
    for name, split in data.items():
        if list(split["idx"]) != list(range(len(split))):
            raise ValueError(f"SST-2 {name} idx column is not the row position")
    return splits, list(names.pop())


def read_setfit_sst2(snapshot):
    splits = {}
    for name, path in SETFIT_FILES.items():
        rows = [json.loads(line) for line in (snapshot / path).read_text().splitlines() if line.strip()]
        splits[name] = {"sentences": [row["text"] for row in rows], "labels": [row["label"] for row in rows],
                        "label_texts": [row["label_text"] for row in rows]}
    return splits


def proportional_stratified_split(labels, selection_size, seed):
    """Carve `selection_size` indices with each label's share of the whole set.

    Per-label counts use largest-remainder rounding (ties go to the smaller
    label). Each label's indices are shuffled in ascending label order by one
    seeded generator and the first ones are taken.
    """
    labels = np.asarray(labels)
    if not 1 <= selection_size < len(labels):
        raise ValueError("selection_size must leave training examples")
    values, counts = np.unique(labels, return_counts=True)
    exact = counts * selection_size / len(labels)
    quota = np.floor(exact).astype(int)
    for position in np.argsort(-(exact - quota), kind="stable")[: selection_size - quota.sum()]:
        quota[position] += 1
    if (quota < 1).any() or (quota >= counts).any():
        raise ValueError("Each label needs selection and training examples")
    rng = np.random.default_rng(seed)
    selection, train = [], []
    for label, count in zip(values, quota):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        selection.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return {"selection": sorted(selection), "train": sorted(train)}


def normalize_sentence(sentence):
    """Collapse whitespace only: GLUE sentences carry a trailing space."""
    return " ".join(sentence.split())


def sentence_overlap(candidates, references):
    """Count candidate sentences that also occur among the references.

    `exact_raw` compares stored strings byte for byte; `exact` ignores only
    whitespace differences; `casefolded` additionally ignores case.
    """
    raw = set(references)
    spaced = {normalize_sentence(sentence) for sentence in references}
    folded = {sentence.casefold() for sentence in spaced}
    matches = sorted({normalize_sentence(sentence) for sentence in candidates if normalize_sentence(sentence) in spaced})
    return {
        "exact_raw": sum(sentence in raw for sentence in candidates),
        "exact": sum(normalize_sentence(sentence) in spaced for sentence in candidates),
        "casefolded": sum(normalize_sentence(sentence).casefold() in folded for sentence in candidates),
        "exact_matches": matches,
    }


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels):
        if len(sentences) != len(labels):
            raise ValueError("Sentences and labels differ in length")
        self.sentences, self.labels = list(sentences), list(labels)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]


def tokenization_metadata(max_length):
    return {
        "tokenizer": "the model repository's own tokenizer at the pinned model revision",
        "input": "one stored sentence per example, unmodified", "add_special_tokens": True,
        "truncation": True, "max_length": max_length, "padding": "longest sequence in each batch",
        "augmentation": None,
    }


def make_collate(tokenizer, max_length):
    """Tokenize a batch of (sentence, label) pairs into a dict of tensors, padded to its longest."""
    def collate(items):
        sentences, labels = zip(*items)
        batch = dict(tokenizer(list(sentences), truncation=True, max_length=max_length, padding="longest", return_tensors="pt"))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
    return collate


def truncation_statistics(tokenizer, sentences, max_length):
    """Token counts including special tokens, measured without truncation."""
    lengths = np.array([len(ids) for ids in tokenizer(list(sentences), add_special_tokens=True, truncation=False)["input_ids"]])
    truncated = int((lengths > max_length).sum())
    return {"num_sentences": len(lengths), "num_truncated": truncated, "fraction_truncated": truncated / len(lengths),
            "max_tokens": int(lengths.max()), "mean_tokens": float(lengths.mean())}
