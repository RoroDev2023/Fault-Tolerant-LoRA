"""Pinned Hugging Face BERT classifiers, their Linear/Embedding layout, and checkpoint restoration."""

from pathlib import Path
import re

import torch
from torch import nn

MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
WEIGHT_FILES = ("model.safetensors",)


def fetch_snapshot(repo_id, revision, cache_dir, files, *, repo_type="model", download=False):
    """Return the local snapshot of exactly one repository commit.

    Only the listed files are requested. Without `download`, nothing is fetched
    and a missing file is an error.
    """
    from huggingface_hub import snapshot_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character commit hash, not a branch or tag")
    snapshot = Path(snapshot_download(repo_id, repo_type=repo_type, revision=revision, cache_dir=str(cache_dir),
                                      allow_patterns=list(files), local_files_only=not download))
    if snapshot.name != revision:
        raise RuntimeError(f"Resolved snapshot {snapshot.name} differs from the pinned revision")
    missing = [name for name in files if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{repo_id}@{revision} is missing {missing}; rerun with --download")
    return snapshot


def load_sequence_classifier(snapshot):
    """Load the model and its own tokenizer from a local snapshot directory only."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model.eval()
    return model, tokenizer


def linear_embedding_layout(model):
    """Name and shape of every nn.Linear and nn.Embedding, in module order."""
    layout = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        bias = getattr(module, "bias", None)
        layout.append({
            "name": name, "type": type(module).__name__,
            "weight_name": f"{name}.weight", "weight_shape": list(module.weight.shape),
            "bias_shape": None if bias is None else list(bias.shape),
            "num_weights": module.weight.numel(), "num_bias": 0 if bias is None else bias.numel(),
        })
    return layout


def load_tinybert_checkpoint(path, *, device="cpu"):
    """Restore a saved clean classifier without any Hugging Face download."""
    from transformers import BertConfig, BertForSequenceClassification

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BertForSequenceClassification(BertConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint
