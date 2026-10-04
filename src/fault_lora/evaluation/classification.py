"""Sample-weighted classification measurements for an explicit dataset split."""

import math

import torch
from torch.nn import functional as F


def evaluate_classifier(model, loader, device):
    previous_modes = [(module, module.training) for module in model.modules()]
    total_loss = 0.0
    correct = 0
    count = 0
    model.eval()
    try:
        with torch.inference_mode():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets, reduction="sum")
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == targets).sum().item()
                count += targets.numel()
    finally:
        for module, mode in previous_modes:
            module.training = mode
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if not math.isfinite(total_loss):
        raise ValueError("Non-finite evaluation loss")
    return {
        "num_examples": count,
        "num_correct": correct,
        "accuracy": correct / count,
        "cross_entropy_loss": total_loss / count,
    }
