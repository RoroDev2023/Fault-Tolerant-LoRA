"""Sample-weighted classification measurements for dict-of-tensor batches.

Sibling of `classification.evaluate_classifier`, which takes `(inputs, targets)`
batches. That module is left byte-identical because completed runs record its
hash. Both return the same four measurements.
"""

import math

import torch
from torch.nn import functional as F


def evaluate_sequence_classifier(model, loader, device, *, label_key="labels"):
    """Evaluate a model called as `model(**batch)` on batches holding their labels.

    Labels are removed from the batch before the forward pass, so the model
    never computes its own loss. The output may be a logits tensor or an object
    with a `logits` attribute. Every label must be a valid class index; hidden
    labels such as -1 are rejected instead of being scored.
    """
    previous_modes = [(module, module.training) for module in model.modules()]
    total_loss = 0.0
    correct = 0
    count = 0
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                if label_key not in batch:
                    raise ValueError(f"Batch has no {label_key!r} entry")
                inputs = {name: value.to(device) for name, value in batch.items() if name != label_key}
                targets = batch[label_key].to(device)
                output = model(**inputs)
                logits = output if isinstance(output, torch.Tensor) else output.logits
                if logits.ndim != 2 or logits.shape[0] != targets.numel():
                    raise ValueError("Logits must have shape (examples, classes)")
                if targets.numel() and not 0 <= targets.min().item() <= targets.max().item() < logits.shape[1]:
                    raise ValueError("Labels must be class indices; hidden or out-of-range labels cannot be scored")
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
