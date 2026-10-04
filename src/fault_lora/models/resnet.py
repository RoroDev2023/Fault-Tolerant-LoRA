"""ResNet-18 construction and clean checkpoint restoration."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes=10, *, pretrained=True, cache_dir=None):
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if cache_dir is not None:
        torch.hub.set_dir(str(Path(cache_dir)))
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if num_classes != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_clean_checkpoint(path, *, device="cpu"):
    """Restore a saved baseline without downloading pretrained weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_resnet18(checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint
