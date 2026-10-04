"""CIFAR-10 preprocessing and saved, disjoint training/validation splits."""

import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stratified_split(targets, validation_per_class, seed):
    targets = np.asarray(targets)
    if validation_per_class < 1:
        raise ValueError("validation_per_class must be positive")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        if validation_per_class >= len(indices):
            raise ValueError("Each class must retain training examples")
        rng.shuffle(indices)
        validation.extend(indices[:validation_per_class].tolist())
        train.extend(indices[validation_per_class:].tolist())
    return {"train": sorted(train), "validation": sorted(validation)}


def cifar10_transforms(image_size):
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    common = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        *common,
    ])
    evaluation = transforms.Compose(common)
    return train, evaluation


def preprocessing_metadata(image_size):
    return {
        "input_size": [3, image_size, image_size],
        "interpolation": "bilinear",
        "antialias": True,
        "pixel_scale": "uint8 [0,255] to float32 [0,1]",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "training_augmentation": {"random_crop": 32, "crop_padding": 4, "horizontal_flip_probability": 0.5},
        "evaluation_augmentation": None,
        "note": "CIFAR-10 images are resized directly; this differs from the pretrained 256-resize/224-center-crop transform.",
    }
