"""Datasets, transforms and class balancing for the Kaggle chest X-ray dataset."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224):
    train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # No horizontal flips: heart position is anatomically meaningful.
    evaluation = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, evaluation


def build_loaders(root: str, image_size: int = 224, batch_size: int = 32, val_fraction: float = 0.1, seed: int = 42):
    """The official val split has only 16 images, so a stratified val set is carved out of train."""
    train_tf, eval_tf = build_transforms(image_size)
    train_full = datasets.ImageFolder(f"{root}/train", transform=train_tf)
    train_eval_view = datasets.ImageFolder(f"{root}/train", transform=eval_tf)
    test = datasets.ImageFolder(f"{root}/test", transform=eval_tf)

    targets = np.array(train_full.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=val_fraction, stratify=targets, random_state=seed
    )
    train_set, val_set = Subset(train_full, train_idx), Subset(train_eval_view, val_idx)

    # Pneumonia outnumbers normal roughly 3:1; sample inversely to class frequency.
    class_counts = np.bincount(targets[train_idx])
    sample_weights = (1.0 / class_counts)[targets[train_idx]]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(train_idx))

    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, sampler=sampler, **kw),
        DataLoader(val_set, shuffle=False, **kw),
        DataLoader(test, shuffle=False, **kw),
        train_full.classes,
    )
