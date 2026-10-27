"""DenseNet-121 classifier (the backbone used by CheXNet) with a single logit output."""

import torch
from torchvision.models import DenseNet121_Weights, densenet121


def build_model(pretrained: bool = True) -> torch.nn.Module:
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    return model


def gradcam_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Last dense block: the deepest layer that still has spatial resolution."""
    return model.features.denseblock4
