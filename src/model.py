"""CIFAR-style ResNet-18 matching the target / suspect architecture.

torchvision resnet18 with the first conv replaced by a 3x3 stride-1 conv,
the maxpool removed, and a 100-class head — the architecture every target
and suspect uses.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from safetensors.torch import load_file


def make_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def load_weights(path: str, device: str = "cpu") -> nn.Module:
    """Load a safetensors checkpoint into make_model() and return it in eval mode."""
    state_dict = load_file(path, device=device)
    model = make_model()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if device != "cpu":
        model.to(device)
    return model
