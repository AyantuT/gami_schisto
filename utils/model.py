"""
model.py – Configurable CNN classifier for vegetation type prediction.

Supports ResNet-18/50 and EfficientNet-B0 backbones with optional band
count adaptation (so you can feed 1-band or 4-band rasters, not just RGB).
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    backbone: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    in_channels: int = 3,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Parameters
    ----------
    backbone    : "resnet18" | "resnet50" | "efficientnet_b0"
    num_classes : number of output classes (2 for emergent/submergent)
    pretrained  : load ImageNet weights
    in_channels : number of input spectral bands
    dropout     : dropout before the final classifier head

    Returns
    -------
    nn.Module ready for training
    """
    weights_arg = "DEFAULT" if pretrained else None

    if backbone.startswith("resnet"):
        model = _build_resnet(backbone, weights_arg, num_classes, in_channels, dropout)
    elif backbone.startswith("efficientnet"):
        model = _build_efficientnet(backbone, weights_arg, num_classes, in_channels, dropout)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose resnet18, resnet50, or efficientnet_b0.")

    return model


# ── backbone builders ─────────────────────────────────────────────────────────

def _build_resnet(backbone, weights_arg, num_classes, in_channels, dropout):
    if backbone == "resnet18":
        model = models.resnet18(weights=weights_arg)
    else:
        model = models.resnet50(weights=weights_arg)

    # Adapt first conv if band count != 3
    if in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        # Initialise new weights: average pretrained RGB weights across bands
        if weights_arg and in_channels != 3:
            with torch.no_grad():
                model.conv1.weight[:] = old.weight.mean(dim=1, keepdim=True).expand_as(model.conv1.weight)

    # Replace classification head
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feats, num_classes),
    )
    return model


def _build_efficientnet(backbone, weights_arg, num_classes, in_channels, dropout):
    model = models.efficientnet_b0(weights=weights_arg)

    if in_channels != 3:
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if weights_arg and in_channels != 3:
            with torch.no_grad():
                model.features[0][0].weight[:] = old.weight.mean(dim=1, keepdim=True).expand_as(
                    model.features[0][0].weight
                )

    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feats, num_classes),
    )
    return model


# ── layer accessors (for GradCAM) ─────────────────────────────────────────────

def get_gradcam_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return the last convolutional layer, which GradCAM hooks into."""
    if backbone.startswith("resnet"):
        return model.layer4[-1]
    elif backbone.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"Unknown backbone: {backbone}")


# ── freeze / unfreeze helpers ──────────────────────────────────────────────────

def freeze_backbone(model: nn.Module, backbone: str):
    """Freeze all parameters except the classification head."""
    head_names = {"resnet18": "fc", "resnet50": "fc", "efficientnet_b0": "classifier"}
    head = head_names.get(backbone, "fc")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head)


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
