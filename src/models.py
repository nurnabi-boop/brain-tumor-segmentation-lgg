"""Model factory wrapping segmentation_models_pytorch.

We compare three architectures from the same library so the only thing
varying is the model class + encoder, not the data pipeline.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

MODEL_REGISTRY = {
    "unet_resnet34": dict(arch="Unet", encoder_name="resnet34", encoder_weights="imagenet"),
    "unetpp_effb0": dict(arch="UnetPlusPlus", encoder_name="efficientnet-b0", encoder_weights="imagenet"),
    "deeplabv3p_resnet34": dict(arch="DeepLabV3Plus", encoder_name="resnet34", encoder_weights="imagenet"),
}


def build_model(name: str, in_channels: int = 3, classes: int = 1) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    cfg = MODEL_REGISTRY[name]
    arch_cls = getattr(smp, cfg["arch"])
    return arch_cls(
        encoder_name=cfg["encoder_name"],
        encoder_weights=cfg["encoder_weights"],
        in_channels=in_channels,
        classes=classes,
        activation=None,  # we apply sigmoid in losses/metrics
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
