"""Loss functions for binary tumor segmentation.

Combined Dice + BCE is the standard recipe: BCE keeps gradients alive when
the prediction is far off, Dice handles the foreground class imbalance
(tumor is typically <2% of voxels in a slice).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice on logits. eps avoids div-by-zero on empty masks."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        target = target.flatten(1)
        intersection = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """alpha * BCE + beta * Dice. Defaults match the SMP recipe."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = self.dice(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Hard Dice per sample. Returns shape (N,)."""
    pred = (torch.sigmoid(logits) > threshold).float().flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return (2 * intersection + eps) / (denom + eps)


@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > threshold).float().flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)
