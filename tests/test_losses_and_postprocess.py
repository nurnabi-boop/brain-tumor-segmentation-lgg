"""Smoke tests for losses and post-processing — run without GPU or data."""
from __future__ import annotations

import numpy as np
import torch

from src.losses import DiceBCELoss, DiceLoss, dice_score, iou_score
from src.postprocess import (
    fill_holes,
    keep_largest_component,
    postprocess_mask,
    remove_small_components,
)


def test_dice_loss_perfect_prediction():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    logits = (target * 20 - 10)  # very confident, correct
    loss = DiceLoss()(logits, target).item()
    assert loss < 0.01, f"expected near-zero loss, got {loss}"


def test_dice_bce_combined():
    target = torch.zeros(4, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1
    logits = torch.randn_like(target)
    loss = DiceBCELoss()(logits, target)
    assert torch.isfinite(loss) and loss.item() > 0


def test_dice_score_shape():
    logits = torch.randn(8, 1, 32, 32)
    target = (torch.rand(8, 1, 32, 32) > 0.5).float()
    s = dice_score(logits, target)
    assert s.shape == (8,)
    assert (s >= 0).all() and (s <= 1).all()
    iou = iou_score(logits, target)
    assert iou.shape == (8,) and (iou >= 0).all() and (iou <= 1).all()


def test_remove_small_components():
    mask = np.zeros((50, 50), np.uint8)
    mask[2:5, 2:5] = 1                 # 9 px speck
    mask[20:35, 20:35] = 1             # 225 px lesion
    cleaned = remove_small_components(mask, min_area_px=50)
    assert cleaned[3, 3] == 0
    assert cleaned[27, 27] == 1


def test_keep_largest_component():
    mask = np.zeros((50, 50), np.uint8)
    mask[2:7, 2:7] = 1
    mask[20:40, 20:40] = 1
    largest = keep_largest_component(mask)
    assert largest[3, 3] == 0
    assert largest[30, 30] == 1


def test_fill_holes():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:40, 10:40] = 1
    mask[20:30, 20:30] = 0  # interior hole
    filled = fill_holes(mask)
    assert filled[25, 25] == 1


def test_postprocess_mask_pipeline():
    mask = np.zeros((64, 64), np.uint8)
    mask[2:5, 2:5] = 1
    mask[20:50, 20:50] = 1
    out = postprocess_mask(mask, min_area_px=50, largest_only=True)
    assert out[3, 3] == 0
    assert out[35, 35] == 1


if __name__ == "__main__":
    test_dice_loss_perfect_prediction()
    test_dice_bce_combined()
    test_dice_score_shape()
    test_remove_small_components()
    test_keep_largest_component()
    test_fill_holes()
    test_postprocess_mask_pipeline()
    print("All loss + postprocess tests passed.")
