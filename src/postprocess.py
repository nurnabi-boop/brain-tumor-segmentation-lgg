"""Post-processing for binary tumor masks.

Connected-component filtering removes small false-positive specks the model
sometimes scatters across non-tumor tissue. Tune `min_area_px` on val.
"""
from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(mask: np.ndarray, min_area_px: int = 50) -> np.ndarray:
    """Drop components whose area is below `min_area_px`. Returns uint8 0/1 mask."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for label_id in range(1, n_labels):  # 0 is background
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == label_id] = 1
    return out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the single largest CC. LGG tumors are typically one lesion."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes via flood-fill from the border."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    binary = (mask > 0).astype(np.uint8) * 255
    h, w = binary.shape
    flood = binary.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return ((binary | holes) > 0).astype(np.uint8)


def postprocess_mask(
    mask: np.ndarray,
    min_area_px: int = 50,
    largest_only: bool = False,
    fill: bool = False,
) -> np.ndarray:
    out = remove_small_components(mask, min_area_px=min_area_px)
    if fill:
        out = fill_holes(out)
    if largest_only:
        out = keep_largest_component(out)
    return out
