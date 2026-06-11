"""LGG MRI segmentation dataset with patient-level splitting.

The Kaggle dataset by Mateusz Buda lays out files as:
    kaggle_3m/
        TCGA_<institution>_<id>_<date>/
            TCGA_<...>_<slice_idx>.tif        # FLAIR slice (RGB-stacked)
            TCGA_<...>_<slice_idx>_mask.tif   # binary tumor mask

Patient ID is the folder name. Splits MUST keep all slices of one patient
in a single split — otherwise leakage inflates Dice by ~5-10 points.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MASK_SUFFIX = "_mask.tif"
SLICE_RE = re.compile(r"_(\d+)\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class SliceRecord:
    patient_id: str
    image_path: str
    mask_path: str
    slice_idx: int
    has_tumor: bool


def discover_slices(root: str | os.PathLike) -> list[SliceRecord]:
    """Walk the kaggle_3m directory and pair every image with its mask.

    Empty masks are kept here; the caller decides whether to filter them.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records: list[SliceRecord] = []
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img_path in sorted(patient_dir.glob("*.tif")):
            name = img_path.name
            if name.endswith(MASK_SUFFIX):
                continue
            mask_path = img_path.with_name(img_path.stem + MASK_SUFFIX)
            if not mask_path.exists():
                continue

            m = SLICE_RE.search(name)
            slice_idx = int(m.group(1)) if m else -1

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            has_tumor = bool(mask is not None and mask.max() > 0)

            records.append(
                SliceRecord(
                    patient_id=patient_dir.name,
                    image_path=str(img_path),
                    mask_path=str(mask_path),
                    slice_idx=slice_idx,
                    has_tumor=has_tumor,
                )
            )
    return records


def patient_level_split(
    records: list[SliceRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[SliceRecord]]:
    """Split slices into train/val/test by patient ID.

    Every slice from one patient ends up in exactly one split. The function
    asserts disjointness before returning; tests rely on this guarantee.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")

    patients = sorted({r.patient_id for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test : n_test + n_val])
    train_p = set(patients[n_test + n_val :])

    overlap = (train_p & val_p) | (train_p & test_p) | (val_p & test_p)
    assert not overlap, f"Patient leakage across splits: {overlap}"

    return {
        "train": [r for r in records if r.patient_id in train_p],
        "val": [r for r in records if r.patient_id in val_p],
        "test": [r for r in records if r.patient_id in test_p],
    }


def filter_tumor_only(records: Iterable[SliceRecord]) -> list[SliceRecord]:
    """Drop empty-mask slices. Use for training; keep all for evaluation."""
    return [r for r in records if r.has_tumor]


class LGGSegmentationDataset(Dataset):
    """FLAIR slice + binary mask, returned as float tensors in [0, 1]."""

    def __init__(
        self,
        records: list[SliceRecord],
        transform=None,
        image_size: int = 256,
    ):
        self.records = records
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        image = cv2.imread(rec.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {rec.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(rec.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {rec.mask_path}")
        mask = (mask > 0).astype(np.uint8)

        if (image.shape[0], image.shape[1]) != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        if not torch.is_tensor(image):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).float()
        mask = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()

        return {
            "image": image,
            "mask": mask,
            "patient_id": rec.patient_id,
            "slice_idx": rec.slice_idx,
            "has_tumor": rec.has_tumor,
        }


def verify_no_patient_leakage(splits: dict[str, list[SliceRecord]]) -> None:
    """Raise AssertionError if any patient ID appears in more than one split."""
    seen: dict[str, str] = {}
    for split_name, recs in splits.items():
        for r in recs:
            prev = seen.get(r.patient_id)
            if prev is not None and prev != split_name:
                raise AssertionError(
                    f"Patient {r.patient_id} appears in both '{prev}' and '{split_name}'"
                )
            seen[r.patient_id] = split_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to kaggle_3m directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = discover_slices(args.root)
    print(f"Discovered {len(records)} slices across {len({r.patient_id for r in records})} patients")
    print(f"  with tumor: {sum(r.has_tumor for r in records)}")
    print(f"  empty:      {sum(not r.has_tumor for r in records)}")

    splits = patient_level_split(records, seed=args.seed)
    verify_no_patient_leakage(splits)
    for name, recs in splits.items():
        n_pat = len({r.patient_id for r in recs})
        n_tum = sum(r.has_tumor for r in recs)
        print(f"  {name:5s}: {len(recs):4d} slices / {n_pat:3d} patients / {n_tum:4d} tumor")
    print("OK: no patient leakage across splits")
