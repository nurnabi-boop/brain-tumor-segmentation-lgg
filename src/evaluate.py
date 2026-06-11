"""Per-patient evaluation with Dice, IoU, Hausdorff, sensitivity, specificity.

Aggregation order matters: compute per-slice metrics, average within patient,
then average across patients. Macro-averaging across patients is what gets
reported in the LGG segmentation literature.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from monai.metrics import compute_hausdorff_distance
    HAS_MONAI = True
except Exception:
    HAS_MONAI = False

from .augmentations import eval_transform
from .dataset import (
    LGGSegmentationDataset,
    discover_slices,
    patient_level_split,
    verify_no_patient_leakage,
)
from .models import build_model
from .postprocess import postprocess_mask


EPS = 1e-6


def slice_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Binary metrics for a single 2D slice. Both arrays are uint8 0/1."""
    pred_b = pred.astype(bool)
    targ_b = target.astype(bool)

    tp = np.logical_and(pred_b, targ_b).sum()
    fp = np.logical_and(pred_b, ~targ_b).sum()
    fn = np.logical_and(~pred_b, targ_b).sum()
    tn = np.logical_and(~pred_b, ~targ_b).sum()

    has_gt = targ_b.any()
    has_pred = pred_b.any()

    if has_gt:
        dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
        iou = (tp + EPS) / (tp + fp + fn + EPS)
        sens = tp / (tp + fn + EPS)
    else:
        # No tumor: by convention Dice=1 if also no prediction, else 0
        dice = 1.0 if not has_pred else 0.0
        iou = 1.0 if not has_pred else 0.0
        sens = float("nan")

    spec = tn / (tn + fp + EPS)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sens) if not np.isnan(sens) else None,
        "specificity": float(spec),
        "has_gt": bool(has_gt),
        "has_pred": bool(has_pred),
    }


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float | None:
    """Symmetric Hausdorff in pixels. None if either mask is empty or monai missing."""
    if not HAS_MONAI:
        return None
    if pred.sum() == 0 or target.sum() == 0:
        return None
    pred_t = torch.from_numpy(pred).bool().unsqueeze(0).unsqueeze(0)
    targ_t = torch.from_numpy(target).bool().unsqueeze(0).unsqueeze(0)
    hd = compute_hausdorff_distance(pred_t, targ_t, include_background=False, percentile=95)
    return float(hd.item())


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    min_area_px: int = 50,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="evaluate"):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].cpu().numpy()
        patient_ids = batch["patient_id"]
        slice_idxs = batch["slice_idx"].cpu().numpy() if torch.is_tensor(batch["slice_idx"]) else batch["slice_idx"]

        logits = model(image)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(np.uint8)

        for i in range(preds.shape[0]):
            p = preds[i, 0]
            t = mask[i, 0].astype(np.uint8)
            p = postprocess_mask(p, min_area_px=min_area_px)
            m = slice_metrics(p, t)
            m["hausdorff"] = hausdorff_distance(p, t)
            m["patient_id"] = patient_ids[i] if isinstance(patient_ids, list) else patient_ids[i]
            m["slice_idx"] = int(slice_idxs[i])
            rows.append(m)
    return pd.DataFrame(rows)


def aggregate_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per patient, ignoring NaN (e.g. sensitivity on empty-GT slices)."""
    return (
        df.groupby("patient_id")
        .agg(
            dice=("dice", "mean"),
            iou=("iou", "mean"),
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
            hausdorff=("hausdorff", "mean"),
            n_slices=("dice", "size"),
            n_tumor_slices=("has_gt", "sum"),
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area-px", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    records = discover_slices(args.root)
    splits = patient_level_split(records, seed=args.seed)
    verify_no_patient_leakage(splits)
    eval_recs = splits[args.split]

    ds = LGGSegmentationDataset(eval_recs, transform=eval_transform(), image_size=256)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    df_slice = evaluate_model(model, loader, device,
                              threshold=args.threshold, min_area_px=args.min_area_px)

    df_patient = aggregate_per_patient(df_slice)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem
    df_slice.to_csv(results_dir / f"{stem}_{args.split}_slice.csv", index=False)
    df_patient.to_csv(results_dir / f"{stem}_{args.split}_patient.csv", index=False)

    summary = {
        "split": args.split,
        "n_patients": int(df_patient.shape[0]),
        "n_slices": int(df_slice.shape[0]),
        "dice_mean": float(df_patient["dice"].mean()),
        "dice_median": float(df_patient["dice"].median()),
        "dice_std": float(df_patient["dice"].std()),
        "iou_mean": float(df_patient["iou"].mean()),
        "sensitivity_mean": float(df_patient["sensitivity"].mean(skipna=True)),
        "specificity_mean": float(df_patient["specificity"].mean()),
        "hausdorff_mean": (
            float(df_patient["hausdorff"].mean(skipna=True))
            if df_patient["hausdorff"].notna().any() else None
        ),
    }
    summary_path = results_dir / f"{stem}_{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"per-slice -> {results_dir / f'{stem}_{args.split}_slice.csv'}")
    print(f"per-patient -> {results_dir / f'{stem}_{args.split}_patient.csv'}")


if __name__ == "__main__":
    main()
