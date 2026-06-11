"""Training loop for binary tumor segmentation.

Usage:
    python -m src.train \
        --root data/lgg-mri-segmentation/kaggle_3m \
        --model unet_resnet34 \
        --epochs 40 \
        --batch-size 16

Logs to W&B if WANDB_API_KEY is set; otherwise --no-wandb keeps it offline.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augmentations import eval_transform, train_transform
from .dataset import (
    LGGSegmentationDataset,
    discover_slices,
    filter_tumor_only,
    patient_level_split,
    verify_no_patient_leakage,
)
from .losses import DiceBCELoss, dice_score, iou_score
from .models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to kaggle_3m directory")
    p.add_argument("--model", default="unet_resnet34")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="models")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="brain-segmentation")
    p.add_argument("--run-name", default=None)
    p.add_argument("--mixed-precision", action="store_true", default=True)
    p.add_argument("--include-empty-train", action="store_true",
                   help="Keep empty-mask slices in train (default: drop)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace):
    records = discover_slices(args.root)
    splits = patient_level_split(records, seed=args.seed)
    verify_no_patient_leakage(splits)

    train_recs = splits["train"] if args.include_empty_train else filter_tumor_only(splits["train"])
    val_recs = splits["val"]
    test_recs = splits["test"]

    print(f"train: {len(train_recs)} slices / {len({r.patient_id for r in train_recs})} patients")
    print(f"val:   {len(val_recs)} slices / {len({r.patient_id for r in val_recs})} patients")
    print(f"test:  {len(test_recs)} slices / {len({r.patient_id for r in test_recs})} patients")

    train_ds = LGGSegmentationDataset(train_recs, transform=train_transform(args.image_size), image_size=args.image_size)
    val_ds = LGGSegmentationDataset(val_recs, transform=eval_transform(args.image_size), image_size=args.image_size)
    test_ds = LGGSegmentationDataset(test_recs, transform=eval_transform(args.image_size), image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, splits


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> dict:
    model.train()
    losses, dices = [], []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(image)
            loss = criterion(logits, mask)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        dices.extend(dice_score(logits.detach().float(), mask).cpu().tolist())
        pbar.set_postfix(loss=f"{np.mean(losses):.4f}", dice=f"{np.mean(dices):.4f}")
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices))}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    losses, dices, ious = [], [], []
    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(image)
        losses.append(criterion(logits, mask).item())
        dices.extend(dice_score(logits.float(), mask).cpu().tolist())
        ious.extend(iou_score(logits.float(), mask).cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, splits = make_loaders(args)

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"model: {args.model} | trainable params: {n_params/1e6:.2f}M")

    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.run_name or f"{args.model}-{int(time.time())}",
                config=vars(args) | {"params_m": n_params / 1e6},
            )
        except Exception as e:
            print(f"[wandb disabled] {e}")
            run = None

    best_dice = -1.0
    history = []
    ckpt_path = output_dir / f"{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "epoch_time_s": time.time() - t0,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {record['train_loss']:.4f} dice {record['train_dice']:.4f} | "
            f"val loss {record['val_loss']:.4f} dice {record['val_dice']:.4f} iou {record['val_iou']:.4f} | "
            f"{record['epoch_time_s']:.1f}s"
        )

        if run is not None:
            run.log(record)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "epoch": epoch,
                    "val_dice": best_dice,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> saved best (dice={best_dice:.4f}) to {ckpt_path}")

    history_path = results_dir / f"{args.model}_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"history: {history_path}")
    print(f"best val dice: {best_dice:.4f}")

    if run is not None:
        run.summary["best_val_dice"] = best_dice
        run.finish()


if __name__ == "__main__":
    main()
