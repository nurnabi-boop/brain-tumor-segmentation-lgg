# Brain Tumor Segmentation (LGG MRI)

U-Net family applied to the [LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
by Mateusz Buda — 3,929 brain MRI slices (FLAIR + manual mask) from 110 low-grade
glioma patients in TCIA. Compares U-Net / U-Net++ / DeepLabV3+ on a strict
**patient-level** split.

## Why patient-level splitting matters
Slices from one patient are highly correlated. Random slice-level splitting
inflates Dice by 5–10 points because train and test see neighboring slices of
the same lesion. Every split here is keyed on the patient folder ID; an
assertion in `dataset.py` refuses to return a leaky split.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Download data (≈800 MB)
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data --unzip

# 2. Sanity-check the split (no patient appears in two splits)
python -m src.dataset --root data/lgg-mri-segmentation/kaggle_3m

# 3. Train baseline
python -m src.train \
    --root data/lgg-mri-segmentation/kaggle_3m \
    --model unet_resnet34 \
    --epochs 40 --batch-size 16

# 4. Evaluate on the held-out test set
python -m src.evaluate \
    --root data/lgg-mri-segmentation/kaggle_3m \
    --checkpoint models/unet_resnet34_best.pt \
    --split test

# 5. Launch demo
python app.py --checkpoint models/unet_resnet34_best.pt
```

## Methodology
| Step | Choice |
| --- | --- |
| Pairing | FLAIR slice ↔ binary mask via `_mask.tif` suffix |
| Training set | tumor-only slices (empty masks dropped, configurable) |
| Eval set | all slices, including empty (Dice=1 on empty if no FP) |
| Split | 70 / 15 / 15 by patient ID, seeded |
| Loss | `0.5 * BCE + 0.5 * Dice` |
| Augmentation | flip, ±20° rotate, scale, elastic, grid distortion, brightness, gaussian noise |
| Optimizer | AdamW, lr=1e-4, cosine schedule |
| Mixed precision | `torch.cuda.amp` |
| Post-processing | drop CCs <50 px (tunable on val) |

### Models
- `unet_resnet34` — U-Net + ResNet34 (ImageNet) ← baseline
- `unetpp_effb0` — U-Net++ + EfficientNet-B0
- `deeplabv3p_resnet34` — DeepLabV3+ + ResNet34

All built via `segmentation-models-pytorch` so the only thing varying between
runs is the architecture.

### Metrics (per patient, then averaged)
- Dice coefficient (primary)
- IoU
- 95th-percentile Hausdorff distance (via MONAI)
- Sensitivity / Specificity

## Project layout
```
brain-segmentation/
├── data/                       # Kaggle dataset extracted here
├── src/
│   ├── dataset.py              # patient-level split + Dataset
│   ├── augmentations.py        # albumentations pipelines
│   ├── models.py               # SMP factory
│   ├── losses.py               # Dice + BCE
│   ├── train.py                # training loop, W&B
│   ├── evaluate.py             # per-patient Dice/Hausdorff/etc.
│   └── postprocess.py          # connected-component cleanup
├── notebooks/
│   ├── eda.ipynb               # split sanity-check + 10 random samples
│   └── failure_analysis.ipynb  # worst-Dice patients
├── models/                     # checkpoints
├── results/                    # CSV + JSON metrics
├── app.py                      # Gradio demo
└── requirements.txt
```

## Validation order (recommended)
1. `python -m src.dataset --root <path>` → confirms no patient leakage.
2. Run `notebooks/eda.ipynb` → visual check of 10 random image+mask pairs.
3. Train U-Net + ResNet34 for ~5 epochs → confirm training Dice climbs.
4. Train the full encoder zoo for 40 epochs.
5. Evaluate on test, then explore failures in `failure_analysis.ipynb`.

## Research extensions (not implemented here, listed for follow-up)
- **3D vs 2D**: stack contiguous slices into volumes and use a 3D U-Net
  (`monai.networks.nets.UNet`). Trade-off: better through-plane context but
  far fewer training samples and ~10× memory.
- **Semi-supervised segmentation**: most clinical datasets have far more
  unlabeled scans than labeled ones. Mean Teacher / FixMatch on the 110
  labeled patients plus unlabeled TCIA volumes is a natural extension.
- **Uncertainty estimation**: enable dropout at inference (Monte Carlo dropout)
  and report per-pixel predictive variance. The variance map flags ambiguous
  tumor borders for clinician review — useful for decision support, not
  autonomous use.

## Disclaimer
Research code. Not validated for clinical use. The Gradio demo is a sanity
tool, not a medical device.
