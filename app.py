"""Gradio demo: upload a brain MRI slice, see predicted tumor mask overlaid.

Launch:
    python app.py --checkpoint models/unet_resnet34_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from src.augmentations import eval_transform
from src.models import build_model
from src.postprocess import postprocess_mask


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: torch.nn.Module | None = None
TRANSFORM = eval_transform(image_size=256)


def load_model(checkpoint_path: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_model(ckpt["model_name"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha: float = 0.45) -> np.ndarray:
    """Blend a binary mask onto an RGB image."""
    out = image_rgb.copy()
    color_arr = np.array(color, dtype=np.uint8)
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * color_arr).astype(np.uint8)

    # outline the mask for clarity
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 1)
    return out


@torch.no_grad()
def predict(image: np.ndarray, threshold: float, min_area: int) -> tuple[np.ndarray, np.ndarray, str]:
    if MODEL is None:
        raise RuntimeError("Model not loaded. Pass --checkpoint at launch.")
    if image is None:
        return None, None, "Upload an MRI slice."

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    img256 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    out = TRANSFORM(image=img256, mask=np.zeros(img256.shape[:2], np.uint8))
    x = out["image"].unsqueeze(0).to(DEVICE)

    logits = MODEL(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    raw_mask = (prob > threshold).astype(np.uint8)
    clean_mask = postprocess_mask(raw_mask, min_area_px=int(min_area))

    overlay = overlay_mask(img256, clean_mask)
    pred_area_pct = clean_mask.mean() * 100
    summary = (
        f"Predicted tumor area: {pred_area_pct:.2f}% of slice  |  "
        f"max prob: {prob.max():.3f}  |  threshold: {threshold:.2f}  |  min CC area: {int(min_area)} px"
    )
    return overlay, (clean_mask * 255).astype(np.uint8), summary


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="LGG Brain Tumor Segmentation") as demo:
        gr.Markdown(
            "# LGG Brain Tumor Segmentation\n"
            "Upload a FLAIR brain MRI slice. The model returns the predicted tumor mask "
            "overlaid on the input. Trained on the LGG MRI dataset (Buda et al.) with patient-level splits."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Input MRI slice (FLAIR)", type="numpy")
                threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Probability threshold")
                min_area = gr.Slider(0, 500, value=50, step=10, label="Min connected-component area (px)")
                btn = gr.Button("Segment", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Overlay")
                out_mask = gr.Image(label="Predicted mask")
                out_text = gr.Textbox(label="Summary", lines=2)
        btn.click(predict, inputs=[inp, threshold, min_area], outputs=[out_img, out_mask, out_text])
        gr.Markdown(
            "_Research demo only — not a medical device. The model has not been validated for clinical use._"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/unet_resnet34_best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    global MODEL
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first via src/train.py.")
    MODEL = load_model(str(ckpt_path))
    print(f"Loaded {ckpt_path} on {DEVICE}")

    demo = build_ui()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
