#!/usr/bin/env python3
"""
Minimal Gradio demo for Community-Forensics final decision pipeline.

Decision rule:
    sigmoid(logit / temperature) >= threshold
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "results" / "final_decision_config.json"
COMMFOR_REPO = os.environ.get("COMMFOR_REPO", "/root/Community-Forensics")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_final_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing config file: {CONFIG_PATH}. "
            "Run adaptive calibration first to generate final config."
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_model(model_repo: str):
    if not Path(COMMFOR_REPO).exists():
        raise FileNotFoundError(
            f"COMMFOR_REPO not found: {COMMFOR_REPO}. "
            "Set COMMFOR_REPO to your Community-Forensics path."
        )
    sys.path.insert(0, COMMFOR_REPO)
    import models  # pylint: disable=import-error

    model = models.ViTClassifier(
        model_size="small",
        input_size=224,
        patch_size=16,
        freeze_backbone=False,
        device=0 if DEVICE == "cuda" else "cpu",
        dtype=torch.float32,
    )
    model = model.from_pretrained(model_repo)
    model = model.to(DEVICE)
    model.eval()
    return model


TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


CFG = load_final_config()
MODEL = load_model(CFG["model_repo"])
TEMPERATURE = float(CFG["temperature"])
THRESHOLD = float(CFG["threshold"])


def predict(image: Image.Image):
    if image is None:
        return "Please upload an image.", {}
    image = image.convert("RGB")
    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = float(MODEL(x).squeeze(0).item())
    calibrated_prob = float(torch.sigmoid(torch.tensor(logit / TEMPERATURE)).item())
    is_ai = calibrated_prob >= THRESHOLD
    label = "AI-generated" if is_ai else "Real"
    confidence = calibrated_prob if is_ai else (1.0 - calibrated_prob)
    summary = (
        f"### Prediction: **{label}**\n"
        f"- Calibrated probability (AI): `{calibrated_prob:.4f}`\n"
        f"- Threshold: `{THRESHOLD:.2f}`\n"
        f"- Confidence: `{confidence:.4f}`\n"
        f"- Device: `{DEVICE}`"
    )
    details = {
        "label": label,
        "logit": logit,
        "temperature": TEMPERATURE,
        "threshold": THRESHOLD,
        "calibrated_prob_ai": calibrated_prob,
        "confidence": confidence,
        "decision_rule": "sigmoid(logit / temperature) >= threshold",
        "model_repo": CFG["model_repo"],
    }
    return summary, details


def build_demo():
    desc = (
        "Upload one image to run Community-Forensics inference with the finalized "
        "adaptive-threshold + temperature-calibrated decision rule."
    )
    with gr.Blocks(title="AI Image Detector Demo") as demo:
        gr.Markdown("# AI-generated Image Detector Demo")
        gr.Markdown(desc)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input Image")
                btn = gr.Button("Run Detection", variant="primary")
            with gr.Column():
                out_summary = gr.Markdown(label="Summary")
                out_json = gr.JSON(label="Details")
        btn.click(fn=predict, inputs=[inp], outputs=[out_summary, out_json])
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
