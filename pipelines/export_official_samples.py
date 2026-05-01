#!/usr/bin/env python3
"""
Stream official dataset and export:
1) first-hit gt0/gt1 samples (default 100 each)
2) from those 200 images, confidence top-K for real/fake (default 5 each)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import PIL.Image
import torch
from datasets import load_dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 224


def resolve_commfor_repo() -> str:
    env_path = os.environ.get("COMMFOR_REPO")
    if env_path:
        return env_path
    candidates = [
        PROJECT_ROOT / "Community-Forensics",
        PROJECT_ROOT.parent / "Community-Forensics",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


def load_model(model_repo: str):
    commfor_repo = resolve_commfor_repo()
    if not Path(commfor_repo).exists():
        raise FileNotFoundError(
            f"COMMFOR_REPO not found: {commfor_repo}. "
            "Set COMMFOR_REPO or place Community-Forensics beside this project."
        )
    sys.path.insert(0, commfor_repo)
    import models  # pylint: disable=import-error

    model = models.ViTClassifier(
        model_size="small",
        input_size=INPUT_SIZE,
        patch_size=16,
        freeze_backbone=False,
        device=0 if DEVICE == "cuda" else "cpu",
        dtype=torch.float32,
    )
    model = model.from_pretrained(model_repo)
    return model.to(DEVICE).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export streamed official samples.")
    parser.add_argument(
        "--dataset-repo",
        default=os.environ.get("HF_TEST_REPO", "OwensLab/CommunityForensics-Eval"),
    )
    parser.add_argument(
        "--split",
        default=os.environ.get("HF_TEST_SPLIT", "CompEval"),
    )
    parser.add_argument(
        "--model-repo",
        default=os.environ.get("HF_MODEL_REPO", "OwensLab/commfor-model-224"),
    )
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "official_samples"))
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Only export picked gt0/gt1 pools, skip confidence scoring/export.",
    )
    return parser.parse_args()


def save_jpg(image_bytes: bytes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(out_path, format="JPEG", quality=95)


def infer_confidence(model, tfm, image_bytes: bytes, temperature: float, gt: int):
    img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = float(model(x).squeeze(0).item())
    prob_ai = float(torch.sigmoid(torch.tensor(logit / temperature)).item())
    # gt=0(real) confidence is 1-prob_ai; gt=1(fake) confidence is prob_ai
    class_conf = prob_ai if gt == 1 else (1.0 - prob_ai)
    return logit, prob_ai, class_conf


def main():
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Streaming dataset: {args.dataset_repo} [{args.split}]")
    ds = load_dataset(args.dataset_repo, split=args.split, streaming=True)

    need = args.per_class
    gt0_samples: List[Dict] = []
    gt1_samples: List[Dict] = []
    seen0 = 0
    seen1 = 0
    total = 0

    for row in ds:
        total += 1
        label = int(row["label"])
        sample = {
            "image_bytes": row["image_data"],
            "label": label,
            "image_name": row.get("image_name", f"row{total}"),
        }
        if label == 0:
            seen0 += 1
            if len(gt0_samples) < need:
                gt0_samples.append(sample)
        elif label == 1:
            seen1 += 1
            if len(gt1_samples) < need:
                gt1_samples.append(sample)
        if len(gt0_samples) >= need and len(gt1_samples) >= need:
            break

    if len(gt0_samples) < need or len(gt1_samples) < need:
        raise RuntimeError(
            f"Insufficient samples: gt0={len(gt0_samples)} gt1={len(gt1_samples)} "
            f"(need each={need})"
        )

    out_root = Path(args.output_dir)
    random_gt0_dir = out_root / "picked_gt0_100"
    random_gt1_dir = out_root / "picked_gt1_100"
    random_gt0_dir.mkdir(parents=True, exist_ok=True)
    random_gt1_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(gt0_samples, 1):
        fname = f"{i:03d}_picked_gt0_{s['image_name']}.jpg"
        save_jpg(s["image_bytes"], random_gt0_dir / fname)
    for i, s in enumerate(gt1_samples, 1):
        fname = f"{i:03d}_picked_gt1_{s['image_name']}.jpg"
        save_jpg(s["image_bytes"], random_gt1_dir / fname)

    print(f"Picked first-hit set from stream. seen gt0={seen0}, gt1={seen1}, total={total}")
    if args.skip_scoring:
        print("Skip scoring enabled. Export finished with picked_gt0_100 and picked_gt1_100 only.")
        return
    print("Scoring confidence on sampled 200 images ...")

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = load_model(args.model_repo)

    scored_real: List[Dict] = []
    scored_fake: List[Dict] = []

    for s in gt0_samples:
        logit, prob_ai, conf = infer_confidence(model, tfm, s["image_bytes"], args.temperature, 0)
        scored_real.append({**s, "logit": logit, "prob_ai": prob_ai, "class_conf": conf})
    for s in gt1_samples:
        logit, prob_ai, conf = infer_confidence(model, tfm, s["image_bytes"], args.temperature, 1)
        scored_fake.append({**s, "logit": logit, "prob_ai": prob_ai, "class_conf": conf})

    scored_real = sorted(scored_real, key=lambda x: x["class_conf"], reverse=True)
    scored_fake = sorted(scored_fake, key=lambda x: x["class_conf"], reverse=True)
    topk = args.topk
    top_real = scored_real[:topk]
    top_fake = scored_fake[:topk]

    top_real_dir = out_root / "conf_top5_real"
    top_fake_dir = out_root / "conf_top5_fake"
    top_real_dir.mkdir(parents=True, exist_ok=True)
    top_fake_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(top_real, 1):
        fname = f"{i:03d}_top_real_conf{s['class_conf']:.4f}_{s['image_name']}.jpg"
        save_jpg(s["image_bytes"], top_real_dir / fname)
    for i, s in enumerate(top_fake, 1):
        fname = f"{i:03d}_top_fake_conf{s['class_conf']:.4f}_{s['image_name']}.jpg"
        save_jpg(s["image_bytes"], top_fake_dir / fname)

    print("Export complete:")
    print(f"  - {random_gt0_dir}")
    print(f"  - {random_gt1_dir}")
    print(f"  - {top_real_dir}")
    print(f"  - {top_fake_dir}")


if __name__ == "__main__":
    main()
