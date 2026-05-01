#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = (
    "cuda"
    if torch.cuda.is_available() and (getattr(torch.version, "cuda", None) is not None)
    else "cpu"
)
INPUT_SIZE = 224


@dataclass
class ScoredFile:
    path: Path
    gt: int
    pred: int
    prob_ai: float
    class_conf: float
    is_correct: bool


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
    model = model.from_pretrained(model_repo, device="cpu")
    return model.to(DEVICE).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter picked samples by confidence")
    parser.add_argument(
        "--samples-root",
        default=str(PROJECT_ROOT / "results" / "official_samples"),
        help="Root containing picked_gt0_100 and picked_gt1_100",
    )
    parser.add_argument(
        "--model-repo",
        default=os.environ.get("HF_MODEL_REPO", "OwensLab/commfor-model-224"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.1098015308380127,
        help="Temperature used in sigmoid(logit/temperature)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.24,
        help="Prediction threshold",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--real-lo", type=float, default=0.76)
    parser.add_argument("--real-hi", type=float, default=0.85)
    parser.add_argument("--fake-lo", type=float, default=0.24)
    parser.add_argument("--fake-hi", type=float, default=0.4)
    return parser.parse_args()


def file_list(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file()])


def score_file(model, tfm, img_path: Path, gt: int, temperature: float, threshold: float) -> ScoredFile:
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = float(model(x).squeeze(0).item())
    prob_ai = float(torch.sigmoid(torch.tensor(logit / temperature)).item())
    pred = 1 if prob_ai >= threshold else 0
    class_conf = prob_ai if gt == 1 else (1.0 - prob_ai)
    return ScoredFile(
        path=img_path,
        gt=gt,
        pred=pred,
        prob_ai=prob_ai,
        class_conf=class_conf,
        is_correct=(pred == gt),
    )


def keep_and_cleanup(
    scored: List[ScoredFile],
    topk: int,
    lo: float,
    hi: float,
) -> tuple[int, int]:
    scored_sorted = sorted(scored, key=lambda s: s.class_conf, reverse=True)
    top = scored_sorted[:topk]
    mid = [s for s in scored if s.is_correct and lo <= s.class_conf <= hi]
    keep_set: Set[Path] = {s.path for s in top} | {s.path for s in mid}

    # Rename kept files to include confidence and correctness metadata.
    for s in sorted((x for x in scored if x.path in keep_set), key=lambda x: x.path.name):
        conf = f"{s.class_conf:.4f}"
        status = "correct" if s.is_correct else "wrong"
        new_name = f"{s.path.stem}__conf{conf}__pred{s.pred}__{status}{s.path.suffix}"
        new_path = s.path.with_name(new_name)
        if new_path != s.path:
            os.replace(s.path, new_path)

    deleted = 0
    for s in scored:
        if s.path not in keep_set and s.path.exists():
            s.path.unlink()
            deleted += 1
    return len(keep_set), deleted


def main():
    args = parse_args()
    root = Path(args.samples_root)
    d0 = root / "picked_gt0_100"
    d1 = root / "picked_gt1_100"
    if not d0.exists() or not d1.exists():
        raise FileNotFoundError(f"Missing picked directories under: {root}")

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = load_model(args.model_repo)

    scored0 = [score_file(model, tfm, p, 0, args.temperature, args.threshold) for p in file_list(d0)]
    scored1 = [score_file(model, tfm, p, 1, args.temperature, args.threshold) for p in file_list(d1)]

    keep0, del0 = keep_and_cleanup(scored0, args.topk, args.real_lo, args.real_hi)
    keep1, del1 = keep_and_cleanup(scored1, args.topk, args.fake_lo, args.fake_hi)

    print("Done.")
    print(f"picked_gt0_100: kept={keep0}, deleted={del0}")
    print(f"picked_gt1_100: kept={keep1}, deleted={del1}")


if __name__ == "__main__":
    main()
