#!/usr/bin/env python3
"""
Adaptive-threshold + temperature-calibration evaluation for Community-Forensics.

This script:
1) loads the same zero-shot model used by baseline;
2) runs inference once on the full eval parquet set;
3) splits predictions into val/test;
4) learns:
   - temperature for probability calibration (on val)
   - optimal decision threshold (on val, max F1);
5) reports baseline vs improved test metrics and writes final config artifact.
"""

from __future__ import annotations

import glob
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import PIL.Image
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMFOR_REPO = os.environ.get(
    "COMMFOR_REPO",
    str((PROJECT_ROOT.parent / "Community-Forensics").resolve()),
)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "OwensLab/commfor-model-224")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE_PER_GPU", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
INPUT_SIZE = 224
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))
SEED = int(os.environ.get("SPLIT_SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_HOME = os.environ.get("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
RESULT_DIR = Path(os.environ.get("RESULT_DIR", str(PROJECT_ROOT / "results")))


def _discover_data_dir() -> str:
    """Pick latest snapshot data dir under HF cache."""
    ds_root = Path(HF_HOME) / "hub" / "datasets--OwensLab--CommunityForensics-Eval" / "snapshots"
    candidates = sorted(glob.glob(str(ds_root / "*" / "data")))
    if not candidates:
        raise FileNotFoundError(
            f"No cached eval parquet found under {ds_root}. "
            "Run zero-shot baseline first to populate cache."
        )
    return candidates[-1]


@dataclass
class EvalArrays:
    labels: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    model_names: List[str]


class ParquetEvalDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        pq_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        self._file_ranges = []
        total = 0
        for fp in pq_files:
            meta = pq.read_metadata(fp)
            n = meta.num_rows
            if n == 0:
                continue
            self._file_ranges.append((total, total + n, fp))
            total += n
        if total == 0:
            raise RuntimeError(f"No rows found in parquet shards at {data_dir}")
        self._total = total
        self._cache = {}

    def __len__(self):
        return self._total

    def _get_table(self, fp):
        if fp not in self._cache:
            self._cache.clear()
            self._cache[fp] = pq.read_table(fp, columns=["image_data", "label", "model_name"])
        return self._cache[fp]

    def __getitem__(self, idx):
        lo, hi = 0, len(self._file_ranges) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._file_ranges[mid][1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        start, _, fp = self._file_ranges[lo]
        row_idx = idx - start
        table = self._get_table(fp)

        image_bytes = table.column("image_data")[row_idx].as_py()
        label = int(table.column("label")[row_idx].as_py())
        model_name = str(table.column("model_name")[row_idx].as_py())
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, model_name


def build_model():
    import sys

    if not Path(COMMFOR_REPO).exists():
        raise FileNotFoundError(
            f"COMMFOR_REPO not found: {COMMFOR_REPO}. "
            "Set COMMFOR_REPO or place Community-Forensics beside this project."
        )
    sys.path.insert(0, COMMFOR_REPO)
    import models  # pylint: disable=import-error

    model = models.ViTClassifier(
        model_size="small",
        input_size=INPUT_SIZE,
        patch_size=16,
        freeze_backbone=False,
        device=0 if DEVICE == "cuda" else "cpu",
        dtype=torch.float32,
    )
    model = model.from_pretrained(HF_MODEL_REPO)
    return model.to(DEVICE).eval()


def infer(model, loader: DataLoader) -> EvalArrays:
    all_labels: List[int] = []
    all_logits: List[float] = []
    all_probs: List[float] = []
    all_model_names: List[str] = []
    with torch.no_grad():
        for imgs, labels, model_names in loader:
            logits = model(imgs.to(DEVICE)).squeeze(1).detach().cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_logits.extend(logits.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_model_names.extend(model_names)
    return EvalArrays(
        labels=np.array(all_labels, dtype=np.int64),
        logits=np.array(all_logits, dtype=np.float64),
        probs=np.array(all_probs, dtype=np.float64),
        model_names=all_model_names,
    )


def _binary_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / n)
    return float(ece)


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    logit_t = torch.tensor(logits_val, dtype=torch.float32)
    y_t = torch.tensor(y_val, dtype=torch.float32)
    log_temp = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = criterion(logit_t / temp, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp = float(torch.exp(log_temp).detach().cpu().item())
    return max(1e-3, min(temp, 100.0))


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece10": _binary_ece(y_true, y_prob, n_bins=10),
        "threshold": float(threshold),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = _discover_data_dir()
    print(f"DEVICE={DEVICE} BATCH={BATCH_SIZE} VAL_RATIO={VAL_RATIO} SEED={SEED}")
    print(f"Using parquet data: {data_dir}")

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ParquetEvalDataset(data_dir, transform=tfm)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    model = build_model()
    arrays = infer(model, loader)
    n = len(arrays.labels)
    print(f"Inference finished on {n} samples.")

    idx_all = np.arange(n)
    val_idx, test_idx = train_test_split(
        idx_all,
        test_size=(1.0 - VAL_RATIO),
        random_state=SEED,
        stratify=arrays.labels,
    )

    y_val = arrays.labels[val_idx]
    y_test = arrays.labels[test_idx]
    logit_val = arrays.logits[val_idx]
    logit_test = arrays.logits[test_idx]
    prob_val = arrays.probs[val_idx]
    prob_test = arrays.probs[test_idx]

    base_metrics = eval_metrics(y_test, prob_test, threshold=0.5)
    best_t_raw = find_best_threshold(y_val, prob_val)
    tuned_metrics = eval_metrics(y_test, prob_test, threshold=best_t_raw)

    temperature = fit_temperature(logit_val, y_val)
    prob_val_cal = 1.0 / (1.0 + np.exp(-(logit_val / temperature)))
    prob_test_cal = 1.0 / (1.0 + np.exp(-(logit_test / temperature)))
    best_t_cal = find_best_threshold(y_val, prob_val_cal)
    cal_metrics = eval_metrics(y_test, prob_test_cal, threshold=best_t_cal)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = RESULT_DIR / "final_decision_config.json"
    report_path = RESULT_DIR / "adaptive_calibration_report.json"
    out = {
        "model_repo": HF_MODEL_REPO,
        "data_dir": data_dir,
        "val_ratio": VAL_RATIO,
        "seed": SEED,
        "n_samples": int(n),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "temperature": temperature,
        "threshold_raw": best_t_raw,
        "threshold_calibrated": best_t_cal,
        "test_metrics_baseline_t05": base_metrics,
        "test_metrics_threshold_only": tuned_metrics,
        "test_metrics_threshold_plus_calibration": cal_metrics,
    }
    final_cfg = {
        "model_repo": HF_MODEL_REPO,
        "decision_type": "sigmoid(logit/temperature) >= threshold",
        "temperature": temperature,
        "threshold": best_t_cal,
        "fit_split": {"val_ratio": VAL_RATIO, "seed": SEED},
    }
    config_path.write_text(json.dumps(final_cfg, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Test Metrics (baseline t=0.5) ===")
    print(json.dumps(base_metrics, indent=2))
    print("\n=== Test Metrics (threshold only) ===")
    print(json.dumps(tuned_metrics, indent=2))
    print("\n=== Test Metrics (threshold + calibration) ===")
    print(json.dumps(cal_metrics, indent=2))
    print(f"\nSaved final config: {config_path}")
    print(f"Saved full report: {report_path}")


if __name__ == "__main__":
    main()
