#!/usr/bin/env python3
"""
Standalone zero-shot evaluation using locally-cached parquet files.
Avoids HF dataset download by loading parquet files directly with PyArrow.
"""
import sys, os, io, glob, time
import torch
import numpy as np
import pyarrow.parquet as pq
import PIL.Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)

COMMFOR_REPO = os.environ.get("COMMFOR_REPO", "/workspaces/Community-Forensics")
DATA_DIR = os.environ.get(
    "EVAL_DATA_DIR",
    "/workspaces/AI-generated-image-detection/.hf_cache/hub/"
    "datasets--OwensLab--CommunityForensics-Eval/snapshots/"
    "7d4a74a88d2cac93b513c0853bf92c260eaceea0/data",
)
MODEL_REPO  = os.environ.get("HF_MODEL_REPO", "OwensLab/commfor-model-224")
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE_PER_GPU", 32))
INPUT_SIZE  = 224
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

sys.path.insert(0, COMMFOR_REPO)
import models  # ViTClassifier


# ---------- dataset ----------------------------------------------------------

class ParquetEvalDataset(Dataset):
    """Memory-efficient dataset: loads one parquet file's table at a time."""

    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        # Build a sorted list of (file_path, n_rows) and a flat index
        pq_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        self._file_ranges = []   # [(start_idx, end_idx, file_path), ...]
        total = 0
        for fp in pq_files:
            meta = pq.read_metadata(fp)
            n = meta.num_rows
            if n == 0:
                continue
            self._file_ranges.append((total, total + n, fp))
            total += n
        self._total = total
        self._cache = {}   # filepath -> pyarrow Table (LRU-1)

    def __len__(self):
        return self._total

    def _get_table(self, fp):
        if fp not in self._cache:
            self._cache.clear()
            self._cache[fp] = pq.read_table(
                fp, columns=["image_data", "label", "model_name"]
            )
        return self._cache[fp]

    def __getitem__(self, idx):
        # Binary search for the file that contains idx
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
        label       = int(table.column("label")[row_idx].as_py())
        model_name  = str(table.column("model_name")[row_idx].as_py())

        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, model_name


# ---------- transform --------------------------------------------------------

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


# ---------- main -------------------------------------------------------------

def main():
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}")

    # --- model ---
    print(f"Loading model from {MODEL_REPO} ...")
    model = models.ViTClassifier(
        model_size="small",
        input_size=INPUT_SIZE,
        patch_size=16,
        freeze_backbone=False,
        device=0 if DEVICE == "cuda" else "cpu",
        dtype=torch.float32,
    )
    model = model.from_pretrained(MODEL_REPO)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    # --- data ---
    print(f"Building dataset from {DATA_DIR} ...")
    dataset = ParquetEvalDataset(DATA_DIR, transform=eval_transform)
    print(f"Dataset size: {len(dataset)} examples")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    # --- inference ---
    all_labels, all_probs, all_model_names = [], [], []
    t0 = time.time()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0

    with torch.no_grad():
        for i, (imgs, labels, model_names) in enumerate(loader):
            imgs   = imgs.to(DEVICE)
            labels_dev = labels.float().unsqueeze(1).to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels_dev)
            total_loss += loss.item()
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_model_names.extend(model_names)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{i+1}/{len(loader)}] "
                    f"loss={loss.item():.4f}  "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed_total = time.time() - t0

    # --- metrics ---
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    avg_loss = total_loss / len(loader)

    print("\n" + "="*50)
    print("=== Zero-Shot Evaluation Results ===")
    print(f"  Examples evaluated : {len(y_true)}")
    print(f"  Loss               : {avg_loss:.4f}")
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision          : {prec:.4f}")
    print(f"  Recall             : {rec:.4f}")
    print(f"  F1                 : {f1:.4f}")
    print(f"  ROC-AUC            : {roc_auc:.4f}")
    print(f"  Avg Precision (AP) : {avg_prec:.4f}")
    print(f"  Elapsed            : {elapsed_total:.1f}s")
    print("="*50)

    # --- per-generator breakdown ---
    unique_models = sorted(set(all_model_names))
    print("\n--- Per-generator breakdown ---")
    for gen in unique_models:
        idx = [i for i, m in enumerate(all_model_names) if m == gen]
        g_true = y_true[idx]
        g_prob = y_prob[idx]
        g_pred = y_pred[idx]
        g_acc  = accuracy_score(g_true, g_pred)
        try:
            g_auc = roc_auc_score(g_true, g_prob)
        except Exception:
            g_auc = float("nan")
        print(f"  {gen:40s}  n={len(idx):5d}  acc={g_acc:.3f}  auc={g_auc:.3f}")


if __name__ == "__main__":
    main()
