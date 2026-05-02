import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import timm
import torch
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SEED = 0
EPOCHS = 8
BATCH_SIZE = 16
LR = 5e-5

STYLE_TRAIN_PER_CLASS = 2000
CIFAKE_TRAIN_PER_CLASS = 2000
STYLE_VAL_PER_CLASS = 400
CIFAKE_VAL_PER_CLASS = 400

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class MixedImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, domain = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), torch.tensor(domain, dtype=torch.long)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_images(folder: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ],
        key=lambda x: str(x).lower(),
    )


def sample_paths(paths: List[Path], n: int, seed: int) -> List[Path]:
    rng = random.Random(seed)
    picked = list(paths)
    rng.shuffle(picked)
    if len(picked) < n:
        print(f"⚠ 样本不足 {n}，当前 {len(picked)}，将全部使用")
    return picked[: min(n, len(picked))]


def inspect_class_mapping(root: Path, tag: str):
    if not root.exists():
        raise FileNotFoundError(f"{tag} 路径不存在: {root}")
    ds = datasets.ImageFolder(root)
    print(f"[{tag}] class_to_idx = {ds.class_to_idx}")


def build_samples(style_train_root: Path, style_valid_root: Path, cifake_train_root: Path, cifake_test_root: Path):
    print(f"StyleGAN train 子目录: {os.listdir(style_train_root)}")
    print(f"StyleGAN valid 子目录: {os.listdir(style_valid_root)}")
    print(f"CIFAKE train 子目录: {os.listdir(cifake_train_root)}")
    print(f"CIFAKE test 子目录: {os.listdir(cifake_test_root)}")

    inspect_class_mapping(style_train_root, "StyleGAN Train")
    inspect_class_mapping(cifake_train_root, "CIFAKE Train")

    style_real = collect_images(style_train_root / "real")
    style_fake = collect_images(style_train_root / "fake")
    cifake_real = collect_images(cifake_train_root / "REAL")
    cifake_fake = collect_images(cifake_train_root / "FAKE")

    train_style_real = sample_paths(style_real, STYLE_TRAIN_PER_CLASS, SEED)
    train_style_fake = sample_paths(style_fake, STYLE_TRAIN_PER_CLASS, SEED)
    train_cifake_real = sample_paths(cifake_real, CIFAKE_TRAIN_PER_CLASS, SEED)
    train_cifake_fake = sample_paths(cifake_fake, CIFAKE_TRAIN_PER_CLASS, SEED)

    # label: fake=0, real=1; domain: 0=StyleGAN, 1=CIFAKE
    train_samples = (
        [(p, 1, 0) for p in train_style_real]
        + [(p, 0, 0) for p in train_style_fake]
        + [(p, 1, 1) for p in train_cifake_real]
        + [(p, 0, 1) for p in train_cifake_fake]
    )

    style_valid_real = collect_images(style_valid_root / "real")
    style_valid_fake = collect_images(style_valid_root / "fake")
    cifake_test_real = collect_images(cifake_test_root / "REAL")
    cifake_test_fake = collect_images(cifake_test_root / "FAKE")

    val_style_real = sample_paths(style_valid_real, STYLE_VAL_PER_CLASS, SEED)
    val_style_fake = sample_paths(style_valid_fake, STYLE_VAL_PER_CLASS, SEED)
    val_cifake_real = sample_paths(cifake_test_real, CIFAKE_VAL_PER_CLASS, SEED)
    val_cifake_fake = sample_paths(cifake_test_fake, CIFAKE_VAL_PER_CLASS, SEED)

    val_samples = (
        [(p, 1, 0) for p in val_style_real]
        + [(p, 0, 0) for p in val_style_fake]
        + [(p, 1, 1) for p in val_cifake_real]
        + [(p, 0, 1) for p in val_cifake_fake]
    )

    rng = random.Random(SEED)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    print(
        "训练集构成: "
        f"StyleGAN REAL {len(train_style_real)}, StyleGAN FAKE {len(train_style_fake)}, "
        f"CIFAKE REAL {len(train_cifake_real)}, CIFAKE FAKE {len(train_cifake_fake)}"
    )
    print(
        "验证集构成: "
        f"StyleGAN REAL {len(val_style_real)}, StyleGAN FAKE {len(val_style_fake)}, "
        f"CIFAKE REAL {len(val_cifake_real)}, CIFAKE FAKE {len(val_cifake_fake)}"
    )

    return train_samples, val_samples


def build_loaders(train_samples, val_samples, device: torch.device):
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    train_ds = MixedImageDataset(train_samples, train_tf)
    val_ds = MixedImageDataset(val_samples, val_tf)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()

    total_correct, total_count = 0, 0
    style_correct, style_count = 0, 0
    cifake_correct, cifake_count = 0, 0

    with torch.no_grad():
        for imgs, labels, domains in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            domains = domains.to(device)

            logits = model(imgs).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = preds == labels

            total_correct += correct.sum().item()
            total_count += labels.size(0)

            style_mask = domains == 0
            cifake_mask = domains == 1

            if style_mask.any():
                style_correct += correct[style_mask].sum().item()
                style_count += style_mask.sum().item()
            if cifake_mask.any():
                cifake_correct += correct[cifake_mask].sum().item()
                cifake_count += cifake_mask.sum().item()

    mixed_acc = total_correct / total_count if total_count else 0.0
    style_acc = style_correct / style_count if style_count else 0.0
    cifake_acc = cifake_correct / cifake_count if cifake_count else 0.0
    return mixed_acc, style_acc, cifake_acc


def main():
    set_seed(SEED)
    project_root = Path(__file__).resolve().parent

    style_train_root = project_root / "data" / "real_vs_fake" / "real-vs-fake" / "train"
    style_valid_root = project_root / "data" / "real_vs_fake" / "real-vs-fake" / "valid"
    cifake_train_root = project_root / "data" / "train"
    cifake_test_root = project_root / "data" / "test"

    start_weight = project_root / "weights" / "efficientnet_b7_deepfake.pth"
    out_weight = project_root / "weights" / "efficientnet_b7_mixed.pth"

    for p in [style_train_root, style_valid_root, cifake_train_root, cifake_test_root]:
        if not p.exists():
            raise FileNotFoundError(f"缺少路径: {p}")
    if not start_weight.exists():
        raise FileNotFoundError(f"初始权重不存在: {start_weight}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠️ 未检测到 CUDA，使用 CPU 训练，预计时间较长")
    print(f"训练设备: {device}")

    train_samples, val_samples = build_samples(
        style_train_root, style_valid_root, cifake_train_root, cifake_test_root
    )
    train_loader, val_loader = build_loaders(train_samples, val_samples, device)

    model = timm.create_model("tf_efficientnet_b7", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(start_weight, map_location=device))
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    best_mixed = 0.0
    out_weight.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for imgs, labels, _domains in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

        scheduler.step()

        epoch_loss = total_loss / total_count if total_count else 0.0
        mixed_acc, style_acc, cifake_acc = evaluate(model, val_loader, device)

        if mixed_acc > best_mixed:
            best_mixed = mixed_acc
            torch.save(model.state_dict(), out_weight)

        print(
            f"[Epoch {epoch}/{EPOCHS}] Loss: {epoch_loss:.4f} | "
            f"Mixed Val: {mixed_acc*100:.2f}% | "
            f"StyleGAN Val: {style_acc*100:.2f}% | "
            f"CIFAKE Val: {cifake_acc*100:.2f}%"
        )

    print(f"\n训练完成，最佳 Mixed Val: {best_mixed*100:.2f}%")
    print(f"权重已保存: {out_weight}")


if __name__ == "__main__":
    main()

