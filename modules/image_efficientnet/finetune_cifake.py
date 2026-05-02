import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import timm
import torch
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SEED = 42
TRAIN_PER_CLASS = 3000
VAL_PER_CLASS = 500
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

FREEZE_PREFIXES = [
    "conv_stem",
    "bn1",
    "blocks.0",
    "blocks.1",
    "blocks.2",
    "blocks.3",
    "blocks.4",
]


class ImageListDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


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


def split_class_samples(paths: List[Path], seed: int):
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    need = TRAIN_PER_CLASS + VAL_PER_CLASS
    if len(shuffled) < need:
        print(f"⚠ 类别样本不足 {need}，当前 {len(shuffled)}，将按可用数量切分")

    train_end = min(TRAIN_PER_CLASS, len(shuffled))
    val_end = min(TRAIN_PER_CLASS + VAL_PER_CLASS, len(shuffled))

    train_list = shuffled[:train_end]
    val_list = shuffled[TRAIN_PER_CLASS:val_end]
    return train_list, val_list


def build_samples(train_root: Path):
    print(f"data/train 目录子项: {os.listdir(train_root)}")
    real_dir = train_root / "REAL"
    fake_dir = train_root / "FAKE"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(f"缺少目录: {real_dir} 或 {fake_dir}")

    real_paths = collect_images(real_dir)
    fake_paths = collect_images(fake_dir)

    print(f"REAL 总数: {len(real_paths)}")
    print(f"FAKE 总数: {len(fake_paths)}")

    real_train, real_val = split_class_samples(real_paths, SEED)
    fake_train, fake_val = split_class_samples(fake_paths, SEED)

    # Label direction: FAKE=0, REAL=1
    train_samples = [(p, 1) for p in real_train] + [(p, 0) for p in fake_train]
    val_samples = [(p, 1) for p in real_val] + [(p, 0) for p in fake_val]

    rng = random.Random(SEED)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    print(
        f"训练子集: REAL {len(real_train)} + FAKE {len(fake_train)} = {len(train_samples)}"
    )
    print(f"验证子集: REAL {len(real_val)} + FAKE {len(fake_val)} = {len(val_samples)}")

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

    train_ds = ImageListDataset(train_samples, train_tf)
    val_ds = ImageListDataset(val_samples, val_tf)

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


def freeze_layers(model):
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in FREEZE_PREFIXES):
            param.requires_grad = False
        else:
            param.requires_grad = True


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) if total > 0 else 0.0


def main():
    set_seed(SEED)
    project_root = Path(__file__).resolve().parent
    train_root = project_root / "data" / "train"
    weight_in = project_root / "weights" / "efficientnet_b7_deepfake.pth"
    weight_out = project_root / "weights" / "efficientnet_b7_cifake_finetuned.pth"

    if not train_root.exists():
        raise FileNotFoundError(f"训练目录不存在: {train_root}")
    if not weight_in.exists():
        raise FileNotFoundError(f"初始权重不存在: {weight_in}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠️ 未检测到 CUDA，使用 CPU 训练，预计时间较长")
    print(f"训练设备: {device}")

    train_samples, val_samples = build_samples(train_root)
    train_loader, val_loader = build_loaders(train_samples, val_samples, device)

    model = timm.create_model("tf_efficientnet_b7", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(weight_in, map_location=device))
    freeze_layers(model)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable}/{total}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    weight_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_count = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total_count += labels.size(0)

        scheduler.step()
        train_loss = total_loss / total_count if total_count else 0.0
        train_acc = correct / total_count if total_count else 0.0
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weight_out)
            tag = "saved"
        else:
            tag = "skipped"

        print(
            f"[Epoch {epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% [{tag}]"
        )

    print(f"\n训练完成，最佳 Val Acc: {best_val_acc*100:.2f}%")
    print(f"权重已保存: {weight_out}")


if __name__ == "__main__":
    main()

