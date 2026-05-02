import torch
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
import os

# ── 路径配置 ──
TRAIN_DIR = r"F:\HYH_LocalFile\8307_GroupProject\data\real_vs_fake\real-vs-fake\train"
VALID_DIR = r"F:\HYH_LocalFile\8307_GroupProject\data\real_vs_fake\real-vs-fake\valid"
SAVE_PATH = r"F:\HYH_LocalFile\8307_GroupProject\weights\efficientnet_b7_deepfake.pth"
os.makedirs("weights", exist_ok=True)

# ── 数据增强 ──
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 加载数据（各取 2000 张加快训练）──
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
valid_dataset = datasets.ImageFolder(VALID_DIR, transform=val_transform)

# 只取子集加快速度
from torch.utils.data import Subset
import random
train_idx = random.sample(range(len(train_dataset)), min(2000, len(train_dataset)))
valid_idx = random.sample(range(len(valid_dataset)), min(400,  len(valid_dataset)))
train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=16, shuffle=True)
valid_loader = DataLoader(Subset(valid_dataset, valid_idx), batch_size=16)

print(f"类别映射: {train_dataset.class_to_idx}")  # 确认 fake=1, real=0
print(f"训练集: {len(train_idx)} 张 | 验证集: {len(valid_idx)} 张")

# ── 模型 ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = timm.create_model('tf_efficientnet_b7', pretrained=True, num_classes=1)
model = model.to(device)

# ── 训练设置 ──
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# ── 训练循环 ──
best_val_acc = 0
for epoch in range(5):
    # 训练
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        out = model(imgs).squeeze(1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total

    # 验证
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            out = model(imgs).squeeze(1)
            preds = (torch.sigmoid(out) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f} "
          f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  💾 最佳模型已保存 (Val Acc: {val_acc:.4f})")

    scheduler.step()

print(f"\n✅ 训练完成！最佳验证准确率: {best_val_acc:.4f}")
print(f"权重保存在: {SAVE_PATH}")