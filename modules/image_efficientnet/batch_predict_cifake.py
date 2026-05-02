import os
import sys
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms


IMAGE_EXTS = (".jpg", ".jpeg", ".png")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def build_model(weight_path: Path, device: torch.device):
    model = timm.create_model("tf_efficientnet_b7", pretrained=False, num_classes=1)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def collect_files(folder_path: Path):
    return sorted(
        [
            p
            for p in folder_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda x: x.name.lower(),
    )


def predict_folder(folder_path: Path, true_label: str, model, transform, device, base_dir: Path):
    records = []
    files = collect_files(folder_path)
    print(f"📁 {true_label} 文件夹 → {len(files)} 张图像")

    for img_path in files:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                real_prob = torch.sigmoid(model(tensor)).item()

            fake_prob = 1.0 - real_prob
            is_fake_pred = fake_prob > 0.5

            records.append(
                {
                    "file": str(img_path.relative_to(base_dir)),
                    "true_label": true_label,
                    "fake_prob": round(fake_prob, 4),
                    "pred_label": "FAKE" if is_fake_pred else "REAL",
                    "correct": (true_label == "FAKE") == is_fake_pred,
                }
            )
        except Exception as exc:
            print(f"  ❌ {img_path.name} 失败: {exc}")

    return records


def main():
    project_root = Path(__file__).resolve().parent
    weight_path = project_root / "weights" / "efficientnet_b7_deepfake.pth"
    input_root = project_root / "test_images_cifake"
    real_dir = input_root / "real"
    fake_dir = input_root / "fake"
    output_csv = project_root / "results" / "predictions_cifake.csv"

    if not weight_path.exists():
        raise FileNotFoundError(f"权重不存在: {weight_path}")
    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"测试目录不存在，请先运行采样脚本: {input_root / 'real'} / {input_root / 'fake'}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = build_model(weight_path, device)
    print("✅ 已加载训练好的 Deepfake 检测权重")

    transform = build_transform()

    all_results = []
    real_results = predict_folder(real_dir, "REAL", model, transform, device, project_root)
    fake_results = predict_folder(fake_dir, "FAKE", model, transform, device, project_root)
    all_results.extend(real_results)
    all_results.extend(fake_results)

    if not all_results:
        raise RuntimeError("未找到可推理图像，请检查 test_images_cifake 目录")

    os.makedirs(project_root / "results", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)

    print("\n✅ CIFAKE 跨域测试完成（Stable Diffusion 生成图像）")
    print(f"📁 REAL 文件夹 → {len(real_results)} 张图像")
    print(f"📁 FAKE 文件夹 → {len(fake_results)} 张图像")

    print("\n📊 Fake 概率分布统计（CIFAKE）：")
    print(df.groupby("true_label")["fake_prob"].describe().round(4))

    print("\n🎯 分类准确率：")
    for label in ["REAL", "FAKE"]:
        sub = df[df["true_label"] == label]
        correct = int(sub["correct"].sum())
        total = len(sub)
        acc = (correct / total * 100.0) if total else 0.0
        print(f"  {label}: {acc:.1f}%  ({correct}/{total} 正确)")

    overall = df["correct"].mean() * 100.0
    print(f"  总体准确率: {overall:.1f}%")
    print(f"\n✅ 结果已保存至 {output_csv}")


if __name__ == "__main__":
    main()
