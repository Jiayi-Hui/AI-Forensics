import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def find_label_dir(root: Path, label: str) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    for d in dirs:
        if d.name.lower() == label.lower():
            return d
    raise FileNotFoundError(f"目录 {root} 下未找到标签目录 {label}")


def collect_images(folder: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda x: x.name.lower(),
    )


def load_model(weight_path: Path, device: torch.device):
    if not weight_path.exists():
        raise FileNotFoundError(f"权重不存在: {weight_path}")
    model = timm.create_model("tf_efficientnet_b7", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_for_root(
    model,
    root: Path,
    device: torch.device,
    transform,
) -> pd.DataFrame:
    real_dir = find_label_dir(root, "real")
    fake_dir = find_label_dir(root, "fake")

    records = []
    for true_label, folder in [("REAL", real_dir), ("FAKE", fake_dir)]:
        files = collect_images(folder)
        for path in files:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                real_prob = torch.sigmoid(model(tensor)).item()
            fake_prob = 1.0 - real_prob
            pred_label = "FAKE" if fake_prob > 0.5 else "REAL"
            correct = (true_label == "FAKE") == (fake_prob > 0.5)
            records.append(
                {
                    "file": str(path.relative_to(root)),
                    "true_label": true_label,
                    "fake_prob": round(fake_prob, 4),
                    "pred_label": pred_label,
                    "correct": bool(correct),
                }
            )
    return pd.DataFrame(records)


def metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    def cls_acc(label: str):
        sub = df[df["true_label"] == label]
        return float(sub["correct"].mean() * 100.0) if len(sub) else 0.0

    real_acc = cls_acc("REAL")
    fake_acc = cls_acc("FAKE")
    overall = float(df["correct"].mean() * 100.0) if len(df) else 0.0

    fake_vals = df[df["true_label"] == "FAKE"]["fake_prob"]
    real_vals = df[df["true_label"] == "REAL"]["fake_prob"]
    fake_mean = float(fake_vals.mean()) if len(fake_vals) else 0.0
    fake_median = float(fake_vals.median()) if len(fake_vals) else 0.0
    real_mean = float(real_vals.mean()) if len(real_vals) else 0.0
    real_median = float(real_vals.median()) if len(real_vals) else 0.0
    separation = fake_mean - real_mean

    return {
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "overall_acc": overall,
        "fake_mean": fake_mean,
        "fake_median": fake_median,
        "real_mean": real_mean,
        "real_median": real_median,
        "separation": separation,
    }


def separation_conclusion(sep: float) -> str:
    if sep > 0.3:
        return "分布分离良好"
    if 0.1 <= sep <= 0.3:
        return "分离有限"
    if 0.0 <= sep < 0.1:
        return "分离不足"
    return "仍然反转"


def print_summary_table(stats: Dict[str, Dict[str, float]]):
    print("╔══════════════════════════════════════════════════════╗")
    print("║          CIFAKE 跨域测试 — 策略对比汇总              ║")
    print("╠══════════════╦════════════╦════════════╦════════════╣")
    print("║ 策略         ║ REAL 准确率 ║ FAKE 准确率 ║ 总体准确率 ║")
    print("╠══════════════╬════════════╬════════════╬════════════╣")
    for key, title in [
        ("baseline", "Baseline"),
        ("strategy_a", "策略A（微调）"),
        ("strategy_b", "策略B（混合）"),
    ]:
        s = stats[key]
        print(
            f"║ {title:<10} ║ {s['real_acc']:>6.1f}%   ║ {s['fake_acc']:>6.1f}%   ║ {s['overall_acc']:>6.1f}%   ║"
        )
    print("╚══════════════╩════════════╩════════════╩════════════╝")


def print_distribution_analysis(stats: Dict[str, Dict[str, float]]):
    print("\n分布分析：")
    for key, title in [
        ("baseline", "Baseline"),
        ("strategy_a", "策略A（微调）"),
        ("strategy_b", "策略B（混合）"),
    ]:
        s = stats[key]
        print(f"\n[{title}]")
        print(f"  FAKE 组 fake_prob：均值 {s['fake_mean']:.4f}，中位数 {s['fake_median']:.4f}")
        print(f"  REAL 组 fake_prob：均值 {s['real_mean']:.4f}，中位数 {s['real_median']:.4f}")
        print(f"  分离度（FAKE均值 - REAL均值）：{s['separation']:+.4f}")
        print(f"  结论：{separation_conclusion(s['separation'])}")


def plot_accuracy(stats: Dict[str, Dict[str, float]], out_path: Path):
    labels = ["Baseline", "策略A", "策略B"]
    real_vals = [stats["baseline"]["real_acc"], stats["strategy_a"]["real_acc"], stats["strategy_b"]["real_acc"]]
    fake_vals = [stats["baseline"]["fake_acc"], stats["strategy_a"]["fake_acc"], stats["strategy_b"]["fake_acc"]]
    overall_vals = [
        stats["baseline"]["overall_acc"],
        stats["strategy_a"]["overall_acc"],
        stats["strategy_b"]["overall_acc"],
    ]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - width, real_vals, width, label="REAL 准确率", color="#2f6db3")
    bars2 = ax.bar(x, fake_vals, width, label="FAKE 准确率", color="#c0392b")
    bars3 = ax.bar(x + width, overall_vals, width, label="总体准确率", color="#7f8c8d")

    for bars in [bars1, bars2, bars3]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.text(2.4, 51.5, "随机猜测基准", fontsize=10, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CIFAKE Strategy Comparison — Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.2, linestyle=":")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_separation(stats: Dict[str, Dict[str, float]], out_path: Path):
    labels = ["Baseline", "策略A", "策略B"]
    fake_means = [stats["baseline"]["fake_mean"], stats["strategy_a"]["fake_mean"], stats["strategy_b"]["fake_mean"]]
    real_means = [stats["baseline"]["real_mean"], stats["strategy_a"]["real_mean"], stats["strategy_b"]["real_mean"]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, fake_means, marker="o", color="#c0392b", linewidth=2, label="FAKE 组均值")
    ax.plot(x, real_means, marker="o", color="#2f6db3", linewidth=2, label="REAL 组均值")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.text(2.2, 0.515, "理想分界位置", fontsize=10, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean Fake Probability")
    ax.set_title("Fake Probability Mean — REAL vs FAKE group")
    ax.legend()
    ax.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_boxplot(results: Dict[str, pd.DataFrame], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    configs = [
        ("baseline", "Baseline"),
        ("strategy_a", "策略A"),
        ("strategy_b", "策略B"),
    ]

    for ax, (key, title) in zip(axes, configs):
        df = results[key]
        real_vals = df[df["true_label"] == "REAL"]["fake_prob"].values
        fake_vals = df[df["true_label"] == "FAKE"]["fake_prob"].values
        bp = ax.boxplot([real_vals, fake_vals], tick_labels=["REAL", "FAKE"], patch_artist=True)
        colors = ["#2f6db3", "#c0392b"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        med_real = float(np.median(real_vals)) if len(real_vals) else 0.0
        med_fake = float(np.median(fake_vals)) if len(fake_vals) else 0.0
        ax.text(1, min(0.98, med_real + 0.05), f"{med_real:.4f}", ha="center", fontsize=9)
        ax.text(2, min(0.98, med_fake + 0.05), f"{med_fake:.4f}", ha="center", fontsize=9)

        ax.set_title(title)
        ax.grid(alpha=0.2, linestyle=":")

    axes[0].set_ylabel("Fake Probability")
    fig.suptitle("CIFAKE Fake Probability Boxplot by Strategy")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def maybe_eval_stylegan(
    strategy_weights: Dict[str, Path], device: torch.device, transform, project_root: Path
):
    style_root = project_root / "test_images"
    if not style_root.exists():
        print("\n未找到 test_images/，跳过 StyleGAN 遗忘检查。")
        return

    try:
        _ = find_label_dir(style_root, "real")
        _ = find_label_dir(style_root, "fake")
    except FileNotFoundError:
        print("\ntest_images/ 缺少 real/fake 子目录，跳过 StyleGAN 遗忘检查。")
        return

    print("\nStyleGAN 测试集附加评估（遗忘检查）：")
    for key, title in [("baseline", "Baseline"), ("strategy_a", "策略A"), ("strategy_b", "策略B")]:
        model = load_model(strategy_weights[key], device)
        df = predict_for_root(model, style_root, device, transform)
        overall = float(df["correct"].mean() * 100.0) if len(df) else 0.0
        print(f"  {title} StyleGAN 总体准确率: {overall:.1f}%")


def main():
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cifake_root = project_root / "test_images_cifake"
    if not cifake_root.exists():
        raise FileNotFoundError(f"缺少目录: {cifake_root}")

    strategy_weights = {
        "baseline": project_root / "weights" / "efficientnet_b7_deepfake.pth",
        "strategy_a": project_root / "weights" / "efficientnet_b7_cifake_finetuned.pth",
        "strategy_b": project_root / "weights" / "efficientnet_b7_mixed.pth",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估设备: {device}")

    transform = build_transform()
    results = {}
    stats = {}

    for key in ["baseline", "strategy_a", "strategy_b"]:
        model = load_model(strategy_weights[key], device)
        df = predict_for_root(model, cifake_root, device, transform)
        results[key] = df
        stats[key] = metrics_from_df(df)

    # Required output files
    results["strategy_a"].to_csv(results_dir / "predictions_cifake_strategyA.csv", index=False)
    results["strategy_b"].to_csv(results_dir / "predictions_cifake_strategyB.csv", index=False)

    print_summary_table(stats)
    print_distribution_analysis(stats)

    plot_accuracy(stats, results_dir / "strategy_comparison_accuracy.png")
    plot_separation(stats, results_dir / "strategy_comparison_separation.png")
    plot_boxplot(results, results_dir / "strategy_comparison_boxplot.png")

    maybe_eval_stylegan(strategy_weights, device, transform, project_root)

    print("\n输出文件：")
    print(f"  - {results_dir / 'predictions_cifake_strategyA.csv'}")
    print(f"  - {results_dir / 'predictions_cifake_strategyB.csv'}")
    print(f"  - {results_dir / 'strategy_comparison_accuracy.png'}")
    print(f"  - {results_dir / 'strategy_comparison_separation.png'}")
    print(f"  - {results_dir / 'strategy_comparison_boxplot.png'}")


if __name__ == "__main__":
    main()

