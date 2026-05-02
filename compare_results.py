from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


STYLEGAN_REAL_ACC = 78.0
STYLEGAN_FAKE_ACC = 84.0
STYLEGAN_OVERALL_ACC = 81.0

STYLEGAN_FAKE_MEAN = 0.8010
STYLEGAN_FAKE_MEDIAN = 0.9814
STYLEGAN_REAL_MEAN = 0.2339
STYLEGAN_REAL_MEDIAN = 0.0348
STYLEGAN_SEPARATION = 0.5671

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["true_label"] = out["true_label"].astype(str).str.upper()
    out["fake_prob"] = pd.to_numeric(out["fake_prob"], errors="coerce")
    out["correct"] = (
        out["correct"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(out["correct"])
        .astype(bool)
    )
    out = out.dropna(subset=["fake_prob"])
    return out


def calc_acc(df: pd.DataFrame, label: str):
    sub = df[df["true_label"] == label]
    total = len(sub)
    correct = int(sub["correct"].sum())
    acc = (correct / total * 100.0) if total else 0.0
    return acc, correct, total


def draw_distribution(style_df: pd.DataFrame, cifake_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    bins = 20

    for ax, df, title in [
        (axes[0], style_df, "StyleGAN Domain"),
        (axes[1], cifake_df, "Stable Diffusion (CIFAKE) Domain"),
    ]:
        real_vals = df[df["true_label"] == "REAL"]["fake_prob"].values
        fake_vals = df[df["true_label"] == "FAKE"]["fake_prob"].values

        ax.hist(real_vals, bins=bins, alpha=0.6, color="#1f77b4", label="REAL", range=(0, 1))
        ax.hist(fake_vals, bins=bins, alpha=0.6, color="#d62728", label="FAKE", range=(0, 1))

        real_mean = float(np.mean(real_vals)) if len(real_vals) else 0.0
        fake_mean = float(np.mean(fake_vals)) if len(fake_vals) else 0.0

        ax.axvline(real_mean, color="#1f77b4", linestyle="--", linewidth=1.5)
        ax.axvline(fake_mean, color="#d62728", linestyle="--", linewidth=1.5)

        y_top = ax.get_ylim()[1]
        ax.text(real_mean, y_top * 0.90, f"REAL mean={real_mean:.4f}", color="#1f77b4", fontsize=10)
        ax.text(fake_mean, y_top * 0.80, f"FAKE mean={fake_mean:.4f}", color="#d62728", fontsize=10)

        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2, linestyle=":")

    axes[1].set_xlabel("Fake Probability", fontsize=10)
    fig.suptitle("Fake Probability Distribution — StyleGAN vs Stable Diffusion", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def draw_boxplot(style_df: pd.DataFrame, cifake_df: pd.DataFrame, out_path: Path):
    groups = [
        style_df[style_df["true_label"] == "REAL"]["fake_prob"].values,
        style_df[style_df["true_label"] == "FAKE"]["fake_prob"].values,
        cifake_df[cifake_df["true_label"] == "REAL"]["fake_prob"].values,
        cifake_df[cifake_df["true_label"] == "FAKE"]["fake_prob"].values,
    ]
    labels = ["StyleGAN-REAL", "StyleGAN-FAKE", "SD-REAL", "SD-FAKE"]
    colors = ["#1f77b4", "#d62728", "#85c1e9", "#f5b7b1"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    for i, vals in enumerate(groups, start=1):
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        y = min(0.98, med + 0.04)
        ax.text(i, y, f"{med:.4f}", ha="center", va="bottom", fontsize=10)

    legend_handles = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="REAL (StyleGAN, darker blue)"),
        Patch(facecolor="#85c1e9", edgecolor="black", label="REAL (Stable Diffusion, lighter blue)"),
        Patch(facecolor="#d62728", edgecolor="black", label="FAKE (StyleGAN, darker red)"),
        Patch(facecolor="#f5b7b1", edgecolor="black", label="FAKE (Stable Diffusion, lighter red)"),
    ]

    ax.legend(handles=legend_handles, fontsize=10, loc="upper left")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fake Probability", fontsize=10)
    ax.set_title("Fake Probability Boxplot — Domain Comparison", fontsize=14)
    ax.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def get_domain_stats(df: pd.DataFrame):
    fake_vals = df[df["true_label"] == "FAKE"]["fake_prob"]
    real_vals = df[df["true_label"] == "REAL"]["fake_prob"]
    fake_mean = float(fake_vals.mean())
    fake_median = float(fake_vals.median())
    real_mean = float(real_vals.mean())
    real_median = float(real_vals.median())
    separation = fake_mean - real_mean
    return fake_mean, fake_median, real_mean, real_median, separation


def conclusion_by_acc(acc: float) -> str:
    if acc >= 75.0:
        return "模型具备初步跨域泛化能力"
    if 60.0 <= acc < 75.0:
        return "跨域泛化能力有限，存在明显性能下降"
    return "模型对 Diffusion 图像泛化失败，接近随机猜测"


def main():
    project_root = Path(__file__).resolve().parent
    style_csv = project_root / "results" / "predictions.csv"
    cifake_csv = project_root / "results" / "predictions_cifake.csv"

    if not style_csv.exists():
        raise FileNotFoundError(f"缺少文件: {style_csv}")
    if not cifake_csv.exists():
        raise FileNotFoundError(f"缺少文件: {cifake_csv}")

    style_df = normalize_df(pd.read_csv(style_csv))
    cifake_df = normalize_df(pd.read_csv(cifake_csv))

    cifake_real_acc, cifake_real_ok, cifake_real_total = calc_acc(cifake_df, "REAL")
    cifake_fake_acc, cifake_fake_ok, cifake_fake_total = calc_acc(cifake_df, "FAKE")
    cifake_overall_acc = float(cifake_df["correct"].mean() * 100.0)

    print("| 数据来源       | 生成方式           | REAL 准确率 | FAKE 准确率 | 总体准确率 |")
    print("|----------------|--------------------|-------------|-------------|------------|")
    print(
        f"| 140K 数据集    | StyleGAN（训练域） |    {STYLEGAN_REAL_ACC:.1f}%    |    {STYLEGAN_FAKE_ACC:.1f}%    |   {STYLEGAN_OVERALL_ACC:.1f}%    |"
    )
    print(
        f"| CIFAKE 数据集  | Stable Diffusion   |    {cifake_real_acc:.1f}%    |    {cifake_fake_acc:.1f}%    |   {cifake_overall_acc:.1f}%    |"
    )

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dist_png = results_dir / "compare_distribution.png"
    box_png = results_dir / "compare_boxplot.png"

    draw_distribution(style_df, cifake_df, dist_png)
    draw_boxplot(style_df, cifake_df, box_png)

    sd_fake_mean, sd_fake_median, sd_real_mean, sd_real_median, sd_sep = get_domain_stats(cifake_df)

    acc_delta = cifake_overall_acc - STYLEGAN_OVERALL_ACC
    print("\n📈 跨域泛化分析：")
    print("\n[StyleGAN → 训练域]")
    print(f"  FAKE 组 fake_prob 均值：{STYLEGAN_FAKE_MEAN:.4f}  中位数：{STYLEGAN_FAKE_MEDIAN:.4f}")
    print(f"  REAL 组 fake_prob 均值：{STYLEGAN_REAL_MEAN:.4f}  中位数：{STYLEGAN_REAL_MEDIAN:.4f}")
    print(f"  均值差（分离度）：{STYLEGAN_SEPARATION:.4f}")

    print("\n[Stable Diffusion → 测试域（跨域）]")
    print(f"  FAKE 组 fake_prob 均值：{sd_fake_mean:.4f}  中位数：{sd_fake_median:.4f}")
    print(f"  REAL 组 fake_prob 均值：{sd_real_mean:.4f}  中位数：{sd_real_median:.4f}")
    print(f"  均值差（分离度）：{sd_sep:.4f}")

    print("\n💡 泛化结论：")
    print(
        f"  - 总体准确率变化：{STYLEGAN_OVERALL_ACC:.1f}% → {cifake_overall_acc:.1f}%（差值 {acc_delta:+.1f}%）"
    )
    print(f"  - FAKE 检测率变化：{STYLEGAN_FAKE_ACC:.1f}% → {cifake_fake_acc:.1f}%")
    print(f"  - 两组分离度变化：{STYLEGAN_SEPARATION:.4f} → {sd_sep:.4f}")
    print(f"  - 初步判断：{conclusion_by_acc(cifake_overall_acc)}")

    print("\n📁 输出文件：")
    print(f"  - {dist_png}")
    print(f"  - {box_png}")
    print(f"  - CIFAKE 明细: {cifake_csv}")
    print(f"  - CIFAKE 准确率: REAL {cifake_real_ok}/{cifake_real_total}, FAKE {cifake_fake_ok}/{cifake_fake_total}")


if __name__ == "__main__":
    main()
