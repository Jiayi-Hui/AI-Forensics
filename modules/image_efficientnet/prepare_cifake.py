import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SAMPLE_SIZE = 50
SEED = 42


def find_label_dir(root: Path, label: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]

    for path in candidates:
        if path.name.lower() == label.lower():
            return path

    for path in candidates:
        if label.lower() in path.name.lower():
            return path

    raise FileNotFoundError(f"未找到类别目录: {label} (root={root})")


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda x: str(x).lower(),
    )


def clear_output_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.iterdir():
        if p.is_file():
            p.unlink()


def copy_samples(paths: list[Path], output_dir: Path) -> int:
    copied = 0
    for src in paths:
        target = output_dir / src.name
        if target.exists():
            idx = 1
            while True:
                new_name = f"{src.stem}_{idx}{src.suffix}"
                target = output_dir / new_name
                if not target.exists():
                    break
                idx += 1
        shutil.copy2(src, target)
        copied += 1
    return copied


def main() -> None:
    project_root = Path(__file__).resolve().parent
    test_root = project_root / "data" / "test"
    output_root = project_root / "test_images_cifake"

    if not test_root.exists():
        raise FileNotFoundError(f"测试目录不存在: {test_root}")

    real_src = find_label_dir(test_root, "REAL")
    fake_src = find_label_dir(test_root, "FAKE")

    real_all = collect_images(real_src)
    fake_all = collect_images(fake_src)

    print("=== CIFAKE 测试集采样 ===")
    print(f"源目录 REAL: {real_src}")
    print(f"源目录 FAKE: {fake_src}")
    print(f"REAL 可用图像: {len(real_all)}")
    print(f"FAKE 可用图像: {len(fake_all)}")

    random.seed(SEED)
    real_k = min(SAMPLE_SIZE, len(real_all))
    fake_k = min(SAMPLE_SIZE, len(fake_all))
    real_pick = random.sample(real_all, k=real_k)
    fake_pick = random.sample(fake_all, k=fake_k)

    if real_k < SAMPLE_SIZE:
        print(f"⚠ REAL 不足 {SAMPLE_SIZE} 张，实际使用 {real_k} 张")
    if fake_k < SAMPLE_SIZE:
        print(f"⚠ FAKE 不足 {SAMPLE_SIZE} 张，实际使用 {fake_k} 张")

    out_real = output_root / "real"
    out_fake = output_root / "fake"
    clear_output_dir(out_real)
    clear_output_dir(out_fake)

    copied_real = copy_samples(real_pick, out_real)
    copied_fake = copy_samples(fake_pick, out_fake)

    print("\n=== 采样完成 ===")
    print(f"输出目录: {output_root}")
    print(f"REAL 已复制: {copied_real} 张")
    print(f"FAKE 已复制: {copied_fake} 张")

    real_count = len([p for p in out_real.iterdir() if p.is_file()])
    fake_count = len([p for p in out_fake.iterdir() if p.is_file()])
    print("\n=== 数量验证 ===")
    print(f"test_images_cifake/real: {real_count} 张")
    print(f"test_images_cifake/fake: {fake_count} 张")


if __name__ == "__main__":
    main()

