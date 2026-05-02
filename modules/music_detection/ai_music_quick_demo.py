#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General local inference script for Fake Music Detection.

This version removes Colab-only code such as:
- google.colab.drive.mount
- /content/drive/MyDrive/... paths
- notebook shell commands like !cp and !pip install

Expected project structure:

Fake Music detection/
├── ai_generated_test_general.py
├── requirements.txt
├── configs/
├── fake_music_detector/
├── output/
├── sonics/
├── test.py
├── train.py
└── test_audio/              # put .mp3/.wav/.flac files here by default

Basic usage from Terminal:

cd "/Users/jingbo/Documents/NLP/Fake Music detection/Fake Music detection"
python3 ai_generated_test_general.py

Optional usage:

python3 ai_generated_test_general.py \
  --audio_dir test_audio \
  --config configs/local-smoke.yaml \
  --ckpt_path output/spectttra_gamma-t=5/best_checkpoint.pth

Notes:
- This script assumes you have already installed dependencies:
    python3 -m pip install -r requirements.txt
- If your project exposes a different detector API, edit build_detector() below.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


AUDIO_EXTENSIONS = ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg")


def project_root() -> Path:
    """Return the folder containing this script."""
    return Path(__file__).resolve().parent


def resolve_path(path_text: str | None, root: Path) -> Path | None:
    """Resolve a path. Relative paths are interpreted from the project root."""
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def install_requirements_if_requested(root: Path, yes: bool) -> None:
    """Optionally install requirements.txt. Disabled by default for safety."""
    if not yes:
        return

    requirements = root / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"requirements.txt not found: {requirements}")

    print(f"Installing dependencies from: {requirements}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Find audio files recursively under audio_dir."""
    files: list[Path] = []
    for pattern in AUDIO_EXTENSIONS:
        files.extend(Path(p) for p in glob.glob(str(audio_dir / "**" / pattern), recursive=True))
    return sorted(files)


def format_result(result: Any) -> str:
    """Format common detector outputs safely."""
    if isinstance(result, dict):
        prediction = result.get("prediction", result.get("label", "UNKNOWN"))
        prob_fake = result.get("prob_fake", result.get("fake_probability", result.get("score")))
        if isinstance(prob_fake, (int, float)):
            return f"prediction={prediction}, prob_fake={prob_fake:.4f}"
        return f"prediction={prediction}, result={result}"
    return str(result)


def build_detector(config_path: Path | None, ckpt_path: Path | None) -> Any:
    """
    Build the detector.

    Because different fake_music_detector projects expose different APIs, this
    function tries several common patterns. If none works, it gives a precise
    error message instead of failing silently.
    """

    import_errors: list[str] = []

    candidates = [
        # module_name, class_or_function_name
        ("fake_music_detector", "FakeMusicDetector"),
        ("fake_music_detector.detector", "FakeMusicDetector"),
        ("fake_music_detector.inference", "FakeMusicDetector"),
        ("fake_music_detector.predictor", "FakeMusicDetector"),
        ("fake_music_detector", "Detector"),
        ("fake_music_detector.detector", "Detector"),
        ("fake_music_detector.inference", "Detector"),
        ("fake_music_detector.predictor", "Detector"),
    ]

    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls_or_fn = getattr(module, attr_name)
        except Exception as exc:
            import_errors.append(f"{module_name}.{attr_name}: {exc}")
            continue

        # Try keyword constructor first.
        constructor_attempts = [
            {"config_path": str(config_path) if config_path else None, "ckpt_path": str(ckpt_path) if ckpt_path else None},
            {"config": str(config_path) if config_path else None, "checkpoint": str(ckpt_path) if ckpt_path else None},
            {"config_file": str(config_path) if config_path else None, "checkpoint_path": str(ckpt_path) if ckpt_path else None},
            {},
        ]

        for kwargs in constructor_attempts:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try:
                detector = cls_or_fn(**kwargs)
                if hasattr(detector, "predict_one"):
                    return detector
                if hasattr(detector, "predict"):
                    return detector
                import_errors.append(f"{module_name}.{attr_name}: object has no predict_one() or predict() method")
            except Exception as exc:
                import_errors.append(f"{module_name}.{attr_name}({kwargs}): {exc}")

    raise RuntimeError(
        "Could not initialize detector automatically.\n\n"
        "Your project package was found only if fake_music_detector exposes a supported API.\n"
        "Open fake_music_detector/ and check the actual detector class/function name, then edit build_detector().\n\n"
        "Tried:\n- " + "\n- ".join(import_errors[-20:])
    )


def predict_one(detector: Any, audio_file: Path) -> Any:
    """Call the detector using either predict_one(path) or predict(path)."""
    if hasattr(detector, "predict_one"):
        return detector.predict_one(str(audio_file))
    if hasattr(detector, "predict"):
        return detector.predict(str(audio_file))
    raise AttributeError("Detector has neither predict_one() nor predict().")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser(
        description="Run fake music detection on local audio files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio_dir",
        default="test_audio",
        help="Folder containing audio files. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional model config path, e.g. configs/local-smoke.yaml.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Optional checkpoint path, e.g. output/spectttra_gamma-t=5/best_checkpoint.pth.",
    )
    parser.add_argument(
        "--install_requirements",
        action="store_true",
        help="Install requirements.txt before running. Usually you only need this once.",
    )
    parser.add_argument(
        "--list_project_files",
        action="store_true",
        help="Print top-level project files before running.",
    )
    parser.add_argument(
        "--allow_missing_model_args",
        action="store_true",
        help="Allow running without --config and --ckpt_path if your detector has built-in defaults.",
    )

    args = parser.parse_args(argv)
    args.root = root
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.root

    print(f"Project root: {root}")

    if args.list_project_files:
        print("\nProject files:")
        for item in sorted(root.iterdir()):
            print(f"- {item.name}")

    install_requirements_if_requested(root, args.install_requirements)

    audio_dir = resolve_path(args.audio_dir, root)
    config_path = resolve_path(args.config, root)
    ckpt_path = resolve_path(args.ckpt_path, root)

    if audio_dir is None or not audio_dir.exists():
        print(f"Audio folder not found: {audio_dir}")
        print("Create a test_audio folder under the project root, or pass --audio_dir /path/to/audio_folder")
        return 1

    if not args.allow_missing_model_args:
        if config_path is None:
            print("Missing --config. Example: --config configs/local-smoke.yaml")
            print("Use --allow_missing_model_args only if your detector has built-in defaults.")
            return 1
        if ckpt_path is None:
            print("Missing --ckpt_path. Example: --ckpt_path output/spectttra_gamma-t=5/best_checkpoint.pth")
            print("Use --allow_missing_model_args only if your detector has built-in defaults.")
            return 1

    if config_path is not None and not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    if ckpt_path is not None and not ckpt_path.exists():
        print(f"Checkpoint file not found: {ckpt_path}")
        return 1

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in: {audio_dir}")
        print(f"Supported extensions: {', '.join(ext.replace('*', '') for ext in AUDIO_EXTENSIONS)}")
        return 1

    print(f"\nFound {len(audio_files)} audio file(s). Building detector...\n")
    detector = build_detector(config_path=config_path, ckpt_path=ckpt_path)

    print("Starting batch detection...\n")
    for audio_file in audio_files:
        print(f"▶ Detecting: {audio_file.name}")
        try:
            result = predict_one(detector, audio_file)
            print(f"  {format_result(result)}\n")
        except Exception as exc:
            print(f"  ERROR: failed to process {audio_file}: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
