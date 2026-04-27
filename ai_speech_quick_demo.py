import argparse
from pathlib import Path

from fake_speech_detector.detector import FakeSpeechDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Quick demo for fake speech detection")
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to one audio file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/rf_enhanced_model.joblib",
        help="Path to trained Random Forest model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Decision threshold for fake prediction"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(__file__).resolve().parent
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = root / model_path

    audio_path = Path(args.audio_path)
    if not audio_path.is_absolute():
        audio_path = root / audio_path

    detector = FakeSpeechDetector(
        model_path=model_path,
        threshold=args.threshold
    )

    result = detector.predict_one(audio_path)

    print("\n===== Fake Speech Detection Result =====")
    print(f"Audio path      : {result['audio_path']}")
    print(f"Prediction      : {result['prediction']}")
    print(f"Fake probability: {result['prob_fake']:.4f}")
    print(f"Threshold       : {result['threshold']}")
    print(f"Risk level      : {result['risk_level']}")


if __name__ == "__main__":
    main()