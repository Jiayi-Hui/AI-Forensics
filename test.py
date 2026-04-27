import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from fake_speech_detector.detector import FakeSpeechDetector


def collect_files(data_root):
    rows = []
    label_map = {"real": 0, "fake": 1}

    for cls, label in label_map.items():
        folder = Path(data_root) / "testing" / cls
        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                rows.append({
                    "path": str(folder / fname),
                    "label": label,
                    "label_name": cls
                })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fake speech detector on testing split")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to FoR for-2seconds dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/rf_enhanced_model.joblib"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = root / model_path

    detector = FakeSpeechDetector(model_path=model_path, threshold=args.threshold)

    df = collect_files(args.data_root)

    y_true = []
    y_pred = []
    probs = []

    for _, row in df.iterrows():
        result = detector.predict_one(row["path"])
        y_true.append(row["label"])
        y_pred.append(1 if result["prediction"] == "fake" else 0)
        probs.append(result["prob_fake"])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== Testing Results =====")
    print(f"Threshold : {args.threshold}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["real", "fake"], zero_division=0))

    out_df = df.copy()
    out_df["prob_fake"] = probs
    out_df["prediction"] = y_pred

    output_path = root / "output" / "speech_test_predictions.csv"
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()