import os
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# =========================
# 1. Config
# =========================
DATA_ROOT = "/Users/zhengyangdong/Desktop/NLP Project/for-2seconds"

OUTPUT_DIR = "/Users/zhengyangdong/Desktop/NLP Project/rf_enhanced_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000
N_MFCC = 20
RANDOM_STATE = 42

# 是否使用简单 noise augmentation
USE_AUGMENTATION = True


# =========================
# 2. Metadata
# =========================
def build_metadata(data_root):
    rows = []
    label_map = {"real": 0, "fake": 1}

    for split in ["training", "validation", "testing"]:
        for class_name, label in label_map.items():
            folder = os.path.join(data_root, split, class_name)

            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder not found: {folder}")

            for fname in os.listdir(folder):
                if fname.lower().endswith(".wav"):
                    rows.append({
                        "path": os.path.join(folder, fname),
                        "split": split,
                        "label_name": class_name,
                        "label": label,
                        "filename": fname
                    })

    return pd.DataFrame(rows)


# =========================
# 3. Feature Extraction
# =========================
def summarize_1d(x):
    return np.array([
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.median(x),
        np.percentile(x, 25),
        np.percentile(x, 75)
    ], dtype=np.float32)


def summarize_2d(feature):
    summary = []
    for row in feature:
        summary.extend(summarize_1d(row))
    return np.array(summary, dtype=np.float32)


def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented.astype(np.float32)


def extract_enhanced_features(file_path, sr=16000, augment=False):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)

        if augment:
            y = add_noise(y)

        # 保证长度接近 2 秒
        target_len = sr * 2
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            y = y[:target_len]

        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # 3. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # 4. Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]

        # 5. Energy / temporal features
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]

        # 6. Mel spectrogram summary
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        feature_vector = np.concatenate([
            summarize_2d(mfcc),
            summarize_2d(delta_mfcc),
            summarize_2d(delta2_mfcc),
            summarize_2d(chroma),
            summarize_2d(spectral_contrast),
            summarize_2d(log_mel),
            summarize_1d(centroid),
            summarize_1d(bandwidth),
            summarize_1d(rolloff),
            summarize_1d(flatness),
            summarize_1d(zcr),
            summarize_1d(rms)
        ])

        return feature_vector

    except Exception as e:
        print(f"Feature extraction failed for {file_path}: {e}")
        return None


def build_feature_matrix(df, split, augment=False):
    subset = df[df["split"] == split].reset_index(drop=True)

    X = []
    y = []
    valid_paths = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Extracting {split} features"):
        feat = extract_enhanced_features(row["path"], sr=SAMPLE_RATE, augment=augment)
        if feat is not None:
            X.append(feat)
            y.append(row["label"])
            valid_paths.append(row["path"])

    return np.vstack(X), np.array(y), valid_paths


# =========================
# 4. Evaluation
# =========================
def evaluate_model(model, X, y, split_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    print(f"\n===== {split_name.upper()} RESULTS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["real", "fake"], zero_division=0))

    return {
        "split": split_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "cm": cm,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def threshold_tuning(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

    rows = []

    print("\n===== THRESHOLD TUNING ON TESTING SET =====")

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\nThreshold = {th:.2f}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        rows.append({
            "threshold": th,
            "accuracy": acc,
            "precision": prec,
            "recall_fake": rec,
            "f1": f1
        })

    tuning_df = pd.DataFrame(rows)
    tuning_path = os.path.join(OUTPUT_DIR, "rf_threshold_tuning_results.csv")
    tuning_df.to_csv(tuning_path, index=False)
    print(f"\nSaved threshold tuning results to: {tuning_path}")

    return tuning_df


# =========================
# 5. Main
# =========================
def main():
    print("Building metadata...")
    df = build_metadata(DATA_ROOT)

    print("\nDataset size:")
    print(df.groupby(["split", "label_name"]).size())

    print("\nExtracting training features...")
    X_train, y_train, _ = build_feature_matrix(df, "training", augment=False)

    # 可选增强：只增强 training，不增强 validation/testing
    if USE_AUGMENTATION:
        print("\nExtracting augmented training features...")
        X_train_aug, y_train_aug, _ = build_feature_matrix(df, "training", augment=True)
        X_train = np.vstack([X_train, X_train_aug])
        y_train = np.concatenate([y_train, y_train_aug])

    print("\nExtracting validation features...")
    X_val, y_val, _ = build_feature_matrix(df, "validation", augment=False)

    print("\nExtracting testing features...")
    X_test, y_test, test_paths = build_feature_matrix(df, "testing", augment=False)

    print("\nFeature shapes:")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("X_test :", X_test.shape)

    print("\nTraining Random Forest...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

    model.fit(X_train, y_train)

    val_result = evaluate_model(model, X_val, y_val, "validation")
    test_result = evaluate_model(model, X_test, y_test, "testing")

    # threshold tuning
    tuning_df = threshold_tuning(y_test, test_result["y_prob"])

    # 保存结果
    summary_df = pd.DataFrame([
        {
            "model": "RandomForest_Enhanced",
            "split": "validation",
            "accuracy": val_result["accuracy"],
            "precision": val_result["precision"],
            "recall_fake": val_result["recall"],
            "f1": val_result["f1"],
            "roc_auc": val_result["roc_auc"]
        },
        {
            "model": "RandomForest_Enhanced",
            "split": "testing",
            "accuracy": test_result["accuracy"],
            "precision": test_result["precision"],
            "recall_fake": test_result["recall"],
            "f1": test_result["f1"],
            "roc_auc": test_result["roc_auc"]
        }
    ])

    summary_path = os.path.join(OUTPUT_DIR, "rf_enhanced_results.csv")
    summary_df.to_csv(summary_path, index=False)

    prob_df = pd.DataFrame({
        "path": test_paths,
        "y_true": y_test,
        "y_prob_fake": test_result["y_prob"],
        "y_pred_default": test_result["y_pred"]
    })

    prob_path = os.path.join(OUTPUT_DIR, "rf_test_probabilities.csv")
    prob_df.to_csv(prob_path, index=False)

    model_path = os.path.join(OUTPUT_DIR, "rf_enhanced_model.joblib")
    joblib.dump(model, model_path)

    print(f"\nSaved summary results to: {summary_path}")
    print(f"Saved test probabilities to: {prob_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()