import os
import pandas as pd


def get_scenefake_data(root_dir):
    data = []

    splits = {
        "train": "training",
        "dev": "validation",
        "eval": "testing",
    }

    for folder_name, split_label in splits.items():
        folder_path = os.path.join(root_dir, folder_name)

        if not os.path.exists(folder_path):
            continue

        for label in ["real", "fake"]:
            label_path = os.path.join(folder_path, label)

            if not os.path.exists(label_path):
                continue

            for file in os.listdir(label_path):
                if file.endswith((".wav", ".mp3", ".flac")):
                    data.append({
                        "file_path": os.path.join(label_path, file),
                        "label": label,
                        "split": split_label,
                        "source": "SceneFake",
                    })

    return pd.DataFrame(data)


def get_fake_or_real_data(root_dir):
    data = []

    base_path = os.path.join(root_dir, "for-2sec", "for-2seconds")

    for split in ["training", "validation", "testing"]:
        split_path = os.path.join(base_path, split)

        if not os.path.exists(split_path):
            continue

        for label in ["real", "fake"]:
            label_path = os.path.join(split_path, label)

            if not os.path.exists(label_path):
                continue

            for file in os.listdir(label_path):
                if file.endswith((".wav", ".mp3", ".flac")):
                    data.append({
                        "file_path": os.path.join(label_path, file),
                        "label": label,
                        "split": split,
                        "source": "FakeOrReal",
                    })

    return pd.DataFrame(data)


def build_combined_dataframe(fake_or_real_root, scenefake_root):
    df_for = get_fake_or_real_data(fake_or_real_root)
    df_scene = get_scenefake_data(scenefake_root)

    df_all = pd.concat([df_for, df_scene], ignore_index=True)

    if df_all.empty:
        raise ValueError(
            "No data loaded. Check data/for-2sec/for-2seconds and data/scenefake."
        )

    return df_all


def oversample_by_label(df, random_state=42):
    max_count = df["label"].value_counts().max()

    df_balanced = pd.concat([
        df[df["label"] == "real"].sample(
            n=max_count,
            replace=True,
            random_state=random_state,
        ),
        df[df["label"] == "fake"].sample(
            n=max_count,
            replace=True,
            random_state=random_state,
        ),
    ])

    return df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)