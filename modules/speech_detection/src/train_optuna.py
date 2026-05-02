import os
import random
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import optuna

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.data_loader import build_combined_dataframe, oversample_by_label
from src.dataset import AudioDeepfakeDataset
from src.model import HybridAST


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seed set to: {seed}")


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=5,
):
    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
        )

        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()

        all_labels = []
        all_preds = []
        all_probs = []

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Val]",
        )

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                probs = model(inputs)
                preds = (probs > 0.5).float()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        print(
            f"\nLoss: {total_loss / len(train_loader):.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"Precision: {prec:.4f} | "
            f"Recall: {rec:.4f} | "
            f"F1: {f1:.4f} | "
            f"AUC: {auc:.4f}\n"
        )

        if auc > best_val_auc:
            best_val_auc = auc

    return best_val_auc


def objective(trial, df_train, df_val, device, config):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    num_heads = trial.suggest_categorical(
        "num_heads",
        config["model"]["num_heads_options"],
    )

    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        config["model"]["hidden_dim_options"],
    )

    audio_cfg = config["audio"]
    train_cfg = config["training"]

    train_dataset = AudioDeepfakeDataset(
        df_train,
        sample_rate=audio_cfg["sample_rate"],
        n_mels=audio_cfg["n_mels"],
        max_len=audio_cfg["max_len"],
    )

    val_dataset = AudioDeepfakeDataset(
        df_val,
        sample_rate=audio_cfg["sample_rate"],
        n_mels=audio_cfg["n_mels"],
        max_len=audio_cfg["max_len"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = HybridAST(
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_auc = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=train_cfg["epochs_per_trial"],
    )

    return val_auc


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    print("\nLoading datasets...")

    df_all = build_combined_dataframe(
        fake_or_real_root=config["paths"]["fake_or_real_root"],
        scenefake_root=config["paths"]["scenefake_root"],
    )

    print("\nDataset loaded:")
    print(df_all["source"].value_counts())
    print(df_all["split"].value_counts())
    print(df_all["label"].value_counts())

    df_train = df_all[df_all["split"] == "training"].reset_index(drop=True)
    df_val = df_all[df_all["split"] == "validation"].reset_index(drop=True)

    print(f"\nBefore oversampling:")
    print(f"Train size: {len(df_train)}")
    print(f"Validation size: {len(df_val)}")

    df_train = oversample_by_label(df_train, random_state=config["seed"])
    df_val = oversample_by_label(df_val, random_state=config["seed"])

    print(f"\nAfter oversampling:")
    print(f"Train size: {len(df_train)}")
    print(f"Validation size: {len(df_val)}")
    print(df_train["label"].value_counts())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\nStarting Bayesian Optimization with Optuna...")

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(trial, df_train, df_val, device, config),
        n_trials=config["training"]["optuna_trials"],
    )

    print("\nBest AUC:", study.best_value)
    print("Best hyperparameters:", study.best_params)
    print("\nStarting final training with best parameters...")

    best_params = study.best_params

    audio_cfg = config["audio"]
    train_cfg = config["training"]

    train_dataset = AudioDeepfakeDataset(
        df_train,
        sample_rate=audio_cfg["sample_rate"],
        n_mels=audio_cfg["n_mels"],
        max_len=audio_cfg["max_len"],
    )

    val_dataset = AudioDeepfakeDataset(
        df_val,
        sample_rate=audio_cfg["sample_rate"],
        n_mels=audio_cfg["n_mels"],
        max_len=audio_cfg["max_len"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    final_model = HybridAST(
        num_heads=best_params["num_heads"],
        hidden_dim=best_params["hidden_dim"],
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"])

    final_auc = train_and_evaluate(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=10,
    )

    torch.save(final_model.state_dict(), config["paths"]["model_output"])

    print(f"\nFinal model saved to: {config['paths']['model_output']}")
    print(f"Final validation AUC: {final_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()

    main(args.config)