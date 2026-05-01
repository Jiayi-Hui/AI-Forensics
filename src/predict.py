import argparse
import yaml
import torch
import pandas as pd

from src.dataset import AudioDeepfakeDataset
from src.model import HybridAST


def predict(audio_path, config_path, model_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridAST(
        num_heads=4,
        hidden_dim=128,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.DataFrame([
        {
            "file_path": audio_path,
            "label": "fake",
            "split": "testing",
            "source": "single_file",
        }
    ])

    dataset = AudioDeepfakeDataset(
        df,
        sample_rate=config["audio"]["sample_rate"],
        n_mels=config["audio"]["n_mels"],
        max_len=config["audio"]["max_len"],
    )

    x, _ = dataset[0]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(x).item()

    label = "fake" if prob > 0.5 else "real"

    print("================================")
    print(f"Audio file: {audio_path}")
    print(f"Prediction: {label}")
    print(f"Fake probability: {prob:.4f}")
    print("================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="models/best_hybrid_ast.pth")

    args = parser.parse_args()

    predict(args.audio, args.config, args.model)