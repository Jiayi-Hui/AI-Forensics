import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for speech deepfake detection")
    parser.add_argument("--audio", required=True, help="Path to an audio file")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model", default="models/best_hybrid_ast.pth", help="Path to trained model")

    args = parser.parse_args()

    print("Running Speech Deepfake Detection Demo...")
    predict(args.audio, args.config, args.model)