from pathlib import Path

import pandas as pd
import torch
import yaml

from sonics.models.model import AudioClassifier
from sonics.utils.config import dict2cfg
from sonics.utils.dataset import get_dataloader


class FakeMusicDetector:
    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        device=None,
        threshold=0.5,
    ):
        root = Path(__file__).resolve().parents[1]
        self.config_path = Path(config_path or root / "configs/local-smoke.yaml")
        self.checkpoint_path = Path(
            checkpoint_path
            or root
            / "output/spectttra_gamma-t=5-stratified-v3-train80/best_checkpoint.pth"
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold

        with open(self.config_path, "r") as f:
            self.cfg = dict2cfg(yaml.safe_load(f))

        self.model = AudioClassifier(self.cfg).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def predict(self, audio, batch_size=16):
        audio_paths = self._collect_audio_paths(audio)
        if not audio_paths:
            raise FileNotFoundError(f"No audio files found in {audio}")

        dataloader = get_dataloader(
            filepaths=[str(path) for path in audio_paths],
            labels=[0] * len(audio_paths),
            skip_times=None,
            max_len=self.cfg.audio.max_len,
            sample_rate=self.cfg.audio.sample_rate,
            batch_size=batch_size,
            num_classes=self.cfg.num_classes,
            train=False,
            random_sampling=False,
            num_workers=0,
            collate_fn=None,
            distributed=False,
        )

        probs = []
        with torch.no_grad():
            for batch in dataloader:
                waveform = batch["audio"].to(self.device)
                logits = self.model(waveform).squeeze(-1)
                probs.extend(torch.sigmoid(logits).detach().cpu().tolist())

        rows = []
        for path, prob_fake in zip(audio_paths, probs):
            rows.append(
                {
                    "filepath": str(path),
                    "prob_fake": float(prob_fake),
                    "prediction": "fake" if prob_fake > self.threshold else "real",
                }
            )
        return pd.DataFrame(rows)

    def predict_one(self, audio):
        row = self.predict(audio, batch_size=1).iloc[0].to_dict()
        return row

    @staticmethod
    def _collect_audio_paths(audio):
        audio = Path(audio)
        if audio.is_file():
            return [audio]

        exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        return sorted(path for path in audio.rglob("*") if path.suffix.lower() in exts)
