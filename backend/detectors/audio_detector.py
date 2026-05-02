import sys
from pathlib import Path

import numpy as np
import torch

_SPEECH_MODULE = Path(__file__).resolve().parents[2] / "modules" / "speech_detection"
_MODEL_PATH = _SPEECH_MODULE / "models" / "best_hybrid_ast.pth"
_CONFIG_PATH = _SPEECH_MODULE / "config.yaml"

_model = None
_config = None
_device = None


def _load():
    global _model, _config, _device

    if _model is not None:
        return _model, _config, _device

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Speech model not found at {_MODEL_PATH}. "
            "Pull the DongZhengyang_Speech_Deepfake branch."
        )

    import yaml
    with open(_CONFIG_PATH) as f:
        _config = yaml.safe_load(f)

    if str(_SPEECH_MODULE) not in sys.path:
        sys.path.insert(0, str(_SPEECH_MODULE))

    from src.model import HybridAST

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = HybridAST(
        n_mels=_config["audio"]["n_mels"],
        num_heads=4,
        hidden_dim=128,
    ).to(_device)
    _model.load_state_dict(torch.load(str(_MODEL_PATH), map_location=_device))
    _model.eval()

    return _model, _config, _device


def _audio_to_tensor(audio_path: str, config: dict) -> torch.Tensor:
    import librosa
    import torch.nn.functional as F

    sample_rate = config["audio"]["sample_rate"]
    n_mels = config["audio"]["n_mels"]
    n_fft = config["audio"]["n_fft"]
    hop_length = config["audio"]["hop_length"]
    max_len = config["audio"]["max_len"]

    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    t = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)

    if t.shape[2] > max_len:
        t = t[:, :, :max_len]
    else:
        t = torch.nn.functional.pad(t, (0, max_len - t.shape[2]))

    return t.unsqueeze(0)  # (1, 1, n_mels, max_len)


def predict(audio_path: str) -> dict:
    model, config, device = _load()

    x = _audio_to_tensor(audio_path, config).to(device)
    with torch.no_grad():
        prob_fake = float(model(x).item())

    prediction = "fake" if prob_fake > 0.5 else "real"
    risk_level = (
        "High" if prob_fake > 0.7
        else "Medium" if prob_fake > 0.4
        else "Low"
    )

    return {
        "prediction": prediction,
        "prob_fake": round(prob_fake, 4),
        "prob_real": round(1 - prob_fake, 4),
        "risk_level": risk_level,
    }
