from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms

_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "weights" / "efficientnet_b7_deepfake.pth"

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    m = timm.create_model("tf_efficientnet_b7", pretrained=False, num_classes=1)
    m.load_state_dict(torch.load(_WEIGHTS_PATH, map_location="cpu"))
    m.eval()
    _model = m
    return m


def predict(image: Image.Image) -> dict:
    model = _load_model()
    tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logit = model(tensor).item()
    real_prob = float(torch.sigmoid(torch.tensor(logit)))
    fake_prob = 1.0 - real_prob
    return {
        "label": "FAKE" if fake_prob > 0.5 else "REAL",
        "fake_prob": round(fake_prob, 4),
        "real_prob": round(real_prob, 4),
        "confidence": round(max(fake_prob, real_prob), 4),
    }
