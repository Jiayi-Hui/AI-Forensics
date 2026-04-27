from pathlib import Path
import joblib

from fake_speech_detector.feature_extraction import extract_enhanced_features


class FakeSpeechDetector:
    def __init__(self, model_path, threshold=0.3):
        self.model_path = Path(model_path)
        self.threshold = threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)

    def predict_one(self, audio_path):
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        features = extract_enhanced_features(str(audio_path))
        prob_fake = float(self.model.predict_proba(features)[0][1])

        prediction = "fake" if prob_fake >= self.threshold else "real"

        if prob_fake < 0.3:
            risk_level = "low"
        elif prob_fake < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "audio_path": str(audio_path),
            "prediction": prediction,
            "prob_fake": prob_fake,
            "threshold": self.threshold,
            "risk_level": risk_level
        }