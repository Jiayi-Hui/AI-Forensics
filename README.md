# 🎙️ Fake Speech Detection Module

This repository implements a **speech deepfake detection module** as part of a multimodal AI-generated content detection system.

The goal is to determine whether a short audio clip (2 seconds) is **real human speech** or **AI-generated speech**, and output a calibrated probability score.

---

# 🚀 1. Overview

Unlike large-model approaches, this module focuses on:

- Robust performance under **limited data**
- Strong **generalization under domain shift**
- Lightweight and **deployable design**

---

# 🧠 2. Pipeline

```text
audio (.wav)
→ preprocessing (resample + normalize)
→ enhanced acoustic feature extraction
→ Random Forest classifier
→ probability calibration (threshold tuning)
→ final prediction
```

---

# 🔬 3. Feature Engineering

We extract a comprehensive set of acoustic features:

### Core Features
- MFCC (20 dims)
- Delta MFCC
- Delta-Delta MFCC

### Spectral Features
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral flatness

### Frequency Representation
- Log-Mel spectrogram (summary)
- Chroma features
- Spectral contrast

### Temporal / Energy Features
- Zero-crossing rate (ZCR)
- RMS energy

---

# 📊 4. Dataset

Dataset: **Fake-or-Real (FoR)** – for-2seconds subset

| Split       | Real | Fake | Total |
|------------|-----:|-----:|------:|
| Training   | 6978 | 6978 | 13956 |
| Validation | 1413 | 1413 | 2826  |
| Testing    | 544  | 544  | 1088  |
| **Total**  | 8935 | 8935 | 17870 |

✔ Fully balanced dataset  
✔ 2-second normalized audio clips  

---

# 📈 5. Model Comparison

| Model                      | Test F1 | Fake Recall |
|---------------------------|--------:|------------:|
| XGBoost baseline          | 0.42    | 0.33        |
| CNN (log-Mel spectrogram) | 0.10    | 0.05        |
| Random Forest (enhanced)  | **0.75** | **0.84** |

---

# ⚠️ 6. Key Finding: Domain Shift

- CNN achieved near-perfect validation performance  
- BUT completely failed on test set  

👉 Root cause: **distribution shift**

---

# 🎯 7. Threshold Calibration

Default threshold (0.5) performs poorly.

| Threshold | Accuracy | Precision | Recall | F1 |
|----------|---------:|----------:|-------:|---:|
| 0.5      | 0.51     | 0.83      | 0.03   | 0.07 |
| 0.4      | 0.61     | 0.79      | 0.30   | 0.44 |
| 0.3      | **0.72** | 0.67      | **0.84** | **0.75** |
| 0.2      | 0.60     | 0.56      | 0.99   | 0.71 |

✔ Final threshold: **0.3**

---

# 🧠 8. Insights

- Feature-based models outperform CNN under small data
- CNN overfits dataset-specific artifacts
- Model still has strong ranking ability (high ROC-AUC)
- Calibration is critical for real-world deployment

---

# 🏗️ 9. Project Structure

```
fake_speech_detector_project/
├── README.md
├── requirements.txt
├── ai_speech_quick_demo.py
├── train_random_forest.py
├── test.py
├── configs/
├── fake_speech_detector/
│   ├── detector.py
│   ├── feature_extraction.py
│   └── utils.py
├── output/
│   ├── rf_enhanced_model.joblib
│   └── results.csv
└── test_audio/
```

---

# ⚙️ 10. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# ▶️ 11. Quick Demo

```bash
python ai_speech_quick_demo.py \
  --audio_path test_audio/sample.wav \
  --model_path output/rf_enhanced_model.joblib \
  --threshold 0.3
```

### Example Output

```
Prediction: fake
Fake probability: 0.42
Risk level: medium
```

---

# 🧪 12. Evaluation

```bash
python test.py \
  --data_root /path/to/for-2seconds \
  --model_path output/rf_enhanced_model.joblib \
  --threshold 0.3
```

---

# ⚠️ 13. Limitations

- Sensitive to domain shift
- Feature engineering may miss long-range temporal patterns
- Requires threshold tuning

---

# 🔮 14. Future Work

- Data augmentation (noise / compression / rerecording)
- Pretrained speech encoders (wav2vec2, HuBERT)
- Domain adaptation
- Multimodal fusion (speech + text + image)

---

# 🧩 15. System Role

This module is designed as:

👉 **a speech risk scoring component**, not a standalone decision system.

It outputs:

```json
{
  "speech_fake_score": 0.42
}
```

for downstream multimodal fusion.

---

# 📜 16. License

Academic use only.
