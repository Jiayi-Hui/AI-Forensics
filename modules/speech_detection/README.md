# Speech Deepfake Detection

This project detects AI-generated speech using log-mel spectrogram features and a hybrid CNN-Transformer model with attention mechanisms.

---

## Quick Demo

```bash
pip install -r requirements.txt
python demo/demo_predict.py --audio demo/sample.wav
```

Expected output:

```text
Prediction: fake
Probability: 0.87
```

---

## Input Requirements

- Format: `.wav`
- Duration: around 2 seconds
- Sample rate: 16000 Hz

---

## Model

The project uses a HybridAST model:

- CNN feature extractor
- Channel Attention
- Spatial Attention
- Transformer Encoder
- Binary classifier for real/fake speech detection

Pretrained model weights are included here:

```text
models/best_hybrid_ast.pth
```

This means the UI/demo team does not need to retrain the model.

---

## Project Structure

```text
speech-deepfake-detection/
├── data/
│   └── README.md
├── demo/
│   ├── README.md
│   └── demo_predict.py
├── models/
│   ├── README.md
│   └── best_hybrid_ast.pth
├── notebooks/
│   └── voice_detection_original.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── model.py
│   ├── predict.py
│   └── train_optuna.py
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Inference

Command-line inference:

```bash
python -m src.predict --audio demo/sample.wav
```

Demo script inference:

```bash
python demo/demo_predict.py --audio demo/sample.wav
```

The inference output should include:

- Prediction: `real` or `fake`
- Fake probability score

---

## UI Integration

For UI integration, use the prediction function from `src.predict`.

Example:

```python
from src.predict import predict_audio

result = predict_audio("demo/sample.wav")
print(result)
```

Expected return format:

```python
{
    "label": "fake",
    "probability": 0.87
}
```

If the actual function name or return format in `src.predict.py` is different, please follow the implementation in that file.

---

## Demo Data

The full training datasets are not included because of size and licensing restrictions.

However, for demo testing, please upload a small number of sample audio files to GitHub, for example:

```text
demo/sample.wav
demo/sample_fake.wav
demo/sample_real.wav
```

Recommended sample files:

- 1 real speech sample
- 1 fake speech sample
- `.wav` format
- around 2 seconds
- 16000 Hz sample rate

These files allow the UI/demo team to test the model without downloading the full dataset or retraining the model.

---

## Dataset Download

The original datasets can be downloaded separately.

Fake-or-Real Dataset:

```bash
kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset --unzip -p data
```

SceneFake Dataset:

```bash
kaggle datasets download -d mohammedabdeldayem/scenefake --unzip -p data/scenefake
```

Expected dataset structure:

```text
data/for-2sec/for-2seconds/
├── training/
├── validation/
└── testing/

data/scenefake/
├── train/
├── dev/
└── eval/
```

---

## Training

To retrain or tune the model:

```bash
python -m src.train_optuna --config config.yaml
```

Training includes:

1. Loading Fake-or-Real and SceneFake datasets
2. Merging and labeling audio data
3. Handling class imbalance with oversampling
4. Converting audio into log-mel spectrograms
5. Training the HybridAST model
6. Hyperparameter tuning with Optuna
7. Saving the best trained model

---

## Results

Final validation performance:

- AUC: 0.9506
- Accuracy: around 0.89
- F1-score: around 0.90
- Recall: around 0.99

The model is especially strong in detecting fake speech, with high recall.

---

## Notes for Demo/UI Team

- Model weights are included in `models/best_hybrid_ast.pth`
- Full datasets are not included
- A small demo audio file should be added under `demo/`
- Use `demo/demo_predict.py` or `src.predict.py` for inference
- No retraining is required for demo usage

---

## Key Skills Demonstrated

- Audio preprocessing
- Log-mel spectrogram feature extraction
- Speech deepfake detection
- CNN + Transformer hybrid modeling
- Attention mechanisms
- Handling imbalanced data
- Hyperparameter optimization with Optuna
- End-to-end machine learning pipeline
- Model inference and demo integration
