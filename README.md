# Fake Music Detection with SpecTTTra

This project reproduces a fake music detection pipeline based on a SONICS-style setup.

The goal is to classify short music clips as:

- `0`: real music
- `1`: AI-generated music

The current experiment uses the **SpecTTTra** model on 5-second audio clips converted into Mel spectrograms.

---

## Project Structure

```text
Fake Music detection/
├── train.py
├── test.py
├── ai_music_quick_demo.py
├── ai_music_quick_demo.ipynb
├── train_results.json
├── requirements.txt
├── configs/
├── fake_music_detector/
├── sonics/
├── test_audio/
└── output/
    └── spectttra_gamma-t=5-stratified-v3-train80/
        ├── best_checkpoint.pth
        ├── last_checkpoint.pth
        ├── best_metrics.json
        ├── valid_predictions.csv
        └── test_predictions.csv
```

---

## Methodology

The pipeline is:

```text
audio file
→ 5-second clip
→ Mel spectrogram
→ SpecTTTra model
→ binary prediction
```

Audio settings:

| Item | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| Clip length | 5 seconds |
| Max samples | 80,000 |
| Input shape | 128 × 128 |

Model settings:

| Item | Value |
|---|---:|
| Model | SpecTTTra |
| Embedding dim | 384 |
| Attention heads | 6 |
| Transformer layers | 12 |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Epochs | 50 |
| Batch size | 32 |

---

## Dataset

The dataset is split into train, validation, and test sets.

| Split | Total | Real | Fake |
|---|---:|---:|---:|
| Train | 55,283 | 16,021 | 39,262 |
| Validation | 6,909 | 2,003 | 4,906 |
| Test | 6,909 | 2,003 | 4,906 |

Labels:

```text
0 = real music
1 = AI-generated music
```

Sources:

- Real: YouTube
- Fake: Suno, Udio

Because the dataset is imbalanced, F1 score, precision, recall, and confusion matrix are more useful than accuracy alone.

---

## Setup

```bash
cd "/Users/jingbo/Documents/NLP/Fake Music detection/Fake Music detection"

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If using `uv`:

```bash
uv pip install -r requirements.txt
```

---

## Train

```bash
python train.py \
  --config configs/your_train_config.yaml
```

The output will be saved to:

```text
output/spectttra_gamma-t=5-stratified-v3-train80/
```

---

## Evaluate

```bash
python test.py \
  --config configs/your_test_config.yaml \
  --ckpt_path output/spectttra_gamma-t=5-stratified-v3-train80/best_checkpoint.pth
```

Important output files:

| File | Meaning |
|---|---|
| `best_checkpoint.pth` | Best saved model |
| `last_checkpoint.pth` | Final epoch model |
| `best_metrics.json` | Best validation metrics |
| `valid_predictions.csv` | Validation predictions |
| `test_predictions.csv` | Test predictions |

---

## Quick Demo

Put custom audio files in:

```text
test_audio/
```

Then run:

```bash
python ai_music_quick_demo.py \
  --ckpt_path output/spectttra_gamma-t=5-stratified-v3-train80/best_checkpoint.pth \
  --audio_path test_audio/your_audio_file.wav
```

Output interpretation:

| Fake Probability | Meaning |
|---:|---|
| 0.00–0.30 | likely real |
| 0.30–0.70 | uncertain |
| 0.70–1.00 | likely AI-generated |
