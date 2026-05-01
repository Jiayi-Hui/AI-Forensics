# Speech Deepfake Detection

This project detects AI-generated speech using log-mel spectrogram features and a hybrid CNN-Transformer model with attention mechanisms.

## Project Overview

This project builds an end-to-end pipeline for speech deepfake detection:

- Combine multiple datasets (Fake-or-Real + SceneFake)
- Handle class imbalance with oversampling
- Convert audio into log-mel spectrograms
- Train a HybridAST model (CNN + Transformer)
- Apply Channel Attention and Spatial Attention
- Tune hyperparameters using Optuna
- Train final model and run inference

## Datasets

### Fake-or-Real Dataset

data/for-2sec/for-2seconds/
- training/
- validation/
- testing/

### SceneFake Dataset

data/scenefake/
- train/
- dev/
- eval/

Datasets are not included due to size and licensing.

## Model Architecture

- Log-mel spectrogram input
- CNN feature extractor
- Channel Attention
- Spatial Attention
- Transformer Encoder
- Binary classifier (real vs fake)

## Training Pipeline

1. Load datasets
2. Merge and label data
3. Oversample to balance classes
4. Convert audio → log-mel spectrogram
5. Train model
6. Hyperparameter tuning (Optuna)
7. Final training with best parameters
8. Save trained model

## Results

Final Validation Performance:

- AUC: 0.9506
- Accuracy: ~0.89
- F1-score: ~0.90
- Recall: ~0.99

The model shows strong ability to detect fake speech, especially high recall.

## Project Structure

speech-deepfake-detection/
- src/
- data/
- models/
- notebooks/
- config.yaml
- requirements.txt
- README.md

## Installation

pip install -r requirements.txt

## Dataset Download

Fake-or-Real:

kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset --unzip -p data

SceneFake:

kaggle datasets download -d mohammedabdeldayem/scenefake --unzip -p data/scenefake

## Training

python -m src.train_optuna --config config.yaml

## Inference

python -m src.predict --audio path/to/audio.wav

Output:
- Prediction (real / fake)
- Fake probability score

## Notes

- Model weights are not included
- Dataset is not included
- Original notebook is provided for reference

## Key Skills Demonstrated

- Audio preprocessing (log-mel spectrogram)
- Deepfake speech detection
- CNN + Transformer hybrid modeling
- Attention mechanisms
- Handling imbalanced data
- Hyperparameter optimization (Optuna)
- End-to-end ML pipeline design
- Model deployment (inference)

