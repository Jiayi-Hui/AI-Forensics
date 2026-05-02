# Demo

This folder provides a simple command-line demo for speech deepfake detection.

## Run Demo

After training the model, run:

python demo/demo_predict.py --audio path/to/audio.wav

Example:

python demo/demo_predict.py --audio data/scenefake/dev/real/B_0092_10_B.wav

Example output:

Prediction: real  
Fake probability: 0.0002

## Notes

The trained model file is expected at:

models/best_hybrid_ast.pth

The model file is not included in this repository due to file size constraints. Please train the model first by running:

python -m src.train_optuna --config config.yaml