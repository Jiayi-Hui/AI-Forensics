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
