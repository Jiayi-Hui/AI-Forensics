import numpy as np
import librosa


SAMPLE_RATE = 16000
N_MFCC = 20


def summarize_1d(x):
    return np.array([
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.median(x),
        np.percentile(x, 25),
        np.percentile(x, 75)
    ], dtype=np.float32)


def summarize_2d(feature):
    summary = []
    for row in feature:
        summary.extend(summarize_1d(row))
    return np.array(summary, dtype=np.float32)


def extract_enhanced_features(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    target_len = sr * 2
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    feature_vector = np.concatenate([
        summarize_2d(mfcc),
        summarize_2d(delta_mfcc),
        summarize_2d(delta2_mfcc),
        summarize_2d(chroma),
        summarize_2d(spectral_contrast),
        summarize_2d(log_mel),
        summarize_1d(centroid),
        summarize_1d(bandwidth),
        summarize_1d(rolloff),
        summarize_1d(flatness),
        summarize_1d(zcr),
        summarize_1d(rms)
    ])

    return feature_vector.reshape(1, -1)