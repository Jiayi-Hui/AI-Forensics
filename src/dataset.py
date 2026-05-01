import torch
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset


class AudioDeepfakeDataset(Dataset):
    def __init__(self, dataframe, sample_rate=16000, n_mels=128, max_len=130):
        self.data = dataframe
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]["file_path"]
        label_str = self.data.iloc[idx]["label"]
        label = 1 if label_str == "fake" else 0

        try:
            waveform, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=True
            )

            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=self.n_mels
            )

            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            log_mel_spec = torch.tensor(log_mel_spec, dtype=torch.float32)
            log_mel_spec = log_mel_spec.unsqueeze(0)

            if log_mel_spec.shape[2] > self.max_len:
                log_mel_spec = log_mel_spec[:, :, :self.max_len]
            else:
                padding = self.max_len - log_mel_spec.shape[2]
                log_mel_spec = F.pad(log_mel_spec, (0, padding))

        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {file_path}") from e

        return log_mel_spec, torch.tensor(label, dtype=torch.float32)