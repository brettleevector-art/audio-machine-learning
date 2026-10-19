
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


SR = 22050
N = SR * 5


def waveform_to_mel_spectrogram(y):
    y = librosa.util.fix_length(y, size=N)

    m = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128,
    )

    m = librosa.power_to_db(m, ref=np.max)
    m = (m + 40) / 40

    return torch.tensor(
        m,
        dtype=torch.float32,
    ).unsqueeze(0)


def load_audio_spectrogram(path):
    y, _ = librosa.load(str(path), sr=SR)
    return waveform_to_mel_spectrogram(y)


class ESC50(Dataset):
    def __init__(self, root, folds):
        self.root = root

        df = pd.read_csv(f"{root}/meta/esc50.csv")
        self.df = df[df.fold.isin(folds)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        audio_path = f"{self.root}/audio/{r.filename}"
        x = load_audio_spectrogram(audio_path)

        return x, int(r.target)
