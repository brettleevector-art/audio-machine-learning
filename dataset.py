
import torch, librosa, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader

SR, N = 22050, 22050 * 5

class ESC50(Dataset):
    def __init__(self, root, folds):
        df = pd.read_csv(f"{root}/meta/esc50.csv")
        self.df = df[df.fold.isin(folds)].reset_index(drop=True)
        self.root = root
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        y, _ = librosa.load(f"{self.root}/audio/{r.filename}", sr=SR)
        y = librosa.util.fix_length(y, size=N)
        m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128)
        m = librosa.power_to_db(m, ref=np.max)   # range -80..0
        m = (m + 40) / 40                        # roughly -1..1
        return torch.tensor(m, dtype=torch.float32).unsqueeze(0), int(r.target)
