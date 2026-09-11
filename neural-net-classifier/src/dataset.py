import numpy as np
import torch
from torch.utils.data import Dataset


class LoanDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X.astype(np.float32))
        self.y = torch.FloatTensor(y.values if hasattr(y, "values") else y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
