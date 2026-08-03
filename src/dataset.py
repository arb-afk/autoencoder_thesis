"""PyTorch Dataset for keystroke sliding-window sequences."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class KeystrokeWindowDataset(Dataset):
    """Autoencoder dataset: each sample is a (seq_len, 2) HT/FT window."""

    def __init__(self, windows):
        self.windows = torch.as_tensor(windows, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]
