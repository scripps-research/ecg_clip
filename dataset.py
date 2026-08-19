# Copyright (c) 2026 The Scripps Research Institute
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset


class PretrainingDataset(Dataset):
    """
    Stub dataset for ECG-text contrastive pretraining.

    Returns random (8, 5000) ECG tensors paired with a placeholder text report.
    Replace with real data loading (ECG arrays + paired clinical reports).
    """

    def __init__(self, size: int = 10000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ecg = torch.randn(8, 5000, dtype=torch.float32)
        text = "Normal sinus rhythm. No acute ST-T wave changes."
        return {"ecg": ecg, "text": text}


class FinetuneDataset(Dataset):
    """
    Stub dataset — replace with your actual data loading logic.

    Currently returns random (8, 5000) ECG tensors (8 leads, 5000 timepoints)
    and random binary float labels. The __getitem__ dict keys ("ecg", "label")
    must be preserved so finetune.py can consume this dataset without changes.
    """

    def __init__(self, size: int = 10000):
        # TODO: replace `size` with your real data source (file paths, dataframe, etc.)
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # TODO: load real ECG (8 leads × 5000 timepoints) and its label here
        ecg = torch.randn(8, 5000, dtype=torch.float32)
        label = torch.randint(0, 2, (1,)).float().squeeze()
        return {"ecg": ecg, "label": label}

