import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

def read_dataset(file_path):
    data = pd.read_csv(file_path)

    return data

class ModelDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features: torch.Tensor = torch.tensor(features, dtype=torch.float32)
        self.labels: torch.Tensor = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
