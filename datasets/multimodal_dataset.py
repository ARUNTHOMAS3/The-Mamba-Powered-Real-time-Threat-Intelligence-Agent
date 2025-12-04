# datasets/multimodal_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os, json

class MultimodalDataset(Dataset):
    """
    Dataset yields:
      x_log: (seq_len, logs_feat)
      x_text: (seq_len, text_feat)
      x_cve: (seq_len, cve_feat)
      label: binary 0/1 (attack in window)
    For quick start we include a synthetic data generator if files missing.
    """
    def __init__(self, path, seq_len=128, create_synthetic=True, n_samples=1000, seed=42):
        self.path = path
        self.seq_len = seq_len
        if os.path.exists(path):
            self.data = json.load(open(path))
        elif create_synthetic:
            np.random.seed(seed)
            self.data = []
            for i in range(n_samples):
                # create benign series with occasional attack windows
                label = 1 if np.random.rand() < 0.2 else 0
                # generate three modalities
                x_log = np.random.normal(0, 1, (seq_len, 32))
                x_text = np.random.normal(0, 1, (seq_len, 64))
                x_cve = np.random.normal(0, 1, (seq_len, 16))
                if label == 1:
                    # inject anomaly pattern near the tail
                    idx = np.random.randint(seq_len//2, seq_len-3)
                    x_log[idx:idx+3] += np.random.normal(3.0, 0.5, (3,32))
                    x_text[idx:idx+3] += np.random.normal(2.0, 0.3, (3,64))
                self.data.append({
                    "x_log": x_log.tolist(),
                    "x_text": x_text.tolist(),
                    "x_cve": x_cve.tolist(),
                    "label": int(label)
                })
        else:
            raise FileNotFoundError(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "x_log": torch.tensor(item["x_log"], dtype=torch.float32),
            "x_text": torch.tensor(item["x_text"], dtype=torch.float32),
            "x_cve": torch.tensor(item["x_cve"], dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.float32)
        }
