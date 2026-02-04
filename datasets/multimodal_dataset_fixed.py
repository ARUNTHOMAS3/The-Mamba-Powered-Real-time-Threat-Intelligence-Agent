"""
FIXED: Proper MultimodalDataset with Train/Test Split & Data Validation
=========================================================================
Purpose:
- Load separate train/test files (NO LEAKAGE)
- Normalize features properly
- Validate class balance
- Provide diagnostic info
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os


class MultimodalDatasetFixed(Dataset):
    """
    Loads multimodal time-series data with proper train/test split.
    
    Expected data format (JSON):
    [
        {
            "x_log": [[...], [...], ...],      # (seq_len, 32)
            "x_text": [[...], [...], ...],     # (seq_len, 64)
            "x_cve": [[...], [...], ...],      # (seq_len, 16)
            "label": 0 or 1
        },
        ...
    ]
    """
    
    def __init__(self, data_path: str, split: str = "train", normalize: bool = True, seq_len: int = 128):
        """
        Args:
            data_path: Path to JSON file (WITHOUT _train/_test suffix)
            split: "train" or "test"
            normalize: If True, normalize features to zero mean, unit variance
        """
        self.data_path = data_path
        self.split = split
        self.normalize = normalize
        self.seq_len = seq_len
        
        # Load data
        if split == "train":
            full_path = data_path.replace(".json", "") + "_train.json"
        elif split == "test":
            full_path = data_path.replace(".json", "") + "_test.json"
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found: {full_path}")
        
        with open(full_path, "r") as f:
            raw_data = json.load(f)
        
        # Validate
        assert len(raw_data) > 0, "Empty dataset"
        assert all("x_log" in item and "x_text" in item and "x_cve" in item and "label" in item 
                  for item in raw_data), "Invalid data format"
        
        # Normalize if requested
        if normalize:
            raw_data = self._normalize_features(raw_data)
        
        self.data = raw_data
        
        # Diagnostics
        self.labels = np.array([x["label"] for x in self.data])
        self.n_normal = np.sum(self.labels == 0)
        self.n_attack = np.sum(self.labels == 1)
        
        print(f"\n✓ Loaded {split} split: {len(self.data)} samples")
        print(f"  Normal: {self.n_normal} ({100*self.n_normal/len(self.data):.1f}%)")
        print(f"  Attack: {self.n_attack} ({100*self.n_attack/len(self.data):.1f}%)")
    
    def _normalize_features(self, data: list) -> list:
        """Normalize each feature modality to zero mean, unit variance."""
        # Collect all feature arrays by modality
        all_logs = np.array([item["x_log"] for item in data])  # (N, seq, 32)
        all_texts = np.array([item["x_text"] for item in data])  # (N, seq, 64)
        all_cves = np.array([item["x_cve"] for item in data])  # (N, seq, 16)
        
        # Compute stats per feature dimension
        log_mean, log_std = all_logs.mean(axis=(0, 1), keepdims=True), all_logs.std(axis=(0, 1), keepdims=True)
        text_mean, text_std = all_texts.mean(axis=(0, 1), keepdims=True), all_texts.std(axis=(0, 1), keepdims=True)
        cve_mean, cve_std = all_cves.mean(axis=(0, 1), keepdims=True), all_cves.std(axis=(0, 1), keepdims=True)
        
        # Avoid division by zero
        log_std = np.where(log_std < 1e-6, 1.0, log_std)
        text_std = np.where(text_std < 1e-6, 1.0, text_std)
        cve_std = np.where(cve_std < 1e-6, 1.0, cve_std)
        
        # Normalize
        normalized_data = []
        for item in data:
            x_log = (np.array(item["x_log"]) - log_mean) / log_std
            x_text = (np.array(item["x_text"]) - text_mean) / text_std
            x_cve = (np.array(item["x_cve"]) - cve_mean) / cve_std
            
            normalized_data.append({
                "x_log": x_log.tolist(),
                "x_text": x_text.tolist(),
                "x_cve": x_cve.tolist(),
                "label": item["label"],
                "attack_span": item.get("attack_span"),
            })
        
        return normalized_data
    
    def __len__(self):
        return len(self.data)
    
    def _pad_or_crop(self, arr: np.ndarray, label: int, attack_span=None):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        n = len(arr)
        if n == self.seq_len:
            return arr, 0
        if n > self.seq_len:
            if attack_span and label == 1:
                start_span, end_span = attack_span
                min_start = max(0, end_span - self.seq_len)
                max_start = min(start_span, n - self.seq_len)
                if max_start < min_start:
                    start = min_start
                else:
                    start = np.random.randint(min_start, max_start + 1)
            else:
                start = np.random.randint(0, n - self.seq_len + 1)
            return arr[start:start + self.seq_len], start
        # pad
        pad_len = self.seq_len - n
        pad = np.zeros((pad_len, arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0), 0

def __getitem__(self, idx):
    item = self.data[idx]
    attack_span = item.get("attack_span")

    x_log, start_idx = self._pad_or_crop(
        np.array(item["x_log"], dtype=np.float32),
        item["label"],
        attack_span
    )

    x_text, _ = self._pad_or_crop(
        np.array(item["x_text"], dtype=np.float32),
        item["label"],
        attack_span
    )

    x_cve, _ = self._pad_or_crop(
        np.array(item["x_cve"], dtype=np.float32),
        item["label"],
        attack_span
    )

    y = float(item["label"])

    # window-level label correction
    if attack_span and y == 1:
        span_start, span_end = attack_span
        window_start = start_idx
        window_end = start_idx + self.seq_len

        overlap = not (
            span_end <= window_start or
            span_start >= window_end
        )

        if not overlap:
            y = 0.0

    return {
        "x_log": torch.tensor(x_log, dtype=torch.float32),
        "x_text": torch.tensor(x_text, dtype=torch.float32),
        "x_cve": torch.tensor(x_cve, dtype=torch.float32),
        "label": torch.tensor(y, dtype=torch.float32),
    }



if __name__ == "__main__":
    # Test loading
    try:
        train_ds = MultimodalDatasetFixed("data/processed/synth_balanced", split="train", normalize=True)
        test_ds = MultimodalDatasetFixed("data/processed/synth_balanced", split="test", normalize=True)
        print("\n✓ Dataset loading works!")
        
        # Test batch
        sample = train_ds[0]
        print(f"\nSample shapes:")
        print(f"  x_log: {sample['x_log'].shape}")
        print(f"  x_text: {sample['x_text'].shape}")
        print(f"  x_cve: {sample['x_cve'].shape}")
        print(f"  label: {sample['label'].item()}")
    except Exception as e:
        print(f"Error: {e}")
