"""
CICIDS2017 Dataset Loader
Reference: https://www.unb.ca/cic/datasets/ids-2017.html

This dataset contains benign and the most up-to-date common attacks, 
which resembles the true real-world data (PCAPs).
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class CICIDS2017Dataset(Dataset):
    """
    Loader for CICIDS2017 Network Intrusion Detection Dataset
    
    Download from: https://www.unb.ca/cic/datasets/ids-2017.html
    Place CSV files in: data/raw/CICIDS2017/
    
    Features: 78 network flow features
    Labels: Multi-class (Benign, DoS, DDoS, Web Attack, etc.)
    """
    
    def __init__(self, root_dir="data/raw/CICIDS2017", train=True, binary=True, max_samples=None):
        """
        Args:
            root_dir: Path to CICIDS2017 CSV files
            train: If True, load training set (70%), else test set (30%)
            binary: If True, convert to binary (benign/attack), else multi-class
            max_samples: Limit dataset size for quick experiments
        """
        self.root_dir = root_dir
        self.train = train
        self.binary = binary
        
        # Load all CSV files
        self.data = self._load_data()
        
        if max_samples:
            self.data = self.data.sample(n=min(max_samples, len(self.data)), random_state=42)
        
        # Preprocess
        self.X, self.y = self._preprocess()
        
        # Train/Test Split (70/30)
        split_idx = int(0.7 * len(self.X))
        if train:
            self.X = self.X[:split_idx]
            self.y = self.y[:split_idx]
        else:
            self.X = self.X[split_idx:]
            self.y = self.y[split_idx:]
    
    def _load_data(self):
        """Load and concatenate all CSV files"""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"\n❌ CICIDS2017 not found at: {self.root_dir}\n"
                f"📥 Download from: https://www.unb.ca/cic/datasets/ids-2017.html\n"
                f"📁 Extract CSVs to: {self.root_dir}/"
            )
        
        csv_files = [f for f in os.listdir(self.root_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.root_dir}")
        
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(os.path.join(self.root_dir, csv_file), encoding='latin1')
                dfs.append(df)
                print(f"✓ Loaded {csv_file}: {len(df)} samples")
            except Exception as e:
                print(f"⚠ Skipping {csv_file}: {e}")
        
        data = pd.concat(dfs, ignore_index=True)
        print(f"📊 Total CICIDS2017 samples: {len(data)}")
        return data
    
    def _preprocess(self):
        """Clean and normalize features"""
        df = self.data.copy()
        
        # Typical label column name in CICIDS2017
        label_col = 'Label'
        if label_col not in df.columns:
            label_col = [col for col in df.columns if 'label' in col.lower()][0]
        
        # Extract labels
        labels = df[label_col].values
        
        # Convert to binary if needed
        if self.binary:
            y = np.array([0 if 'BENIGN' in str(l).upper() else 1 for l in labels])
        else:
            le = LabelEncoder()
            y = le.fit_transform(labels)
        
        # Drop non-numeric columns
        X = df.drop(columns=[label_col], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        
        # Handle inf/nan
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'label': self.y[idx].float()
        }


def get_cicids_stats():
    """Print dataset statistics"""
    try:
        train_ds = CICIDS2017Dataset(train=True, binary=True, max_samples=10000)
        test_ds = CICIDS2017Dataset(train=False, binary=True, max_samples=10000)
        
        print("\n" + "="*60)
        print("CICIDS2017 Dataset Statistics")
        print("="*60)
        print(f"Train samples: {len(train_ds)}")
        print(f"Test samples: {len(test_ds)}")
        print(f"Feature dimension: {train_ds.X.shape[1]}")
        print(f"Attack ratio (train): {train_ds.y.float().mean():.1%}")
        print(f"Attack ratio (test): {test_ds.y.float().mean():.1%}")
        print("="*60)
    except Exception as e:
        print(f"❌ Could not load CICIDS2017: {e}")


if __name__ == "__main__":
    get_cicids_stats()
