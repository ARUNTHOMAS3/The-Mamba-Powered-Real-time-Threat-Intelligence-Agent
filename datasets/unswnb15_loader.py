"""
UNSW-NB15 Dataset Loader
Reference: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Modern network intrusion dataset with realistic attack patterns (2015)
49 features with 9 attack categories
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


class UNSWNB15Dataset(Dataset):
    """
    Loader for UNSW-NB15 Network Intrusion Detection Dataset
    
    Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    Place CSV files in: data/raw/UNSW-NB15/
    
    Features: 49 network flow features
    Labels: Binary (0=normal, 1=attack) or 10-class
    """
    
    def __init__(self, root_dir="data/raw/UNSW-NB15", train=True, binary=True):
        """
        Args:
            root_dir: Path to UNSW-NB15 CSV files
            train: If True, load UNSW_NB15_training-set.csv, else testing-set
            binary: If True, use binary labels, else multi-class
        """
        self.root_dir = root_dir
        self.train = train
        self.binary = binary
        
        # Load data
        self.data = self._load_data()
        
        # Preprocess
        self.X, self.y = self._preprocess()
    
    def _load_data(self):
        """Load train or test CSV"""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"\n❌ UNSW-NB15 not found at: {self.root_dir}\n"
                f"📥 Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
                f"📁 Extract CSVs to: {self.root_dir}/"
            )
        
        filename = "UNSW_NB15_training-set.csv" if self.train else "UNSW_NB15_testing-set.csv"
        filepath = os.path.join(self.root_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ {filename} not found in {self.root_dir}")
        
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded {filename}: {len(df)} samples")
        return df
    
    def _preprocess(self):
        """Clean and normalize features"""
        df = self.data.copy()
        
        # UNSW-NB15 has 'attack_cat' (category) and 'label' (binary)
        if self.binary:
            if 'label' in df.columns:
                y = df['label'].values
            else:
                # Fallback: any attack category → 1
                y = (df['attack_cat'] != 'Normal').astype(int).values
        else:
            le = LabelEncoder()
            y = le.fit_transform(df['attack_cat'].values)
        
        # Drop label columns
        X = df.drop(columns=['label', 'attack_cat', 'id'], errors='ignore')
        
        # Convert categorical features (proto, service, state)
        for col in ['proto', 'service', 'state']:
            if col in X.columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Select numeric columns
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


def get_unswnb15_stats():
    """Print dataset statistics"""
    try:
        train_ds = UNSWNB15Dataset(train=True, binary=True)
        test_ds = UNSWNB15Dataset(train=False, binary=True)
        
        print("\n" + "="*60)
        print("UNSW-NB15 Dataset Statistics")
        print("="*60)
        print(f"Train samples: {len(train_ds)}")
        print(f"Test samples: {len(test_ds)}")
        print(f"Feature dimension: {train_ds.X.shape[1]}")
        print(f"Attack ratio (train): {train_ds.y.float().mean():.1%}")
        print(f"Attack ratio (test): {test_ds.y.float().mean():.1%}")
        print("="*60)
    except Exception as e:
        print(f"❌ Could not load UNSW-NB15: {e}")


if __name__ == "__main__":
    get_unswnb15_stats()
