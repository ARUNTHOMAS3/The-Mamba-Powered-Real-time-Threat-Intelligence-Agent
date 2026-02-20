"""
CIC-IDS2018 Dataset Loader (Memory-Efficient Lazy Windowing)
Reference: https://www.unb.ca/cic/datasets/ids-2018.html

Follows the same interface as CICIDS2017/UNSW-NB15 loaders:
- Lazy windowing (windows generated on-the-fly)
- Strict temporal split (70/10/20)
- Leakage-free scaling (scaler fitted on train only)
- Binary + multi-class label support

Expected data: Processed CSV files from the "Processed Traffic Data for ML Algorithms" folder.
Download command:
    aws s3 sync --no-sign-request --region ap-south-1 \
        "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
        data/raw/CIC-IDS2018/ --exclude "*" --include "*.csv"
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
import gc


class CICIDS2018Dataset(Dataset):
    """
    Memory-efficient CIC-IDS2018 loader with lazy windowing.
    
    Expected CSV files in root_dir (processed traffic data):
        Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
        Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
        Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
        Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
        Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
        Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
        Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
        Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
        Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
        Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv
    """
    
    def __init__(self, root_dir="data/raw/CIC-IDS2018", split="train", binary=True, seq_len=50):
        self.root_dir = root_dir
        self.split = split
        self.binary = binary
        self.seq_len = seq_len
        self.scaler_path = "outputs/scaler_cicids2018.pkl"
        
        self.X_raw, self.y_raw, self.attack_labels = self._load_process_split()
        
        if len(self.X_raw) >= self.seq_len:
            self.num_windows = len(self.X_raw) - self.seq_len + 1
        else:
            self.num_windows = 0
    
    def _load_process_split(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)
        
        # Find all CSV files in the directory
        csv_files = sorted([
            f for f in os.listdir(self.root_dir)
            if f.endswith('.csv')
        ])
        
        if not csv_files:
            print(f"⚠ No CIC-IDS2018 CSV files found in {self.root_dir}.")
            print(f"  Download with:")
            print(f'  aws s3 sync --no-sign-request --region ap-south-1 '
                  f'"s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" '
                  f'{self.root_dir}/ --exclude "*" --include "*.csv"')
            return np.array([]), np.array([]), np.array([])
        
        print(f"[{self.split.upper()}] Loading CIC-IDS2018...")
        print(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        drop_cols_set = {'flow id', 'src ip', 'src port', 'dst ip', 'dst port', 'protocol'}
        
        for f in csv_files:
            print(f"  -> Processing {f}...")
            try:
                chunk_iter = pd.read_csv(
                    os.path.join(self.root_dir, f),
                    encoding='latin1',
                    chunksize=100000,
                    low_memory=False
                )
                
                for chunk in chunk_iter:
                    chunk.columns = [c.strip() for c in chunk.columns]
                    
                    # Drop metadata columns
                    cols_to_drop = [c for c in chunk.columns if c.strip().lower() in drop_cols_set]
                    chunk.drop(columns=cols_to_drop, inplace=True, errors='ignore')
                    
                    # Downcast numeric
                    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                    chunk[numeric_cols] = chunk[numeric_cols].astype(np.float32)
                    
                    dfs.append(chunk)
                
                gc.collect()
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not dfs:
            return np.array([]), np.array([]), np.array([])
        
        print("Concatenating chunks...")
        full_df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        
        print(f"Loaded {len(full_df)} total rows.")
        
        # Sort by Timestamp if available
        ts_col = None
        for c in full_df.columns:
            if 'timestamp' in c.lower():
                ts_col = c
                break
        
        if ts_col:
            print("Sorting by Timestamp...")
            try:
                full_df[ts_col] = pd.to_datetime(full_df[ts_col], dayfirst=True, errors='coerce')
                full_df = full_df.sort_values(ts_col)
            except Exception as e:
                print(f"Timestamp sort failed ({e}), using file order.")
        
        # Label processing
        label_col = None
        for c in full_df.columns:
            if 'label' in c.lower():
                label_col = c
                break
        
        if label_col is None:
            print("⚠ No label column found! Using last column as label.")
            label_col = full_df.columns[-1]
        
        attack_labels = full_df[label_col].astype(str).str.strip().values
        
        if self.binary:
            y_all = np.where(
                np.isin(np.char.lower(attack_labels.astype('<U100')), ['benign', 'normal']),
                0, 1
            )
        else:
            le = LabelEncoder()
            y_all = le.fit_transform(attack_labels)
        
        print(f"Class Distribution: Benign={np.sum(y_all == 0)}, Attack={np.sum(y_all == 1)}")
        
        # Drop non-feature columns
        cols_to_drop = [label_col]
        if ts_col:
            cols_to_drop.append(ts_col)
        full_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Convert remaining object columns
        for col in full_df.select_dtypes(include=['object', 'datetime64']).columns:
            try:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            except:
                le = LabelEncoder()
                full_df[col] = le.fit_transform(full_df[col].astype(str))
        
        # Convert to float32
        X_all = full_df.values.astype(np.float32)
        
        # Clean NaN/Inf
        for i in range(X_all.shape[1]):
            col = X_all[:, i]
            col[np.isnan(col)] = 0.0
            col[np.isinf(col)] = 0.0
        
        del full_df
        gc.collect()
        
        # Strict temporal split
        total_len = len(X_all)
        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.80)
        
        print(f"Total samples: {total_len}")
        print(f"Train: 0-{train_end}, Val: {train_end}-{val_end}, Test: {val_end}-{total_len}")
        
        if self.split == 'train':
            X_part = X_all[:train_end]
            y_part = y_all[:train_end]
            atk_part = attack_labels[:train_end]
        elif self.split == 'val':
            X_part = X_all[train_end:val_end]
            y_part = y_all[train_end:val_end]
            atk_part = attack_labels[train_end:val_end]
        elif self.split == 'test':
            X_part = X_all[val_end:]
            y_part = y_all[val_end:]
            atk_part = attack_labels[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        del X_all, y_all, attack_labels
        gc.collect()
        
        # Leakage-free scaling
        if self.split == 'train':
            scaler = StandardScaler()
            X_part = scaler.fit_transform(X_part)
            os.makedirs("outputs", exist_ok=True)
            joblib.dump(scaler, self.scaler_path)
            print(f"Saved scaler to {self.scaler_path}")
        else:
            if os.path.exists(self.scaler_path):
                scaler = joblib.load(self.scaler_path)
                X_part = scaler.transform(X_part)
            else:
                raise FileNotFoundError("Scaler not found. Run training split first.")
        
        print(f"[{self.split.upper()}] Raw shape: {X_part.shape}")
        print(f"[OK] Will generate {max(0, len(X_part) - self.seq_len + 1)} windows lazily")
        
        return X_part, y_part, atk_part
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        if idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of range for {self.num_windows} windows")
        
        window = self.X_raw[idx:idx + self.seq_len]
        label = self.y_raw[idx + self.seq_len - 1]
        
        return torch.FloatTensor(window), torch.tensor(label, dtype=torch.float32)
    
    def get_attack_types(self):
        """Return unique attack types in this split."""
        if len(self.attack_labels) == 0:
            return []
        return list(set(self.attack_labels))
    
    def get_attack_label(self, idx):
        """Get the attack category label for a specific window."""
        if idx >= self.num_windows:
            raise IndexError
        return self.attack_labels[idx + self.seq_len - 1]
    
    def get_feature_count(self):
        """Return number of features."""
        if len(self.X_raw) == 0:
            return 0
        return self.X_raw.shape[1]
