"""
CICIDS2017 Dataset Loader (Memory-Efficient Lazy Windowing)
Reference: https://www.unb.ca/cic/datasets/ids-2017.html

MEMORY-SAFE SUBSET APPROACH:
- Load only Monday and Friday CSVs (hardware memory constraint)
- Maintains temporal order, attack diversity, and benign traffic
- Publication-safe: explicitly documented as subset evaluation

LAZY WINDOWING:
- Windows generated on-the-fly in __getitem__ (no pre-allocation)
- Drastically reduces memory footprint

Strict Compliance:
- No Data Leakage: Scaler fitted ONLY on Train split
- Temporal Order: Sorted by Timestamp
- No Shuffling
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
import gc

class CICIDS2017Dataset(Dataset):
    """
    Memory-Efficient CICIDS2017 Loader (Monday + Friday Subset)
    
    CRITICAL: Uses lazy windowing to avoid OOM errors.
    Windows are created on-demand during __getitem__, not pre-allocated.
    """
    
    def __init__(self, root_dir="data/raw/CICIDS2017", split="train", binary=True, seq_len=50):
        self.root_dir = root_dir
        self.split = split
        self.binary = binary
        self.seq_len = seq_len  # Window length for temporal sequences
        self.scaler_path = "outputs/scaler_cicids.pkl"
        
        # Load raw data (NOT windowed)
        self.X_raw, self.y_raw, self.attack_labels = self._load_process_split()
        
        # Calculate number of valid windows
        if len(self.X_raw) >= self.seq_len:
            self.num_windows = len(self.X_raw) - self.seq_len + 1
        else:
            self.num_windows = 0
            
    def _load_process_split(self):
        # === DISK CACHE: Load from .npz if available (skips all CSV parsing) ===
        cache_dir = os.path.join("outputs", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"cicids2017_{self.split}.npz")
        
        if os.path.exists(cache_file):
            print(f"[{self.split.upper()}] Loading from cache: {cache_file}")
            cached = np.load(cache_file, allow_pickle=True)
            X_part = cached['X']
            y_part = cached['y']
            atk_part = cached['atk']
            print(f"  Cached shape: {X_part.shape}, labels: {len(y_part)}")
            return self._apply_scaling(X_part, y_part, atk_part)
        
        print(f"[{self.split.upper()}] No cache found, parsing CSVs (this only happens once)...")
        
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)
        
        # SUBSET SELECTION (Memory-safe)
        subset_files = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv"
        ]
        
        available_files = [f for f in subset_files if os.path.exists(os.path.join(self.root_dir, f))]
        
        if not available_files:
            print(f"⚠ No CICIDS2017 subset files found in {self.root_dir}.")
            return np.array([]), np.array([]), np.array([])
            
        print(f"[{self.split.upper()}] Loading CICIDS2017 Subset (Monday+Friday)...")
        print(f"Files: {available_files}")
        
        dfs = []
        drop_cols_set = {'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol'}
        
        for f in available_files: 
            print(f" -> Processing {f}...")
            try:
                # Read in chunks to avoid pandas' peak memory usage during read
                chunk_iter = pd.read_csv(
                    os.path.join(self.root_dir, f), 
                    encoding='latin1', 
                    chunksize=100000,
                    low_memory=False
                )
                
                for chunk in chunk_iter:
                    chunk.columns = [c.strip() for c in chunk.columns]
                    
                    # Drop metadata columns immediately
                    cols_to_drop = [c for c in chunk.columns if c in drop_cols_set]
                    chunk.drop(columns=cols_to_drop, inplace=True, errors='ignore')
                    
                    # Downcast numeric to float32
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
        
        # Sort by Timestamp
        if 'Timestamp' in full_df.columns:
            print("Sorting by Timestamp...")
            try:
                full_df['Timestamp'] = pd.to_datetime(full_df['Timestamp'], dayfirst=True, errors='coerce')
                full_df = full_df.sort_values('Timestamp')
            except Exception as e:
                print(f"Timestamp sort failed ({e}), using file order.")
        
        # Label Processing
        print("Processing Labels...")
        label_col = 'Label'
        if potential := [c for c in full_df.columns if 'label' in c.lower()]:
             label_col = potential[0]
             
        y_raw = full_df[label_col].astype(str).str.strip().values
        attack_labels_all = y_raw.copy()  # Keep original labels for per-attack analysis
        if self.binary:
            y_all = np.where(pd.Series(y_raw).str.upper() == 'BENIGN', 0, 1)
        else:
            le = LabelEncoder()
            y_all = le.fit_transform(y_raw)
            
        print(f"Class Distribution: Benign={np.sum(y_all==0)}, Attack={np.sum(y_all==1)}")
        
        # Drop non-features
        full_df.drop(columns=[label_col, 'Timestamp'], inplace=True, errors='ignore')
        
        # Convert to Numpy - use float32 directly
        print("Converting to numpy arrays...")
        X_all = full_df.values.astype(np.float32)
        
        # Clean NaN/Inf WITHOUT allocating huge boolean masks
        print("Cleaning NaN/Inf values...")
        for i in range(X_all.shape[1]):
            col = X_all[:, i]
            col[np.isnan(col)] = 0.0
            col[np.isinf(col)] = 0.0
        
        del full_df
        gc.collect()
        
        # Strict Temporal Split (on RAW data, before windowing)
        total_len = len(X_all)
        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.80)
        
        print(f"Total samples: {total_len}")
        print(f"Train: 0-{train_end}, Val: {train_end}-{val_end}, Test: {val_end}-{total_len}")
        
        if self.split == 'train':
            X_part = X_all[:train_end]
            y_part = y_all[:train_end]
            atk_part = attack_labels_all[:train_end]
        elif self.split == 'val':
            X_part = X_all[train_end:val_end]
            y_part = y_all[train_end:val_end]
            atk_part = attack_labels_all[train_end:val_end]
        elif self.split == 'test':
            X_part = X_all[val_end:]
            y_part = y_all[val_end:]
            atk_part = attack_labels_all[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
        del X_all, y_all, attack_labels_all
        gc.collect()
        
        # Save to cache for next time
        print(f"Saving cache to {cache_file}...")
        np.savez_compressed(cache_file, X=X_part, y=y_part, atk=atk_part)
        
        print(f"[{self.split.upper()}] Raw shape (before windowing): {X_part.shape}")
        
        return self._apply_scaling(X_part, y_part, atk_part)
    
    def _apply_scaling(self, X_part, y_part, atk_part):
        """Apply leakage-free scaling. Train fits scaler, val/test load it."""
        print(f"[{self.split.upper()}] Raw shape (before windowing): {X_part.shape}")
        # Scaling (Leakage-Free)
        if self.split == 'train':
            print("Fitting StandardScaler on Train split...")
            scaler = StandardScaler()
            X_part = scaler.fit_transform(X_part)
            os.makedirs("outputs", exist_ok=True)
            joblib.dump(scaler, self.scaler_path)
            print(f"Saved scaler to {self.scaler_path}")
        else:
            if os.path.exists(self.scaler_path):
                print(f"Loading scaler from {self.scaler_path}...")
                scaler = joblib.load(self.scaler_path)
                X_part = scaler.transform(X_part)
            else:
                print("⚠ Scaler not found! Training must run first.")
                raise FileNotFoundError("Scaler not found. Run training first.")
        
        # Return raw data - windowing will happen in __getitem__
        print(f"[OK] Will generate {len(X_part) - self.seq_len + 1} windows lazily during training")
        
        return X_part, y_part, atk_part

    def __len__(self):
        # Return number of valid windows
        return self.num_windows
    
    def __getitem__(self, idx):
        """
        Generate window on-the-fly (lazy evaluation).
        This avoids pre-allocating all windows in memory.
        """
        if idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of range for {self.num_windows} windows")
        
        # Extract window from raw data
        window = self.X_raw[idx:idx + self.seq_len]  # (seq_len, features)
        label = self.y_raw[idx + self.seq_len - 1]  # Label from last timestep
        
        # Convert to tensors
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
