
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
import time
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.cicids2017_loader import CICIDS2017Dataset
from models.tabular_models import LSTMClassifier
from utils.reproducibility import set_seeds, log_reproducibility_info, print_system_summary, compute_config_hash
from utils.config_loader import load_config, get_hyperparams

def count_params(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):
    # Load centralized configuration
    config_path = args.config
    config = load_config(config_path)
    config_hash = compute_config_hash(config_path)
    hparams = get_hyperparams(config)
    
    # Reproducibility Setup with CLI seed
    seed = args.seed if args.seed is not None else config['reproducibility']['default_seed']
    set_seeds(seed)
    print_system_summary()
    log_reproducibility_info(seed=seed, config_hash=config_hash)
    
    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[LSTM Tabular] Starting Training on {DEVICE}...")
    print(f"Random Seed: {seed}")
    print(f"Config Hash: {config_hash[:16]}...\n")
    
    # Extract hyperparameters from config
    SEQ_LEN = hparams['seq_len']
    BATCH_SIZE = hparams['batch_size']
    EPOCHS = hparams['epochs']
    LR = hparams['lr']
    D_MODEL = hparams['d_model']
    N_LAYERS = hparams['n_layers']
    
    # 1. Load Data (Pre-windowed by dataset loader)
    # CRITICAL: Windowing is now done in CICIDS2017Dataset BEFORE splitting
    # This prevents boundary leakage where windows span train/val/test splits
    print("Loading CICIDS2017 (Train) - Pre-windowed...")
    train_ds = CICIDS2017Dataset(split='train', seq_len=SEQ_LEN)
    if len(train_ds) == 0:
        print("❌ Dataset empty. Please add CSVs to data/raw/CICIDS2017/")
        return
        
    # No shuffling as per strict temporal requirement
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Val
    print("Loading CICIDS2017 (Val) - Pre-windowed...")
    val_ds = CICIDS2017Dataset(split='val', seq_len=SEQ_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Input dimension from dataset
    # With lazy windowing, get dimension from raw data
    input_dim = train_ds.X_raw.shape[1]  # Number of features
    
    print(f"Input Dimension: {input_dim} features\n")
    
    # 2. Model
    model = LSTMClassifier(input_dim=input_dim, d_model=D_MODEL, n_layers=N_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    # Parameter count logging for fair comparison
    params = count_params(model)
    print(f"[MODEL PARAMS] {model.__class__.__name__}: {params:,}")
    
    # Save parameter count for cross-model comparison
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/param_counts.txt", "w") as f:  # Overwrite to ensure clean comparison
        f.write(f"LSTM: {params}\n")
    
    print(f"  Parameter count logged to outputs/param_counts.txt\n")
    
    # 3. Training Loop with Best Model Tracking
    best_val_f1 = 0.0
    best_epoch = 0
    training_history = []
    
    print(f"Training for {EPOCHS} epochs (No Shuffling - Strict Temporal Order)...\n")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10} {'Val F1':<10} {'Time':<8} {'Best':<6}")
    print("="*76)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        t0 = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits.squeeze(), y)
                val_loss += loss.item()
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Metrics
        val_acc = np.mean(np.array(all_preds) == np.array(all_targets))
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        epoch_time = time.time() - t0
        
        # Track best model
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            # Save best model
            torch.save(model.state_dict(), "outputs/lstm_tabular.pth")
        
        # Log
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "time_seconds": epoch_time,
            "is_best": is_best
        })
        
        # Print progress
        best_marker = "  *" if is_best else ""
        print(f"{epoch+1:<6} {total_loss/len(train_loader):<12.4f} {val_loss/len(val_loader):<12.4f} "
              f"{val_acc:<10.4f} {val_f1:<10.4f} {epoch_time:<8.1f}s {best_marker}")
    
    # Save training history
    print("\n" + "="*76)
    print(f"Training Complete!")
    print(f"Best Validation F1: {best_val_f1:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: outputs/lstm_tabular.pth")
    
    with open("outputs/lstm_training_history.json", "w") as f:
        json.dump({
            "model": "LSTM_Tabular",
            "best_val_f1": float(best_val_f1),
            "best_epoch": best_epoch,
            "parameter_count": params,
            "config_hash": config_hash,
            "seed": seed,
            "hyperparameters": hparams,
            "history": training_history
        }, f, indent=2)
    
    print(f"Training history saved to: outputs/lstm_training_history.json\n")
    
    return params  # Return for capacity comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model on CICIDS2017")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to config file")
    
    args = parser.parse_args()
    train(args)
