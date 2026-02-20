
import torch
import torch.nn as nn
import sys
import os
import json
import platform
# import psutil  <-- Removed
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.cicids2017_loader import CICIDS2017Dataset
from models.tabular_models import MambaClassifier, LSTMClassifier
from utils.reproducibility import set_seeds

# Initialize seeds for reproducibility
set_seeds(42)

# --- CONFIG ---
SEQ_LEN = 50
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_system_info():
    return {
        "System": platform.system(),
        "Version": platform.version(),
        "Processor": platform.processor(),
        "Python": sys.version,
        "PyTorch": torch.__version__,
        "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "RAM_GB": "N/A"
    }

def evaluate_model(name, model, test_loader):
    print(f"\nEvaluating {name} on TEST set...")
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            logits = model(x)
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
        
    cm = confusion_matrix(all_targets, all_preds)
    if cm.size == 1:
        tn = cm[0,0] if all_targets[0]==0 else 0
        tp = cm[0,0] if all_targets[0]==1 else 0
        fp, fn = 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC-ROC": auc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def main(seed=42):
    # Reproducibility
    set_seeds(seed)
    
    print("--- STARTING OFFLINE EVALUATION (CICIDS2017 Tabular) ---")
    print(f"Random Seed: {seed}\n")
    sys_info = get_system_info()
    print(json.dumps(sys_info, indent=2))
    
    # Load Test Data (Pre-windowed by dataset loader)
    # CRITICAL: Windowing is now done in CICIDS2017Dataset BEFORE splitting
    # This prevents boundary leakage where windows span train/val/test splits
    print("Loading Test Data (Pre-windowed, Strict Temporal Split)...")
    try:
        test_ds = CICIDS2017Dataset(split="test", binary=True, seq_len=SEQ_LEN)
    except MemoryError:
        print("❌ MemoryError: Test set too large. Trying to sample or fail gracefully.")
        return
        
    if len(test_ds) == 0:
        print("❌ Dataset empty. Run terminated.")
        return
        
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Input dimension from dataset (lazy windowing)
    input_dim = test_ds.X_raw.shape[1]
    
    results = []
    
    # Models to Evaluate
    models_to_run = [
        ("Mamba_Tabular", MambaClassifier, "outputs/mamba_tabular.pth"),
        ("LSTM_Tabular", LSTMClassifier, "outputs/lstm_tabular.pth")
    ]
    
    for name, cls, path in models_to_run:
        if not os.path.exists(path):
            print(f"⚠️  Skipping {name}: Weight file not found at {path}")
            continue
            
        print(f"Loading {name} from {path}...")
        model = cls(input_dim=input_dim, d_model=128).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        
        metrics = evaluate_model(name, model, test_loader)
        results.append(metrics)
        
    # Save
    with open("outputs/metrics_offline.json", "w") as f:
        json.dump({"system_info": sys_info, "results": results}, f, indent=2)
        
    print("\nOffline Metrics Saved to outputs/metrics_offline.json")
    if results:
        print(pd.DataFrame(results).to_markdown(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline evaluation for CICIDS2017")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    main(seed=args.seed)
