
import torch
import torch.nn as nn
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.cicids2017_loader import CICIDS2017Dataset
from models.tabular_models import MambaClassifier, LSTMClassifier
from utils.reproducibility import set_seeds

# Initialize seeds for reproducibility
set_seeds(42)

# --- CONFIG ---
SEQ_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_streaming(name, model, test_ds):
    print(f"\n[Stream] Evaluating {name} (Single-sample Latency)...")
    model.eval()
    
    # 1. Latency Measurement
    print("   Measuring Latency...")
    latencies = []
    warmup = 10
    limit = 200 # Measure first 200
    
    with torch.no_grad():
        for i in range(min(len(test_ds), limit + warmup)):
            x, _ = test_ds[i]
            x = x.unsqueeze(0).to(DEVICE) # (1, Seq, D)
            
            # Flush
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            if i >= warmup:
                latencies.append((t1 - t0) * 1000.0) # ms
                
    avg_latency = np.mean(latencies) if latencies else 0
    std_latency = np.std(latencies) if latencies else 0
    
    # 2. Throughput & Streaming Accuracy (Batched for system capacity)
    print("   Measuring Throughput & Metrics...")
    
    BATCH_EVAL = 64
    loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_EVAL, shuffle=False)
    
    preds = []
    targets = []
    
    t0_throughput = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            p = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            preds.extend(p.cpu().numpy())
            targets.extend(y.cpu().numpy())
            
    t1_throughput = time.perf_counter()
    total_time = t1_throughput - t0_throughput
    num_samples = len(targets)
    batched_throughput = num_samples / total_time
    
    # Metrics
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
    
    # FP Stats
    tn = 0
    fp = 0
    for t, p in zip(targets, preds):
        if t == 0 and p == 0: tn += 1
        if t == 0 and p == 1: fp += 1
        
    if (fp + tn) > 0:
        fp_rate = fp / (fp + tn)
    else:
        fp_rate = 0
        
    projected_fp_per_min = fp_rate * 60 * 1000 # @ 1k EPS
    
    results = {
        "Model": name,
        "Latency_Mean_ms": avg_latency,
        "Latency_Std_ms": std_latency,
        "Throughput_Seq_req_s": 1000.0 / avg_latency if avg_latency > 0 else 0,
        "Throughput_Batched_req_s": batched_throughput,
        "Streaming_F1": f1,
        "FP_Count": int(fp),
        "FP_Rate": fp_rate,
        "Projected_FP_per_min_at_1kEPS": projected_fp_per_min
    }
    return results

def main(seed=42):
    # Reproducibility
    set_seeds(seed)
    
    print("--- STARTING STREAMING EVALUATION (CICIDS2017 Tabular) ---")
    print(f"Random Seed: {seed}\n")
    
    # Load Data (Pre-windowed by dataset loader)
    # CRITICAL: Windowing is now done in CICIDS2017Dataset BEFORE splitting
    # This prevents boundary leakage where windows span train/val/test splits
    test_ds = CICIDS2017Dataset(split="test", binary=True, seq_len=SEQ_LEN)
    if len(test_ds) == 0:
        print("❌ Dataset empty.")
        return
        
    # Input dimension from dataset (lazy windowing)
    input_dim = test_ds.X_raw.shape[1]
    
    results = []
    
    models_to_run = [
        ("Mamba_Tabular", MambaClassifier, "outputs/mamba_tabular.pth"),
        ("LSTM_Tabular", LSTMClassifier, "outputs/lstm_tabular.pth")
    ]
    
    for name, cls, path in models_to_run:
        if not os.path.exists(path):
            results.append({"Model": name, "Error": "Weights not found"})
            continue
            
        model = cls(input_dim=input_dim, d_model=128).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        
        metrics = eval_streaming(name, model, test_ds)
        results.append(metrics)
        
    with open("outputs/metrics_streaming.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nStreaming Metrics Saved to outputs/metrics_streaming.json")
    if results:
        print(pd.DataFrame(results).to_markdown(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Streaming evaluation for CICIDS2017")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    main(seed=args.seed)
