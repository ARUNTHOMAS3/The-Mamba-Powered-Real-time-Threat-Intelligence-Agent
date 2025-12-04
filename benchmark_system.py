import torch
import time
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets.multimodal_dataset import MultimodalDataset
from models.classifier import ThreatModel
from models.lstm_baseline import LSTMModel
from torch.utils.data import DataLoader
from utils.utils import load_config

def measure_performance(model, dl, device, name):
    model.to(device)
    model.eval()
    
    all_y = []
    all_pred = []
    latencies = []
    
    print(f"⚡ Benchmarking {name}...")
    
    with torch.no_grad():
        for batch in dl:
            x_log = batch["x_log"].to(device)
            x_text = batch["x_text"].to(device)
            x_cve = batch["x_cve"].to(device)
            y = batch["label"].to(device)
            
            # Timer
            start_time = time.time()
            out = model(x_log, x_text, x_cve)
            end_time = time.time()
            
            latencies.append((end_time - start_time) * 1000)
            
            all_y.extend(y.cpu().numpy())
            
            # Handle Scalar Outputs
            probs = out["score"].cpu().numpy()
            if probs.ndim == 0: probs = np.array([probs])
            
            preds = [1 if p > 0.5 else 0 for p in probs]
            all_pred.extend(preds)
            
    avg_latency = np.mean(latencies)
    
    # --- NEW METRICS ---
    f1 = f1_score(all_y, all_pred, zero_division=0)
    prec = precision_score(all_y, all_pred, zero_division=0)
    rec = recall_score(all_y, all_pred, zero_division=0)
    
    return f1, prec, rec, avg_latency

def run_benchmark():
    cfg = load_config("configs/default.yaml")
    device = torch.device("cpu")
    input_dims = (cfg["data"]["logs_emb_dim"], cfg["data"]["text_emb_dim"], cfg["data"]["cve_emb_dim"])
    d_model = cfg["model"]["d_model"]
    
    # Load Data
    data_path = cfg["paths"]["processed_data"]
    if not os.path.exists(data_path): data_path = "data/processed/synth_small.json"
    ds = MultimodalDataset(path=data_path)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    # Load Models
    mamba = ThreatModel(input_dims=input_dims, d_model=d_model)
    if os.path.exists("outputs/supervised_epoch5.pt"):
        mamba.load_state_dict(torch.load("outputs/supervised_epoch5.pt", map_location="cpu"))

    lstm = LSTMModel(input_dims=input_dims, d_model=d_model)
    if os.path.exists("outputs/lstm_baseline.pt"):
        lstm.load_state_dict(torch.load("outputs/lstm_baseline.pt", map_location="cpu"))
    
    # Measure
    f1_m, prec_m, rec_m, lat_m = measure_performance(mamba, dl, device, "Proposed Mamba Agent")
    f1_l, prec_l, rec_l, lat_l = measure_performance(lstm, dl, device, "LSTM Baseline")
    
    # Print Full Table
    print("\n" + "="*75)
    print(f"{'Metric':<20} | {'LSTM Baseline':<20} | {'Proposed Mamba':<20}")
    print("-" * 75)
    print(f"{'F1-Score':<20} | {f1_l:.2%}             | {f1_m:.2%}")
    print(f"{'Precision':<20} | {prec_l:.2%}             | {prec_m:.2%}")
    print(f"{'Recall':<20} | {rec_l:.2%}             | {rec_m:.2%}")
    print(f"{'Inference Latency':<20} | {lat_l:.2f} ms            | {lat_m:.2f} ms")
    print("-" * 75)

if __name__ == "__main__":
    run_benchmark()
