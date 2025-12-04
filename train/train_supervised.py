import argparse, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasets.multimodal_dataset import MultimodalDataset
from models.classifier import ThreatModel
from utils.utils import load_config, set_seed, mkdirp
from utils.metrics import classification_metrics, auc_metric

def train(cfg, max_epochs=5):
    set_seed(42)
    device = torch.device(cfg.get("device", "cpu"))
    
    # 1. Load Dataset
    # Ensure this path matches where preprocess.py saved your file
    data_path = "data/processed/synth_train.json" 
    
    # Check if file exists to avoid path errors
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Using 'data/processed/synth_small.json' or regenerating...")
        data_path = "data/processed/synth_small.json"

    # Note: removed create_synthetic arg as the simple dataset class might not support it 
    # directly depending on which version you pasted. This assumes the file exists.
    ds = MultimodalDataset(path=data_path)
    
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)

    # 2. Initialize Model with Correct Dimensions
    # input_dims=(Log=32, Text=64, CVE=16) -> Matches your preprocess.py output
    model = ThreatModel(input_dims=(32, 64, 16), d_model=cfg["model"]["d_model"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    bce = nn.BCELoss()

    print(f"Starting training on device: {device}")

    for epoch in range(max_epochs):
        model.train()
        losses = []
        all_y, all_p = [], []
        
        for batch in dl:
            x_log = batch["x_log"].to(device)
            x_text = batch["x_text"].to(device)
            x_cve = batch["x_cve"].to(device)
            y = batch["label"].to(device)
            
            # Forward pass
            out = model(x_log, x_text, x_cve)
            pred = out["score"]
            
            # Loss calculation
            loss = bce(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            all_y.extend(y.detach().cpu().numpy().tolist())
            all_p.extend(pred.detach().cpu().numpy().tolist())

        # Metrics
        yhat = [1 if p > 0.5 else 0 for p in all_p]
        m = classification_metrics(all_y, yhat)
        auc = auc_metric(all_y, all_p)
        
        avg_loss = sum(losses) / len(losses) if losses else 0
        print(f"[SupTrain] epoch {epoch+1}/{max_epochs} | loss {avg_loss:.4f} | f1 {m['f1']:.3f} | auc {auc:.3f}")
        
        # Save Checkpoint
        mkdirp("outputs")
        torch.save(model.state_dict(), f"outputs/supervised_epoch{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_epochs", type=int, default=5)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    train(cfg, args.max_epochs)