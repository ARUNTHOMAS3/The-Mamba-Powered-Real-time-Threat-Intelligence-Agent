import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.multimodal_dataset import MultimodalDataset
from models.lstm_baseline import LSTMModel
from utils.utils import load_config, set_seed, mkdirp
import os

def train_lstm():
    # 1. Load the Master Config
    cfg = load_config("configs/default.yaml")
    set_seed(42)
    device = torch.device(cfg["device"])
    
    # 2. Dynamic Input Dims from Config
    input_dims = (
        cfg["data"]["logs_emb_dim"], 
        cfg["data"]["text_emb_dim"], 
        cfg["data"]["cve_emb_dim"]
    )
    model_size = cfg["model"]["d_model"]
    
    # 3. Load Data
    data_path = cfg["paths"]["processed_data"]
    if not os.path.exists(data_path):
        data_path = "data/processed/synth_small.json" # Fallback
        
    ds = MultimodalDataset(path=data_path)
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    
    # 4. Init Model using Config Variables
    print(f"🥊 Training LSTM (Size: {model_size}, Inputs: {input_dims})...")
    model = LSTMModel(input_dims=input_dims, d_model=model_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    loss_fn = nn.BCELoss()
    
    # 5. Train
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in dl:
            x_log = batch["x_log"].to(device)
            x_text = batch["x_text"].to(device)
            x_cve = batch["x_cve"].to(device)
            y = batch["label"].to(device)
            
            optimizer.zero_grad()
            out = model(x_log, x_text, x_cve)
            loss = loss_fn(out["score"], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"LSTM Epoch {epoch+1} Loss: {total_loss/len(dl):.4f}")

    mkdirp(cfg["paths"]["outputs"])
    save_path = os.path.join(cfg["paths"]["outputs"], "lstm_baseline.pt")
    torch.save(model.state_dict(), save_path)
    print(f"✅ LSTM Model Saved to {save_path}")

if __name__ == "__main__":
    train_lstm()
