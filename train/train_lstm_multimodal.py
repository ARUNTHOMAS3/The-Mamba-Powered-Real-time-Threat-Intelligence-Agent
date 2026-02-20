
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.unswnb15_loader import UNSWNB15Dataset
from train.train_mamba import HashTextVectorizer, window_data

# --- CONFIG ---
SEQ_LEN = 50           # Same as Mamba
BATCH_SIZE = 64
EPOCHS = 15
PATIENCE = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "lstm_multimodal"

# --- MODEL DEFINITION ---
class LSTM_Backbone(nn.Module):
    def __init__(self, d_input, d_model=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, D_in)
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.layernorm(out)

class LSTM_Fusion_Full(nn.Module):
    def __init__(self, tabular_dim, d_model=128):
        super().__init__()
        # Identical Architecture to Mamba Fusion, just swapping Backbone
        self.lstm_tab = LSTM_Backbone(d_input=tabular_dim, d_model=d_model)
        self.lstm_log = LSTM_Backbone(d_input=128, d_model=d_model)
        self.lstm_cve = LSTM_Backbone(d_input=128, d_model=d_model)
        
        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)
        
    def forward(self, x_tab, x_log, x_cve):
        z_tab = self.lstm_tab(x_tab)
        z_log = self.lstm_log(x_log)
        z_cve = self.lstm_cve(x_cve)
        
        z_concat = torch.cat([z_tab, z_log, z_cve], dim=-1)
        # Fuse
        z_fused = self.fusion_norm(self.fusion_proj(z_concat))
        # Classify Last Step
        return self.classifier(z_fused[:, -1, :])

def train():
    print(f"[{MODEL_NAME}] Starting Training on {DEVICE}...")
    
    # 1. Load Data (Strict Split)
    print("Loading Datasets...")
    train_ds = UNSWNB15Dataset(split="train", binary=True)
    val_ds = UNSWNB15Dataset(split="val", binary=True)
    
    # 2. Vectorize & Window
    vectorizer = HashTextVectorizer(dim=128)
    print("Windowing Data...")
    train_data = window_data(train_ds, SEQ_LEN, vectorizer)
    val_data = window_data(val_ds, SEQ_LEN, vectorizer)
    
    # Unpack
    trn_xt, trn_xl, trn_xc, trn_y = train_data
    val_xt, val_xl, val_xc, val_y = val_data
    
    tabular_dim = trn_xt.shape[-1]
    
    # 3. Model Setup
    model = LSTM_Fusion_Full(tabular_dim=tabular_dim, d_model=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(DEVICE))
    
    # 4. Training Loop
    best_f1 = 0
    patience_ctr = 0
    history = []
    
    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        total_loss = 0
        
        indices = torch.randperm(len(trn_xt))
        
        for i in range(0, len(trn_xt), BATCH_SIZE):
            idx = indices[i:i+BATCH_SIZE]
            if len(idx) < 2: continue
            
            b_xt = trn_xt[idx].to(DEVICE)
            b_xl = trn_xl[idx].to(DEVICE)
            b_xc = trn_xc[idx].to(DEVICE)
            b_y = trn_y[idx].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(b_xt, b_xl, b_xc)
            loss = criterion(logits.squeeze(), b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        
        with torch.no_grad():
            for i in range(0, len(val_xt), BATCH_SIZE):
                end = min(i+BATCH_SIZE, len(val_xt))
                b_xt = val_xt[i:end].to(DEVICE)
                b_xl = val_xl[i:end].to(DEVICE)
                b_xc = val_xc[i:end].to(DEVICE)
                b_y = val_y[i:end].to(DEVICE)
                
                logits = model(b_xt, b_xl, b_xc)
                probs = torch.sigmoid(logits.squeeze())
                preds = (probs > 0.5).float()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(b_y.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                
        prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary', zero_division=0)
        acc = accuracy_score(val_targets, val_preds)
        try:
            auc = roc_auc_score(val_targets, val_probs)
        except:
            auc = 0.5
            
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f} | Time: {epoch_time:.1f}s")
        
        history.append({
            "epoch": epoch+1,
            "f1": f1,
            "accuracy": acc,
            "loss": total_loss,
            "time": epoch_time
        })
        
        if f1 > best_f1:
            best_f1 = f1
            patience_ctr = 0
            torch.save(model.state_dict(), f"outputs/{MODEL_NAME}.pth")
            print("   >>> Saved Best Model")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("   [Early Stopping]")
                break
        
        scheduler.step(f1)
        
    # Save History
    with open(f"outputs/{MODEL_NAME}_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    train()
