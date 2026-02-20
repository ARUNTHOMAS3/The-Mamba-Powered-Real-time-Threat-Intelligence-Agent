
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.unswnb15_loader import UNSWNB15Dataset
from models.mamba_backbone import MambaBackbone
from train.train_mamba import HashTextVectorizer, window_data

from sklearn.ensemble import RandomForestClassifier

# --- CONFIG ---
SEQ_LEN = 20
EPOCHS = 10      # Increased for convergence
BATCH_SIZE = 64  # Larger batch for stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODIFIED MODELS FOR ABLATION ---

class LogisticRegression_Tabular(nn.Module):
    def __init__(self, tabular_dim, d_model=None): # d_model unused but kept for signature compat
        super().__init__()
        self.linear = nn.Linear(tabular_dim, 1)
        
    def forward(self, x_tab, x_log, x_cve):
        # Ignores sequence nature (just treats each step as independent or pools)
        # For sequence input (B, L, D), we classify the last step equivalent to 'Sequence-Last'
        out = self.linear(x_tab[:, -1, :]) 
        return out

class LSTM_Tabular(nn.Module):
    def __init__(self, tabular_dim, d_model=128):
        super().__init__()
        self.lstm = nn.LSTM(tabular_dim, d_model, batch_first=True)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x_tab, x_log, x_cve):
        # Ignores log and cve
        out, _ = self.lstm(x_tab)
        out = out[:, -1, :]
        return self.fc(out)

class Mamba_Tabular(nn.Module):
    def __init__(self, tabular_dim, d_model=128):
        super().__init__()
        self.backbone = MambaBackbone(d_input=tabular_dim, d_model=d_model)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x_tab, x_log, x_cve):
        # Ignores log and cve
        z = self.backbone(x_tab)
        z_last = z[:, -1, :]
        return self.fc(z_last)

class Mamba_LogOnly(nn.Module):
    def __init__(self, tabular_dim, d_model=128):
        super().__init__()
        self.backbone = MambaBackbone(d_input=128, d_model=d_model) # input is 128-dim hash
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x_tab, x_log, x_cve):
        # Ignores tabular and cve
        z = self.backbone(x_log)
        z_last = z[:, -1, :]
        return self.fc(z_last)

class Mamba_Fusion_Full(nn.Module):
    def __init__(self, tabular_dim, d_model=128):
        super().__init__()
        self.mamba_tab = MambaBackbone(d_input=tabular_dim, d_model=d_model)
        self.mamba_log = MambaBackbone(d_input=128, d_model=d_model)
        self.mamba_cve = MambaBackbone(d_input=128, d_model=d_model)
        
        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)
        
    def forward(self, x_tab, x_log, x_cve):
        z_tab = self.mamba_tab(x_tab)
        z_log = self.mamba_log(x_log)
        z_cve = self.mamba_cve(x_cve)
        
        z_concat = torch.cat([z_tab, z_log, z_cve], dim=-1)
        z_fused = self.fusion_norm(self.fusion_proj(z_concat))
        return self.classifier(z_fused[:, -1, :])

# --- EXPERIMENT RUNNER ---

def run_experiment(name, model_class, train_data, test_data, tabular_dim):
    print(f"\n[Run] Experiment: {name}")
    
    # Unpack Data
    train_xt, train_xl, train_xc, train_y = train_data
    test_xt, test_xl, test_xc, test_y = test_data
    
    # Init Components
    model = model_class(tabular_dim=tabular_dim, d_model=256).to(DEVICE) # Increased d_model
    optimizer = optim.Adam(model.parameters(), lr=0.002) # Higher initial LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Calculate simple class weight: approx 1.5-2.0 for attacks usually
    pos_weight = torch.tensor([2.0]).to(DEVICE) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    vectorizer = HashTextVectorizer(dim=128)
    
    best_f1 = 0.0
    best_metrics = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1 Score": 0}
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Training (Simple Non-Batched Loop for Clarity/Speed in script)
        # Using same manual batching as train_mamba.py
        indices = torch.randperm(len(train_xt))
        
        for i in range(0, len(train_xt), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            if len(batch_idx) < BATCH_SIZE: continue
            
            b_xt = train_xt[batch_idx].to(DEVICE)
            b_y = train_y[batch_idx].to(DEVICE)
            raw_logs = [train_xl[k] for k in batch_idx.tolist()]
            raw_cves = [train_xc[k] for k in batch_idx.tolist()]
            
            b_xl = vectorizer.encode(raw_logs).to(DEVICE)
            b_xc = vectorizer.encode(raw_cves).to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(b_xt, b_xl, b_xc)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for i in range(0, len(test_xt), BATCH_SIZE):
                end = min(i+BATCH_SIZE, len(test_xt))
                b_xt = test_xt[i:end].to(DEVICE)
                b_y = test_y[i:end].to(DEVICE)
                raw_logs = test_xl[i:end]
                raw_cves = test_xc[i:end]
                
                b_xl = vectorizer.encode(raw_logs).to(DEVICE)
                b_xc = vectorizer.encode(raw_cves).to(DEVICE)
                
                logits = model(b_xt, b_xl, b_xc)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(b_y.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary', zero_division=0)
        acc = accuracy_score(val_targets, val_preds)
        print(f"   Epoch {epoch+1} | Loss: {total_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
        
        scheduler.step(f1)
            
    return best_metrics

def run_rf_baseline(train_ds, test_ds):
    print(f"\n[Run] Experiment: Random Forest (Static Baseline - Last 30%)")
    X_train = train_ds.X
    y_train = train_ds.y
    X_test = test_ds.X
    y_test = test_ds.y
    
    # Flatten if needed? No, dataset.X is (N, D)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    acc = accuracy_score(y_test, preds)
    print(f"   RF Result | F1: {f1:.4f} | Acc: {acc:.4f}")
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def main():
    print("Loading Data...")
    results = []
    
    train_ds = UNSWNB15Dataset(split="train", binary=True)
    test_ds = UNSWNB15Dataset(split="test", binary=True)
    
    # Prepare Data
    train_data = window_data(train_ds, SEQ_LEN)
    test_data = window_data(test_ds, SEQ_LEN)
    
    tabular_dim = train_data[0].shape[-1]
    print(f"Data Prepared. Features: {tabular_dim}")
    
    
    # Run Static Baseline (RF)
    rf_metrics = run_rf_baseline(train_ds, test_ds)
    results.append({
        "Model Config": "Random Forest (Static)", 
        "Accuracy": rf_metrics["Accuracy"],
        "Precision": rf_metrics["Precision"],
        "Recall": rf_metrics["Recall"],
        "F1 Score": rf_metrics["F1 Score"]
    })
    
    # Define Sequence Experiments
    experiments = [
        ("LSTM (Tabular Temporal)", LSTM_Tabular),
        ("Mamba (Tabular Only)", Mamba_Tabular),
        ("Mamba (Log Text Only)", Mamba_LogOnly),
        ("Mamba (Full Fusion)", Mamba_Fusion_Full)
    ]
    
    
    print("\n" + "="*50)
    print("STARTING ABLATION STUDY")
    print("="*50)
    
    for name, cls in experiments:
        metrics = run_experiment(name, cls, train_data, test_data, tabular_dim)
        results.append({
            "Model Config": name, 
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"]
        })
        
    print("\n" + "="*50)
    print("FINAL ABLATION RESULTS")
    print("="*50)
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    # Save results
    df.to_csv("outputs/ablation_results.csv", index=False)
    print("\nResults saved to outputs/ablation_results.csv")

if __name__ == "__main__":
    main()
