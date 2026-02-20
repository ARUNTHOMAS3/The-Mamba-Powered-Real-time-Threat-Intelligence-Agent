
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import hashlib
import time
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.unswnb15_loader import UNSWNB15Dataset
from models.mamba_backbone import MambaBackbone

# --- 1. CONFIGURATION ---
SEQ_LEN = 128
BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. VECTORIZER ---
class HashTextVectorizer:
    """
    Deterministic lightweight vectorizer using hashing.
    Converts list of strings -> Tensor (Batch, Seq, Dim).
    """
    def __init__(self, dim=128):
        self.dim = dim

    def encode(self, text_matrix):
        # text_matrix: List of Lists of strings (Batch, Seq)
        # Output: (Batch, Seq, Dim)
        
        batch_out = []
        for seq_list in text_matrix:
            seq_out = []
            for text in seq_list:
                # Deterministic Seed from hash
                seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
                
                # We use numpy random with seeded state to generate a consistent vector
                # This is "feature engineering", effectively projecting text into a fixed random basis
                rs = np.random.RandomState(seed)
                vec = rs.normal(0, 1, size=(self.dim,))
                seq_out.append(vec)
            batch_out.append(seq_out)
            
        return torch.tensor(np.array(batch_out), dtype=torch.float32)

# --- 3. MODEL ARCHITECTURE ---
class MultiModalMamba(nn.Module):
    def __init__(self, tabular_dim, d_model=128):
        super().__init__()
        
        # Encoders
        self.mamba_tab = MambaBackbone(d_input=tabular_dim, d_model=d_model)
        self.mamba_log = MambaBackbone(d_input=128, d_model=d_model) # generated logs are 128-dim
        self.mamba_cve = MambaBackbone(d_input=128, d_model=d_model) # cves are 128-dim
        
        # Fusion
        # Concatenate 3 * d_model -> Project to d_model
        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # Classifier
        self.classifier = nn.Linear(d_model, 1)
        
    def forward(self, x_tab, x_log, x_cve, return_embeddings=False):
        # All inputs: (B, L, D)
        
        # 1. Encode Streams
        z_tab = self.mamba_tab(x_tab) # (B, L, d_model)
        z_log = self.mamba_log(x_log) 
        z_cve = self.mamba_cve(x_cve)
        
        # 2. Fusion
        z_concat = torch.cat([z_tab, z_log, z_cve], dim=-1) # (B, L, 3*d_model)
        z_fused = self.fusion_proj(z_concat)
        z_fused = self.fusion_norm(z_fused) # (B, L, d_model)
        
        # 3. Pooling (Last Token)
        # We classify based on the state after seeing the sequence
        z_last = z_fused[:, -1, :] # (B, d_model)
        
        if return_embeddings:
            return z_last
        
        # 4. Classification
        logits = self.classifier(z_last)
        return logits

# --- 4. PREPARATION UTILS ---
def window_data(dataset, seq_len):
    """
    Chunks the flat dataset into sequences.
    Returns:
       x_tab_seq: (N_seq, L, F_tab)
       x_log_seq: List of Lists of strings (N_seq, L)
       x_cve_seq: List of Lists of strings (N_seq, L)
       y_seq:     (N_seq, 1) - Label of the last item in window
    """
    print("Windowing data...")
    X_flat = dataset.X
    y_flat = dataset.y
    logs_flat = dataset.logs
    cves_flat = dataset.cve_contexts
    
    num_samples = len(X_flat) // seq_len
    
    # Process Tabular
    X_trunc = X_flat[:num_samples*seq_len]
    x_tab_seq = X_trunc.view(num_samples, seq_len, -1)
    
    # Process Labels (Take last label of sequence)
    y_trunc = y_flat[:num_samples*seq_len]
    y_seq = y_trunc.view(num_samples, seq_len)[:, -1].unsqueeze(1).float()
    
    # Process Strings (Manual chunking)
    logs_trunc = logs_flat[:num_samples*seq_len]
    cves_trunc = cves_flat[:num_samples*seq_len]
    
    x_log_seq = [logs_trunc[i*seq_len : (i+1)*seq_len] for i in range(num_samples)]
    x_cve_seq = [cves_trunc[i*seq_len : (i+1)*seq_len] for i in range(num_samples)]
    
    return x_tab_seq, x_log_seq, x_cve_seq, y_seq

# --- 5. TRAINING LOOP ---
def main():
    print(f"Running Mamba Training on {DEVICE}")
    
    # Load Data
    print("Loading UNSW-NB15 (Stratified Split)...")
    train_ds = UNSWNB15Dataset(split="train", binary=True)
    test_ds = UNSWNB15Dataset(split="val", binary=True) # Use val for monitoring
    
    # Vectorizer
    vectorizer = HashTextVectorizer(dim=128)
    
    # Windowing
    train_xt, train_xl, train_xc, train_y = window_data(train_ds, SEQ_LEN)
    test_xt, test_xl, test_xc, test_y = window_data(test_ds, SEQ_LEN)
    
    # Initialize Model
    # Determine tabular input dim from data
    tabular_dim = train_xt.shape[-1]
    print(f"Tabular Feature Dim: {tabular_dim}")
    
    model = MultiModalMamba(tabular_dim=tabular_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = 0.0
    
    # Training
    # Since we have custom string lists, we iterate manually or custom collate.
    # Manual iteration is simpler for this structure.
    
    num_batches = len(train_xt) // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Shuffle indices
        indices = torch.randperm(len(train_xt))
        
        start_time = time.time()
        
        for i in range(0, len(train_xt), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            if len(batch_idx) < BATCH_SIZE: continue # Drop last incomplete
            
            # Prepare Batch
            b_xt = train_xt[batch_idx].to(DEVICE)
            b_y = train_y[batch_idx].to(DEVICE)
            
            # Vectorize Text on-the-fly (Simulating real-time embedding)
            # Fetch lists using indices (tensor indices -> list access)
            raw_logs = [train_xl[k] for k in batch_idx.tolist()]
            raw_cves = [train_xc[k] for k in batch_idx.tolist()]
            
            b_xl = vectorizer.encode(raw_logs).to(DEVICE)
            b_xc = vectorizer.encode(raw_cves).to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            logits = model(b_xt, b_xl, b_xc)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"Epoch {epoch+1} [{i}/{len(train_xt)}] Loss: {loss.item():.4f}", end='\r')
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for i in range(0, len(test_xt), BATCH_SIZE):
                # No shuffle for test
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
        
        # Advanced Metrics
        try:
            auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            auc = 0.5 # Handle single class edge case
            
        tn, fp, fn, tp = confusion_matrix(val_targets, val_preds).ravel() if len(val_targets) > 0 else (0,0,0,0)
        
        if num_batches > 0:
            print(f"\nEpoch {epoch+1} Summary: Loss={total_loss/num_batches:.4f} | Prec={precision:.3f} Rec={recall:.3f} F1={f1:.3f} | AUC={auc:.3f} | FP={fp}")
        else:
            print(f"\nEpoch {epoch+1} Summary: Loss=0.0000 | Prec={precision:.3f} Rec={recall:.3f} F1={f1:.3f} | AUC={auc:.3f} | FP={fp}")
        if f1 > best_f1:
            best_f1 = f1
            print(f"[+] New Best Model! Saving to outputs/mamba_multimodal.pt")
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/mamba_multimodal.pt")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
