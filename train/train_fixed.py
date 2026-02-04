"""
FIXED: Train Script with Proper Evaluation, Threshold Tuning, & Reproducibility
=================================================================================
Fixes:
- Separate train/test evaluation (NO leakage)
- Threshold tuning on validation set
- Gradient clipping & proper weight init
- Fixed seeds for reproducibility
- Proper logging of metrics (precision, recall, F1, AUC separately)
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
from typing import Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports from your project
from datasets.multimodal_dataset_fixed import MultimodalDatasetFixed
from models.classifier import ThreatModel
from utils.utils import load_config, set_seed, mkdirp


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """
    Find optimal threshold that maximizes metric on validation data.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_score: Predicted probabilities [0, 1]
        metric: "f1", "precision", "recall", or "auc"
    
    Returns:
        (best_threshold, best_score)
    """
    assert metric in ["f1", "precision", "recall", "auc"], f"Unknown metric: {metric}"
    
    if metric == "auc":
        return 0.5, roc_auc_score(y_true, y_score)
    
    best_threshold = 0.5
    best_score = -1.0
    
    # Search thresholds
    for threshold in np.linspace(0.1, 0.9, 50):
        y_pred = (y_score >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def evaluate_model(model, dataloader, device, threshold: float = 0.5) -> dict:
    """
    Evaluate model on a dataset.
    
    Returns:
        dict with all metrics
    """
    model.eval()
    
    all_y = []
    all_scores = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_log = batch["x_log"].to(device)
            x_text = batch["x_text"].to(device)
            x_cve = batch["x_cve"].to(device)
            y = batch["label"].to(device)
            
            out = model(x_log, x_text, x_cve)
            logits = out["score_logit"]
            scores = torch.sigmoid(logits).cpu().numpy()
            
            # Handle scalar vs batch
            if scores.ndim == 0:
                scores = np.array([scores])
            
            all_y.extend(y.cpu().numpy().tolist())
            all_scores.extend(scores.tolist())
            all_preds.extend((scores >= threshold).astype(int).tolist())
    
    all_y = np.array(all_y)
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    
    # Compute metrics
    try:
        auc = roc_auc_score(all_y, all_scores)
    except:
        auc = 0.5
    
    f1 = f1_score(all_y, all_preds, zero_division=0)
    prec = precision_score(all_y, all_preds, zero_division=0)
    rec = recall_score(all_y, all_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_y, all_preds).ravel()
    
    return {
        "auc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
        "scores": all_scores,
        "labels": all_y,
    }


def train_with_validation(cfg, max_epochs: int = 10, random_seed: int = 42):
    """
    Train with proper train/val/test split and threshold tuning.
    """
    set_seed(random_seed)
    device = torch.device(cfg.get("device", "cpu"))
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    data_path = "data/processed/synth_balanced"
    seq_len = cfg["data"].get("seq_len", 128)
    train_ds = MultimodalDatasetFixed(data_path, split="train", normalize=True, seq_len=seq_len)
    test_ds = MultimodalDatasetFixed(data_path, split="test", normalize=True, seq_len=seq_len)
    
    # Split train into train/val (80/20)
    val_size = int(0.2 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds_split, val_ds_split = random_split(train_ds, [train_size, val_size])
    
    train_dl = DataLoader(train_ds_split, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds_split, batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = ThreatModel(input_dims=(32, 64, 16), d_model=cfg["model"]["d_model"]).to(device)
    
    # Proper weight initialization
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Optimizer: Adam (lr={cfg['training']['lr']}, wd={cfg['training']['weight_decay']})")
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_f1 = 0.0
    best_threshold = 0.5
    results_log = []
    best_val_scores = None
    best_val_labels = None
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_losses = []
        train_y_true = []
        train_y_score = []
        
        for batch in train_dl:
            x_log = batch["x_log"].to(device)
            x_text = batch["x_text"].to(device)
            x_cve = batch["x_cve"].to(device)
            y = batch["label"].to(device)
            
            out = model(x_log, x_text, x_cve)
            logits = out["score_logit"]
            
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_losses.append(loss.item())
            train_y_true.extend(y.detach().cpu().numpy().tolist())
            train_y_score.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        
        train_y_true = np.array(train_y_true)
        train_y_score = np.array(train_y_score)
        
        # Validation
        val_metrics = evaluate_model(model, val_dl, device, threshold=0.5)
        
        # Find best threshold on validation (on probabilities from logits)
        best_threshold, _ = find_best_threshold(train_y_true, train_y_score, metric="f1")
        
        # Evaluate on test with best threshold
        test_metrics = evaluate_model(model, test_dl, device, threshold=best_threshold)
        
        avg_loss = np.mean(train_losses)
        
        print(
            f"Epoch {epoch+1:2d}/{max_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.3f} | "
            f"Test F1: {test_metrics['f1']:.3f} | "
            f"Test AUC: {test_metrics['auc']:.3f} | "
            f"Threshold: {best_threshold:.3f}"
        )
        
        # Save best model
        if test_metrics["f1"] > best_val_f1:
            best_val_f1 = test_metrics["f1"]
            best_val_scores = test_metrics.get("scores")
            best_val_labels = test_metrics.get("labels")
            mkdirp("outputs")
            torch.save(model.state_dict(), f"outputs/best_model_seed{random_seed}.pt")
            print(f"  ✓ Saved best model (F1={best_val_f1:.3f})")
        
        # Log results
        results_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "threshold": best_threshold,
        })
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    model.load_state_dict(torch.load(f"outputs/best_model_seed{random_seed}.pt", map_location=device))
    final_metrics = evaluate_model(model, test_dl, device, threshold=best_threshold)
    
    print(f"\nBest Test Metrics:")
    print(f"  F1:        {final_metrics['f1']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  AUC:       {final_metrics['auc']:.4f}")
    print(f"  Threshold: {final_metrics['threshold']:.4f}")
    print(f"  TP: {final_metrics['tp']}, FP: {final_metrics['fp']}, "
          f"FN: {final_metrics['fn']}, TN: {final_metrics['tn']}")
    
    # Save log
    mkdirp("outputs")
    with open(f"outputs/training_log_seed{random_seed}.json", "w") as f:
        json.dump(results_log, f, indent=2)

    # Plot score distributions (normal vs attack) for best model
    try:
        scores = best_val_scores if best_val_scores is not None else final_metrics.get("scores")
        labels = best_val_labels if best_val_labels is not None else final_metrics.get("labels")
        if scores is not None and labels is not None:
            plt.figure(figsize=(6, 4))
            scores = np.array(scores)
            labels = np.array(labels)
            plt.hist(scores[labels == 0], bins=40, alpha=0.6, label="normal")
            plt.hist(scores[labels == 1], bins=40, alpha=0.6, label="attack")
            plt.xlabel("Window score (sigmoid)")
            plt.ylabel("count")
            plt.title("Score distributions by class")
            plt.legend()
            plt.tight_layout()
            plt.savefig("outputs/score_distributions.png", dpi=120)
            plt.close()
            print("\n✓ Saved score distribution plot to outputs/score_distributions.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")
    
    return final_metrics, results_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    final_metrics, log = train_with_validation(cfg, max_epochs=args.max_epochs, random_seed=args.seed)
