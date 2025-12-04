# utils/metrics.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def classification_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}

def auc_metric(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except:
        return None
