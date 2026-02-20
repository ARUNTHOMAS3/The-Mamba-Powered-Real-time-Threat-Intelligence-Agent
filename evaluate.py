"""
Evaluation Module for Benchmark Study

Provides:
- evaluate_model(): Classification metrics (accuracy, precision, recall, F1, AUC-ROC)
- measure_efficiency(): Latency, throughput, memory, parameter count
- per_attack_metrics(): Per-attack-type breakdown
- statistical_tests(): Pairwise Wilcoxon signed-rank tests
"""

import time
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: Device to run on
    
    Returns:
        dict with accuracy, precision, recall, f1, auc_roc, predictions, targets, probabilities
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0  # Only one class present
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc),
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
    }


def measure_efficiency(model, sample_input, device='cpu', n_warmup=10, n_measure=100):
    """
    Measure model efficiency metrics.
    
    Args:
        model: PyTorch model
        sample_input: Single sample tensor (1, seq_len, features)
        device: Device
        n_warmup: Warmup iterations
        n_measure: Measurement iterations
    
    Returns:
        dict with latency_ms, throughput_per_sec, memory_mb, param_count
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    model.eval()
    model = model.to(device)
    sample_input = sample_input.to(device)
    
    # Parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory footprint (approximate model size)
    memory_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    memory_mb = memory_bytes / (1024 * 1024)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample_input)
    
    # Latency measurement (single sample)
    latencies = []
    with torch.no_grad():
        for _ in range(n_measure):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(sample_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # Convert to ms
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    # Throughput measurement (batch processing)
    batch_input = sample_input.repeat(32, 1, 1)  # Batch of 32
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_measure):
            _ = model(batch_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    
    total_samples = 32 * n_measure
    total_time = t1 - t0
    throughput = total_samples / total_time
    
    return {
        'latency_ms': float(avg_latency),
        'latency_std_ms': float(std_latency),
        'throughput_per_sec': float(throughput),
        'memory_mb': float(memory_mb),
        'param_count': int(param_count),
    }


def per_attack_metrics(model, test_dataset, test_loader, device='cpu'):
    """
    Compute per-attack-type metrics.
    
    Args:
        model: Trained model
        test_dataset: Dataset with get_attack_label() method
        test_loader: DataLoader
        device: Device
    
    Returns:
        dict mapping attack_type -> {accuracy, precision, recall, f1, count}
    """
    model.eval()
    
    # Get all predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x).squeeze()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Group by attack type
    attack_results = defaultdict(lambda: {'preds': [], 'targets': []})
    
    for idx in range(len(all_preds)):
        try:
            attack_type = test_dataset.get_attack_label(idx)
        except (IndexError, AttributeError):
            attack_type = 'Unknown'
        
        attack_results[attack_type]['preds'].append(all_preds[idx])
        attack_results[attack_type]['targets'].append(all_targets[idx])
    
    # Compute metrics per attack type
    results = {}
    for attack_type, data in attack_results.items():
        preds = np.array(data['preds'])
        targets = np.array(data['targets'])
        
        if len(set(targets)) < 2:
            # Only one class present for this attack type
            acc = accuracy_score(targets, preds)
            results[attack_type] = {
                'accuracy': float(acc),
                'precision': float(np.mean(preds == targets)),
                'recall': 1.0 if np.all(targets == 1) and np.all(preds == 1) else 0.0,
                'f1': float(2 * acc / (acc + 1)) if acc > 0 else 0.0,
                'count': int(len(targets)),
            }
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, preds, average='binary', zero_division=0
            )
            results[attack_type] = {
                'accuracy': float(accuracy_score(targets, preds)),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'count': int(len(targets)),
            }
    
    return results


def statistical_tests(results_dict, metric='f1'):
    """
    Perform pairwise Wilcoxon signed-rank tests between all model pairs.
    
    Args:
        results_dict: dict of model_name -> list of metric values (one per seed)
        metric: which metric to test (default: 'f1')
    
    Returns:
        dict of (model_a, model_b) -> {statistic, p_value, significant}
    """
    model_names = list(results_dict.keys())
    pairwise_results = {}
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            
            values_a = np.array(results_dict[name_a])
            values_b = np.array(results_dict[name_b])
            
            if len(values_a) < 5 or len(values_b) < 5:
                # Not enough samples for Wilcoxon test
                pairwise_results[(name_a, name_b)] = {
                    'statistic': None,
                    'p_value': None,
                    'significant': None,
                    'note': 'Insufficient samples (need ≥5)',
                }
                continue
            
            try:
                stat, p_value = stats.wilcoxon(values_a, values_b)
                pairwise_results[(name_a, name_b)] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                }
            except Exception as e:
                pairwise_results[(name_a, name_b)] = {
                    'statistic': None,
                    'p_value': None,
                    'significant': None,
                    'note': str(e),
                }
    
    return pairwise_results


def compute_confidence_interval(values, confidence=0.95):
    """Compute confidence interval for a list of values."""
    n = len(values)
    if n < 2:
        return float(np.mean(values)), 0.0, float(np.mean(values)), float(np.mean(values))
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_val * se
    
    return float(mean), float(std), float(mean - margin), float(mean + margin)
