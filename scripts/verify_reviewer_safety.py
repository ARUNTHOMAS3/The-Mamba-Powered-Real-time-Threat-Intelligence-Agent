#!/usr/bin/env python3
"""
Reviewer Safety Verification Script

Validates that all publication-ready requirements are met:
1. Model capacity parity (±10%)
2. No data leakage in scaler fitting
3. Config hash consistency
4. Seed reproducibility

Run this before submitting to a journal to ensure compliance.
"""

import os
import json
import sys
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")

def check_capacity_parity():
    """Check that LSTM and Mamba models have ±10% parameter parity."""
    print(f"{Colors.BLUE}[1/4] Checking Model Capacity Parity...{Colors.END}")
    
    param_file = "outputs/param_counts.txt"
    if not os.path.exists(param_file):
        print(f"{Colors.RED}✗ FAIL: {param_file} not found{Colors.END}")
        print(f"  Run training scripts first: python train/train_lstm_tabular.py --seed 42")
        return False
    
    with open(param_file, 'r') as f:
        lines = f.readlines()
    
    lstm_params = None
    mamba_params = None
    
    for line in lines:
        if 'LSTM' in line or 'LSTMClassifier' in line:
            lstm_params = int(line.split(':')[1].strip())
        if 'Mamba' in line or 'MambaClassifier' in line:
            mamba_params = int(line.split(':')[1].strip())
    
    if lstm_params is None or mamba_params is None:
        print(f"{Colors.RED}✗ FAIL: Could not parse parameter counts{Colors.END}")
        return False
    
    diff_percent = abs(lstm_params - mamba_params) / max(lstm_params, mamba_params) * 100
    
    print(f"  LSTM Parameters:  {lstm_params:,}")
    print(f"  Mamba Parameters: {mamba_params:,}")
    print(f"  Difference: {diff_percent:.2f}%")
    
    if diff_percent <= 10.0:
        print(f"{Colors.GREEN}✓ PASS: Models are capacity-matched (within ±10%){Colors.END}")
        return True
    else:
        print(f"{Colors.RED}✗ FAIL: Capacity mismatch exceeds 10% ({diff_percent:.2f}%){Colors.END}")
        return False

def check_scaler_safety():
    """Verify scaler is fit only on train split."""
    print(f"\n{Colors.BLUE}[2/4] Checking Scaler Safety (No Test Leakage)...{Colors.END}")
    
    # Check dataset loader code
    loader_file = "datasets/cicids2017_loader.py"
    if not os.path.exists(loader_file):
        print(f"{Colors.RED}✗ FAIL: {loader_file} not found{Colors.END}")
        return False
    
    with open(loader_file, 'r') as f:
        code = f.read()
    
    # Check for proper scaler fitting
    issues = []
    
    if 'if self.split == \'train\':' not in code:
        issues.append("Missing train-only scaler fitting check")
    
    if code.count('scaler.fit') > 1:
        issues.append("Multiple scaler.fit() calls detected")
    
    if 'scaler.fit_transform' in code and code.index('scaler.fit_transform') < code.index('if self.split'):
        issues.append("fit_transform called before split check")
    
    # Check for leaky multimodal dataset
    if os.path.exists("datasets/multimodal_dataset_fixed.py"):
        print(f"{Colors.YELLOW}⚠  WARNING: datasets/multimodal_dataset_fixed.py still exists{Colors.END}")
        print(f"  This file was flagged for potential leakage. Consider deleting if unused.")
    
    if issues:
        print(f"{Colors.RED}✗ FAIL: Scaler safety issues detected:{Colors.END}")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print(f"{Colors.GREEN}✓ PASS: Scaler fit only on train split (no leakage){Colors.END}")
    return True

def check_config_consistency():
    """Verify both models use same config hash."""
    print(f"\n{Colors.BLUE}[3/4] Checking Config Hash Consistency...{Colors.END}")
    
    lstm_history = "outputs/lstm_training_history.json"
    mamba_history = "outputs/mamba_training_history.json"
    
    if not os.path.exists(lstm_history):
        print(f"{Colors.YELLOW}⚠  WARNING: {lstm_history} not found (train LSTM first){Colors.END}")
        return None  # Not a failure, just not run yet
    
    if not os.path.exists(mamba_history):
        print(f"{Colors.YELLOW}⚠  WARNING: {mamba_history} not found (train Mamba first){Colors.END}")
        return None
    
    with open(lstm_history, 'r') as f:
        lstm_data = json.load(f)
    
    with open(mamba_history, 'r') as f:
        mamba_data = json.load(f)
    
    lstm_hash = lstm_data.get('config_hash', 'N/A')
    mamba_hash = mamba_data.get('config_hash', 'N/A')
    
    print(f"  LSTM Config Hash:  {lstm_hash[:16]}...")
    print(f"  Mamba Config Hash: {mamba_hash[:16]}...")
    
    if lstm_hash == mamba_hash and lstm_hash != 'N/A':
        print(f"{Colors.GREEN}✓ PASS: Config hashes match (identical hyperparameters){Colors.END}")
        return True
    else:
        print(f"{Colors.RED}✗ FAIL: Config hashes don't match or missing{Colors.END}")
        return False

def check_seed_reproducibility():
    """Verify seed is logged in reproducibility.json."""
    print(f"\n{Colors.BLUE}[4/4] Checking Seed Reproducibility...{Colors.END}")
    
    repro_file = "outputs/reproducibility.json"
    if not os.path.exists(repro_file):
        print(f"{Colors.RED}✗ FAIL: {repro_file} not found{Colors.END}")
        return False
    
    with open(repro_file, 'r') as f:
        data = json.load(f)
    
    seed = data.get('seed')
    config_hash = data.get('config_hash')
    
    if seed is None:
        print(f"{Colors.RED}✗ FAIL: Seed not logged in reproducibility.json{Colors.END}")
        return False
    
    print(f"  Seed: {seed}")
    print(f"  Config Hash: {config_hash[:16] if config_hash else 'N/A'}...")
    print(f"  Deterministic: {data.get('deterministic_settings', {})}")
    
    print(f"{Colors.GREEN}✓ PASS: Seed logged for reproducibility{Colors.END}")
    return True

def main():
    print_header("REVIEWER SAFETY VERIFICATION")
    
    print("This script verifies that all publication-ready requirements are met.")
    print("Any failures must be fixed before journal submission.\n")
    
    results = {
        "Capacity Parity": check_capacity_parity(),
        "Scaler Safety": check_scaler_safety(),
        "Config Consistency": check_config_consistency(),
        "Seed Reproducibility": check_seed_reproducibility()
    }
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for check, result in results.items():
        if result is True:
            print(f"{Colors.GREEN}✓{Colors.END} {check}")
        elif result is False:
            print(f"{Colors.RED}✗{Colors.END} {check}")
        else:
            print(f"{Colors.YELLOW}⊘{Colors.END} {check} (not run yet)")
    
    print(f"\n{Colors.BLUE}Final Score: {passed}/{len(results)} checks passed{Colors.END}")
    
    if failed == 0 and skipped == 0:
        print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}✓ PUBLICATION-READY: All checks passed!{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
        return 0
    elif failed > 0:
        print(f"\n{Colors.RED}{'='*70}{Colors.END}")
        print(f"{Colors.RED}✗ NOT READY: {failed} check(s) failed{Colors.END}")
        print(f"{Colors.RED}{'='*70}{Colors.END}\n")
        return 1
    else:
        print(f"\n{Colors.YELLOW}{'='*70}{Colors.END}")
        print(f"{Colors.YELLOW}⚠ INCOMPLETE: Run both training scripts first{Colors.END}")
        print(f"{Colors.YELLOW}{'='*70}{Colors.END}\n")
        return 2

if __name__ == "__main__":
    sys.exit(main())
