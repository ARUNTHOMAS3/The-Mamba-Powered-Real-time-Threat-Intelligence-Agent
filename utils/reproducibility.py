"""
Reproducibility Utilities for Publication-Ready ML Experiments

This module ensures strict reproducibility by:
- Setting all random seeds (Python, NumPy, PyTorch)
- Configuring deterministic CUDA operations
- Logging system and software version information
"""

import torch
import numpy as np
import random
import platform
import sys
import json
import os
import hashlib

# Fixed seed for all experiments
SEED = 42

def compute_config_hash(config_path):
    """
    Compute SHA256 hash of a configuration file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        str: SHA256 hash of the file contents
    """
    with open(config_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def set_seeds(seed=SEED):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    print(f"[Reproducibility] Setting all random seeds to {seed}")
    
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("[Reproducibility] [OK] All seeds set and deterministic mode enabled")

def get_system_info():
    """
    Collect system and software version information.
    
    Returns:
        dict: System information including OS, Python, PyTorch versions, and hardware
    """
    info = {
        "os": {
            "system": platform.system(),
            "version": platform.version(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "python": {
            "version": sys.version,
            "version_info": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A"
        },
        "device": {
            "type": "cuda" if torch.cuda.is_available() else "cpu",
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 1
        }
    }
    
    # Try to get RAM info
    try:
        import psutil
        info["memory"] = {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        }
    except ImportError:
        info["memory"] = {
            "total_gb": "N/A (psutil not installed)",
            "available_gb": "N/A"
        }
    
    # Get NumPy and scikit-learn versions
    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        info["sklearn_version"] = "N/A"
    
    info["numpy_version"] = np.__version__
    
    return info

def log_reproducibility_info(output_path="outputs/reproducibility.json", seed=SEED, config_hash=None):
    """
    Log reproducibility information to a JSON file.
    
    Args:
        output_path (str): Path to save reproducibility information
        seed (int): Random seed used
        config_hash (str): SHA256 hash of the configuration file (optional)
    """
    print(f"[Reproducibility] Logging system info to {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    info = {
        "seed": seed,
        "config_hash": config_hash,
        "deterministic_settings": {
            "torch.backends.cudnn.deterministic": True,
            "torch.backends.cudnn.benchmark": False
        },
        "system_info": get_system_info()
    }
    
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"[Reproducibility] [OK] Saved to {output_path}")
    return info

def print_system_summary():
    """Print a human-readable summary of system information."""
    info = get_system_info()
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"OS: {info['os']['system']} {info['os']['release']}")
    print(f"Processor: {info['os']['processor']}")
    print(f"Python: {info['python']['version_info']}")
    print(f"PyTorch: {info['pytorch']['version']}")
    print(f"NumPy: {info['numpy_version']}")
    print(f"scikit-learn: {info['sklearn_version']}")
    print(f"Device: {info['device']['type'].upper()} ({info['device']['name']})")
    
    if isinstance(info['memory']['total_gb'], (int, float)):
        print(f"RAM: {info['memory']['total_gb']} GB (Available: {info['memory']['available_gb']} GB)")
    else:
        print(f"RAM: {info['memory']['total_gb']}")
    
    print(f"Random Seed: {SEED}")
    print(f"Deterministic Mode: [ENABLED]")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Test the module
    set_seeds()
    print_system_summary()
    log_reproducibility_info()
