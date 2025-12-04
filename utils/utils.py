# utils/utils.py
import os, random, yaml, time
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mkdirp(path):
    os.makedirs(path, exist_ok=True)

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def now():
    return time.strftime("%Y%m%d_%H%M%S")
