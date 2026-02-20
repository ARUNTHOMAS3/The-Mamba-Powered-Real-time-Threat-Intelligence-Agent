# Efficient Real-Time Intrusion Detection via State Space Models: A Comprehensive Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Comprehensive benchmark study comparing State Space Models (Mamba), Transformers, and Recurrent Neural Networks for network intrusion detection across multiple datasets.**

---

## 🎯 Overview

This repository provides a **rigorous, fair, and reproducible** benchmark comparing **6 sequence-modeling architectures** for Network Intrusion Detection Systems (NIDS), evaluated across **3 standard datasets** with **5 random seeds** and full statistical significance testing.

### Models Compared

| Model | Type | Key Mechanism |
|-------|------|---------------|
| **Mamba (SSM)** | State Space Model | Selective scan, linear-time sequence modeling |
| **LSTM** | Recurrent | Gated memory cells |
| **GRU** | Recurrent | Gated recurrent unit (simplified LSTM) |
| **Transformer** | Attention-based | Multi-head self-attention |
| **CNN-LSTM** | Hybrid | 1D-CNN local features + LSTM temporal |
| **TCN** | Convolutional | Dilated causal convolutions |

### Datasets

| Dataset | Year | Samples | Features | Attack Types |
|---------|------|---------|----------|--------------|
| CICIDS2017 | 2017 | ~2.8M | 77 | DDoS, PortScan, Bot, etc. |
| UNSW-NB15 | 2015 | ~2.5M | 49 | Fuzzers, Exploits, DoS, etc. |
| CIC-IDS2018 | 2018 | ~16M | 80 | Brute Force, DoS, Botnet, etc. |

### Key Contributions

1. **First comprehensive SSM benchmark for IDS** — Systematic comparison of Mamba vs attention-based and recurrent models
2. **Fair evaluation framework** — All models use identical hyperparameters, matched parameter counts (within ±15%), and the same data pipeline
3. **Multi-dimensional analysis** — Classification accuracy, computational efficiency (latency, throughput, memory), and per-attack-type breakdown
4. **Statistical rigor** — 5-seed evaluation with Wilcoxon signed-rank tests and 95% confidence intervals

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/mamba-threat-intel.git
cd mamba-threat-intel

python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

### Dataset Setup

Download datasets into `data/raw/`:

| Dataset | Download |
|---------|----------|
| CICIDS2017 | [UNB Website](https://www.unb.ca/cic/datasets/ids-2017.html) → `data/raw/CICIDS2017/` |
| UNSW-NB15 | [UNSW Website](https://research.unsw.edu.au/projects/unsw-nb15-dataset) → `data/raw/UNSW-NB15/` |
| CIC-IDS2018 | `aws s3 sync --no-sign-request --region ap-south-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" data/raw/CIC-IDS2018/ --exclude "*" --include "*.csv"` |

### Run Benchmark

```bash
# Full benchmark (all datasets, all models, 5 seeds)
python run_benchmark.py

# Quick test (single dataset, 2 models, 1 seed)
python run_benchmark.py --datasets CICIDS2017 --models Mamba LSTM --seeds 42 --quick

# Specific configuration
python run_benchmark.py --datasets CICIDS2017 UNSW-NB15 --models Mamba LSTM GRU Transformer --seeds 42 123 456
```

### Generate Publication Tables

```bash
python generate_tables.py
```

---

## 📁 Project Structure

```
mamba-threat-intel/
├── models/
│   ├── mamba_backbone.py       # Mamba SSM (pure PyTorch S6 implementation)
│   ├── tabular_models.py       # MambaClassifier + LSTMClassifier
│   └── benchmark_models.py     # GRU, Transformer, CNN-LSTM, TCN + model registry
│
├── datasets/
│   ├── cicids2017_loader.py    # CICIDS2017 loader (lazy windowing, temporal split)
│   ├── unswnb15_loader.py      # UNSW-NB15 loader
│   ├── cicids2018_loader.py    # CIC-IDS2018 loader
│   └── dataset_factory.py     # Unified dataset factory
│
├── configs/
│   └── experiment.yaml         # Single source of truth for all hyperparameters
│
├── run_benchmark.py            # Main benchmark runner (single entry point)
├── evaluate.py                 # Evaluation metrics, efficiency, statistical tests
├── generate_tables.py          # Publication table generator
├── verify_benchmark.py         # Verification script
│
├── utils/
│   ├── reproducibility.py      # Seed setting, system info logging
│   ├── config_loader.py        # YAML config loader
│   └── metrics.py              # Classification metrics
│
└── outputs/
    └── benchmark_results/      # All experiment results (JSON)
```

---

## 🔬 Experimental Protocol

### Fair Comparison Guarantees

All models are compared under **strictly identical conditions**:

- ✅ **Same data pipeline**: Identical preprocessing, windowing, and temporal splits (70/10/20)
- ✅ **Same hyperparameters**: Learning rate, batch size, optimizer, loss function
- ✅ **Matched capacity**: All models are within ±15% parameter count
- ✅ **No data leakage**: Scaler fitted on training set only, strict temporal ordering
- ✅ **No shuffling**: Preserves temporal causality
- ✅ **Early stopping**: Patience=5 on validation F1

### Evaluation Metrics

**Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

**Efficiency**: Inference latency (ms), throughput (samples/sec), memory footprint (MB), parameter count

**Analysis**: Per-attack-type F1 breakdown, statistical significance (Wilcoxon signed-rank, p<0.05)

### Reproducibility

- 5 fixed random seeds (42, 123, 456, 789, 1024)
- Deterministic PyTorch operations enabled
- System info and config hash logged for every run
- All results saved as JSON for independent verification

---

## 📊 Configuration

All hyperparameters are controlled from a single file: [`configs/experiment.yaml`](configs/experiment.yaml)

```yaml
datasets: [CICIDS2017, UNSW-NB15, CIC-IDS2018]
models: [Mamba, LSTM, GRU, Transformer, CNN-LSTM, TCN]
seeds: [42, 123, 456, 789, 1024]

dataset:
  seq_len: 50
model:
  d_model: 128
  n_layers: 2
training:
  batch_size: 128
  epochs: 30
  learning_rate: 0.001
  early_stopping:
    patience: 5
```

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- Mamba SSM: [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)
- CICIDS2017: [Sharafaldin et al., 2018](https://www.unb.ca/cic/datasets/ids-2017.html)
- UNSW-NB15: [Moustafa & Slay, 2015](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- CIC-IDS2018: [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
