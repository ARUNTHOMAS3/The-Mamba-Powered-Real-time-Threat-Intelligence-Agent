# 🚀 Running the Benchmark on Kaggle

## Step 1: Create a Kaggle Dataset

Kaggle doesn't use Google Drive — it uses **Kaggle Datasets** for file storage.

### Upload Your Project as a Kaggle Dataset
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"+ New Dataset"**
3. Name it: `mamba-threat-intel`
4. Upload your entire project folder (**without `.venv312/` and `outputs/`**)
5. Make sure `data/raw/CICIDS2017/*.csv` files are included
6. Click **"Create"**

### OR: Use the CICIDS2017 Dataset Already on Kaggle
The CICIDS2017 dataset is already available on Kaggle:
- Search for "CICIDS2017" on Kaggle Datasets
- Add it to your notebook

---

## Step 2: Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. On the right panel, click **"Accelerator"** → Select **GPU T4 x2** or **GPU P100**
4. Click **"+ Add Data"** → search for your uploaded dataset → Add it

---

## Step 3: Paste These Cells

### Cell 1: Setup Project
```python
import os
import shutil

# Your dataset is mounted at /kaggle/input/
# List available datasets
!ls /kaggle/input/

# Copy project to working directory (Kaggle input is read-only)
!cp -r /kaggle/input/mamba-threat-intel /kaggle/working/mamba-threat-intel
os.chdir('/kaggle/working/mamba-threat-intel')

# If CICIDS2017 data is in a separate Kaggle dataset:
# !mkdir -p data/raw/CICIDS2017
# !cp /kaggle/input/cicids2017/*.csv data/raw/CICIDS2017/

# Verify files
!ls data/raw/CICIDS2017/
```

### Cell 2: Install Dependencies
```python
!pip install -q scikit-learn pandas pyyaml joblib

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### Cell 3: Run Full Benchmark
```python
# Full benchmark: 6 models × 5 seeds × 30 epochs
!python -u run_benchmark.py \
    --datasets CICIDS2017 \
    --seeds 42 123 456 789 1024
```

### Cell 4: Generate Tables
```python
!python generate_tables.py
```

### Cell 5: Save Results (Download from Kaggle)
```python
# Results are in /kaggle/working/mamba-threat-intel/outputs/
# Kaggle auto-saves everything in /kaggle/working/ as output

# Zip for easy download
!cd /kaggle/working && zip -r results.zip mamba-threat-intel/outputs/

print("Download from: Notebook → Output tab → results.zip")
```

---

## Key Differences: Kaggle vs Colab

| Feature | Kaggle | Colab |
|---|---|---|
| GPU | T4 x2 or P100 | T4 |
| Session time | 12 hours | ~12 hours (with timeout) |
| Storage | 20 GB | 15 GB |
| Data upload | Kaggle Datasets | Google Drive |
| File access | `/kaggle/input/` (read-only) | `/content/drive/` |
| Save results | `/kaggle/working/` → Output tab | Download manually |
| Internet | Available | Available |

## Saving Trained Models

After training, model checkpoints are saved to:
```
outputs/checkpoints/CICIDS2017_Mamba_seed42.pt
outputs/checkpoints/CICIDS2017_LSTM_seed42.pt
...etc
```

Each `.pt` file contains:
- `model_state_dict` — trained weights
- `model_name`, `dataset`, `seed` — metadata
- `input_dim`, `d_model`, `n_layers` — architecture config
- `best_val_f1`, `epochs_trained` — training info

### Loading a Saved Model Later
```python
import torch
from models.tabular_models import get_model

# Load checkpoint
ckpt = torch.load('outputs/checkpoints/CICIDS2017_Mamba_seed42.pt')

# Recreate model
model = get_model(
    ckpt['model_name'],
    input_dim=ckpt['input_dim'],
    d_model=ckpt['d_model'],
    n_layers=ckpt['n_layers']
)

# Load trained weights
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"Loaded {ckpt['model_name']} (Val F1: {ckpt['best_val_f1']:.4f})")
```

## Estimated Time on Kaggle

| GPU | Estimated Time |
|---|---|
| T4 x2 | ~4-5 hours |
| P100 | ~3-4 hours |

> **Tip**: Enable "Always Save Output" in Notebook Settings so results are preserved even if the session disconnects.
