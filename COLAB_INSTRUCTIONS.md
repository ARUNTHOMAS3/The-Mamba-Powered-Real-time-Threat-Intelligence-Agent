# 🚀 Running the Full Benchmark on Google Colab

## Step 1: Upload Project to Google Drive

1. **Zip** your entire `mamba-threat-intel` folder
2. Upload the zip file to your **Google Drive** (root or any folder)
3. Also upload your `data/raw/CICIDS2017/` CSV files to Drive

---

## Step 2: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a **New Notebook**
3. Go to **Runtime → Change runtime type → GPU (T4)**
4. Paste the cells below **one by one**

---

## Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2: Unzip and Setup Project
```python
import os

# Change this to wherever you uploaded the zip file
ZIP_PATH = '/content/drive/MyDrive/mamba-threat-intel.zip'

!unzip -q "{ZIP_PATH}" -d /content/
%cd /content/mamba-threat-intel

# If your CICIDS2017 data is elsewhere in Drive, copy it:
# !cp -r "/content/drive/MyDrive/CICIDS2017_DATA/"* data/raw/CICIDS2017/

# Verify data
!ls data/raw/CICIDS2017/
```

## Cell 3: Install Dependencies
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install scikit-learn pandas pyyaml joblib

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

## Cell 4: Run FULL Publication Benchmark (6 models × 5 seeds = 30 experiments)
```python
# This runs the full benchmark: 30 epochs, 5 seeds, all data
# Estimated time on T4: ~4-6 hours

!python -u run_benchmark.py \
    --datasets CICIDS2017 \
    --seeds 42 123 456 789 1024
```

## Cell 5: Generate Publication Tables
```python
!python generate_tables.py
```

## Cell 6: Download Results
```python
# Zip results for download
!zip -r /content/benchmark_results.zip outputs/benchmark_results/

from google.colab import files
files.download('/content/benchmark_results.zip')
```

---

## Expected Output

After the benchmark finishes, you'll have:
- `outputs/benchmark_results/evaluation_results.json` — All metrics for all models
- Publication-ready tables with **mean ± std** across 5 seeds
- Per-attack metrics breakdown

## Estimated Timeline on Colab T4 GPU

| Model | Per Seed | Total (5 seeds) |
|---|---|---|
| Mamba | ~15 min | ~75 min |
| LSTM | ~5 min | ~25 min |
| GRU | ~5 min | ~25 min |
| Transformer | ~5 min | ~25 min |
| CNN-LSTM | ~5 min | ~25 min |
| TCN | ~30 min | ~150 min |
| **Total** | | **~5-6 hours** |

> **Tip**: Keep the Colab tab open and active to prevent timeout.
> Use `Runtime → Manage sessions` to monitor.
