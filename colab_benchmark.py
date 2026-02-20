#!/usr/bin/env python
"""
Google Colab Publication-Ready Benchmark Runner
================================================
Run this in Google Colab to train all 6 models × 5 seeds on a T4 GPU.

Instructions:
    1. Upload this entire project to Google Drive
    2. Open a new Colab notebook with GPU runtime
    3. Mount Drive: drive.mount('/content/drive')
    4. cd to the project: %cd /content/drive/MyDrive/mamba-threat-intel
    5. Install deps: !pip install torch scikit-learn pandas pyyaml
    6. Run: !python -u colab_benchmark.py
"""

import subprocess
import sys
import os
import time

def check_gpu():
    """Verify GPU is available and print info."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ NO GPU DETECTED!")
            print("Go to Runtime → Change runtime type → GPU")
            print("Then restart and try again.")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   PyTorch: {torch.__version__}")
        return True
    except ImportError:
        print("❌ PyTorch not installed!")
        print("Run: !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_data():
    """Check CICIDS2017 data files exist."""
    data_dir = "data/raw/CICIDS2017"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Make sure your CICIDS2017 CSV files are in data/raw/CICIDS2017/")
        return False
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if len(csv_files) == 0:
        print(f"❌ No CSV files found in {data_dir}")
        return False
    
    print(f"✅ Found {len(csv_files)} CSV files:")
    for f in csv_files:
        size_mb = os.path.getsize(os.path.join(data_dir, f)) / 1e6
        print(f"   {f} ({size_mb:.1f} MB)")
    return True


def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "scikit-learn", "pandas", "pyyaml", "joblib"])
    print("✅ Dependencies installed")


def run_benchmark():
    """Run the full publication-ready benchmark."""
    print("\n" + "=" * 70)
    print("🚀 PUBLICATION-READY BENCHMARK")
    print("=" * 70)
    print("Models:  Mamba, LSTM, GRU, Transformer, CNN-LSTM, TCN")
    print("Seeds:   42, 123, 456, 789, 1024")
    print("Epochs:  30 (with early stopping, patience=5)")
    print("Data:    Full CICIDS2017 dataset (no subsampling)")
    print("=" * 70)
    
    start_time = time.time()
    
    result = subprocess.run([
        sys.executable, "-u", "run_benchmark.py",
        "--datasets", "CICIDS2017",
        "--seeds", "42", "123", "456", "789", "1024"
    ])
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print(f"\n{'=' * 70}")
    print(f"⏱️  Total time: {hours}h {minutes}m")
    print(f"{'=' * 70}")
    
    if result.returncode == 0:
        print("✅ Benchmark completed successfully!")
        print("\n📊 Generating publication tables...")
        subprocess.run([sys.executable, "generate_tables.py"])
        print("\n📁 Results saved to: outputs/benchmark_results/")
        print("   → evaluation_results.json (raw metrics)")
        print("   → Publication tables with mean ± std")
    else:
        print(f"❌ Benchmark failed with exit code {result.returncode}")
    
    return result.returncode


def main():
    print("=" * 70)
    print("  Mamba Threat Intelligence — Publication Benchmark")
    print("  Colab GPU Runner")
    print("=" * 70)
    
    # Step 1: Check GPU
    print("\n[1/4] Checking GPU...")
    if not check_gpu():
        sys.exit(1)
    
    # Step 2: Check data
    print("\n[2/4] Checking dataset...")
    if not check_data():
        sys.exit(1)
    
    # Step 3: Install deps
    print("\n[3/4] Installing dependencies...")
    install_dependencies()
    
    # Step 4: Run benchmark
    print("\n[4/4] Starting benchmark...")
    exit_code = run_benchmark()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
