#!/usr/bin/env python
"""
Dual-GPU Parallel Benchmark Runner
====================================
Splits experiments across 2 GPUs for ~2x speedup on Kaggle T4 x2.

Usage:
    python kaggle_dual_gpu.py
"""

import subprocess
import sys
import os
import torch
import time


def main():
    n_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {n_gpus} GPU(s):")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i}: {name} ({vram:.1f} GB)")
    
    if n_gpus < 2:
        print("\n⚠ Only 1 GPU detected. Running normally on GPU 0...")
        os.execv(sys.executable, [sys.executable, "-u", "run_benchmark.py",
                                   "--datasets", "CICIDS2017",
                                   "--seeds", "42", "123", "456", "789", "1024"])
        return
    
    print(f"\n🚀 Running on {n_gpus} GPUs in parallel!")
    print("   GPU 0: Seeds [42, 123, 456]")
    print("   GPU 1: Seeds [789, 1024]")
    print("=" * 60)
    
    start = time.time()
    
    env0 = os.environ.copy()
    env0["CUDA_VISIBLE_DEVICES"] = "0"
    
    env1 = os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Start both processes
    p0 = subprocess.Popen(
        [sys.executable, "-u", "run_benchmark.py",
         "--datasets", "CICIDS2017",
         "--seeds", "42", "123", "456"],
        env=env0,
        stdout=open("gpu0_log.txt", "w"),
        stderr=subprocess.STDOUT
    )
    print(f"✅ GPU 0 started (PID {p0.pid}) — seeds [42, 123, 456]")
    
    p1 = subprocess.Popen(
        [sys.executable, "-u", "run_benchmark.py",
         "--datasets", "CICIDS2017",
         "--seeds", "789", "1024"],
        env=env1,
        stdout=open("gpu1_log.txt", "w"),
        stderr=subprocess.STDOUT
    )
    print(f"✅ GPU 1 started (PID {p1.pid}) — seeds [789, 1024]")
    
    print("\n⏳ Both GPUs training in parallel...")
    print("   Monitor progress: !tail -f gpu0_log.txt")
    print("   Monitor progress: !tail -f gpu1_log.txt")
    
    # Wait for both
    r0 = p0.wait()
    elapsed0 = time.time() - start
    print(f"\n✅ GPU 0 finished (exit={r0}, time={elapsed0/60:.0f}min)")
    
    r1 = p1.wait()
    elapsed1 = time.time() - start
    print(f"✅ GPU 1 finished (exit={r1}, time={elapsed1/60:.0f}min)")
    
    total = time.time() - start
    hours = int(total // 3600)
    mins = int((total % 3600) // 60)
    print(f"\n{'='*60}")
    print(f"🏁 All done! Total time: {hours}h {mins}m")
    print(f"{'='*60}")
    
    # Merge results
    print("\n📊 Merging results from both GPUs...")
    merge_results()
    
    # Generate tables
    print("📋 Generating publication tables...")
    subprocess.run([sys.executable, "generate_tables.py"])
    print("\n✅ Results saved to outputs/benchmark_results/")


def merge_results():
    """Merge evaluation_results.json from both GPU runs."""
    import json
    
    results_path = "outputs/benchmark_results/evaluation_results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"   Found {len(results)} experiment results")
    else:
        print("   ⚠ No results file found — check gpu0_log.txt and gpu1_log.txt for errors")


if __name__ == '__main__':
    main()
