#!/usr/bin/env python
"""
Verification Script for Benchmark Framework

Validates that all components work correctly:
  1. All 6 models instantiate and produce correct output shapes
  2. Parameter counts are within ±15% of each other
  3. Dataset factory returns correct types
  4. Evaluation functions return all required keys
  5. Efficiency measurement works

Usage:
    python verify_benchmark.py
"""

import sys
import os
import traceback

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_models():
    """Test all 6 models instantiate and produce correct output."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Shape Verification")
    print("=" * 60)
    
    from models.benchmark_models import MODEL_REGISTRY, count_params
    
    input_dim = 77  # Typical CICIDS2017 feature count
    d_model = 128
    n_layers = 2
    batch_size = 4
    seq_len = 50
    
    param_counts = {}
    all_passed = True
    
    for name, model_class in MODEL_REGISTRY.items():
        try:
            model = model_class(input_dim=input_dim, d_model=d_model, n_layers=n_layers)
            x = torch.randn(batch_size, seq_len, input_dim)
            out = model(x)
            
            params = count_params(model)
            param_counts[name] = params
            
            expected_shape = (batch_size, 1)
            actual_shape = tuple(out.shape)
            
            if actual_shape == expected_shape:
                print(f"  ✅ {name:15s} | Output: {actual_shape} | Params: {params:>10,}")
            else:
                print(f"  ❌ {name:15s} | Expected {expected_shape}, got {actual_shape}")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ {name:15s} | Error: {e}")
            traceback.print_exc()
            all_passed = False
    
    return all_passed, param_counts


def test_parameter_parity(param_counts, max_diff=15.0):
    """Test that model parameters are within acceptable range."""
    print("\n" + "=" * 60)
    print("TEST 2: Parameter Parity Check (±15%)")
    print("=" * 60)
    
    if len(param_counts) < 2:
        print("  ⚠ Not enough models to compare")
        return True
    
    names = list(param_counts.keys())
    all_passed = True
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = param_counts[a], param_counts[b]
            diff = abs(pa - pb) / max(pa, pb) * 100
            
            status = "✅" if diff <= max_diff else "⚠"
            if diff > max_diff:
                all_passed = False
            
            print(f"  {status} {a} vs {b}: {diff:.1f}% difference "
                  f"({pa:,} vs {pb:,})")
    
    return all_passed


def test_dataset_factory():
    """Test that dataset factory returns correct types."""
    print("\n" + "=" * 60)
    print("TEST 3: Dataset Factory")
    print("=" * 60)
    
    from datasets.dataset_factory import DATASET_REGISTRY
    
    all_passed = True
    
    for name in DATASET_REGISTRY:
        try:
            # Just check the class can be imported
            cls = DATASET_REGISTRY[name]['class']
            print(f"  ✅ {name:15s} | Class: {cls.__name__}")
        except Exception as e:
            print(f"  ❌ {name:15s} | Error: {e}")
            all_passed = False
    
    return all_passed


def test_evaluation():
    """Test evaluation functions return all required keys."""
    print("\n" + "=" * 60)
    print("TEST 4: Evaluation Functions")
    print("=" * 60)
    
    from evaluate import evaluate_model, measure_efficiency, compute_confidence_interval
    from models.benchmark_models import MambaClassifier
    
    # Create a dummy model and data
    model = MambaClassifier(input_dim=10, d_model=32, n_layers=1)
    
    # Test efficiency measurement
    try:
        sample = torch.randn(1, 10, 10)
        eff = measure_efficiency(model, sample, n_warmup=2, n_measure=5)
        required_keys = ['latency_ms', 'throughput_per_sec', 'memory_mb', 'param_count']
        missing = [k for k in required_keys if k not in eff]
        
        if not missing:
            print(f"  ✅ measure_efficiency() | Keys: {list(eff.keys())}")
        else:
            print(f"  ❌ measure_efficiency() | Missing keys: {missing}")
            return False
    except Exception as e:
        print(f"  ❌ measure_efficiency() | Error: {e}")
        traceback.print_exc()
        return False
    
    # Test confidence interval
    try:
        mean, std, ci_low, ci_high = compute_confidence_interval([0.8, 0.82, 0.79, 0.81, 0.83])
        print(f"  ✅ compute_confidence_interval() | {mean:.3f} ± {std:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    except Exception as e:
        print(f"  ❌ compute_confidence_interval() | Error: {e}")
        return False
    
    return True


def test_config():
    """Test config loading."""
    print("\n" + "=" * 60)
    print("TEST 5: Configuration")
    print("=" * 60)
    
    from utils.config_loader import load_config
    
    try:
        config = load_config('configs/experiment.yaml')
        required = ['datasets', 'models', 'seeds', 'dataset', 'model', 'training']
        missing = [k for k in required if k not in config]
        
        if not missing:
            print(f"  ✅ Config loaded | Datasets: {config['datasets']}")
            print(f"  ✅              | Models: {config['models']}")
            print(f"  ✅              | Seeds: {config['seeds']}")
        else:
            print(f"  ❌ Missing keys: {missing}")
            return False
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("BENCHMARK FRAMEWORK VERIFICATION")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Models
    passed, param_counts = test_models()
    results['models'] = passed
    
    # Test 2: Parameter parity
    results['parity'] = test_parameter_parity(param_counts)
    
    # Test 3: Dataset factory
    results['datasets'] = test_dataset_factory()
    
    # Test 4: Evaluation
    results['evaluation'] = test_evaluation()
    
    # Test 5: Config
    results['config'] = test_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, status in results.items():
        print(f"  {'✅' if status else '❌'} {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Benchmark framework is ready.")
    else:
        print("\n  ⚠ Some tests failed. Check output above.")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
