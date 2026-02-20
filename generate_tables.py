#!/usr/bin/env python
"""
Generate Publication-Ready Tables from Benchmark Results

Reads all results from outputs/benchmark_results/ and generates:
  Table 1: Main results (mean ± std across seeds, per dataset, per model)
  Table 2: Efficiency comparison (latency, throughput, memory, params)
  Table 3: Per-attack-type F1 breakdown
  Table 4: Statistical significance matrix (pairwise p-values)

Usage:
    python generate_tables.py
    python generate_tables.py --results-dir outputs/benchmark_results
    python generate_tables.py --format markdown  # or csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import statistical_tests, compute_confidence_interval


def load_results(results_dir):
    """Load all individual result JSON files."""
    results = []
    
    # Try combined file first
    combined = os.path.join(results_dir, 'all_results.json')
    if os.path.exists(combined):
        with open(combined) as f:
            return json.load(f)
    
    # Load individual files
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.json') and fname != 'all_results.json':
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    
    return results


def generate_main_table(results):
    """Table 1: Main classification results (mean ± std)."""
    # Group by dataset -> model -> list of metrics
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r['dataset']][r['model']].append(r['test'])
    
    tables = {}
    for ds_name, models in grouped.items():
        rows = []
        for model_name, metrics_list in models.items():
            accs = [m['accuracy'] for m in metrics_list]
            precs = [m['precision'] for m in metrics_list]
            recs = [m['recall'] for m in metrics_list]
            f1s = [m['f1'] for m in metrics_list]
            aucs = [m['auc_roc'] for m in metrics_list]
            
            row = {
                'Model': model_name,
                'Accuracy': f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
                'Precision': f"{np.mean(precs)*100:.2f} ± {np.std(precs)*100:.2f}",
                'Recall': f"{np.mean(recs)*100:.2f} ± {np.std(recs)*100:.2f}",
                'F1-Score': f"{np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}",
                'AUC-ROC': f"{np.mean(aucs)*100:.2f} ± {np.std(aucs)*100:.2f}",
                '_f1_mean': np.mean(f1s),  # For sorting
            }
            rows.append(row)
        
        # Sort by F1 descending
        rows.sort(key=lambda x: x['_f1_mean'], reverse=True)
        tables[ds_name] = rows
    
    return tables


def generate_efficiency_table(results):
    """Table 2: Efficiency comparison."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r['dataset']][r['model']].append(r)
    
    tables = {}
    for ds_name, models in grouped.items():
        rows = []
        for model_name, result_list in models.items():
            lats = [r['efficiency']['latency_ms'] for r in result_list]
            tputs = [r['efficiency']['throughput_per_sec'] for r in result_list]
            mem = result_list[0]['efficiency']['memory_mb']
            params = result_list[0]['param_count']
            train_times = [r['train']['train_time_sec'] for r in result_list]
            
            rows.append({
                'Model': model_name,
                'Params': f"{params:,}",
                'Memory (MB)': f"{mem:.2f}",
                'Latency (ms)': f"{np.mean(lats):.2f} ± {np.std(lats):.2f}",
                'Throughput (/s)': f"{np.mean(tputs):.0f}",
                'Train Time (s)': f"{np.mean(train_times):.1f}",
                '_lat': np.mean(lats),
            })
        
        rows.sort(key=lambda x: x['_lat'])
        tables[ds_name] = rows
    
    return tables


def generate_attack_table(results):
    """Table 3: Per-attack-type F1 breakdown."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for r in results:
        if 'per_attack' in r and r['per_attack']:
            for atk_type, metrics in r['per_attack'].items():
                grouped[r['dataset']][r['model']][atk_type].append(metrics['f1'])
    
    tables = {}
    for ds_name, models in grouped.items():
        # Get all attack types
        all_attacks = set()
        for model_attacks in models.values():
            all_attacks.update(model_attacks.keys())
        all_attacks = sorted(all_attacks)
        
        rows = []
        for model_name, attacks in models.items():
            row = {'Model': model_name}
            for atk in all_attacks:
                if atk in attacks and attacks[atk]:
                    f1s = attacks[atk]
                    row[atk] = f"{np.mean(f1s)*100:.1f}"
                else:
                    row[atk] = '-'
            rows.append(row)
        
        tables[ds_name] = {'rows': rows, 'attacks': all_attacks}
    
    return tables


def generate_significance_table(results):
    """Table 4: Pairwise statistical significance."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r['dataset']][r['model']].append(r['test']['f1'])
    
    tables = {}
    for ds_name, models in grouped.items():
        if len(models) < 2:
            continue
        
        sig_results = statistical_tests(dict(models))
        
        rows = []
        for (a, b), result in sig_results.items():
            p_val = result.get('p_value')
            sig = result.get('significant')
            
            rows.append({
                'Model A': a,
                'Model B': b,
                'p-value': f"{p_val:.4f}" if p_val is not None else 'N/A',
                'Significant (p<0.05)': '✓' if sig else '✗' if sig is not None else 'N/A',
            })
        
        tables[ds_name] = rows
    
    return tables


def format_markdown_table(headers, rows):
    """Format a list of dicts as a markdown table."""
    # Filter out internal keys starting with _
    headers = [h for h in headers if not h.startswith('_')]
    
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['-' * (len(h) + 2) for h in headers]) + '|')
    
    for row in rows:
        values = [str(row.get(h, '-')) for h in headers]
        lines.append('| ' + ' | '.join(values) + ' |')
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark tables')
    parser.add_argument('--results-dir', default='outputs/benchmark_results',
                        help='Directory containing result JSON files')
    parser.add_argument('--format', choices=['markdown', 'csv'], default='markdown')
    parser.add_argument('--output', default=None,
                        help='Output file (default: print to stdout)')
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        print("   Run 'python run_benchmark.py' first.")
        return
    
    results = load_results(args.results_dir)
    if not results:
        print("❌ No results found. Run experiments first.")
        return
    
    print(f"Loaded {len(results)} experiment results.\n")
    
    output_lines = []
    
    # Table 1: Main Results
    main_tables = generate_main_table(results)
    for ds_name, rows in main_tables.items():
        output_lines.append(f"## Table 1: Classification Results — {ds_name}")
        output_lines.append(f"*Mean ± std across {len(set(r['seed'] for r in results if r['dataset']==ds_name))} seeds*\n")
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        output_lines.append(format_markdown_table(headers, rows))
        output_lines.append("")
    
    # Table 2: Efficiency
    eff_tables = generate_efficiency_table(results)
    for ds_name, rows in eff_tables.items():
        output_lines.append(f"## Table 2: Efficiency Comparison — {ds_name}\n")
        headers = ['Model', 'Params', 'Memory (MB)', 'Latency (ms)', 'Throughput (/s)', 'Train Time (s)']
        output_lines.append(format_markdown_table(headers, rows))
        output_lines.append("")
    
    # Table 3: Per-Attack
    atk_tables = generate_attack_table(results)
    for ds_name, data in atk_tables.items():
        output_lines.append(f"## Table 3: Per-Attack-Type F1 (%) — {ds_name}\n")
        headers = ['Model'] + list(data['attacks'])
        output_lines.append(format_markdown_table(headers, data['rows']))
        output_lines.append("")
    
    # Table 4: Statistical Significance
    sig_tables = generate_significance_table(results)
    for ds_name, rows in sig_tables.items():
        output_lines.append(f"## Table 4: Statistical Significance (Wilcoxon) — {ds_name}\n")
        headers = ['Model A', 'Model B', 'p-value', 'Significant (p<0.05)']
        output_lines.append(format_markdown_table(headers, rows))
        output_lines.append("")
    
    output_text = '\n'.join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Tables saved to {args.output}")
    else:
        print(output_text)
    
    # Also save as markdown file in results dir
    md_path = os.path.join(args.results_dir, 'benchmark_tables.md')
    with open(md_path, 'w') as f:
        f.write(output_text)
    print(f"\n✅ Tables saved to {md_path}")


if __name__ == '__main__':
    main()
