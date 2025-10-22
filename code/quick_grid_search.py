#!/usr/bin/env python
"""
Quick grid search for Dual-Graph LightGCN - tests key parameter combinations.
Faster version with fewer epochs for rapid prototyping.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from itertools import product
import pandas as pd

# Quick search configuration - fewer combinations, fewer epochs
GRID_CONFIG = {
    # Key parameters to search
    'semantic_weight': [0.1, 0.3, 0.5, 0.7],  # 4 values
    'semantic_layers': [1, 2, 3],  # 3 values
    # Total: 12 experiments

    # Fixed parameters
    'dataset': 'amazon-book_subset_1500',
    'path': '../data/amazon-book_subset_1500',
    'epochs': 50,  # Reduced for speed
    'recdim': 64,
    'layer': 3,
    'lr': 0.001,
    'decay': 1e-4,
    'bpr_batch': 2048,
    'seed': 2020,
}

RESULTS_DIR = '../results/quick_search'
LOG_DIR = '../log/quick_search'


def setup_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def parse_log_file(log_path):
    """Extract metrics from log file"""
    if not os.path.exists(log_path):
        return None

    metrics = {
        'best_recall': 0.0,
        'best_epoch': 0,
        'final_recall': 0.0,
        'final_precision': 0.0,
        'final_ndcg': 0.0,
    }

    with open(log_path, 'r') as f:
        content = f.read()

    # Find all "New best" lines
    for line in content.split('\n'):
        if '✓ New best! Recall@20:' in line:
            try:
                recall = float(line.split('Recall@20: ')[1].strip())
                if recall > metrics['best_recall']:
                    metrics['best_recall'] = recall
            except:
                pass

    # Find final results
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Final Results:' in line:
            # Parse next few lines
            for j in range(i+1, min(i+10, len(lines))):
                if 'Recall@[20]:' in lines[j]:
                    try:
                        metrics['final_recall'] = float(lines[j].split('[')[2].split(']')[0])
                    except: pass
                elif 'Precision@[20]:' in lines[j]:
                    try:
                        metrics['final_precision'] = float(lines[j].split('[')[2].split(']')[0])
                    except: pass
                elif 'NDCG@[20]:' in lines[j]:
                    try:
                        metrics['final_ndcg'] = float(lines[j].split('[')[2].split(']')[0])
                    except: pass
            break

    return metrics


def run_experiment(semantic_weight, semantic_layers, exp_id):
    """Run single experiment"""
    print(f"\n{'='*70}")
    print(f"[{exp_id}/12] weight={semantic_weight}, layers={semantic_layers}")
    print(f"{'='*70}")

    cmd = [
        sys.executable, 'train_dualgraph_standalone.py',
        '--dataset', GRID_CONFIG['dataset'],
        '--path', GRID_CONFIG['path'],
        '--epochs', str(GRID_CONFIG['epochs']),
        '--recdim', str(GRID_CONFIG['recdim']),
        '--layer', str(GRID_CONFIG['layer']),
        '--semantic_weight', str(semantic_weight),
        '--semantic_layers', str(semantic_layers),
        '--tensorboard', '0',
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    # Find and rename log
    import glob
    default_logs = glob.glob('../log/train_dualgraph_*.log')
    if default_logs:
        latest_log = max(default_logs, key=os.path.getctime)
        new_name = f"exp{exp_id:02d}_w{semantic_weight}_l{semantic_layers}.log"
        new_path = os.path.join(LOG_DIR, new_name)
        os.rename(latest_log, new_path)
        metrics = parse_log_file(new_path)
    else:
        metrics = None
        new_path = None

    result_dict = {
        'exp_id': exp_id,
        'semantic_weight': semantic_weight,
        'semantic_layers': semantic_layers,
        'elapsed_time': elapsed,
        'success': result.returncode == 0,
    }

    if metrics:
        result_dict.update(metrics)
        print(f"✓ Best Recall@20: {metrics['best_recall']:.4f} ({elapsed:.1f}s)")

    return result_dict


def main():
    print("🚀 Quick Grid Search - Dual-Graph LightGCN")
    print("="*70)

    setup_directories()

    combinations = list(product(
        GRID_CONFIG['semantic_weight'],
        GRID_CONFIG['semantic_layers']
    ))

    print(f"\nExperiments: {len(combinations)}")
    print(f"Epochs per experiment: {GRID_CONFIG['epochs']}")
    print(f"Estimated time: ~{len(combinations) * 1.5:.0f} minutes\n")

    results = []
    start_time = time.time()

    for i, (weight, layers) in enumerate(combinations, 1):
        result = run_experiment(weight, layers, i)
        results.append(result)

    total_time = time.time() - start_time

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'quick_search_{timestamp}.csv')
    df.to_csv(csv_path, index=False)

    # Display results
    print("\n" + "="*70)
    print("📊 Results Summary")
    print("="*70)

    df_sorted = df.sort_values('best_recall', ascending=False)
    print("\n🏆 Top 5 Configurations:")
    for idx, row in df_sorted.head(5).iterrows():
        print(f"  {row['exp_id']:2d}. w={row['semantic_weight']:.1f}, l={int(row['semantic_layers'])}, "
              f"Recall={row['best_recall']:.4f}")

    baseline = 0.1853
    best = df_sorted.iloc[0]
    improvement = (best['best_recall'] - baseline) / baseline * 100

    print(f"\n📈 vs Baseline:")
    print(f"  Baseline: {baseline:.4f}")
    print(f"  Best:     {best['best_recall']:.4f}")
    print(f"  Change:   {improvement:+.2f}%")

    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📁 Results: {csv_path}")

    # Recommendation
    if best['best_recall'] > baseline:
        print(f"\n✅ RECOMMENDATION: Use semantic_weight={best['semantic_weight']}, semantic_layers={int(best['semantic_layers'])}")
        print(f"   Run full 100-epoch training with these parameters!")
    else:
        print(f"\n⚠️  No configuration beats baseline yet.")
        print(f"   Consider: (1) full grid search, (2) larger dataset, (3) different semantic graph")


if __name__ == '__main__':
    main()
