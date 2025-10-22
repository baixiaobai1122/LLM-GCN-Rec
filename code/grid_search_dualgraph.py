#!/usr/bin/env python
"""
Grid search for Dual-Graph LightGCN hyperparameters.
Systematically tests different combinations of semantic_weight and semantic_layers.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from itertools import product
import pandas as pd

# Grid search configuration
GRID_CONFIG = {
    'semantic_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'semantic_layers': [1, 2, 3],

    # Fixed parameters
    'dataset': 'amazon-book_subset_1500',
    'path': '../data/amazon-book_subset_1500',
    'epochs': 100,
    'recdim': 64,
    'layer': 3,
    'lr': 0.001,
    'decay': 1e-4,
    'bpr_batch': 2048,
    'seed': 2020,
}

# Results directory
RESULTS_DIR = '../results/grid_search'
LOG_DIR = '../log/grid_search'

def setup_directories():
    """Create necessary directories"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Logs will be saved to: {LOG_DIR}")


def parse_log_file(log_path):
    """
    Parse training log to extract best metrics.

    Returns:
        dict with best_recall, best_epoch, final_recall, final_precision, final_ndcg
    """
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
        lines = f.readlines()

    # Parse best recall
    for line in lines:
        if '✓ New best! Recall@' in line:
            try:
                # Extract: "✓ New best! Recall@20: 0.1803"
                recall_str = line.split('Recall@20: ')[1].strip()
                recall = float(recall_str)
                if recall > metrics['best_recall']:
                    metrics['best_recall'] = recall
                    # Find epoch number from context
                    for prev_line in reversed(lines[:lines.index(line)]):
                        if '[TEST] Epoch' in prev_line:
                            epoch = int(prev_line.split('Epoch ')[1].strip())
                            metrics['best_epoch'] = epoch
                            break
            except:
                pass

    # Parse final results
    in_final_section = False
    for line in lines:
        if 'Final Results:' in line:
            in_final_section = True
        elif in_final_section:
            if 'Recall@[20]:' in line:
                try:
                    recall_str = line.split('[')[2].split(']')[0]
                    metrics['final_recall'] = float(recall_str)
                except:
                    pass
            elif 'Precision@[20]:' in line:
                try:
                    precision_str = line.split('[')[2].split(']')[0]
                    metrics['final_precision'] = float(precision_str)
                except:
                    pass
            elif 'NDCG@[20]:' in line:
                try:
                    ndcg_str = line.split('[')[2].split(']')[0]
                    metrics['final_ndcg'] = float(ndcg_str)
                except:
                    pass
                break

    return metrics


def run_experiment(semantic_weight, semantic_layers, exp_id):
    """
    Run a single experiment with given hyperparameters.

    Returns:
        dict with experiment results
    """
    print(f"\n{'='*80}")
    print(f"Experiment {exp_id}: semantic_weight={semantic_weight}, semantic_layers={semantic_layers}")
    print(f"{'='*80}")

    # Prepare command
    cmd = [
        sys.executable, 'train_dualgraph_standalone.py',
        '--dataset', GRID_CONFIG['dataset'],
        '--path', GRID_CONFIG['path'],
        '--epochs', str(GRID_CONFIG['epochs']),
        '--recdim', str(GRID_CONFIG['recdim']),
        '--layer', str(GRID_CONFIG['layer']),
        '--lr', str(GRID_CONFIG['lr']),
        '--decay', str(GRID_CONFIG['decay']),
        '--bpr_batch', str(GRID_CONFIG['bpr_batch']),
        '--seed', str(GRID_CONFIG['seed']),
        '--semantic_weight', str(semantic_weight),
        '--semantic_layers', str(semantic_layers),
        '--use_semantic_graph', '1',
        '--tensorboard', '0',  # Disable tensorboard for grid search
    ]

    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time

    # Find the log file (most recent one)
    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.startswith('train_dualgraph_')],
                      reverse=True)

    if log_files:
        latest_log = os.path.join(LOG_DIR, log_files[0])
        # Rename to include experiment ID
        new_log_name = f"exp{exp_id:03d}_w{semantic_weight}_l{semantic_layers}_{datetime.now().strftime('%H%M%S')}.log"
        new_log_path = os.path.join(LOG_DIR, new_log_name)

        # Move log from default location
        import glob
        default_logs = glob.glob('../log/train_dualgraph_*.log')
        if default_logs:
            latest_default_log = max(default_logs, key=os.path.getctime)
            os.rename(latest_default_log, new_log_path)
            log_path = new_log_path
        else:
            log_path = None
    else:
        log_path = None

    # Parse results
    if log_path and os.path.exists(log_path):
        metrics = parse_log_file(log_path)
    else:
        metrics = None

    result_dict = {
        'exp_id': exp_id,
        'semantic_weight': semantic_weight,
        'semantic_layers': semantic_layers,
        'elapsed_time': elapsed_time,
        'log_file': os.path.basename(log_path) if log_path else None,
        'success': result.returncode == 0,
    }

    if metrics:
        result_dict.update(metrics)

    print(f"✓ Completed in {elapsed_time:.1f}s")
    if metrics:
        print(f"  Best Recall@20: {metrics['best_recall']:.4f} (Epoch {metrics['best_epoch']})")
        print(f"  Final Recall@20: {metrics['final_recall']:.4f}")

    return result_dict


def main():
    print("="*80)
    print("Dual-Graph LightGCN - Grid Search")
    print("="*80)

    setup_directories()

    # Generate all parameter combinations
    param_combinations = list(product(
        GRID_CONFIG['semantic_weight'],
        GRID_CONFIG['semantic_layers']
    ))

    total_experiments = len(param_combinations)
    print(f"\nTotal experiments to run: {total_experiments}")
    print(f"Parameters:")
    print(f"  semantic_weight: {GRID_CONFIG['semantic_weight']}")
    print(f"  semantic_layers: {GRID_CONFIG['semantic_layers']}")
    print(f"\nEstimated time: ~{total_experiments * 2.5:.1f} minutes")

    # Confirm start
    response = input("\nStart grid search? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Grid search cancelled.")
        return

    # Run experiments
    results = []
    start_time = time.time()

    for i, (weight, layers) in enumerate(param_combinations, 1):
        result = run_experiment(weight, layers, i)
        results.append(result)

        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(RESULTS_DIR, 'grid_search_results_partial.csv'), index=False)

    total_time = time.time() - start_time

    # Save final results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(RESULTS_DIR, f'grid_search_results_{timestamp}.csv')
    df.to_csv(results_file, index=False)

    # Print summary
    print("\n" + "="*80)
    print("Grid Search Completed!")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {results_file}")

    # Sort by best recall and display top 5
    df_sorted = df.sort_values('best_recall', ascending=False)
    print("\n🏆 Top 5 Configurations:")
    print("-"*80)
    for idx, row in df_sorted.head(5).iterrows():
        print(f"{row['exp_id']:3d}. weight={row['semantic_weight']:.1f}, layers={int(row['semantic_layers'])}, "
              f"Recall@20={row['best_recall']:.4f} (Epoch {int(row['best_epoch'])})")

    # Compare with baseline
    baseline_recall = 0.1853  # From your baseline experiment
    best_result = df_sorted.iloc[0]
    improvement = (best_result['best_recall'] - baseline_recall) / baseline_recall * 100

    print(f"\n📊 Comparison with Baseline:")
    print(f"  Baseline Recall@20: {baseline_recall:.4f}")
    print(f"  Best Dual-Graph Recall@20: {best_result['best_recall']:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_experiments': total_experiments,
        'total_time_minutes': total_time / 60,
        'best_config': {
            'semantic_weight': float(best_result['semantic_weight']),
            'semantic_layers': int(best_result['semantic_layers']),
            'best_recall': float(best_result['best_recall']),
            'best_epoch': int(best_result['best_epoch']),
        },
        'baseline_recall': baseline_recall,
        'improvement_percent': float(improvement),
    }

    summary_file = os.path.join(RESULTS_DIR, f'grid_search_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
