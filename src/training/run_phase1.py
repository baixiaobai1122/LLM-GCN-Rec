"""
Phase 1 Quick Validation Pipeline

Runs the complete Phase 1 pipeline:
1. Extract CLIP features from book metadata
2. Build semantic similarity graph using Faiss
3. Train Dual-Graph LightGCN

Usage:
    python run_phase1.py --dataset amazon-book --skip_feature_extraction --skip_graph_building

Options:
    --skip_feature_extraction: Skip CLIP feature extraction (if already done)
    --skip_graph_building: Skip Faiss graph construction (if already done)
    --k: Number of nearest neighbors for semantic graph (default: 10)
    --semantic_weight: Weight for semantic graph (default: 0.5)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and check for errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n{description} completed successfully!")
    return result


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ Found {description}: {filepath}")
        return True
    else:
        print(f"✗ Missing {description}: {filepath}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Quick Validation Pipeline")

    # Dataset config
    parser.add_argument('--dataset', type=str, default='amazon-book',
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default='../data/amazon-book',
                       help='Path to dataset directory')

    # Pipeline control
    parser.add_argument('--skip_feature_extraction', action='store_true',
                       help='Skip CLIP feature extraction')
    parser.add_argument('--skip_graph_building', action='store_true',
                       help='Skip semantic graph building')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip model training')

    # Feature extraction config
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')

    # Graph building config
    parser.add_argument('--k', type=int, default=10,
                       help='Number of nearest neighbors for semantic graph')
    parser.add_argument('--graph_method', type=str, default='knn',
                       choices=['knn', 'threshold'],
                       help='Graph construction method')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for threshold method')
    parser.add_argument('--normalize', type=str, default='symmetric',
                       choices=['symmetric', 'random_walk', 'none'],
                       help='Graph normalization method')

    # Training config
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of LightGCN layers')
    parser.add_argument('--semantic_weight', type=float, default=0.5,
                       help='Weight for semantic graph (0.0-1.0)')
    parser.add_argument('--semantic_layers', type=int, default=2,
                       help='Number of semantic propagation layers')

    args = parser.parse_args()

    # File paths
    data_path = Path(args.data_path)
    features_file = data_path / "clip_features.npy"
    semantic_graph_file = data_path / "semantic_graph.npz"
    metadata_file = data_path / "book_metadata.json"

    print("\n" + "="*60)
    print("PHASE 1: DUAL-GRAPH LIGHTGCN QUICK VALIDATION")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {data_path}")
    print(f"CLIP model: {args.clip_model}")
    print(f"k-NN: {args.k}")
    print(f"Semantic weight: {args.semantic_weight}")
    print("="*60)

    # Check prerequisites
    print("\nChecking prerequisites...")
    if not check_file_exists(metadata_file, "Book metadata"):
        print(f"\nERROR: Book metadata file not found: {metadata_file}")
        print("Please run fetch_book_metadata.py first")
        sys.exit(1)

    # Step 1: Extract CLIP features
    if not args.skip_feature_extraction:
        print("\n" + "="*60)
        print("STEP 1: Extract CLIP Features")
        print("="*60)

        cmd = [
            sys.executable, "extract_clip_features.py",
            "--data_path", str(data_path),
            "--model_name", args.clip_model,
            "--batch_size", str(args.batch_size),
            "--output_name", "clip_features.npy"
        ]

        run_command(cmd, "CLIP Feature Extraction")
    else:
        print("\nSkipping CLIP feature extraction...")
        if not check_file_exists(features_file, "CLIP features"):
            print("ERROR: Features file not found but skip requested")
            sys.exit(1)

    # Step 2: Build semantic graph
    if not args.skip_graph_building:
        print("\n" + "="*60)
        print("STEP 2: Build Semantic Graph with Faiss")
        print("="*60)

        cmd = [
            sys.executable, "build_semantic_graph.py",
            "--data_path", str(data_path),
            "--features_file", "clip_features.npy",
            "--method", args.graph_method,
            "--k", str(args.k),
            "--threshold", str(args.threshold),
            "--normalize", args.normalize,
            "--output_name", "semantic_graph.npz"
        ]

        # Add GPU flag if available
        try:
            import torch
            if torch.cuda.is_available():
                cmd.append("--use_gpu")
        except ImportError:
            pass

        run_command(cmd, "Semantic Graph Construction")
    else:
        print("\nSkipping semantic graph building...")
        if not check_file_exists(semantic_graph_file, "Semantic graph"):
            print("ERROR: Semantic graph file not found but skip requested")
            sys.exit(1)

    # Step 3: Train Dual-Graph LightGCN
    if not args.skip_training:
        print("\n" + "="*60)
        print("STEP 3: Train Dual-Graph LightGCN")
        print("="*60)

        cmd = [
            sys.executable, "train_dual_graph.py",
            "--dataset", args.dataset,
            "--path", str(data_path),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--recdim", str(args.embed_dim),
            "--layer", str(args.layers),
            "--use_semantic_graph", "1",
            "--semantic_weight", str(args.semantic_weight),
            "--semantic_layers", str(args.semantic_layers),
            "--semantic_graph_file", "semantic_graph.npz",
            "--comment", f"phase1-k{args.k}-sw{args.semantic_weight}"
        ]

        run_command(cmd, "Dual-Graph LightGCN Training")
    else:
        print("\nSkipping training...")

    # Summary
    print("\n" + "="*60)
    print("PHASE 1 PIPELINE COMPLETED!")
    print("="*60)
    print("\nGenerated files:")
    if features_file.exists():
        print(f"  ✓ CLIP features: {features_file}")
    if semantic_graph_file.exists():
        print(f"  ✓ Semantic graph: {semantic_graph_file}")

    print("\nNext steps:")
    print("  1. Check tensorboard logs for training curves")
    print("  2. Evaluate model performance")
    print("  3. Tune hyperparameters (k, semantic_weight, etc.)")
    print("  4. Compare with baseline LightGCN")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
