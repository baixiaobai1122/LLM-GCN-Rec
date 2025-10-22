"""
Training script for Dual-Graph LightGCN.

This script trains a dual-graph LightGCN model that combines:
1. User-Item interaction graph (collaborative filtering)
2. Item-Item semantic similarity graph (content-based from CLIP features)
"""

import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
from os.path import join
import argparse
import sys
import os

# Defer world import to avoid early argument parsing
import importlib

# Parse additional arguments for dual-graph model
def parse_dual_graph_args():
    parser = argparse.ArgumentParser(description="Train Dual-Graph LightGCN")

    # Basic arguments (matching parse.py)
    parser.add_argument('--bpr_batch', type=int, default=2048, help="batch size for BPR training")
    parser.add_argument('--recdim', type=int, default=64, help="embedding dimension")
    parser.add_argument('--layer', type=int, default=3, help="number of GCN layers")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--dropout', type=int, default=0, help="dropout")
    parser.add_argument('--keepprob', type=float, default=0.6, help="keep probability for dropout")
    parser.add_argument('--a_fold', type=int, default=100, help="fold num for adjacency matrix")
    parser.add_argument('--testbatch', type=int, default=100, help="test batch size")
    parser.add_argument('--dataset', type=str, default='amazon-book', help="dataset")
    parser.add_argument('--path', type=str, default="../data/amazon-book", help="dataset path")
    parser.add_argument('--topks', type=str, default="[20]", help="top k for evaluation")
    parser.add_argument('--tensorboard', type=int, default=1, help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="dual-graph-lgn", help="comment for this run")
    parser.add_argument('--load', type=int, default=0, help="load pretrained model")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--multicore', type=int, default=0, help="use multicore for evaluation")
    parser.add_argument('--pretrain', type=int, default=0, help="use pretrained embeddings")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--model', type=str, default='dual-lgn', help="model name")

    # Dual-graph specific arguments
    parser.add_argument('--use_semantic_graph', type=int, default=1,
                       help="use semantic similarity graph (1) or not (0)")
    parser.add_argument('--semantic_weight', type=float, default=0.5,
                       help="weight for semantic graph (0.0-1.0)")
    parser.add_argument('--semantic_layers', type=int, default=2,
                       help="number of propagation layers for semantic graph")
    parser.add_argument('--semantic_graph_file', type=str, default='semantic_graph.npz',
                       help="semantic graph filename")

    args = parser.parse_args()
    return args


def setup_config(args):
    """Setup configuration from arguments."""
    config = {}
    config['bpr_batch_size'] = args.bpr_batch
    config['latent_dim_rec'] = args.recdim
    config['lightGCN_n_layers'] = args.layer
    config['dropout'] = args.dropout
    config['keep_prob'] = args.keepprob
    config['A_n_fold'] = args.a_fold
    config['test_u_batch_size'] = args.testbatch
    config['multicore'] = args.multicore
    config['lr'] = args.lr
    config['decay'] = args.decay
    config['pretrain'] = args.pretrain
    config['A_split'] = False
    config['bigdata'] = False

    # Dual-graph specific config
    config['use_semantic_graph'] = bool(args.use_semantic_graph)
    config['semantic_weight'] = args.semantic_weight
    config['semantic_layers'] = args.semantic_layers
    config['semantic_graph_file'] = args.semantic_graph_file

    return config


def main():
    # Parse arguments FIRST (before importing world)
    args = parse_dual_graph_args()

    # Now import world and other modules that parse args
    import world
    import utils
    import Procedure
    from world import cprint

    # Override world's args with ours
    sys.argv = ['train_dual_graph.py']  # Clear argv to prevent world from re-parsing

    # Setup config
    config = setup_config(args)

    # Set random seed
    utils.set_seed(args.seed)
    print(">>SEED:", args.seed)

    # Device setup
    GPU = torch.cuda.is_available()
    device = torch.device('cuda' if GPU else "cpu")
    print(f">>DEVICE: {device}")

    # Load dataset
    print(f"\n{'='*50}")
    print(f"Loading dataset: {args.dataset}")
    print(f"{'='*50}")

    from dual_graph_dataloader import DualGraphLoader
    dataset = DualGraphLoader(
        config=config,
        path=args.path,
        semantic_graph_file=args.semantic_graph_file
    )

    # Print dataset statistics
    dataset.print_statistics()

    # Load semantic graph if enabled
    if config['use_semantic_graph']:
        try:
            semantic_graph = dataset.getSemanticGraph()
            print("Semantic graph loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load semantic graph: {e}")
            print("Falling back to single-graph mode")
            config['use_semantic_graph'] = False

    # Build model
    print(f"\n{'='*50}")
    print(f"Building model: Dual-Graph LightGCN")
    print(f"{'='*50}")

    from dual_graph_model import DualGraphLightGCN
    Recmodel = DualGraphLightGCN(config, dataset)
    Recmodel = Recmodel.to(device)

    print(f"Model parameters:")
    print(f"  - Embedding dimension: {config['latent_dim_rec']}")
    print(f"  - UI graph layers: {config['lightGCN_n_layers']}")
    print(f"  - Semantic graph layers: {config['semantic_layers']}")
    print(f"  - Semantic weight: {config['semantic_weight']}")
    print(f"  - Learning rate: {config['lr']}")
    print(f"  - Weight decay: {config['decay']}")

    # Setup BPR loss
    bpr = utils.BPRLoss(Recmodel, config)

    # Setup file paths
    weight_file = join(world.FILE_PATH, f"dual-lgn-{args.dataset}-{args.recdim}.pth.tar")
    print(f"\nModel checkpoint: {weight_file}")

    # Load pretrained weights if requested
    if args.load:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=device))
            cprint(f"Loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not found, starting from scratch")

    # Setup tensorboard
    if args.tensorboard:
        log_dir = join(world.BOARD_PATH,
                      time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + args.comment)
        w = SummaryWriter(log_dir)
        print(f"Tensorboard logging to: {log_dir}")
    else:
        w = None
        cprint("Tensorboard disabled")

    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*50}\n")

    topks = eval(args.topks)
    best_recall = 0.0
    best_epoch = 0

    try:
        for epoch in range(args.epochs):
            start = time.time()

            # Test every 10 epochs
            if epoch % 10 == 0:
                cprint("[TEST]")
                results = Procedure.Test(dataset, Recmodel, epoch, w, config['multicore'])

                # Track best model
                if 'recall' in results and len(results['recall']) > 0:
                    current_recall = results['recall'][0]  # Recall@20
                    if current_recall > best_recall:
                        best_recall = current_recall
                        best_epoch = epoch
                        # Save best model
                        best_weight_file = weight_file.replace('.pth.tar', '_best.pth.tar')
                        torch.save(Recmodel.state_dict(), best_weight_file)
                        cprint(f"New best model saved! Recall@{topks[0]}: {best_recall:.4f}")

            # Train
            output_information = Procedure.BPR_train_original(
                dataset, Recmodel, bpr, epoch, neg_k=1, w=w
            )

            elapsed = time.time() - start
            print(f'EPOCH[{epoch+1}/{args.epochs}] {output_information} Time: {elapsed:.2f}s')

            # Save checkpoint
            torch.save(Recmodel.state_dict(), weight_file)

        # Final test
        print(f"\n{'='*50}")
        print("Final Evaluation")
        print(f"{'='*50}")
        Procedure.Test(dataset, Recmodel, args.epochs, w, config['multicore'])

        print(f"\nBest model at epoch {best_epoch} with Recall@{topks[0]}: {best_recall:.4f}")

    finally:
        if args.tensorboard:
            w.close()

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
