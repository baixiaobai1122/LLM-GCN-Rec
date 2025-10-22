"""
Standalone training script for Dual-Graph LightGCN.
This version doesn't depend on world.py to avoid argument parsing conflicts.
"""

import os
import sys
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
from os.path import join
import argparse
import multiprocessing
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dual-Graph LightGCN")

    # Basic arguments
    parser.add_argument('--bpr_batch', type=int, default=2048)
    parser.add_argument('--recdim', type=int, default=64)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--keepprob', type=float, default=0.6)
    parser.add_argument('--a_fold', type=int, default=100)
    parser.add_argument('--testbatch', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='amazon-book')
    parser.add_argument('--path', type=str, default="../data/amazon-book")
    parser.add_argument('--topks', type=str, default="[20]")
    parser.add_argument('--tensorboard', type=int, default=1)
    parser.add_argument('--comment', type=str, default="dual-graph")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--multicore', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2020)

    # Dual-graph specific arguments
    parser.add_argument('--use_semantic_graph', type=int, default=1)
    parser.add_argument('--semantic_weight', type=float, default=0.5)
    parser.add_argument('--semantic_layers', type=int, default=2)
    parser.add_argument('--semantic_graph_file', type=str, default='semantic_graph.npz')

    return parser.parse_args()


def setup_logger(log_dir, dataset_name):
    """Setup logger to output to both console and file"""
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = join(log_dir, f"train_dualgraph_{dataset_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger('DualGraphTraining')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')

    # File handler (detailed format)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_file}")

    return logger, log_file


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()

    # Setup logger
    LOG_DIR = join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
    logger, log_file = setup_logger(LOG_DIR, args.dataset)

    # Set seed
    set_seed(args.seed)
    logger.info(f">>SEED: {args.seed}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    logger.info(f">>DEVICE: {device}")

    # Setup config
    config = {
        'bpr_batch_size': args.bpr_batch,
        'latent_dim_rec': args.recdim,
        'lightGCN_n_layers': args.layer,
        'dropout': args.dropout,
        'keep_prob': args.keepprob,
        'A_n_fold': args.a_fold,
        'test_u_batch_size': args.testbatch,
        'multicore': args.multicore,
        'lr': args.lr,
        'decay': args.decay,
        'pretrain': args.pretrain,
        'A_split': False,
        'bigdata': False,
        'use_semantic_graph': bool(args.use_semantic_graph),
        'semantic_weight': args.semantic_weight,
        'semantic_layers': args.semantic_layers,
        'semantic_graph_file': args.semantic_graph_file,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Loading dataset: {args.dataset}")
    logger.info(f"{'='*60}")

    # Load dataset
    from dual_graph_dataloader import DualGraphLoader
    dataset = DualGraphLoader(
        config=config,
        path=args.path,
        semantic_graph_file=args.semantic_graph_file
    )

    # Log dataset statistics
    logger.info(f"Dataset Statistics:")
    logger.info(f"  - Users: {dataset.n_users}")
    logger.info(f"  - Items: {dataset.m_items}")
    logger.info(f"  - Train interactions: {dataset.trainDataSize}")
    logger.info(f"  - Test interactions: {dataset.testDataSize}")

    # Load semantic graph if enabled
    if config['use_semantic_graph']:
        try:
            semantic_graph = dataset.getSemanticGraph()
            logger.info("✓ Semantic graph loaded successfully")
        except Exception as e:
            logger.error(f"ERROR: Failed to load semantic graph: {e}")
            logger.warning("Falling back to single-graph mode")
            config['use_semantic_graph'] = False

    # Build model
    logger.info(f"\n{'='*60}")
    logger.info(f"Building Dual-Graph LightGCN")
    logger.info(f"{'='*60}")

    from dual_graph_model import DualGraphLightGCN
    model = DualGraphLightGCN(config, dataset).to(device)

    logger.info(f"Model parameters:")
    logger.info(f"  - Embedding dim: {config['latent_dim_rec']}")
    logger.info(f"  - UI graph layers: {config['lightGCN_n_layers']}")
    logger.info(f"  - Semantic layers: {config['semantic_layers']}")
    logger.info(f"  - Semantic weight: {config['semantic_weight']}")
    logger.info(f"  - Learning rate: {config['lr']}")

    # Setup optimizer
    from utils import BPRLoss
    bpr = BPRLoss(model, config)

    # File paths
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CODE_PATH = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = join(CODE_PATH, 'checkpoints')
    BOARD_PATH = join(CODE_PATH, 'runs')

    os.makedirs(FILE_PATH, exist_ok=True)
    os.makedirs(BOARD_PATH, exist_ok=True)

    weight_file = join(FILE_PATH, f"dualgraph-{args.dataset}-{args.recdim}.pth.tar")
    logger.info(f"\nModel checkpoint: {weight_file}")

    # Load pretrained if requested
    if args.load:
        try:
            model.load_state_dict(torch.load(weight_file, map_location=device))
            logger.info(f"✓ Loaded model from {weight_file}")
        except:
            logger.warning(f"Checkpoint not found, starting from scratch")

    # Setup tensorboard
    w = None
    if args.tensorboard:
        log_dir = join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + args.comment)
        w = SummaryWriter(log_dir)
        logger.info(f"Tensorboard: {log_dir}")

    # Training
    logger.info(f"\n{'='*60}")
    logger.info(f"Training for {args.epochs} epochs")
    logger.info(f"{'='*60}\n")

    import Procedure
    topks = eval(args.topks)
    best_recall = 0.0
    best_epoch = 0

    try:
        for epoch in range(args.epochs):
            start = time.time()

            # Test every 10 epochs
            if epoch % 10 == 0:
                logger.info(f"\n[TEST] Epoch {epoch}")
                results = Procedure.Test(dataset, model, epoch, w, config['multicore'])

                # Log test results
                if 'recall' in results and len(results['recall']) > 0:
                    logger.info(f"  Recall@{topks}: {results['recall']}")
                    logger.info(f"  Precision@{topks}: {results['precision']}")
                    logger.info(f"  NDCG@{topks}: {results['ndcg']}")

                    current_recall = results['recall'][0]
                    if current_recall > best_recall:
                        best_recall = current_recall
                        best_epoch = epoch
                        best_file = weight_file.replace('.pth.tar', '_best.pth.tar')
                        torch.save(model.state_dict(), best_file)
                        logger.info(f"✓ New best! Recall@{topks[0]}: {best_recall:.4f}")

            # Train
            output = Procedure.BPR_train_original(dataset, model, bpr, epoch, neg_k=1, w=w)

            elapsed = time.time() - start
            logger.info(f'EPOCH[{epoch+1}/{args.epochs}] {output} Time: {elapsed:.1f}s')

            # Save checkpoint
            torch.save(model.state_dict(), weight_file)

        # Final test
        logger.info(f"\n{'='*60}")
        logger.info("Final Evaluation")
        logger.info(f"{'='*60}")
        final_results = Procedure.Test(dataset, model, args.epochs, w, config['multicore'])

        # Log final results
        logger.info(f"\nFinal Results:")
        logger.info(f"  Recall@{topks}: {final_results['recall']}")
        logger.info(f"  Precision@{topks}: {final_results['precision']}")
        logger.info(f"  NDCG@{topks}: {final_results['ndcg']}")
        logger.info(f"\nBest: Epoch {best_epoch}, Recall@{topks[0]}: {best_recall:.4f}")

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise
    finally:
        if w:
            w.close()

    logger.info("\n✓ Training completed!")
    logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
