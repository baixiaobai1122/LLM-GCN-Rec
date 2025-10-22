import sys
import os
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.baseline import world, utils, Procedure, register
from src.baseline.world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
from os.path import join
import logging
from datetime import datetime

# Setup logger
def setup_logger(log_dir, dataset_name):
    """Setup logger to output to both console and file"""
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = join(log_dir, f"train_baseline_{dataset_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger('LightGCN')
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

# Setup logger
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
LOG_DIR = join(PROJECT_ROOT, 'logs/baseline')
logger, log_file = setup_logger(LOG_DIR, world.dataset)

# ==============================
utils.set_seed(world.seed)
logger.info(f">>SEED: {world.seed}")
logger.info(f">>DEVICE: {world.device}")
logger.info(f">>DATASET: {world.dataset}")
# ==============================
from src.baseline.register import dataset

# Log dataset statistics
logger.info(f"\n{'='*60}")
logger.info(f"Dataset Statistics")
logger.info(f"{'='*60}")
logger.info(f"  - Users: {dataset.n_users}")
logger.info(f"  - Items: {dataset.m_items}")
logger.info(f"  - Train interactions: {dataset.trainDataSize}")
logger.info(f"  - Test interactions: {dataset.testDataSize}")

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

# Log model configuration
logger.info(f"\n{'='*60}")
logger.info(f"Model Configuration")
logger.info(f"{'='*60}")
logger.info(f"  - Model: {world.model_name}")
logger.info(f"  - Embedding dim: {world.config['latent_dim_rec']}")
logger.info(f"  - GCN layers: {world.config['lightGCN_n_layers']}")
logger.info(f"  - Learning rate: {world.config['lr']}")
logger.info(f"  - Weight decay: {world.config['decay']}")
logger.info(f"  - Batch size: {world.config['bpr_batch_size']}")

weight_file = utils.getFileName()
logger.info(f"\nModel checkpoint: {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        logger.info(f"✓ Loaded model weights from {weight_file}")
    except FileNotFoundError:
        logger.warning(f"Checkpoint not found, starting from scratch")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
    logger.info(f"Tensorboard enabled: {world.BOARD_PATH}")
else:
    w = None
    logger.info("Tensorboard disabled")

logger.info(f"\n{'='*60}")
logger.info(f"Training for {world.TRAIN_epochs} epochs")
logger.info(f"{'='*60}\n")

best_recall = 0.0
best_epoch = 0

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            logger.info(f"\n[TEST] Epoch {epoch}")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

            # Log test results
            if 'recall' in results and len(results['recall']) > 0:
                logger.info(f"  Recall@{world.topks}: {results['recall']}")
                logger.info(f"  Precision@{world.topks}: {results['precision']}")
                logger.info(f"  NDCG@{world.topks}: {results['ndcg']}")

                current_recall = results['recall'][0]
                if current_recall > best_recall:
                    best_recall = current_recall
                    best_epoch = epoch
                    logger.info(f"✓ New best! Recall@{world.topks[0]}: {best_recall:.4f}")

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        elapsed = time.time() - start
        logger.info(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} Time: {elapsed:.1f}s')
        torch.save(Recmodel.state_dict(), weight_file)

    # Final evaluation
    logger.info(f"\n{'='*60}")
    logger.info("Final Evaluation")
    logger.info(f"{'='*60}")
    final_results = Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, w, world.config['multicore'])

    # Log final results
    logger.info(f"\nFinal Results:")
    logger.info(f"  Recall@{world.topks}: {final_results['recall']}")
    logger.info(f"  Precision@{world.topks}: {final_results['precision']}")
    logger.info(f"  NDCG@{world.topks}: {final_results['ndcg']}")
    logger.info(f"\nBest: Epoch {best_epoch}, Recall@{world.topks[0]}: {best_recall:.4f}")

except Exception as e:
    logger.error(f"Training error: {e}", exc_info=True)
    raise
finally:
    if world.tensorboard:
        w.close()

logger.info("\n✓ Training completed!")
logger.info(f"Log saved to: {log_file}")