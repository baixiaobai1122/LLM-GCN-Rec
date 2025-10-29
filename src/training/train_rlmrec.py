"""
Training script for RLMRec models (LightGCN_plus and LightGCN_gene)
Loads LLM embeddings and trains with contrastive or generative alignment
"""
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
def setup_logger(log_dir, dataset_name, model_name):
    """Setup logger to output to both console and file"""
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = join(log_dir, f"train_{model_name}_{dataset_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger('RLMRec')
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


def load_llm_embeddings(dataset_path, num_users, num_items):
    """
    Load LLM embeddings for users and items from RLMRec pickle files.
    Files should be in dataset_path/rlmrec/ directory.
    """
    import pickle

    rlmrec_dir = join(dataset_path, 'rlmrec')

    if not os.path.exists(rlmrec_dir):
        raise FileNotFoundError(f"RLMRec directory not found at {rlmrec_dir}")

    usr_emb_path = join(rlmrec_dir, 'usr_emb_np.pkl')
    itm_emb_path = join(rlmrec_dir, 'itm_emb_np.pkl')

    # Check if files exist
    if not os.path.exists(usr_emb_path):
        raise FileNotFoundError(f"User LLM embeddings not found at {usr_emb_path}")
    if not os.path.exists(itm_emb_path):
        raise FileNotFoundError(f"Item LLM embeddings not found at {itm_emb_path}")

    # Load user LLM embeddings
    with open(usr_emb_path, 'rb') as f:
        user_llm_emb = pickle.load(f)
    cprint(f"Loaded user LLM embeddings from rlmrec/usr_emb_np.pkl: shape={user_llm_emb.shape}")

    # Load item LLM embeddings
    with open(itm_emb_path, 'rb') as f:
        item_llm_emb = pickle.load(f)
    cprint(f"Loaded item LLM embeddings from rlmrec/itm_emb_np.pkl: shape={item_llm_emb.shape}")

    # Check if dimensions match
    if user_llm_emb.shape[0] != num_users:
        raise ValueError(
            f"User LLM embedding count ({user_llm_emb.shape[0]}) "
            f"does not match dataset users ({num_users})"
        )

    if item_llm_emb.shape[0] != num_items:
        raise ValueError(
            f"Item LLM embedding count ({item_llm_emb.shape[0]}) "
            f"does not match dataset items ({num_items})"
        )

    # Convert to float32 for efficiency
    user_llm_emb = user_llm_emb.astype(np.float32)
    item_llm_emb = item_llm_emb.astype(np.float32)

    return user_llm_emb, item_llm_emb


# Setup logger
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
LOG_DIR = join(PROJECT_ROOT, 'logs/rlmrec')
logger, log_file = setup_logger(LOG_DIR, world.dataset, world.model_name)

# Log full configuration
logger.info("\n" + "="*80)
logger.info(f"RLMRec TRAINING - {world.model_name.upper()}")
logger.info("="*80)
logger.info(f"Configuration: {world.config}")
logger.info("="*80 + "\n")

# ==============================
utils.set_seed(world.seed)
logger.info(f">>SEED: {world.seed}")
logger.info(f">>DEVICE: {world.device}")
logger.info(f">>DATASET: {world.dataset}")
logger.info(f">>MODEL: {world.model_name}")
# ==============================

from src.baseline.register import dataset

# Log dataset statistics
logger.info(f"\n{'='*80}")
logger.info(f"Dataset Statistics")
logger.info(f"{'='*80}")
logger.info(f"  - Users: {dataset.n_users}")
logger.info(f"  - Items: {dataset.m_items}")
logger.info(f"  - Train interactions: {dataset.trainDataSize}")
logger.info(f"  - Test interactions: {dataset.testDataSize}")
logger.info(f"  - Sparsity: {1 - (dataset.trainDataSize / (dataset.n_users * dataset.m_items)):.6f}")

# Load LLM embeddings
logger.info(f"\n{'='*80}")
logger.info(f"Loading LLM Embeddings")
logger.info(f"{'='*80}")

DATA_PATH = join(PROJECT_ROOT, 'datasets', world.dataset)

user_llm_emb, item_llm_emb = load_llm_embeddings(DATA_PATH, dataset.n_users, dataset.m_items)

logger.info(f"  - Item LLM embeddings: {item_llm_emb.shape}")
logger.info(f"  - User LLM embeddings: {user_llm_emb.shape}")
logger.info(f"  - LLM embedding dimension: {item_llm_emb.shape[1]}")

# Add LLM embeddings to config
world.config['llm_user_emb'] = user_llm_emb
world.config['llm_item_emb'] = item_llm_emb

# Create model
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

# Create BPR loss wrapper with custom loss handling for RLMRec
class RLMRecBPRLoss:
    def __init__(self, model, config):
        self.model = model
        self.weight_decay = config['decay']
        self.opt = torch.optim.Adam(model.parameters(), lr=config['lr'])

    def stageOne(self, users, pos, neg):
        loss, reg_loss, extra_loss = self.model.bpr_loss(users, pos, neg)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.cpu().item(), reg_loss.cpu().item(), extra_loss.cpu().item()


bpr = RLMRecBPRLoss(Recmodel, world.config)

# Log model configuration
logger.info(f"\n{'='*80}")
logger.info(f"Model Configuration - {world.model_name}")
logger.info(f"{'='*80}")
logger.info(f"  - Model: {world.model_name}")
logger.info(f"  - CF Embedding dim: {world.config['latent_dim_rec']}")
logger.info(f"  - GCN layers: {world.config['lightGCN_n_layers']}")
logger.info(f"  - LLM embedding dim: {item_llm_emb.shape[1]}")

if world.model_name == 'lgn_plus':
    logger.info(f"  - KD weight: {world.config['kd_weight']}")
    logger.info(f"  - KD temperature: {world.config['kd_temperature']}")
elif world.model_name == 'lgn_gene':
    logger.info(f"  - Mask ratio: {world.config['mask_ratio']}")
    logger.info(f"  - Reconstruction weight: {world.config['recon_weight']}")
    logger.info(f"  - Reconstruction temperature: {world.config['re_temperature']}")

logger.info(f"\nOptimization:")
logger.info(f"  - Learning rate: {world.config['lr']}")
logger.info(f"  - Weight decay: {world.config['decay']}")
logger.info(f"  - Batch size: {world.config['bpr_batch_size']}")
logger.info(f"  - Epochs: {world.TRAIN_epochs}")
logger.info(f"  - Dropout: {bool(world.config['dropout'])} (keep_prob={world.config['keep_prob']})")
logger.info(f"  - Seed: {world.seed}")

# Metrics
logger.info(f"\nEvaluation:")
logger.info(f"  - Top-K: {world.topks}")
logger.info(f"  - Test batch size: {world.config['test_u_batch_size']}")
logger.info(f"{'='*80}\n")

# Load weights if specified
weight_file = utils.getFileName()
logger.info(f"Weight file: {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file))
        logger.info(f"Loaded model weights from {weight_file}")
    except FileNotFoundError:
        logger.info(f"Weights file not found, starting from scratch")

# Create tensorboard writer
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + f"{world.model_name}-{world.comment}")
    )
else:
    w = None
    logger.info("Tensorboard disabled")

# Training loop with custom RLMRec loss logging
logger.info("\n" + "="*80)
logger.info("Training Started")
logger.info("="*80)

try:
    best_recall = 0.0
    best_epoch = 0
    patience = 50
    epochs_no_improve = 0

    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        # Test every 10 epochs
        if epoch % 10 == 0:
            cprint(f"[TEST at Epoch {epoch}]")
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch} - Testing")
            logger.info(f"{'='*80}")

            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

            # Log results
            for key, value in results.items():
                # Handle both scalar and array values
                if isinstance(value, np.ndarray):
                    value_str = f"{value[0]:.6f}" if len(value) == 1 else str(value)
                else:
                    value_str = f"{value:.6f}"
                logger.info(f"  {key}: {value_str}")

            # Track best model
            recall_key = 'recall'  # Key is 'recall', not 'recall@20'
            if recall_key in results:
                current_recall = results[recall_key][0] if isinstance(results[recall_key], np.ndarray) else results[recall_key]
                if current_recall > best_recall:
                    best_recall = current_recall
                    best_epoch = epoch
                    epochs_no_improve = 0
                    # Save best model
                    torch.save(Recmodel.state_dict(), weight_file)
                    logger.info(f"  >>> New best model saved! Recall@{world.topks[0]}: {best_recall:.6f}")
                else:
                    epochs_no_improve += 1
            else:
                epochs_no_improve += 1

            logger.info(f"  Best so far: Epoch {best_epoch}, Recall@{world.topks[0]}: {best_recall:.6f}")
            logger.info(f"  Epochs without improvement: {epochs_no_improve}/{patience}")
            logger.info(f"{'='*80}")

            # Early stopping
            if epochs_no_improve >= patience:
                logger.info(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break

        # Training
        output_information = ""
        loss_values = []
        reg_losses = []
        extra_losses = []

        # Sample and train
        S = utils.UniformSample_original(dataset)
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()

        users = users.to(world.device)
        posItems = posItems.to(world.device)
        negItems = negItems.to(world.device)
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)

        total_batch = len(users) // world.config['bpr_batch_size'] + 1
        aver_loss = 0.
        aver_reg_loss = 0.
        aver_extra_loss = 0.

        for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
                utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
            loss, reg_loss, extra_loss = bpr.stageOne(batch_users, batch_pos, batch_neg)
            aver_loss += loss
            aver_reg_loss += reg_loss
            aver_extra_loss += extra_loss

        aver_loss = aver_loss / total_batch
        aver_reg_loss = aver_reg_loss / total_batch
        aver_extra_loss = aver_extra_loss / total_batch

        time_info = time.time() - start

        # Loss name based on model
        extra_loss_name = "KD Loss" if world.model_name == 'lgn_plus' else "Recon Loss"

        output_information += f"Epoch {epoch:>3d} [{time_info:.2f}s]: "
        output_information += f"BPR Loss={aver_loss:.5f} | "
        output_information += f"Reg Loss={aver_reg_loss:.5f} | "
        output_information += f"{extra_loss_name}={aver_extra_loss:.5f}"

        logger.info(output_information)

        # Tensorboard logging
        if w:
            w.add_scalar(f'Loss/BPR-{world.model_name}', aver_loss, epoch)
            w.add_scalar(f'Loss/Reg-{world.model_name}', aver_reg_loss, epoch)
            w.add_scalar(f'Loss/{extra_loss_name}-{world.model_name}', aver_extra_loss, epoch)

    # Final evaluation
    logger.info("\n" + "="*80)
    logger.info("Training Completed - Final Evaluation")
    logger.info("="*80)

    # Load best model
    Recmodel.load_state_dict(torch.load(weight_file))
    logger.info(f"Loaded best model from epoch {best_epoch}")

    results = Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, w, world.config['multicore'])

    logger.info("\nFinal Results:")
    for key, value in results.items():
        # Handle both scalar and array values
        if isinstance(value, np.ndarray):
            value_str = f"{value[0]:.6f}" if len(value) == 1 else str(value)
        else:
            value_str = f"{value:.6f}"
        logger.info(f"  {key}: {value_str}")

    logger.info(f"\nBest epoch: {best_epoch}")
    logger.info(f"Best Recall@{world.topks[0]}: {best_recall:.6f}")
    logger.info("="*80)

finally:
    if w:
        w.close()

logger.info(f"\nTraining log saved to: {log_file}")
logger.info("="*80)