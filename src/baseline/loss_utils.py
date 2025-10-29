"""
Loss utility functions for RLMRec models
Includes InfoNCE loss for contrastive alignment and reconstruction losses
"""
import torch
import torch.nn.functional as F


def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    """
    Calculate InfoNCE contrastive loss for aligning CF embeddings with LLM embeddings.

    Args:
        embeds1: CF embeddings (batch_size, embedding_dim)
        embeds2: Corresponding LLM embeddings (batch_size, embedding_dim)
        all_embeds2: All LLM embeddings for negative sampling (num_total, embedding_dim)
        temp: Temperature parameter for InfoNCE loss

    Returns:
        InfoNCE contrastive loss
    """
    # Normalize embeddings
    normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))

    # Positive pair similarity (numerator)
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)

    # All pairs similarity (denominator)
    deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))

    # InfoNCE loss
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss


def ssl_con_loss(x, y, temp=1.0):
    """
    Self-supervised contrastive loss for reconstruction task.

    Args:
        x: Predicted embeddings (batch_size, embedding_dim)
        y: Target embeddings (batch_size, embedding_dim)
        temp: Temperature parameter

    Returns:
        Contrastive reconstruction loss
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    # Positive similarity
    mole = torch.exp(torch.sum(x * y, dim=1) / temp)

    # All pairwise similarities
    deno = torch.sum(torch.exp(x @ y.T / temp), dim=1)

    # Contrastive loss
    loss = -torch.log(mole / (deno + 1e-8) + 1e-8).mean()
    return loss


def sce_loss(x, y, alpha=3):
    """
    Supervised contrastive embedding loss.

    Args:
        x: Embeddings 1 (batch_size, embedding_dim)
        y: Embeddings 2 (batch_size, embedding_dim)
        alpha: Power parameter

    Returns:
        SCE loss
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def alignment(x, y, alpha=2):
    """
    Alignment loss between two embedding sets.

    Args:
        x: Embeddings 1 (batch_size, embedding_dim)
        y: Embeddings 2 (batch_size, embedding_dim)
        alpha: Power parameter

    Returns:
        Alignment loss
    """
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x):
    """
    Uniformity loss to encourage embeddings to be uniformly distributed.

    Args:
        x: Embeddings (batch_size, embedding_dim)

    Returns:
        Uniformity loss
    """
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()