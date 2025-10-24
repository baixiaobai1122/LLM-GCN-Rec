"""
Dual-Graph LightGCN Model with Contrastive Alignment

Extends the standard Dual-Graph LightGCN with contrastive learning to align
the CF space and semantic space.

Key features:
1. User-Item Interaction Graph (collaborative filtering)
2. Item-Item Semantic Graph (content-based similarity)
3. Contrastive alignment loss (InfoNCE) to align CF and semantic embeddings

Reference: InfoNCE Contrastive Learning
"""

import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.baseline import world
from src.baseline.dataloader import BasicDataset


class DualGraphLightGCN_Align(nn.Module):
    """
    LightGCN with dual graph architecture and contrastive alignment.

    Architecture:
    - Graph 1: User-Item bipartite graph (collaborative filtering signal)
    - Graph 2: Item-Item semantic graph (content-based signal)
    - Contrastive loss: Aligns CF and semantic embeddings using InfoNCE

    New features compared to DualGraphLightGCN:
    - alignment_loss(): InfoNCE contrastive loss for space alignment
    - computer() returns separate CF and semantic embeddings before fusion
    - bpr_loss() integrates alignment loss with recommendation loss
    """

    def __init__(self, config: dict, dataset: BasicDataset):
        super(DualGraphLightGCN_Align, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        """Initialize model parameters."""
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # Dual graph specific parameters
        self.use_semantic_graph = self.config.get('use_semantic_graph', True)
        self.semantic_weight = self.config.get('semantic_weight', 0.5)  # Weight for semantic graph
        self.semantic_layers = self.config.get('semantic_layers', 2)  # Layers for semantic propagation

        # Contrastive alignment parameters
        self.use_alignment = self.config.get('use_alignment', True)  # Enable contrastive alignment
        self.alignment_weight = self.config.get('alignment_weight', 0.1)  # Weight for alignment loss
        self.temperature = self.config.get('temperature', 0.07)  # Temperature for InfoNCE

        # Embeddings
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim
        )

        # Initialize embeddings
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')

        self.f = nn.Sigmoid()

        # Load graphs
        self.Graph_UI = self.dataset.getSparseGraph()  # User-Item graph

        if self.use_semantic_graph:
            self.Graph_II = self.dataset.getSemanticGraph()  # Item-Item semantic graph
            print(f"Dual-graph LightGCN with Alignment initialized:")
            print(f"  - User-Item graph loaded")
            print(f"  - Semantic graph loaded")
            print(f"  - Semantic weight: {self.semantic_weight}")
            print(f"  - Semantic layers: {self.semantic_layers}")
            print(f"  - Contrastive alignment: {self.use_alignment}")
            if self.use_alignment:
                print(f"  - Alignment weight: {self.alignment_weight}")
                print(f"  - Temperature: {self.temperature}")
        else:
            self.Graph_II = None
            print("Single-graph LightGCN (semantic graph disabled)")

    def __dropout_x(self, x, keep_prob):
        """Apply dropout to sparse tensor."""
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob, graph):
        """Apply dropout to graph."""
        if self.A_split:
            graph_list = []
            for g in graph:
                graph_list.append(self.__dropout_x(g, keep_prob))
            return graph_list
        else:
            return self.__dropout_x(graph, keep_prob)

    def computer(self):
        """
        Propagate embeddings on both graphs.

        MODIFIED: Returns separate embeddings for contrastive learning.

        Returns:
            users_emb: Final user embeddings from CF graph
            items_emb_cf: Item embeddings from CF graph (User-Item)
            items_emb_semantic: Item embeddings from semantic graph (Item-Item)
                                None if semantic graph is disabled
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        # ========== Graph 1: User-Item Graph ==========
        all_emb = torch.cat([users_emb, items_emb])
        embs_ui = [all_emb]

        # Dropout for training
        if self.config['dropout']:
            if self.training:
                g_droped_ui = self.__dropout(self.keep_prob, self.Graph_UI)
            else:
                g_droped_ui = self.Graph_UI
        else:
            g_droped_ui = self.Graph_UI

        # Propagate on User-Item graph
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped_ui)):
                    temp_emb.append(torch.sparse.mm(g_droped_ui[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped_ui, all_emb)
            embs_ui.append(all_emb)

        embs_ui = torch.stack(embs_ui, dim=1)
        light_out_ui = torch.mean(embs_ui, dim=1)
        users_emb_ui, items_emb_ui = torch.split(light_out_ui, [self.num_users, self.num_items])

        # ========== Graph 2: Item-Item Semantic Graph ==========
        if self.use_semantic_graph and self.Graph_II is not None:
            items_emb_semantic = items_emb
            embs_ii = [items_emb_semantic]

            # Dropout for semantic graph
            if self.config['dropout']:
                if self.training:
                    g_droped_ii = self.__dropout(self.keep_prob, self.Graph_II)
                else:
                    g_droped_ii = self.Graph_II
            else:
                g_droped_ii = self.Graph_II

            # Propagate on Item-Item semantic graph
            for layer in range(self.semantic_layers):
                items_emb_semantic = torch.sparse.mm(g_droped_ii, items_emb_semantic)
                embs_ii.append(items_emb_semantic)

            embs_ii = torch.stack(embs_ii, dim=1)
            items_emb_semantic = torch.mean(embs_ii, dim=1)

            # Return separate embeddings for contrastive learning
            return users_emb_ui, items_emb_ui, items_emb_semantic
        else:
            # No semantic graph, return None for semantic embedding
            return users_emb_ui, items_emb_ui, None

    def alignment_loss(self, emb_cf, emb_sem, temperature=None):
        """
        InfoNCE contrastive loss for aligning CF and semantic embeddings.

        For each item i in the batch:
        - Positive: (emb_cf[i], emb_sem[i])  - same item's two views
        - Negatives: (emb_cf[i], emb_sem[j]) where j != i

        Args:
            emb_cf: CF embeddings [batch_size, dim]
            emb_sem: Semantic embeddings [batch_size, dim]
            temperature: Temperature parameter (default: self.temperature)

        Returns:
            loss: Contrastive alignment loss (scalar)
        """
        if emb_sem is None:
            return 0.0

        if temperature is None:
            temperature = self.temperature

        batch_size = emb_cf.shape[0]

        # L2 normalization (important for cosine similarity)
        emb_cf = F.normalize(emb_cf, dim=1)
        emb_sem = F.normalize(emb_sem, dim=1)

        # Compute similarity matrix [batch_size, batch_size]
        # sim[i,j] = cos_sim(emb_cf[i], emb_sem[j]) / temperature
        sim_matrix = torch.matmul(emb_cf, emb_sem.T) / temperature

        # Labels: diagonal elements are positive pairs (same item)
        labels = torch.arange(batch_size).to(emb_cf.device)

        # Bidirectional contrastive loss
        # 1. CF -> Semantic: maximize similarity of (emb_cf[i], emb_sem[i])
        loss_cf2sem = F.cross_entropy(sim_matrix, labels)

        # 2. Semantic -> CF: symmetric loss
        loss_sem2cf = F.cross_entropy(sim_matrix.T, labels)

        # Average of both directions
        loss = (loss_cf2sem + loss_sem2cf) / 2.0

        return loss

    def getUsersRating(self, users):
        """
        Get predicted ratings for users on all items.

        Args:
            users: User indices

        Returns:
            Predicted ratings
        """
        all_users, all_items_cf, all_items_sem = self.computer()

        # Fuse CF and semantic embeddings for recommendation
        if all_items_sem is not None:
            all_items = (1 - self.semantic_weight) * all_items_cf + \
                       self.semantic_weight * all_items_sem
        else:
            all_items = all_items_cf

        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        """
        Get embeddings for users and items.

        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices

        Returns:
            Tuple of embeddings (propagated and ego)
        """
        all_users, all_items_cf, all_items_sem = self.computer()

        # Fuse CF and semantic embeddings
        if all_items_sem is not None:
            all_items = (1 - self.semantic_weight) * all_items_cf + \
                       self.semantic_weight * all_items_sem
        else:
            all_items = all_items_cf

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        """
        Compute BPR loss with contrastive alignment.

        MODIFIED: Integrates contrastive alignment loss.

        Args:
            users: User indices
            pos: Positive item indices
            neg: Negative item indices

        Returns:
            (total_loss, reg_loss, align_loss): Total loss includes BPR + alignment
        """
        # Get separate CF and semantic embeddings
        all_users, all_items_cf, all_items_sem = self.computer()

        # Fuse for recommendation
        if all_items_sem is not None:
            all_items = (1 - self.semantic_weight) * all_items_cf + \
                       self.semantic_weight * all_items_sem
        else:
            all_items = all_items_cf

        # Get embeddings
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos.long()]
        neg_emb = all_items[neg.long()]

        # Ego embeddings for regularization
        userEmb0 = self.embedding_user(users.long())
        posEmb0 = self.embedding_item(pos.long())
        negEmb0 = self.embedding_item(neg.long())

        # === 1. BPR Loss ===
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # === 2. Regularization Loss ===
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        # === 3. Contrastive Alignment Loss ===
        align_loss = 0.0
        if self.use_alignment and all_items_sem is not None:
            # Only compute alignment loss on positive items in the batch
            pos_items_cf = all_items_cf[pos.long()]
            pos_items_sem = all_items_sem[pos.long()]
            align_loss = self.alignment_loss(pos_items_cf, pos_items_sem)

        # === 4. Total Loss ===
        total_loss = bpr_loss + self.config['decay'] * reg_loss + \
                    self.alignment_weight * align_loss

        return total_loss, reg_loss, align_loss

    def forward(self, users, items):
        """
        Forward pass.

        Args:
            users: User indices
            items: Item indices

        Returns:
            Predicted scores
        """
        all_users, all_items_cf, all_items_sem = self.computer()

        # Fuse CF and semantic embeddings
        if all_items_sem is not None:
            all_items = (1 - self.semantic_weight) * all_items_cf + \
                       self.semantic_weight * all_items_sem
        else:
            all_items = all_items_cf

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma