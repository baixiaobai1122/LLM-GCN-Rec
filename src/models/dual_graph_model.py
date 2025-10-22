"""
Dual-Graph LightGCN Model

Extends LightGCN with two graphs:
1. User-Item Interaction Graph (collaborative filtering)
2. Item-Item Semantic Graph (content-based similarity from CLIP features)

The model propagates information on both graphs and combines them for final predictions.
"""

import os
import sys
import torch
from torch import nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.baseline import world
from src.baseline.dataloader import BasicDataset


class DualGraphLightGCN(nn.Module):
    """
    LightGCN with dual graph architecture.

    Architecture:
    - Graph 1: User-Item bipartite graph (collaborative filtering signal)
    - Graph 2: Item-Item semantic graph (content-based signal)
    - Combines both graphs for enhanced item representations
    """

    def __init__(self, config: dict, dataset: BasicDataset):
        super(DualGraphLightGCN, self).__init__()
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
            print(f"Dual-graph LightGCN initialized:")
            print(f"  - User-Item graph loaded")
            print(f"  - Semantic graph loaded")
            print(f"  - Semantic weight: {self.semantic_weight}")
            print(f"  - Semantic layers: {self.semantic_layers}")
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

        Returns:
            users_emb: Final user embeddings
            items_emb: Final item embeddings (combined from both graphs)
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

            # Combine item embeddings from both graphs
            items_emb_final = (1 - self.semantic_weight) * items_emb_ui + \
                              self.semantic_weight * items_emb_semantic
        else:
            items_emb_final = items_emb_ui

        return users_emb_ui, items_emb_final

    def getUsersRating(self, users):
        """
        Get predicted ratings for users on all items.

        Args:
            users: User indices

        Returns:
            Predicted ratings
        """
        all_users, all_items = self.computer()
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
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        """
        Compute BPR loss.

        Args:
            users: User indices
            pos: Positive item indices
            neg: Negative item indices

        Returns:
            (loss, reg_loss)
        """
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        """
        Forward pass.

        Args:
            users: User indices
            items: Item indices

        Returns:
            Predicted scores
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma