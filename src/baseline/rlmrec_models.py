"""
RLMRec: Representation Learning with LLMs for Recommendation
Implements LightGCN_plus (contrastive alignment) and LightGCN_gene (generative alignment)
"""
import torch
from torch import nn
import numpy as np

from . import world
from .dataloader import BasicDataset
from .model import BasicModel
from .loss_utils import cal_infonce_loss, ssl_con_loss


class LightGCN_plus(BasicModel):
    """
    RLMRec-Con: LightGCN with contrastive alignment to LLM embeddings.
    Uses InfoNCE loss to align CF embeddings with LLM-generated semantic embeddings.
    """

    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN_plus, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # CF embeddings
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Initialize CF embeddings
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initializer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')

        # Load LLM embeddings
        if 'llm_user_emb' in self.config and 'llm_item_emb' in self.config:
            self.llm_user_embeds = torch.tensor(self.config['llm_user_emb']).float()
            self.llm_item_embeds = torch.tensor(self.config['llm_item_emb']).float()
            llm_dim = self.llm_item_embeds.shape[1]

            # MLP projection from LLM embeddings to CF embedding space
            hidden_dim = (llm_dim + self.latent_dim) // 2
            self.mlp = nn.Sequential(
                nn.Linear(llm_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, self.latent_dim)
            )

            # Initialize MLP weights
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

            world.cprint(f'Loaded LLM embeddings: users {self.llm_user_embeds.shape}, items {self.llm_item_embeds.shape}')
        else:
            raise ValueError("LLM embeddings not provided in config. Please add 'llm_user_emb' and 'llm_item_emb'.")

        # RLMRec hyperparameters
        self.kd_weight = self.config.get('kd_weight', 0.01)  # Knowledge distillation weight
        self.kd_temperature = self.config.get('kd_temperature', 0.2)  # Temperature for InfoNCE

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"LightGCN_plus is ready (kd_weight={self.kd_weight}, kd_temp={self.kd_temperature})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        Propagate methods for LightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                print("dropping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        # Sum all layers (same as RLMRec source code)
        light_out = sum(embs)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
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
        BPR loss with InfoNCE contrastive alignment to LLM embeddings
        """
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        # Regularization loss: L2 norm of all model parameters (same as RLMRec)
        reg_weight = 1e-07  # Same as RLMRec config
        reg_loss = 0
        for W in self.parameters():
            reg_loss += W.norm(2).square()
        reg_loss = reg_loss * reg_weight
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # InfoNCE contrastive alignment loss
        # Move LLM embeddings to the same device as CF embeddings
        device = users_emb.device
        llm_user_embeds = self.llm_user_embeds.to(device)
        llm_item_embeds = self.llm_item_embeds.to(device)

        # Project LLM embeddings to CF space
        usrprf_embeds = self.mlp(llm_user_embeds)
        itmprf_embeds = self.mlp(llm_item_embeds)

        # Pick batch embeddings
        ancprf_embeds = usrprf_embeds[users.long()]
        posprf_embeds = itmprf_embeds[pos.long()]
        negprf_embeds = itmprf_embeds[neg.long()]

        # Calculate InfoNCE loss for users and items
        # IMPORTANT: pos/neg use batch embeddings (not global) as negative samples
        kd_loss = cal_infonce_loss(users_emb, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_emb, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_emb, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss = kd_loss / float(len(users)) * self.kd_weight

        total_loss = bpr_loss + reg_loss + kd_loss

        return total_loss, reg_loss, kd_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class LightGCN_gene(BasicModel):
    """
    RLMRec-Gen: LightGCN with generative reconstruction of LLM embeddings.
    Masks nodes and reconstructs their LLM embeddings via MLP decoder.
    """

    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN_gene, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # CF embeddings
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Initialize CF embeddings
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initializer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')

        # Load LLM embeddings
        if 'llm_user_emb' in self.config and 'llm_item_emb' in self.config:
            self.llm_user_embeds = torch.tensor(self.config['llm_user_emb']).float()
            self.llm_item_embeds = torch.tensor(self.config['llm_item_emb']).float()
            llm_dim = self.llm_item_embeds.shape[1]

            # MLP decoder to reconstruct LLM embeddings from CF embeddings
            hidden_dim = (llm_dim + self.latent_dim) // 2
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, llm_dim)
            )

            # Initialize decoder weights
            for m in self.decoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

            # Learnable mask token
            self.mask_token = nn.Parameter(torch.zeros(1, self.latent_dim))
            nn.init.normal_(self.mask_token, std=0.1)

            world.cprint(f'Loaded LLM embeddings: users {self.llm_user_embeds.shape}, items {self.llm_item_embeds.shape}')
        else:
            raise ValueError("LLM embeddings not provided in config. Please add 'llm_user_emb' and 'llm_item_emb'.")

        # RLMRec-Gen hyperparameters
        self.mask_ratio = self.config.get('mask_ratio', 0.1)  # Ratio of nodes to mask
        self.recon_weight = self.config.get('recon_weight', 0.1)  # Reconstruction loss weight
        self.re_temperature = self.config.get('re_temperature', 0.2)  # Temperature for reconstruction

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"LightGCN_gene is ready (mask_ratio={self.mask_ratio}, recon_weight={self.recon_weight})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def _mask_nodes(self, embeds):
        """
        Randomly mask a portion of node embeddings for reconstruction
        Returns: masked embeddings and indices of masked nodes
        """
        num_nodes = embeds.shape[0]
        num_mask = int(num_nodes * self.mask_ratio)

        # Random sample nodes to mask
        perm = torch.randperm(num_nodes, device=embeds.device)
        mask_indices = perm[:num_mask]

        # Create masked embeddings
        masked_embeds = embeds.clone()
        masked_embeds[mask_indices] = self.mask_token

        return masked_embeds, mask_indices

    def computer(self, apply_mask=False):
        """
        Propagate methods for LightGCN with optional masking
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        # Apply masking during training for reconstruction task
        masked_indices = None
        if apply_mask and self.training:
            all_emb, masked_indices = self._mask_nodes(all_emb)

        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                print("dropping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        # Sum all layers (same as RLMRec source code)
        light_out = sum(embs)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if apply_mask:
            return users, items, masked_indices
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer(apply_mask=False)
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer(apply_mask=False)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        """
        BPR loss with generative reconstruction of LLM embeddings
        """
        # Standard BPR forward pass
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        # Standard BPR loss
        reg_loss = (1/2) * (userEmb0.norm(2).pow(2) +
                           posEmb0.norm(2).pow(2) +
                           negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # Reconstruction task: mask nodes and reconstruct their LLM embeddings
        all_users_masked, all_items_masked, masked_indices = self.computer(apply_mask=True)
        all_emb_masked = torch.cat([all_users_masked, all_items_masked])

        # Get masked CF embeddings
        masked_cf_embeds = all_emb_masked[masked_indices]

        # Decode to LLM embedding space
        predicted_llm_embeds = self.decoder(masked_cf_embeds)

        # Get target LLM embeddings
        device = masked_cf_embeds.device
        llm_user_embeds = self.llm_user_embeds.to(device)
        llm_item_embeds = self.llm_item_embeds.to(device)
        all_llm_embeds = torch.cat([llm_user_embeds, llm_item_embeds])
        target_llm_embeds = all_llm_embeds[masked_indices]

        # Contrastive reconstruction loss
        recon_loss = ssl_con_loss(predicted_llm_embeds, target_llm_embeds, self.re_temperature)
        recon_loss = recon_loss * self.recon_weight

        total_loss = bpr_loss + reg_loss + recon_loss

        return total_loss, reg_loss, recon_loss

    def forward(self, users, items):
        all_users, all_items = self.computer(apply_mask=False)
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma