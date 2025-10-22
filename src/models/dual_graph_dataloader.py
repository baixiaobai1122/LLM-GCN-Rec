"""
Extended dataloader for dual-graph LightGCN.

Extends the standard Loader class to support loading semantic item-item graphs.
"""

import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.baseline import world
from src.baseline.world import cprint
from src.baseline.dataloader import Loader


class DualGraphLoader(Loader):
    """
    Extended Loader that supports dual graphs:
    1. User-Item interaction graph (from parent class)
    2. Item-Item semantic similarity graph (new)
    """

    def __init__(self, config=world.config, path="../data/gowalla", semantic_graph_file=None):
        """
        Initialize dual-graph loader.

        Args:
            config: Configuration dictionary
            path: Path to dataset directory
            semantic_graph_file: Filename of semantic graph (e.g., 'semantic_graph.npz')
                                If None, will try to load from default location
        """
        # Initialize parent class (loads user-item interactions)
        super().__init__(config, path)

        # Semantic graph parameters
        self.semantic_graph_file = semantic_graph_file or config.get('semantic_graph_file', 'semantic_graph.npz')
        self.use_semantic_graph = config.get('use_semantic_graph', True)
        self.SemanticGraph = None

        if self.use_semantic_graph:
            print(f"Dual-graph loader initialized")
            print(f"  - Semantic graph file: {self.semantic_graph_file}")
        else:
            print("Single-graph loader (semantic graph disabled)")

    def getSemanticGraph(self):
        """
        Load or create item-item semantic similarity graph.

        Returns:
            PyTorch sparse tensor of shape (m_items, m_items)
        """
        if not self.use_semantic_graph:
            print("Semantic graph is disabled")
            return None

        print("Loading semantic graph")
        if self.SemanticGraph is None:
            semantic_path = join(self.path, self.semantic_graph_file)

            try:
                # Try to load pre-computed semantic graph
                semantic_adj = sp.load_npz(semantic_path)
                print(f"Successfully loaded semantic graph from {semantic_path}")
                print(f"  - Shape: {semantic_adj.shape}")
                print(f"  - Edges: {semantic_adj.nnz}")
                print(f"  - Sparsity: {semantic_adj.nnz / (semantic_adj.shape[0] ** 2):.6f}")

                # Verify dimensions
                if semantic_adj.shape[0] != self.m_items or semantic_adj.shape[1] != self.m_items:
                    raise ValueError(
                        f"Semantic graph shape {semantic_adj.shape} does not match "
                        f"number of items {self.m_items}"
                    )

                # Convert to PyTorch sparse tensor
                if self.split:
                    # Split into folds for large graphs
                    self.SemanticGraph = self._split_semantic_graph(semantic_adj)
                    print("Semantic graph split into folds")
                else:
                    self.SemanticGraph = self._convert_sp_mat_to_sp_tensor(semantic_adj)
                    self.SemanticGraph = self.SemanticGraph.coalesce().to(world.device)
                    print("Semantic graph loaded as single tensor")

            except FileNotFoundError:
                print(f"ERROR: Semantic graph file not found: {semantic_path}")
                print("Please run extract_clip_features.py and build_semantic_graph.py first")
                raise

            except Exception as e:
                print(f"ERROR loading semantic graph: {e}")
                raise

        return self.SemanticGraph

    def _split_semantic_graph(self, semantic_adj):
        """
        Split semantic graph into folds for large datasets.

        Args:
            semantic_adj: scipy sparse matrix (m_items, m_items)

        Returns:
            List of PyTorch sparse tensors
        """
        semantic_fold = []
        fold_len = self.m_items // self.folds

        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.m_items
            else:
                end = (i_fold + 1) * fold_len

            fold_tensor = self._convert_sp_mat_to_sp_tensor(semantic_adj[start:end])
            semantic_fold.append(fold_tensor.coalesce().to(world.device))

        return semantic_fold

    def get_statistics(self):
        """
        Get dataset statistics including both graphs.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'n_users': self.n_users,
            'n_items': self.m_items,
            'train_interactions': self.trainDataSize,
            'test_interactions': self.testDataSize,
            'sparsity': (self.trainDataSize + self.testDataSize) / self.n_users / self.m_items,
        }

        # Add semantic graph statistics if available
        if self.use_semantic_graph and self.SemanticGraph is not None:
            if isinstance(self.SemanticGraph, list):
                # Count edges across all folds
                semantic_edges = sum([g._nnz() for g in self.SemanticGraph])
            else:
                semantic_edges = self.SemanticGraph._nnz()

            stats['semantic_edges'] = semantic_edges
            stats['semantic_sparsity'] = semantic_edges / (self.m_items ** 2)

        return stats

    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()

        print("\n" + "="*50)
        print("Dataset Statistics")
        print("="*50)
        print(f"Users: {stats['n_users']}")
        print(f"Items: {stats['n_items']}")
        print(f"Train interactions: {stats['train_interactions']}")
        print(f"Test interactions: {stats['test_interactions']}")
        print(f"Overall sparsity: {stats['sparsity']:.6f}")

        if 'semantic_edges' in stats:
            print("\nSemantic Graph:")
            print(f"  Edges: {stats['semantic_edges']}")
            print(f"  Sparsity: {stats['semantic_sparsity']:.6f}")
        print("="*50 + "\n")