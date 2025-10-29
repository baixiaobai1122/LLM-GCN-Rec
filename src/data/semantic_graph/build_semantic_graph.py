"""
Build semantic similarity graph using Faiss from CLIP features.

This script:
1. Loads CLIP features
2. Uses Faiss to find k-nearest neighbors
3. Constructs item-item semantic similarity graph
4. Saves graph in sparse format for LightGCN
"""

import numpy as np
import scipy.sparse as sp
import argparse
import json
from pathlib import Path
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("Installing faiss-cpu...")
    import os
    os.system("pip install faiss-cpu")
    import faiss


class SemanticGraphBuilder:
    """Build semantic similarity graph using Faiss."""

    def __init__(self, features, use_gpu=False):
        """
        Initialize graph builder.

        Args:
            features: numpy array of shape (num_items, feature_dim)
            use_gpu: whether to use GPU for Faiss (if available)
        """
        self.features = features.astype('float32')
        self.num_items = features.shape[0]
        self.feature_dim = features.shape[1]
        self.use_gpu = use_gpu

        print(f"Initialized graph builder:")
        print(f"  - Number of items: {self.num_items}")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Use GPU: {self.use_gpu}")

        # Normalize features for cosine similarity
        faiss.normalize_L2(self.features)

    def build_knn_graph(self, k=10, metric="cosine"):
        """
        Build k-NN graph using Faiss.

        Args:
            k: number of nearest neighbors
            metric: similarity metric ("cosine" or "l2")

        Returns:
            scipy sparse matrix (num_items x num_items)
        """
        print(f"\nBuilding k-NN graph with k={k}, metric={metric}")

        # Create Faiss index
        if metric == "cosine":
            # For cosine similarity, use inner product on normalized vectors
            index = faiss.IndexFlatIP(self.feature_dim)
        else:
            # L2 distance
            index = faiss.IndexFlatL2(self.feature_dim)

        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU for Faiss")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Add features to index
        print("Adding features to index...")
        index.add(self.features)

        # Search for k+1 neighbors (including self)
        print(f"Searching for {k+1} nearest neighbors...")
        distances, indices = index.search(self.features, k + 1)

        # Build sparse adjacency matrix
        print("Building sparse adjacency matrix...")
        row_indices = []
        col_indices = []
        edge_weights = []

        for i in tqdm(range(self.num_items), desc="Processing neighbors"):
            for j in range(1, k + 1):  # Skip first neighbor (self)
                neighbor_idx = indices[i, j]
                similarity = distances[i, j]

                # For L2, convert to similarity
                if metric == "l2":
                    # Convert L2 distance to similarity (closer = higher similarity)
                    similarity = 1.0 / (1.0 + similarity)

                # Add edge (bidirectional)
                row_indices.append(i)
                col_indices.append(neighbor_idx)
                edge_weights.append(similarity)

        # Create sparse matrix
        adjacency_matrix = sp.csr_matrix(
            (edge_weights, (row_indices, col_indices)),
            shape=(self.num_items, self.num_items)
        )

        # Make symmetric (take maximum similarity)
        adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)

        print(f"Graph constructed:")
        print(f"  - Number of edges: {adjacency_matrix.nnz}")
        print(f"  - Sparsity: {adjacency_matrix.nnz / (self.num_items ** 2):.6f}")

        return adjacency_matrix

    def build_threshold_graph(self, threshold=0.5, max_edges_per_node=50):
        """
        Build graph by thresholding similarity scores.

        Args:
            threshold: minimum similarity threshold
            max_edges_per_node: maximum edges per node to prevent dense connections

        Returns:
            scipy sparse matrix (num_items x num_items)
        """
        print(f"\nBuilding threshold graph:")
        print(f"  - Similarity threshold: {threshold}")
        print(f"  - Max edges per node: {max_edges_per_node}")

        # Create Faiss index for cosine similarity
        index = faiss.IndexFlatIP(self.feature_dim)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(self.features)

        # Search for neighbors above threshold
        # Use large k to find all potential neighbors
        k_search = min(max_edges_per_node * 2, self.num_items - 1)
        distances, indices = index.search(self.features, k_search + 1)

        row_indices = []
        col_indices = []
        edge_weights = []

        for i in tqdm(range(self.num_items), desc="Filtering by threshold"):
            edges_added = 0
            for j in range(1, k_search + 1):
                if edges_added >= max_edges_per_node:
                    break

                similarity = distances[i, j]
                if similarity >= threshold:
                    neighbor_idx = indices[i, j]
                    row_indices.append(i)
                    col_indices.append(neighbor_idx)
                    edge_weights.append(similarity)
                    edges_added += 1

        adjacency_matrix = sp.csr_matrix(
            (edge_weights, (row_indices, col_indices)),
            shape=(self.num_items, self.num_items)
        )

        # Make symmetric
        adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)

        print(f"Graph constructed:")
        print(f"  - Number of edges: {adjacency_matrix.nnz}")
        print(f"  - Avg edges per node: {adjacency_matrix.nnz / self.num_items:.2f}")

        return adjacency_matrix

    def normalize_graph(self, adjacency_matrix, method="symmetric"):
        """
        Normalize adjacency matrix.

        Args:
            adjacency_matrix: scipy sparse matrix
            method: "symmetric" (D^{-1/2} A D^{-1/2}) or "random_walk" (D^{-1} A)

        Returns:
            normalized sparse matrix
        """
        print(f"\nNormalizing graph using {method} normalization...")

        # Compute degree matrix
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()

        if method == "symmetric":
            # D^{-1/2}
            d_inv_sqrt = np.power(degrees, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            D_inv_sqrt = sp.diags(d_inv_sqrt)

            # D^{-1/2} A D^{-1/2}
            normalized_matrix = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt

        elif method == "random_walk":
            # D^{-1}
            d_inv = np.power(degrees, -1.0)
            d_inv[np.isinf(d_inv)] = 0.0
            D_inv = sp.diags(d_inv)

            # D^{-1} A
            normalized_matrix = D_inv @ adjacency_matrix

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized_matrix.tocsr()


def load_features(features_path):
    """Load CLIP features from file."""
    print(f"Loading features from {features_path}")
    features = np.load(features_path)
    print(f"Loaded features: {features.shape}")
    return features


def save_graph(graph, output_path, format="npz"):
    """Save graph to file."""
    print(f"Saving graph to {output_path}")

    if format == "npz":
        sp.save_npz(output_path, graph)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Graph saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Build semantic similarity graph with Faiss")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/amazon-book",
        help="Path to data directory"
    )
    parser.add_argument(
        "--features_file",
        type=str,
        default="clip_features.npy",
        help="CLIP features file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="knn",
        choices=["knn", "threshold"],
        help="Graph construction method"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors (for knn method)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold (for threshold method)"
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=50,
        help="Max edges per node (for threshold method)"
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="symmetric",
        choices=["symmetric", "random_walk", "none"],
        help="Normalization method"
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for Faiss (if available)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="semantic_graph.npz",
        help="Output graph filename"
    )

    args = parser.parse_args()

    # Paths
    data_path = Path(args.data_path)
    features_path = data_path / args.features_file

    # Create semantic_graph directory
    graph_dir = data_path / "semantic_graph"
    graph_dir.mkdir(exist_ok=True)
    output_path = graph_dir / args.output_name

    # Load features
    features = load_features(features_path)

    # Build graph
    builder = SemanticGraphBuilder(features, use_gpu=args.use_gpu)

    if args.method == "knn":
        adjacency_matrix = builder.build_knn_graph(k=args.k)
    elif args.method == "threshold":
        adjacency_matrix = builder.build_threshold_graph(
            threshold=args.threshold,
            max_edges_per_node=args.max_edges
        )

    # Normalize if requested
    if args.normalize != "none":
        adjacency_matrix = builder.normalize_graph(adjacency_matrix, method=args.normalize)

    # Save graph
    save_graph(adjacency_matrix, output_path)

    # Save metadata
    metadata = {
        "num_items": adjacency_matrix.shape[0],
        "num_edges": int(adjacency_matrix.nnz),
        "sparsity": float(adjacency_matrix.nnz / (adjacency_matrix.shape[0] ** 2)),
        "method": args.method,
        "k": args.k if args.method == "knn" else None,
        "threshold": args.threshold if args.method == "threshold" else None,
        "normalization": args.normalize,
        "features_file": args.features_file
    }

    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*50)
    print("Semantic graph construction completed!")
    print(f"Graph saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Number of items: {metadata['num_items']}")
    print(f"Number of edges: {metadata['num_edges']}")
    print(f"Sparsity: {metadata['sparsity']:.6f}")
    print("="*50)


if __name__ == "__main__":
    main()