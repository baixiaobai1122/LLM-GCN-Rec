"""
Analyze and visualize semantic similarity graph.

This script provides insights into the quality of the constructed semantic graph:
- Graph statistics
- Example similar items
- Similarity distribution
"""

import numpy as np
import scipy.sparse as sp
import json
import argparse
from pathlib import Path
from collections import Counter


def load_item_mapping(item_list_path):
    """Load item ID to ISBN mapping."""
    id_to_isbn = {}
    with open(item_list_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                isbn, remap_id = parts
                id_to_isbn[int(remap_id)] = isbn
    return id_to_isbn


def load_metadata(metadata_path):
    """Load book metadata."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def analyze_graph(semantic_graph):
    """Analyze graph statistics."""
    print("\n" + "="*60)
    print("Semantic Graph Statistics")
    print("="*60)

    n_nodes = semantic_graph.shape[0]
    n_edges = semantic_graph.nnz

    print(f"Number of nodes (items): {n_nodes}")
    print(f"Number of edges: {n_edges}")
    print(f"Sparsity: {n_edges / (n_nodes ** 2):.6f}")
    print(f"Average degree: {n_edges / n_nodes:.2f}")

    # Degree distribution
    degrees = np.array(semantic_graph.sum(axis=1)).flatten()
    print(f"\nDegree Statistics:")
    print(f"  Min degree: {degrees.min():.0f}")
    print(f"  Max degree: {degrees.max():.0f}")
    print(f"  Mean degree: {degrees.mean():.2f}")
    print(f"  Median degree: {np.median(degrees):.2f}")

    # Weight statistics (if weighted)
    if semantic_graph.dtype != np.int32:
        weights = semantic_graph.data
        print(f"\nEdge Weight Statistics:")
        print(f"  Min weight: {weights.min():.4f}")
        print(f"  Max weight: {weights.max():.4f}")
        print(f"  Mean weight: {weights.mean():.4f}")
        print(f"  Median weight: {np.median(weights):.4f}")

    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'degrees': degrees,
        'weights': semantic_graph.data if semantic_graph.dtype != np.int32 else None
    }


def show_similar_items(semantic_graph, id_to_isbn, metadata, n_examples=5, top_k=10):
    """Show examples of similar items."""
    print("\n" + "="*60)
    print(f"Examples of Similar Items (Top {top_k})")
    print("="*60)

    # Convert to CSR for efficient row access
    graph_csr = semantic_graph.tocsr()

    # Sample some items with many neighbors
    degrees = np.array(graph_csr.sum(axis=1)).flatten()
    high_degree_items = np.argsort(degrees)[::-1][:n_examples*3]

    # Randomly sample from high-degree items
    np.random.seed(42)
    sample_items = np.random.choice(high_degree_items, size=n_examples, replace=False)

    for item_id in sample_items:
        isbn = id_to_isbn.get(item_id)
        if not isbn or isbn not in metadata:
            continue

        book = metadata[isbn]
        title = book.get('title', 'Unknown')
        authors = ', '.join(book.get('authors', []))
        subjects = [s['name'] if isinstance(s, dict) else s
                   for s in book.get('subjects', [])[:3]]

        print(f"\n📚 Item {item_id}: {title}")
        if authors:
            print(f"   Authors: {authors}")
        if subjects:
            print(f"   Subjects: {', '.join(subjects)}")

        # Get neighbors
        row = graph_csr.getrow(item_id)
        neighbor_indices = row.indices
        neighbor_weights = row.data

        # Sort by weight
        sorted_idx = np.argsort(neighbor_weights)[::-1][:top_k]

        print(f"\n   Top {min(top_k, len(sorted_idx))} Similar Books:")
        for rank, idx in enumerate(sorted_idx, 1):
            neighbor_id = neighbor_indices[idx]
            weight = neighbor_weights[idx]

            neighbor_isbn = id_to_isbn.get(neighbor_id)
            if neighbor_isbn and neighbor_isbn in metadata:
                neighbor_book = metadata[neighbor_isbn]
                neighbor_title = neighbor_book.get('title', 'Unknown')
                neighbor_authors = ', '.join(neighbor_book.get('authors', [])[:2])

                print(f"   {rank}. [{weight:.3f}] {neighbor_title}")
                if neighbor_authors:
                    print(f"       by {neighbor_authors}")


def plot_degree_distribution(degrees, output_path=None):
    """Plot degree distribution."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)

        # Log-log plot
        plt.subplot(1, 2, 2)
        degree_counts = Counter(degrees.astype(int))
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]

        plt.loglog(degrees_sorted, counts, 'o-', alpha=0.7)
        plt.xlabel('Degree (log scale)')
        plt.ylabel('Frequency (log scale)')
        plt.title('Degree Distribution (Log-Log)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("\nMatplotlib not installed. Skipping plot.")


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic similarity graph")
    parser.add_argument('--data_path', type=str, default='../data/amazon-book',
                       help='Path to data directory')
    parser.add_argument('--semantic_graph_file', type=str, default='semantic_graph.npz',
                       help='Semantic graph filename')
    parser.add_argument('--n_examples', type=int, default=5,
                       help='Number of example items to show')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of similar items to show per example')
    parser.add_argument('--plot', action='store_true',
                       help='Generate degree distribution plot')

    args = parser.parse_args()

    # Paths
    data_path = Path(args.data_path)
    semantic_graph_path = data_path / args.semantic_graph_file
    item_list_path = data_path / "item_list.txt"
    metadata_path = data_path / "book_metadata.json"

    print("\n" + "="*60)
    print("Semantic Graph Analysis")
    print("="*60)
    print(f"Data path: {data_path}")
    print(f"Graph file: {semantic_graph_path}")

    # Load data
    print("\nLoading data...")
    semantic_graph = sp.load_npz(semantic_graph_path)
    id_to_isbn = load_item_mapping(item_list_path)
    metadata = load_metadata(metadata_path)

    print(f"✓ Loaded semantic graph: {semantic_graph.shape}")
    print(f"✓ Loaded {len(id_to_isbn)} item mappings")
    print(f"✓ Loaded {len(metadata)} book metadata entries")

    # Analyze graph
    stats = analyze_graph(semantic_graph)

    # Show similar items
    show_similar_items(
        semantic_graph,
        id_to_isbn,
        metadata,
        n_examples=args.n_examples,
        top_k=args.top_k
    )

    # Plot degree distribution
    if args.plot:
        plot_path = data_path / "semantic_graph_degree_distribution.png"
        plot_degree_distribution(stats['degrees'], output_path=plot_path)

    print("\n" + "="*60)
    print("Analysis completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
