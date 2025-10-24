"""
Extract embeddings from GPT-generated profiles using sentence-transformers.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import os
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer


def load_profiles(profiles_path):
    """Load GPT profiles."""
    print(f"Loading profiles from {profiles_path}")
    with open(profiles_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    # Convert to list sorted by item_id
    num_items = len(profiles)
    profiles_list = []
    for i in range(num_items):
        profiles_list.append(profiles[str(i)])

    print(f"Loaded {len(profiles_list)} profiles")
    return profiles_list


def extract_embeddings(profiles, model_name="all-MiniLM-L6-v2", batch_size=32):
    """Extract embeddings using sentence-transformers."""
    print(f"Loading embedding model: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)

    print(f"Extracting embeddings for {len(profiles)} profiles...")
    embeddings = model.encode(
        profiles,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from GPT profiles"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--profiles_file",
        type=str,
        default="gpt_item_profiles.json",
        help="GPT profiles JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpt_embeddings.npy",
        help="Output filename"
    )

    args = parser.parse_args()

    # Paths
    data_path = Path(args.data_path)
    profiles_path = data_path / args.profiles_file
    output_path = data_path / args.output

    # Load profiles
    profiles = load_profiles(profiles_path)

    # Extract embeddings
    embeddings = extract_embeddings(
        profiles,
        model_name=args.model,
        batch_size=args.batch_size
    )

    # Save embeddings
    print(f"\nSaving embeddings to {output_path}")
    np.save(output_path, embeddings)

    # Save metadata
    metadata = {
        "num_items": len(profiles),
        "embedding_dim": embeddings.shape[1],
        "model": args.model,
        "source": args.profiles_file,
        "description": "Sentence-transformer embeddings of GPT content profiles"
    }

    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*70)
    print("✅ Embedding extraction completed!")
    print("="*70)
    print(f"Number of items: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
