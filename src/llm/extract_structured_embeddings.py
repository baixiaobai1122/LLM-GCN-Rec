"""
Extract embeddings from structured GPT profiles using multi-vector fusion approach.

This script processes structured item profiles with 6 distinct fields:
- core_themes
- genre_style
- content_features
- comparable_works
- distinctive_traits
- target_readers

Each field is embedded separately and then fused using weighted averaging.
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


# Field names in the structured profile
PROFILE_FIELDS = [
    "core_themes",
    "genre_style",
    "content_features",
    "comparable_works",
    "distinctive_traits",
    "target_readers"
]


def load_structured_profiles(profiles_path):
    """Load structured GPT profiles."""
    print(f"Loading structured profiles from {profiles_path}")
    with open(profiles_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    # Convert to list sorted by item_id
    num_items = len(profiles)
    profiles_list = []
    for i in range(num_items):
        profiles_list.append(profiles[str(i)])

    print(f"Loaded {len(profiles_list)} structured profiles")

    # Validate structure
    if profiles_list:
        sample = profiles_list[0]
        missing_fields = [f for f in PROFILE_FIELDS if f not in sample]
        if missing_fields:
            raise ValueError(f"Missing fields in profile: {missing_fields}")
        print(f"Profile fields: {list(sample.keys())}")

    return profiles_list


def extract_field_texts(profiles, field_name):
    """Extract text for a specific field from all profiles."""
    texts = []
    for profile in profiles:
        text = profile.get(field_name, "")
        if not text:
            print(f"Warning: Empty {field_name} found, using placeholder")
            text = f"No {field_name} available"
        texts.append(text)
    return texts


def extract_multi_vector_embeddings(
    profiles,
    model_name="all-mpnet-base-v2",
    batch_size=32,
    field_weights=None
):
    """
    Extract embeddings using multi-vector fusion approach.

    Args:
        profiles: List of structured profile dictionaries
        model_name: Sentence-transformer model name
        batch_size: Batch size for encoding
        field_weights: Dictionary mapping field names to weights (default: equal weights)

    Returns:
        final_embeddings: Fused embeddings (num_items, embedding_dim)
        field_embeddings: Dictionary of per-field embeddings
    """
    # Default to equal weights if not specified
    if field_weights is None:
        weight_value = 1.0 / len(PROFILE_FIELDS)
        field_weights = {field: weight_value for field in PROFILE_FIELDS}

    # Normalize weights to sum to 1.0
    weight_sum = sum(field_weights.values())
    field_weights = {k: v/weight_sum for k, v in field_weights.items()}

    print(f"\nField weights (normalized):")
    for field, weight in field_weights.items():
        print(f"  {field}: {weight:.4f}")

    print(f"\nLoading embedding model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    num_items = len(profiles)

    print(f"\nExtracting embeddings for {num_items} profiles...")
    print(f"Embedding dimension: {embedding_dim}")

    # Store embeddings for each field
    field_embeddings = {}

    # Process each field separately
    for field in PROFILE_FIELDS:
        print(f"\n--- Processing field: {field} ---")

        # Extract texts for this field
        texts = extract_field_texts(profiles, field)

        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better fusion
        )

        field_embeddings[field] = embeddings
        print(f"  Shape: {embeddings.shape}")

    # Fuse embeddings using weighted averaging
    print(f"\n--- Fusing embeddings ---")
    final_embeddings = np.zeros((num_items, embedding_dim), dtype=np.float32)

    for field, weight in field_weights.items():
        final_embeddings += weight * field_embeddings[field]

    # Normalize final embeddings
    norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    final_embeddings = final_embeddings / norms

    print(f"Final embeddings shape: {final_embeddings.shape}")

    return final_embeddings, field_embeddings


def save_embeddings(
    output_path,
    final_embeddings,
    field_embeddings,
    metadata
):
    """Save embeddings and metadata."""
    # Create gpt_embeddings directory
    gpt_emb_dir = output_path.parent / "gpt_embeddings"
    gpt_emb_dir.mkdir(exist_ok=True)

    # Update output path to be inside gpt_embeddings/
    final_output_path = gpt_emb_dir / output_path.name

    # Save final fused embeddings
    print(f"\nSaving final embeddings to {final_output_path}")
    np.save(final_output_path, final_embeddings)

    # Save per-field embeddings
    output_dir = output_path.parent
    field_dir = output_dir / "field_embeddings"
    field_dir.mkdir(exist_ok=True)

    print(f"Saving per-field embeddings to {field_dir}/")
    for field, embeddings in field_embeddings.items():
        field_path = field_dir / f"{field}_embeddings.npy"
        np.save(field_path, embeddings)
        print(f"  Saved: {field_path.name}")

    # Save metadata in gpt_embeddings/
    metadata_path = gpt_emb_dir / f"{final_output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from structured GPT profiles using multi-vector fusion"
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
        default="gpt_item_profiles_structured.json",
        help="Structured profiles JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence-transformer model name (e.g., all-mpnet-base-v2, all-MiniLM-L6-v2)"
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
        default="gpt_structured_embeddings.npy",
        help="Output filename for final fused embeddings"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="JSON string of field weights, e.g., '{\"core_themes\": 0.25, \"genre_style\": 0.20, ...}'"
    )

    args = parser.parse_args()

    # Paths
    data_path = Path(args.data_path)
    profiles_path = data_path / args.profiles_file
    output_path = data_path / args.output

    # Parse custom weights if provided
    field_weights = None
    if args.weights:
        try:
            field_weights = json.loads(args.weights)
            print(f"Using custom field weights: {field_weights}")
        except json.JSONDecodeError as e:
            print(f"Error parsing weights JSON: {e}")
            print("Using default equal weights")

    # Load profiles
    profiles = load_structured_profiles(profiles_path)

    # Extract embeddings
    final_embeddings, field_embeddings = extract_multi_vector_embeddings(
        profiles,
        model_name=args.model,
        batch_size=args.batch_size,
        field_weights=field_weights
    )

    # Prepare metadata
    weights_used = field_weights if field_weights else {
        field: 1.0/len(PROFILE_FIELDS) for field in PROFILE_FIELDS
    }

    metadata = {
        "num_items": len(profiles),
        "embedding_dim": final_embeddings.shape[1],
        "model": args.model,
        "source": args.profiles_file,
        "method": "multi_vector_fusion",
        "field_weights": weights_used,
        "fields": PROFILE_FIELDS,
        "description": "Multi-vector fusion embeddings: each field embedded separately then weighted fusion"
    }

    # Save everything
    save_embeddings(output_path, final_embeddings, field_embeddings, metadata)

    # Print summary
    print("\n" + "="*70)
    print("✅ Structured embedding extraction completed!")
    print("="*70)
    print(f"Number of items: {final_embeddings.shape[0]}")
    print(f"Embedding dimension: {final_embeddings.shape[1]}")
    print(f"Model: {args.model}")
    print(f"Method: Multi-vector fusion with weighted averaging")
    print(f"Field weights:")
    for field, weight in weights_used.items():
        print(f"  {field}: {weight:.4f}")
    print(f"\nOutput files:")
    print(f"  Final embeddings: {output_path.parent / 'gpt_embeddings' / output_path.name}")
    print(f"  Field embeddings: {output_path.parent / 'field_embeddings'}/")
    print(f"  Metadata: {output_path.parent / 'gpt_embeddings' / f'{output_path.stem}_metadata.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
