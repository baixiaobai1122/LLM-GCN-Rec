"""
Extract SBERT features from book metadata for semantic similarity graph construction.

This script:
1. Loads book metadata (title, authors, subjects)
2. Creates text descriptions for each book
3. Uses Sentence-BERT to extract semantic features
4. Saves features for graph construction
"""

import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer


class BookSBERTExtractor:
    """Extract SBERT features from book metadata."""

    def __init__(self, model_name="all-mpnet-base-v2", device=None):
        """
        Initialize SBERT model.

        Args:
            model_name: SentenceTransformer model name
            device: torch device (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SBERT model: {model_name} on {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name

        print(f"SBERT model loaded successfully")
        print(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def create_text_description(self, book_data):
        """
        Create text description from book metadata.

        Args:
            book_data: Dictionary with book metadata

        Returns:
            Text description string
        """
        parts = []

        # Add title
        if "title" in book_data and book_data["title"]:
            parts.append(f"Title: {book_data['title']}")

        # Add authors
        if "authors" in book_data and book_data["authors"]:
            authors_str = ", ".join(book_data["authors"])
            parts.append(f"Authors: {authors_str}")

        # Add subjects/topics
        if "subjects" in book_data and book_data["subjects"]:
            # Extract subject names
            subject_names = []
            for subj in book_data["subjects"][:5]:  # Limit to top 5 subjects
                if isinstance(subj, dict) and "name" in subj:
                    subject_names.append(subj["name"])
                elif isinstance(subj, str):
                    subject_names.append(subj)

            if subject_names:
                subjects_str = ", ".join(subject_names)
                parts.append(f"Topics: {subjects_str}")

        # Add description if available (SBERT can handle longer text better than CLIP)
        if "description" in book_data and book_data["description"]:
            # Truncate long descriptions
            desc = book_data["description"]
            if len(desc) > 300:  # SBERT can handle more than CLIP
                desc = desc[:300] + "..."
            parts.append(f"Description: {desc}")

        # Join all parts
        description = ". ".join(parts)

        # Fallback if no metadata available
        if not description.strip():
            description = f"Book: {book_data.get('isbn', 'Unknown')}"

        return description

    def extract_features(self, texts, batch_size=32):
        """
        Extract SBERT features from text descriptions.

        Args:
            texts: List of text descriptions
            batch_size: Batch size for processing

        Returns:
            numpy array of features (N x feature_dim)
        """
        print(f"Extracting features for {len(texts)} items with batch size {batch_size}...")

        features = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        return features


def load_book_metadata(metadata_path):
    """Load book metadata from JSON file."""
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} books")
    return metadata


def load_item_mapping(item_list_path):
    """
    Load item ID mapping from item_list.txt.

    Returns:
        dict: {remap_id: isbn}
    """
    print(f"Loading item mapping from {item_list_path}")
    id_to_isbn = {}

    with open(item_list_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                isbn, remap_id = parts
                id_to_isbn[int(remap_id)] = isbn

    print(f"Loaded mapping for {len(id_to_isbn)} items")
    return id_to_isbn


def main():
    parser = argparse.ArgumentParser(description="Extract SBERT features from book metadata")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformer model name (e.g., all-mpnet-base-v2, all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="sbert_features.npy",
        help="Output filename for features"
    )

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    # Paths
    data_path = Path(args.data_path)
    metadata_path = data_path / "book_metadata.json"
    item_list_path = data_path / "item_list.txt"
    output_path = data_path / args.output_name
    descriptions_path = data_path / "book_descriptions_sbert.json"

    # Load data
    metadata = load_book_metadata(metadata_path)
    id_to_isbn = load_item_mapping(item_list_path)

    # Create extractor
    extractor = BookSBERTExtractor(model_name=args.model_name, device=device)

    # Create text descriptions for each item in order
    num_items = len(id_to_isbn)
    descriptions = []
    description_dict = {}

    print(f"Creating text descriptions for {num_items} items...")
    for item_id in range(num_items):
        isbn = id_to_isbn.get(item_id)

        if isbn and isbn in metadata:
            book_data = metadata[isbn]
            description = extractor.create_text_description(book_data)
        else:
            # Fallback for missing metadata
            description = f"Book item {item_id}"

        descriptions.append(description)
        description_dict[item_id] = description

    # Save descriptions
    print(f"Saving descriptions to {descriptions_path}")
    with open(descriptions_path, 'w', encoding='utf-8') as f:
        json.dump(description_dict, f, indent=2, ensure_ascii=False)

    # Extract SBERT features
    print(f"Extracting SBERT features for {len(descriptions)} items...")
    features = extractor.extract_features(descriptions, batch_size=args.batch_size)

    print(f"Features shape: {features.shape}")
    print(f"Feature dimension: {features.shape[1]}")

    # Save features
    print(f"Saving features to {output_path}")
    np.save(output_path, features)

    # Save metadata
    metadata_output = {
        "num_items": num_items,
        "feature_dim": features.shape[1],
        "model_name": args.model_name,
        "description": f"SBERT ({args.model_name}) text features for book items"
    }

    metadata_output_path = data_path / f"{args.output_name.replace('.npy', '_metadata.json')}"
    with open(metadata_output_path, 'w') as f:
        json.dump(metadata_output, f, indent=2)

    print("\n" + "="*70)
    print("✅ Feature extraction completed!")
    print("="*70)
    print(f"Features saved to: {output_path}")
    print(f"Descriptions saved to: {descriptions_path}")
    print(f"Metadata saved to: {metadata_output_path}")
    print(f"Total items: {num_items}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Model: {args.model_name}")
    print("="*70)


if __name__ == "__main__":
    main()