"""
Extract Multimodal CLIP features from book metadata (Text + Images).

This enhanced version:
1. Loads book metadata (title, authors, subjects)
2. Downloads book cover images from URLs
3. Uses CLIP to extract both text and image features
4. Fuses multimodal features for better semantic representation
5. Saves features for Faiss indexing
"""

import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import time

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("Installing transformers and required dependencies...")
    os.system("pip install transformers torch torchvision pillow")
    from transformers import CLIPProcessor, CLIPModel


class MultimodalBookCLIPExtractor:
    """Extract CLIP features from both text and images."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize CLIP model.

        Args:
            model_name: HuggingFace model name for CLIP
            device: torch device (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model: {model_name} on {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        print(f"CLIP model loaded successfully (device: {self.device})")

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
            subject_names = []
            for subj in book_data["subjects"][:5]:
                if isinstance(subj, dict) and "name" in subj:
                    subject_names.append(subj["name"])
                elif isinstance(subj, str):
                    subject_names.append(subj)

            if subject_names:
                subjects_str = ", ".join(subject_names)
                parts.append(f"Topics: {subjects_str}")

        # Add description if available
        if "description" in book_data and book_data["description"]:
            desc = book_data["description"]
            if len(desc) > 200:
                desc = desc[:200] + "..."
            parts.append(f"Description: {desc}")

        description = ". ".join(parts)

        if not description.strip():
            description = f"Book: {book_data.get('isbn', 'Unknown')}"

        return description

    def download_image(self, url, timeout=10, max_retries=3):
        """
        Download image from URL.

        Args:
            url: Image URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts

        Returns:
            PIL Image or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    return image
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Wait before retry
                continue
        return None

    @torch.no_grad()
    def extract_text_features(self, texts, batch_size=32):
        """
        Extract CLIP features from text descriptions.

        Args:
            texts: List of text descriptions
            batch_size: Batch size for processing

        Returns:
            numpy array of features (N x feature_dim)
        """
        all_features = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
            batch_texts = texts[i:i + batch_size]

            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_features.append(text_features.cpu().numpy())

        return np.vstack(all_features)

    @torch.no_grad()
    def extract_image_features(self, images, batch_size=32):
        """
        Extract CLIP features from images.

        Args:
            images: List of PIL Images (or None for missing images)
            batch_size: Batch size for processing

        Returns:
            numpy array of features (N x feature_dim)
        """
        all_features = []
        num_images = len(images)

        for i in tqdm(range(0, num_images, batch_size), desc="Extracting image features"):
            batch_images = images[i:i + batch_size]

            # Filter out None images
            valid_images = [img for img in batch_images if img is not None]

            if len(valid_images) == 0:
                # If no valid images in batch, create zero features
                feature_dim = 512  # CLIP feature dimension
                batch_features = np.zeros((len(batch_images), feature_dim), dtype=np.float32)
            else:
                inputs = self.processor(
                    images=valid_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Map back to original batch (handle None images)
                batch_features = []
                valid_idx = 0
                for img in batch_images:
                    if img is not None:
                        batch_features.append(image_features[valid_idx].cpu().numpy())
                        valid_idx += 1
                    else:
                        batch_features.append(np.zeros(512, dtype=np.float32))
                batch_features = np.array(batch_features)

            all_features.append(batch_features)

        return np.vstack(all_features)

    def fuse_features(self, text_features, image_features, fusion_method="average",
                     text_weight=0.5):
        """
        Fuse text and image features.

        Args:
            text_features: Text feature array (N x D)
            image_features: Image feature array (N x D)
            fusion_method: "average", "weighted", "concat", "text_only", "image_only"
            text_weight: Weight for text features (for "weighted" method)

        Returns:
            Fused features
        """
        if fusion_method == "average":
            # Simple average
            return (text_features + image_features) / 2

        elif fusion_method == "weighted":
            # Weighted average
            return text_weight * text_features + (1 - text_weight) * image_features

        elif fusion_method == "concat":
            # Concatenate features
            return np.concatenate([text_features, image_features], axis=1)

        elif fusion_method == "text_only":
            return text_features

        elif fusion_method == "image_only":
            return image_features

        elif fusion_method == "adaptive":
            # Use image when available, text as fallback
            fused = text_features.copy()
            # If image feature is not all zeros, use average
            has_image = np.any(image_features != 0, axis=1)
            fused[has_image] = (text_features[has_image] + image_features[has_image]) / 2
            return fused

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")


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
    parser = argparse.ArgumentParser(
        description="Extract multimodal CLIP features (text + images)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="adaptive",
        choices=["average", "weighted", "concat", "text_only", "image_only", "adaptive"],
        help="Feature fusion method"
    )
    parser.add_argument(
        "--text_weight",
        type=float,
        default=0.5,
        help="Weight for text features (for weighted fusion)"
    )
    parser.add_argument(
        "--download_images",
        action="store_true",
        help="Download and use cover images"
    )
    parser.add_argument(
        "--image_delay",
        type=float,
        default=0.1,
        help="Delay between image downloads (seconds)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="clip_features_multimodal.npy",
        help="Output filename for features"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )

    args = parser.parse_args()

    # Set GPU device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Paths
    data_path = Path(args.data_path)
    metadata_path = data_path / "book_metadata.json"
    item_list_path = data_path / "item_list.txt"
    output_path = data_path / args.output_name
    descriptions_path = data_path / "book_descriptions.json"

    # Load data
    metadata = load_book_metadata(metadata_path)
    id_to_isbn = load_item_mapping(item_list_path)

    # Create extractor
    extractor = MultimodalBookCLIPExtractor(model_name=args.model_name, device=device)

    # Prepare data
    num_items = len(id_to_isbn)
    descriptions = []
    description_dict = {}
    cover_urls = []

    print(f"\n{'='*70}")
    print(f"Preparing data for {num_items} items...")
    print(f"{'='*70}")

    for item_id in range(num_items):
        isbn = id_to_isbn.get(item_id)

        if isbn and isbn in metadata:
            book_data = metadata[isbn]
            description = extractor.create_text_description(book_data)
            cover_url = book_data.get('cover_url', None)
        else:
            description = f"Book item {item_id}"
            cover_url = None

        descriptions.append(description)
        description_dict[item_id] = description
        cover_urls.append(cover_url)

    # Save descriptions
    print(f"Saving descriptions to {descriptions_path}")
    with open(descriptions_path, 'w', encoding='utf-8') as f:
        json.dump(description_dict, f, indent=2, ensure_ascii=False)

    # Extract text features
    print(f"\n{'='*70}")
    print(f"Extracting text features...")
    print(f"{'='*70}")
    text_features = extractor.extract_text_features(descriptions, batch_size=args.batch_size)

    # Extract image features if requested
    if args.download_images:
        print(f"\n{'='*70}")
        print(f"Downloading and extracting image features...")
        print(f"{'='*70}")

        images = []
        successful_downloads = 0

        for i, url in enumerate(tqdm(cover_urls, desc="Downloading covers")):
            if url:
                image = extractor.download_image(url)
                if image is not None:
                    successful_downloads += 1
                images.append(image)
                time.sleep(args.image_delay)  # Rate limiting
            else:
                images.append(None)

            # Progress update every 1000 items
            if (i + 1) % 1000 == 0:
                print(f"  Downloaded: {successful_downloads}/{i+1} ({successful_downloads/(i+1)*100:.1f}%)")

        print(f"\nSuccessfully downloaded: {successful_downloads}/{num_items} ({successful_downloads/num_items*100:.1f}%)")

        image_features = extractor.extract_image_features(images, batch_size=args.batch_size)

        # Fuse features
        print(f"\n{'='*70}")
        print(f"Fusing features using '{args.fusion_method}' method...")
        print(f"{'='*70}")
        features = extractor.fuse_features(
            text_features,
            image_features,
            fusion_method=args.fusion_method,
            text_weight=args.text_weight
        )
    else:
        print(f"\nUsing text features only (--download_images not specified)")
        features = text_features

    print(f"\nFinal features shape: {features.shape}")
    print(f"Feature dimension: {features.shape[1]}")

    # Save features
    print(f"\nSaving features to {output_path}")
    np.save(output_path, features)

    # Save metadata
    metadata_output = {
        "num_items": num_items,
        "feature_dim": features.shape[1],
        "model_name": args.model_name,
        "fusion_method": args.fusion_method if args.download_images else "text_only",
        "text_weight": args.text_weight if args.fusion_method == "weighted" else None,
        "used_images": args.download_images,
        "description": f"CLIP {'multimodal (text+image)' if args.download_images else 'text-only'} features"
    }

    if args.download_images:
        metadata_output["images_downloaded"] = successful_downloads
        metadata_output["image_coverage"] = f"{successful_downloads/num_items*100:.1f}%"

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
    print(f"Modality: {'Text + Image' if args.download_images else 'Text only'}")
    if args.download_images:
        print(f"Images used: {successful_downloads}/{num_items} ({successful_downloads/num_items*100:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
