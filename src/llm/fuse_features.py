"""
Fuse CLIP and GPT features for hybrid semantic graph.
"""

import argparse
from pathlib import Path
import numpy as np
import json
from sklearn.preprocessing import normalize


def load_features(features_path):
    """Load feature numpy array."""
    print(f"Loading features from {features_path}")
    features = np.load(features_path)
    print(f"  Shape: {features.shape}")
    return features


def fuse_concat(clip_features, gpt_features):
    """Concatenate CLIP and GPT features."""
    print("Fusion method: concatenation")
    fused = np.concatenate([clip_features, gpt_features], axis=1)
    return fused


def fuse_weighted_avg(clip_features, gpt_features, clip_weight=0.5):
    """Weighted average of normalized features."""
    print(f"Fusion method: weighted average (CLIP: {clip_weight}, GPT: {1-clip_weight})")

    # Normalize
    clip_norm = normalize(clip_features, norm='l2', axis=1)
    gpt_norm = normalize(gpt_features, norm='l2', axis=1)

    # Weighted sum
    fused = clip_weight * clip_norm + (1 - clip_weight) * gpt_norm

    # Re-normalize
    fused = normalize(fused, norm='l2', axis=1)

    return fused


def fuse_pca_concat(clip_features, gpt_features, target_dim=512):
    """Concatenate and reduce dimensionality with PCA."""
    print(f"Fusion method: PCA concatenation (target_dim: {target_dim})")

    from sklearn.decomposition import PCA

    # Concatenate
    concat = np.concatenate([clip_features, gpt_features], axis=1)
    print(f"  Concatenated shape: {concat.shape}")

    # PCA
    pca = PCA(n_components=target_dim)
    fused = pca.fit_transform(concat)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return fused


def main():
    parser = argparse.ArgumentParser(
        description="Fuse CLIP and GPT features"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--clip_features",
        type=str,
        default="clip_features.npy",
        help="CLIP features file"
    )
    parser.add_argument(
        "--gpt_features",
        type=str,
        default="gpt_embeddings.npy",
        help="GPT embeddings file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="concat",
        choices=["concat", "weighted_avg", "pca_concat"],
        help="Fusion method"
    )
    parser.add_argument(
        "--clip_weight",
        type=float,
        default=0.5,
        help="Weight for CLIP features (for weighted_avg method)"
    )
    parser.add_argument(
        "--target_dim",
        type=int,
        default=512,
        help="Target dimension (for pca_concat method)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hybrid_features.npy",
        help="Output filename"
    )

    args = parser.parse_args()

    # Paths
    data_path = Path(args.data_path)
    clip_path = data_path / args.clip_features
    gpt_path = data_path / args.gpt_features

    # Create fuse_embedding directory
    fuse_dir = data_path / "fuse_embedding"
    fuse_dir.mkdir(exist_ok=True)
    output_path = fuse_dir / args.output

    # Load features
    clip_features = load_features(clip_path)
    gpt_features = load_features(gpt_path)

    print(f"\nFusing features...")
    print(f"CLIP shape: {clip_features.shape}")
    print(f"GPT shape: {gpt_features.shape}")

    # Fuse
    if args.method == "concat":
        fused = fuse_concat(clip_features, gpt_features)
    elif args.method == "weighted_avg":
        fused = fuse_weighted_avg(clip_features, gpt_features, args.clip_weight)
    elif args.method == "pca_concat":
        fused = fuse_pca_concat(clip_features, gpt_features, args.target_dim)

    print(f"\nFused features shape: {fused.shape}")

    # Save
    print(f"Saving to {output_path}")
    np.save(output_path, fused)

    # Metadata
    metadata = {
        "num_items": fused.shape[0],
        "feature_dim": fused.shape[1],
        "fusion_method": args.method,
        "clip_features": args.clip_features,
        "gpt_features": args.gpt_features,
        "clip_dim": clip_features.shape[1],
        "gpt_dim": gpt_features.shape[1],
        "description": f"Hybrid features: CLIP + GPT ({args.method})"
    }

    if args.method == "weighted_avg":
        metadata["clip_weight"] = args.clip_weight
        metadata["gpt_weight"] = 1 - args.clip_weight
    elif args.method == "pca_concat":
        metadata["target_dim"] = args.target_dim

    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*70)
    print("✅ Feature fusion completed!")
    print("="*70)
    print(f"Output shape: {fused.shape}")
    print(f"Fusion method: {args.method}")
    print(f"Output: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
