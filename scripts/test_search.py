#!/usr/bin/env python3
"""
Test stamp search by uploading an image and finding similar stamps.

Usage:
    python scripts/test_search.py path/to/stamp_image.jpg
    python scripts/test_search.py path/to/stamp_image.jpg --top 10
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import custom layers before loading model
from ml.model_utils import L2Normalize
from ml.inference import predict_stamp_value_from_image


def main():
    parser = argparse.ArgumentParser(description="Test stamp similarity search")
    parser.add_argument("image", type=str, help="Path to query stamp image")
    parser.add_argument("--top", type=int, default=5, help="Number of results to return")
    parser.add_argument("--model", type=str, default="models/stamp_embed.keras", help="Path to embedding model")
    parser.add_argument("--index-dir", type=str, default="indexes", help="Path to index directory")
    args = parser.parse_args()

    # Check image exists
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    # Load model (with custom objects)
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model, custom_objects={'L2Normalize': L2Normalize})

    # Load index
    emb_path = os.path.join(args.index_dir, "ref_embeddings.npy")
    rows_path = os.path.join(args.index_dir, "ref_rows.pkl")

    print(f"Loading index from {args.index_dir}...")
    ref_embeddings = np.load(emb_path)
    with open(rows_path, "rb") as f:
        ref_rows = pickle.load(f)

    print(f"Index loaded: {len(ref_rows)} stamps")
    print("-" * 60)

    # Search
    print(f"Searching for similar stamps to: {args.image}")
    print("-" * 60)

    results = predict_stamp_value_from_image(
        model=model,
        ref_embeddings=ref_embeddings,
        ref_rows=ref_rows,
        query_image_path=args.image,
        top_k=args.top,
    )

    # Display results
    print(f"\nTop {len(results)} matches:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. Similarity: {r['similarity']:.3f}")
        print(f"   Title: {r['title']}")
        print(f"   Country: {r['country']}, Year: {r['year']}")
        print(f"   Price: {r['price_original']['value']} {r['price_original']['currency']}")
        print(f"   Condition: {r['condition_text']}")
        print(f"   Image: {r['image_path_ref']}")
        print()


if __name__ == "__main__":
    main()
