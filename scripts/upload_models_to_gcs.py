#!/usr/bin/env python3
"""
Upload ML models to Google Cloud Storage.

Prerequisites:
1. Install google-cloud-storage: pip install google-cloud-storage
2. Authenticate: gcloud auth application-default login
3. Create bucket: gsutil mb gs://obelisk-stamps-models

Usage:
    python scripts/upload_models_to_gcs.py
    python scripts/upload_models_to_gcs.py --bucket my-custom-bucket
"""

import argparse
import os
from pathlib import Path

from google.cloud import storage


# Files to upload
MODEL_FILES = [
    ("models/stamp_embed.keras", "stamp_embed.keras"),
    ("indexes/ref_embeddings.npy", "ref_embeddings.npy"),
    ("indexes/ref_rows.pkl", "ref_rows.pkl"),
    ("ml/stamp_detector/weights/stamp_detector.pt", "stamp_detector.pt"),
]


def upload_file(bucket, local_path: Path, blob_name: str):
    """Upload a file to GCS."""
    blob = bucket.blob(blob_name)

    # Get file size
    file_size = local_path.stat().st_size / (1024 * 1024)  # MB

    print(f"Uploading {local_path.name} ({file_size:.1f} MB) -> gs://{bucket.name}/{blob_name}")
    blob.upload_from_filename(str(local_path))
    print(f"  ✓ Uploaded successfully")


def main():
    parser = argparse.ArgumentParser(description="Upload ML models to GCS")
    parser.add_argument("--bucket", type=str, default="obelisk-stamps-models",
                        help="GCS bucket name")
    parser.add_argument("--project", type=str, default="obelisk-stamps",
                        help="GCP project ID")
    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent

    # Initialize GCS client
    print(f"Connecting to GCS bucket: {args.bucket}")
    client = storage.Client(project=args.project)

    # Create bucket if it doesn't exist
    try:
        bucket = client.get_bucket(args.bucket)
        print(f"Using existing bucket: {args.bucket}")
    except Exception:
        print(f"Creating bucket: {args.bucket}")
        bucket = client.create_bucket(args.bucket, location="europe-west2")

    print("-" * 60)

    # Upload each file
    for local_rel_path, blob_name in MODEL_FILES:
        local_path = project_root / local_rel_path

        if not local_path.exists():
            print(f"WARNING: {local_path} not found, skipping")
            continue

        upload_file(bucket, local_path, blob_name)

    print("-" * 60)
    print("All models uploaded!")
    print(f"\nBucket URL: https://storage.googleapis.com/{args.bucket}/")
    print("\nTo verify:")
    print(f"  gsutil ls gs://{args.bucket}/")


if __name__ == "__main__":
    main()
