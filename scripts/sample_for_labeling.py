#!/usr/bin/env python3
"""
Sample images from verification folders for additional labeling.

Copies ORIGINAL images (not annotated ones) to the training folder
so they can be labeled with LabelMe.
"""

import os
import random
import re
import shutil
import sys
from pathlib import Path

# Folders to sample from
VERIFICATION_DIR = Path("D:/CodingFolder/obelisk-stamps/verification_output")
TRAINING_IMAGES_DIR = Path("D:/CodingFolder/obelisk-stamps/ml/stamp_detector/training_data/images")

# Sample sizes
SAMPLE_SIZE = 150


def extract_original_path(annotated_filename: str) -> str:
    """
    Extract original image path from annotated filename.

    Annotated files are named: {id}_{original_filename}
    We need to find the original in the data folder.
    """
    # Remove the ID prefix (e.g., "12345_1864_netherlands_365.jpg" -> "1864_netherlands_365.jpg")
    match = re.match(r'^\d+_(.+)$', annotated_filename)
    if match:
        return match.group(1)
    return annotated_filename


def find_original_image(filename: str, data_dir: Path = Path("D:/CodingFolder/obelisk-stamps/data")) -> Path:
    """Find the original image in the data directory."""
    # Search recursively in data folder
    for img_path in data_dir.rglob(filename):
        return img_path
    return None


def sample_and_copy(source_dir: Path, sample_size: int, dest_dir: Path) -> int:
    """Sample images from source and copy originals to destination."""
    if not source_dir.exists():
        print(f"  Source folder not found: {source_dir}")
        return 0

    # Get all annotated images
    annotated_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))

    if not annotated_files:
        print(f"  No images found in {source_dir}")
        return 0

    # Sample randomly
    sample_size = min(sample_size, len(annotated_files))
    sampled = random.sample(annotated_files, sample_size)

    copied = 0
    for annotated_path in sampled:
        original_filename = extract_original_path(annotated_path.name)
        original_path = find_original_image(original_filename)

        if original_path and original_path.exists():
            dest_path = dest_dir / original_filename
            if not dest_path.exists():
                shutil.copy2(original_path, dest_path)
                copied += 1
        else:
            print(f"  WARNING: Original not found for {annotated_path.name}")

    return copied


def main():
    print("Sampling images for additional labeling...")
    print(f"Sample size per folder: {SAMPLE_SIZE}")
    print("-" * 60)

    # Ensure training images folder exists
    TRAINING_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing = len(list(TRAINING_IMAGES_DIR.glob("*.jpg"))) + len(list(TRAINING_IMAGES_DIR.glob("*.png")))
    print(f"Existing images in training folder: {existing}")
    print()

    total_copied = 0

    # Sample from each folder
    folders = ["no_detection", "suspicious_single", "multi_stamp"]

    for folder in folders:
        source_dir = VERIFICATION_DIR / folder
        print(f"Sampling from {folder}/...")
        copied = sample_and_copy(source_dir, SAMPLE_SIZE, TRAINING_IMAGES_DIR)
        print(f"  Copied {copied} new images")
        total_copied += copied

    print("-" * 60)
    print(f"Total new images copied: {total_copied}")

    # Count final total
    final_count = len(list(TRAINING_IMAGES_DIR.glob("*.jpg"))) + len(list(TRAINING_IMAGES_DIR.glob("*.png")))
    print(f"Total images in training folder: {final_count}")
    print()
    print(f"Training images folder: {TRAINING_IMAGES_DIR}")
    print()
    print("Next step: Run LabelMe to label the new images:")
    print("  python ml/stamp_detector/prepare_data.py --launch-labelme")


if __name__ == "__main__":
    main()
