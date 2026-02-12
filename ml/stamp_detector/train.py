#!/usr/bin/env python3
"""
Training script for the Stamp Detector YOLO model.

Usage:
    python ml/stamp_detector/train.py

Requirements:
    - Labeled images in ml/stamp_detector/training_data/images/
    - YOLO format labels in ml/stamp_detector/training_data/labels/
    - ultralytics package installed

Label format (YOLO):
    Each image needs a corresponding .txt file with the same name.
    Each line: class_id center_x center_y width height (normalized 0-1)
    Example: 0 0.5 0.5 0.3 0.4
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODULE_DIR = Path(__file__).parent
TRAINING_DATA_DIR = MODULE_DIR / "training_data"
WEIGHTS_DIR = MODULE_DIR / "weights"
CONFIG_PATH = MODULE_DIR / "config.yaml"


def check_training_data():
    """Check if training data is available and properly formatted."""
    images_dir = TRAINING_DATA_DIR / "images"
    labels_dir = TRAINING_DATA_DIR / "labels"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return False, 0, 0

    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return False, 0, 0

    # Count images and labels
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    labels = [f for f in labels_dir.iterdir() if f.suffix == '.txt']

    # Check for matching pairs
    image_stems = {f.stem for f in images}
    label_stems = {f.stem for f in labels}

    matched = image_stems & label_stems
    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems

    print(f"\n{'='*60}")
    print("TRAINING DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Images found: {len(images)}")
    print(f"Labels found: {len(labels)}")
    print(f"Matched pairs: {len(matched)}")

    if images_without_labels:
        print(f"\nWARNING: {len(images_without_labels)} images without labels")
        if len(images_without_labels) <= 5:
            for stem in images_without_labels:
                print(f"  - {stem}")

    if labels_without_images:
        print(f"\nWARNING: {len(labels_without_images)} labels without images")

    print(f"{'='*60}\n")

    return len(matched) > 0, len(matched), len(images)


def train(
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    model_size: str = "n",  # n=nano, s=small, m=medium, l=large, x=xlarge
    patience: int = 20,
    resume: bool = False,
):
    """
    Train the YOLO stamp detector model.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        image_size: Input image size (will be resized).
        model_size: YOLO model size (n/s/m/l/x).
        patience: Early stopping patience.
        resume: Resume from last checkpoint.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("Run: pip install ultralytics")
        return False

    # Check training data
    has_data, matched, total = check_training_data()
    if not has_data:
        print("ERROR: No valid training data found.")
        print("\nTo prepare training data:")
        print("1. Copy images to: ml/stamp_detector/training_data/images/")
        print("2. Create YOLO labels in: ml/stamp_detector/training_data/labels/")
        print("3. Or run: python ml/stamp_detector/prepare_data.py")
        return False

    if matched < 10:
        print(f"WARNING: Only {matched} labeled images found.")
        print("Recommend at least 50-100 images for basic training.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            return False

    # Create weights directory
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Select base model
    base_model = f"yolo11{model_size}.pt"  # YOLOv11 nano by default
    print(f"Using base model: {base_model}")

    # Initialize model
    if resume and (WEIGHTS_DIR / "last.pt").exists():
        print("Resuming from last checkpoint...")
        model = YOLO(str(WEIGHTS_DIR / "last.pt"))
    else:
        model = YOLO(base_model)

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}")
    print(f"  Patience: {patience}")

    results = model.train(
        data=str(CONFIG_PATH),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        patience=patience,
        project=str(WEIGHTS_DIR.parent),
        name="training_run",
        exist_ok=True,
        verbose=True,
        # Augmentation settings good for stamps
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.4,    # Saturation augmentation
        hsv_v=0.3,    # Value augmentation
        degrees=5,    # Rotation (stamps are usually upright)
        translate=0.1,
        scale=0.3,
        flipud=0.0,   # No vertical flip (stamps have orientation)
        fliplr=0.5,   # Horizontal flip OK
        mosaic=0.5,   # Mosaic augmentation
    )

    # Copy best weights to standard location
    best_weights = WEIGHTS_DIR.parent / "training_run" / "weights" / "best.pt"
    if best_weights.exists():
        import shutil
        dest = WEIGHTS_DIR / "stamp_detector.pt"
        shutil.copy(best_weights, dest)
        print(f"\nBest weights saved to: {dest}")

    print("\nTraining complete!")
    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO stamp detector")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--size", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.size,
        model_size=args.model,
        patience=args.patience,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
