#!/usr/bin/env python3
"""
Data preparation script for stamp detector training.

This script helps you:
1. Copy sample images from your scraped data for labeling
2. Launch the Label Studio annotation tool
3. Convert annotations to YOLO format

Usage:
    python ml/stamp_detector/prepare_data.py --sample 200
    python ml/stamp_detector/prepare_data.py --launch-labelme
    python ml/stamp_detector/prepare_data.py --convert-labelme
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODULE_DIR = Path(__file__).parent
TRAINING_DATA_DIR = MODULE_DIR / "training_data"
IMAGES_DIR = TRAINING_DATA_DIR / "images"
LABELS_DIR = TRAINING_DATA_DIR / "labels"

# Source directories for scraped images
SCRAPED_IMAGES_DIRS = [
    PROJECT_ROOT / "data" / "postbeeld" / "images",
    PROJECT_ROOT / "data" / "stamps" / "images",
    # Add more scraper image directories here
]


def sample_images(count: int = 200, include_multi_stamp: bool = True):
    """
    Sample images from scraped data for labeling.

    Tries to get a good mix of:
    - Single stamp images
    - Multi-stamp images (wider aspect ratio)

    Args:
        count: Number of images to sample.
        include_multi_stamp: Prioritize images that look like multi-stamps.
    """
    from PIL import Image

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all available images
    all_images = []
    for src_dir in SCRAPED_IMAGES_DIRS:
        if src_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                all_images.extend(src_dir.glob(ext))

    if not all_images:
        print("ERROR: No source images found.")
        print("Make sure you have scraped images in:")
        for d in SCRAPED_IMAGES_DIRS:
            print(f"  - {d}")
        return

    print(f"Found {len(all_images)} source images")

    # Categorize by aspect ratio (potential multi-stamp indicator)
    wide_images = []  # Likely 2-up or 3-up horizontal
    tall_images = []  # Likely 2-up vertical
    normal_images = []  # Likely single stamp

    print("Analyzing images...")
    for img_path in all_images:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                aspect = w / h if h > 0 else 1

                # Skip already-split images (have _stamp or _p in name)
                if '_stamp' in img_path.stem or '_p1' in img_path.stem or '_p2' in img_path.stem:
                    continue

                if aspect > 1.6:  # Wide - likely 2-up horizontal
                    wide_images.append(img_path)
                elif aspect < 0.6:  # Tall - likely 2-up vertical
                    tall_images.append(img_path)
                else:
                    normal_images.append(img_path)
        except Exception:
            continue

    print(f"  Wide (potential multi-stamp): {len(wide_images)}")
    print(f"  Tall (potential multi-stamp): {len(tall_images)}")
    print(f"  Normal (likely single): {len(normal_images)}")

    # Sample with preference for multi-stamps
    selected = []

    if include_multi_stamp:
        # Take more wide/tall images as they're more interesting for training
        n_wide = min(len(wide_images), count // 3)
        n_tall = min(len(tall_images), count // 6)
        n_normal = count - n_wide - n_tall

        selected.extend(random.sample(wide_images, n_wide) if wide_images else [])
        selected.extend(random.sample(tall_images, n_tall) if tall_images else [])
        selected.extend(random.sample(normal_images, min(n_normal, len(normal_images))))
    else:
        selected = random.sample(all_images, min(count, len(all_images)))

    # Copy selected images
    print(f"\nCopying {len(selected)} images to {IMAGES_DIR}")
    for src_path in selected:
        dst_path = IMAGES_DIR / src_path.name

        # Handle name conflicts
        counter = 1
        while dst_path.exists():
            dst_path = IMAGES_DIR / f"{src_path.stem}_{counter}{src_path.suffix}"
            counter += 1

        shutil.copy(src_path, dst_path)

    print(f"Done! Images ready for labeling in: {IMAGES_DIR}")
    print("\nNext steps:")
    print("1. Install LabelMe: pip install labelme")
    print("2. Run: python ml/stamp_detector/prepare_data.py --launch-labelme")
    print("3. Label all stamps with bounding boxes (class: 'stamp')")
    print("4. Run: python ml/stamp_detector/prepare_data.py --convert-labelme")


def launch_labelme(only_unlabeled: bool = False):
    """Launch LabelMe annotation tool."""
    try:
        import subprocess

        target_dir = IMAGES_DIR

        if only_unlabeled:
            # Create temp folder with symlinks to unlabeled images only
            unlabeled_dir = TRAINING_DATA_DIR / "unlabeled"
            unlabeled_dir.mkdir(parents=True, exist_ok=True)

            # Clear old symlinks
            for f in unlabeled_dir.glob("*"):
                f.unlink()

            # Find images without .json labels
            all_images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
            unlabeled = []
            for img_path in all_images:
                json_path = img_path.with_suffix('.json')
                if not json_path.exists():
                    unlabeled.append(img_path)

            if not unlabeled:
                print("All images are already labeled!")
                return

            print(f"Found {len(unlabeled)} unlabeled images (out of {len(all_images)} total)")

            # Copy unlabeled images to temp folder
            for img_path in unlabeled:
                dest = unlabeled_dir / img_path.name
                shutil.copy2(img_path, dest)

            target_dir = unlabeled_dir
            print(f"\nLaunching LabelMe on unlabeled images only: {target_dir}")
            print("NOTE: After labeling, .json files will be saved in the unlabeled folder.")
            print("      Run --move-labels to move them back to the main images folder.")
        else:
            print(f"Launching LabelMe on: {IMAGES_DIR}")

        print("\nInstructions:")
        print("1. Open an image")
        print("2. Click 'Create Rectangle' or press 'R'")
        print("3. Draw a box around each stamp")
        print("4. Label it as 'stamp'")
        print("5. Save (Ctrl+S) - saves .json alongside image")
        print("6. Next image (D key)")
        print("\nPress Ctrl+C to stop when done.")

        subprocess.run(["labelme", str(target_dir), "--labels", "stamp"])
    except FileNotFoundError:
        print("ERROR: LabelMe not installed.")
        print("Run: pip install labelme")


def move_labels_from_unlabeled():
    """Move .json label files from unlabeled folder back to images folder."""
    unlabeled_dir = TRAINING_DATA_DIR / "unlabeled"

    if not unlabeled_dir.exists():
        print("No unlabeled folder found.")
        return

    json_files = list(unlabeled_dir.glob("*.json"))

    if not json_files:
        print("No label files found in unlabeled folder.")
        return

    moved = 0
    for json_path in json_files:
        dest = IMAGES_DIR / json_path.name
        shutil.move(str(json_path), str(dest))
        moved += 1

    print(f"Moved {moved} label files to {IMAGES_DIR}")

    # Clean up - remove copied images from unlabeled folder
    for f in unlabeled_dir.glob("*"):
        f.unlink()

    print("Cleaned up unlabeled folder.")


def convert_labelme_to_yolo():
    """Convert LabelMe JSON annotations to YOLO format."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(IMAGES_DIR.glob("*.json"))

    if not json_files:
        print(f"No LabelMe annotations found in {IMAGES_DIR}")
        print("Label some images first using LabelMe.")
        return

    print(f"Found {len(json_files)} LabelMe annotations")

    converted = 0
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            img_width = data['imageWidth']
            img_height = data['imageHeight']

            yolo_lines = []
            for shape in data['shapes']:
                if shape['shape_type'] != 'rectangle':
                    continue

                label = shape['label'].lower()
                if label != 'stamp':
                    continue

                # LabelMe format: [[x1,y1], [x2,y2]]
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]

                # Ensure correct order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Convert to YOLO format (normalized center x, y, width, height)
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # Class 0 = stamp
                yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

            if yolo_lines:
                # Save YOLO label file
                label_path = LABELS_DIR / f"{json_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                converted += 1

        except Exception as e:
            print(f"Error converting {json_path.name}: {e}")

    print(f"Converted {converted} annotations to YOLO format")
    print(f"Labels saved to: {LABELS_DIR}")

    if converted > 0:
        print("\nReady to train! Run:")
        print("  python ml/stamp_detector/train.py")


def show_stats():
    """Show current training data statistics."""
    images = list(IMAGES_DIR.glob("*")) if IMAGES_DIR.exists() else []
    images = [f for f in images if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]

    labels = list(LABELS_DIR.glob("*.txt")) if LABELS_DIR.exists() else []
    json_annotations = list(IMAGES_DIR.glob("*.json")) if IMAGES_DIR.exists() else []

    print("\n" + "=" * 60)
    print("TRAINING DATA STATUS")
    print("=" * 60)
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Labels directory: {LABELS_DIR}")
    print()
    print(f"Images collected: {len(images)}")
    print(f"LabelMe annotations (.json): {len(json_annotations)}")
    print(f"YOLO labels (.txt): {len(labels)}")

    if labels:
        # Count total stamps labeled
        total_stamps = 0
        for label_path in labels:
            with open(label_path) as f:
                total_stamps += len(f.readlines())
        print(f"Total stamps labeled: {total_stamps}")
        print(f"Average stamps per image: {total_stamps / len(labels):.1f}")

    print("=" * 60)

    if len(images) == 0:
        print("\nNext step: Sample images for labeling")
        print("  python ml/stamp_detector/prepare_data.py --sample 200")
    elif len(json_annotations) == 0:
        print("\nNext step: Label images with LabelMe")
        print("  python ml/stamp_detector/prepare_data.py --launch-labelme")
    elif len(labels) == 0:
        print("\nNext step: Convert annotations to YOLO format")
        print("  python ml/stamp_detector/prepare_data.py --convert-labelme")
    elif len(labels) < 50:
        print(f"\nRecommendation: Label more images ({len(labels)}/50+ recommended)")
    else:
        print("\nReady to train!")
        print("  python ml/stamp_detector/train.py")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data for stamp detector")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="Sample N images from scraped data for labeling")
    parser.add_argument("--launch-labelme", action="store_true",
                        help="Launch LabelMe annotation tool")
    parser.add_argument("--only-unlabeled", action="store_true",
                        help="Only show unlabeled images in LabelMe")
    parser.add_argument("--move-labels", action="store_true",
                        help="Move label files from unlabeled folder back to images folder")
    parser.add_argument("--convert-labelme", action="store_true",
                        help="Convert LabelMe annotations to YOLO format")
    parser.add_argument("--stats", action="store_true",
                        help="Show training data statistics")

    args = parser.parse_args()

    if args.sample:
        sample_images(args.sample)
    elif args.launch_labelme:
        launch_labelme(only_unlabeled=args.only_unlabeled)
    elif args.move_labels:
        move_labels_from_unlabeled()
    elif args.convert_labelme:
        convert_labelme_to_yolo()
    else:
        # Default: show stats
        show_stats()


if __name__ == "__main__":
    main()
