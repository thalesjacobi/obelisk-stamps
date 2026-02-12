#!/usr/bin/env python3
"""
Test script for the trained stamp detector.

Usage:
    # Test on a single image
    python ml/stamp_detector/test_detector.py path/to/image.jpg

    # Test on multiple images
    python ml/stamp_detector/test_detector.py image1.jpg image2.jpg image3.jpg

    # Test on a folder of images (first 20)
    python ml/stamp_detector/test_detector.py --folder data/postbeeld/images --limit 20

    # Test and save annotated images with bounding boxes drawn
    python ml/stamp_detector/test_detector.py --folder data/postbeeld/images --visualize

Run from project root: D:\CodingFolder\obelisk-stamps
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.stamp_detector import StampDetector


def test_single_image(detector, image_path, visualize=False, output_dir=None):
    """Test detection on a single image."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"  ERROR: File not found: {image_path}")
        return

    detections = detector.detect(str(image_path))
    count = len(detections)

    status = "MULTI-STAMP" if count > 1 else "SINGLE" if count == 1 else "NO DETECTION"
    print(f"  [{status}] {image_path.name}: {count} stamp(s) detected")

    for i, det in enumerate(detections, 1):
        bbox = det['bbox']
        conf = det['confidence']
        print(f"    Stamp {i}: bbox={bbox}, confidence={conf:.3f}")

    # Optionally save annotated image with bounding boxes
    if visualize and detections:
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            for i, det in enumerate(detections, 1):
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                # Draw label
                label = f"stamp {i} ({conf:.2f})"
                draw.text((x1, max(0, y1 - 12)), label, fill="red")

            # Save annotated image
            out_dir = Path(output_dir) if output_dir else image_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{image_path.stem}_detected{image_path.suffix}"
            img.save(str(out_path))
            print(f"    -> Saved annotated: {out_path}")
        except Exception as e:
            print(f"    -> Visualization error: {e}")

    return count


def test_folder(detector, folder_path, limit=20, visualize=False, output_dir=None):
    """Test detection on a folder of images."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return

    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in sorted(folder.iterdir())
              if f.suffix.lower() in extensions
              and '_detected' not in f.stem  # Skip annotated images
              and '_stamp' not in f.stem     # Skip already-split images
              and '_p1' not in f.stem and '_p2' not in f.stem and '_p3' not in f.stem]

    if not images:
        print("No images found in folder.")
        return

    total = min(limit, len(images)) if limit else len(images)
    print(f"\nTesting on {total} images from {folder}...")
    print("-" * 60)

    single_count = 0
    multi_count = 0
    no_detect = 0

    for img_path in images[:total]:
        count = test_single_image(detector, img_path, visualize, output_dir)
        if count is None:
            continue
        if count > 1:
            multi_count += 1
        elif count == 1:
            single_count += 1
        else:
            no_detect += 1

    print("-" * 60)
    print(f"Results: {total} images tested")
    print(f"  Single stamp:  {single_count}")
    print(f"  Multi-stamp:   {multi_count}")
    print(f"  No detection:  {no_detect}")


def main():
    parser = argparse.ArgumentParser(description="Test the stamp detector on images")
    parser.add_argument("images", nargs="*", help="Image file(s) to test")
    parser.add_argument("--folder", type=str, help="Folder of images to test")
    parser.add_argument("--limit", type=int, default=20, help="Max images to test from folder")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated images with bounding boxes drawn")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for annotated images (default: same as input)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold (0.0-1.0)")

    args = parser.parse_args()

    # Initialize detector
    print("Loading stamp detector model...")
    detector = StampDetector(confidence_threshold=args.confidence)

    if not detector.is_ready:
        print("ERROR: Model not ready. Make sure stamp_detector.pt exists in weights/")
        sys.exit(1)

    print(f"Model loaded! Confidence threshold: {args.confidence}")

    if args.folder:
        test_folder(detector, args.folder, args.limit, args.visualize, args.output_dir)
    elif args.images:
        print(f"\nTesting {len(args.images)} image(s)...")
        print("-" * 60)
        for img in args.images:
            test_single_image(detector, img, args.visualize, args.output_dir)
    else:
        # Default: test on scraped postbeeld images
        default_folder = PROJECT_ROOT / "data" / "postbeeld" / "images"
        if default_folder.exists():
            test_folder(detector, default_folder, args.limit, args.visualize, args.output_dir)
        else:
            print("No images specified. Usage:")
            print("  python ml/stamp_detector/test_detector.py image.jpg")
            print("  python ml/stamp_detector/test_detector.py --folder data/postbeeld/images")


if __name__ == "__main__":
    main()
