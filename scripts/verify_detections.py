#!/usr/bin/env python3
"""
Visual verification script for stamp detection.

Scans images and saves annotated copies with bounding boxes for:
1. Wide images (aspect ratio > 1.5) marked as single stamp - potential missed multi-stamps
2. Images with 0 detections - YOLO found nothing
3. All multi-stamp detections - to verify before splitting

This reduces manual review from 100k+ images to just the suspicious cases.
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.stamp_detector.detector import StampDetector
from scripts.scrape_postbeeld import get_mysql_conn


def draw_detections(image_path: str, detections: list, output_path: str):
    """Draw bounding boxes on image and save."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Colors for different detection counts
    colors = {
        0: "red",      # No detection - problem
        1: "green",    # Single stamp
        2: "blue",     # Multi-stamp
        3: "purple",
        4: "orange",
    }

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors.get(len(detections), "yellow")

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        conf = det.get('confidence', 0)
        label = f"#{i+1} {conf:.0%}"
        draw.text((x1 + 5, y1 + 5), label, fill=color)

    # Add summary text at top
    w, h = img.size
    aspect = w / h if h > 0 else 0
    summary = f"Detected: {len(detections)} stamps | Size: {w}x{h} | Aspect: {aspect:.2f}"
    draw.rectangle([0, 0, w, 25], fill="black")
    draw.text((5, 5), summary, fill="white")

    img.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Visual verification of stamp detection")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images to scan (0=all)")
    parser.add_argument("--output-dir", type=str, default="verification_output",
                        help="Directory to save annotated images")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--aspect-threshold", type=float, default=1.5,
                        help="Aspect ratio threshold for 'wide' images")
    args = parser.parse_args()

    # Create output subdirectories
    output_base = Path(args.output_dir)
    dir_no_detection = output_base / "no_detection"
    dir_suspicious_single = output_base / "suspicious_single"
    dir_multi_stamp = output_base / "multi_stamp"

    for d in [dir_no_detection, dir_suspicious_single, dir_multi_stamp]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    print("Loading stamp detector (YOLO)...")
    detector = StampDetector(confidence_threshold=args.confidence)
    if not detector.is_ready:
        print("ERROR: Detector not ready. YOLO model not found.")
        sys.exit(1)
    print(f"Detector loaded. Confidence threshold: {args.confidence}")

    # Query records
    conn = get_mysql_conn()
    cur = conn.cursor(dictionary=True)

    table = os.getenv("STAMP_TABLE", "postbeeld_stamps")
    limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""

    cur.execute(
        f"SELECT id, title, image_path FROM {table} "
        f"WHERE image_path IS NOT NULL AND variant_key = 'single' "
        f"ORDER BY id {limit_clause}"
    )
    rows = cur.fetchall()
    print(f"Scanning {len(rows)} images...")
    print(f"Output directories:")
    print(f"  No detection:      {dir_no_detection}")
    print(f"  Suspicious single: {dir_suspicious_single}")
    print(f"  Multi-stamp:       {dir_multi_stamp}")
    print("-" * 70)

    stats = {
        "scanned": 0,
        "missing": 0,
        "no_detection": 0,
        "suspicious_single": 0,
        "multi_stamp": 0,
        "normal_single": 0,
    }

    for row in rows:
        img_path = row["image_path"]
        if not img_path or not os.path.exists(img_path):
            stats["missing"] += 1
            continue

        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            stats["missing"] += 1
            continue

        aspect_ratio = w / h if h > 0 else 0
        detections = detector.detect(img_path)
        stamp_count = len(detections)
        stats["scanned"] += 1

        filename = os.path.basename(img_path)
        output_name = f"{row['id']}_{filename}"

        # Categorize and save suspicious cases
        if stamp_count == 0:
            # No detection - save for review
            stats["no_detection"] += 1
            output_path = dir_no_detection / output_name
            draw_detections(img_path, detections, str(output_path))
            print(f"  [NONE]    {filename} -> no_detection/")

        elif stamp_count == 1 and aspect_ratio > args.aspect_threshold:
            # Wide image marked as single - suspicious
            stats["suspicious_single"] += 1
            output_path = dir_suspicious_single / output_name
            draw_detections(img_path, detections, str(output_path))
            print(f"  [WIDE-1]  {filename} (aspect={aspect_ratio:.2f}) -> suspicious_single/")

        elif stamp_count > 1:
            # Multi-stamp - save for verification
            stats["multi_stamp"] += 1
            output_path = dir_multi_stamp / output_name
            draw_detections(img_path, detections, str(output_path))
            print(f"  [MULTI-{stamp_count}] {filename} -> multi_stamp/")

        else:
            # Normal single stamp - don't save
            stats["normal_single"] += 1

        # Progress update every 500 images
        if stats["scanned"] % 500 == 0:
            print(f"  ... scanned {stats['scanned']}/{len(rows)}")

    cur.close()
    conn.close()

    print("-" * 70)
    print("Summary:")
    print(f"  Scanned:            {stats['scanned']}")
    print(f"  Missing:            {stats['missing']}")
    print(f"  Normal single:      {stats['normal_single']} (not saved)")
    print(f"  No detection:       {stats['no_detection']} -> {dir_no_detection}")
    print(f"  Suspicious single:  {stats['suspicious_single']} -> {dir_suspicious_single}")
    print(f"  Multi-stamp:        {stats['multi_stamp']} -> {dir_multi_stamp}")
    print("-" * 70)
    print(f"Review the images in: {output_base.absolute()}")


if __name__ == "__main__":
    main()
