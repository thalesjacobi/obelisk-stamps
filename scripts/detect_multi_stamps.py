#!/usr/bin/env python3
"""
Scan existing images in the database, detect multi-stamp images,
split them into individual stamp crops, and insert new records
with fractional prices.

This script:
1. Queries all postbeeld_stamps records with an image_path
2. Runs the YOLO stamp detector on each image
3. For multi-stamp images: crops each stamp, saves the crop file,
   and inserts a new DB record with price split equally
4. Updates the original record's group_count
5. Prints a summary of multi-stamp images found

Usage:
    python scripts/detect_multi_stamps.py                  # Scan all, limit 500
    python scripts/detect_multi_stamps.py --limit 100      # Scan first 100
    python scripts/detect_multi_stamps.py --limit 0        # Scan all (no limit)
    python scripts/detect_multi_stamps.py --confidence 0.3 # Custom confidence
    python scripts/detect_multi_stamps.py --dry-run        # Preview only

Run from project root: D:/CodingFolder/obelisk-stamps
"""

import os
import sys
import hashlib
import argparse
from decimal import Decimal
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from ml.stamp_detector import StampDetector


def sha256_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()


def get_db_connection():
    import mysql.connector
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", ""),
        autocommit=True,
    )


def insert_stamp_part(cur, table, original_row, part_path, part_idx, total_parts):
    """Insert a new record for a cropped stamp part with fractional price."""
    source_url = original_row["source_url"]
    variant_key = f"p{part_idx}of{total_parts}"

    # Split price equally
    part_price = None
    if original_row.get("price_value") is not None:
        try:
            part_price = Decimal(str(original_row["price_value"])) / total_parts
            part_price = part_price.quantize(Decimal("0.01"))
        except Exception:
            part_price = original_row["price_value"]

    sql = f"""
    INSERT INTO {table} (
        source, source_url, source_url_hash, title, country, year,
        price_value, price_currency, condition_text,
        image_url, image_url_full, image_path,
        variant_key, group_key, group_part, group_count
    ) VALUES (
        %(source)s, %(source_url)s, %(source_url_hash)s, %(title)s,
        %(country)s, %(year)s, %(price_value)s, %(price_currency)s,
        %(condition_text)s, %(image_url)s, %(image_url_full)s, %(image_path)s,
        %(variant_key)s, %(group_key)s, %(group_part)s, %(group_count)s
    )
    ON DUPLICATE KEY UPDATE
        title=VALUES(title), country=VALUES(country), year=VALUES(year),
        price_value=VALUES(price_value), price_currency=VALUES(price_currency),
        condition_text=VALUES(condition_text), image_url=VALUES(image_url),
        image_url_full=VALUES(image_url_full),
        image_path=VALUES(image_path), group_key=VALUES(group_key),
        group_part=VALUES(group_part), group_count=VALUES(group_count),
        scraped_at=CURRENT_TIMESTAMP
    """

    payload = {
        "source": original_row.get("source", "postbeeld"),
        "source_url": source_url,
        "source_url_hash": sha256_bytes(source_url),
        "title": original_row.get("title"),
        "country": original_row.get("country"),
        "year": original_row.get("year"),
        "price_value": part_price,
        "price_currency": original_row.get("price_currency"),
        "condition_text": original_row.get("condition_text"),
        "image_url": original_row.get("image_url"),
        "image_url_full": original_row.get("image_url_full"),
        "image_path": part_path,
        "variant_key": variant_key,
        "group_key": original_row.get("group_key"),
        "group_part": part_idx,
        "group_count": total_parts,
    }

    cur.execute(sql, payload)


def main():
    parser = argparse.ArgumentParser(description="Detect multi-stamp images, split and insert records")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max images to scan (0 = unlimited)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--only-unscanned", action="store_true",
                        help="Only scan records where group_count is NULL or 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't update DB or save crops, just report findings")
    args = parser.parse_args()

    # Initialize YOLO-based detector
    print("Loading stamp detector (YOLO)...")
    detector = StampDetector(
        confidence_threshold=args.confidence,
    )
    if not detector.is_ready:
        print("ERROR: Detector not ready. YOLO model not found.")
        print("Run training first: python ml/stamp_detector/train.py")
        sys.exit(1)
    print(f"Detector loaded. Confidence threshold: {args.confidence}")

    # Query records
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    where = "WHERE image_path IS NOT NULL AND variant_key = 'single'"
    if args.only_unscanned:
        where += " AND (group_count IS NULL OR group_count = 1)"

    limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""
    table = os.getenv("STAMP_TABLE", "postbeeld_stamps")

    cur.execute(
        f"SELECT id, source, source_url, title, country, year, "
        f"price_value, price_currency, condition_text, "
        f"image_url, image_url_full, image_path, "
        f"group_key, group_count "
        f"FROM {table} {where} ORDER BY id {limit_clause}"
    )
    rows = cur.fetchall()
    print(f"Scanning {len(rows)} records...")
    print("-" * 70)

    multi_stamp_records = []
    scanned = 0
    records_updated = 0
    parts_inserted = 0
    missing = 0

    for row in rows:
        img_path = row["image_path"]
        if not img_path or not os.path.exists(img_path):
            missing += 1
            continue

        stamp_count = detector.count_stamps(img_path)
        scanned += 1
        filename = os.path.basename(img_path)

        # Always print the result for visibility
        if stamp_count == 0:
            print(f"  [NONE]    id={row['id']} {filename} | {row['title'][:50]}")
        elif stamp_count == 1:
            print(f"  [SINGLE]  id={row['id']} {filename} | {row['title'][:50]}")
        else:
            print(f"  [MULTI-{stamp_count}] id={row['id']} {filename} | {row['title'][:50]}")

        if stamp_count > 1:
            multi_stamp_records.append({
                "id": row["id"],
                "title": row["title"],
                "stamp_count": stamp_count,
                "image_path": img_path,
            })

            if not args.dry_run:
                # Crop and save individual stamps
                saved_paths = detector.detect_and_save(
                    img_path, output_dir=None, padding=2, quality=95
                )

                if saved_paths:
                    num_parts = len(saved_paths)

                    # Insert a record for each cropped stamp
                    for idx, part_path in enumerate(saved_paths, start=1):
                        insert_stamp_part(cur, table, row, part_path, idx, num_parts)
                        parts_inserted += 1

                    # Update the original record
                    cur.execute(
                        f"UPDATE {table} SET group_count = %s, variant_key = 'combo' WHERE id = %s",
                        (num_parts, row["id"]),
                    )
                    records_updated += 1
                else:
                    # Detection found multi but cropping failed; just update count
                    cur.execute(
                        f"UPDATE {table} SET group_count = %s WHERE id = %s",
                        (stamp_count, row["id"]),
                    )
                    records_updated += 1

        elif not args.dry_run and stamp_count > 0:
            # Single stamp: just update group_count
            cur.execute(
                f"UPDATE {table} SET group_count = %s WHERE id = %s",
                (stamp_count, row["id"]),
            )
            records_updated += 1

        if scanned % 100 == 0:
            print(f"  ... scanned {scanned}/{len(rows)}")

    cur.close()
    conn.close()

    # Summary
    print("-" * 70)
    print(f"Scanned:         {scanned}")
    print(f"Missing imgs:    {missing}")
    print(f"Multi-stamp:     {len(multi_stamp_records)}")
    if not args.dry_run:
        print(f"Records updated: {records_updated}")
        print(f"Parts inserted:  {parts_inserted}")
    else:
        print("DB changes:      SKIPPED (dry-run)")

    if multi_stamp_records:
        print(f"\nMulti-stamp images found:")
        for r in multi_stamp_records:
            filename = os.path.basename(r['image_path'])
            print(f"  id={r['id']:>6}  stamps={r['stamp_count']}  {filename}  |  {r['title'][:50]}")


if __name__ == "__main__":
    main()
