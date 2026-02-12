#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PostBeeld scraper (grid pages)

Features:
- Scrapes PostBeeld stamp listing pages
- Downloads full-size images (not thumbnails) when available
- Detects multi-stamp images and splits them into individual records
- Splits prices equally among stamps in multi-stamp images
- Proper group_key linking for combo + part records
- Upserts into postbeeld_stamps with UNIQUE(source_url_hash, variant_key)
"""

import os
import re
import sys
import time
import hashlib
from dataclasses import dataclass, asdict
from decimal import Decimal
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

# Import ML stamp detector (optional - falls back to legacy if not available)
try:
    from ml.stamp_detector import StampDetector
    _stamp_detector = None  # Lazy initialization
except ImportError:
    StampDetector = None
    _stamp_detector = None


# ----------------------------
# Config
# ----------------------------

# ENABLE_IMAGE_SPLITTING controls whether multi-stamp images are split into individual parts.
# - False: Images are saved as-is (no splitting). Each scraped item = 1 database row.
# - True:  Multi-stamp images are detected and split.
#          Each part becomes a separate database row with group_part/group_count metadata.
ENABLE_IMAGE_SPLITTING = False  # Disabled - run splitting separately after scraping

# ENABLE_STAMP_DETECTION controls whether the ML detector runs to COUNT stamps per image.
# - When True: Each image is analyzed by the detector. group_count is set to the
#   number of stamps detected. Images with group_count > 1 are multi-stamp images.
#   No splitting occurs unless ENABLE_IMAGE_SPLITTING is also True.
# - When False: group_count defaults to 1 (legacy behavior).
ENABLE_STAMP_DETECTION = False  # Disabled - run splitting separately after scraping

BASE_LIST_URL = "https://www.postbeeld.com/stamps/page/{page}/mode/grid/show/120"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

DEBUG_DIR = "debug"
DATA_DIR = os.path.join("data", "postbeeld")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

SOURCE_NAME = "postbeeld"

# PostBeeld Magento image cache pattern
# Cached:   /media/catalog/product/cache/2/small_image/{size}/{hash}/{path}
# Original: /media/catalog/product/{path}
_CACHE_RE = re.compile(
    r"/media/catalog/product/cache/\d+/[^/]+/[^/]+/[0-9a-f]{32}(/.*)"
)
_PRODUCT_BASE = "https://www.postbeeld.com/media/catalog/product"

# Your .env keys
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "")
STAMP_TABLE = os.getenv("STAMP_TABLE", "postbeeld_stamps")


# ----------------------------
# Model
# ----------------------------

@dataclass
class StampItem:
    source: str
    source_url: str
    sku: Optional[str]
    title: str
    country: Optional[str]
    year: Optional[int]
    condition_text: Optional[str]
    price_value: Optional[Decimal]
    price_currency: Optional[str]
    image_url: Optional[str]          # Thumbnail URL from listing page
    image_url_full: Optional[str]     # Full-size original URL
    image_path: Optional[str]         # Local file path (full-size download)
    group_key: Optional[str]          # Links combo + parts together
    group_count: Optional[int]        # Total stamps in the group
    group_part: Optional[int]         # Which part (1, 2, 3...)
    variant_key: Optional[str]        # single, combo, p1of4, etc.


# ----------------------------
# Utilities
# ----------------------------

def ensure_dirs() -> None:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def sha256_bytes(s: str) -> bytes:
    """SHA256 digest for source_url_hash BINARY(32) column."""
    return hashlib.sha256(s.encode("utf-8")).digest()


def absolutize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return "https://www.postbeeld.com" + url
    return "https://www.postbeeld.com/" + url


def derive_full_image_url(cached_url: str) -> Optional[str]:
    """
    Convert a PostBeeld cached thumbnail URL to the full-size original URL.

    Cached:   .../media/catalog/product/cache/2/small_image/400x400/{hash}/s/p/sp6509_0.jpg
    Original: .../media/catalog/product/s/p/sp6509_0.jpg

    Returns the full-size URL, or None if the URL doesn't match the cache pattern.
    """
    if not cached_url:
        return None
    m = _CACHE_RE.search(cached_url)
    if m:
        return _PRODUCT_BASE + m.group(1)
    return None


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch(url: str, session: requests.Session, timeout: int = 30) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def save_debug_html(page: int, html: str) -> str:
    path = os.path.join(DEBUG_DIR, f"postbeeld_page_{page}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def safe_decimal(x: str) -> Optional[Decimal]:
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return None
    x = x.replace(",", ".")
    x = re.sub(r"[^0-9.]", "", x)
    if not x:
        return None
    try:
        return Decimal(x)
    except Exception:
        return None


def normalize_currency(cur: Optional[str]) -> Optional[str]:
    if not cur:
        return None
    cur = cur.strip()
    if cur == "€":
        return "EUR"
    return cur


# ----------------------------
# Parsing
# ----------------------------

def parse_list_page(html: str) -> List[StampItem]:
    """
    Parse PostBeeld listing page HTML into StampItem objects.

    Current PostBeeld markup:
    - card: div.product-in-grid
    - title link: div.descrip-wrapper h3 a[href]
    - url meta: meta[itemprop="url"][content]
    - name meta: meta[itemprop="name"][content]
    - sku: meta[itemprop="sku"][content] OR p.extra-info span[itemprop="sku"]
    - image: link[itemprop="image"][content]
    - first condition row only: tr.condition_row
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.product-in-grid")
    items: List[StampItem] = []

    for card in cards:
        title = None
        source_url = None

        a = card.select_one("div.descrip-wrapper h3 a[href]")
        if a:
            source_url = (a.get("href") or "").strip()
            title = a.get_text(" ", strip=True)

        if not source_url:
            meta_url = card.select_one('meta[itemprop="url"][content]')
            if meta_url and meta_url.get("content"):
                source_url = meta_url["content"].strip()

        if not title:
            meta_name = card.select_one('meta[itemprop="name"][content]')
            if meta_name and meta_name.get("content"):
                title = meta_name["content"].strip()

        if not source_url or not title:
            continue

        source_url = absolutize_url(source_url)

        # Exclude folds/letters
        t_upper = title.upper()
        if "FOLD" in t_upper or "LETTER" in t_upper:
            continue

        # SKU
        sku = None
        sku_meta = card.select_one('meta[itemprop="sku"][content]')
        if sku_meta and sku_meta.get("content"):
            sku = sku_meta["content"].strip()
        else:
            sku_span = card.select_one('p.extra-info span[itemprop="sku"]')
            if sku_span:
                sku = sku_span.get_text(" ", strip=True)

        # Image URL (thumbnail from listing page)
        image_url = None
        img_tag = card.select_one('link[itemprop="image"][content], link[itemprop="image"][href]')
        if img_tag:
            image_url = (img_tag.get("content") or img_tag.get("href") or "").strip()
        image_url = absolutize_url(image_url) if image_url else None

        # Derive full-size image URL
        image_url_full = derive_full_image_url(image_url) if image_url else None

        # Country + year
        country = None
        year = None
        info_spans = card.select("p.extra-info span")
        if len(info_spans) >= 3:
            country = info_spans[1].get_text(" ", strip=True)
            ytxt = info_spans[2].get_text(" ", strip=True)
            m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", ytxt)
            year = int(m.group(1)) if m else None

        # FIRST condition only
        condition_text = None
        price_value = None
        price_currency = None

        first_row = card.select_one("tr.condition_row")
        if first_row:
            cspan = first_row.select_one("span.condition-name")
            if cspan:
                condition_text = cspan.get_text(" ", strip=True)

            p = first_row.select_one('[itemprop="price"]')
            cur = first_row.select_one('[itemprop="priceCurrency"]')

            if cur:
                price_currency = normalize_currency(cur.get("content") or cur.get_text(" ", strip=True))

            if p:
                ptxt = p.get("content") or p.get_text(" ", strip=True)
                price_value = safe_decimal(ptxt)

        group_key = f"{SOURCE_NAME}:{sha1(source_url)}"

        items.append(
            StampItem(
                source=SOURCE_NAME,
                source_url=source_url,
                sku=sku,
                title=title,
                country=country,
                year=year,
                condition_text=condition_text,
                price_value=price_value,
                price_currency=price_currency,
                image_url=image_url,
                image_url_full=image_url_full,
                image_path=None,
                group_key=group_key,
                group_count=1,
                group_part=None,
                variant_key="single",
            )
        )

    return items


# ----------------------------
# Image download
# ----------------------------

def _sanitize_for_filename(s: Optional[str]) -> str:
    """Sanitize a string for use in a filename."""
    if not s:
        return "unknown"
    # Lowercase, replace spaces/special chars with underscore
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:20] if s else "unknown"  # Limit length


def _generate_image_filename(
    url: str,
    year: Optional[int] = None,
    country: Optional[str] = None,
    variant_key: Optional[str] = None,
) -> str:
    """
    Generate a human-readable image filename.

    Format: {year}_{country}_{3char_hash}.jpg
    Examples:
        1867_turkey_a7f.jpg
        1920_netherlands_b3c.jpg
        unknown_unknown_f2e.jpg (if metadata missing)

    The 3-char hash ensures uniqueness when year+country collide.
    """
    year_str = str(year) if year else "unknown"
    country_str = _sanitize_for_filename(country)

    # Generate a short hash from URL for uniqueness
    url_hash = sha1(url)[:3]

    # Add variant suffix if it's a part (p1of4, p2of4, etc.)
    if variant_key and variant_key.startswith("p") and "of" in variant_key:
        return f"{year_str}_{country_str}_{url_hash}_{variant_key}.jpg"

    return f"{year_str}_{country_str}_{url_hash}.jpg"


def download_image(
    session: requests.Session,
    url: str,
    out_dir: str,
    fallback_url: Optional[str] = None,
    year: Optional[int] = None,
    country: Optional[str] = None,
    variant_key: Optional[str] = None,
) -> Optional[str]:
    """
    Download an image, trying the primary URL first and falling back if needed.

    Args:
        session: requests session
        url: Primary URL to download (typically full-size original)
        out_dir: Output directory
        fallback_url: Fallback URL if primary fails (typically thumbnail)
        year: Stamp year (for filename)
        country: Stamp country (for filename)
        variant_key: Variant key like 'single', 'combo', 'p1of4' (for filename)

    Returns:
        Local file path, or None on failure.

    Filename format: {year}_{country}_{3char_hash}.jpg
    """
    if not url:
        return None
    os.makedirs(out_dir, exist_ok=True)

    # Generate human-readable filename
    fname = _generate_image_filename(url, year, country, variant_key)
    path = os.path.join(out_dir, fname)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    # Try primary URL
    try:
        r = session.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            with open(path, "wb") as f:
                f.write(r.content)
            return path
    except Exception:
        pass

    # Try fallback URL
    if fallback_url and fallback_url != url:
        # Use same naming convention for fallback
        fname_fb = _generate_image_filename(fallback_url, year, country, variant_key)
        path_fb = os.path.join(out_dir, fname_fb)
        if os.path.exists(path_fb) and os.path.getsize(path_fb) > 0:
            return path_fb
        try:
            r = session.get(fallback_url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100:
                with open(path_fb, "wb") as f:
                    f.write(r.content)
                return path_fb
        except Exception:
            pass

    return None


# ----------------------------
# Multi-stamp detection
# ----------------------------

def _get_detector() -> Optional["StampDetector"]:
    """Lazily initialize the hybrid stamp detector (YOLO + valley fallback)."""
    global _stamp_detector

    if StampDetector is None:
        return None

    if _stamp_detector is None:
        _stamp_detector = StampDetector(
            confidence_threshold=0.3,
            use_valley_fallback=True,
        )

    if not _stamp_detector.is_ready:
        return None

    return _stamp_detector


def detect_stamp_count(image_path: str) -> int:
    """Count stamps in an image using the hybrid detector."""
    detector = _get_detector()
    if detector is None:
        return 0
    return detector.count_stamps(image_path)


def detect_and_crop_stamps(image_path: str, year: Optional[int] = None, country: Optional[str] = None) -> List[str]:
    """
    Detect stamps in an image and save each as a separate cropped file.

    Filenames follow the pattern: {year}_{country}_{hash}_p{idx}of{total}.jpg
    Example: 1864_netherlands_365_p1of3.jpg, 1864_netherlands_365_p2of3.jpg

    Returns list of saved crop file paths. Empty if single/no stamps.
    """
    detector = _get_detector()
    if detector is None:
        return []

    stamp_count = detector.count_stamps(image_path)
    if stamp_count <= 1:
        return []

    # Get the crops as PIL images
    crops = detector.detect_and_crop(image_path, padding=2)
    if not crops:
        return []

    num_stamps = len(crops)
    out_dir = os.path.dirname(image_path)

    # Extract base name without extension
    # e.g., "1864_netherlands_365.jpg" -> "1864_netherlands_365"
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    saved_paths = []
    for idx, crop in enumerate(crops, start=1):
        # Generate filename with part suffix: {base}_p{idx}of{total}.jpg
        crop_name = f"{base_name}_p{idx}of{num_stamps}.jpg"
        crop_path = os.path.join(out_dir, crop_name)

        crop.save(crop_path, quality=95)
        saved_paths.append(crop_path)

    return saved_paths


# ----------------------------
# Database
# ----------------------------

def get_mysql_conn():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_NAME", "")

    if not database:
        raise RuntimeError("DB_NAME is missing in .env")

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True,
    )


def coerce_group_fields(d: dict) -> dict:
    """Ensure group_part and group_count are None or int."""
    gp = d.get("group_part")
    gc = d.get("group_count")

    if isinstance(gp, str):
        m = re.match(r"^p(\d+)of(\d+)$", gp.strip(), re.IGNORECASE)
        if m:
            d["group_part"] = int(m.group(1))
            if d.get("group_count") in (None, "", 0, "0"):
                d["group_count"] = int(m.group(2))
        else:
            d["group_part"] = None

    if isinstance(d.get("group_part"), str) and d["group_part"].isdigit():
        d["group_part"] = int(d["group_part"])

    if isinstance(gc, str) and gc.isdigit():
        d["group_count"] = int(gc)

    if d.get("group_part") is not None and not isinstance(d["group_part"], int):
        d["group_part"] = None
    if d.get("group_count") is not None and not isinstance(d["group_count"], int):
        d["group_count"] = None

    return d


def upsert_items(items) -> int:
    """
    Upsert into postbeeld_stamps.
    Key: UNIQUE(source_url_hash, variant_key)
    """
    if not items:
        return 0

    table = os.getenv("STAMP_TABLE", "postbeeld_stamps")
    conn = get_mysql_conn()
    cur = conn.cursor()

    sql = f"""
    INSERT INTO {table} (
        source,
        source_url,
        source_url_hash,
        title,
        country,
        year,
        price_value,
        price_currency,
        condition_text,
        image_url,
        image_url_full,
        image_path,
        variant_key,
        group_key,
        group_part,
        group_count
    ) VALUES (
        %(source)s,
        %(source_url)s,
        %(source_url_hash)s,
        %(title)s,
        %(country)s,
        %(year)s,
        %(price_value)s,
        %(price_currency)s,
        %(condition_text)s,
        %(image_url)s,
        %(image_url_full)s,
        %(image_path)s,
        %(variant_key)s,
        %(group_key)s,
        %(group_part)s,
        %(group_count)s
    )
    ON DUPLICATE KEY UPDATE
        source=VALUES(source),
        source_url=VALUES(source_url),
        title=VALUES(title),
        country=VALUES(country),
        year=VALUES(year),
        price_value=VALUES(price_value),
        price_currency=VALUES(price_currency),
        condition_text=VALUES(condition_text),
        image_url=VALUES(image_url),
        image_url_full=VALUES(image_url_full),
        image_path=VALUES(image_path),
        group_key=VALUES(group_key),
        group_part=VALUES(group_part),
        group_count=VALUES(group_count),
        scraped_at=CURRENT_TIMESTAMP
    """

    n = 0
    for it in items:
        d = asdict(it) if hasattr(it, "__dataclass_fields__") else dict(it)

        source_url = (d.get("source_url") or "").strip()
        if not source_url:
            continue

        variant_key = (d.get("variant_key") or "single").strip()[:32] or "single"

        payload = {
            "source": d.get("source") or "postbeeld",
            "source_url": source_url,
            "source_url_hash": sha256_bytes(source_url),

            "title": d.get("title"),
            "country": d.get("country"),
            "year": d.get("year"),
            "price_value": d.get("price_value"),
            "price_currency": d.get("price_currency"),
            "condition_text": d.get("condition_text"),
            "image_url": d.get("image_url"),
            "image_url_full": d.get("image_url_full"),
            "image_path": d.get("image_path"),

            "variant_key": variant_key,
            "group_key": d.get("group_key"),
            "group_part": d.get("group_part"),
            "group_count": d.get("group_count"),
        }

        payload = coerce_group_fields(payload)

        cur.execute(sql, payload)
        n += 1

    cur.close()
    conn.close()
    return n


# ----------------------------
# Main scrape
# ----------------------------

def scrape_pages(
    start_page: int,
    end_page: int,
    split_combos: bool = ENABLE_IMAGE_SPLITTING,
    detect_stamps: bool = ENABLE_STAMP_DETECTION,
) -> Tuple[int, int]:
    """
    Scrape PostBeeld stamp pages and save to database.

    For each listing:
    1. Parse title, price, country, year, image URL from listing page
    2. Derive full-size image URL from the cached thumbnail URL
    3. Download full-size image (fall back to thumbnail if unavailable)
    4. Run multi-stamp detection on the downloaded image
    5. If multi-stamp: crop each stamp, create separate DB records with split prices
    6. Upsert all records

    Returns:
        Tuple of (upsert_count, images_downloaded)
    """
    ensure_dirs()
    session = get_session()

    all_items: List[StampItem] = []
    images_downloaded = 0
    multi_stamp_count = 0
    full_size_count = 0

    print(f"[config] Image splitting: {'ENABLED' if split_combos else 'DISABLED'}")
    print(f"[config] Stamp detection: {'ENABLED' if detect_stamps else 'DISABLED'}")

    for page in range(start_page, end_page + 1):
        url = BASE_LIST_URL.format(page=page)
        print(f"[page {page}] fetching: {url}")

        html = fetch(url, session=session)
        debug_path = save_debug_html(page, html)

        items = parse_list_page(html)
        print(f"[page {page}] found {len(items)} items  DEBUG: saved {debug_path}")

        expanded: List[StampItem] = []

        for it in items:
            # Download image: prefer full-size, fall back to thumbnail
            if it.image_url:
                download_url = it.image_url_full or it.image_url
                p = download_image(
                    session,
                    download_url,
                    IMAGES_DIR,
                    fallback_url=it.image_url,  # Thumbnail as fallback
                    year=it.year,
                    country=it.country,
                    variant_key=it.variant_key,
                )
                if p:
                    images_downloaded += 1
                    it.image_path = p
                    if it.image_url_full and download_url == it.image_url_full:
                        full_size_count += 1

            # Run stamp detection on downloaded image
            if it.image_path and os.path.exists(it.image_path) and (split_combos or detect_stamps):
                crop_paths = detect_and_crop_stamps(it.image_path, year=it.year, country=it.country) if split_combos else []

                if crop_paths:
                    num_stamps = len(crop_paths)
                    multi_stamp_count += 1
                    print(f"    [MULTI] {it.title}: {num_stamps} stamps -> splitting")

                    # Split price equally
                    part_price = None
                    if it.price_value is not None:
                        part_price = Decimal(str(it.price_value)) / num_stamps
                        part_price = part_price.quantize(Decimal("0.01"))

                    # Create a record for each individual stamp
                    for idx, part_path in enumerate(crop_paths, start=1):
                        part = StampItem(**asdict(it))
                        part.image_path = part_path
                        part.group_part = idx
                        part.group_count = num_stamps
                        part.variant_key = f"p{idx}of{num_stamps}"
                        part.price_value = part_price
                        # group_key stays the same — links all parts + combo together
                        expanded.append(part)

                    # Keep the original combo record too
                    it.group_count = num_stamps
                    it.variant_key = "combo"
                    expanded.append(it)
                    continue
                else:
                    # Single stamp or detection found <= 1
                    if detect_stamps:
                        stamp_count = detect_stamp_count(it.image_path)
                        if stamp_count > 0:
                            it.group_count = stamp_count

            expanded.append(it)

        all_items.extend(expanded)

        time.sleep(0.6)  # be polite

    print(f"\n[stats] Pages: {end_page - start_page + 1}")
    print(f"[stats] Images downloaded: {images_downloaded} ({full_size_count} full-size)")
    print(f"[stats] Multi-stamp images split: {multi_stamp_count}")
    print(f"[stats] Total records to upsert: {len(all_items)}")

    upserts = upsert_items(all_items)
    return upserts, images_downloaded


def main():
    """
    Main entry point for the scraper.

    Usage:
        python scripts/scrape_postbeeld.py [start_page] [end_page]

    Examples:
        python scripts/scrape_postbeeld.py           # Scrape page 1 only
        python scripts/scrape_postbeeld.py 1 10      # Scrape pages 1-10
        python scripts/scrape_postbeeld.py 1 650     # Full re-scrape (~76K stamps)
    """
    start_page = 1
    end_page = 1

    if len(sys.argv) >= 2:
        try:
            start_page = int(sys.argv[1])
        except Exception:
            pass
    if len(sys.argv) >= 3:
        try:
            end_page = int(sys.argv[2])
        except Exception:
            pass

    upserts, imgs = scrape_pages(
        start_page,
        end_page,
        split_combos=ENABLE_IMAGE_SPLITTING,
        detect_stamps=ENABLE_STAMP_DETECTION,
    )
    print("\nDone.")
    print(f"Upserts: {upserts}. Images downloaded: {imgs}.")
    print(f"Images folder: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
