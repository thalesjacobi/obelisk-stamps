#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PostBeeld scraper (grid pages)

Fixes:
- Uses current PostBeeld HTML selectors (no more "found 0 items" on page 4)
- Takes ONLY the first condition row on each card (ignores 2nd/others)
- Uses your .env keys: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, STAMP_TABLE
- Downloads images
- Detects + splits multi-stamp images (2-up / 3-up / simple grids) and inserts multiple records
- Marks groups via: is_group, group_key, group_count, group_part, variant_key

Important:
- This assumes your table has the columns used in the UPSERT SQL below.
  If the UPSERT fails, paste your CREATE TABLE for STAMP_TABLE and I will return a full corrected file.
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


# ----------------------------
# Config
# ----------------------------

# ENABLE_IMAGE_SPLITTING controls whether multi-stamp images are split into individual parts.
# - False: Images are saved as-is (no splitting). Each scraped item = 1 database row.
# - True:  Multi-stamp images (2-up, 3-up, 2x2 grids) are detected and split.
#          Each part becomes a separate database row with group_part/group_count metadata.
ENABLE_IMAGE_SPLITTING = False

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
    image_url: Optional[str]
    image_path: Optional[str]
    group_key: Optional[str]
    group_count: Optional[int]
    group_part: Optional[str]
    variant_key: Optional[str]
    is_group: int  # 0/1


# ----------------------------
# Utilities
# ----------------------------

def ensure_dirs() -> None:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def absolutize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return "https://www.postbeeld.com" + url
    return "https://www.postbeeld.com/" + url


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
# Parsing (FIXES "0 items")
# ----------------------------

def parse_list_page(html: str) -> List[StampItem]:
    """
    Current PostBeeld markup (as of your debug HTML):
    - card: div.product-in-grid
    - title link: div.descrip-wrapper h3 a[href]
    - url meta: meta[itemprop="url"][content]
    - name meta: meta[itemprop="name"][content]
    - sku: meta[itemprop="sku"][content] OR p.extra-info span[itemprop="sku"]
    - image: link[itemprop="image"][content]
    - first condition row only: tr.condition_row
      - condition: span.condition-name
      - price: [itemprop=price] content="7.50"
      - currency: [itemprop=priceCurrency] content="EUR"
    - extra info: p.extra-info spans -> [sku, country, year]
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.product-in-grid")
    items: List[StampItem] = []

    for card in cards:
        title = None
        source_url = None

        # Preferred visible title link
        a = card.select_one("div.descrip-wrapper h3 a[href]")
        if a:
            source_url = (a.get("href") or "").strip()
            title = a.get_text(" ", strip=True)

        # Fallback to schema.org
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

        # Image URL
        image_url = None
        img_tag = card.select_one('link[itemprop="image"][content], link[itemprop="image"][href]')
        if img_tag:
            image_url = (img_tag.get("content") or img_tag.get("href") or "").strip()
        image_url = absolutize_url(image_url) if image_url else None

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
                image_path=None,
                group_key=group_key,
                group_count=1,
                group_part=None,
                variant_key="single",
                is_group=0,
            )
        )

    return items


# ----------------------------
# Image download + splitting
# ----------------------------

def download_image(session: requests.Session, url: str, out_dir: str) -> Optional[str]:
    if not url:
        return None
    os.makedirs(out_dir, exist_ok=True)

    ext = ".jpg"
    m = re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", url, re.IGNORECASE)
    if m:
        ext = "." + m.group(1).lower().replace("jpeg", "jpg")

    fname = sha1(url) + ext
    path = os.path.join(out_dir, fname)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None

    with open(path, "wb") as f:
        f.write(r.content)

    return path


def _ink_profile(img: Image.Image, axis: str, ignore_bottom_frac: float = 0.22) -> List[int]:
    """
    Sum of "ink" pixels per column (axis='x') or per row (axis='y').

    We ignore a bottom slice because PostBeeld watermark often sits there and can confuse splits.
    """
    gray = ImageOps.grayscale(img)
    w, h = gray.size
    px = gray.load()

    y_max = int(h * (1.0 - ignore_bottom_frac))
    y_max = max(1, min(h, y_max))

    if axis == "x":
        prof = [0] * w
        for x in range(w):
            s = 0
            for y in range(y_max):
                if px[x, y] < 240:
                    s += 1
            prof[x] = s
        return prof

    prof = [0] * y_max
    for y in range(y_max):
        s = 0
        for x in range(w):
            if px[x, y] < 240:
                s += 1
        prof[y] = s
    return prof


def _find_valleys(profile: List[int], min_run: int, low_quantile: float = 0.12) -> List[Tuple[int, int]]:
    if not profile:
        return []
    sorted_vals = sorted(profile)
    q_idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * low_quantile)))
    thresh = sorted_vals[q_idx]

    valleys = []
    in_run = False
    start = 0
    for i, v in enumerate(profile):
        if v <= thresh:
            if not in_run:
                in_run = True
                start = i
        else:
            if in_run:
                end = i - 1
                if (end - start + 1) >= min_run:
                    valleys.append((start, end))
                in_run = False
    if in_run:
        end = len(profile) - 1
        if (end - start + 1) >= min_run:
            valleys.append((start, end))
    return valleys


def split_combo_image(image_path: str) -> List[str]:
    """
    Try to split multi-stamp images:
    - 2 side-by-side (your 2-stamp example)
    - 3 side-by-side (your 3-stamp example)
    - simple 2x2 grid if obvious

    Returns list of new image paths (parts). If no confident split, returns [].
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return []

    w, h = img.size

    # Allow small images like your examples (177x104, 267x131)
    if w < 140 or h < 90:
        return []

    col_prof = _ink_profile(img, "x")
    row_prof = _ink_profile(img, "y")

    # Run-length thresholds adapt to size
    v_valleys = _find_valleys(col_prof, min_run=max(3, w // 60))
    h_valleys = _find_valleys(row_prof, min_run=max(3, h // 60))

    # candidate cut positions: center of each valley range
    v_cuts = sorted({(a + b) // 2 for a, b in v_valleys})
    h_cuts = sorted({(a + b) // 2 for a, b in h_valleys})

    # Helper: reject cuts too close to edges
    def valid_cuts(cuts: List[int], size: int) -> List[int]:
        out = []
        for c in cuts:
            if c < int(size * 0.18):
                continue
            if c > int(size * 0.82):
                continue
            out.append(c)
        return out

    v_cuts = valid_cuts(v_cuts, w)
    h_cuts = valid_cuts(h_cuts, h)

    # For horizontal strips (2 or 3 stamps), we want 1 or 2 cuts
    # Pick up to 2 cuts that are reasonably spaced.
    def pick_up_to_two(cuts: List[int], size: int) -> List[int]:
        if not cuts:
            return []
        # Prefer cuts closest to 1/3 and 2/3 (works for 3-up) then center (2-up)
        targets = [size // 3, (2 * size) // 3, size // 2]
        scored = []
        for c in cuts:
            score = min(abs(c - t) for t in targets)
            scored.append((score, c))
        scored.sort()
        chosen = []
        for _, c in scored:
            if all(abs(c - x) > max(8, size // 10) for x in chosen):
                chosen.append(c)
            if len(chosen) == 2:
                break
        return sorted(chosen)

    v_cuts = pick_up_to_two(v_cuts, w)
    h_cuts = pick_up_to_two(h_cuts, h)

    # Crop+save helper
    def save_crop(crop: Image.Image, suffix: str) -> Optional[str]:
        cw, ch = crop.size
        # keep small but not tiny
        if cw < 60 or ch < 60:
            return None
        out_path = image_path.replace(".", f"_{suffix}.")
        try:
            crop.save(out_path, quality=95)
            return out_path
        except Exception:
            return None

    parts: List[str] = []

    # Try vertical splits first (common for 2-up / 3-up)
    if v_cuts:
        xs = [0] + v_cuts + [w]
        crops = []
        for i in range(len(xs) - 1):
            x1, x2 = xs[i], xs[i + 1]
            crop = img.crop((x1, 0, x2, h))
            crops.append(crop)

        saved = [save_crop(c, f"p{i+1}") for i, c in enumerate(crops)]
        saved = [p for p in saved if p]
        if len(saved) in (2, 3):
            return saved  # confident 2-up or 3-up

    # Try horizontal split (less common but supported)
    if h_cuts:
        ys = [0] + h_cuts + [h]
        crops = []
        for i in range(len(ys) - 1):
            y1, y2 = ys[i], ys[i + 1]
            crop = img.crop((0, y1, w, y2))
            crops.append(crop)

        saved = [save_crop(c, f"p{i+1}") for i, c in enumerate(crops)]
        saved = [p for p in saved if p]
        if len(saved) == 2:
            return saved

    # Try 2x2 grid if both exist
    if len(v_cuts) == 1 and len(h_cuts) == 1 and w >= 180 and h >= 120:
        xc = v_cuts[0]
        yc = h_cuts[0]
        tiles = [
            img.crop((0, 0, xc, yc)),
            img.crop((xc, 0, w, yc)),
            img.crop((0, yc, xc, h)),
            img.crop((xc, yc, w, h)),
        ]
        saved = [save_crop(t, f"p{i+1}") for i, t in enumerate(tiles)]
        saved = [p for p in saved if p]
        if len(saved) in (3, 4):
            return saved

    return parts

def sha256_bytes(s: str) -> bytes:
    # matches source_url_hash BINARY(32)
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).digest()

# ----------------------------
# Database
# ----------------------------
def get_mysql_conn():
    import os
    import mysql.connector

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
    """
    Ensures:
      - group_part is None or int
      - group_count is None or int
    Also supports the case where someone accidentally put 'p1of2' into group_part.
    """
    gp = d.get("group_part")
    gc = d.get("group_count")

    # If group_part was accidentally set to a variant string like 'p1of2'
    if isinstance(gp, str):
        m = re.match(r"^p(\d+)of(\d+)$", gp.strip(), re.IGNORECASE)
        if m:
            d["group_part"] = int(m.group(1))
            # if group_count missing, infer it from the variant string
            if d.get("group_count") in (None, "", 0, "0"):
                d["group_count"] = int(m.group(2))
        else:
            # Not parseable -> NULL it (safe for INT column)
            d["group_part"] = None

    # If group_part is numeric string
    if isinstance(d.get("group_part"), str) and d["group_part"].isdigit():
        d["group_part"] = int(d["group_part"])

    # group_count numeric string -> int
    if isinstance(gc, str) and gc.isdigit():
        d["group_count"] = int(gc)

    # Any other weird type -> NULL it rather than breaking MySQL
    if d.get("group_part") is not None and not isinstance(d["group_part"], int):
        d["group_part"] = None
    if d.get("group_count") is not None and not isinstance(d["group_count"], int):
        d["group_count"] = None

    return d

def upsert_items(items) -> int:
    """
    Upsert into your postbeeld_stamps schema ONLY (no non-existent fields).
    Key: UNIQUE(source_url_hash, variant_key)
    Enforces INT types for group_part/group_count.
    """
    import os
    from dataclasses import asdict

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
            "source_url_hash": sha256_bytes(source_url),  # BINARY(32)

            "title": d.get("title"),
            "country": d.get("country"),
            "year": d.get("year"),
            "price_value": d.get("price_value"),
            "price_currency": d.get("price_currency"),
            "condition_text": d.get("condition_text"),
            "image_url": d.get("image_url"),
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

def scrape_pages(start_page: int, end_page: int, split_combos: bool = ENABLE_IMAGE_SPLITTING) -> Tuple[int, int]:
    """
    Scrape PostBeeld stamp pages and save to database.

    Args:
        start_page: First page number to scrape
        end_page: Last page number to scrape (inclusive)
        split_combos: Whether to split multi-stamp images into individual stamps.
                      Defaults to ENABLE_IMAGE_SPLITTING config value.

    Returns:
        Tuple of (upsert_count, images_downloaded)
    """
    ensure_dirs()
    session = get_session()

    all_items: List[StampItem] = []
    images_downloaded = 0

    print(f"[config] Image splitting is {'ENABLED' if split_combos else 'DISABLED'}")

    for page in range(start_page, end_page + 1):
        url = BASE_LIST_URL.format(page=page)
        print(f"[page {page}] fetching: {url}")

        html = fetch(url, session=session)
        debug_path = save_debug_html(page, html)

        items = parse_list_page(html)
        print(f"[page {page}] found {len(items)} items  DEBUG: saved {debug_path}")

        expanded: List[StampItem] = []

        for it in items:
            # Download image
            if it.image_url:
                p = download_image(session, it.image_url, IMAGES_DIR)
                if p:
                    # count as download only if new file just created
                    if not it.image_path and (not os.path.exists(p) or os.path.getsize(p) > 0):
                        images_downloaded += 1
                    it.image_path = p

            # Split combos if enabled and image exists
            if split_combos and it.image_path and os.path.exists(it.image_path):
                parts = split_combo_image(it.image_path)
                if parts:
                    it.is_group = 1
                    it.group_count = len(parts)
                    it.variant_key = "combo"

                    for idx, part_path in enumerate(parts, start=1):
                        part = StampItem(**asdict(it))
                        part.image_path = part_path
                        part.group_part = f"p{idx}of{len(parts)}"
                        part.variant_key = part.group_part
                        expanded.append(part)
                else:
                    expanded.append(it)
            else:
                # No splitting - keep image as-is
                expanded.append(it)

        all_items.extend(expanded)

        time.sleep(0.6)  # be polite

    print("About to upsert", len(all_items), "rows")
    upserts = upsert_items(all_items)
    return upserts, images_downloaded


def main():
    """
    Main entry point for the scraper.

    Usage:
        python scripts/scrape_postbeeld.py [start_page] [end_page]

    Examples:
        python scripts/scrape_postbeeld.py           # Scrape page 4 only
        python scripts/scrape_postbeeld.py 1 10      # Scrape pages 1-10

    Configuration:
        Set ENABLE_IMAGE_SPLITTING = True/False at the top of this file
        to control whether multi-stamp images are split into parts.
    """
    # Default: scrape page 4 (your failing case)
    start_page = 4
    end_page = 4

    # Usage: python scripts/scrape_postbeeld.py 1 10
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

    # Use the global ENABLE_IMAGE_SPLITTING config
    upserts, imgs = scrape_pages(start_page, end_page, split_combos=ENABLE_IMAGE_SPLITTING)
    print("Done.")
    print(f"Upserts: {upserts}. Images downloaded: {imgs}. Images folder: {IMAGES_DIR}")
    print(f"Image splitting was: {'ENABLED' if ENABLE_IMAGE_SPLITTING else 'DISABLED'}")


if __name__ == "__main__":
    main()
