import os
import re
import time
from decimal import Decimal, InvalidOperation
from urllib.parse import urljoin
import hashlib
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import mysql.connector
from dotenv import load_dotenv

import random
import requests

from PIL import Image
import math

load_dotenv()

BASE_URL = "https://www.postbeeld.com"
START_URL = "https://www.postbeeld.com/stamps"

HEADERS = {
    "User-Agent": "ObeliskStampsResearchBot/0.1 (contact: youremail@example.com)"
}

SLEEP_SECONDS = 1.2  # be polite

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Connection": "keep-alive",
})

def get_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

def safe_filename(text: str, max_len: int = 80) -> str:
    """Make a filesystem-safe filename chunk."""
    if not text:
        return "item"
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if text else "item"

def guess_ext_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return ".png"
    if path.endswith(".webp"):
        return ".webp"
    if path.endswith(".jpeg") or path.endswith(".jpg"):
        return ".jpg"
    return ".jpg"

def download_image(session: requests.Session, image_url: str, dest_dir: str, filename_hint: str = None) -> str | None:
    """
    Downloads image_url to dest_dir. Returns local filepath, or None on failure.
    Safe to rerun: if file exists, won't download again.
    """
    if not image_url:
        return None

    os.makedirs(dest_dir, exist_ok=True)

    # Stable, unique component from URL (prevents collisions)
    url_hash = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    ext = guess_ext_from_url(image_url)

    hint = safe_filename(filename_hint) if filename_hint else "stamp"
    filename = f"{hint}_{url_hash}{ext}"
    filepath = os.path.join(dest_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return filepath

    try:
        r = session.get(image_url, timeout=30, stream=True)
        r.raise_for_status()

        # Basic validation
        content_type = (r.headers.get("Content-Type") or "").lower()
        if "image" not in content_type:
            # Sometimes servers lie, but usually this means we got HTML/block page
            return None

        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)

        return filepath

    except Exception:
        # Don't crash the whole scrape because one image fails
        return None

def upsert_listing(conn, item: dict):
    sql = """
        INSERT INTO postbeeld_stamps
        (source_url, variant_key, group_key, group_part, group_count,
         title, country, year,
         price_value, price_currency, condition_text,
         image_url, image_path)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          group_key=VALUES(group_key),
          group_part=VALUES(group_part),
          group_count=VALUES(group_count),
          title=VALUES(title),
          country=VALUES(country),
          year=VALUES(year),
          price_value=VALUES(price_value),
          price_currency=VALUES(price_currency),
          condition_text=VALUES(condition_text),
          image_url=VALUES(image_url),
          image_path=VALUES(image_path),
          scraped_at=CURRENT_TIMESTAMP
    """
    cur = conn.cursor()
    cur.execute(
        sql,
        (
            item.get("source_url"),
            item.get("variant_key", "single"),
            item.get("group_key"),
            item.get("group_part"),
            item.get("group_count"),
            item.get("title"),
            item.get("country"),
            item.get("year"),
            item.get("price_value"),
            item.get("price_currency"),
            item.get("condition_text"),
            item.get("image_url"),
            item.get("image_path"),
        ),
    )
    conn.commit()
    cur.close()

def parse_price(text: str):
    """
    Examples you might see: "€140.00", "€ 3.00"
    Returns (Decimal price, 'EUR') or (None, None)
    """
    if not text:
        return None, None

    t = text.strip()
    currency = None

    if "€" in t:
        currency = "EUR"
        t = t.replace("€", "").strip()

    # remove non-numeric except dot/comma
    t = re.sub(r"[^\d,\.]", "", t)

    # handle comma decimals if present
    if t.count(",") == 1 and t.count(".") == 0:
        t = t.replace(",", ".")

    try:
        return Decimal(t), currency
    except (InvalidOperation, ValueError):
        return None, currency

def extract_country_year_from_meta(meta_text: str):
    """
    PostBeeld cards often include something like:
      "Belgium, 1849" or "France, 1849"
    """
    if not meta_text:
        return None, None
    parts = [p.strip() for p in meta_text.split(",")]
    if len(parts) >= 2:
        country = parts[-2]
        year_match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", parts[-1])
        year = int(year_match.group(1)) if year_match else None
        return country, year
    return None, None

def split_combo_image(image_path: str, out_dir: str, base_name: str) -> list[dict]:
    """
    If an image appears to contain multiple stamps on a white background,
    split it into multiple crops and return metadata for each crop.

    Returns a list like:
      [{"path": "..._p1of2.jpg", "variant_key": "p1of2", "group_part": 1, "group_count": 2}, ...]
    If no split is detected, returns:
      [{"path": image_path, "variant_key": "single", "group_part": None, "group_count": None}]
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return [{"path": image_path, "variant_key": "single", "group_part": None, "group_count": None}]

    w, h = img.size
    # Heuristic: if it's not wide enough, likely a single stamp
    if w < int(1.15 * h):
        return [{"path": image_path, "variant_key": "single", "group_part": None, "group_count": None}]

    # Build a "non-white" mask by scanning columns
    # (Postbeeld images are generally stamp(s) on white background with watermark)
    px = img.load()

    # Count "ink" per column
    col_ink = [0] * w
    for x in range(w):
        ink = 0
        for y in range(h):
            r, g, b = px[x, y]
            # Treat near-white as background
            if (r < 245) or (g < 245) or (b < 245):
                ink += 1
        col_ink[x] = ink

    max_ink = max(col_ink) if col_ink else 0
    if max_ink == 0:
        return [{"path": image_path, "variant_key": "single", "group_part": None, "group_count": None}]

    # Find "valleys" (columns with very low ink) that could be separators between stamps
    # Threshold is a fraction of max ink
    valley_threshold = max(10, int(0.05 * max_ink))
    valleys = []
    run_start = None
    for x in range(w):
        if col_ink[x] <= valley_threshold:
            if run_start is None:
                run_start = x
        else:
            if run_start is not None:
                valleys.append((run_start, x - 1))
                run_start = None
    if run_start is not None:
        valleys.append((run_start, w - 1))

    # Keep only valleys that are "wide enough" to be a real gap
    valleys = [(a, b) for (a, b) in valleys if (b - a + 1) >= max(8, w // 80)]

    if not valleys:
        return [{"path": image_path, "variant_key": "single", "group_part": None, "group_count": None}]

    # Convert valleys into cut positions (use midpoint of each valley)
    cut_positions = []
    for (a, b) in valleys:
        mid = (a + b) // 2
        # avoid cuts too close to edges
        if mid > int(0.12 * w) and mid < int(0.88 * w):
            cut_positions.append(mid)

    # Deduplicate / sort
    cut_positions = sorted(set(cut_positions))

    # If we have too many cuts (watermark noise), pick the strongest few by valley width
    if len(cut_positions) > 5:
        valley_by_width = sorted(valleys, key=lambda t: (t[1] - t[0]), reverse=True)
        cut_positions = sorted(set([(a + b) // 2 for (a, b) in valley_by_width[:5]]))

    # Build segments from cuts
    bounds = [0] + cut_positions + [w]
    segments = []
    for i in range(len(bounds) - 1):
        x1, x2 = bounds[i], bounds[i + 1]
        seg_w = x2 - x1
        # Reject tiny slices
        if seg_w < int(0.20 * h):
            continue
        segments.append((x1, 0, x2, h))

    # If we ended up with just 1 meaningful segment, don't split
    if len(segments) <= 1:
        return [{"path": image_path, "variant_key": "single", "group_part": None, "group_count": None}]

    os.makedirs(out_dir, exist_ok=True)

    out = []
    n = len(segments)
    for idx, box in enumerate(segments, start=1):
        crop = img.crop(box)
        out_path = os.path.join(out_dir, f"{base_name}_p{idx}of{n}.jpg")
        crop.save(out_path, quality=95)
        out.append(
            {
                "path": out_path,
                "variant_key": f"p{idx}of{n}",
                "group_part": idx,
                "group_count": n,
            }
        )

    return out

def parse_list_page(html: str, session: requests.Session, image_dir: str):
    """
    Parse one PostBeeld listing page and return a list of dicts.

    Decisions:
    - If multiple conditions are listed (multiple rows), we store ONLY the first row.
    - If the image seems to contain multiple stamps, we split and store multiple rows
      linked by group_key/group_part/group_count.
    """
    soup = BeautifulSoup(html, "html.parser")
    items = []

    cards = soup.select("div.product-in-grid")
    for card in cards:
        # URL + Title
        a = card.select_one("a.grid-product-name[href]")
        if not a:
            continue
        source_url = a.get("href", "").strip()
        title = a.get_text(" ", strip=True)
        if not source_url or not title:
            continue

        # Exclude letter/postal-history style listings if you want
        title_upper = title.upper()
        if "FOLD" in title_upper or "LETTER" in title_upper:
            continue

        # SKU (for filenames)
        sku = None
        sku_meta = card.select_one('meta[itemprop="sku"]')
        if sku_meta and sku_meta.get("content"):
            sku = sku_meta["content"].strip()

        # Image URL
        image_url = None
        img_link = card.select_one('link[itemprop="image"][href], link[itemprop="image"][content]')
        if img_link:
            image_url = (img_link.get("href") or img_link.get("content") or "").strip()

        # Country/year from meta line like: "sde0676, Denmark, 1979"
        meta_text = None
        meta_div = card.select_one(".grid-product-meta")
        if meta_div:
            meta_text = meta_div.get_text(" ", strip=True)
        country, year = extract_country_year_from_meta(meta_text or "")

        # FIRST condition row only
        condition_text = None
        price_value = None
        price_currency = None

        first_row = card.select_one("tr.condition_row")
        if first_row:
            # condition
            btn = first_row.select_one("button.condition_button")
            if btn:
                condition_text = btn.get_text(" ", strip=True)

            # price (first span.price)
            pspan = first_row.select_one("span.price")
            if pspan:
                price_value, price_currency = parse_price(pspan.get_text(" ", strip=True))

        # Download image (one file)
        image_path = None
        if image_url:
            filename_hint = sku or title
            image_path = download_image(session=session, image_url=image_url, dest_dir=image_dir, filename_hint=filename_hint)

        # If we have an image, attempt to split it into multiple stamp crops
        crops = []
        if image_path:
            # stable group key for this listing
            group_key = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:16]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            crops = split_combo_image(image_path=image_path, out_dir=image_dir, base_name=base_name)

            # If it split into multiple parts, attach group metadata
            if len(crops) > 1:
                for c in crops:
                    items.append(
                        {
                            "source_url": source_url,
                            "variant_key": c["variant_key"],
                            "group_key": group_key,
                            "group_part": c["group_part"],
                            "group_count": c["group_count"],
                            "title": title,
                            "country": country,
                            "year": year,
                            "price_value": price_value,
                            "price_currency": price_currency or "EUR",
                            "condition_text": condition_text,
                            "image_url": image_url,
                            "image_path": c["path"],
                        }
                    )
                continue  # done with this card

        # Normal single record (no split)
        items.append(
            {
                "source_url": source_url,
                "variant_key": "single",
                "group_key": None,
                "group_part": None,
                "group_count": None,
                "title": title,
                "country": country,
                "year": year,
                "price_value": price_value,
                "price_currency": price_currency or "EUR",
                "condition_text": condition_text,
                "image_url": image_url,
                "image_path": image_path,
            }
        )

    return items

def fetch(url: str, max_retries: int = 8) -> str:
    """
    Fetch a URL with polite retry/backoff.
    Handles 429 by respecting Retry-After when present.
    """
    backoff = 2.0  # seconds

    for attempt in range(1, max_retries + 1):
        r = SESSION.get(url, timeout=30)

        # Success
        if r.status_code == 200:
            return r.text

        # Rate limited
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_for = float(retry_after)
            else:
                # exponential backoff with jitter
                sleep_for = backoff + random.uniform(0.0, 1.5)
                backoff = min(backoff * 1.8, 120.0)  # cap at 2 minutes

            print(f"  429 rate-limited. Sleeping {sleep_for:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_for)
            continue

        # Other errors: retry a bit (transient), then fail
        if 500 <= r.status_code < 600:
            sleep_for = backoff + random.uniform(0.0, 1.5)
            backoff = min(backoff * 1.6, 60.0)
            print(f"  {r.status_code} server error. Sleeping {sleep_for:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_for)
            continue

        # Non-retryable (403, 404, etc.)
        r.raise_for_status()

    raise requests.HTTPError(f"Failed after {max_retries} retries (likely rate limited): {url}")

def main():
    conn = get_db()

    start_page = 4#631
    end_page = 4#650  # START SMALL to test; raise later (2030/12176/etc.)
    base_sleep_seconds = 2.0

    image_dir = os.path.join("data", "postbeeld", "images")
    os.makedirs(image_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Connection": "keep-alive",
    })

    total_upserts = 0
    total_images = 0

    try:
        for page in range(start_page, end_page + 1):
            url = f"{BASE_URL}/stamps/page/{page}/mode/grid/show/120"
            print(f"[page {page}] fetching: {url}")

            html = fetch(url)  # keep your existing fetch/backoff logic

            items = parse_list_page(html, session=session, image_dir=image_dir)
            print(f"  found {len(items)} items")

            if len(items) == 0:
                os.makedirs("debug", exist_ok=True)
                debug_path = f"debug/postbeeld_page_{page}.html"
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"  DEBUG: saved {debug_path} (0 items)")
                break

            for item in items:
                upsert_listing(conn, item)
                total_upserts += 1
                if item.get("image_path"):
                    total_images += 1

            print(f"  total upserts: {total_upserts} | images downloaded: {total_images}")
            time.sleep(base_sleep_seconds)

        print(f"Done. Upserts: {total_upserts}. Images downloaded: {total_images}.")
        print(f"Images folder: {image_dir}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
    #app.run(debug=True)
    #client.parse_request_body_response(json.dumps(token_response.json()))
    #userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    #uri, headers, body = client.add_token(userinfo_endpoint)
    #userinfo_response = requests.get(uri, headers=headers, data=body)
    #user_info = userinfo_response.json()
    #username = user_info["email"].split("@")[0]
    #email = user_info["email"]
    #name = user_info["name"]
