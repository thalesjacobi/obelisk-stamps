"""
Main Flask app for Obelisk Stamps.
# deploy: refresh FB_PAGE_ACCESS_TOKEN

This application calls a separate ML API service for stamp recognition.
The ML API handles model loading and inference, keeping the website lightweight.
"""

import os
import re
import json
import time
import base64
import random
import threading
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from functools import wraps
from datetime import datetime, timedelta

import mysql.connector
import requests as http_requests
import stripe
from mysql.connector import Error
from oauthlib.oauth2 import WebApplicationClient

from dotenv import load_dotenv
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, \
    send_from_directory, session, stream_with_context, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from werkzeug.middleware.proxy_fix import ProxyFix

# ------------------------------------------------------------
# ENV + APP
# ------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
# Fix for running behind Cloud Run's load balancer - ensures url_for generates https:// URLs
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Custom domain configuration
SITE_URL = os.getenv("SITE_URL", "").rstrip("/")  # e.g. https://obelisk-stamps.com
app.config["PREFERRED_URL_SCHEME"] = "https"

# ------------------------------------------------------------
# ML API Configuration
# ------------------------------------------------------------
ML_API_URL = os.getenv("ML_API_URL", "http://localhost:8081")

# ------------------------------------------------------------
# AI CHATBOT (OpenAI + ChromaDB RAG)
# ------------------------------------------------------------
try:
    import openai as _openai_module
except ImportError:
    _openai_module = None
    print("WARNING: openai package not installed — chatbot disabled")

try:
    import chromadb as _chromadb_module
except ImportError:
    _chromadb_module = None
    print("WARNING: chromadb package not installed — chatbot KB disabled")

_openai_client = None
_chroma_collection = None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY and _openai_module:
    try:
        _openai_client = _openai_module.OpenAI(api_key=OPENAI_API_KEY.strip())
        print("Chatbot: OpenAI client initialised")
    except Exception as e:
        print(f"WARNING: Failed to init OpenAI client: {e}")
else:
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set — chatbot disabled")

# Style suffix appended to every DALL-E 3 prompt in the Instagram carousel
CAROUSEL_STYLE_SUFFIX = (
    "Style: vintage editorial illustration, reminiscent of classic postage-stamp engravings "
    "and collector's print ephemera. Rich, deep colour palette — navy blue, burgundy, warm "
    "gold and ivory — with fine-line detail and a tactile, handcrafted quality. Strong focal "
    "composition with a sense of heritage and craftsmanship. "
    "Square 1:1 format for Instagram. No text, lettering, numbers, or watermarks in the image."
)

# Default prompt for Cinemagraph generation (stored in site_settings, editable by admins)
_CINE_DEFAULT_PROMPT = (
    "Subtle gentle atmospheric motion, cinematic still life, "
    "minimal movement, soft breathing effect, loop-friendly"
)

# Default OpenAI prompt for Instagram caption generation (stored in site_settings, editable by admins)
_IG_CAPTION_DEFAULT_PROMPT = (
    "You are a social media copywriter for an online store called Obelisk Stamps "
    "that sells handcrafted framed stamp displays. Write an Instagram caption for "
    "a post about the article described below. The caption should convey an inviting "
    "idea of the article's content and encourage readers to visit the link. Include "
    "relevant hashtags at the end. Keep the total caption under 2000 characters. "
    "Return ONLY the caption text, no extra commentary."
)

# --- Luma AI client (image-to-video, used by cinemagraph worker) ---
try:
    from lumaai import LumaAI as _LumaAI
except ImportError:
    _LumaAI = None

LUMAAI_API_KEY = os.getenv("LUMAAI_API_KEY", "")
_luma_client = None
_luma_enabled = False

if LUMAAI_API_KEY and _LumaAI:
    try:
        _luma_client = _LumaAI(auth_token=LUMAAI_API_KEY)
        _luma_enabled = True
        print("Luma AI enabled")
    except Exception as e:
        print(f"WARNING: Luma AI init failed: {e}")
else:
    if not LUMAAI_API_KEY:
        print("INFO: LUMAAI_API_KEY not set — Luma cinemagraph generation disabled")


def _is_luma_billing_error(exc):
    """Non-transient Luma errors — stop immediately instead of retrying."""
    msg_lower = str(exc).lower()
    return any(phrase in msg_lower for phrase in [
        "insufficient", "credits", "billing", "quota", "balance",
        "permissiondenied", "403",
    ])


def _luma_create_task(image_url, prompt_text):
    """Submit an image-to-video task to Luma. Returns generation ID."""
    gen = _luma_client.generations.create(
        prompt=prompt_text or "gentle subtle motion",
        model="ray-2",
        keyframes={"frame0": {"type": "image", "url": image_url}},
        aspect_ratio="1:1",
        duration="5s",
    )
    return gen.id


def _luma_poll_task(gen_id):
    """Poll a Luma generation. Returns (state, mp4_url_or_None, failure_reason)."""
    gen = _luma_client.generations.get(id=gen_id)
    state = gen.state  # "queued", "dreaming", "completed", "failed"
    mp4_url = gen.assets.video if state == "completed" and gen.assets else None
    reason = gen.failure_reason if state == "failed" else ""
    return state, mp4_url, reason or ""


# --- Google Cloud Storage (optional — for persistent file hosting) ---
_gcs_bucket = None
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
GCS_PATH_PREFIX = os.getenv("GCS_PATH_PREFIX", "").strip("/")

if GCS_BUCKET_NAME:
    try:
        from google.cloud import storage as _gcs_storage
        _gcs_client = _gcs_storage.Client()
        _gcs_bucket = _gcs_client.bucket(GCS_BUCKET_NAME)
        print(f"GCS: Connected to bucket '{GCS_BUCKET_NAME}'")
    except Exception as _e:
        print(f"WARNING: GCS init failed: {_e} — falling back to local storage")
        _gcs_bucket = None
else:
    print("GCS: GCS_BUCKET_NAME not set — using local storage")

# --- Instagram Graph API (optional — for carousel posting) ---
IG_USER_ID      = os.getenv("IG_USER_ID", "")
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN", "")
_IG_GRAPH_URL   = "https://graph.facebook.com/v21.0"
_FB_VIDEO_URL   = "https://graph-video.facebook.com/v21.0"
if IG_USER_ID and IG_ACCESS_TOKEN:
    print("Instagram: credentials configured")
else:
    print("Instagram: IG_USER_ID / IG_ACCESS_TOKEN not set — posting disabled")

# --- Facebook Page API (optional — for Page posting) ---
FB_PAGE_ID           = os.getenv("FB_PAGE_ID", "")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "")
if FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN:
    print("Facebook: Page credentials configured")
else:
    print("Facebook: FB_PAGE_ID / FB_PAGE_ACCESS_TOKEN not set — posting disabled")

# --- Knowledge Base (ChromaDB) ---
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")


def _chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def _load_knowledge_base():
    """Load all .txt files from kb/ into ChromaDB with OpenAI embeddings."""
    global _chroma_collection
    if not _chromadb_module or not _openai_client:
        return

    try:
        chroma_client = _chromadb_module.Client()
        # Delete and recreate to pick up any file changes
        try:
            chroma_client.delete_collection("obelisk_kb")
        except Exception:
            pass
        _chroma_collection = chroma_client.create_collection(
            name="obelisk_kb",
            metadata={"hnsw:space": "cosine"},
        )

        if not os.path.isdir(KB_DIR):
            print(f"Chatbot: No kb/ directory found at {KB_DIR}")
            return

        documents = []
        metadatas = []
        ids = []
        idx = 0

        for filename in sorted(os.listdir(KB_DIR)):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(KB_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                continue
            chunks = _chunk_text(content)
            for chunk in chunks:
                documents.append(chunk)
                metadatas.append({"source": filename})
                ids.append(f"chunk_{idx}")
                idx += 1

        if not documents:
            print("Chatbot: No knowledge base documents found")
            return

        # Embed all chunks using OpenAI
        embed_response = _openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=documents,
        )
        embeddings = [item.embedding for item in embed_response.data]

        _chroma_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Chatbot: Loaded {len(documents)} knowledge base chunks from {idx and len(set(m['source'] for m in metadatas))} files")
    except Exception as e:
        print(f"WARNING: Failed to load knowledge base: {e}")
        _chroma_collection = None


def _search_knowledge(query, n_results=3):
    """Search the knowledge base for relevant chunks."""
    if not _chroma_collection or not _openai_client:
        return []
    try:
        embed_response = _openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
        )
        query_embedding = embed_response.data[0].embedding
        results = _chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"[Chatbot] KB search error: {e}")
        return []


# Load knowledge base at startup
_load_knowledge_base()

CHATBOT_SYSTEM_PROMPT = """You are the Obelisk Stamps assistant, a friendly and knowledgeable expert in stamps and philately, working for a UK-based e-commerce site that sells handcrafted framed stamp displays.

You have access to the company's business information provided as context below. Use it to answer questions about delivery, returns, payments, and products accurately.

For questions about stamp history, values, collecting tips, and philately in general, use your own knowledge.

Guidelines:
- Be concise — keep responses under 3 short paragraphs
- Be warm, enthusiastic, and encouraging to collectors of all levels
- When relevant, mention that Obelisk Stamps offers curated framed stamp displays as gifts or collection pieces
- If asked about specific product pricing or availability, suggest the user check the catalogue at obelisk-stamps.com
- Do not provide financial investment advice — you can discuss historical trends but always note that values can fluctuate
- If a question is completely unrelated to stamps, philately, or Obelisk Stamps, politely redirect the conversation back to stamps
- Always answer based on the provided context when the question is about business policies"""

# In-memory rate limiting for chatbot
_chat_rate_limits = {}
CHAT_RATE_LIMIT = 10   # max messages per window
CHAT_RATE_WINDOW = 60   # window in seconds


def _check_chat_rate_limit(ip_address):
    """Return True if the IP is within rate limits, False if exceeded."""
    now = time.time()
    if ip_address not in _chat_rate_limits:
        _chat_rate_limits[ip_address] = []
    _chat_rate_limits[ip_address] = [
        ts for ts in _chat_rate_limits[ip_address]
        if now - ts < CHAT_RATE_WINDOW
    ]
    if len(_chat_rate_limits[ip_address]) >= CHAT_RATE_LIMIT:
        return False
    _chat_rate_limits[ip_address].append(now)
    return True


# ------------------------------------------------------------
# MAIL (Gmail SMTP with App Password)
# ------------------------------------------------------------
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", "587"))
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")  # Your Gmail address
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")  # Gmail App Password (not regular password)
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")

mail = Mail(app)

# ------------------------------------------------------------
# STRIPE PAYMENTS
# ------------------------------------------------------------
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# ------------------------------------------------------------
# GOOGLE OAUTH
# ------------------------------------------------------------
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
client = WebApplicationClient(GOOGLE_CLIENT_ID) if GOOGLE_CLIENT_ID else None

# ------------------------------------------------------------
# YOUTUBE OAUTH
# ------------------------------------------------------------
YT_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
YT_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
YT_REDIRECT_URI  = os.getenv("SITE_URL", "").rstrip("/") + "/admin/youtube-oauth-callback"
YT_SCOPES        = ["https://www.googleapis.com/auth/youtube.upload"]

# ------------------------------------------------------------
# PATHS + UPLOADS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# GCS HELPERS
# ------------------------------------------------------------
def _gcs_object_name(name):
    """Prepend the optional GCS_PATH_PREFIX to an object name."""
    return f"{GCS_PATH_PREFIX}/{name}" if GCS_PATH_PREFIX else name


def upload_to_gcs(local_path, object_name, content_type=None):
    """Upload a local file to GCS. Returns public URL or None if GCS is disabled/fails."""
    if not _gcs_bucket:
        return None
    try:
        blob = _gcs_bucket.blob(_gcs_object_name(object_name))
        blob.upload_from_filename(str(local_path), content_type=content_type)
        url = blob.public_url
        print(f"GCS: Uploaded {object_name} → {url}")
        return url
    except Exception as e:
        import traceback
        print(f"ERROR: GCS upload failed for {object_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def upload_bytes_to_gcs(data, object_name, content_type=None):
    """Upload bytes to GCS. Returns public URL or None if GCS is disabled/fails."""
    if not _gcs_bucket:
        return None
    try:
        blob = _gcs_bucket.blob(_gcs_object_name(object_name))
        blob.upload_from_string(data, content_type=content_type or "application/octet-stream")
        url = blob.public_url
        print(f"GCS: Uploaded {object_name} → {url}")
        return url
    except Exception as e:
        import traceback
        print(f"ERROR: GCS upload failed for {object_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def resolve_image_to_local_path(img_url):
    """
    Given an image URL from the DB (GCS URL, relative path, or /static/ path),
    return a local Path to the file — downloading from GCS if needed.
    Returns None if the file cannot be resolved.
    """
    if not img_url:
        return None

    # Full GCS URL → download to local cache
    if img_url.startswith("https://storage.googleapis.com/"):
        # Extract object path after the bucket name
        marker = f"/{GCS_BUCKET_NAME}/"
        idx = img_url.find(marker)
        if idx >= 0:
            gcs_name = img_url[idx + len(marker):]
            # Strip prefix if present
            prefix = GCS_PATH_PREFIX + "/" if GCS_PATH_PREFIX else ""
            if prefix and gcs_name.startswith(prefix):
                gcs_name = gcs_name[len(prefix):]
            local_path = Path("static") / gcs_name
            if local_path.exists():
                return local_path
            # Download from GCS
            if _gcs_bucket:
                try:
                    full_name = _gcs_object_name(gcs_name)
                    blob = _gcs_bucket.blob(full_name)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(local_path))
                    return local_path
                except Exception as e:
                    print(f"WARNING: GCS download failed for {gcs_name}: {e}")
        return None

    # Local path resolution (existing pattern)
    u = img_url.lstrip("/")
    img_path = Path(u) if u.startswith("static/") else Path("static") / u
    return img_path if img_path.exists() else None


# ------------------------------------------------------------
# CAROUSEL INSTAGRAM COMPOSITING
# ------------------------------------------------------------
_FONT_PATH = Path("static/fonts/Roboto-Bold.ttf")


def compose_carousel_slide(img_bytes, punchline, slide_index, total_slides, band_top=None):
    """
    Composite a dark gradient + punchline text + optional swipe hint onto a
    carousel image.  Returns JPEG bytes.  The original img_bytes are never
    modified — this is only called when posting to Instagram (or for preview).

    Args:
        img_bytes:    Raw bytes of the original image (PNG or JPEG).
        punchline:    Caption text (uppercased automatically).
        slide_index:  0-based index of this slide in the carousel.
        total_slides: Total number of slides — swipe hint hidden on last slide.
        band_top:     If set, use a solid dark band (same style as cinemagraph
                      overlay) starting at this Y. Used in cinemagraph carousels
                      for visual consistency with video slides.
    """
    from PIL import Image, ImageDraw, ImageFont
    import io

    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = img.size

    if band_top is not None:
        # ── Solid-band mode (matches cinemagraph overlay style) ────────────
        band_alpha = 235
        fade_h     = int(h * 0.06)
        band_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_b     = ImageDraw.Draw(band_layer)
        for y in range(fade_h):
            fy = band_top - fade_h + y
            if 0 <= fy < h:
                alpha = int(band_alpha * (y / fade_h))
                draw_b.line([(0, fy), (w, fy)], fill=(0, 0, 0, alpha))
        draw_b.rectangle([(0, max(0, band_top)), (w, h)], fill=(0, 0, 0, band_alpha))
        img = Image.alpha_composite(img, band_layer)

        # Use same font sizing as cinemagraph overlay for consistency
        try:
            font_main = ImageFont.truetype(str(_FONT_PATH), size=int(w * 0.048))
            font_hint = ImageFont.truetype(str(_FONT_PATH), size=int(w * 0.021))
        except Exception:
            font_main = ImageFont.load_default(size=32)
            font_hint = ImageFont.load_default(size=20)
        line_h_mult   = 0.88
        bottom_pad    = 0.015
    else:
        # ── Gradient mode (original static carousel style) ─────────────────
        gradient_h = int(h * 0.55)
        gradient   = Image.new("RGBA", (w, gradient_h), (0, 0, 0, 0))
        draw_g     = ImageDraw.Draw(gradient)
        for y in range(gradient_h):
            alpha = int(255 * (y / gradient_h) ** 1.0)
            draw_g.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
        img.paste(gradient, (0, h - gradient_h), gradient)

        try:
            font_main = ImageFont.truetype(str(_FONT_PATH), size=int(w * 0.092))
            font_hint = ImageFont.truetype(str(_FONT_PATH), size=int(w * 0.021))
        except Exception:
            font_main = ImageFont.load_default(size=40)
            font_hint = ImageFont.load_default(size=20)
        line_h_mult   = 1.05
        bottom_pad    = 0.04

    draw  = ImageDraw.Draw(img)
    pad_x = int(w * 0.06)

    # ── Punchline text (white, uppercase, left-aligned, word-wrapped) ───────
    text  = (punchline or "").upper()
    max_w = w - pad_x * 2
    words = text.split()
    lines, line = [], ""
    for word in words:
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=font_main) <= max_w:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    line_h       = int(font_main.size * line_h_mult)
    hint_h       = int(font_hint.size * 2.4)
    total_text_h = len(lines) * line_h + hint_h
    text_y       = h - total_text_h - int(h * bottom_pad)

    for line_text in lines:
        draw.text((pad_x, text_y), line_text, font=font_main,
                  fill=(255, 255, 255, 255))
        text_y += line_h

    # ── Swipe hint (yellow, all slides except the last) ─────────────────────
    if slide_index < total_slides - 1:
        hint_text = "Slide right for more »»»"
        hint_y    = h - hint_h + int(font_hint.size * 0.4)
        draw.text((pad_x, hint_y), hint_text, font=font_hint,
                  fill=(255, 215, 0, 255))

    # ── Return as JPEG bytes ─────────────────────────────────────────────────
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


def _compute_max_overlay_band_top(punchlines, width=960, height=960):
    """Pre-compute the band_top Y that fits the tallest punchline in the set.

    Call this BEFORE the per-slide loop and pass the result as band_top=
    to _make_cinemagraph_overlay_png / compose_carousel_slide so every
    slide in the carousel has the same dark-band height.
    """
    from PIL import ImageFont, ImageDraw, Image

    w, h = width, height
    try:
        font_main = ImageFont.truetype(str(_FONT_PATH), size=int(w * 0.048))
        font_hint = ImageFont.truetype(str(_FONT_PATH), size=int(w * 0.021))
    except Exception:
        font_main = ImageFont.load_default(size=32)
        font_hint = ImageFont.load_default(size=20)

    pad_x = int(w * 0.06)
    max_w = w - pad_x * 2
    line_h = int(font_main.size * 0.88)
    hint_h = int(font_hint.size * 2.4)

    tmp  = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(tmp)

    max_text_h = 0
    for p in (punchlines or []):
        text  = (p or "").upper()
        words = text.split()
        lines, line = [], ""
        for word in words:
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=font_main) <= max_w:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        text_h = len(lines) * line_h + hint_h
        if text_h > max_text_h:
            max_text_h = text_h

    text_y   = h - max_text_h - int(h * 0.015)
    band_top = text_y - int(h * 0.03)
    return band_top


def _make_cinemagraph_overlay_png(punchline, slide_index, total_slides, width=960, height=960, band_top=None):
    """
    Create a transparent RGBA PNG with a dark gradient + punchline text +
    optional swipe hint — identical visual style to compose_carousel_slide()
    but on a fully transparent canvas so it can be blended over a video.

    Args:
        punchline:    Caption text (uppercased automatically).
        slide_index:  0-based index of this slide.
        total_slides: Total slides — swipe hint hidden on last slide.
        width/height: Video frame dimensions (default 960x960).

    Returns PNG bytes.
    """
    from PIL import Image, ImageDraw, ImageFont
    import io as _io

    w, h = width, height
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # fully transparent

    # ── Font loading ─────────────────────────────────────────────────────────
    try:
        font_main = ImageFont.truetype(str(_FONT_PATH), size=int(w * 0.048))
        font_hint = ImageFont.truetype(str(_FONT_PATH), size=int(w * 0.021))
    except Exception:
        font_main = ImageFont.load_default(size=32)
        font_hint = ImageFont.load_default(size=20)

    draw  = ImageDraw.Draw(overlay)
    pad_x = int(w * 0.06)

    # ── Word-wrap punchline text ─────────────────────────────────────────────
    text  = (punchline or "").upper()
    max_w = w - pad_x * 2
    words = text.split()
    lines, line = [], ""
    for word in words:
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=font_main) <= max_w:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    line_h       = int(font_main.size * 0.88)
    hint_h       = int(font_hint.size * 2.4)
    total_text_h = len(lines) * line_h + hint_h
    text_y       = h - total_text_h - int(h * 0.015)

    # ── Dark band: solid from text top to bottom, fade above ─────────────────
    if band_top is None:
        band_top = text_y - int(h * 0.03)     # auto: fit this slide's text
    band_alpha = 235                           # ~92% opacity
    fade_h     = int(h * 0.06)                 # smooth transition zone

    # Short gradient fade above the solid band
    for y in range(fade_h):
        fy    = band_top - fade_h + y
        if 0 <= fy < h:
            alpha = int(band_alpha * (y / fade_h))
            draw.line([(0, fy), (w, fy)], fill=(0, 0, 0, alpha))

    # Solid dark rectangle from band_top to very bottom
    draw.rectangle([(0, max(0, band_top)), (w, h)], fill=(0, 0, 0, band_alpha))

    # ── Draw punchline text (white, uppercase, left-aligned) ─────────────────
    ty = text_y
    for line_text in lines:
        draw.text((pad_x, ty), line_text, font=font_main,
                  fill=(255, 255, 255, 255))
        ty += line_h

    # ── Swipe hint (yellow, all slides except the last) ──────────────────────
    if slide_index < total_slides - 1:
        hint_text = "Slide right for more \u00bb\u00bb\u00bb"
        hint_y    = h - hint_h + int(font_hint.size * 0.4)
        draw.text((pad_x, hint_y), hint_text, font=font_hint,
                  fill=(255, 215, 0, 255))

    out = _io.BytesIO()
    overlay.save(out, format="PNG")
    return out.getvalue()


def _apply_overlay_to_video(video_url_or_path, overlay_png_bytes, output_path):
    """
    Blend a transparent overlay PNG onto all frames of a video using FFmpeg.
    Downloads the video first if a URL is given.

    Args:
        video_url_or_path: URL string or Path to the raw MP4.
        overlay_png_bytes: PNG bytes from _make_cinemagraph_overlay_png().
        output_path:       Path where the composited MP4 is written.

    Returns True on success, False on any error.
    """
    import subprocess, tempfile, shutil
    import requests as _req

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("_apply_overlay_to_video: imageio-ffmpeg not installed", flush=True)
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_files = []

    try:
        # ── Resolve video to a local file ────────────────────────────────────
        video_url_or_path = str(video_url_or_path)
        if video_url_or_path.startswith("http"):
            tmp_vid = output_path.with_suffix(".input.mp4")
            tmp_files.append(tmp_vid)
            r = _req.get(video_url_or_path, timeout=60, stream=True)
            r.raise_for_status()
            with open(tmp_vid, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            video_input = str(tmp_vid)
        else:
            video_input = video_url_or_path

        # ── Write overlay PNG to a temp file ─────────────────────────────────
        tmp_png = output_path.with_suffix(".overlay.png")
        tmp_files.append(tmp_png)
        tmp_png.write_bytes(overlay_png_bytes)

        # ── FFmpeg blend ─────────────────────────────────────────────────────
        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_input,
            "-i", str(tmp_png),
            "-filter_complex", "[0:v]scale=960:960[vid];[vid][1:v]overlay=0:0",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-an",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"_apply_overlay_to_video: FFmpeg error: "
                  f"{result.stderr.decode(errors='replace')[-500:]}", flush=True)
            return False
        return True

    except Exception as e:
        print(f"_apply_overlay_to_video: {e}", flush=True)
        return False

    finally:
        for f in tmp_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


# ------------------------------------------------------------
# DATABASE HELPERS
# ------------------------------------------------------------
def get_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


def query_one(sql, params=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(sql, params or ())
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def query_all(sql, params=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(sql, params or ())
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def execute(sql, params=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(sql, params or ())
    conn.commit()
    last_id = cur.lastrowid
    cur.close()
    conn.close()
    return last_id


def get_setting(key, default=None):
    """Return a value from site_settings, or default if not found."""
    row = query_one("SELECT value FROM site_settings WHERE `key` = %s", (key,))
    return row[0] if row else default


def init_articles_table():
    """Create the articles table if it doesn't exist."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            slug VARCHAR(255) UNIQUE NOT NULL,
            title VARCHAR(500) NOT NULL,
            subtitle VARCHAR(500),
            content LONGTEXT,
            excerpt VARCHAR(1000),
            image_url VARCHAR(1000),
            is_published BOOLEAN DEFAULT FALSE,
            published_at DATETIME NULL,
            carousel_prompts TEXT,
            carousel_images TEXT,
            carousel_punchlines TEXT,
            carousel_style TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_slug (slug),
            INDEX idx_published (is_published, published_at DESC)
        )
    """)
    # Migration: add carousel + video columns to existing tables
    for col in ("carousel_prompts TEXT", "carousel_images TEXT",
                "carousel_punchlines TEXT", "carousel_style TEXT",
                "video_narrated_url TEXT", "video_narrated_script TEXT",
                "video_ai_url TEXT", "video_narrated_runs TEXT",
                "video_ai_status TEXT", "video_narrated_status TEXT",
                "carousel_cinemagraphs TEXT",
                "carousel_cinemagraph_log TEXT",
                "carousel_cinemagraph_prompts TEXT",
                "carousel_cinemagraph_archived TEXT",
                "video_narrated_log TEXT",
                "video_ai_log TEXT",
                "carousel_created_at TEXT",
                "carousel_archived_meta TEXT",
                "carousel_cinemagraph_created_at TEXT",
                "show_slideshow BOOLEAN DEFAULT FALSE"):
        try:
            cur.execute(f"ALTER TABLE articles ADD COLUMN {col}")
        except Exception:
            pass  # column already exists
    conn.commit()
    cur.close()
    conn.close()

try:
    init_articles_table()
except Exception as e:
    print(f"WARNING: Could not initialise articles table: {e}")


def init_site_settings():
    """Create site_settings table and seed default values if missing."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS site_settings (
            `key` VARCHAR(100) PRIMARY KEY,
            value MEDIUMTEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)
    # Migrate existing TEXT column to MEDIUMTEXT if not already upgraded
    cur.execute("""
        SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'site_settings'
          AND COLUMN_NAME = 'value'
    """)
    col_type = cur.fetchone()
    if col_type and col_type[0].lower() == 'text':
        cur.execute("ALTER TABLE site_settings MODIFY COLUMN value MEDIUMTEXT")
    # Seed carousel_style default only on first run (INSERT IGNORE skips if key exists)
    cur.execute(
        "INSERT IGNORE INTO site_settings (`key`, value) VALUES ('carousel_style', %s)",
        (CAROUSEL_STYLE_SUFFIX,),
    )
    # Seed cinemagraph_prompt default only on first run
    cur.execute(
        "INSERT IGNORE INTO site_settings (`key`, value) VALUES ('cinemagraph_prompt', %s)",
        (_CINE_DEFAULT_PROMPT,),
    )
    # Clear stale cinemagraph status keys left over from a server restart mid-run
    # (the daemon thread is killed but the DB key remains set to "running:N/M")
    cur.execute("DELETE FROM site_settings WHERE `key` LIKE 'cinemagraph_status_%'")
    # Clear stale narrated-video status left over from a server restart mid-run
    cur.execute(
        "UPDATE articles SET video_narrated_status = NULL "
        "WHERE video_narrated_status LIKE 'running%%'"
    )
    conn.commit()
    cur.close()
    conn.close()


try:
    init_site_settings()
except Exception as e:
    print(f"WARNING: Could not initialise site_settings table: {e}")


def log_activity(user_id, activity_type, description=None, metadata=None):
    """Log a user activity. Silently fails to avoid disrupting the main flow."""
    try:
        execute(
            "INSERT INTO user_activity (user_id, activity_type, description, metadata) "
            "VALUES (%s, %s, %s, %s)",
            (user_id, activity_type, description,
             json.dumps(metadata) if metadata else None),
        )
    except Exception:
        pass


# ------------------------------------------------------------
# AUTH (Flask-Login)
# ------------------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id, username, email, name, active, picture=None, currency="GBP", country=None, currency_locked=False):
        self.id = id
        self.username = username
        self.email = email
        self.name = name
        self.active = active
        self.picture = picture  # Profile picture as bytes (BLOB)
        self.currency = currency or "GBP"
        self.country = country
        self.currency_locked = bool(currency_locked)


# Mapping of country codes to default currencies
COUNTRY_CURRENCY_MAP = {
    # Americas
    "US": "USD", "BR": "BRL", "AR": "ARS", "CA": "CAD",
    # Europe (non-EUR)
    "GB": "GBP", "IE": "GBP",
    "CH": "CHF", "LI": "CHF",
    "SE": "SEK", "NO": "NOK", "DK": "DKK",
    "PL": "PLN", "CZ": "CZK", "HU": "HUF", "TR": "TRY",
    # Eurozone
    "DE": "EUR", "FR": "EUR", "ES": "EUR", "IT": "EUR", "NL": "EUR",
    "BE": "EUR", "AT": "EUR", "PT": "EUR", "GR": "EUR", "FI": "EUR",
    "LU": "EUR", "EE": "EUR", "LV": "EUR", "LT": "EUR", "SK": "EUR",
    "SI": "EUR", "CY": "EUR", "MT": "EUR", "HR": "EUR",
    # Asia-Pacific
    "AU": "AUD", "NZ": "NZD", "JP": "JPY", "CN": "CNY", "IN": "INR",
    # Africa
    "ZA": "ZAR",
}


def detect_currency_from_request():
    """Detect the user's likely currency based on their IP geolocation."""
    try:
        # Use the CF-IPCountry header (available on Cloud Run behind Cloud Load Balancer)
        # or X-AppEngine-Country, or fall back to an external lookup
        country_code = (
            request.headers.get("CF-IPCountry")
            or request.headers.get("X-AppEngine-Country")
            or request.headers.get("X-Country-Code")
        )

        if not country_code:
            # Lightweight fallback: use a free IP geolocation API
            forwarded_for = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            client_ip = forwarded_for or request.remote_addr
            if client_ip and client_ip not in ("127.0.0.1", "::1"):
                resp = http_requests.get(f"https://ipapi.co/{client_ip}/country/", timeout=3)
                if resp.status_code == 200 and len(resp.text.strip()) == 2:
                    country_code = resp.text.strip()

        if country_code:
            country_code = country_code.upper()
            currency = COUNTRY_CURRENCY_MAP.get(country_code, "EUR")
            return currency, country_code

    except Exception as e:
        print(f"[GeoIP] Could not detect location: {e}")

    return "GBP", None  # Default for a UK-based business


@login_manager.user_loader
def load_user(user_id):
    user_data = query_one(
        "SELECT id, username, email, name, active, picture, currency, country, currency_locked FROM users WHERE id = %s",
        (user_id,),
    )
    if user_data:
        return User(*user_data)
    return None


# ------------------------------------------------------------
# ADMIN CONFIG
# ------------------------------------------------------------
ADMIN_EMAIL = "thalesjacobi@gmail.com"


def is_admin():
    """Check if the current user is an admin."""
    return (
        hasattr(current_user, "is_authenticated")
        and current_user.is_authenticated
        and current_user.email == ADMIN_EMAIL
    )


def admin_required(f):
    """Decorator that requires admin access. Must be used after @login_required."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin():
            flash("Access denied.", "danger")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function


# ------------------------------------------------------------
# CURRENCY HELPERS
# ------------------------------------------------------------
CURRENCY_SYMBOLS = {
    "EUR": "\u20ac", "GBP": "\u00a3", "USD": "$",
    "BRL": "R$", "ARS": "AR$", "HUF": "Ft", "ZAR": "R",
    "AUD": "A$", "JPY": "\u00a5", "NZD": "NZ$",
    "CHF": "CHF", "SEK": "kr", "NOK": "kr", "DKK": "kr",
    "PLN": "z\u0142", "CZK": "K\u010d", "CAD": "C$",
    "CNY": "\u00a5", "INR": "\u20b9", "TRY": "\u20ba",
}

# In-memory cache for exchange rates: {"rates": {...}, "fetched_at": timestamp}
_fx_cache = {"rates": None, "fetched_at": 0}
FX_CACHE_TTL = 300  # 5 minutes


def get_fx_rates():
    """
    Return a dict of currency -> rate (EUR-based) from the database.
    Uses an in-memory cache with a 5-minute TTL.
    Falls back to hardcoded defaults if the database is unavailable.
    """
    now = time.time()
    if _fx_cache["rates"] and (now - _fx_cache["fetched_at"]) < FX_CACHE_TTL:
        return _fx_cache["rates"]

    try:
        # Get the latest rate for each currency
        rows = query_all(
            """SELECT currency, rate
               FROM exchange_rates e1
               WHERE fetched_at = (
                   SELECT MAX(fetched_at) FROM exchange_rates e2
                   WHERE e2.currency = e1.currency
               )"""
        )
        rates = {"EUR": 1.0}
        for currency, rate in rows:
            rates[currency] = float(rate)

        _fx_cache["rates"] = rates
        _fx_cache["fetched_at"] = now
        return rates

    except Exception as e:
        print(f"[FX] Failed to load rates from DB: {e}")
        # Fallback to hardcoded rates
        return {"EUR": 1.0, "USD": 1.08, "GBP": 0.86}


def fetch_ecb_rates():
    """
    Fetch latest exchange rates from the ECB Data API.
    Returns a dict like {"USD": 1.08, "GBP": 0.86, ...} (EUR-based).
    Raises an exception on failure so the caller can handle it.
    """
    # ECB SDMX-JSON endpoint for daily exchange rates (EUR base)
    ecb_url = (
        "https://data-api.ecb.europa.eu/service/data/EXR/"
        "D.USD+GBP+BRL+ARS+HUF+ZAR+AUD+JPY+NZD"
        "+CHF+SEK+NOK+DKK+PLN+CZK+CAD+CNY+INR+TRY"
        ".EUR.SP00.A"
        "?lastNObservations=1&format=jsondata"
    )
    resp = http_requests.get(ecb_url, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    rates = {}

    # Navigate the SDMX-JSON structure
    datasets = data["dataSets"][0]["series"]
    dimensions = data["structure"]["dimensions"]["series"]

    # Find the CURRENCY dimension (usually index 1)
    currency_dim = None
    for dim in dimensions:
        if dim["id"] == "CURRENCY":
            currency_dim = dim
            break

    if not currency_dim:
        raise ValueError("Could not find CURRENCY dimension in ECB response")

    for series_key, series_data in datasets.items():
        # series_key is like "0:0:0:0:0" — extract currency dimension index
        key_parts = series_key.split(":")
        currency_idx = int(key_parts[1])  # CURRENCY is dimension index 1
        currency_code = currency_dim["values"][currency_idx]["id"]

        # Get the latest observation value
        observations = series_data["observations"]
        if observations:
            latest_key = max(observations.keys(), key=int)
            rate_value = observations[latest_key][0]
            rates[currency_code] = float(rate_value)

    return rates


def get_active_currency():
    """Return the active currency code for the current user or visitor."""
    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        return current_user.currency or "GBP"
    return "GBP"  # Default for anonymous visitors


def convert_catalogue_price(price_gbp, target_currency):
    """Convert a catalogue price (stored in GBP) to the target currency."""
    return convert_price(price_gbp, "GBP", target_currency)


@app.context_processor
def inject_currency_helpers():
    """Make currency helpers available in all templates."""
    currency = get_active_currency()
    return {
        "active_currency": currency,
        "currency_symbol": CURRENCY_SYMBOLS.get(currency, currency),
        "currency_symbols": CURRENCY_SYMBOLS,
        "convert_catalogue_price": convert_catalogue_price,
        "is_admin": is_admin(),
        "contact_email": os.getenv("CONTACT_TO_EMAIL", "thalesjacobi@gmail.com"),
        "site_url": SITE_URL,
    }


# ------------------------------------------------------------
# PAGES
# ------------------------------------------------------------
@app.route("/")
def home():
    catalogue_items = query_all("SELECT * FROM catalogue")
    currency = get_active_currency()
    return render_template("home.html", catalogue_items=catalogue_items, user_currency=currency)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/articles")
def articles():
    rows = query_all(
        "SELECT id, slug, title, subtitle, excerpt, image_url, published_at "
        "FROM articles WHERE is_published = TRUE ORDER BY published_at DESC"
    )
    article_list = [
        {
            "id": r[0], "slug": r[1], "title": r[2], "subtitle": r[3],
            "excerpt": r[4], "image_url": r[5],
            "published_at": r[6].strftime("%d %B %Y").lstrip("0") if r[6] else None,
        }
        for r in rows
    ]
    return render_template("articles.html", articles=article_list)


@app.route("/articles/<slug>")
def article_view(slug):
    row = query_one(
        "SELECT id, slug, title, subtitle, content, excerpt, image_url, published_at, "
        "carousel_images, carousel_punchlines, carousel_cinemagraphs, show_slideshow "
        "FROM articles WHERE slug = %s AND is_published = TRUE",
        (slug,),
    )
    if not row:
        return render_template("404.html"), 404
    article = {
        "id": row[0], "slug": row[1], "title": row[2], "subtitle": row[3],
        "content": row[4], "excerpt": row[5] or "",
        "image_url": row[6],
        "published_at": row[7].strftime("%d %B %Y").lstrip("0") if row[7] else None,
        "carousel_images":       json.loads(row[8])  if row[8]  else [],
        "carousel_punchlines":   json.loads(row[9])  if row[9]  else [],
        "carousel_cinemagraphs": json.loads(row[10]) if row[10] else [],
        "show_slideshow": bool(row[11]),
    }
    return render_template("article.html", article=article)


@app.route("/catalogue")
def catalogue():
    catalogue_items = query_all("SELECT * FROM catalogue")
    currency = get_active_currency()
    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        log_activity(current_user.id, "page_view", "Viewed catalogue", {"page": "/catalogue"})
    return render_template("catalogue.html", catalogue_items=catalogue_items, user_currency=currency)


RECAPTCHA_SITE_KEY = "6LdO7nMsAAAAAFfZQbtr3vbztflfcyNA0uYL2opK"
RECAPTCHA_API_KEY = os.getenv("RECAPTCHA_API_KEY", "")
RECAPTCHA_GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "")


def verify_recaptcha(token, expected_action="contact_submit"):
    """Verify reCAPTCHA Enterprise token. Returns True if valid, False otherwise."""
    if not RECAPTCHA_API_KEY or not RECAPTCHA_GCP_PROJECT:
        print("WARNING: reCAPTCHA not configured (missing RECAPTCHA_API_KEY or GCP_PROJECT_ID). Skipping verification.")
        return True  # allow through if not configured yet

    url = f"https://recaptchaenterprise.googleapis.com/v1/projects/{RECAPTCHA_GCP_PROJECT}/assessments?key={RECAPTCHA_API_KEY}"
    body = {
        "event": {
            "token": token,
            "siteKey": RECAPTCHA_SITE_KEY,
            "expectedAction": expected_action,
        }
    }
    try:
        resp = http_requests.post(url, json=body, timeout=5)
        result = resp.json()
        token_valid = result.get("tokenProperties", {}).get("valid", False)
        action_match = result.get("tokenProperties", {}).get("action") == expected_action
        score = result.get("riskAnalysis", {}).get("score", 0)
        print(f"reCAPTCHA assessment: valid={token_valid}, action_match={action_match}, score={score}")
        return token_valid and action_match and score >= 0.5
    except Exception as e:
        print(f"reCAPTCHA verification error: {e}")
        return True  # fail open to avoid blocking legit users if API is down


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # Verify reCAPTCHA Enterprise token
        recaptcha_token = request.form.get("g-recaptcha-response", "")
        if not verify_recaptcha(recaptcha_token):
            flash("reCAPTCHA verification failed. Please try again.", "danger")
            return render_template("contact.html")

        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        try:
            # Sender must be the authenticated Gmail account
            # Reply-To is set to the visitor's email for easy replies
            msg = Message(
                subject=f"[Obelisk Stamps] Contact from {name}",
                sender=os.getenv("MAIL_USERNAME"),
                recipients=[os.getenv("CONTACT_TO_EMAIL", "thalesjacobi@gmail.com")],
                reply_to=email,
            )
            msg.body = f"""New contact form submission from Obelisk Stamps website:

Name: {name}
Email: {email}

Message:
{message}

---
You can reply directly to this email to respond to {name}.
"""
            mail.send(msg)
            flash(f"Thank you, {name}. Your message has been sent!", "success")
        except Exception as e:
            flash("Email sending failed. Please try again later.", "danger")
            print(f"Error sending email: {e}")

    return render_template("contact.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/stamp-quest-ai")
def stamp_quest_ai():
    return render_template("stamp_quest_ai.html")


# ------------------------------------------------------------
# AI CHATBOT ENDPOINT
# ------------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    """AI chatbot endpoint — accepts a conversation and returns a response."""
    if not _openai_client:
        return jsonify({"error": "Chatbot is not configured."}), 503

    # Rate limiting by IP
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr
    if not _check_chat_rate_limit(client_ip):
        return jsonify({"error": "Too many messages. Please wait a moment and try again."}), 429

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided."}), 400

    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if len(user_message) > 500:
        return jsonify({"error": "Message is too long (max 500 characters)."}), 400

    # Retrieve relevant knowledge base context
    kb_docs = _search_knowledge(user_message, n_results=3)
    context_block = ""
    if kb_docs:
        context_block = "\n\nRelevant business information:\n" + "\n---\n".join(kb_docs)

    system_message = CHATBOT_SYSTEM_PROMPT + context_block

    # Build conversation from client-side history (last 10 messages)
    history = data.get("history", [])
    messages = [{"role": "system", "content": system_message}]
    for msg in history[-10:]:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=400,
        )
        assistant_reply = response.choices[0].message.content
        if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
            log_activity(current_user.id, "chatbot_message", "Sent chatbot message",
                         {"message_length": len(user_message)})
        return jsonify({"reply": assistant_reply})
    except _openai_module.RateLimitError:
        return jsonify({"error": "The chatbot is busy. Please try again in a moment."}), 429
    except _openai_module.APIError as e:
        print(f"[Chatbot] OpenAI API error: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500
    except Exception as e:
        print(f"[Chatbot] Unexpected error: {e}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# ------------------------------------------------------------
# LOGIN VIA GOOGLE
# ------------------------------------------------------------
@app.route("/login")
def login():
    if client is None:
        flash("Google login is not configured. Set GOOGLE_CLIENT_ID/SECRET.", "danger")
        return redirect(url_for("home"))

    # Pass the return URL through OAuth state so it survives the redirect
    next_url = request.args.get("next", "")
    state_data = base64.urlsafe_b64encode(json.dumps({"next": next_url}).encode()).decode()

    google_provider_cfg = http_requests.get(GOOGLE_DISCOVERY_URL).json()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Use explicit SITE_URL for the callback to ensure correct domain
    if SITE_URL:
        callback_url = f"{SITE_URL}/login/callback"
    else:
        callback_url = url_for("login_callback", _external=True)

    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=callback_url,
        scope=["openid", "email", "profile"],
        state=state_data,
    )
    return redirect(request_uri)


@app.route("/login/callback")
def login_callback():
    if client is None:
        flash("Google login is not configured.", "danger")
        return redirect(url_for("home"))

    # Check if user cancelled or there was an error
    error = request.args.get("error")
    if error:
        if error == "access_denied":
            flash("Login cancelled.", "info")
        else:
            flash(f"Login failed: {error}", "danger")
        return redirect(url_for("home"))

    try:
        return _handle_oauth_callback()
    except Exception as e:
        import traceback
        print(f"[OAuth Error] {e}")
        traceback.print_exc()
        flash("Login failed. Please try again.", "danger")
        return redirect(url_for("home"))


def _handle_oauth_callback():

    code = request.args.get("code")
    google_provider_cfg = http_requests.get(GOOGLE_DISCOVERY_URL).json()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Use explicit SITE_URL for the redirect_uri to match what was sent during login
    if SITE_URL:
        callback_url = f"{SITE_URL}/login/callback"
    else:
        callback_url = request.base_url

    # Exchange authorization code for tokens directly (avoid stateful client issues)
    token_response = http_requests.post(
        token_endpoint,
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": callback_url,
            "grant_type": "authorization_code",
        },
    )

    token_data = token_response.json()
    if "error" in token_data:
        print(f"[OAuth Error] Token exchange failed: {token_data}")
        raise Exception(f"Token exchange failed: {token_data.get('error_description', token_data.get('error'))}")

    # Get user info using the access token
    access_token = token_data["access_token"]
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    userinfo_response = http_requests.get(
        userinfo_endpoint,
        headers={"Authorization": f"Bearer {access_token}"},
    )

    user_info = userinfo_response.json()
    username = user_info["email"].split("@")[0]
    email = user_info["email"]
    name = user_info.get("name", username)

    # Download profile picture from Google
    picture_bytes = None
    picture_url = user_info.get("picture")
    if picture_url:
        try:
            picture_response = http_requests.get(picture_url, timeout=10)
            if picture_response.status_code == 200:
                picture_bytes = picture_response.content
        except Exception as e:
            print(f"Failed to download profile picture: {e}")

    user_data = query_one(
        "SELECT id, username, email, name, active, picture, currency, country, currency_locked FROM users WHERE email = %s",
        (email,),
    )

    if user_data:
        user = User(*user_data)
        if not user.active:
            flash("Your account is inactive. Please contact support.", "danger")
            return redirect(url_for("home"))
        # Update profile picture if we got a new one
        if picture_bytes:
            execute(
                "UPDATE users SET picture = %s WHERE id = %s",
                (picture_bytes, user.id),
            )
            user.picture = picture_bytes
    else:
        # New user: detect their location to set default currency
        detected_currency, detected_country = detect_currency_from_request()
        user_id = execute(
            "INSERT INTO users (username, email, name, active, date_created, picture, currency, country) VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s)",
            (username, email, name, 1, picture_bytes, detected_currency, detected_country),
        )
        user = User(user_id, username, email, name, 1, picture_bytes, detected_currency, detected_country)

    login_user(user)
    log_activity(user.id, "login", f"{user.name} logged in")

    # Retrieve return URL from OAuth state parameter
    next_url = None
    state_param = request.args.get("state")
    if state_param:
        try:
            state_data = json.loads(base64.urlsafe_b64decode(state_param))
            next_url = state_data.get("next")
        except Exception:
            pass
    return redirect(next_url or url_for("account"))


@app.route("/account")
@login_required
def account():
    return render_template("account.html")


@app.route("/account/update-currency", methods=["POST"])
@login_required
def update_currency():
    """Update user's preferred currency (one-time only)."""
    if current_user.currency_locked:
        return jsonify({"error": "Currency has already been set and cannot be changed. Please contact us via the contact form."}), 403

    data = request.get_json()
    currency = data.get("currency", "").upper()
    if currency not in CURRENCY_SYMBOLS:
        return jsonify({"error": "Invalid currency"}), 400
    execute(
        "UPDATE users SET currency = %s, currency_locked = 1 WHERE id = %s",
        (currency, current_user.id),
    )
    current_user.currency = currency
    current_user.currency_locked = True
    log_activity(current_user.id, "currency_changed", f"Changed currency to {currency}",
                 {"currency": currency})
    return jsonify({"success": True, "currency": currency})


@app.route("/user/picture/<int:user_id>")
def user_picture(user_id):
    """Serve user profile picture from database."""
    from flask import Response
    user_data = query_one(
        "SELECT picture FROM users WHERE id = %s",
        (user_id,),
    )
    if user_data and user_data[0]:
        return Response(user_data[0], mimetype='image/jpeg')
    # Return a default placeholder or 404
    return '', 404


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# ------------------------------------------------------------
# ML PREDICT ROUTE (calls external ML API)
# ------------------------------------------------------------
def convert_price(price_value, price_currency, target_currency):
    """Convert price to target currency using DB-backed rates."""
    fx_rates = get_fx_rates()

    if not price_value or not price_currency:
        return None

    # Ensure price_value is a float (MySQL returns decimal.Decimal)
    price_value = float(price_value)
    price_currency = price_currency.upper()
    target_currency = target_currency.upper()

    # Convert to EUR first
    if price_currency == "EUR":
        amount_eur = price_value
    elif price_currency in fx_rates:
        amount_eur = price_value / fx_rates[price_currency]
    else:
        amount_eur = price_value

    # Convert to target
    if target_currency == "EUR":
        return round(amount_eur, 2)
    elif target_currency in fx_rates:
        return round(amount_eur * fx_rates[target_currency], 2)
    else:
        return round(amount_eur, 2)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Proxy endpoint that forwards requests to the ML API service.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Use multipart field name 'image'."}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file locally (for preview)
    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))

    currency = request.form.get("currency", "EUR").upper()

    # Forward to ML API
    try:
        with open(save_path, "rb") as img_file:
            files = {"image": (f.filename, img_file, "image/jpeg")}
            data = {"top_k": 3, "confidence": 0.3}

            response = http_requests.post(
                f"{ML_API_URL}/predict",
                files=files,
                data=data,
                timeout=60,
            )

        if response.status_code != 200:
            return jsonify({
                "error": "ML API error",
                "details": response.text,
            }), response.status_code

        result = response.json()

        if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
            log_activity(current_user.id, "stamp_identified", "Used Stamp Quest AI",
                         {"result_count": len(result.get("matches", result.get("stamps", [])))})

        # Process the response to add currency conversion
        if result.get("multi_stamp"):
            # Multi-stamp response
            for stamp in result.get("stamps", []):
                for match in stamp.get("matches", []):
                    _add_price_conversion(match, currency)
            return jsonify(result)
        else:
            # Single stamp response
            for match in result.get("matches", []):
                _add_price_conversion(match, currency)
            return jsonify(result.get("matches", []))

    except http_requests.exceptions.ConnectionError:
        return jsonify({
            "error": "ML API not available",
            "details": f"Could not connect to {ML_API_URL}",
        }), 503
    except http_requests.exceptions.Timeout:
        return jsonify({
            "error": "ML API timeout",
            "details": "The request took too long",
        }), 504
    except Exception as e:
        return jsonify({
            "error": "Unexpected error",
            "details": str(e),
        }), 500


def _add_price_conversion(match: dict, target_currency: str):
    """Add price conversion fields to a match result."""
    price_value = match.get("price_value")
    price_currency = match.get("price_currency")

    match["price_original"] = {
        "value": price_value,
        "currency": price_currency,
    }

    converted = convert_price(price_value, price_currency, target_currency)
    match["price_converted"] = {
        "value": converted,
        "currency": target_currency,
    }


# ------------------------------------------------------------
# SERVE UPLOADED FILES
# ------------------------------------------------------------
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


# ------------------------------------------------------------
# USER ALBUM FEATURE
# ------------------------------------------------------------
@app.route("/album")
@login_required
def my_album():
    """View user's stamp album."""
    # Get all stamps for this user
    stamps = query_all(
        """SELECT id, stamp_image, title, country, year, condition_text,
                  price_value, price_currency, source_url, similarity, date_added
           FROM user_stamps
           WHERE user_id = %s
           ORDER BY date_added DESC""",
        (current_user.id,),
    )

    # Calculate totals — convert each stamp's value to the user's preferred currency
    total_stamps = len(stamps) if stamps else 0
    user_currency = current_user.currency or "EUR"
    total_value = 0.0
    if stamps:
        for s in stamps:
            price_val = s[6]   # price_value
            price_cur = s[7]   # price_currency
            if price_val:
                converted = convert_price(price_val, price_cur or "EUR", user_currency)
                total_value += converted if converted else float(price_val)

    return render_template("my_album.html",
                         stamps=stamps,
                         total_stamps=total_stamps,
                         total_value=total_value,
                         primary_currency=user_currency)


@app.route("/album/add", methods=["POST"])
@login_required
def add_to_album():
    """Add a stamp to user's album."""
    from flask import Response
    import base64

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Get stamp image from base64 or from uploaded file
    stamp_image = None
    if data.get("image_base64"):
        # Remove data URL prefix if present
        img_data = data["image_base64"]
        if "," in img_data:
            img_data = img_data.split(",")[1]
        stamp_image = base64.b64decode(img_data)

    # Insert stamp into database
    try:
        stamp_id = execute(
            """INSERT INTO user_stamps
               (user_id, stamp_image, title, country, year, condition_text,
                price_value, price_currency, source_url, similarity)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                current_user.id,
                stamp_image,
                data.get("title"),
                data.get("country"),
                data.get("year"),
                data.get("condition_text"),
                data.get("price_value"),
                data.get("price_currency"),
                data.get("source_url"),
                data.get("similarity"),
            ),
        )
        log_activity(current_user.id, "stamp_saved", f"Saved stamp: {data.get('title', 'Unknown')}",
                     {"stamp_id": stamp_id, "title": data.get("title"), "country": data.get("country")})
        return jsonify({"success": True, "stamp_id": stamp_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/album/delete/<int:stamp_id>", methods=["POST"])
@login_required
def delete_from_album(stamp_id):
    """Delete a stamp from user's album."""
    # Verify the stamp belongs to the current user
    stamp = query_one(
        "SELECT id FROM user_stamps WHERE id = %s AND user_id = %s",
        (stamp_id, current_user.id),
    )

    if not stamp:
        return jsonify({"error": "Stamp not found"}), 404

    execute("DELETE FROM user_stamps WHERE id = %s", (stamp_id,))
    log_activity(current_user.id, "stamp_deleted", "Removed stamp from album",
                 {"stamp_id": stamp_id})
    flash("Stamp removed from your album.", "success")
    return redirect(url_for("my_album"))


@app.route("/album/stamp-image/<int:stamp_id>")
@login_required
def album_stamp_image(stamp_id):
    """Serve stamp image from user's album."""
    from flask import Response

    stamp = query_one(
        "SELECT stamp_image FROM user_stamps WHERE id = %s AND user_id = %s",
        (stamp_id, current_user.id),
    )

    if stamp and stamp[0]:
        return Response(stamp[0], mimetype='image/jpeg')
    return '', 404


# ------------------------------------------------------------
# SHOPPING CART
# ------------------------------------------------------------
@app.route("/cart")
@login_required
def cart():
    """View shopping cart."""
    cart_items = query_all(
        """SELECT ci.id, ci.catalogue_id, ci.quantity, c.title, c.price, c.image_url
           FROM cart_items ci
           JOIN catalogue c ON ci.catalogue_id = c.id
           WHERE ci.user_id = %s""",
        (current_user.id,),
    )

    # Calculate total
    total = sum((item[4] or 0) * item[2] for item in cart_items)  # price * quantity

    return render_template("cart.html",
                         cart_items=cart_items,
                         total=total,
                         stripe_key=STRIPE_PUBLISHABLE_KEY)


@app.route("/cart/add/<int:catalogue_id>", methods=["POST"])
@login_required
def add_to_cart(catalogue_id):
    """Add item to cart."""
    # Check if item exists in catalogue
    item = query_one("SELECT id, title FROM catalogue WHERE id = %s", (catalogue_id,))
    if not item:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": "Item not found"}), 404
        flash("Item not found.", "danger")
        return redirect(url_for("catalogue"))

    # Check if already in cart
    existing = query_one(
        "SELECT id, quantity FROM cart_items WHERE user_id = %s AND catalogue_id = %s",
        (current_user.id, catalogue_id),
    )

    if existing:
        # Update quantity
        execute(
            "UPDATE cart_items SET quantity = quantity + 1 WHERE id = %s",
            (existing[0],),
        )
    else:
        # Insert new cart item
        execute(
            "INSERT INTO cart_items (user_id, catalogue_id, quantity) VALUES (%s, %s, 1)",
            (current_user.id, catalogue_id),
        )

    log_activity(current_user.id, "add_to_cart", f"Added {item[1]} to cart",
                 {"catalogue_id": catalogue_id, "title": item[1]})

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        count = get_cart_count(current_user.id)
        return jsonify({"success": True, "cart_count": count, "message": f"{item[1]} added to cart"})

    flash(f"{item[1]} added to cart!", "success")
    return redirect(request.referrer or url_for("catalogue"))


@app.route("/cart/remove/<int:cart_item_id>", methods=["POST"])
@login_required
def remove_from_cart(cart_item_id):
    """Remove item from cart."""
    # Verify the item belongs to the current user
    item = query_one(
        "SELECT id FROM cart_items WHERE id = %s AND user_id = %s",
        (cart_item_id, current_user.id),
    )

    if not item:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": "Item not found"}), 404
        flash("Item not found.", "danger")
        return redirect(url_for("cart"))

    execute("DELETE FROM cart_items WHERE id = %s", (cart_item_id,))
    log_activity(current_user.id, "remove_from_cart", "Removed item from cart",
                 {"cart_item_id": cart_item_id})

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        count = get_cart_count(current_user.id)
        return jsonify({"success": True, "cart_count": count})

    flash("Item removed from cart.", "success")
    return redirect(url_for("cart"))


@app.route("/cart/count")
def cart_count():
    """Get cart item count for navbar badge."""
    if current_user.is_authenticated:
        count = get_cart_count(current_user.id)
        return jsonify({"count": count})
    return jsonify({"count": 0})


def get_cart_count(user_id):
    """Helper to get cart item count."""
    result = query_one(
        "SELECT COALESCE(SUM(quantity), 0) FROM cart_items WHERE user_id = %s",
        (user_id,),
    )
    return int(result[0]) if result else 0


# ------------------------------------------------------------
# CHECKOUT & PAYMENTS
# ------------------------------------------------------------
@app.route("/checkout")
@login_required
def checkout():
    """Checkout page."""
    cart_items = query_all(
        """SELECT ci.id, ci.catalogue_id, ci.quantity, c.title, c.price, c.image_url
           FROM cart_items ci
           JOIN catalogue c ON ci.catalogue_id = c.id
           WHERE ci.user_id = %s""",
        (current_user.id,),
    )

    if not cart_items:
        flash("Your cart is empty.", "info")
        return redirect(url_for("catalogue"))

    total = sum((item[4] or 0) * item[2] for item in cart_items)

    return render_template("checkout.html",
                         cart_items=cart_items,
                         total=total,
                         stripe_key=STRIPE_PUBLISHABLE_KEY)


@app.route("/checkout/create-session", methods=["POST"])
@login_required
def create_checkout_session():
    """Create Stripe Checkout Session."""
    if not stripe.api_key:
        return jsonify({"error": "Payment system not configured"}), 500

    cart_items = query_all(
        """SELECT ci.catalogue_id, ci.quantity, c.title, c.price, c.image_url
           FROM cart_items ci
           JOIN catalogue c ON ci.catalogue_id = c.id
           WHERE ci.user_id = %s""",
        (current_user.id,),
    )

    if not cart_items:
        return jsonify({"error": "Cart is empty"}), 400

    # Build line items for Stripe in user's preferred currency
    user_currency = get_active_currency().lower()
    line_items = []
    for item in cart_items:
        catalogue_id, quantity, title, price, image_url = item
        converted_price = convert_catalogue_price(price or 0, user_currency.upper()) or 0
        line_items.append({
            "price_data": {
                "currency": user_currency,
                "product_data": {
                    "name": title or "Stamp Collection",
                    "images": [url_for("static", filename=image_url, _external=True)] if image_url else [],
                },
                "unit_amount": int(converted_price * 100),  # Stripe uses smallest currency unit
            },
            "quantity": quantity,
        })

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=line_items,
            mode="payment",
            success_url=url_for("checkout_success", _external=True) + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=url_for("checkout_cancel", _external=True),
            customer_email=current_user.email,
            shipping_address_collection={
                "allowed_countries": ["GB", "US", "CA", "AU", "NZ", "IE", "DE", "FR", "ES", "IT", "NL", "BE"],
            },
            metadata={
                "user_id": str(current_user.id),
            },
            payment_intent_data={
                "metadata": {
                    "user_id": str(current_user.id),
                },
            },
        )
        return jsonify({"checkout_url": checkout_session.url})
    except stripe.error.StripeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/checkout/success")
@login_required
def checkout_success():
    """Payment success page."""
    session_id = request.args.get("session_id")

    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)

            # Check if order already exists for this session
            existing_order = query_one(
                "SELECT id FROM orders WHERE stripe_checkout_session_id = %s",
                (session_id,),
            )

            if not existing_order and session.payment_status == "paid":
                # Create order from session
                _create_order_from_session(session)
        except stripe.error.StripeError:
            pass  # Webhook will handle it

    return render_template("checkout_success.html")


@app.route("/checkout/cancel")
@login_required
def checkout_cancel():
    """Payment cancelled page."""
    return render_template("checkout_cancel.html")


def _create_order_from_session(session):
    """Create order record from Stripe session."""
    user_id = session.metadata.get("user_id")
    if not user_id:
        return None

    user_id = int(user_id)

    # Get shipping details
    shipping = session.shipping_details or {}
    shipping_address = shipping.get("address", {})
    address_str = ", ".join(filter(None, [
        shipping_address.get("line1"),
        shipping_address.get("line2"),
        shipping_address.get("city"),
        shipping_address.get("state"),
        shipping_address.get("postal_code"),
        shipping_address.get("country"),
    ]))

    # Get cart items
    cart_items = query_all(
        """SELECT ci.catalogue_id, ci.quantity, c.title, c.price
           FROM cart_items ci
           JOIN catalogue c ON ci.catalogue_id = c.id
           WHERE ci.user_id = %s""",
        (user_id,),
    )

    if not cart_items:
        return None

    total = session.amount_total / 100 if session.amount_total else sum((item[3] or 0) * item[1] for item in cart_items)
    order_currency = (session.currency or "gbp").upper()

    # Create order
    order_id = execute(
        """INSERT INTO orders
           (user_id, stripe_checkout_session_id, stripe_payment_intent_id, status,
            total_amount, currency, shipping_name, shipping_email, shipping_address)
           VALUES (%s, %s, %s, 'paid', %s, %s, %s, %s, %s)""",
        (
            user_id,
            session.id,
            session.payment_intent,
            total,
            order_currency,
            shipping.get("name"),
            session.customer_email,
            address_str,
        ),
    )

    # Create order items
    for item in cart_items:
        catalogue_id, quantity, title, price = item
        execute(
            """INSERT INTO order_items (order_id, catalogue_id, quantity, price_at_purchase, title)
               VALUES (%s, %s, %s, %s, %s)""",
            (order_id, catalogue_id, quantity, price, title),
        )

    # Clear cart
    execute("DELETE FROM cart_items WHERE user_id = %s", (user_id,))

    log_activity(user_id, "order_placed", f"Order #{order_id} placed",
                 {"order_id": order_id, "total": float(total), "currency": order_currency})

    return order_id


@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhooks."""
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature")

    if not STRIPE_WEBHOOK_SECRET:
        # If no webhook secret, skip signature verification (dev mode)
        event = json.loads(payload)
    else:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            return jsonify({"error": "Invalid payload"}), 400
        except stripe.error.SignatureVerificationError:
            return jsonify({"error": "Invalid signature"}), 400

    # Handle the event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        if session.get("payment_status") == "paid":
            _create_order_from_session(stripe.checkout.Session.retrieve(
                session["id"],
                expand=["shipping_details"],
            ))

    return jsonify({"status": "success"})


# ------------------------------------------------------------
# ORDERS
# ------------------------------------------------------------
@app.route("/orders")
@login_required
def orders():
    """View order history."""
    user_orders = query_all(
        """SELECT id, status, total_amount, currency, shipping_name, created_at
           FROM orders
           WHERE user_id = %s
           ORDER BY created_at DESC""",
        (current_user.id,),
    )
    return render_template("orders.html", orders=user_orders)


@app.route("/orders/<int:order_id>")
@login_required
def order_detail(order_id):
    """View single order details."""
    order = query_one(
        """SELECT id, status, total_amount, currency, shipping_name, shipping_email,
                  shipping_address, created_at, stripe_checkout_session_id
           FROM orders
           WHERE id = %s AND user_id = %s""",
        (order_id, current_user.id),
    )

    if not order:
        flash("Order not found.", "danger")
        return redirect(url_for("orders"))

    order_items = query_all(
        """SELECT oi.quantity, oi.price_at_purchase, oi.title, c.image_url
           FROM order_items oi
           LEFT JOIN catalogue c ON oi.catalogue_id = c.id
           WHERE oi.order_id = %s""",
        (order_id,),
    )

    return render_template("order_detail.html", order=order, order_items=order_items)


# ------------------------------------------------------------
# ADMIN PANEL
# ------------------------------------------------------------
@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    """Admin panel — redirects to the first tab (currency exchange)."""
    return redirect(url_for("admin_currency"))


@app.route("/admin/currency")
@login_required
@admin_required
def admin_currency():
    """Currency exchange management page."""
    # Get the latest rate for each currency
    rates = query_all(
        """SELECT e1.currency, e1.rate, e1.source, e1.fetched_at
           FROM exchange_rates e1
           WHERE e1.fetched_at = (
               SELECT MAX(e2.fetched_at) FROM exchange_rates e2
               WHERE e2.currency = e1.currency
           )
           ORDER BY e1.currency"""
    )

    # Get recent history (last 20 entries)
    history = query_all(
        """SELECT currency, rate, source, fetched_at
           FROM exchange_rates
           ORDER BY fetched_at DESC, currency
           LIMIT 20"""
    )

    return render_template("admin.html",
                         active_tab="currency",
                         rates=rates,
                         history=history)


@app.route("/admin/currency/fetch", methods=["POST"])
@login_required
@admin_required
def admin_currency_fetch():
    """Fetch latest rates from ECB API (AJAX endpoint)."""
    try:
        rates = fetch_ecb_rates()
        return jsonify({"success": True, "rates": rates})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/admin/currency/save", methods=["POST"])
@login_required
@admin_required
def admin_currency_save():
    """Save exchange rates to the database (AJAX endpoint)."""
    data = request.get_json()
    if not data or "rates" not in data:
        return jsonify({"success": False, "error": "No rates provided"}), 400

    source = data.get("source", "manual")
    now = datetime.now()

    try:
        conn = get_db()
        cur = conn.cursor()
        for currency, rate in data["rates"].items():
            currency = currency.upper().strip()
            if len(currency) != 3:
                continue
            rate = float(rate)
            if rate <= 0:
                continue
            cur.execute(
                """INSERT INTO exchange_rates (currency, rate, source, fetched_at)
                   VALUES (%s, %s, %s, %s)""",
                (currency, rate, source, now),
            )
        conn.commit()
        cur.close()
        conn.close()

        # Invalidate the in-memory cache so new rates take effect immediately
        _fx_cache["rates"] = None
        _fx_cache["fetched_at"] = 0

        return jsonify({"success": True, "message": f"Saved {len(data['rates'])} rate(s)."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Admin: User Management ──────────────────────────────────

@app.route("/admin/users")
@login_required
@admin_required
def admin_users():
    """Show all users in the admin panel."""
    users = query_all(
        "SELECT id, username, email, name, active, currency, country, "
        "currency_locked, date_created FROM users ORDER BY id"
    )
    return render_template("admin.html", active_tab="users", users=users)


@app.route("/admin/users/<int:user_id>")
@login_required
@admin_required
def admin_user_detail(user_id):
    """Return a single user's editable fields as JSON."""
    row = query_one(
        "SELECT id, username, email, name, active, currency, country, "
        "currency_locked FROM users WHERE id = %s",
        (user_id,),
    )
    if not row:
        return jsonify({"error": "User not found"}), 404
    return jsonify({
        "id": row[0],
        "username": row[1],
        "email": row[2],
        "name": row[3],
        "active": bool(row[4]),
        "currency": row[5],
        "country": row[6] or "",
        "currency_locked": bool(row[7]),
    })


@app.route("/admin/users/<int:user_id>/save", methods=["POST"])
@login_required
@admin_required
def admin_user_save(user_id):
    """Update a user's fields from the admin panel."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Validate that user exists
    existing = query_one("SELECT id FROM users WHERE id = %s", (user_id,))
    if not existing:
        return jsonify({"error": "User not found"}), 404

    # Build update fields
    allowed = {"username", "email", "name", "active", "currency", "country", "currency_locked"}
    updates = []
    values = []
    for field in allowed:
        if field in data:
            val = data[field]
            if field == "active":
                val = 1 if val else 0
            elif field == "currency_locked":
                val = 1 if val else 0
            elif field == "currency":
                val = str(val).upper().strip()
                if val not in CURRENCY_SYMBOLS:
                    return jsonify({"error": f"Invalid currency: {val}"}), 400
            else:
                val = str(val).strip()
            updates.append(f"{field} = %s")
            values.append(val)

    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400

    values.append(user_id)
    try:
        execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = %s",
            tuple(values),
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/users/<int:user_id>/activity")
@login_required
@admin_required
def admin_user_activity(user_id):
    """Return recent activity for a user as JSON (paginated)."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    per_page = min(per_page, 100)
    offset = (page - 1) * per_page

    rows = query_all(
        "SELECT id, activity_type, description, metadata, created_at "
        "FROM user_activity WHERE user_id = %s "
        "ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (user_id, per_page, offset),
    )

    total_row = query_one(
        "SELECT COUNT(*) FROM user_activity WHERE user_id = %s",
        (user_id,),
    )
    total = total_row[0] if total_row else 0

    activities = []
    for r in rows:
        meta = None
        if r[3]:
            try:
                meta = json.loads(r[3]) if isinstance(r[3], str) else r[3]
            except (json.JSONDecodeError, TypeError):
                meta = None
        activities.append({
            "id": r[0],
            "activity_type": r[1],
            "description": r[2],
            "metadata": meta,
            "created_at": r[4].strftime("%d %b %Y %H:%M") if r[4] else None,
        })

    return jsonify({
        "activities": activities,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if total > 0 else 0,
    })


# ------------------------------------------------------------
# ADMIN — SETTINGS
# ------------------------------------------------------------
@app.route("/admin/settings")
@login_required
@admin_required
def admin_settings():
    carousel_style     = get_setting('carousel_style', CAROUSEL_STYLE_SUFFIX)
    cinemagraph_prompt = get_setting('cinemagraph_prompt', _CINE_DEFAULT_PROMPT)
    ig_caption_prompt  = get_setting('ig_caption_prompt', _IG_CAPTION_DEFAULT_PROMPT)
    return render_template("admin.html", active_tab="settings",
                           carousel_style=carousel_style,
                           cinemagraph_prompt=cinemagraph_prompt,
                           ig_caption_prompt=ig_caption_prompt,
                           ig_user_id_set=bool(IG_USER_ID),
                           ig_token_set=bool(IG_ACCESS_TOKEN),
                           fb_page_id_set=bool(FB_PAGE_ID),
                           fb_token_set=bool(FB_PAGE_ACCESS_TOKEN),
                           yt_connected=bool(get_setting("youtube_refresh_token")))


@app.route("/admin/settings/save", methods=["POST"])
@login_required
@admin_required
def admin_settings_save():
    data = request.get_json()
    # Handle carousel style
    carousel_style = (data.get("carousel_style") or "").strip()
    if carousel_style:
        execute(
            "INSERT INTO site_settings (`key`, value) VALUES ('carousel_style', %s) "
            "ON DUPLICATE KEY UPDATE value = %s",
            (carousel_style, carousel_style),
        )
    # Handle cinemagraph prompt (global or per-article)
    cinemagraph_prompt = (data.get("cinemagraph_prompt") or "").strip()
    if cinemagraph_prompt:
        aid = data.get("article_id")
        key = f"cinemagraph_prompt_{aid}" if aid else "cinemagraph_prompt"
        execute(
            "INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s",
            (key, cinemagraph_prompt, cinemagraph_prompt),
        )
    # Handle Instagram caption prompt (global or per-article)
    ig_caption_prompt = (data.get("ig_caption_prompt") or "").strip()
    if ig_caption_prompt:
        aid = data.get("article_id")
        key = f"ig_caption_prompt_{aid}" if aid else "ig_caption_prompt"
        execute(
            "INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s",
            (key, ig_caption_prompt, ig_caption_prompt),
        )
    if not carousel_style and not cinemagraph_prompt and not ig_caption_prompt:
        return jsonify({"error": "Nothing to save"}), 400
    return jsonify({"success": True})


# ------------------------------------------------------------
# ADMIN — ARTICLES
# ------------------------------------------------------------
@app.route("/admin/articles")
@login_required
@admin_required
def admin_articles():
    rows = query_all(
        "SELECT id, title, slug, is_published, published_at, updated_at "
        "FROM articles ORDER BY updated_at DESC"
    )
    article_list = [
        {
            "id": r[0], "title": r[1], "slug": r[2],
            "is_published": bool(r[3]),
            "published_at": r[4].strftime("%d %b %Y") if r[4] else None,
            "updated_at": r[5].strftime("%d %b %Y %H:%M") if r[5] else None,
        }
        for r in rows
    ]
    return render_template("admin.html", active_tab="articles", articles=article_list)


@app.route("/admin/articles/new")
@login_required
@admin_required
def admin_article_new():
    return render_template("article_edit.html", article=None, carousel_style=CAROUSEL_STYLE_SUFFIX,
                           cinemagraph_prompt=get_setting('cinemagraph_prompt', _CINE_DEFAULT_PROMPT),
                           ig_caption_prompt=get_setting('ig_caption_prompt', _IG_CAPTION_DEFAULT_PROMPT),
                           ig_article_caption_prompt='',
                           ig_configured=bool(IG_USER_ID and IG_ACCESS_TOKEN),
                           fb_configured=bool(FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN),
                           yt_configured=bool(get_setting("youtube_refresh_token")))


@app.route("/admin/articles/new", methods=["POST"])
@login_required
@admin_required
def admin_article_create():
    data = request.get_json()
    slug = data.get("slug", "").strip()
    title = data.get("title", "").strip()
    if not slug or not title:
        return jsonify({"error": "Title and slug are required"}), 400
    is_published = bool(data.get("is_published", False))
    published_at_sql = "NOW()" if is_published else "NULL"
    try:
        article_id = execute(
            f"INSERT INTO articles (slug, title, subtitle, content, excerpt, image_url, is_published, published_at) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s, {published_at_sql if is_published else 'NULL'})",
            (slug, title, data.get("subtitle", ""), data.get("content", ""),
             data.get("excerpt", ""), data.get("image_url", ""), is_published),
        )
        return jsonify({"success": True, "id": article_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/articles/<int:article_id>/edit")
@login_required
@admin_required
def admin_article_edit(article_id):
    row = query_one(
        "SELECT id, slug, title, subtitle, content, excerpt, image_url, "
        "is_published, published_at, carousel_prompts, carousel_images, "
        "carousel_punchlines, carousel_style, show_slideshow "
        "FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row:
        flash("Article not found.", "danger")
        return redirect(url_for("admin_articles"))
    article = {
        "id": row[0], "slug": row[1], "title": row[2], "subtitle": row[3],
        "content": row[4] or "", "excerpt": row[5] or "",
        "image_url": row[6] or "", "is_published": bool(row[7]),
        "published_at": row[8].strftime("%d %b %Y") if row[8] else None,
        "carousel_prompts": json.loads(row[9]) if row[9] else [],
        "carousel_images": json.loads(row[10]) if row[10] else [],
        "carousel_punchlines": json.loads(row[11]) if row[11] else [],
        "show_slideshow": bool(row[13]),
    }
    # Show this article's saved style; fall back to current site setting if never generated
    article_carousel_style = row[12] or get_setting('carousel_style', CAROUSEL_STYLE_SUFFIX)
    # Load last-used Instagram captions from post snapshots
    cine_snap_raw = get_setting(f"ig_post_snapshot_{article_id}")
    car_snap_raw  = get_setting(f"ig_car_post_snapshot_{article_id}")
    cine_caption  = json.loads(cine_snap_raw).get("caption", "") if cine_snap_raw else ""
    car_caption   = json.loads(car_snap_raw).get("caption", "") if car_snap_raw else ""

    return render_template("article_edit.html", article=article,
                           carousel_style=article_carousel_style,
                           cinemagraph_prompt=get_setting('cinemagraph_prompt', _CINE_DEFAULT_PROMPT),
                           cinemagraph_article_prompt=get_setting(f'cinemagraph_prompt_{article_id}', ''),
                           ig_caption_prompt=get_setting('ig_caption_prompt', _IG_CAPTION_DEFAULT_PROMPT),
                           ig_article_caption_prompt=get_setting(f'ig_caption_prompt_{article_id}', ''),
                           ig_configured=bool(IG_USER_ID and IG_ACCESS_TOKEN),
                           fb_configured=bool(FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN),
                           yt_configured=bool(get_setting("youtube_refresh_token")),
                           ig_cine_caption=cine_caption,
                           ig_car_caption=car_caption)


@app.route("/admin/articles/<int:article_id>/toggle-slideshow", methods=["POST"])
@login_required
@admin_required
def admin_article_toggle_slideshow(article_id):
    data = request.get_json() or {}
    enabled = bool(data.get("enabled", False))
    execute("UPDATE articles SET show_slideshow = %s WHERE id = %s", (enabled, article_id))
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/save", methods=["POST"])
@login_required
@admin_required
def admin_article_save(article_id):
    data = request.get_json()
    slug = data.get("slug", "").strip()
    title = data.get("title", "").strip()
    if not slug or not title:
        return jsonify({"error": "Title and slug are required"}), 400
    is_published = bool(data.get("is_published", False))
    # Set published_at only on first publish (don't overwrite existing)
    try:
        existing = query_one("SELECT is_published, published_at FROM articles WHERE id = %s", (article_id,))
        if not existing:
            return jsonify({"error": "Article not found"}), 404
        was_published, existing_published_at = existing
        if is_published and not was_published and not existing_published_at:
            # First time publishing
            execute(
                "UPDATE articles SET slug=%s, title=%s, subtitle=%s, content=%s, excerpt=%s, "
                "image_url=%s, is_published=%s, published_at=NOW() WHERE id=%s",
                (slug, title, data.get("subtitle", ""), data.get("content", ""),
                 data.get("excerpt", ""), data.get("image_url", ""), is_published, article_id),
            )
        else:
            execute(
                "UPDATE articles SET slug=%s, title=%s, subtitle=%s, content=%s, excerpt=%s, "
                "image_url=%s, is_published=%s WHERE id=%s",
                (slug, title, data.get("subtitle", ""), data.get("content", ""),
                 data.get("excerpt", ""), data.get("image_url", ""), is_published, article_id),
            )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/articles/<int:article_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_article_delete(article_id):
    try:
        execute("DELETE FROM articles WHERE id = %s", (article_id,))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/articles/<int:article_id>/carousel-images")
@login_required
@admin_required
def admin_article_carousel_images(article_id):
    """Return carousel prompts, punchlines, and image URLs from DB."""
    row = query_one(
        "SELECT carousel_prompts, carousel_images, carousel_punchlines, "
        "carousel_created_at, carousel_archived_meta FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row or not row[1]:
        return jsonify({"images": [], "prompts": [], "punchlines": [], "created_at": [], "archived_meta": {}})
    prompts       = json.loads(row[0]) if row[0] else []
    images        = json.loads(row[1]) if row[1] else []
    punchlines    = json.loads(row[2]) if row[2] else []
    created_at    = json.loads(row[3]) if row[3] else []
    archived_meta = json.loads(row[4]) if row[4] else {}
    urls = [
        p if (p and p.startswith("http")) else (url_for("static", filename=p) if p else None)
        for p in images
    ]
    return jsonify({"images": urls, "prompts": prompts, "punchlines": punchlines,
                     "created_at": created_at, "archived_meta": archived_meta})


@app.route("/admin/articles/<int:article_id>/generate-carousel", methods=["POST"])
@login_required
@admin_required
def admin_article_generate_carousel(article_id):
    """
    SSE endpoint: generate up to 10 Instagram carousel images via DALL-E 3.
    Uses a storyboard approach — GPT breaks the article into 10 sequential
    narrative scenes, then DALL-E illustrates each one.

    SSE events:
      event: prompts  data: {"prompts": ["...", ...]}
      event: image    data: {"index": N, "url": "...", "prompt": "..."}
      event: error    data: {"index": N, "message": "..."}
      event: done     data: {"count": N}
    """
    if not _openai_client:
        return jsonify({"error": "OpenAI is not configured."}), 503

    row = query_one(
        "SELECT id, title, content, excerpt, carousel_prompts, carousel_images, carousel_punchlines, "
        "carousel_created_at "
        "FROM articles WHERE id = %s", (article_id,)
    )
    if not row:
        return jsonify({"error": "Article not found."}), 404

    _, title, content, excerpt, _ex_p, _ex_i, _ex_pl, _ex_ca = row
    plain = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', content or '')).strip()[:6000]

    # Read active style from site settings (fallback to hardcoded constant)
    active_style = get_setting('carousel_style', CAROUSEL_STYLE_SUFFIX)

    # Preserve archived items (index 10+) across a full regeneration
    _existing_p  = json.loads(_ex_p)  if _ex_p  else []
    _existing_i  = json.loads(_ex_i)  if _ex_i  else []
    _existing_pl = json.loads(_ex_pl) if _ex_pl else []
    _existing_ca = json.loads(_ex_ca) if _ex_ca else []
    _archived_p  = _existing_p[10:]
    _archived_i  = _existing_i[10:]
    _archived_pl = _existing_pl[10:]
    _archived_ca = _existing_ca[10:]

    # Current 10 slots (pad to length 10 with None)
    _current_p  = (_existing_p[:10]  + [None] * 10)[:10]
    _current_i  = (_existing_i[:10]  + [None] * 10)[:10]
    _current_pl = (_existing_pl[:10] + [None] * 10)[:10]
    _current_ca = (_existing_ca[:10] + [None] * 10)[:10]
    # Identify which of the first 10 slots have no image yet
    empty_slots = [idx for idx, url in enumerate(_current_i) if not url]
    N = len(empty_slots)

    def generate():
        if N == 0:
            yield f"event: done\ndata: {json.dumps({'count': 0})}\n\n"
            return

        # ── Step 1: GPT storyboard — break article into N sequential scenes ──
        try:
            resp = _openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are a visual storyteller and Instagram content strategist. "
                        f"Your job is to break an article into exactly {N} sequential scenes "
                        "that narrate the article's story from beginning to end.\n\n"
                        "For each scene, produce TWO things:\n"
                        "1. A detailed DALL-E 3 image prompt that captures that section's "
                        "key message, moment, or concept.\n"
                        "2. A short punchline (max 15 words) — a compelling caption that "
                        "gives context to the image so an Instagram viewer understands "
                        "the story beat without reading the full article.\n\n"
                        f"A viewer swiping through the {N} images and reading just the "
                        "punchlines should understand the article's narrative.\n\n"
                        "Rules:\n"
                        "- Each prompt must be specific to its section of the article (not generic)\n"
                        "- Prompts progress chronologically through the article\n"
                        "- Include concrete visual details: setting, subjects, actions, mood, colours\n"
                        f"- All {N} prompts must share the same art style for visual cohesion\n"
                        "- Punchlines should be intriguing, concise, and encourage swiping\n"
                        f"- Output ONLY a JSON array of {N} objects, each with keys "
                        "\"prompt\" and \"punchline\", no other text"
                    )},
                    {"role": "user", "content": (
                        f"Article title: {title}\n\n"
                        f"Full article content:\n{plain}\n\n"
                        f"Generate a JSON array of exactly {N} objects, each with:\n"
                        "- \"prompt\": a DALL-E 3 image prompt for that scene\n"
                        "- \"punchline\": a short Instagram caption (max 15 words)\n\n"
                        f"These {N} scenes should narrate this article as a visual Instagram "
                        "carousel, from introduction to conclusion."
                    )},
                ],
                max_tokens=3000,
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'index': 0, 'message': str(e)})}\n\n"
            yield f"event: done\ndata: {json.dumps({'count': 0})}\n\n"
            return

        # Parse JSON array of {prompt, punchline} objects from GPT response
        try:
            # Handle markdown code blocks if GPT wraps the JSON
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            scenes = json.loads(cleaned)
            if not isinstance(scenes, list):
                raise ValueError("Expected a JSON array")
            # Extract prompts and punchlines from objects
            prompts = []
            punchlines = []
            for item in scenes[:N]:
                if isinstance(item, dict):
                    prompts.append(str(item.get("prompt", "")).strip())
                    punchlines.append(str(item.get("punchline", "")).strip())
                else:
                    # Fallback: if GPT returns plain strings, use them as prompts
                    prompts.append(str(item).strip())
                    punchlines.append("")
            # Remove any empty prompts
            filtered = [(p, pl) for p, pl in zip(prompts, punchlines) if p]
            if filtered:
                prompts, punchlines = zip(*filtered)
                prompts = list(prompts)
                punchlines = list(punchlines)
            else:
                prompts, punchlines = [], []
        except (json.JSONDecodeError, ValueError) as e:
            yield f"event: error\ndata: {json.dumps({'index': 0, 'message': f'Failed to parse prompts: {e}'})}\n\n"
            yield f"event: done\ndata: {json.dumps({'count': 0})}\n\n"
            return

        if not prompts:
            yield f"event: error\ndata: {json.dumps({'index': 0, 'message': 'No prompts generated.'})}\n\n"
            yield f"event: done\ndata: {json.dumps({'count': 0})}\n\n"
            return

        # Stream prompts + punchlines to frontend (so UI can show them immediately)
        yield f"event: prompts\ndata: {json.dumps({'prompts': prompts, 'punchlines': punchlines})}\n\n"

        # ── Step 2: Create directory ──
        carousel_dir = BASE_DIR / "static" / "articles" / str(article_id) / "carousel"
        carousel_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        # Start from the current 10 slots; fill in only the empty ones
        merged_p  = list(_current_p)
        merged_i  = list(_current_i)
        merged_pl = list(_current_pl)
        merged_ca = list(_current_ca)
        slot_assignments = list(zip(empty_slots, zip(prompts, punchlines)))

        # ── Step 3: Generate each image and stream the result ──
        for seq, (slot_idx, (prompt, punchline)) in enumerate(slot_assignments, 1):
            slot_num = slot_idx + 1  # 1-based slot number for frontend/filename
            try:
                img_resp = _openai_client.images.generate(
                    model="dall-e-3",
                    prompt=f"{prompt}. {active_style}",
                    size="1024x1024",
                    quality="standard",
                    style="vivid",
                    n=1,
                )
                dl = http_requests.get(img_resp.data[0].url, timeout=30)
                dl.raise_for_status()
                local_filename = f"image_{slot_num}.png"
                gcs_object_name = f"articles/{article_id}/carousel/{local_filename}"
                (carousel_dir / local_filename).write_bytes(dl.content)
                gcs_url = upload_bytes_to_gcs(dl.content, gcs_object_name, content_type="image/png")
                if gcs_url:
                    merged_i[slot_idx] = gcs_url
                    static_url = gcs_url
                else:
                    rel_path = f"articles/{article_id}/carousel/{local_filename}"
                    merged_i[slot_idx] = rel_path
                    static_url = url_for("static", filename=rel_path)
                merged_p[slot_idx]  = prompt
                merged_pl[slot_idx] = punchline
                import datetime as _dt
                merged_ca[slot_idx] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                ok += 1
                # Persist each image immediately so nothing is lost if the stream
                # is cut by a proxy before the final batch save at step 4.
                try:
                    execute(
                        "UPDATE articles SET carousel_prompts = %s, carousel_punchlines = %s, "
                        "carousel_images = %s, carousel_style = %s, carousel_created_at = %s WHERE id = %s",
                        (json.dumps(merged_p + _archived_p),
                         json.dumps(merged_pl + _archived_pl),
                         json.dumps(merged_i  + _archived_i),
                         active_style,
                         json.dumps(merged_ca + _archived_ca), article_id),
                    )
                except Exception:
                    pass  # non-fatal — step 4 will retry
                yield f"event: image\ndata: {json.dumps({'index': slot_num, 'seq': seq, 'of': N, 'url': static_url, 'prompt': prompt, 'punchline': punchline})}\n\n"
            except _openai_module.BadRequestError as e:
                yield f"event: error\ndata: {json.dumps({'index': slot_num, 'message': f'Content policy: {e}'})}\n\n"
            except _openai_module.RateLimitError:
                yield f"event: error\ndata: {json.dumps({'index': slot_num, 'message': 'Rate limited — remaining images skipped.'})}\n\n"
                break
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'index': slot_num, 'message': str(e)})}\n\n"

        # ── Step 4: Merge new results into current slots, preserve archived (10+) ──
        try:
            execute(
                "UPDATE articles SET carousel_prompts = %s, carousel_punchlines = %s, "
                "carousel_images = %s, carousel_style = %s, carousel_created_at = %s WHERE id = %s",
                (
                    json.dumps(merged_p  + _archived_p),
                    json.dumps(merged_pl + _archived_pl),
                    json.dumps(merged_i  + _archived_i),
                    active_style,
                    json.dumps(merged_ca + _archived_ca),
                    article_id,
                ),
            )
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'index': 0, 'message': f'DB save failed: {e}'})}\n\n"

        # Activity log
        try:
            _add_activity_log(article_id, "Carousel generation",
                              f"Generated {ok} of {N} images\nModel: dall-e-3 (1024×1024, quality=standard, style=vivid)",
                              component="carousel")
        except Exception:
            pass  # non-fatal

        yield f"event: done\ndata: {json.dumps({'count': ok})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/admin/articles/<int:article_id>/archive-carousel", methods=["POST"])
@login_required
@admin_required
def admin_article_archive_carousel(article_id):
    """
    Move current carousel items (index 0–9) to the archive (index 10+).
    Slots 0–9 are set to None so the next Generate Carousel run starts fresh
    while all previous images are preserved in the Removed tab.
    """
    row = query_one(
        "SELECT carousel_prompts, carousel_images, carousel_punchlines, "
        "carousel_created_at, carousel_archived_meta FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row:
        return jsonify({"error": "Article not found."}), 404

    prompts       = json.loads(row[0]) if row[0] else []
    images        = json.loads(row[1]) if row[1] else []
    punchlines    = json.loads(row[2]) if row[2] else []
    created_at    = json.loads(row[3]) if row[3] else []
    archived_meta = json.loads(row[4]) if row[4] else {}

    # Separate current (0-9) from already-archived (10+)
    current_p  = (prompts    + [None] * 10)[:10]
    current_i  = (images     + [None] * 10)[:10]
    current_pl = (punchlines + [None] * 10)[:10]
    current_ca = (created_at + [None] * 10)[:10]

    already_archived_p  = prompts[10:]    if len(prompts)    > 10 else []
    already_archived_i  = images[10:]     if len(images)     > 10 else []
    already_archived_pl = punchlines[10:] if len(punchlines) > 10 else []
    already_archived_ca = created_at[10:] if len(created_at) > 10 else []

    # Filter out None slots — keep track of original slot index for metadata
    import datetime as _dt
    new_archived_p  = []
    new_archived_i  = []
    new_archived_pl = []
    new_archived_ca = []
    now_str = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    base_idx = 10 + len(already_archived_i)  # first index of newly archived items
    for slot_idx in range(10):
        if slot_idx < len(current_i) and current_i[slot_idx] is not None:
            new_archived_p.append(current_p[slot_idx])
            new_archived_i.append(current_i[slot_idx])
            new_archived_pl.append(current_pl[slot_idx])
            new_archived_ca.append(current_ca[slot_idx] if slot_idx < len(current_ca) else None)
            arch_idx = base_idx + len(new_archived_i) - 1
            archived_meta[str(arch_idx)] = {
                "archived_at": now_str,
                "slot_source": slot_idx + 1,
            }

    archived_count = len(new_archived_i)

    # New arrays: 10 empty slots (None) + all archived
    final_p  = [None] * 10 + already_archived_p  + new_archived_p
    final_i  = [None] * 10 + already_archived_i  + new_archived_i
    final_pl = [None] * 10 + already_archived_pl + new_archived_pl
    final_ca = [None] * 10 + already_archived_ca + new_archived_ca

    try:
        execute(
            "UPDATE articles SET carousel_prompts = %s, carousel_punchlines = %s, "
            "carousel_images = %s, carousel_created_at = %s, carousel_archived_meta = %s WHERE id = %s",
            (json.dumps(final_p), json.dumps(final_pl), json.dumps(final_i),
             json.dumps(final_ca), json.dumps(archived_meta), article_id),
        )
    except Exception as e:
        return jsonify({"error": f"DB save failed: {e}"}), 500

    # Build list of archived items with metadata for the frontend
    archived_items = []
    for i, slot_idx_offset in enumerate(range(len(new_archived_i))):
        arch_idx = base_idx + i
        meta = archived_meta.get(str(arch_idx), {})
        archived_items.append({
            "archived_idx": arch_idx,
            "url": new_archived_i[i],
            "prompt": new_archived_p[i],
            "punchline": new_archived_pl[i],
            "created_at": new_archived_ca[i],
            "archived_at": meta.get("archived_at"),
            "slot_source": meta.get("slot_source"),
        })

    return jsonify({"success": True, "archived": archived_count,
                     "archived_items": archived_items})


@app.route("/admin/articles/<int:article_id>/restore-carousel-image", methods=["POST"])
@login_required
@admin_required
def admin_article_restore_carousel_image(article_id):
    """Move an archived carousel image back to a chosen slot (auto-archiving any displaced image)."""
    body = request.get_json(silent=True) or {}
    archived_idx = body.get("archived_idx")  # absolute index in arrays (>= 10)
    target_slot  = body.get("target_slot")   # 0-9
    if archived_idx is None or archived_idx < 10:
        return jsonify({"error": "archived_idx (>= 10) required"}), 400
    if target_slot is None or target_slot < 0 or target_slot > 9:
        return jsonify({"error": "target_slot (0-9) required"}), 400

    row = query_one(
        "SELECT carousel_images, carousel_prompts, carousel_punchlines, "
        "carousel_created_at, carousel_archived_meta FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row:
        return jsonify({"error": "Article not found"}), 404

    images        = json.loads(row[0]) if row[0] else []
    prompts       = json.loads(row[1]) if row[1] else []
    punchlines    = json.loads(row[2]) if row[2] else []
    created_at    = json.loads(row[3]) if row[3] else []
    archived_meta = json.loads(row[4]) if row[4] else {}

    # Pad arrays to reach archived_idx
    while len(images)     <= archived_idx: images.append(None)
    while len(prompts)    <= archived_idx: prompts.append(None)
    while len(punchlines) <= archived_idx: punchlines.append(None)
    while len(created_at) <= archived_idx: created_at.append(None)

    if not images[archived_idx]:
        return jsonify({"error": "Archived image not found"}), 404

    restored_url        = images[archived_idx]
    restored_prompt     = prompts[archived_idx]
    restored_punchline  = punchlines[archived_idx]
    restored_created_at = created_at[archived_idx]

    # Auto-archive displaced image if target slot is occupied
    displaced = None
    if images[target_slot]:
        import datetime as _dt
        displaced_idx = len(images)  # append at end = new archived slot
        displaced_created_at = created_at[target_slot] if target_slot < len(created_at) else None
        images.append(images[target_slot])
        prompts.append(prompts[target_slot])
        punchlines.append(punchlines[target_slot])
        created_at.append(displaced_created_at)
        archived_meta[str(displaced_idx)] = {
            "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slot_source": target_slot + 1,
        }
        displaced = {
            "archived_idx": displaced_idx,
            "url": images[target_slot],
            "prompt": prompts[target_slot],
            "punchline": punchlines[target_slot],
            "created_at": displaced_created_at,
            "archived_at": archived_meta[str(displaced_idx)].get("archived_at"),
            "slot_source": target_slot + 1,
        }

    # Place restored image into target slot
    images[target_slot]     = restored_url
    prompts[target_slot]    = restored_prompt
    punchlines[target_slot] = restored_punchline
    created_at[target_slot] = restored_created_at

    # Clear the archived slot
    images[archived_idx]     = None
    prompts[archived_idx]    = None
    punchlines[archived_idx] = None
    created_at[archived_idx] = None
    archived_meta.pop(str(archived_idx), None)

    execute(
        "UPDATE articles SET carousel_images = %s, carousel_prompts = %s, carousel_punchlines = %s, "
        "carousel_created_at = %s, carousel_archived_meta = %s WHERE id = %s",
        (json.dumps(images), json.dumps(prompts), json.dumps(punchlines),
         json.dumps(created_at), json.dumps(archived_meta), article_id),
    )
    return jsonify({
        "ok": True, "url": restored_url, "slot_idx": target_slot,
        "archived_idx": archived_idx,
        "prompt": restored_prompt, "punchline": restored_punchline,
        "displaced": displaced,
    })


@app.route("/admin/articles/<int:article_id>/carousel-composed/<int:slide_index>")
@login_required
@admin_required
def admin_article_carousel_composed(article_id, slide_index):
    """
    Return a composited carousel slide JPEG (gradient + punchline + swipe hint)
    for admin preview — exactly what will be posted to Instagram.
    Original images in GCS are never modified.
    """
    row = query_one(
        "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row:
        return jsonify({"error": "Article not found"}), 404

    images     = json.loads(row[0]) if row[0] else []
    punchlines = json.loads(row[1]) if row[1] else []

    # Only the first 10 slots are active slides
    active_images     = [x for x in images[:10]     if x]
    active_punchlines = [x for x in punchlines[:10] if x is not None]
    total = len(active_images)

    if slide_index < 0 or slide_index >= total:
        return jsonify({"error": f"Slide index {slide_index} out of range (0–{total - 1})"}), 404

    img_url   = active_images[slide_index]
    punchline = active_punchlines[slide_index] if slide_index < len(active_punchlines) else ""

    # Fetch raw image bytes (GCS URL or local file)
    try:
        if img_url.startswith("https://"):
            import requests as _req
            resp = _req.get(img_url, timeout=15)
            resp.raise_for_status()
            img_bytes = resp.content
        else:
            local_path = resolve_image_to_local_path(img_url)
            if not local_path or not local_path.exists():
                return jsonify({"error": f"Image not found: {img_url}"}), 404
            img_bytes = local_path.read_bytes()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {e}"}), 500

    carousel_band_top = _compute_max_overlay_band_top(active_punchlines)
    jpeg_bytes = compose_carousel_slide(img_bytes, punchline, slide_index, total, band_top=carousel_band_top)

    from flask import Response
    return Response(jpeg_bytes, mimetype="image/jpeg",
                    headers={"Content-Disposition": f"inline; filename=slide_{slide_index + 1}_composed.jpg"})


@app.route("/admin/articles/<int:article_id>/regenerate-carousel-image", methods=["POST"])
@login_required
@admin_required
def admin_article_regenerate_carousel_image(article_id):
    """
    Re-generate a single carousel image at a given index (0-based, 0–9).
    Archives the old image/prompt/punchline to the end of the arrays.

    Request JSON: { "index": N, "prompt": "...", "punchline": "..." }
    Returns JSON: { "url": "...", "prompt": "...", "punchline": "..." }
             or:  { "error": "..." }
    """
    if not _openai_client:
        return jsonify({"error": "OpenAI is not configured."}), 503

    data = request.get_json()
    index = data.get("index")
    new_prompt = (data.get("prompt") or "").strip()
    new_punchline = (data.get("punchline") or "").strip()

    if not isinstance(index, int) or not (0 <= index <= 9):
        return jsonify({"error": "index must be an integer between 0 and 9"}), 400
    if not new_prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Load existing carousel data from DB
    row = query_one(
        "SELECT carousel_prompts, carousel_images, carousel_punchlines, "
        "carousel_created_at, carousel_archived_meta FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row:
        return jsonify({"error": "Article not found."}), 404

    prompts       = json.loads(row[0]) if row[0] else []
    images        = json.loads(row[1]) if row[1] else []
    punchlines    = json.loads(row[2]) if row[2] else []
    created_at    = json.loads(row[3]) if row[3] else []
    archived_meta = json.loads(row[4]) if row[4] else {}

    # Pad arrays to at least index + 1 so we can safely access/replace by index
    while len(prompts) <= index:
        prompts.append(None)
    while len(images) <= index:
        images.append(None)
    while len(punchlines) <= index:
        punchlines.append(None)
    while len(created_at) <= index:
        created_at.append(None)

    # Save old values before replacing
    old_prompt     = prompts[index]
    old_image      = images[index]
    old_punchline  = punchlines[index]
    old_created_at = created_at[index]

    # Read active style from site settings (fallback to hardcoded constant)
    active_style = get_setting('carousel_style', CAROUSEL_STYLE_SUFFIX)

    # ── Generate new image via DALL-E 3 ──
    carousel_dir = BASE_DIR / "static" / "articles" / str(article_id) / "carousel"
    carousel_dir.mkdir(parents=True, exist_ok=True)
    try:
        img_resp = _openai_client.images.generate(
            model="dall-e-3",
            prompt=f"{new_prompt}. {active_style}",
            size="1024x1024",
            quality="standard",
            style="vivid",
            n=1,
        )
        dl = http_requests.get(img_resp.data[0].url, timeout=30)
        dl.raise_for_status()
    except _openai_module.BadRequestError as e:
        _add_activity_log(article_id, f"Carousel re-run slide {index + 1} failed",
                          f"Content policy: {e}", component="carousel")
        return jsonify({"error": f"Content policy: {e}"}), 400
    except _openai_module.RateLimitError:
        _add_activity_log(article_id, f"Carousel re-run slide {index + 1} failed",
                          "Rate limited", component="carousel")
        return jsonify({"error": "Rate limited — please try again later."}), 429
    except Exception as e:
        _add_activity_log(article_id, f"Carousel re-run slide {index + 1} failed",
                          f"Error: {e}", component="carousel")
        return jsonify({"error": str(e)}), 500

    # Save with timestamp so old file (used by archive) is not overwritten
    import time as _time
    ts = int(_time.time())
    filename = f"image_{index + 1}_{ts}.png"
    gcs_object_name = f"articles/{article_id}/carousel/{filename}"
    (carousel_dir / filename).write_bytes(dl.content)
    gcs_url = upload_bytes_to_gcs(dl.content, gcs_object_name, content_type="image/png")

    # ── Archive old values + replace current slot ──
    import datetime as _dt
    prompts[index] = new_prompt
    images[index] = gcs_url if gcs_url else f"articles/{article_id}/carousel/{filename}"
    punchlines[index] = new_punchline
    created_at[index] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    archived_idx = None
    if old_image is not None:
        archived_idx = len(images)  # index of the new archived position
        prompts.append(old_prompt)
        images.append(old_image)
        punchlines.append(old_punchline)
        created_at.append(old_created_at)
        archived_meta[str(archived_idx)] = {
            "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slot_source": index + 1,
        }

    # ── Persist to DB ──
    try:
        execute(
            "UPDATE articles SET carousel_prompts = %s, carousel_punchlines = %s, "
            "carousel_images = %s, carousel_style = %s, carousel_created_at = %s, "
            "carousel_archived_meta = %s WHERE id = %s",
            (json.dumps(prompts), json.dumps(punchlines), json.dumps(images),
             active_style, json.dumps(created_at), json.dumps(archived_meta), article_id),
        )
    except Exception as e:
        return jsonify({"error": f"DB save failed: {e}"}), 500

    _add_activity_log(article_id, f"Carousel re-run slide {index + 1}",
                      f"Model: dall-e-3\nPrompt: {new_prompt[:120]}...",
                      component="carousel")
    static_url = gcs_url if gcs_url else url_for("static", filename=f"articles/{article_id}/carousel/{filename}")
    return jsonify({
        "url": static_url, "prompt": new_prompt, "punchline": new_punchline,
        "archived_idx": archived_idx,
        "old_image": old_image, "old_prompt": old_prompt, "old_punchline": old_punchline,
        "old_created_at": old_created_at,
        "archived_at": archived_meta.get(str(archived_idx), {}).get("archived_at") if archived_idx else None,
        "slot_source": index + 1,
    })


# ------------------------------------------------------------
# ADMIN — VIDEO GENERATION (shared data + narrated + AI)
# ------------------------------------------------------------

@app.route("/admin/articles/<int:article_id>/video-data")
@login_required
@admin_required
def admin_article_video_data(article_id):
    """Return existing video URLs and narration script from DB (shared by Components B & D)."""
    row = query_one(
        "SELECT video_narrated_url, video_narrated_script, video_narrated_runs, "
        "video_narrated_status, carousel_cinemagraphs, carousel_cinemagraph_log, "
        "carousel_cinemagraph_prompts, carousel_cinemagraph_archived, "
        "carousel_cinemagraph_created_at, video_narrated_log "
        "FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row:
        return jsonify({"error": "Article not found"}), 404
    (narrated_url, narrated_script_raw, narrated_runs_raw, narrated_status,
     cinemagraphs_raw, cinemagraph_log_raw, cinemagraph_prompts_raw, cinemagraph_archived_raw,
     cinemagraph_created_at_raw, narrated_log_raw) = row
    narrated_script          = json.loads(narrated_script_raw)          if narrated_script_raw          else []
    narrated_runs            = json.loads(narrated_runs_raw)            if narrated_runs_raw            else []
    cinemagraph_urls         = json.loads(cinemagraphs_raw)             if cinemagraphs_raw             else []
    cinemagraph_prompts      = json.loads(cinemagraph_prompts_raw)      if cinemagraph_prompts_raw      else []
    cinemagraph_archived     = json.loads(cinemagraph_archived_raw)     if cinemagraph_archived_raw     else []
    cinemagraph_created_at   = json.loads(cinemagraph_created_at_raw)   if cinemagraph_created_at_raw   else []
    cinemagraph_status   = get_setting(f"cinemagraph_status_{article_id}")
    cinemagraph_result   = get_setting(f"cinemagraph_result_{article_id}")
    return jsonify({
        "narrated_url":           narrated_url or None,
        "narrated_script":        narrated_script,
        "narrated_runs":          narrated_runs,
        "narrated_status":        narrated_status or None,
        "cinemagraph_urls":       cinemagraph_urls,
        "cinemagraph_prompts":    cinemagraph_prompts,
        "cinemagraph_archived":   cinemagraph_archived,
        "cinemagraph_created_at": cinemagraph_created_at,
        "cinemagraph_status":     cinemagraph_status or None,
        "cinemagraph_result":     cinemagraph_result or None,
        "has_cinemagraph_log":    bool(cinemagraph_log_raw and cinemagraph_log_raw.strip()),
        "has_narrated_log":       bool(narrated_log_raw and narrated_log_raw.strip()),
        "hidden_narrated_runs":   json.loads(get_setting(f"narrated_hidden_runs_{article_id}") or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/toggle-narrated-run-visibility", methods=["POST"])
@login_required
@admin_required
def admin_toggle_narrated_run_visibility(article_id):
    data = request.get_json() or {}
    ts = data.get("ts")
    hidden = data.get("hidden", True)
    if not ts:
        return jsonify({"error": "Missing ts"}), 400
    key = f"narrated_hidden_runs_{article_id}"
    current = json.loads(get_setting(key) or "[]")
    ts = int(ts)
    if hidden and ts not in current:
        current.append(ts)
    elif not hidden and ts in current:
        current.remove(ts)
    val = json.dumps(current)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (key, val, val))
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/component-log/<component>")
@login_required
@admin_required
def admin_article_component_log(article_id, component):
    """Return per-component run log + filtered activity entries."""
    # Run log from dedicated TEXT column (if the component has one)
    col_map = {
        "cinemagraph": "carousel_cinemagraph_log",
        "narrated":    "video_narrated_log",
    }
    run_log = None
    col = col_map.get(component)
    if col:
        row = query_one(
            f"SELECT `{col}` FROM articles WHERE id = %s", (article_id,)
        )
        run_log = row[0] if row and row[0] else None

    # Activity entries filtered by component tag.
    # Legacy entries (written before the component field was added) are classified
    # by title so they still appear in the correct component's log.
    def _infer_component(entry):
        title = entry.get("title", "")
        if "Cinemagraph" in title or "(cine)" in title:
            return "cinemagraph"
        if "Narrated" in title:
            return "narrated"
        return "carousel"

    key = f"ig_activity_log_{article_id}"
    raw = get_setting(key)
    all_entries = json.loads(raw) if raw else []
    entries = [
        e for e in all_entries
        if (e.get("component") or _infer_component(e)) == component
    ]

    return jsonify({"run_log": run_log, "entries": entries})


@app.route("/admin/articles/<int:article_id>/cinemagraph-result", methods=["DELETE"])
@login_required
@admin_required
def admin_article_clear_cinemagraph_result(article_id):
    """Clear the cinemagraph generation result so it doesn't persist on reload."""
    execute("DELETE FROM site_settings WHERE `key` = %s",
            (f"cinemagraph_result_{article_id}",))
    return jsonify({"success": True})




def _narrated_video_worker(article_id, cfg):
    """Background thread: GPT script → OpenAI TTS → FFmpeg per-clip → FFmpeg concat → save MP4."""
    fmt        = cfg.get("format",     "vertical")
    voice      = cfg.get("voice",      "onyx")
    tts_model  = cfg.get("tts_model",  "tts-1")
    script_len = cfg.get("script_len", "medium")
    kb_speed   = cfg.get("kb_speed",   "slow")
    crf        = int(cfg.get("crf",    23))
    fps        = int(cfg.get("fps",    25))

    W, H         = (720, 1280) if fmt == "vertical" else (720, 720)
    render_fps   = 12   # Ken Burns on still images looks identical at 12fps vs 25fps
    speed_factor = 1.35  # Speed up final video for social-media pacing
    zoom_step    = {"slow": 0.0010, "medium": 0.0015, "fast": 0.0025}.get(kb_speed, 0.0010)
    word_range = {"short": "20-40", "medium": "40-70", "long": "70-100"}.get(script_len, "40-70")

    # ── Log capture (mirrors cinemagraph pattern) ────────────────────────────
    log_lines = []
    ts_start  = time.strftime("%Y-%m-%d %H:%M:%S")

    def _log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_lines.append(line)
        print(f"NarratedVideo: {msg}", flush=True)

    def _flush_log():
        """Write current log_lines to DB so the activity-log modal shows live progress."""
        if log_lines:
            log_text = f"Run started: {ts_start}\n" + "\n".join(log_lines)
            try:
                execute("UPDATE articles SET video_narrated_log = %s WHERE id = %s",
                        (log_text[:65000], article_id))
            except Exception:
                pass
    # ─────────────────────────────────────────────────────────────────────────

    row = query_one(
        "SELECT carousel_images, carousel_prompts, carousel_punchlines, title, carousel_cinemagraphs FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row or not row[0]:
        _log("ERROR: No carousel images found.")
        _flush_log()
        execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                ("error:No carousel images found.", article_id))
        return

    images    = (json.loads(row[0]) if row[0] else [])[:10]
    prompts   = (json.loads(row[1]) if row[1] else [])[:10]
    punchlines= (json.loads(row[2]) if row[2] else [])[:10]
    article_title = row[3] if row[3] else "Unknown"
    cinemagraphs  = (json.loads(row[4]) if row[4] else [])[:10]
    n_slides  = len(images)
    n_cine    = sum(1 for c in cinemagraphs if c)

    _log(f"Article: \"{article_title}\", {n_slides} slides, {n_cine} cinemagraphs")
    _log(f"Settings: format={fmt}, voice={voice}, tts={tts_model}, script={script_len}, kb={kb_speed}, crf={crf}, fps={fps}, render_fps={render_fps}, res={W}x{H}, speed={speed_factor}x")

    ts        = int(time.time())
    video_dir = Path(f"static/articles/{article_id}/video")
    audio_dir = video_dir / "audio"
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    temp_files = []

    try:
        # ── 1. GPT narration script ────────────────────────────────────────────
        execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                ("running:script", article_id))
        _log("Generating narration script via GPT…")
        scenes_desc = "\n".join(
            f"Slide {i+1}: {prompts[i] if i < len(prompts) else '(no prompt)'} — "
            f"Punchline: {punchlines[i] if i < len(punchlines) else ''}"
            for i in range(n_slides)
        )
        script_prompt = (
            f"You are writing a voiceover script for a {n_slides}-slide Instagram carousel "
            f'about the article: "{article_title}".\n\n'
            f"For each slide, write a narration segment of {word_range} words that matches "
            f"the scene description and punchline. The narration should sound natural when "
            f"read aloud as a voiceover.\n\n"
            f'Return ONLY a JSON object in this exact format: {{"segments": ["text for slide 1", "text for slide 2", ...]}}\n\n'
            f"Slide descriptions:\n{scenes_desc}"
        )
        gpt_resp = _openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": script_prompt}],
            response_format={"type": "json_object"},
        )
        raw_json = gpt_resp.choices[0].message.content
        _log(f"GPT raw response: {raw_json[:500]}")
        parsed   = json.loads(raw_json)
        if isinstance(parsed, dict):
            segments = parsed.get("segments")
            # Fallback: grab first value that is actually a list
            if not isinstance(segments, list):
                segments = next((v for v in parsed.values() if isinstance(v, list)), None)
            # Last resort: split a string value into one entry per slide
            if not isinstance(segments, list):
                first_val = list(parsed.values())[0] if parsed else ""
                segments = [str(first_val)] if first_val else []
        else:
            segments = parsed if isinstance(parsed, list) else []
        segments = [str(s) for s in segments[:n_slides]]
        while len(segments) < n_slides:
            segments.append(punchlines[len(segments)] if len(segments) < len(punchlines) else "")

        # Validate: if any segment has fewer than 3 words, replace with punchline
        for i, seg in enumerate(segments):
            if len(seg.split()) < 3:
                segments[i] = punchlines[i] if i < len(punchlines) else f"Slide {i+1}."
                _log(f"Segment {i+1} too short, replaced with fallback: {segments[i]}")

        _log(f"Final segments ({len(segments)}): {segments}")
        _flush_log()

        # ── 2. TTS per segment ─────────────────────────────────────────────────
        audio_paths = []
        for i, seg in enumerate(segments):
            execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                    (f"running:audio:{i+1}/{n_slides}", article_id))
            audio_path = audio_dir / f"segment_{i+1}.mp3"
            tts_resp = _openai_client.audio.speech.create(
                model=tts_model, voice=voice, input=seg, response_format="mp3"
            )
            audio_path.write_bytes(tts_resp.content)
            _log(f"TTS {i+1}/{n_slides}: {len(seg.split())} words, {len(tts_resp.content)} bytes")
            _flush_log()
            audio_paths.append(audio_path)
            temp_files.append(audio_path)

        # ── 3. FFmpeg binary ───────────────────────────────────────────────────
        import shutil
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as ff_err:
                _log(f"imageio_ffmpeg fallback failed: {ff_err}")
        if not ffmpeg_exe:
            _log("ERROR: FFmpeg not found (system or imageio_ffmpeg)")
            execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                    ("error:FFmpeg not found.", article_id))
            return
        _log(f"Using FFmpeg: {ffmpeg_exe}")
        _flush_log()

        def _run_ffmpeg(cmd, label, timeout=120):
            """Run an FFmpeg command with timeout and limited output capture."""
            _log(f"FFmpeg [{label}]: {' '.join(str(c) for c in cmd[:8])}…")
            try:
                result = subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                _log(f"FFmpeg [{label}] TIMED OUT after {timeout}s")
                return None, "Timed out"
            if result.returncode != 0:
                return None, result.stderr[-300:] if result.stderr else "Unknown error"
            return result, None

        def probe_duration(mp3_path):
            # Try ffprobe first — it is lighter and faster than ffmpeg -i
            ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe") if ffmpeg_exe else "ffprobe"
            try:
                result = subprocess.run(
                    [ffprobe_exe, "-v", "quiet",
                     "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1",
                     str(mp3_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, timeout=10,
                )
                return float(result.stdout.strip())
            except Exception:
                pass
            # Fallback: ffmpeg -i (reads duration from header)
            try:
                result = subprocess.run(
                    [ffmpeg_exe, "-i", str(mp3_path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=30,
                )
                m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
                if m:
                    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    return h * 3600 + mn * 60 + s
            except Exception as e:
                _log(f"probe_duration failed: {e}")
            return 5.0

        # ── 4. FFmpeg per-clip (cinemagraph or Ken Burns + audio) ────────────
        clip_paths = []
        for i, img_url in enumerate(images):
            execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                    (f"running:clip:{i+1}/{n_slides}", article_id))

            audio_path = audio_paths[i]
            dur        = probe_duration(audio_path)
            dur_video  = dur + 0.3
            clip_path  = video_dir / f"clip_{i+1}.mp4"
            temp_files.append(clip_path)

            # ── Check for cinemagraph ─────────────────────────────────────
            cine_url  = cinemagraphs[i] if i < len(cinemagraphs) else None
            use_cine  = False
            cine_path = None
            if cine_url:
                cine_path = resolve_image_to_local_path(cine_url)
                if cine_path and cine_path.exists():
                    use_cine = True
                    _log(f"Clip {i+1}: cinemagraph resolved → {cine_path} ({cine_path.stat().st_size} bytes)")
                else:
                    _log(f"Clip {i+1}: cinemagraph not resolved ({cine_url}), falling back to Ken Burns")

            if use_cine:
                # ── Cinemagraph: loop video + scale/crop to target size ───
                S = max(W, H)
                cine_filter = (
                    f"[0:v]fps={render_fps},"
                    f"scale={S}:{S}:flags=fast_bilinear,"
                    f"crop={W}:{H},"
                    f"setsar=1[v]"
                )
                cmd = [
                    ffmpeg_exe, "-y",
                    "-stream_loop", "-1",
                    "-t", str(dur_video),
                    "-i", str(cine_path),
                    "-i", str(audio_path),
                    "-filter_complex", cine_filter,
                    "-map", "[v]", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", str(crf),
                    "-c:a", "aac", "-shortest",
                    str(clip_path)
                ]
                result, err = _run_ffmpeg(cmd, f"clip {i+1}/{n_slides} (cine)", timeout=240)
                if err:
                    _log(f"Clip {i+1} cinemagraph FAILED: {err}")
                    execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                            (f"error:FFmpeg clip {i+1} cinemagraph failed: {err[:200]}", article_id))
                    return
                _log(f"Clip {i+1}/{n_slides}: cinemagraph, dur={dur:.1f}s, rendered OK")
            else:
                # ── Ken Burns: scale + animated crop on still image ────────
                img_path = resolve_image_to_local_path(img_url)
                if not img_path or not img_path.exists():
                    _log(f"ERROR: Image not found: {img_url}")
                    execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                            (f"error:Image not found: {img_url}", article_id))
                    return
                _log(f"Clip {i+1}: image resolved → {img_path} ({img_path.stat().st_size} bytes)")

                zoom_factor = min(1.0 + zoom_step * 200, 1.3)
                ZW = int(W * zoom_factor)
                ZH = int(H * zoom_factor)
                px = f"({ZW}-{W})*(1-t/{dur_video:.4f})"
                py = f"({ZH}-{H})*(1-t/{dur_video:.4f})"
                zp_filter = (
                    f"[0:v]"
                    f"fps={render_fps},"
                    f"scale={ZW}:{ZH}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                    f"crop={ZW}:{ZH},"
                    f"crop={W}:{H}:x='{px}':y='{py}',"
                    f"setsar=1[v]"
                )
                cmd = [
                    ffmpeg_exe, "-y",
                    "-framerate", str(render_fps),
                    "-loop", "1", "-t", str(dur_video), "-i", str(img_path),
                    "-i", str(audio_path),
                    "-filter_complex", zp_filter,
                    "-map", "[v]", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
                    "-crf", str(crf),
                    "-c:a", "aac", "-shortest",
                    str(clip_path)
                ]
                result, err = _run_ffmpeg(cmd, f"clip {i+1}/{n_slides} (kb)", timeout=120)
                if err:
                    _log(f"Clip {i+1} FAILED: {err}")
                    execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                            (f"error:FFmpeg clip {i+1} failed: {err[:200]}", article_id))
                    return
                _log(f"Clip {i+1}/{n_slides}: Ken Burns, dur={dur:.1f}s, rendered OK")

            _flush_log()
            clip_paths.append(clip_path)

        # ── 5. Concat all clips ────────────────────────────────────────────────
        execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                ("running:final", article_id))
        _log("Concatenating clips…")
        concat_txt = video_dir / f"concat_{ts}.txt"
        temp_files.append(concat_txt)
        concat_txt.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in clip_paths), encoding="utf-8"
        )
        out_path   = video_dir / f"narrated_{ts}.mp4"
        cmd_concat = [
            ffmpeg_exe, "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-c", "copy", str(out_path)
        ]
        result, err = _run_ffmpeg(cmd_concat, "concat", timeout=60)
        if err:
            _log(f"Concat FAILED: {err}")
            execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                    (f"error:FFmpeg concat failed: {err[:200]}", article_id))
            return
        _flush_log()

        # ── 5b. Speed up final video for social-media pacing ─────────────────
        if speed_factor and speed_factor != 1.0:
            _log(f"Speeding up video by {speed_factor}x…")
            fast_path = video_dir / f"narrated_{ts}_fast.mp4"
            speed_filter = (
                f"[0:v]setpts=PTS/{speed_factor:.4f}[v];"
                f"[0:a]atempo={speed_factor:.4f}[a]"
            )
            cmd_speed = [
                ffmpeg_exe, "-y",
                "-i", str(out_path),
                "-filter_complex", speed_filter,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", str(crf),
                "-c:a", "aac",
                str(fast_path)
            ]
            result, err = _run_ffmpeg(cmd_speed, "speed-up", timeout=180)
            if err:
                _log(f"Speed-up FAILED: {err} — using original speed")
            else:
                _log(f"Speed-up OK ({speed_factor}x)")
                temp_files.append(out_path)   # clean up the slow version
                temp_files.append(fast_path)  # fast version cleaned up after GCS upload
                out_path = fast_path
            _flush_log()

        # ── 6. Upload to GCS + save run to history ──────────────────────────────
        gcs_obj = f"articles/{article_id}/video/narrated_{ts}.mp4"
        gcs_url = upload_to_gcs(out_path, gcs_obj, content_type="video/mp4")
        static_url = gcs_url if gcs_url else ("/" + str(out_path).replace("\\", "/"))
        if gcs_url:
            temp_files.append(out_path)  # clean up local copy; it's in GCS now
            _log(f"Uploaded to GCS: {gcs_url}")
        else:
            _log(f"Saved locally: {static_url}")
        from datetime import datetime
        ts_label = datetime.now().strftime("%d %b %Y, %H:%M").lstrip("0")
        run_obj  = {
            "ts": ts, "ts_label": ts_label, "url": static_url,
            "params": {"format": fmt, "voice": voice, "tts_model": tts_model,
                       "script_len": script_len, "kb_speed": kb_speed, "crf": crf, "fps": fps,
                       "speed": speed_factor},
            "segments": segments,
            "cinemagraph_slides": [i+1 for i in range(n_slides) if i < len(cinemagraphs) and cinemagraphs[i]],
        }

        existing_row      = query_one(
            "SELECT video_narrated_runs, video_narrated_url, video_narrated_script "
            "FROM articles WHERE id = %s", (article_id,)
        )
        existing_runs_raw = existing_row[0] if existing_row else None
        legacy_url        = existing_row[1] if existing_row else None
        legacy_script_raw = existing_row[2] if existing_row else None
        existing_runs     = json.loads(existing_runs_raw) if existing_runs_raw else []

        if legacy_url and not any(r.get("url") == legacy_url for r in existing_runs):
            legacy_script = json.loads(legacy_script_raw) if legacy_script_raw else []
            existing_runs.append({
                "ts": 0, "ts_label": "Previous run", "url": legacy_url,
                "params": {"format": "square", "voice": "onyx", "tts_model": "tts-1",
                           "script_len": "medium", "kb_speed": "slow", "crf": 23, "fps": 25},
                "segments": legacy_script,
            })

        all_runs = [run_obj] + existing_runs
        execute(
            "UPDATE articles SET video_narrated_url = %s, video_narrated_script = %s, "
            "video_narrated_runs = %s, video_narrated_status = NULL WHERE id = %s",
            (static_url, json.dumps(segments), json.dumps(all_runs), article_id),
        )
        _log(f"DONE — {n_slides} slides, video: {static_url}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _log(f"ERROR: {e}")
        execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                (f"error:{str(e)[:200]}", article_id))
    finally:
        # ── Persist run log to DB and GCS ────────────────────────────────────
        try:
            if log_lines:
                log_text = f"Run started: {ts_start}\n" + "\n".join(log_lines)
                # GCS (best-effort)
                gcs_log_obj = f"articles/{article_id}/video/narrated_run_{ts}.log"
                upload_bytes_to_gcs(log_text.encode(), gcs_log_obj, content_type="text/plain")
                try:
                    execute("UPDATE articles SET video_narrated_log = %s WHERE id = %s",
                            (log_text[:65000], article_id))
                except Exception as db_err:
                    print(f"NarratedVideo: Could not save log to DB: {db_err}", flush=True)
        except Exception as log_err:
            print(f"NarratedVideo: Log persistence failed: {log_err}", flush=True)

        # ── Activity log (brief summary — full log lives in video_narrated_log) ─
        try:
            last_line = log_lines[-1] if log_lines else "No log output"
            _add_activity_log(article_id, "Narrated video generation",
                              f"Voice: {voice}, Format: {fmt}, Script: {script_len}\n{last_line}",
                              component="narrated")
        except Exception as al_err:
            print(f"NarratedVideo: Activity log failed: {al_err}", flush=True)

        # ── Clean up temp files ──────────────────────────────────────────────
        for f in temp_files:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass


@app.route("/admin/articles/<int:article_id>/generate-narrated-video", methods=["POST"])
@login_required
@admin_required
def admin_article_generate_narrated_video(article_id):
    """Start a background thread for narrated MP4 generation; return immediately."""
    if not _openai_client:
        return jsonify({"error": "OpenAI client not initialised."}), 400

    row = query_one("SELECT video_narrated_status FROM articles WHERE id = %s", (article_id,))
    if row and row[0] and row[0].startswith("running"):
        return jsonify({"error": "already_running"}), 409

    row = query_one("SELECT carousel_images FROM articles WHERE id = %s", (article_id,))
    if not row or not row[0]:
        return jsonify({"error": "No carousel images found."}), 400
    images = (json.loads(row[0]) if row[0] else [])[:10]
    if not images:
        return jsonify({"error": "No carousel images found."}), 400

    cfg = request.get_json() or {}
    execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
            ("running:script", article_id))
    t = threading.Thread(target=_narrated_video_worker, args=(article_id, cfg), daemon=True)
    t.start()
    return jsonify({"status": "started", "n_slides": len(images)})


@app.route("/admin/articles/<int:article_id>/stop-narrated-video", methods=["POST"])
@login_required
@admin_required
def admin_article_stop_narrated_video(article_id):
    """Reset narrated-video status so the user can re-trigger generation."""
    execute(
        "UPDATE articles SET video_narrated_status = NULL WHERE id = %s",
        (article_id,),
    )
    return jsonify({"ok": True})


def _cinemagraph_worker(article_id, images, prompts=None,
                        slot_indices=None, existing_urls=None):
    """Background thread: Luma AI per carousel slide → store MP4 GCS URLs in DB.

    Args:
        images:         list of source image URLs to generate clips for
        prompts:        parallel list of per-slide prompts
        slot_indices:   list mapping each item in ``images`` to a 0-based slot
                        in the 10-slot cinemagraph array (e.g. [0, 3] means we
                        are generating clips for slots 1 and 4 only)
        existing_urls:  full 10-slot cinemagraph URL array (to merge new results into)
    """
    n = len(images)
    ts = int(time.time())
    status_key = f"cinemagraph_status_{article_id}"

    def _set_status(s):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, s, s))

    def _clear_status():
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    result_key = f"cinemagraph_result_{article_id}"
    def _set_result(s):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, s, s))

    # ── Cancel support ────────────────────────────────────────────────────────
    cancel_key = f"cinemagraph_cancel_{article_id}"
    execute("DELETE FROM site_settings WHERE `key` = %s", (cancel_key,))

    # ── Log capture ────────────────────────────────────────────────────────────
    log_lines = []
    ts_start  = time.strftime("%Y-%m-%d %H:%M:%S")

    def _log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_lines.append(line)
        print(f"Cinemagraph worker: {msg}", flush=True)

    def _flush_log():
        """Write current log_lines to DB so the log modal shows live progress."""
        if log_lines:
            log_text = f"Run started: {ts_start}\n" + "\n".join(log_lines)
            try:
                execute("UPDATE articles SET carousel_cinemagraph_log = %s WHERE id = %s",
                        (log_text[:65000], article_id))
            except Exception:
                pass

    def _is_cancelled():
        return bool(get_setting(cancel_key))
    # ──────────────────────────────────────────────────────────────────────────

    # Start from existing arrays so we preserve clips that aren't being regenerated
    cinemagraph_urls  = list(existing_urls  or [])[:10]
    while len(cinemagraph_urls)  < 10: cinemagraph_urls.append(None)
    # Load created_at timestamps for active slots
    _ca_row = query_one("SELECT carousel_cinemagraph_created_at FROM articles WHERE id = %s", (article_id,))
    cinemagraph_created_at = json.loads(_ca_row[0]) if _ca_row and _ca_row[0] else []
    while len(cinemagraph_created_at) < 10: cinemagraph_created_at.append(None)
    new_succeeded = 0
    consecutive_failures = 0
    billing_abort = False
    billing_abort_msg = ""

    try:
        for i, img_url in enumerate(images):
            real_idx = slot_indices[i] if slot_indices else i
            slide_num = real_idx + 1  # 1-based for display

            if _is_cancelled():
                _log("Cancelled by user")
                break
            _set_status(f"running:{i}/{n}")
            _flush_log()
            _log(f"slide {slide_num} ({i+1}/{n}) submitting to Luma")

            # Resolve image URL for Luma (needs public URL)
            if img_url.startswith("https://"):
                luma_img_url = img_url
                _log(f"slide {slide_num} using GCS URL")
            else:
                img_path = resolve_image_to_local_path(img_url)
                if img_path and img_path.exists():
                    gcs_obj = f"articles/{article_id}/temp/slide_{slide_num}_{ts}.jpg"
                    luma_img_url = upload_to_gcs(img_path, gcs_obj, content_type="image/jpeg")
                    if not luma_img_url:
                        _log(f"slide {slide_num} local image upload to GCS failed")
                        continue
                    _log(f"slide {slide_num} using local file → uploaded to GCS")
                else:
                    _log(f"slide {slide_num} image not found: {img_url}")
                    continue

            slide_prompt = (prompts[i] if prompts and i < len(prompts) and prompts[i] else None) or get_setting(f'cinemagraph_prompt_{article_id}') or get_setting('cinemagraph_prompt', _CINE_DEFAULT_PROMPT)
            _log(f"slide {slide_num} prompt: {slide_prompt[:80]}")
            try:
                gen_id = _luma_create_task(luma_img_url, slide_prompt)
                _log(f"slide {slide_num} Luma gen_id={gen_id}")
            except Exception as e:
                _log(f"slide {slide_num} Luma submit FAILED: {e}")
                if _is_luma_billing_error(e):
                    _log("Non-transient billing error — skipping remaining slides")
                    billing_abort = True
                    billing_abort_msg = str(e)
                    break
                continue

            # Poll Luma until the clip is ready
            mp4_url = None
            poll_count = 0
            last_sub_status = None
            while True:
                time.sleep(6)
                if _is_cancelled():
                    _log(f"slide {slide_num} cancelled by user during poll")
                    break
                poll_count += 1
                try:
                    luma_status, mp4_url, status_msg = _luma_poll_task(gen_id)
                except Exception as e:
                    _log(f"slide {slide_num} poll error: {e}")
                    break
                if luma_status == "completed":
                    _log(f"slide {slide_num} ({i+1}/{n}) Luma SUCCEEDED after {poll_count} polls")
                    break
                elif luma_status == "failed":
                    _log(f"slide {slide_num} ({i+1}/{n}) Luma FAILED (reason: {status_msg})")
                    mp4_url = None
                    break
                else:
                    _log(f"slide {slide_num} still generating (poll {poll_count}, status={luma_status})")
                    if luma_status != last_sub_status:
                        _set_status(f"running:{i}/{n}:{luma_status}")
                        _flush_log()
                        last_sub_status = luma_status

            if mp4_url:
                try:
                    r = http_requests.get(mp4_url, timeout=120)
                    # Save locally first — needed as fallback when GCS is not configured
                    local_dir = Path(f"static/articles/{article_id}/cinemagraph")
                    local_dir.mkdir(parents=True, exist_ok=True)
                    local_path = local_dir / f"slide_{slide_num}_{ts}.mp4"
                    local_path.write_bytes(r.content)
                    # Upload to GCS; fall back to local /static/... URL if unavailable
                    gcs_obj = f"articles/{article_id}/cinemagraph/slide_{slide_num}_{ts}.mp4"
                    gcs_url = upload_bytes_to_gcs(r.content, gcs_obj, content_type="video/mp4")
                    if gcs_url:
                        local_path.unlink(missing_ok=True)  # GCS has it — clean up local copy
                        saved_url = gcs_url
                    else:
                        saved_url = "/" + str(local_path).replace("\\", "/")
                    import datetime as _dt
                    cinemagraph_urls[real_idx] = saved_url
                    cinemagraph_created_at[real_idx] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                    new_succeeded += 1
                    consecutive_failures = 0
                    _log(f"slide {slide_num} ({i+1}/{n}) saved → {saved_url}")
                    # Persist immediately so the frontend picks it up on the next poll
                    execute(
                        "UPDATE articles SET carousel_cinemagraphs = %s, "
                        "carousel_cinemagraph_created_at = %s WHERE id = %s",
                        (json.dumps(cinemagraph_urls),
                         json.dumps(cinemagraph_created_at), article_id),
                    )
                except Exception as e:
                    _log(f"slide {slide_num} save FAILED: {e}")
                    consecutive_failures += 1
            else:
                if not _is_cancelled():
                    consecutive_failures += 1

            if consecutive_failures >= 3:
                _log(f"ABORTING — {consecutive_failures} consecutive failures, stopping to prevent further cost")
                break

            _set_status(f"running:{i+1}/{n}")
            _flush_log()

        if billing_abort:
            _log(f"ABORTED — billing error after {new_succeeded}/{n} new clips: {billing_abort_msg}")
            _set_result(f"error:Luma billing — not enough credits to complete generation")
        elif consecutive_failures >= 3:
            _log(f"ABORTED — {consecutive_failures} consecutive failures after {new_succeeded}/{n} new clips")
            _set_result(f"error:Aborted after {consecutive_failures} consecutive failures — check logs for details")
        elif new_succeeded == 0:
            _set_result("error:All slides failed — check logs for details")
        else:
            _log(f"DONE — {new_succeeded}/{n} new clips generated")
            _set_result(f"done:{new_succeeded}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _log(f"EXCEPTION: {e}")
        try:
            _set_result(f"error:{e}")
        except Exception:
            pass
    finally:
        _clear_status()
        execute("DELETE FROM site_settings WHERE `key` = %s", (cancel_key,))
        # ── Persist run log to DB and GCS ──────────────────────────────────────
        if log_lines:
            log_text = f"Run started: {ts_start}\n" + "\n".join(log_lines)
            # GCS (best-effort — may be None in local dev)
            gcs_log_obj = f"articles/{article_id}/cinemagraph/run_{ts}.log"
            upload_bytes_to_gcs(log_text.encode(), gcs_log_obj, content_type="text/plain")
            # Always save to DB (truncated to 64 KB to stay within TEXT column limit)
            execute(
                "UPDATE articles SET carousel_cinemagraph_log = %s WHERE id = %s",
                (log_text[:65000], article_id),
            )
        # ──────────────────────────────────────────────────────────────────────
        summary = "\n".join(log_lines) if log_lines else "No log output"
        _add_activity_log(article_id, "Cinemagraph generation",
                          "Model: luma ray-2 (1:1, 5s)\n" + summary,
                          component="cinemagraph")


@app.route("/admin/articles/<int:article_id>/generate-cinemagraphs", methods=["POST"])
@login_required
@admin_required
def admin_article_generate_cinemagraphs(article_id):
    """Start a background thread for per-slide cinemagraph generation via Luma AI."""
    if not _luma_enabled:
        return jsonify({"error": "Luma AI not configured. Set LUMAAI_API_KEY."}), 400

    status_key = f"cinemagraph_status_{article_id}"
    current_status = get_setting(status_key)
    if current_status and (current_status.startswith("running") or current_status.startswith("slide_running")):
        return jsonify({"error": "already_running"}), 409

    row = query_one(
        "SELECT carousel_images, carousel_cinemagraph_prompts, "
        "carousel_cinemagraphs "
        "FROM articles WHERE id = %s", (article_id,))
    if not row or not row[0]:
        return jsonify({"error": "No carousel images found."}), 400

    all_images = (json.loads(row[0]) if row[0] else [])[:10]
    # Pad to 10 so indexing is safe
    all_images = (all_images + [None] * 10)[:10]

    body = request.get_json(silent=True) or {}
    global_prompt = (body.get("global_prompt") or "").strip()
    incoming_prompts = body.get("prompts") or []  # per-slide list from client

    # Build saved prompts for all 10 slots
    existing_prompts = json.loads(row[1]) if row[1] else []
    resolved_default = get_setting(f'cinemagraph_prompt_{article_id}') or get_setting('cinemagraph_prompt', _CINE_DEFAULT_PROMPT)
    saved_prompts = []
    for i in range(10):
        if i < len(incoming_prompts) and incoming_prompts[i]:
            saved_prompts.append(incoming_prompts[i])
        elif global_prompt:
            saved_prompts.append(global_prompt)
        elif i < len(existing_prompts) and existing_prompts[i]:
            saved_prompts.append(existing_prompts[i])
        else:
            saved_prompts.append(resolved_default)

    execute("UPDATE articles SET carousel_cinemagraph_prompts = %s WHERE id = %s",
            (json.dumps(saved_prompts), article_id))

    # Identify which slots are empty (no existing cinemagraph clip)
    existing_cine  = (json.loads(row[2]) if row[2] else [])[:10]
    existing_cine_padded  = (existing_cine  + [None] * 10)[:10]

    empty_indices = [i for i in range(10) if all_images[i] and not existing_cine_padded[i]]
    if not empty_indices:
        return jsonify({"error": "All slides already have clips."}), 400

    empty_images  = [all_images[i]    for i in empty_indices]
    empty_prompts = [saved_prompts[i] for i in empty_indices]
    n_empty = len(empty_images)

    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s",
            (status_key, f"running:0/{n_empty}", f"running:0/{n_empty}"))
    # Clear any stale result from a previous run
    execute("DELETE FROM site_settings WHERE `key` = %s",
            (f"cinemagraph_result_{article_id}",))

    t = threading.Thread(
        target=_cinemagraph_worker,
        args=(article_id, empty_images, empty_prompts),
        kwargs=dict(slot_indices=empty_indices,
                    existing_urls=existing_cine_padded),
        daemon=True,
    )
    t.start()
    return jsonify({"started": True, "n_slides": n_empty})


@app.route("/admin/articles/<int:article_id>/cancel-cinemagraphs", methods=["POST"])
@login_required
@admin_required
def admin_article_cancel_cinemagraphs(article_id):
    """Signal the cinemagraph worker to stop after the current slide."""
    cancel_key = f"cinemagraph_cancel_{article_id}"
    execute(
        "INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE value = %s",
        (cancel_key, "1", "1"),
    )
    return jsonify({"cancelled": True})


def _cinemagraph_slide_worker(article_id, slide_idx, img_url, prompt):
    """Background thread: regenerate a single cinemagraph slide."""
    ts = int(time.time())
    status_key = f"cinemagraph_status_{article_id}"

    def _clear_status():
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    log_lines = []
    ts_start = time.strftime("%Y-%m-%d %H:%M:%S")

    def _log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] slide {slide_idx + 1}: {msg}"
        log_lines.append(line)
        print(f"Cinemagraph slide-worker: {msg}", flush=True)

    def _flush_log():
        """Write current log_lines to DB so the log modal shows live progress."""
        if log_lines:
            existing = query_one("SELECT carousel_cinemagraph_log FROM articles WHERE id = %s", (article_id,))
            prev = (existing[0] or "") if existing else ""
            log_text = prev + f"\n\nRe-run slide {slide_idx + 1} started: {ts_start}\n" + "\n".join(log_lines)
            try:
                execute("UPDATE articles SET carousel_cinemagraph_log = %s WHERE id = %s",
                        (log_text.strip()[:65000], article_id))
            except Exception:
                pass

    try:
        _log(f"re-running slide {slide_idx + 1} with prompt: {prompt[:80]}")

        # Resolve image URL for Luma (needs public URL)
        if img_url.startswith("https://"):
            luma_img_url = img_url
            _log("using GCS URL")
        else:
            img_path = resolve_image_to_local_path(img_url)
            if img_path and img_path.exists():
                gcs_obj = f"articles/{article_id}/temp/slide_{slide_idx + 1}_{ts}.jpg"
                luma_img_url = upload_to_gcs(img_path, gcs_obj, content_type="image/jpeg")
                if not luma_img_url:
                    _log("local image upload to GCS failed")
                    return
                _log("using local file → uploaded to GCS")
            else:
                _log(f"image not found: {img_url}")
                return

        slide_prompt = prompt or get_setting(f'cinemagraph_prompt_{article_id}') or get_setting('cinemagraph_prompt', _CINE_DEFAULT_PROMPT)
        try:
            gen_id = _luma_create_task(luma_img_url, slide_prompt)
            _log(f"Luma gen_id={gen_id}")
            _flush_log()
        except Exception as e:
            if _is_luma_billing_error(e):
                _log(f"Luma submit FAILED (billing/credits): {e}")
            else:
                _log(f"Luma submit FAILED: {e}")
            return

        # Poll Luma until ready
        mp4_url = None
        poll_count = 0
        last_sub_status = None
        while True:
            time.sleep(6)
            poll_count += 1
            try:
                luma_status, mp4_url, status_msg = _luma_poll_task(gen_id)
            except Exception as e:
                _log(f"poll error: {e}")
                break
            if luma_status == "completed":
                _log(f"SUCCEEDED after {poll_count} polls")
                break
            elif luma_status == "failed":
                _log(f"FAILED (reason: {status_msg})")
                mp4_url = None
                break
            else:
                _log(f"still generating (poll {poll_count}, status={luma_status})")
                if luma_status != last_sub_status:
                    _flush_log()
                    last_sub_status = luma_status

        if mp4_url:
            r = http_requests.get(mp4_url, timeout=120)
            local_dir = Path(f"static/articles/{article_id}/cinemagraph")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"slide_{slide_idx + 1}_{ts}.mp4"
            local_path.write_bytes(r.content)
            gcs_obj = f"articles/{article_id}/cinemagraph/slide_{slide_idx + 1}_{ts}.mp4"
            gcs_url = upload_bytes_to_gcs(r.content, gcs_obj, content_type="video/mp4")
            if gcs_url:
                local_path.unlink(missing_ok=True)
                saved_url = gcs_url
            else:
                saved_url = "/" + str(local_path).replace("\\", "/")

            # Patch just this slot in the DB array
            import datetime as _dt
            row = query_one("SELECT carousel_cinemagraphs, carousel_cinemagraph_prompts, carousel_cinemagraph_created_at FROM articles WHERE id = %s", (article_id,))
            urls       = json.loads(row[0]) if row and row[0] else []
            prompts    = json.loads(row[1]) if row and row[1] else []
            created_at = json.loads(row[2]) if row and row[2] else []
            while len(urls)       <= slide_idx: urls.append(None)
            while len(prompts)    <= slide_idx: prompts.append(None)
            while len(created_at) <= slide_idx: created_at.append(None)
            urls[slide_idx]       = saved_url
            prompts[slide_idx]    = prompt or _CINE_DEFAULT_PROMPT
            created_at[slide_idx] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            execute(
                "UPDATE articles SET carousel_cinemagraphs = %s, carousel_cinemagraph_prompts = %s, "
                "carousel_cinemagraph_created_at = %s WHERE id = %s",
                (json.dumps(urls), json.dumps(prompts), json.dumps(created_at), article_id),
            )
            _log(f"saved → {saved_url}")
        else:
            _log("no output URL — slide not saved")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _log(f"EXCEPTION: {e}")
    finally:
        _clear_status()
        # Append to existing log
        if log_lines:
            existing_log = query_one("SELECT carousel_cinemagraph_log FROM articles WHERE id = %s", (article_id,))
            prev = (existing_log[0] or "") if existing_log else ""
            new_log = (prev + f"\n\nRe-run slide {slide_idx + 1} started: {ts_start}\n" + "\n".join(log_lines)).strip()
            upload_bytes_to_gcs(new_log.encode(), f"articles/{article_id}/cinemagraph/run_{ts}.log", content_type="text/plain")
            execute("UPDATE articles SET carousel_cinemagraph_log = %s WHERE id = %s",
                    (new_log[:65000], article_id))
        # Activity log entry
        summary = "\n".join(log_lines) if log_lines else "No log output"
        _add_activity_log(article_id, f"Cinemagraph re-run slide {slide_idx + 1}",
                          "Model: luma ray-2 (1:1, 5s)\n" + summary,
                          component="cinemagraph")


@app.route("/admin/articles/<int:article_id>/regenerate-cinemagraph-slide", methods=["POST"])
@login_required
@admin_required
def admin_article_regenerate_cinemagraph_slide(article_id):
    """Re-run Luma AI for a single cinemagraph slide."""
    if not _luma_enabled:
        return jsonify({"error": "Luma AI not configured. Set LUMAAI_API_KEY."}), 400

    status_key = f"cinemagraph_status_{article_id}"
    current_status = get_setting(status_key)
    if current_status:
        return jsonify({"error": "already_running"}), 409

    body = request.get_json(silent=True) or {}
    slide_idx = body.get("slide_idx")
    prompt    = (body.get("prompt") or "").strip() or get_setting(f'cinemagraph_prompt_{article_id}') or get_setting('cinemagraph_prompt', _CINE_DEFAULT_PROMPT)

    if slide_idx is None:
        return jsonify({"error": "slide_idx required"}), 400

    row = query_one(
        "SELECT carousel_images, carousel_cinemagraphs, carousel_cinemagraph_archived, "
        "carousel_cinemagraph_prompts, carousel_cinemagraph_created_at "
        "FROM articles WHERE id = %s", (article_id,)
    )
    if not row or not row[0]:
        return jsonify({"error": "No carousel images found."}), 400

    images       = json.loads(row[0]) if row[0] else []
    cine_urls    = json.loads(row[1]) if row[1] else []
    archived     = json.loads(row[2]) if row[2] else []
    cine_prompts = json.loads(row[3]) if row[3] else []
    cine_created = json.loads(row[4]) if row[4] else []

    if slide_idx >= len(images) or not images[slide_idx]:
        return jsonify({"error": "Slide image not found."}), 400

    # Auto-archive the existing clip for this slot (if any) before re-running
    archived_idx = None
    existing_url = cine_urls[slide_idx] if slide_idx < len(cine_urls) else None
    if existing_url:
        import datetime as _dt
        archived_idx = len(archived)
        archived.append({
            "url": existing_url,
            "prompt": cine_prompts[slide_idx] if slide_idx < len(cine_prompts) else None,
            "created_at": cine_created[slide_idx] if slide_idx < len(cine_created) else None,
            "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slot_source": slide_idx + 1,
        })
        while len(cine_urls) <= slide_idx:
            cine_urls.append(None)
        cine_urls[slide_idx] = None
        if slide_idx < len(cine_created):
            cine_created[slide_idx] = None
        execute(
            "UPDATE articles SET carousel_cinemagraphs = %s, carousel_cinemagraph_archived = %s, "
            "carousel_cinemagraph_created_at = %s WHERE id = %s",
            (json.dumps(cine_urls), json.dumps(archived), json.dumps(cine_created), article_id),
        )

    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s",
            (status_key, f"slide_running:{slide_idx}", f"slide_running:{slide_idx}"))

    t = threading.Thread(
        target=_cinemagraph_slide_worker,
        args=(article_id, slide_idx, images[slide_idx], prompt),
        daemon=True,
    )
    t.start()
    return jsonify({"started": True, "slide_idx": slide_idx, "archived_idx": archived_idx})


@app.route("/admin/articles/<int:article_id>/archive-cinemagraph-slide", methods=["POST"])
@login_required
@admin_required
def admin_article_archive_cinemagraph_slide(article_id):
    """Move a single cinemagraph slot into the archived array."""
    body = request.get_json(silent=True) or {}
    slide_idx = body.get("slide_idx")
    if slide_idx is None:
        return jsonify({"error": "slide_idx required"}), 400

    row = query_one(
        "SELECT carousel_cinemagraphs, carousel_cinemagraph_prompts, carousel_cinemagraph_archived, "
        "carousel_cinemagraph_created_at "
        "FROM articles WHERE id = %s", (article_id,)
    )
    if not row:
        return jsonify({"error": "Article not found"}), 404

    urls       = json.loads(row[0]) if row[0] else []
    prompts    = json.loads(row[1]) if row[1] else []
    archived   = json.loads(row[2]) if row[2] else []
    created_at = json.loads(row[3]) if row[3] else []

    while len(urls)       <= slide_idx: urls.append(None)
    while len(prompts)    <= slide_idx: prompts.append(None)
    while len(created_at) <= slide_idx: created_at.append(None)

    archived_idx = len(archived)
    if slide_idx < len(urls) and urls[slide_idx]:
        import datetime as _dt
        archived.append({
            "url": urls[slide_idx],
            "prompt": prompts[slide_idx] if slide_idx < len(prompts) else None,
            "created_at": created_at[slide_idx],
            "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slot_source": slide_idx + 1,
        })
    urls[slide_idx]       = None
    prompts[slide_idx]    = None
    created_at[slide_idx] = None

    execute(
        "UPDATE articles SET carousel_cinemagraphs = %s, carousel_cinemagraph_prompts = %s, "
        "carousel_cinemagraph_archived = %s, "
        "carousel_cinemagraph_created_at = %s WHERE id = %s",
        (json.dumps(urls), json.dumps(prompts), json.dumps(archived),
         json.dumps(created_at), article_id),
    )
    return jsonify({"ok": True, "archived_idx": archived_idx})


@app.route("/admin/articles/<int:article_id>/restore-cinemagraph-clip", methods=["POST"])
@login_required
@admin_required
def admin_article_restore_cinemagraph_clip(article_id):
    """Move a clip from the archived array back to a chosen slot (auto-archiving any displaced clip)."""
    body = request.get_json(silent=True) or {}
    archived_idx = body.get("archived_idx")
    target_slot  = body.get("target_slot")
    if archived_idx is None:
        return jsonify({"error": "archived_idx required"}), 400
    if target_slot is None or target_slot < 0 or target_slot > 9:
        return jsonify({"error": "target_slot (0-9) required"}), 400

    row = query_one(
        "SELECT carousel_cinemagraphs, carousel_cinemagraph_archived, "
        "carousel_cinemagraph_prompts, carousel_cinemagraph_created_at "
        "FROM articles WHERE id = %s",
        (article_id,)
    )
    if not row:
        return jsonify({"error": "Article not found"}), 404

    urls       = json.loads(row[0]) if row[0] else []
    archived   = json.loads(row[1]) if row[1] else []
    prompts    = json.loads(row[2]) if row[2] else []
    created_at = json.loads(row[3]) if row[3] else []

    if archived_idx >= len(archived) or not archived[archived_idx]:
        return jsonify({"error": "Archived clip not found"}), 404

    # Parse archived entry (string = legacy, dict = new format)
    entry = archived[archived_idx]
    if isinstance(entry, dict):
        clip_url        = entry.get("url")
        clip_prompt     = entry.get("prompt")
        clip_created_at = entry.get("created_at")
    else:
        clip_url        = entry
        clip_prompt     = None
        clip_created_at = None

    archived[archived_idx] = None  # clear from archived (keep index stable)

    # Pad arrays to reach target_slot
    while len(urls)       <= target_slot: urls.append(None)
    while len(prompts)    <= target_slot: prompts.append(None)
    while len(created_at) <= target_slot: created_at.append(None)

    # Auto-archive displaced clip if target slot is occupied
    displaced = None
    if urls[target_slot]:
        import datetime as _dt
        displaced_entry = {
            "url": urls[target_slot],
            "prompt": prompts[target_slot] if target_slot < len(prompts) else None,
            "created_at": created_at[target_slot] if target_slot < len(created_at) else None,
            "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slot_source": target_slot + 1,
        }
        displaced_idx = len(archived)
        archived.append(displaced_entry)
        displaced = {"archived_idx": displaced_idx, "entry": displaced_entry}

    # Place restored clip into target slot
    urls[target_slot]       = clip_url
    prompts[target_slot]    = clip_prompt
    created_at[target_slot] = clip_created_at

    execute(
        "UPDATE articles SET carousel_cinemagraphs = %s, carousel_cinemagraph_archived = %s, "
        "carousel_cinemagraph_prompts = %s, "
        "carousel_cinemagraph_created_at = %s WHERE id = %s",
        (json.dumps(urls), json.dumps(archived), json.dumps(prompts),
         json.dumps(created_at), article_id),
    )
    return jsonify({
        "ok": True, "url": clip_url, "slot_idx": target_slot,
        "archived_idx": archived_idx, "prompt": clip_prompt,
        "created_at": clip_created_at,
        "displaced": displaced,
    })


@app.route("/admin/articles/<int:article_id>/archive-all-cinemagraphs", methods=["POST"])
@login_required
@admin_required
def admin_article_archive_all_cinemagraphs(article_id):
    """Move all current cinemagraph clips into the archived array."""
    row = query_one(
        "SELECT carousel_cinemagraphs, carousel_cinemagraph_prompts, carousel_cinemagraph_archived, "
        "carousel_cinemagraph_created_at "
        "FROM articles WHERE id = %s", (article_id,)
    )
    if not row:
        return jsonify({"error": "Article not found"}), 404

    urls       = json.loads(row[0]) if row[0] else []
    prompts    = json.loads(row[1]) if row[1] else []
    archived   = json.loads(row[2]) if row[2] else []
    created_at = json.loads(row[3]) if row[3] else []

    import datetime as _dt
    archived_start_idx = len(archived)
    for i in range(min(10, len(urls))):
        if urls[i]:
            archived.append({
                "url": urls[i],
                "prompt": prompts[i] if i < len(prompts) else None,
                "created_at": created_at[i] if i < len(created_at) else None,
                "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "slot_source": i + 1,
            })
            urls[i] = None
            if i < len(prompts): prompts[i] = None
            if i < len(created_at): created_at[i] = None

    execute(
        "UPDATE articles SET carousel_cinemagraphs = %s, carousel_cinemagraph_prompts = %s, "
        "carousel_cinemagraph_archived = %s, "
        "carousel_cinemagraph_created_at = %s WHERE id = %s",
        (json.dumps(urls), json.dumps(prompts), json.dumps(archived),
         json.dumps(created_at), article_id),
    )
    return jsonify({"ok": True, "archived_start_idx": archived_start_idx})


# ------------------------------------------------------------
# YOUTUBE UPLOAD WORKER + ROUTES
# ------------------------------------------------------------
def _upload_to_youtube_worker(article_id, video_url, title, desc, run_ts, refresh_token):
    """Background thread: download video → upload to YouTube as a Short."""
    import requests as _req
    import tempfile as _tmp
    import os as _os

    status_key = f"yt_status_{article_id}_{run_ts}"
    result_key = f"yt_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        # 1. Get fresh access token
        _set_status("running:token")
        token_resp = _req.post("https://oauth2.googleapis.com/token", data={
            "client_id":     YT_CLIENT_ID,
            "client_secret": YT_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type":    "refresh_token",
        }, timeout=15)
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            _set_result(f"error:Failed to get access token: {token_data.get('error_description', token_data)}")
            _add_activity_log(article_id, "YouTube Upload Failed",
                              f"Token refresh failed: {token_data}", component="narrated")
            return

        # 2. Download the video to a temp file
        _set_status("running:download")
        with _tmp.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_path = tmp_f.name
        try:
            with _req.get(video_url, stream=True, timeout=120) as dl:
                dl.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
        except Exception as dl_err:
            _os.unlink(tmp_path) if _os.path.exists(tmp_path) else None
            _set_result(f"error:Download failed: {dl_err}")
            _add_activity_log(article_id, "YouTube Upload Failed",
                              f"Download error: {dl_err}", component="narrated")
            return

        # 3. Upload to YouTube using resumable upload
        _set_status("running:upload")
        file_size = _os.path.getsize(tmp_path)

        # Initiate resumable upload
        metadata = {
            "snippet": {
                "title":       title or "Obelisk Stamps",
                "description": desc  or "",
                "tags":        ["stamps", "philately", "shorts"],
                "categoryId":  "26",  # Howto & Style
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        }
        init_resp = _req.post(
            "https://www.googleapis.com/upload/youtube/v3/videos"
            "?uploadType=resumable&part=snippet,status",
            headers={
                "Authorization":           f"Bearer {access_token}",
                "Content-Type":            "application/json; charset=UTF-8",
                "X-Upload-Content-Type":   "video/mp4",
                "X-Upload-Content-Length": str(file_size),
            },
            json=metadata,
            timeout=30,
        )
        if init_resp.status_code not in (200, 201):
            _os.unlink(tmp_path) if _os.path.exists(tmp_path) else None
            try:
                err_json = init_resp.json()
                err_msg  = err_json.get("error", {}).get("message", "") or init_resp.text[:300]
                err_reason = (err_json.get("error", {}).get("errors") or [{}])[0].get("reason", "")
                if err_reason == "youtubeSignupRequired":
                    err_msg = "The authorized Google account has no YouTube channel. Go to youtube.com and create one first."
            except Exception:
                err_msg = init_resp.text[:300]
            _set_result(f"error:{err_msg}")
            _add_activity_log(article_id, "YouTube Upload Failed",
                              f"HTTP {init_resp.status_code}: {err_msg}", component="narrated")
            return

        upload_url = init_resp.headers.get("Location")
        if not upload_url:
            _os.unlink(tmp_path) if _os.path.exists(tmp_path) else None
            _set_result("error:No upload URL returned from YouTube")
            return

        # Upload the file
        with open(tmp_path, "rb") as f:
            up_resp = _req.put(
                upload_url,
                headers={
                    "Content-Type":   "video/mp4",
                    "Content-Length": str(file_size),
                },
                data=f,
                timeout=600,
            )
        _os.unlink(tmp_path) if _os.path.exists(tmp_path) else None

        if up_resp.status_code not in (200, 201):
            _set_result(f"error:Upload failed: {up_resp.text[:200]}")
            _add_activity_log(article_id, "YouTube Upload Failed",
                              f"Upload error {up_resp.status_code}: {up_resp.text[:200]}",
                              component="narrated")
            return

        video_id = up_resp.json().get("id", "")
        yt_url   = f"https://www.youtube.com/shorts/{video_id}" if video_id else "https://youtube.com"
        _set_result(f"done:{yt_url}")
        _add_activity_log(article_id, "Posted to YouTube Shorts",
                          f"video_id={video_id}\nurl={yt_url}",
                          component="narrated")

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            _os.unlink(tmp_path) if _os.path.exists(tmp_path) else None
        except Exception:
            pass
        _set_result(f"error:{str(e)[:200]}")
        _add_activity_log(article_id, "YouTube Upload Failed",
                          f"Exception: {e}", component="narrated")


@app.route("/admin/youtube-connect")
@login_required
@admin_required
def admin_youtube_connect():
    """Redirect admin to Google OAuth to authorize YouTube upload access."""
    import urllib.parse as _urlparse
    params = {
        "client_id":     YT_CLIENT_ID,
        "redirect_uri":  YT_REDIRECT_URI,
        "response_type": "code",
        "scope":         " ".join(YT_SCOPES),
        "access_type":   "offline",
        "prompt":        "consent",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + _urlparse.urlencode(params)
    return redirect(url)


@app.route("/admin/youtube-oauth-callback")
@login_required
@admin_required
def admin_youtube_oauth_callback():
    """Exchange auth code for YouTube refresh token and store it."""
    import requests as _req
    code = request.args.get("code")
    if not code:
        flash("YouTube authorization failed — no code returned.", "danger")
        return redirect(url_for("admin_panel"))
    resp = _req.post("https://oauth2.googleapis.com/token", data={
        "code":          code,
        "client_id":     YT_CLIENT_ID,
        "client_secret": YT_CLIENT_SECRET,
        "redirect_uri":  YT_REDIRECT_URI,
        "grant_type":    "authorization_code",
    }, timeout=15)
    token_data = resp.json()
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        flash(f"YouTube auth failed: {token_data.get('error_description', token_data)}", "danger")
        return redirect(url_for("admin_panel"))
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
            ("youtube_refresh_token", refresh_token, refresh_token))
    flash("YouTube account connected successfully!", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/articles/<int:article_id>/post-to-youtube", methods=["POST"])
@login_required
@admin_required
def admin_post_to_youtube(article_id):
    """Upload a narrated video to YouTube Shorts as a background task."""
    refresh_token = get_setting("youtube_refresh_token")
    if not refresh_token:
        return jsonify({"error": "YouTube not connected. Go to Admin → Connect YouTube."}), 400
    data      = request.get_json() or {}
    video_url = (data.get("video_url") or "").strip()
    title     = (data.get("title")     or "").strip()[:100]
    desc      = (data.get("description") or "").strip()[:5000]
    run_ts    = int(data.get("run_ts") or 0)
    if not video_url:
        return jsonify({"error": "No video URL provided."}), 400
    status_key = f"yt_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already uploading this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
            (status_key, "running:start", "running:start"))
    _add_activity_log(article_id, "YouTube Upload Started",
                      f"run_ts={run_ts}\nvideo_url={video_url[:120]}",
                      component="narrated")
    threading.Thread(target=_upload_to_youtube_worker,
                     args=(article_id, video_url, title, desc, run_ts, refresh_token),
                     daemon=True).start()
    return jsonify({"started": True})


@app.route("/admin/articles/<int:article_id>/youtube-status")
@login_required
@admin_required
def admin_youtube_status(article_id):
    """Poll YouTube upload status for a given run_ts."""
    run_ts     = request.args.get("ts", "0")
    status_key = f"yt_status_{article_id}_{run_ts}"
    result_key = f"yt_result_{article_id}_{run_ts}"
    status = get_setting(status_key) or "idle"
    result = get_setting(result_key) or ""
    return jsonify({"status": status, "result": result})


# ------------------------------------------------------------
# SEO: robots.txt + sitemap.xml
# ------------------------------------------------------------
@app.route("/robots.txt")
def robots_txt():
    from flask import Response
    site = SITE_URL or request.url_root.rstrip("/")
    content = f"User-agent: *\nAllow: /\nSitemap: {site}/sitemap.xml\n"
    return Response(content, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap_xml():
    from flask import Response
    site = SITE_URL or request.url_root.rstrip("/")
    static_pages = [
        ("",                "weekly",  "1.0"),
        ("/about",          "monthly", "0.7"),
        ("/catalogue",      "weekly",  "0.9"),
        ("/articles",       "weekly",  "0.8"),
        ("/contact",        "monthly", "0.5"),
        ("/stamp-quest-ai", "monthly", "0.6"),
    ]
    rows = query_all(
        "SELECT slug, updated_at FROM articles WHERE is_published = TRUE ORDER BY updated_at DESC"
    )
    urls = []
    for path, freq, pri in static_pages:
        urls.append(
            f"  <url>\n    <loc>{site}{path}</loc>\n"
            f"    <changefreq>{freq}</changefreq>\n    <priority>{pri}</priority>\n  </url>"
        )
    for slug, updated_at in rows:
        lastmod = f"\n    <lastmod>{updated_at.strftime('%Y-%m-%d')}</lastmod>" if updated_at else ""
        urls.append(
            f"  <url>\n    <loc>{site}/articles/{slug}</loc>{lastmod}\n"
            f"    <changefreq>monthly</changefreq>\n    <priority>0.7</priority>\n  </url>"
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls)
        + "\n</urlset>"
    )
    return Response(xml, mimetype="application/xml")


# ------------------------------------------------------------
# INSTAGRAM CAROUSEL POSTING
# ------------------------------------------------------------

def _ig_keys(post_type, article_id):
    """Return dict of site_settings keys for a given post type ('cine' or 'car')."""
    prefix = "ig_car_post_" if post_type == "car" else "ig_post_"
    return {
        "status":   f"{prefix}status_{article_id}",
        "result":   f"{prefix}result_{article_id}",
        "media_id": f"{prefix}media_id_{article_id}",
        "history":  f"{prefix}history_{article_id}",
        "snapshot": f"{prefix}snapshot_{article_id}",
    }


def _fb_keys(post_type, article_id):
    """Return dict of site_settings keys for Facebook posting ('cine' or 'car')."""
    prefix = "fb_car_post_" if post_type == "car" else "fb_post_"
    return {
        "status":   f"{prefix}status_{article_id}",
        "result":   f"{prefix}result_{article_id}",
        "media_id": f"{prefix}media_id_{article_id}",
        "history":  f"{prefix}history_{article_id}",
        "snapshot": f"{prefix}snapshot_{article_id}",
    }


def _add_activity_log(article_id, title, content, component=None):
    """Append an entry to the article's activity log (stored in site_settings)."""
    import datetime as _dt
    key = f"ig_activity_log_{article_id}"
    raw = get_setting(key)
    entries = json.loads(raw) if raw else []
    # Cap each entry's content to avoid hitting column size limits
    if len(content) > 4000:
        content = content[:4000] + "\n… (truncated)"
    entry = {
        "title":     title,
        "timestamp": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "content":   content,
    }
    if component:
        entry["component"] = component
    entries.append(entry)
    # Keep only the last 50 entries
    entries = entries[-50:]
    val = json.dumps(entries)
    execute(
        "INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE value = %s",
        (key, val, val),
    )


def _ig_api_call(method, url, **kwargs):
    """Instagram Graph API call with one automatic retry on rate-limit."""
    import requests as _req
    resp = _req.post(url, **kwargs) if method == "POST" else _req.get(url, **kwargs)
    try:
        data = resp.json()
    except Exception:
        return resp
    err_msg = data.get("error", {}).get("message", "")
    if "request limit" in err_msg.lower():
        print(f"IG API rate-limited, waiting 30s before retry…", flush=True)
        _time.sleep(30)
        resp = _req.post(url, **kwargs) if method == "POST" else _req.get(url, **kwargs)
    return resp


def _post_to_instagram_worker(article_id, caption, post_type="cine"):
    """
    Background thread: compose each carousel slide → upload to GCS →
    create IG child containers → create carousel container → publish.
    post_type: 'cine' = use cinemagraph video clips; 'car' = use static carousel images.
    Progress stored in site_settings key ig_post_status_{article_id} (cine)
    or ig_car_post_status_{article_id} (car).
    """
    import time as _time
    import requests as _req

    keys       = _ig_keys(post_type, article_id)
    status_key = keys["status"]
    result_key = keys["result"]
    _log_component = "carousel" if post_type == "car" else "cinemagraph"

    def _set_status(s):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, s, s))

    def _set_result(s):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, s, s))

    def _clear_status():
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    ig_user_id    = IG_USER_ID
    access_token  = IG_ACCESS_TOKEN

    try:
        # ── 1. Load carousel data ──────────────────────────────────────────
        row = query_one(
            "SELECT carousel_images, carousel_punchlines, carousel_cinemagraphs "
            "FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            _clear_status()
            return

        images        = [x for x in (json.loads(row[0]) if row[0] else [])[:10] if x]
        punchlines    = json.loads(row[1]) if row[1] else []
        cinemagraphs  = json.loads(row[2]) if row[2] else []
        n             = len(images)
        # For carousel posts use only static images; for cine posts use videos if available
        use_video     = (post_type != "car") and bool(cinemagraphs and any(u for u in cinemagraphs if u))

        # Pre-compute consistent dark-band position for all slides
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

        if n < 2:
            _set_result("error:Need at least 2 images to post a carousel")
            _clear_status()
            return

        print(f"IG worker: article={article_id} posting {n} slides ig_user={ig_user_id} "
              f"mode={'video' if use_video else 'image'}", flush=True)

        # ── 2a. Video carousel path (cinemagraphs exist) ───────────────────
        if use_video:
            _set_status(f"running:containers:0/{n}")
            container_ids = []
            for i, img_url in enumerate(images):
                _set_status(f"running:containers:{i+1}/{n}")
                video_url = cinemagraphs[i] if i < len(cinemagraphs) else None
                if video_url:
                    # Apply punchline + swipe hint overlay to the video before posting
                    this_punchline = punchlines[i] if i < len(punchlines) else None
                    posted_video_url = video_url  # default: raw
                    if this_punchline:
                        try:
                            overlay_png = _make_cinemagraph_overlay_png(this_punchline, i, n, band_top=carousel_band_top)
                            cine_dir    = BASE_DIR / "static" / "articles" / str(article_id) / "cinemagraph"
                            cine_dir.mkdir(parents=True, exist_ok=True)
                            tmp_mp4 = cine_dir / f"ig_composed_{i}_{int(_time.time())}.mp4"
                            try:
                                if _apply_overlay_to_video(video_url, overlay_png, tmp_mp4):
                                    gcs_obj = (f"articles/{article_id}/cinemagraph/"
                                               f"ig_composed_{i}_{int(_time.time())}.mp4")
                                    gcs_url = upload_bytes_to_gcs(
                                        tmp_mp4.read_bytes(), gcs_obj,
                                        content_type="video/mp4"
                                    )
                                    if gcs_url:
                                        posted_video_url = gcs_url
                            finally:
                                tmp_mp4.unlink(missing_ok=True)
                        except Exception as _ov_err:
                            print(f"IG overlay error slide {i+1}: {_ov_err}", flush=True)
                    # Post as video carousel item
                    resp = _ig_api_call("POST",
                        f"{_IG_GRAPH_URL}/{ig_user_id}/media",
                        data={"video_url": posted_video_url, "media_type": "VIDEO",
                              "is_carousel_item": "true", "access_token": access_token},
                        timeout=30,
                    )
                else:
                    # Fallback: this slide has no cinemagraph → use static image
                    ts_fb = int(_time.time())
                    try:
                        if img_url.startswith("https://"):
                            resp_img = _req.get(img_url, timeout=20)
                            img_bytes = resp_img.content
                        else:
                            local = resolve_image_to_local_path(img_url)
                            img_bytes = local.read_bytes() if local and local.exists() else b""
                    except Exception:
                        img_bytes = b""
                    punchline  = punchlines[i] if i < len(punchlines) else ""
                    jpeg_bytes = compose_carousel_slide(img_bytes, punchline, i, n, band_top=carousel_band_top)
                    gcs_obj    = f"articles/{article_id}/instagram/composed_{i+1}_{ts_fb}.jpg"
                    public_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
                    resp = _ig_api_call("POST",
                        f"{_IG_GRAPH_URL}/{ig_user_id}/media",
                        data={"image_url": public_url or img_url,
                              "is_carousel_item": "true", "access_token": access_token},
                        timeout=30,
                    )

                data = resp.json()
                if "id" not in data:
                    err = data.get("error", {}).get("message", str(data))
                    print(f"IG worker: child container {i+1} FAILED: {err}", flush=True)
                    _set_result(f"error:Child container {i+1} failed: {err}")
                    _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                                      f"Child container {i+1}/{n} creation failed.\nAPI error: {err}",
                                      component=_log_component)
                    _clear_status()
                    return
                print(f"IG worker: child container {i+1}/{n} id={data['id']}", flush=True)
                container_ids.append(data["id"])

        # ── 2b. Image carousel path (no cinemagraphs) ─────────────────────
        else:
            ts = int(_time.time())
            composed_urls = []
            for i, img_url in enumerate(images):
                _set_status(f"running:compose:{i+1}/{n}")
                try:
                    if img_url.startswith("https://"):
                        resp_img = _req.get(img_url, timeout=20)
                        resp_img.raise_for_status()
                        img_bytes = resp_img.content
                    else:
                        local = resolve_image_to_local_path(img_url)
                        if not local or not local.exists():
                            _set_result(f"error:Image not found: {img_url}")
                            _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                                              f"Image not found: {img_url}",
                                              component=_log_component)
                            _clear_status()
                            return
                        img_bytes = local.read_bytes()
                except Exception as e:
                    _set_result(f"error:Could not fetch image {i+1}: {e}")
                    _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                                      f"Could not fetch image {i+1}: {e}",
                                      component=_log_component)
                    _clear_status()
                    return

                punchline   = punchlines[i] if i < len(punchlines) else ""
                jpeg_bytes  = compose_carousel_slide(img_bytes, punchline, i, n, band_top=carousel_band_top)
                gcs_obj     = f"articles/{article_id}/instagram/composed_{i+1}_{ts}.jpg"
                public_url  = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
                if not public_url:
                    _set_result("error:GCS upload failed — GCS_BUCKET_NAME must be set")
                    _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                                      "GCS upload failed — GCS_BUCKET_NAME must be set",
                                      component=_log_component)
                    _clear_status()
                    return
                print(f"IG worker: slide {i+1}/{n} composed → {public_url}", flush=True)
                composed_urls.append(public_url)

            # ── 3. Create IG child containers ─────────────────────────────────
            _set_status(f"running:containers:0/{n}")
            container_ids = []
            for i, url in enumerate(composed_urls):
                _set_status(f"running:containers:{i+1}/{n}")
                resp = _ig_api_call("POST",
                    f"{_IG_GRAPH_URL}/{ig_user_id}/media",
                    data={"image_url": url, "is_carousel_item": "true",
                          "access_token": access_token},
                    timeout=30,
                )
                data = resp.json()
                if "id" not in data:
                    err = data.get("error", {}).get("message", str(data))
                    print(f"IG worker: child container {i+1} FAILED: {err}", flush=True)
                    _set_result(f"error:Child container {i+1} failed: {err}")
                    _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                                      f"Child container {i+1}/{n} creation failed.\nAPI error: {err}",
                                      component=_log_component)
                    _clear_status()
                    return
                print(f"IG worker: child container {i+1}/{n} id={data['id']}", flush=True)
                container_ids.append(data["id"])

        # ── 4. Poll each container until FINISHED ─────────────────────────
        _set_status(f"running:poll:0/{n}")
        for i, cid in enumerate(container_ids):
            _set_status(f"running:poll:{i+1}/{n}")
            deadline = _time.time() + 300  # 5-minute timeout
            _poll_n = 0
            while _time.time() < deadline:
                r = _ig_api_call("GET",
                    f"{_IG_GRAPH_URL}/{cid}",
                    params={"fields": "status_code,status", "access_token": access_token},
                    timeout=15,
                )
                poll_data    = r.json()
                status_code  = poll_data.get("status_code", "")
                status_detail = poll_data.get("status", "")
                if status_code == "FINISHED":
                    print(f"IG worker: container {i+1}/{n} FINISHED", flush=True)
                    break
                if status_code == "ERROR":
                    err_msg = f"Container {i+1} status ERROR: {status_detail}" if status_detail else f"Container {i+1} status ERROR"
                    print(f"IG worker: {err_msg}", flush=True)
                    _set_result(f"error:{err_msg}")
                    _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                                      f"Container {i+1}/{n} (id={cid}) returned ERROR.\nStatus detail: {status_detail or 'none'}",
                                      component=_log_component)
                    _clear_status()
                    return
                _time.sleep(min(5 + _poll_n * 5, 30))
                _poll_n += 1
            else:
                err_msg = f"Container {i+1} timed out waiting for FINISHED"
                _set_result(f"error:{err_msg}")
                _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                                  f"Container {i+1}/{n} (id={cid}) timed out after 5 minutes.",
                                  component=_log_component)
                _clear_status()
                return

        # ── 5. Create carousel container ──────────────────────────────────
        _set_status("running:carousel")
        print(f"IG worker: creating carousel container with {n} children", flush=True)
        resp = _ig_api_call("POST",
            f"{_IG_GRAPH_URL}/{ig_user_id}/media",
            data={"media_type": "CAROUSEL",
                  "children": ",".join(container_ids),
                  "caption": caption or "",
                  "access_token": access_token},
            timeout=30,
        )
        data = resp.json()
        if "id" not in data:
            err = data.get("error", {}).get("message", str(data))
            print(f"IG worker: carousel container FAILED: {err}", flush=True)
            _set_result(f"error:Carousel container failed: {err}")
            _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                              f"Carousel container creation failed.\nChildren: {n}\nAPI error: {err}",
                              component=_log_component)
            _clear_status()
            return
        carousel_id = data["id"]
        print(f"IG worker: carousel container id={carousel_id}", flush=True)

        # ── 5b. Wait for carousel container to be FINISHED ────────────────
        _set_status("running:carousel_poll")
        print(f"IG worker: waiting for carousel container {carousel_id} to be FINISHED", flush=True)
        deadline_c = _time.time() + 300
        _cpoll_n = 0
        while _time.time() < deadline_c:
            rc = _ig_api_call("GET",
                f"{_IG_GRAPH_URL}/{carousel_id}",
                params={"fields": "status_code,status", "access_token": access_token},
                timeout=15,
            )
            cpoll   = rc.json()
            cstatus = cpoll.get("status_code", "")
            cdetail = cpoll.get("status", "")
            if cstatus == "FINISHED":
                print(f"IG worker: carousel container FINISHED", flush=True)
                break
            if cstatus == "ERROR":
                err_msg = f"Carousel container ERROR: {cdetail}" if cdetail else "Carousel container ERROR"
                print(f"IG worker: {err_msg}", flush=True)
                _set_result(f"error:{err_msg}")
                _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                                  f"Carousel container (id={carousel_id}) returned ERROR.\n"
                                  f"Status detail: {cdetail or 'none'}",
                                  component=_log_component)
                _clear_status()
                return
            _time.sleep(min(5 + _cpoll_n * 5, 30))
            _cpoll_n += 1
        else:
            _set_result("error:Carousel container timed out waiting for FINISHED")
            _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                              f"Carousel container (id={carousel_id}) timed out after 5 minutes.",
                              component=_log_component)
            _clear_status()
            return

        # ── 6. Publish ─────────────────────────────────────────────────────
        _set_status("running:publish")
        print(f"IG worker: publishing carousel {carousel_id}", flush=True)
        resp = _req.post(
            f"{_IG_GRAPH_URL}/{ig_user_id}/media_publish",
            data={"creation_id": carousel_id, "access_token": access_token},
            timeout=30,
        )
        data = resp.json()

        # If publish returned an error, check if it actually went through
        # (Instagram sometimes returns rate-limit/Fatal but publishes anyway)
        if "id" not in data:
            err = data.get("error", {}).get("message", str(data))
            print(f"IG worker: publish returned error ({err}), checking if post went through…", flush=True)
            _set_status("running:publish_verify")
            _time.sleep(10)  # give Instagram a moment
            try:
                check_resp = _req.get(
                    f"{_IG_GRAPH_URL}/{ig_user_id}/media",
                    params={"limit": "3", "fields": "id,timestamp,media_type",
                            "access_token": access_token},
                    timeout=15,
                )
                recent = check_resp.json().get("data", [])
                if recent:
                    # If the most recent post was created in the last 2 minutes,
                    # it's almost certainly our publish that went through
                    import datetime as _dt
                    for post in recent:
                        try:
                            ts = _dt.datetime.strptime(post["timestamp"], "%Y-%m-%dT%H:%M:%S+0000")
                            age = (_dt.datetime.utcnow() - ts).total_seconds()
                            if age < 120:
                                print(f"IG worker: post {post['id']} found (age={age:.0f}s) — publish succeeded despite error", flush=True)
                                data = {"id": post["id"]}
                                break
                        except Exception:
                            continue
            except Exception as verify_err:
                print(f"IG worker: verify check failed: {verify_err}", flush=True)

        if "id" not in data:
            err = data.get("error", {}).get("message", str(data))
            print(f"IG worker: publish FAILED: {err}", flush=True)
            if "request limit" in err.lower():
                # Rate-limited even after retry — save carousel_id + snapshot
                # so the user can still archive the post later
                import datetime as _dt
                media_id_key = keys["media_id"]
                execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                        "ON DUPLICATE KEY UPDATE value = %s",
                        (media_id_key, str(carousel_id), str(carousel_id)))
                snapshot = json.dumps({
                    "caption":    caption,
                    "image_urls": images[:10] if not use_video else [],
                    "video_urls": [u for u in cinemagraphs[:10] if u] if use_video else [],
                    "post_type":  post_type,
                    "posted_at":  _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
                execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                        "ON DUPLICATE KEY UPDATE value = %s",
                        (keys["snapshot"], snapshot, snapshot))
                _set_result("warn:Rate limit reached. Your post was likely published — please check Instagram.")
                _add_activity_log(article_id, f"Instagram Rate Limit ({post_type})",
                                  f"Publish returned rate limit for carousel {carousel_id} "
                                  f"(after retry). Carousel ID saved for archive.\n"
                                  f"Post may have been published — check Instagram.",
                                  component=_log_component)
                _clear_status()
                return
            _set_result(f"error:Publish failed: {err}")
            _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                              f"Publish failed for carousel {carousel_id}.\nAPI error: {err}",
                              component=_log_component)
            _clear_status()
            return

        post_id = data["id"]
        # Fetch the real shortcode permalink (numeric ID ≠ shortcode in URL)
        try:
            plink_resp = _ig_api_call("GET",
                f"{_IG_GRAPH_URL}/{post_id}",
                params={"fields": "permalink", "access_token": access_token},
                timeout=15,
            )
            plink_data = plink_resp.json()
            permalink  = plink_data.get("permalink") or f"https://www.instagram.com/p/{post_id}/"
        except Exception:
            permalink = f"https://www.instagram.com/p/{post_id}/"
        print(f"IG worker: SUCCESS post_id={post_id} permalink={permalink}", flush=True)
        _set_result(f"done:{permalink}")
        # Store media_id for later caption editing / archive
        media_id_key = keys["media_id"]
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (media_id_key, str(post_id), str(post_id)))
        # Store snapshot of what was posted (for archive modal)
        import datetime as _dt
        snapshot = json.dumps({
            "caption":    caption,
            "image_urls": images[:10] if not use_video else [],
            "video_urls": [u for u in cinemagraphs[:10] if u] if use_video else [],
            "post_type":  post_type,
            "posted_at":  _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (keys["snapshot"], snapshot, snapshot))
        _add_activity_log(article_id, f"Posted to Instagram ({post_type})",
                          f"post_id={post_id}\npermalink={permalink}\nSlides: {len(images)}",
                          component=_log_component)
        _clear_status()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"IG worker EXCEPTION: {e}", flush=True)
        _set_result(f"error:Unexpected error: {e}")
        _add_activity_log(article_id, f"Instagram Post Failed ({post_type})",
                          f"Error: {e}",
                          component=_log_component)
        _clear_status()


@app.route("/admin/articles/<int:article_id>/post-to-instagram", methods=["POST"])
@login_required
@admin_required
def admin_article_post_to_instagram(article_id):
    """Start a background thread to post to Instagram.
    Body: {caption, type} where type is 'cine' (default) or 'car'.
    """
    if not IG_USER_ID or not IG_ACCESS_TOKEN:
        return jsonify({"error": "Instagram credentials not configured. "
                        "Set IG_USER_ID and IG_ACCESS_TOKEN environment variables."}), 400

    data      = request.get_json() or {}
    caption   = (data.get("caption") or "").strip()[:2200]
    post_type = data.get("type", "cine")
    if post_type not in ("cine", "car"):
        post_type = "cine"

    keys       = _ig_keys(post_type, article_id)
    status_key = keys["status"]
    result_key = keys["result"]

    current_status = get_setting(status_key)
    if current_status and current_status.startswith("running"):
        return jsonify({"error": "already_running"}), 409

    # Validate media availability based on post type
    row = query_one(
        "SELECT carousel_images, carousel_cinemagraphs FROM articles WHERE id = %s",
        (article_id,),
    )
    if not row:
        return jsonify({"error": "Article not found."}), 404

    if post_type == "car":
        images = [x for x in (json.loads(row[0]) if row[0] else [])[:10] if x]
        if len(images) < 2:
            return jsonify({"error": "Need at least 2 carousel images to post."}), 400
        n_media = len(images)
    else:
        clips = [x for x in (json.loads(row[1]) if row[1] else [])[:10] if x]
        if len(clips) < 2:
            return jsonify({"error": "Need at least 2 cinemagraph clips to post."}), 400
        n_media = len(clips)

    # Clear stale result, seed running status
    execute("DELETE FROM site_settings WHERE `key` = %s", (result_key,))
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s",
            (status_key, "running:starting", "running:starting"))

    t = threading.Thread(
        target=_post_to_instagram_worker,
        args=(article_id, caption, post_type),
        daemon=True,
    )
    t.start()
    return jsonify({"started": True, "n_media": n_media, "type": post_type})


@app.route("/admin/articles/<int:article_id>/ig-post-status")
@login_required
@admin_required
def admin_article_ig_post_status(article_id):
    """Poll endpoint: returns current Instagram post status and last result.
    Query param: ?type=cine (default) or ?type=car
    """
    post_type   = request.args.get("type", "cine")
    if post_type not in ("cine", "car"):
        post_type = "cine"
    keys        = _ig_keys(post_type, article_id)
    status      = get_setting(keys["status"])
    result      = get_setting(keys["result"])
    media_id    = get_setting(keys["media_id"])
    history_raw = get_setting(keys["history"])
    history     = json.loads(history_raw) if history_raw else []
    return jsonify({"status": status, "result": result, "media_id": media_id, "history": history})


@app.route("/admin/articles/<int:article_id>/edit-instagram-caption", methods=["POST"])
@login_required
@admin_required
def admin_article_edit_instagram_caption(article_id):
    """Update the caption of the most recently published Instagram post.
    Body: {caption, type} where type is 'cine' (default) or 'car'.
    """
    if not IG_ACCESS_TOKEN:
        return jsonify({"error": "Instagram credentials not configured."}), 400

    data      = request.get_json() or {}
    post_type = data.get("type", "cine")
    if post_type not in ("cine", "car"):
        post_type = "cine"
    keys      = _ig_keys(post_type, article_id)
    media_id  = get_setting(keys["media_id"])
    if not media_id:
        import re as _re
        result_val = get_setting(keys["result"]) or ""
        if result_val.startswith("done:"):
            m = _re.search(r'/p/(\d{10,})', result_val)
            if m:
                media_id = m.group(1)
                execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                        "ON DUPLICATE KEY UPDATE value = %s",
                        (keys["media_id"], media_id, media_id))
    if not media_id:
        return jsonify({"error": "No published post found for this article."}), 400

    caption = (data.get("caption") or "").strip()[:2200]
    if not caption:
        return jsonify({"error": "Caption cannot be empty."}), 400

    import requests as _req
    resp = _req.post(
        f"{_IG_GRAPH_URL}/{media_id}",
        data={"caption": caption, "access_token": IG_ACCESS_TOKEN},
        timeout=30,
    )
    result = resp.json()
    if result.get("success") or result.get("id"):
        _add_activity_log(article_id, f"Caption Edited on Instagram ({post_type})",
                          f"media_id={media_id}\nNew caption: {caption[:200]}…",
                          component=("carousel" if post_type == "car" else "cinemagraph"))
        return jsonify({"ok": True})
    err = result.get("error", {}).get("message", str(result))
    _add_activity_log(article_id, f"Caption Edit Failed ({post_type})",
                      f"media_id={media_id}, error: {err}",
                      component=("carousel" if post_type == "car" else "cinemagraph"))
    return jsonify({"error": err}), 400


@app.route("/admin/articles/<int:article_id>/check-instagram-post")
@login_required
@admin_required
def admin_article_check_instagram_post(article_id):
    """Check if the published Instagram post is still live via the Graph API.
    Query param: ?type=cine (default) or ?type=car
    """
    if not IG_ACCESS_TOKEN:
        return jsonify({"error": "Instagram credentials not configured."}), 400
    post_type = request.args.get("type", "cine")
    if post_type not in ("cine", "car"):
        post_type = "cine"
    ig_keys  = _ig_keys(post_type, article_id)
    media_id = get_setting(ig_keys["media_id"])
    if not media_id:
        # Fallback: older posts stored "done:https://instagram.com/p/{numeric_id}/" in the
        # result key before media_id was saved separately.  Extract and back-fill.
        import re as _re
        result_val = get_setting(ig_keys["result"]) or ""
        if result_val.startswith("done:"):
            m = _re.search(r'/p/(\d{10,})', result_val)
            if m:
                media_id = m.group(1)
                # Persist so subsequent calls are instant
                execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                        "ON DUPLICATE KEY UPDATE value = %s",
                        (ig_keys["media_id"], media_id, media_id))
                print(f"IG check-post-live: back-filled media_id={media_id!r} from result URL", flush=True)
    if not media_id:
        return jsonify({"error": "No published post found for this article."}), 400
    import requests as _req
    print(f"IG check-post-live: media_id={media_id!r} type={post_type}", flush=True)
    try:
        resp = _req.get(
            f"{_IG_GRAPH_URL}/{media_id}",
            params={"fields": "id,permalink", "access_token": IG_ACCESS_TOKEN},
            timeout=15,
        )
        data = resp.json()
        print(f"IG check-post-live response: {data}", flush=True)
        if "id" in data:
            _add_activity_log(article_id, f"Check Post Live ({post_type})",
                              f"Post is live. media_id={media_id}, permalink={data.get('permalink', '')}",
                              component=("carousel" if post_type == "car" else "cinemagraph"))
            return jsonify({"live": True, "permalink": data.get("permalink", "")})
        api_err = data.get("error", {}).get("message", "Post not found on Instagram.")
        # Strip the boilerplate "Please read the Graph API documentation at …" suffix
        for _suffix in (". Please read the Graph API", ". See the Graph API"):
            if _suffix in api_err:
                api_err = api_err[:api_err.index(_suffix)]
                break
        _add_activity_log(article_id, f"Check Post Live ({post_type})",
                          f"Post not found. media_id={media_id}, API response: {api_err}",
                          component=("carousel" if post_type == "car" else "cinemagraph"))
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        print(f"IG check-post-live exception: {exc}", flush=True)
        _add_activity_log(article_id, f"Check Post Live Failed ({post_type})",
                          f"Exception: {exc}",
                          component=("carousel" if post_type == "car" else "cinemagraph"))
        return jsonify({"error": f"Failed to reach Instagram API: {exc}"}), 500


def _post_narrated_reel_worker(article_id, video_url, caption, run_ts):
    """Background thread: post a single narrated video MP4 as an Instagram Reel."""
    import time as _time, requests as _req
    status_key = f"narrated_ig_status_{article_id}_{run_ts}"
    result_key = f"narrated_ig_result_{article_id}_{run_ts}"

    def _set_status(s):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, s, s))
    def _set_result(r):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, r, r))
    def _clear_status():
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        # ── 1. Create Reel container ──────────────────────────────────────
        _set_status("running:container")
        print(f"IG Reel: creating container article={article_id}", flush=True)
        resp = _ig_api_call("POST",
            f"{_IG_GRAPH_URL}/{IG_USER_ID}/media",
            data={"media_type": "REELS", "video_url": video_url,
                  "caption": (caption or "")[:2200],
                  "access_token": IG_ACCESS_TOKEN},
            timeout=30,
        )
        data = resp.json()
        if "id" not in data:
            err = data.get("error", {}).get("message", str(data))
            print(f"IG Reel: container FAILED: {err}", flush=True)
            _set_result(f"error:Container failed: {err}")
            _add_activity_log(article_id, "Instagram Reel Post Failed",
                              f"Container creation failed.\nAPI error: {err}",
                              component="narrated")
            _clear_status(); return
        container_id = data["id"]
        print(f"IG Reel: container id={container_id}", flush=True)

        # ── 2. Poll container until FINISHED ──────────────────────────────
        _set_status("running:poll")
        deadline = _time.time() + 300
        _poll_n  = 0
        while _time.time() < deadline:
            r = _ig_api_call("GET", f"{_IG_GRAPH_URL}/{container_id}",
                params={"fields": "status_code,status", "access_token": IG_ACCESS_TOKEN},
                timeout=15)
            pd           = r.json()
            status_code  = pd.get("status_code", "")
            status_detail = pd.get("status", "")
            if status_code == "FINISHED":
                print(f"IG Reel: container FINISHED", flush=True); break
            if status_code == "ERROR":
                err_msg = f"Container ERROR: {status_detail}" if status_detail else "Container ERROR"
                _set_result(f"error:{err_msg}")
                _add_activity_log(article_id, "Instagram Reel Post Failed",
                                  f"Container (id={container_id}) ERROR.\nDetail: {status_detail or 'none'}",
                                  component="narrated")
                _clear_status(); return
            _time.sleep(min(5 + _poll_n * 5, 30))
            _poll_n += 1
        else:
            _set_result("error:Container timed out waiting for FINISHED")
            _add_activity_log(article_id, "Instagram Reel Post Failed",
                              f"Container {container_id} timed out after 5 minutes.",
                              component="narrated")
            _clear_status(); return

        # ── 3. Publish ────────────────────────────────────────────────────
        _set_status("running:publish")
        print(f"IG Reel: publishing {container_id}", flush=True)
        resp = _req.post(f"{_IG_GRAPH_URL}/{IG_USER_ID}/media_publish",
                         data={"creation_id": container_id, "access_token": IG_ACCESS_TOKEN},
                         timeout=30)
        data = resp.json()

        # If publish returned an error, check if it actually went through
        if "id" not in data:
            err = data.get("error", {}).get("message", str(data))
            print(f"IG Reel: publish returned error ({err}), checking if post went through…", flush=True)
            _set_status("running:publish_verify")
            _time.sleep(10)
            try:
                check_resp = _req.get(
                    f"{_IG_GRAPH_URL}/{IG_USER_ID}/media",
                    params={"limit": "3", "fields": "id,timestamp,media_type",
                            "access_token": IG_ACCESS_TOKEN},
                    timeout=15,
                )
                recent = check_resp.json().get("data", [])
                if recent:
                    import datetime as _dt
                    for post in recent:
                        try:
                            ts = _dt.datetime.strptime(post["timestamp"], "%Y-%m-%dT%H:%M:%S+0000")
                            age = (_dt.datetime.utcnow() - ts).total_seconds()
                            if age < 120:
                                print(f"IG Reel: post {post['id']} found (age={age:.0f}s) — publish succeeded despite error", flush=True)
                                data = {"id": post["id"]}
                                break
                        except Exception:
                            continue
            except Exception as verify_err:
                print(f"IG Reel: verify check failed: {verify_err}", flush=True)

        if "id" not in data:
            err = data.get("error", {}).get("message", str(data))
            _set_result(f"error:Publish failed: {err}")
            _add_activity_log(article_id, "Instagram Reel Post Failed",
                              f"Publish failed for {container_id}.\nAPI error: {err}",
                              component="narrated")
            _clear_status(); return

        post_id = data["id"]
        # Persist media_id for live-check and archive operations
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"narrated_ig_media_id_{article_id}_{run_ts}", post_id, post_id))

        # ── 4. Fetch permalink ────────────────────────────────────────────
        try:
            plink = _ig_api_call("GET", f"{_IG_GRAPH_URL}/{post_id}",
                params={"fields": "permalink", "access_token": IG_ACCESS_TOKEN},
                timeout=15).json()
            permalink = plink.get("permalink") or f"https://www.instagram.com/p/{post_id}/"
        except Exception:
            permalink = f"https://www.instagram.com/p/{post_id}/"

        print(f"IG Reel: SUCCESS post_id={post_id} permalink={permalink}", flush=True)
        _set_result(f"done:{permalink}")
        _add_activity_log(article_id, "Posted Narrated Video to Instagram",
                          f"post_id={post_id}\npermalink={permalink}", component="narrated")
        _clear_status()

    except Exception as e:
        import traceback; traceback.print_exc()
        _set_result(f"error:Unexpected error: {e}")
        _add_activity_log(article_id, "Instagram Reel Post Failed",
                          f"Error: {e}", component="narrated")
        try: _clear_status()
        except Exception: pass


@app.route("/admin/articles/<int:article_id>/post-narrated-to-instagram", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_instagram(article_id):
    """Start a background thread to post a narrated video as an Instagram Reel.
    Body: {video_url, caption, run_ts}
    """
    if not IG_USER_ID or not IG_ACCESS_TOKEN:
        return jsonify({"error": "Instagram credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = (data.get("video_url") or "").strip()
    caption   = (data.get("caption")   or "").strip()[:2200]
    run_ts    = int(data.get("run_ts") or 0)
    if not video_url:
        return jsonify({"error": "No video URL provided."}), 400
    status_key = f"narrated_ig_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s",
            (status_key, "running:start", "running:start"))
    _add_activity_log(article_id, "Instagram Reel Post Started",
                      f"run_ts={run_ts}\nvideo_url={video_url[:120]}",
                      component="narrated")
    threading.Thread(target=_post_narrated_reel_worker,
                     args=(article_id, video_url, caption, run_ts),
                     daemon=True).start()
    return jsonify({"started": True})


@app.route("/admin/articles/<int:article_id>/narrated-ig-status")
@login_required
@admin_required
def admin_narrated_ig_status(article_id):
    """Poll status of a narrated video Instagram post. Query: ?ts=<run_ts>"""
    run_ts = request.args.get("ts", "0")
    history_raw = get_setting(f"narrated_ig_history_{article_id}_{run_ts}")
    history = json.loads(history_raw) if history_raw else []
    return jsonify({
        "status":  get_setting(f"narrated_ig_status_{article_id}_{run_ts}") or "",
        "result":  get_setting(f"narrated_ig_result_{article_id}_{run_ts}") or "",
        "history": history,
    })


@app.route("/admin/articles/<int:article_id>/check-narrated-ig-post")
@login_required
@admin_required
def admin_check_narrated_ig_post(article_id):
    """Check if a narrated video Instagram post is still live. Query: ?ts=<run_ts>"""
    if not IG_ACCESS_TOKEN:
        return jsonify({"error": "Instagram credentials not configured."}), 400
    run_ts   = request.args.get("ts", "0")
    media_id = get_setting(f"narrated_ig_media_id_{article_id}_{run_ts}")
    if not media_id:
        import re as _re
        result_val = get_setting(f"narrated_ig_result_{article_id}_{run_ts}") or ""
        if result_val.startswith("done:"):
            m = _re.search(r'/p/(\d{10,})', result_val)
            if m:
                media_id = m.group(1)
                execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                        "ON DUPLICATE KEY UPDATE value = %s",
                        (f"narrated_ig_media_id_{article_id}_{run_ts}", media_id, media_id))
    if not media_id:
        return jsonify({"error": "No published post found for this run."}), 400
    import requests as _req
    try:
        resp = _req.get(
            f"{_IG_GRAPH_URL}/{media_id}",
            params={"fields": "id,permalink", "access_token": IG_ACCESS_TOKEN},
            timeout=15,
        )
        data = resp.json()
        if "id" in data:
            _add_activity_log(article_id, "Check Narrated Post Live",
                              f"Post is live. media_id={media_id}, ts={run_ts}",
                              component="narrated")
            return jsonify({"live": True, "permalink": data.get("permalink", "")})
        api_err = data.get("error", {}).get("message", "Post not found.")
        for _suffix in (". Please read the Graph API", ". See the Graph API"):
            if _suffix in api_err:
                api_err = api_err[:api_err.index(_suffix)]
                break
        _add_activity_log(article_id, "Check Narrated Post Live",
                          f"Post not found. media_id={media_id}, ts={run_ts}: {api_err}",
                          component="narrated")
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        return jsonify({"error": f"Failed to reach Instagram API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/edit-narrated-ig-caption", methods=["POST"])
@login_required
@admin_required
def admin_edit_narrated_ig_caption(article_id):
    """Update the caption of a published narrated video Instagram post.
    Body: {caption, run_ts}
    """
    if not IG_ACCESS_TOKEN:
        return jsonify({"error": "Instagram credentials not configured."}), 400
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts") or "0")
    caption = (data.get("caption") or "").strip()[:2200]
    if not caption:
        return jsonify({"error": "Caption cannot be empty."}), 400
    media_id = get_setting(f"narrated_ig_media_id_{article_id}_{run_ts}")
    if not media_id:
        import re as _re
        result_val = get_setting(f"narrated_ig_result_{article_id}_{run_ts}") or ""
        if result_val.startswith("done:"):
            m = _re.search(r'/p/(\d{10,})', result_val)
            if m:
                media_id = m.group(1)
                execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                        "ON DUPLICATE KEY UPDATE value = %s",
                        (f"narrated_ig_media_id_{article_id}_{run_ts}", media_id, media_id))
    if not media_id:
        return jsonify({"error": "No published post found for this run."}), 400
    import requests as _req
    resp = _req.post(
        f"{_IG_GRAPH_URL}/{media_id}",
        data={"caption": caption, "access_token": IG_ACCESS_TOKEN},
        timeout=30,
    )
    result = resp.json()
    if result.get("success") or result.get("id"):
        _add_activity_log(article_id, "Caption Edited on Instagram (narrated)",
                          f"media_id={media_id}, ts={run_ts}\nNew caption: {caption[:200]}…",
                          component="narrated")
        return jsonify({"ok": True})
    err = result.get("error", {}).get("message", str(result))
    _add_activity_log(article_id, "Caption Edit Failed (narrated)",
                      f"media_id={media_id}, ts={run_ts}, error: {err}",
                      component="narrated")
    return jsonify({"error": err}), 400


@app.route("/admin/articles/<int:article_id>/archive-narrated-ig-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_ig_post(article_id):
    """Archive a narrated video Instagram post record. Body: {run_ts}"""
    data     = request.get_json() or {}
    run_ts   = int(data.get("run_ts") or 0)
    media_id_key = f"narrated_ig_media_id_{article_id}_{run_ts}"
    result_key   = f"narrated_ig_result_{article_id}_{run_ts}"
    history_key  = f"narrated_ig_history_{article_id}_{run_ts}"
    media_id = get_setting(media_id_key)
    result   = get_setting(result_key)
    if not media_id:
        import re as _re
        result_val = result or ""
        if result_val.startswith("done:"):
            m = _re.search(r'/p/(\d{10,})', result_val)
            if m:
                media_id = m.group(1)
    if not media_id:
        return jsonify({"error": "No published post found for this run."}), 400
    permalink = result[5:] if (result and result.startswith("done:")) else None
    history_raw = get_setting(history_key)
    history = json.loads(history_raw) if history_raw else []
    import datetime as _dt
    history.append({
        "media_id":    media_id,
        "permalink":   permalink,
        "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s",
            (history_key, json.dumps(history), json.dumps(history)))
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (media_id_key, "", ""))
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (result_key, "", ""))
    _add_activity_log(article_id, "Archived Narrated Post",
                      f"run_ts={run_ts}, media_id={media_id}, permalink={permalink or 'N/A'}",
                      component="narrated")
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/archive-instagram-post", methods=["POST"])
@login_required
@admin_required
def admin_article_archive_instagram_post(article_id):
    """Archive the current Instagram post record, preserving history.
    Body: {type} where type is 'cine' (default) or 'car'.
    """
    data      = request.get_json() or {}
    post_type = data.get("type", "cine")
    if post_type not in ("cine", "car"):
        post_type = "cine"
    keys     = _ig_keys(post_type, article_id)
    media_id = get_setting(keys["media_id"])
    result   = get_setting(keys["result"])
    if not media_id:
        import re as _re
        result_val = result or ""
        if result_val.startswith("done:"):
            m = _re.search(r'/p/(\d{10,})', result_val)
            if m:
                media_id = m.group(1)
                execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                        "ON DUPLICATE KEY UPDATE value = %s",
                        (keys["media_id"], media_id, media_id))
    if not media_id:
        return jsonify({"error": "No published post found for this article."}), 400

    # Parse permalink from the stored result (format: "done:https://…")
    permalink = None
    if result and result.startswith("done:"):
        permalink = result[5:]

    # Load existing history
    history_raw = get_setting(keys["history"])
    history = json.loads(history_raw) if history_raw else []

    # Load snapshot of the original post (caption, image/video URLs)
    snapshot_raw = get_setting(keys["snapshot"])
    snapshot     = json.loads(snapshot_raw) if snapshot_raw else {}

    # Append new archive entry with full snapshot data
    import datetime as _dt
    history.append({
        "media_id":    media_id,
        "permalink":   permalink,
        "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "caption":     snapshot.get("caption", ""),
        "image_urls":  snapshot.get("image_urls", []),
        "video_urls":  snapshot.get("video_urls", []),
        "post_type":   snapshot.get("post_type", post_type),
        "posted_at":   snapshot.get("posted_at", ""),
    })

    # Persist history, clear current post keys + snapshot
    execute(
        "INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
        (keys["history"], json.dumps(history), json.dumps(history)),
    )
    execute(
        "INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
        (keys["media_id"], "", ""),
    )
    execute(
        "INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
        (keys["result"], "", ""),
    )
    execute(
        "INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
        (keys["snapshot"], "", ""),
    )
    _add_activity_log(article_id, f"Post Archived ({post_type})",
                      f"media_id={media_id}, permalink={permalink or 'N/A'}",
                      component=("carousel" if post_type == "car" else "cinemagraph"))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/generate-ig-caption", methods=["POST"])
@login_required
@admin_required
def admin_article_generate_ig_caption(article_id):
    """Use OpenAI to generate an Instagram caption for this article."""
    if not _openai_client:
        return jsonify({"error": "OpenAI is not configured on this server."}), 400

    try:
        row = query_one(
            "SELECT title, content, slug FROM articles WHERE id = %s",
            (article_id,),
        )
        if not row:
            return jsonify({"error": "Article not found."}), 404

        title   = row[0] or ""
        content = row[1] or ""
        slug    = row[2] or ""

        # Strip HTML tags to get plain text excerpt for context
        import re as _re
        plain = _re.sub(r"<[^>]+>", " ", content)
        plain = _re.sub(r"\s+", " ", plain).strip()[:500]

        site_url = os.getenv("SITE_URL", "").rstrip("/")
        article_link = f"{site_url}/articles/{slug}" if site_url and slug else ""

        # Per-article prompt takes priority, then global, then hardcoded default
        system_msg = (
            get_setting(f'ig_caption_prompt_{article_id}')
            or get_setting('ig_caption_prompt', _IG_CAPTION_DEFAULT_PROMPT)
        )
        user_msg = f"Article title: {title}\n\nArticle summary: {plain}"

        response = _openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=600,
        )
        caption = response.choices[0].message.content.strip()

        # Append call-to-action with article link
        if article_link:
            caption += f"\n\nWant to read the full article? Access the following link:\n{article_link}"

        log_component = (request.get_json() or {}).get("component") or "carousel"
        if log_component not in ("carousel", "cinemagraph", "narrated"):
            log_component = "carousel"
        _add_activity_log(article_id, "Caption Generated (OpenAI)",
                          f"Prompt: {system_msg[:120]}…\n\nGenerated caption:\n{caption[:300]}…",
                          component=log_component)
        return jsonify({"caption": caption})

    except Exception as e:
        print(f"IG caption generation error: {e}", flush=True)
        try:
            log_component = (request.get_json() or {}).get("component") or "carousel"
            _add_activity_log(article_id, "Caption Generation Failed", f"Error: {e}",
                              component=log_component)
        except Exception:
            pass
        return jsonify({"error": f"Caption generation failed: {e}"}), 500


# ------------------------------------------------------------
# FACEBOOK PAGE POSTING
# ------------------------------------------------------------

def _post_to_facebook_worker(article_id, caption, post_type="cine"):
    """
    Background thread: post carousel images or cinemagraph video to Facebook Page.
    post_type: 'car' = multi-photo post; 'cine' = video post.
    """
    import time as _time
    import requests as _req
    import datetime as _dt

    keys       = _fb_keys(post_type, article_id)
    status_key = keys["status"]
    result_key = keys["result"]

    def _set_status(s):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, s, s))

    def _set_result(r):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, r, r))

    def _set_kv(k, v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (k, v, v))

    page_id      = FB_PAGE_ID
    access_token = FB_PAGE_ACCESS_TOKEN

    try:
        _set_status("running:0")

        # ── Fetch article data ─────────────────────────────────
        row = query_one(
            "SELECT title, carousel_images, carousel_cinemagraphs, carousel_punchlines "
            "FROM articles WHERE id = %s", (article_id,))
        if not row:
            _set_status("idle")
            _set_result("error:Article not found")
            _add_activity_log(article_id, f"Facebook Post Failed",
                              f"Article {article_id} not found in database.",
                              component="carousel" if post_type == "car" else "cinemagraph")
            return
        title       = row[0] or "Untitled"
        images_raw  = row[1]
        cines_raw   = row[2]
        punches_raw = row[3]
        images     = (json.loads(images_raw) if images_raw else [])[:10]
        cines      = (json.loads(cines_raw) if cines_raw else [])[:10]
        punchlines = (json.loads(punches_raw) if punches_raw else [])[:10]

        component_label = "carousel" if post_type == "car" else "cinemagraph"

        if post_type == "car":
            # ── Multi-photo post ────────────────────────────────
            valid_urls = [u for u in images if u]
            if not valid_urls:
                _set_status("idle")
                _set_result("error:No carousel images to post")
                _add_activity_log(article_id, f"Facebook Carousel Post Failed",
                                  f"No carousel images found.\nimages_raw length={len(images_raw) if images_raw else 0}",
                                  component=component_label)
                return

            _add_activity_log(article_id, f"Facebook Carousel Uploading Photos",
                              f"Uploading {len(valid_urls)} photos to Facebook as unpublished…\ncaption length={len(caption)}",
                              component=component_label)

            photo_ids = []
            for idx, img_url in enumerate(valid_urls):
                _set_status(f"running:upload:{idx+1}/{len(valid_urls)}")
                api_url = f"{_IG_GRAPH_URL}/{page_id}/photos"
                resp = _req.post(
                    api_url,
                    data={
                        "url": img_url,
                        "published": "false",
                        "access_token": access_token,
                    },
                    timeout=60,
                )
                data = resp.json()
                if "id" not in data:
                    err = data.get("error", {}).get("message", str(data))
                    _set_status("idle")
                    _set_result(f"error:Photo upload failed (slide {idx+1}): {err}")
                    _add_activity_log(article_id, f"Facebook Carousel Post Failed",
                                      f"Photo upload failed at slide {idx+1}/{len(valid_urls)}.\n"
                                      f"image_url={img_url}\n"
                                      f"API endpoint={api_url}\n"
                                      f"HTTP status={resp.status_code}\n"
                                      f"API response={json.dumps(data, indent=2)[:1500]}",
                                      component=component_label)
                    return
                photo_ids.append(data["id"])
                print(f"[FB] Uploaded photo {idx+1}/{len(valid_urls)}: {data['id']}", flush=True)

            # Create feed post with attached media
            _set_status("running:publishing")
            post_data = {
                "message": caption,
                "access_token": access_token,
            }
            for i, pid in enumerate(photo_ids):
                post_data[f"attached_media[{i}]"] = json.dumps({"media_fbid": pid})

            api_url = f"{_IG_GRAPH_URL}/{page_id}/feed"
            resp = _req.post(api_url, data=post_data, timeout=60)
            data = resp.json()
            if "id" not in data:
                err = data.get("error", {}).get("message", str(data))
                _set_status("idle")
                _set_result(f"error:Feed post failed: {err}")
                _add_activity_log(article_id, f"Facebook Carousel Post Failed",
                                  f"Feed post creation failed.\n"
                                  f"photo_ids={photo_ids}\n"
                                  f"API endpoint={api_url}\n"
                                  f"HTTP status={resp.status_code}\n"
                                  f"API response={json.dumps(data, indent=2)[:1500]}",
                                  component=component_label)
                return

            post_id = data["id"]
            permalink = f"https://www.facebook.com/{post_id}"
            print(f"[FB] Carousel posted: {permalink}", flush=True)

        else:
            # ── Cinemagraph video post ──────────────────────────
            # Find the first valid cinemagraph URL
            video_url = None
            for c in cines:
                if c:
                    video_url = c
                    break
            if not video_url:
                _set_status("idle")
                _set_result("error:No cinemagraph video to post")
                _add_activity_log(article_id, f"Facebook Cinemagraph Post Failed",
                                  f"No cinemagraph video URL found.\ncines count={len(cines)}",
                                  component=component_label)
                return

            _add_activity_log(article_id, f"Facebook Cinemagraph Uploading Video",
                              f"Uploading video to Facebook Page…\nvideo_url={video_url}\ncaption length={len(caption)}",
                              component=component_label)

            _set_status("running:upload")
            # Download video bytes first, then upload directly to avoid URL access issues
            vid_resp = _req.get(video_url, timeout=60)
            if vid_resp.status_code != 200:
                _set_status("idle")
                _set_result(f"error:Could not download video from GCS (HTTP {vid_resp.status_code})")
                _add_activity_log(article_id, f"Facebook Cinemagraph Post Failed",
                                  f"Could not download video from GCS.\nvideo_url={video_url}\nHTTP {vid_resp.status_code}",
                                  component=component_label)
                return
            api_url = f"{_FB_VIDEO_URL}/{page_id}/videos"
            resp = _req.post(
                api_url,
                data={
                    "description": caption,
                    "access_token": access_token,
                },
                files={
                    "source": ("video.mp4", vid_resp.content, "video/mp4"),
                },
                timeout=120,
            )
            data = resp.json()
            if "id" not in data:
                err = data.get("error", {}).get("message", str(data))
                _set_status("idle")
                _set_result(f"error:Video upload failed: {err}")
                _add_activity_log(article_id, f"Facebook Cinemagraph Post Failed",
                                  f"Video upload failed.\n"
                                  f"video_url={video_url}\n"
                                  f"API endpoint={api_url}\n"
                                  f"HTTP status={resp.status_code}\n"
                                  f"API response={json.dumps(data, indent=2)[:1500]}",
                                  component=component_label)
                return

            post_id = data["id"]
            permalink = f"https://www.facebook.com/{page_id}/videos/{post_id}"
            print(f"[FB] Video posted: {permalink}", flush=True)

        # ── Save results ───────────────────────────────────────
        _set_kv(keys["media_id"], str(post_id))
        snapshot = {
            "caption": caption,
            "post_type": post_type,
            "posted_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image_urls": images if post_type == "car" else [],
            "video_urls": cines if post_type == "cine" else [],
        }
        _set_kv(keys["snapshot"], json.dumps(snapshot))
        _set_result(f"done:{permalink}")

        slides_info = f"{len(valid_urls)} photos" if post_type == "car" else f"video_url={video_url}"
        _add_activity_log(
            article_id,
            f"Facebook {component_label.title()} Post Published",
            f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
            f"post_id={post_id}\n{slides_info}\ncaption length={len(caption)}",
            component=component_label,
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[FB] Worker error: {e}\n{tb}", flush=True)
        _set_status("idle")
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Facebook Post Failed (Exception)",
                          f"Unhandled exception:\n{str(e)}\n\nTraceback:\n{tb[:2000]}",
                          component="carousel" if post_type == "car" else "cinemagraph")


def _post_narrated_fb_worker(article_id, video_url, caption, run_ts):
    """Background thread: post narrated video to Facebook Page."""
    import requests as _req

    status_key = f"narrated_fb_status_{article_id}_{run_ts}"
    result_key = f"narrated_fb_result_{article_id}_{run_ts}"

    def _set_status(s):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, s, s))

    def _set_result(r):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, r, r))

    def _set_kv(k, v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (k, v, v))

    try:
        _set_status("running:upload")

        _add_activity_log(
            article_id,
            "Facebook Narrated Video Uploading",
            f"Uploading narrated video to Facebook Page…\n"
            f"video_url={video_url}\nrun_ts={run_ts}\n"
            f"page_id={FB_PAGE_ID}\ncaption length={len(caption)}",
            component="narrated",
        )

        # Download video bytes first, then upload directly
        vid_resp = _req.get(video_url, timeout=60)
        if vid_resp.status_code != 200:
            _set_status("idle")
            _set_result(f"error:Could not download video from GCS (HTTP {vid_resp.status_code})")
            _add_activity_log(article_id, "Facebook Narrated Video Post Failed",
                              f"Could not download video from GCS.\nvideo_url={video_url}\nHTTP {vid_resp.status_code}",
                              component="narrated")
            return
        api_url = f"{_FB_VIDEO_URL}/{FB_PAGE_ID}/videos"
        resp = _req.post(
            api_url,
            data={
                "description": caption,
                "access_token": FB_PAGE_ACCESS_TOKEN,
            },
            files={
                "source": ("video.mp4", vid_resp.content, "video/mp4"),
            },
            timeout=180,
        )
        data = resp.json()

        if "id" not in data:
            err = data.get("error", {}).get("message", str(data))
            _set_status("idle")
            _set_result(f"error:{err}")
            _add_activity_log(article_id, "Facebook Narrated Video Post Failed",
                              f"Video upload failed.\n"
                              f"video_url={video_url}\n"
                              f"API endpoint={api_url}\n"
                              f"HTTP status={resp.status_code}\n"
                              f"API response={json.dumps(data, indent=2)[:1500]}",
                              component="narrated")
            return

        video_id  = data["id"]
        permalink = f"https://www.facebook.com/{FB_PAGE_ID}/videos/{video_id}"

        _set_kv(f"narrated_fb_media_id_{article_id}_{run_ts}", str(video_id))
        _set_result(f"done:{permalink}")

        _add_activity_log(
            article_id,
            "Facebook Narrated Video Post Published",
            f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
            f"video_id={video_id}\nrun_ts={run_ts}\ncaption length={len(caption)}",
            component="narrated",
        )
        print(f"[FB] Narrated video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[FB] Narrated worker error: {e}\n{tb}", flush=True)
        _set_status("idle")
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Facebook Narrated Video Post Failed (Exception)",
                          f"Unhandled exception:\n{str(e)}\n\nTraceback:\n{tb[:2000]}",
                          component="narrated")


# ── Facebook: Post carousel / cinemagraph ──────────────────────

@app.route("/admin/articles/<int:article_id>/post-to-facebook", methods=["POST"])
@login_required
@admin_required
def admin_article_post_to_facebook(article_id):
    if not FB_PAGE_ID or not FB_PAGE_ACCESS_TOKEN:
        return jsonify({"error": "Facebook Page credentials not configured. "
                        "Set FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN environment variables."}), 400
    data      = request.get_json() or {}
    caption   = data.get("caption", "")
    post_type = data.get("type", "cine")

    keys       = _fb_keys(post_type, article_id)
    status_key = keys["status"]
    result_key = keys["result"]

    component_label = "carousel" if post_type == "car" else "cinemagraph"
    caption_preview = (caption[:120] + "…") if len(caption) > 120 else caption
    _add_activity_log(article_id, f"Facebook {component_label.title()} Post Started",
                      f"type={post_type}\ncaption={caption_preview}\npage_id={FB_PAGE_ID}",
                      component=component_label)

    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (result_key, "", ""))

    t = threading.Thread(target=_post_to_facebook_worker,
                         args=(article_id, caption, post_type), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/fb-post-status")
@login_required
@admin_required
def admin_article_fb_post_status(article_id):
    post_type = request.args.get("type", "cine")
    keys      = _fb_keys(post_type, article_id)
    status    = get_setting(keys["status"]) or "idle"
    result    = get_setting(keys["result"]) or ""
    media_id  = get_setting(keys["media_id"]) or ""
    history   = json.loads(get_setting(keys["history"]) or "[]")
    snap_raw  = get_setting(keys["snapshot"]) or "{}"
    snapshot  = json.loads(snap_raw) if snap_raw else {}
    caption   = snapshot.get("caption", "")
    return jsonify({"status": status, "result": result, "media_id": media_id,
                     "history": history, "caption": caption})


@app.route("/admin/articles/<int:article_id>/check-facebook-post")
@login_required
@admin_required
def admin_article_check_facebook_post(article_id):
    import requests as _req
    post_type = request.args.get("type", "cine")
    keys      = _fb_keys(post_type, article_id)
    media_id  = get_setting(keys["media_id"])
    if not media_id:
        return jsonify({"live": False, "reason": "No post ID stored"})
    try:
        resp = _req.get(
            f"{_IG_GRAPH_URL}/{media_id}",
            params={"fields": "id,permalink_url", "access_token": FB_PAGE_ACCESS_TOKEN},
            timeout=15,
        )
        data = resp.json()
        if "id" in data:
            return jsonify({"live": True, "permalink": data.get("permalink_url", "")})
        return jsonify({"live": False, "reason": data.get("error", {}).get("message", "Not found")})
    except Exception as e:
        return jsonify({"live": False, "reason": str(e)})


@app.route("/admin/articles/<int:article_id>/archive-facebook-post", methods=["POST"])
@login_required
@admin_required
def admin_article_archive_facebook_post(article_id):
    data      = request.get_json() or {}
    post_type = data.get("type", "cine")
    keys      = _fb_keys(post_type, article_id)
    media_id  = get_setting(keys["media_id"])
    if not media_id:
        return jsonify({"error": "No post to archive"}), 400

    snap_raw = get_setting(keys["snapshot"])
    snapshot = json.loads(snap_raw) if snap_raw else {}
    result   = get_setting(keys["result"]) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""

    history = json.loads(get_setting(keys["history"]) or "[]")
    history.append({
        "media_id":    media_id,
        "permalink":   permalink,
        "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "caption":     snapshot.get("caption", ""),
        "post_type":   snapshot.get("post_type", post_type),
        "posted_at":   snapshot.get("posted_at", ""),
    })
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (keys["history"], hist_val, hist_val))

    # Clear current post
    for k in ("media_id", "result", "status", "snapshot"):
        execute("DELETE FROM site_settings WHERE `key` = %s", (keys[k],))

    component_label = "carousel" if post_type == "car" else "cinemagraph"
    _add_activity_log(article_id, f"Facebook {component_label.title()} Post Archived",
                      f"Post {media_id} archived", component=component_label)

    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/edit-facebook-caption", methods=["POST"])
@login_required
@admin_required
def admin_article_edit_facebook_caption(article_id):
    import requests as _req
    data      = request.get_json() or {}
    caption   = data.get("caption", "")
    post_type = data.get("type", "cine")
    keys      = _fb_keys(post_type, article_id)
    media_id  = get_setting(keys["media_id"])
    if not media_id or not FB_PAGE_ACCESS_TOKEN:
        return jsonify({"error": "No post or credentials"}), 400
    try:
        resp = _req.post(
            f"{_IG_GRAPH_URL}/{media_id}",
            data={"message": caption, "access_token": FB_PAGE_ACCESS_TOKEN},
            timeout=15,
        )
        data = resp.json()
        if data.get("success") or "id" in data:
            return jsonify({"ok": True})
        return jsonify({"error": data.get("error", {}).get("message", "Unknown error")}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Facebook: Post narrated video ──────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-facebook", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_facebook(article_id):
    if not FB_PAGE_ID or not FB_PAGE_ACCESS_TOKEN:
        return jsonify({"error": "Facebook Page credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    status_key = f"narrated_fb_status_{article_id}_{run_ts}"
    result_key = f"narrated_fb_result_{article_id}_{run_ts}"
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (result_key, "", ""))

    t = threading.Thread(target=_post_narrated_fb_worker,
                         args=(article_id, video_url, caption, run_ts), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-fb-status")
@login_required
@admin_required
def admin_narrated_fb_status(article_id):
    run_ts     = request.args.get("ts", "0")
    status_key = f"narrated_fb_status_{article_id}_{run_ts}"
    result_key = f"narrated_fb_result_{article_id}_{run_ts}"
    status     = get_setting(status_key) or "idle"
    result     = get_setting(result_key) or ""
    history    = json.loads(get_setting(f"narrated_fb_history_{article_id}_{run_ts}") or "[]")
    return jsonify({"status": status, "result": result, "history": history})


@app.route("/admin/articles/<int:article_id>/check-narrated-fb-post")
@login_required
@admin_required
def admin_check_narrated_fb_post(article_id):
    import requests as _req
    run_ts   = request.args.get("ts", "0")
    media_id = get_setting(f"narrated_fb_media_id_{article_id}_{run_ts}")
    if not media_id:
        return jsonify({"live": False, "reason": "No post ID stored"})
    try:
        resp = _req.get(
            f"{_IG_GRAPH_URL}/{media_id}",
            params={"fields": "id,permalink_url", "access_token": FB_PAGE_ACCESS_TOKEN},
            timeout=15,
        )
        data = resp.json()
        if "id" in data:
            return jsonify({"live": True, "permalink": data.get("permalink_url", "")})
        return jsonify({"live": False, "reason": data.get("error", {}).get("message", "Not found")})
    except Exception as e:
        return jsonify({"live": False, "reason": str(e)})


@app.route("/admin/articles/<int:article_id>/archive-narrated-fb-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_fb_post(article_id):
    data   = request.get_json() or {}
    run_ts = str(data.get("run_ts", "0"))

    media_key = f"narrated_fb_media_id_{article_id}_{run_ts}"
    result_key = f"narrated_fb_result_{article_id}_{run_ts}"
    status_key = f"narrated_fb_status_{article_id}_{run_ts}"
    history_key = f"narrated_fb_history_{article_id}_{run_ts}"

    media_id  = get_setting(media_key)
    if not media_id:
        return jsonify({"error": "No post to archive"}), 400

    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""

    history = json.loads(get_setting(history_key) or "[]")
    history.append({
        "media_id":    media_id,
        "permalink":   permalink,
        "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))

    for k in (media_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))

    _add_activity_log(article_id, "Facebook Video Post Archived",
                      f"Post {media_id} archived", component="narrated")

    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-facebook-post", methods=["POST"])
@login_required
@admin_required
def admin_article_delete_facebook_post(article_id):
    import requests as _req
    import traceback as _tb
    import datetime as _dt
    data      = request.get_json() or {}
    post_type = data.get("type", "cine")
    keys      = _fb_keys(post_type, article_id)
    media_id  = get_setting(keys["media_id"])
    component_label = "carousel" if post_type == "car" else "cinemagraph"

    if not media_id:
        return jsonify({"error": "No post record found to archive."}), 400

    # Step 1: Attempt to delete from Facebook
    already_gone = False
    try:
        _add_activity_log(article_id, f"Facebook {component_label.title()} Delete Started",
                          f"Attempting to delete post {media_id} from Facebook…", component=component_label)
        resp = _req.delete(
            f"{_IG_GRAPH_URL}/{media_id}",
            params={"access_token": FB_PAGE_ACCESS_TOKEN},
            timeout=15,
        )
        if resp.ok or resp.status_code == 404:
            # Successfully deleted, or already gone (404)
            already_gone = resp.status_code == 404
        else:
            try:
                err_data = resp.json().get("error", {})
                msg = err_data.get("message", resp.text)
            except Exception:
                msg = resp.text
            # If Facebook says post doesn't exist or can't be deleted, treat as already gone
            if "does not exist" in msg or "Unsupported delete" in msg or "cannot be loaded" in msg:
                already_gone = True
                _add_activity_log(article_id, f"Facebook {component_label.title()} Post Not Found",
                                  f"Post {media_id} no longer exists on Facebook (HTTP {resp.status_code}).\n{msg}",
                                  component=component_label)
            else:
                # Genuine error — log and return
                _add_activity_log(article_id, f"Facebook {component_label.title()} Delete Failed",
                                  f"media_id={media_id}\nHTTP {resp.status_code}: {msg}", component=component_label)
                return jsonify({"error": f"Could not delete the post from Facebook: {msg}"}), 400
    except Exception as e:
        # Network error, timeout, etc. — still allow archiving
        already_gone = True
        _add_activity_log(article_id, f"Facebook {component_label.title()} Delete Error",
                          f"media_id={media_id}\nException: {e}\n{_tb.format_exc()}", component=component_label)

    # Step 2: Archive locally (always runs if we get here)
    try:
        snap_raw  = get_setting(keys["snapshot"])
        snapshot  = json.loads(snap_raw) if snap_raw else {}
        result_val = get_setting(keys["result"]) or ""
        permalink = result_val.replace("done:", "") if result_val.startswith("done:") else ""

        history = json.loads(get_setting(keys["history"]) or "[]")
        history.append({
            "media_id":    media_id,
            "permalink":   permalink,
            "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "caption":     snapshot.get("caption", ""),
            "post_type":   snapshot.get("post_type", post_type),
            "posted_at":   snapshot.get("posted_at", ""),
            "image_urls":  snapshot.get("image_urls", []),
            "video_urls":  snapshot.get("video_urls", []),
        })
        hist_val = json.dumps(history)
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (keys["history"], hist_val, hist_val))
        for k in ("media_id", "result", "status", "snapshot"):
            execute("DELETE FROM site_settings WHERE `key` = %s", (keys[k],))

        action = "Post already removed from Facebook, archived locally" if already_gone else f"Post {media_id} deleted from Facebook and archived"
        _add_activity_log(article_id, f"Facebook {component_label.title()} Post Archived",
                          f"{action}\npermalink={permalink}", component=component_label)

        return jsonify({"ok": True, "history": history})
    except Exception as e:
        _add_activity_log(article_id, f"Facebook {component_label.title()} Archive Failed",
                          f"media_id={media_id}\nException: {e}\n{_tb.format_exc()}", component=component_label)
        return jsonify({"error": f"Post was removed from Facebook but archiving failed: {e}"}), 500


@app.route("/admin/articles/<int:article_id>/delete-narrated-fb-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_fb_post(article_id):
    import requests as _req
    import traceback as _tb
    import datetime as _dt
    data   = request.get_json() or {}
    run_ts = str(data.get("run_ts", "0"))

    media_key   = f"narrated_fb_media_id_{article_id}_{run_ts}"
    result_key  = f"narrated_fb_result_{article_id}_{run_ts}"
    status_key  = f"narrated_fb_status_{article_id}_{run_ts}"
    history_key = f"narrated_fb_history_{article_id}_{run_ts}"

    media_id = get_setting(media_key)
    if not media_id:
        return jsonify({"error": "No post record found to archive."}), 400

    # Step 1: Attempt to delete from Facebook
    already_gone = False
    try:
        _add_activity_log(article_id, "Facebook Video Delete Started",
                          f"Attempting to delete post {media_id} from Facebook…", component="narrated")
        resp = _req.delete(
            f"{_IG_GRAPH_URL}/{media_id}",
            params={"access_token": FB_PAGE_ACCESS_TOKEN},
            timeout=15,
        )
        if resp.ok or resp.status_code == 404:
            already_gone = resp.status_code == 404
        else:
            try:
                err_data = resp.json().get("error", {})
                msg = err_data.get("message", resp.text)
            except Exception:
                msg = resp.text
            if "does not exist" in msg or "Unsupported delete" in msg or "cannot be loaded" in msg:
                already_gone = True
                _add_activity_log(article_id, "Facebook Video Post Not Found",
                                  f"Post {media_id} no longer exists on Facebook (HTTP {resp.status_code}).\n{msg}",
                                  component="narrated")
            else:
                _add_activity_log(article_id, "Facebook Video Delete Failed",
                                  f"media_id={media_id}\nHTTP {resp.status_code}: {msg}", component="narrated")
                return jsonify({"error": f"Could not delete the post from Facebook: {msg}"}), 400
    except Exception as e:
        already_gone = True
        _add_activity_log(article_id, "Facebook Video Delete Error",
                          f"media_id={media_id}\nException: {e}\n{_tb.format_exc()}", component="narrated")

    # Step 2: Archive locally (always runs if we get here)
    try:
        result_val = get_setting(result_key) or ""
        permalink  = result_val.replace("done:", "") if result_val.startswith("done:") else ""

        history = json.loads(get_setting(history_key) or "[]")
        history.append({
            "media_id":    media_id,
            "permalink":   permalink,
            "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        hist_val = json.dumps(history)
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
        for k in (media_key, result_key, status_key):
            execute("DELETE FROM site_settings WHERE `key` = %s", (k,))

        action = "Post already removed from Facebook, archived locally" if already_gone else f"Post {media_id} deleted from Facebook and archived"
        _add_activity_log(article_id, "Facebook Video Post Archived",
                          f"{action}\npermalink={permalink}", component="narrated")

        return jsonify({"ok": True, "history": history})
    except Exception as e:
        _add_activity_log(article_id, "Facebook Video Archive Failed",
                          f"media_id={media_id}\nException: {e}\n{_tb.format_exc()}", component="narrated")
        return jsonify({"error": f"Post was removed from Facebook but archiving failed: {e}"}), 500


# ------------------------------------------------------------
# ML API HEALTH CHECK
# ------------------------------------------------------------
@app.route("/ml-health")
def ml_health():
    """Check if the ML API is available."""
    try:
        response = http_requests.get(f"{ML_API_URL}/health", timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "error": str(e),
            "ml_api_url": ML_API_URL,
        }), 503


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"[Website] Starting on port {port}")
    print(f"[Website] ML API URL: {ML_API_URL}")
    app.run(host="0.0.0.0", port=port, debug=debug)
