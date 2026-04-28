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
GIFT_PERSONA_PRESETS = [
    "🏛️ Architecture enthusiasts",
    "🎨 Art lovers",
    "✈️ Aviation enthusiasts",
    "🎂 Birthdays & anniversaries",
    "🔬 Biology lovers",
    "🐦 Bird lovers",
    "🎄 Christmas & holiday gifts",
    "📚 Education & learning",
    "🌸 Flower enthusiasts",
    "⚽ Football fans",
    "🌍 Geography lovers",
    "🏔️ Hiking & adventure lovers",
    "📜 History enthusiasts",
    "🏠 Home & office décor",
    "🦋 Insect enthusiasts",
    "⚓ Maritime lovers",
    "🎖️ Military history fans",
    "🎵 Music lovers",
    "🌿 Nature lovers",
    "🏅 Olympics fans",
    "🚂 Railway enthusiasts",
    "🎓 Retirement gifts",
    "👑 Royal & monarchy fans",
    "🧳 Stamp collectors",
    "🚀 Space & astronomy fans",
    "⚽ Sports fans",
    "💍 Wedding & engagement gifts",
]

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

# --- X (Twitter) API (optional — for posting) ---
X_API_KEY             = os.getenv("X_API_KEY", "")
X_API_SECRET          = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN        = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET", "")
X_CONFIGURED          = bool(X_API_KEY and X_API_SECRET and X_ACCESS_TOKEN and X_ACCESS_TOKEN_SECRET)
if X_CONFIGURED:
    print("X (Twitter): credentials configured")
else:
    print("X (Twitter): credentials not set — posting disabled")

_X_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
_X_TWEET_URL  = "https://api.twitter.com/2/tweets"


def _x_auth():
    from requests_oauthlib import OAuth1
    return OAuth1(X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)


# --- Threads API (optional — for posting) ---
THREADS_USER_ID      = os.getenv("THREADS_USER_ID", "")
THREADS_ACCESS_TOKEN = os.getenv("THREADS_ACCESS_TOKEN", "")
THREADS_CONFIGURED   = bool(THREADS_USER_ID and THREADS_ACCESS_TOKEN)
_THREADS_API_URL     = "https://graph.threads.net/v1.0"
if THREADS_CONFIGURED:
    print("Threads: credentials configured")
else:
    print("Threads: THREADS_USER_ID / THREADS_ACCESS_TOKEN not set — posting disabled")


# --- Pinterest API (optional — for pinning) ---
PINTEREST_ACCESS_TOKEN  = os.getenv("PINTEREST_ACCESS_TOKEN", "")
PINTEREST_BOARD_ID      = os.getenv("PINTEREST_BOARD_ID", "")
PINTEREST_CLIENT_ID     = os.getenv("PINTEREST_CLIENT_ID", "")
PINTEREST_CLIENT_SECRET = os.getenv("PINTEREST_CLIENT_SECRET", "")
PINTEREST_REDIRECT_URI  = os.getenv("SITE_URL", "").rstrip("/") + "/admin/pinterest-oauth-callback"
# Token can come from env var OR from DB (OAuth flow) — only board ID is required at startup
PINTEREST_CONFIGURED    = bool(PINTEREST_BOARD_ID)
_PINTEREST_API_URL      = "https://api.pinterest.com/v5"
if PINTEREST_BOARD_ID:
    print("Pinterest: board ID configured")
else:
    print("Pinterest: PINTEREST_BOARD_ID not set — posting disabled")


# --- TikTok API (optional — for posting videos) ---
TIKTOK_CLIENT_KEY    = os.getenv("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")
TIKTOK_ACCESS_TOKEN  = os.getenv("TIKTOK_ACCESS_TOKEN", "")
TIKTOK_CONFIGURED    = bool(TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET and TIKTOK_ACCESS_TOKEN)
_TIKTOK_API_URL      = "https://open.tiktokapis.com/v2"
if TIKTOK_CONFIGURED:
    print("TikTok: credentials configured")
else:
    print("TikTok: credentials not set — posting disabled")


# --- LinkedIn API (optional — for sharing posts) ---
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
LINKEDIN_ORG_ID       = os.getenv("LINKEDIN_ORG_ID", "")
LINKEDIN_CONFIGURED   = bool(LINKEDIN_ACCESS_TOKEN and LINKEDIN_ORG_ID)
_LINKEDIN_API_URL     = "https://api.linkedin.com/rest"
if LINKEDIN_CONFIGURED:
    print("LinkedIn: credentials configured")
else:
    print("LinkedIn: credentials not set — posting disabled")

# --- Bluesky (AT Protocol) ---
BLUESKY_HANDLE       = os.getenv("BLUESKY_HANDLE", "")
BLUESKY_APP_PASSWORD = os.getenv("BLUESKY_APP_PASSWORD", "")
BLUESKY_CONFIGURED   = bool(BLUESKY_HANDLE and BLUESKY_APP_PASSWORD)
if BLUESKY_CONFIGURED:
    print("Bluesky: credentials configured")
else:
    print("Bluesky: BLUESKY_HANDLE / BLUESKY_APP_PASSWORD not set — posting disabled")

# --- Reddit API ---
REDDIT_CLIENT_ID      = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET  = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_REFRESH_TOKEN  = os.getenv("REDDIT_REFRESH_TOKEN", "")
REDDIT_SUBREDDIT      = os.getenv("REDDIT_SUBREDDIT", "")
REDDIT_CONFIGURED     = bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_REFRESH_TOKEN and REDDIT_SUBREDDIT)
if REDDIT_CONFIGURED:
    print("Reddit: credentials configured")
else:
    print("Reddit: credentials not set — posting disabled")

# --- Telegram Bot ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_CONFIGURED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
if TELEGRAM_CONFIGURED:
    print("Telegram: credentials configured")
else:
    print("Telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — posting disabled")

# --- Vimeo ---
VIMEO_ACCESS_TOKEN = os.getenv("VIMEO_ACCESS_TOKEN", "")
VIMEO_CONFIGURED   = bool(VIMEO_ACCESS_TOKEN)
if VIMEO_CONFIGURED:
    print("Vimeo: credentials configured")
else:
    print("Vimeo: VIMEO_ACCESS_TOKEN not set — posting disabled")

# --- Mastodon ---
MASTODON_INSTANCE_URL  = os.getenv("MASTODON_INSTANCE_URL", "").rstrip("/")
MASTODON_ACCESS_TOKEN  = os.getenv("MASTODON_ACCESS_TOKEN", "")
MASTODON_CONFIGURED    = bool(MASTODON_INSTANCE_URL and MASTODON_ACCESS_TOKEN)
if MASTODON_CONFIGURED:
    print("Mastodon: credentials configured")
else:
    print("Mastodon: credentials not set — posting disabled")

# --- VKontakte (VK) ---
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN", "")
VK_OWNER_ID     = os.getenv("VK_OWNER_ID", "")
VK_CONFIGURED   = bool(VK_ACCESS_TOKEN and VK_OWNER_ID)
if VK_CONFIGURED:
    print("VK: credentials configured")
else:
    print("VK: VK_ACCESS_TOKEN / VK_OWNER_ID not set — posting disabled")

# --- Tumblr ---
TUMBLR_ACCESS_TOKEN = os.getenv("TUMBLR_ACCESS_TOKEN", "")
TUMBLR_BLOG_NAME    = os.getenv("TUMBLR_BLOG_NAME", "")
TUMBLR_CONFIGURED   = bool(TUMBLR_ACCESS_TOKEN and TUMBLR_BLOG_NAME)
if TUMBLR_CONFIGURED:
    print("Tumblr: credentials configured")
else:
    print("Tumblr: TUMBLR_ACCESS_TOKEN / TUMBLR_BLOG_NAME not set — posting disabled")

# --- Google Analytics 4 ---
GA_MEASUREMENT_ID = os.getenv("GA_MEASUREMENT_ID", "")
if GA_MEASUREMENT_ID:
    print(f"GA4: Measurement ID configured ({GA_MEASUREMENT_ID})")
else:
    print("GA4: GA_MEASUREMENT_ID not set — analytics disabled")


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


def save_catalogue_image(file_storage) -> Optional[str]:
    """
    Save an uploaded catalogue image (werkzeug FileStorage).
    Uploads to GCS when configured, otherwise falls back to static/uploads/catalogue/.
    Returns the stored URL/relative-path (or None on failure).
    """
    if not file_storage or not file_storage.filename:
        return None
    import uuid
    ext = Path(file_storage.filename).suffix.lower() or ".jpg"
    # Normalise weird extensions
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        ext = ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    data = file_storage.read()
    if not data:
        return None
    content_type = file_storage.content_type or "image/jpeg"

    # Try GCS first
    gcs_url = upload_bytes_to_gcs(data, f"catalogue/{fname}", content_type=content_type)
    if gcs_url:
        return gcs_url

    # Fallback: save locally under static/uploads/catalogue/
    local_dir = Path("static") / "uploads" / "catalogue"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / fname
    with open(local_path, "wb") as fp:
        fp.write(data)
    return f"uploads/catalogue/{fname}"


def slugify(value: str) -> str:
    """Turn an arbitrary string into a URL-safe slug (lowercase, a-z/0-9/hyphens)."""
    if not value:
        return ""
    value = value.strip().lower()
    # Replace anything that isn't alphanumeric with a hyphen
    value = re.sub(r"[^a-z0-9]+", "-", value)
    # Collapse runs and trim
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:200]


def catalogue_img_url(url):
    """
    Resolve a catalogue image URL for the browser.
    - Full http(s) URLs (GCS, external) → returned unchanged.
    - Relative paths → served via /static/.
    """
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://") or url.startswith("/"):
        return url
    # Relative static path
    return url_for("static", filename=url)


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


def compose_carousel_slide(img_bytes, punchline, slide_index, total_slides, band_top=None, hint_text=None):
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
        hint_text:    If set, replaces the default "Slide right for more"
                      swipe hint and is shown on ALL slides (including last).
                      Used for branding (e.g. domain name on Facebook posts).
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

    # ── Swipe hint / branding line ───────────────────────────────────────────
    if hint_text:
        # Custom hint (e.g. domain branding) — shown on ALL slides
        hint_y = h - hint_h + int(font_hint.size * 0.4)
        draw.text((pad_x, hint_y), hint_text, font=font_hint,
                  fill=(255, 215, 0, 255))
    elif slide_index < total_slides - 1:
        # Default swipe hint — all slides except the last
        hint_y = h - hint_h + int(font_hint.size * 0.4)
        draw.text((pad_x, hint_y), "Slide right for more »»»", font=font_hint,
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
    cur.execute("""CREATE TABLE IF NOT EXISTS article_engagement (
        id INT AUTO_INCREMENT PRIMARY KEY,
        article_id INT NOT NULL,
        platform VARCHAR(20) NOT NULL,
        content_type VARCHAR(20) NOT NULL,
        post_id VARCHAR(255),
        permalink VARCHAR(500),
        likes INT DEFAULT 0,
        views INT DEFAULT 0,
        shares INT DEFAULT 0,
        comments INT DEFAULT 0,
        saves INT DEFAULT 0,
        clicks INT DEFAULT 0,
        impressions INT DEFAULT 0,
        reach INT DEFAULT 0,
        fetched_at DATETIME NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_article (article_id),
        INDEX idx_platform (platform),
        INDEX idx_fetched (fetched_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    cur.execute("""CREATE TABLE IF NOT EXISTS post_metrics_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        article_id INT NOT NULL,
        platform VARCHAR(20) NOT NULL,
        content_type VARCHAR(20) NOT NULL,
        post_id VARCHAR(255),
        likes INT DEFAULT 0,
        views INT DEFAULT 0,
        shares INT DEFAULT 0,
        comments_count INT DEFAULT 0,
        saves INT DEFAULT 0,
        clicks INT DEFAULT 0,
        impressions INT DEFAULT 0,
        reach INT DEFAULT 0,
        fetched_at DATETIME NOT NULL,
        hours_since_post INT DEFAULT 0,
        INDEX idx_article_platform (article_id, platform),
        INDEX idx_fetched (fetched_at),
        INDEX idx_hours (hours_since_post)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    cur.execute("""CREATE TABLE IF NOT EXISTS post_comments (
        id INT AUTO_INCREMENT PRIMARY KEY,
        article_id INT NOT NULL,
        platform VARCHAR(20) NOT NULL,
        content_type VARCHAR(20) NOT NULL,
        post_id VARCHAR(255),
        comment_id VARCHAR(255),
        comment_text TEXT,
        comment_author VARCHAR(255),
        comment_timestamp DATETIME,
        sentiment_score FLOAT DEFAULT NULL,
        fetched_at DATETIME NOT NULL,
        INDEX idx_article (article_id),
        INDEX idx_platform_post (platform, post_id),
        UNIQUE KEY uk_comment (platform, comment_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    cur.execute("""CREATE TABLE IF NOT EXISTS posting_log (
        id INT AUTO_INCREMENT PRIMARY KEY,
        article_id INT NOT NULL,
        platform VARCHAR(20) NOT NULL,
        content_type VARCHAR(20) NOT NULL,
        post_id VARCHAR(255),
        permalink VARCHAR(500),
        caption TEXT,
        hashtags TEXT,
        posted_at DATETIME NOT NULL,
        posted_day_of_week TINYINT,
        posted_hour TINYINT,
        posted_is_weekend BOOLEAN DEFAULT FALSE,
        article_title VARCHAR(500),
        article_word_count INT DEFAULT 0,
        article_slug VARCHAR(255),
        image_count INT DEFAULT 0,
        video_duration_seconds INT DEFAULT 0,
        INDEX idx_article (article_id),
        INDEX idx_platform (platform),
        INDEX idx_posted (posted_at),
        INDEX idx_timing (posted_day_of_week, posted_hour)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    cur.execute("""CREATE TABLE IF NOT EXISTS follower_snapshots (
        id INT AUTO_INCREMENT PRIMARY KEY,
        platform VARCHAR(20) NOT NULL,
        follower_count INT DEFAULT 0,
        following_count INT DEFAULT 0,
        post_count INT DEFAULT 0,
        snapshot_date DATE NOT NULL,
        fetched_at DATETIME NOT NULL,
        UNIQUE KEY uk_platform_date (platform, snapshot_date),
        INDEX idx_platform (platform)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    cur.execute("""CREATE TABLE IF NOT EXISTS short_links (
        id INT AUTO_INCREMENT PRIMARY KEY,
        code VARCHAR(8) UNIQUE NOT NULL,
        article_id INT NOT NULL,
        platform VARCHAR(20) NOT NULL,
        target_url VARCHAR(1000) NOT NULL,
        click_count INT DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_code (code),
        UNIQUE KEY uk_article_platform (article_id, platform)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    conn.commit()
    # Add UTM attribution columns to orders table if not present
    try:
        cur.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'orders' AND COLUMN_NAME = 'utm_source'
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE orders ADD COLUMN utm_source VARCHAR(50) DEFAULT NULL")
            cur.execute("ALTER TABLE orders ADD COLUMN utm_campaign VARCHAR(255) DEFAULT NULL")
    except Exception:
        pass  # orders table may not exist yet
    try:
        cur.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'cart_items' AND COLUMN_NAME = 'gift_message'
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE cart_items ADD COLUMN gift_message TEXT DEFAULT NULL")
    except Exception:
        pass  # cart_items table may not exist yet
    try:
        cur.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'catalogue' AND COLUMN_NAME = 'gift_personas'
        """)
        if not cur.fetchone():
            cur.execute("ALTER TABLE catalogue ADD COLUMN gift_personas TEXT DEFAULT NULL")
    except Exception:
        pass  # catalogue table may not exist yet
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


def init_article_queue_table():
    execute("""
        CREATE TABLE IF NOT EXISTS article_queue (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            subtitle VARCHAR(500),
            description TEXT,
            target_date DATE NOT NULL,
            status ENUM('pending','accepted','generating','ready','failed','rejected') DEFAULT 'pending',
            article_id INT NULL,
            generation_error TEXT NULL,
            batch_id VARCHAR(50) NULL,
            prompt_used TEXT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_status (status),
            INDEX idx_target_date (target_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

try:
    init_article_queue_table()
    # Add prompt_used column if table was created before this column existed
    try:
        execute("ALTER TABLE article_queue ADD COLUMN prompt_used TEXT NULL AFTER batch_id")
        print("article_queue: added prompt_used column")
    except Exception:
        pass  # Column already exists
except Exception as e:
    print(f"WARNING: Could not initialise article_queue table: {e}")


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


_UTM_PLATFORM_MAP = {
    'ig': 'instagram', 'fb': 'facebook', 'yt': 'youtube', 'x': 'twitter',
    'threads': 'threads', 'pinterest': 'pinterest', 'tiktok': 'tiktok',
    'linkedin': 'linkedin', 'bluesky': 'bluesky', 'reddit': 'reddit',
    'telegram': 'telegram', 'vimeo': 'vimeo', 'mastodon': 'mastodon',
    'vk': 'vk', 'tumblr': 'tumblr',
}

def make_short_url(article_id, platform_key):
    """Get or create a short URL for article+platform combo."""
    import hashlib as _hl, string as _str
    existing = query_one(
        "SELECT code FROM short_links WHERE article_id = %s AND platform = %s",
        (article_id, platform_key))
    if existing:
        return f"{SITE_URL}/a/{existing[0]}"
    slug_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    slug = slug_row[0] if slug_row else str(article_id)
    source = _UTM_PLATFORM_MAP.get(platform_key, platform_key)
    target = f"{SITE_URL}/articles/{slug}?utm_source={source}&utm_medium=social&utm_campaign={slug}"
    chars = _str.ascii_letters + _str.digits
    h = _hl.sha256(f"{article_id}:{platform_key}".encode()).hexdigest()
    code = ''.join(chars[int(h[i:i+2], 16) % len(chars)] for i in range(0, 12, 2))
    try:
        execute("INSERT INTO short_links (code, article_id, platform, target_url) VALUES (%s,%s,%s,%s)",
                (code, article_id, platform_key, target))
    except Exception:
        pass  # Duplicate key race condition — re-query
        existing = query_one("SELECT code FROM short_links WHERE article_id = %s AND platform = %s",
                             (article_id, platform_key))
        if existing:
            return f"{SITE_URL}/a/{existing[0]}"
    return f"{SITE_URL}/a/{code}"


def make_utm_url(article_slug, platform_key, article_id=None):
    """Build article URL with UTM tracking. Uses short URL when article_id is provided."""
    if article_id and SITE_URL:
        try:
            return make_short_url(article_id, platform_key)
        except Exception:
            pass
    source = _UTM_PLATFORM_MAP.get(platform_key, platform_key)
    base = f"{SITE_URL}/articles/{article_slug}"
    return f"{base}?utm_source={source}&utm_medium=social&utm_campaign={article_slug}"


def log_social_post(article_id, platform, content_type, post_id, permalink, caption):
    """Log a social media post with full metadata for ML training."""
    import re as _re
    from datetime import datetime as _dt_cls

    now = _dt_cls.utcnow()

    # Extract hashtags from caption
    hashtags = ' '.join(_re.findall(r'#\w+', caption or ''))

    # Get article metadata
    row = query_one("SELECT title, content, slug FROM articles WHERE id = %s", (article_id,))
    title = row[0] if row else ''
    content = row[1] if row else ''
    slug = row[2] if row else ''

    # Count words in article content (strip HTML)
    plain = _re.sub(r'<[^>]+>', ' ', content or '')
    word_count = len(plain.split())

    # Count images for carousel posts
    image_count = 0
    if content_type in ('carousel', 'car', 'cinemagraph', 'cine'):
        imgs = query_all("SELECT COUNT(*) FROM site_settings WHERE `key` LIKE %s AND value LIKE '%%http%%'",
                         (f'carousel_image_%_{article_id}',))
        image_count = imgs[0][0] if imgs else 0

    execute("""INSERT INTO posting_log
        (article_id, platform, content_type, post_id, permalink, caption, hashtags,
         posted_at, posted_day_of_week, posted_hour, posted_is_weekend,
         article_title, article_word_count, article_slug, image_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (article_id, platform, content_type, post_id, permalink, caption, hashtags,
         now, now.weekday(), now.hour, now.weekday() >= 5,
         title, word_count, slug, image_count))


@app.context_processor
def inject_currency_helpers():
    """Make currency helpers available in all templates."""
    currency = get_active_currency()
    unrecognized_count = 0
    if is_admin():
        try:
            row = query_one("SELECT COUNT(*) FROM unrecognized_stamps WHERE reviewed = 0")
            unrecognized_count = int(row[0]) if row and row[0] else 0
        except Exception:
            unrecognized_count = 0
    return {
        "active_currency": currency,
        "currency_symbol": CURRENCY_SYMBOLS.get(currency, currency),
        "currency_symbols": CURRENCY_SYMBOLS,
        "convert_catalogue_price": convert_catalogue_price,
        "catalogue_img_url": catalogue_img_url,
        "is_admin": is_admin(),
        "contact_email": os.getenv("CONTACT_TO_EMAIL", "thalesjacobi@gmail.com"),
        "site_url": SITE_URL,
        "ga_measurement_id": GA_MEASUREMENT_ID,
        "unrecognized_count": unrecognized_count,
    }


# ------------------------------------------------------------
# ENGAGEMENT FETCHERS
# ------------------------------------------------------------

def _fetch_ig_engagement(article_id):
    """Fetch Instagram engagement metrics for all post types."""
    import requests as _req
    results = []
    for content_type, prefix in [('carousel', 'ig_car'), ('cinemagraph', 'ig_cine')]:
        post_id = get_setting(f"instagram_media_id_{article_id}") if content_type == 'carousel' else get_setting(f"ig_cine_post_id_{article_id}")
        if not post_id:
            continue
        try:
            resp = _req.get(f"https://graph.instagram.com/{post_id}",
                params={"fields": "like_count,comments_count,media_type", "access_token": IG_ACCESS_TOKEN}, timeout=15)
            data = resp.json()
            if 'error' not in data:
                results.append({
                    'platform': 'ig', 'content_type': content_type,
                    'post_id': post_id,
                    'likes': data.get('like_count', 0),
                    'comments': data.get('comments_count', 0),
                })
        except Exception as e:
            print(f"IG engagement fetch error: {e}")
    # Also check NV posts
    rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s", (f"ig_narrated_post_id_{article_id}_%",))
    for row in (rows or []):
        post_id = row[1]
        if not post_id:
            continue
        try:
            resp = _req.get(f"https://graph.instagram.com/{post_id}",
                params={"fields": "like_count,comments_count,media_type", "access_token": IG_ACCESS_TOKEN}, timeout=15)
            data = resp.json()
            if 'error' not in data:
                results.append({
                    'platform': 'ig', 'content_type': 'narrated_video',
                    'post_id': post_id,
                    'likes': data.get('like_count', 0),
                    'comments': data.get('comments_count', 0),
                })
        except Exception as e:
            print(f"IG NV engagement fetch error: {e}")
    return results


def _fetch_fb_engagement(article_id):
    """Fetch Facebook engagement metrics."""
    import requests as _req
    results = []
    for content_type, key_prefix in [('carousel', 'fb_car'), ('narrated_video', 'fb_narrated')]:
        if content_type == 'carousel':
            post_id = get_setting(f"fb_car_post_id_{article_id}")
        else:
            rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s", (f"fb_narrated_post_id_{article_id}_%",))
            for row in (rows or []):
                post_id = row[1]
                if not post_id:
                    continue
                try:
                    resp = _req.get(f"https://graph.facebook.com/v21.0/{post_id}",
                        params={"fields": "likes.summary(true),shares,comments.summary(true)",
                                "access_token": FB_PAGE_ACCESS_TOKEN}, timeout=15)
                    data = resp.json()
                    if 'error' not in data:
                        results.append({
                            'platform': 'fb', 'content_type': content_type,
                            'post_id': post_id,
                            'likes': data.get('likes', {}).get('summary', {}).get('total_count', 0),
                            'shares': data.get('shares', {}).get('count', 0),
                            'comments': data.get('comments', {}).get('summary', {}).get('total_count', 0),
                        })
                except Exception as e:
                    print(f"FB engagement fetch error: {e}")
            continue
        if not post_id:
            continue
        try:
            resp = _req.get(f"https://graph.facebook.com/v21.0/{post_id}",
                params={"fields": "likes.summary(true),shares,comments.summary(true)",
                        "access_token": FB_PAGE_ACCESS_TOKEN}, timeout=15)
            data = resp.json()
            if 'error' not in data:
                results.append({
                    'platform': 'fb', 'content_type': content_type,
                    'post_id': post_id,
                    'likes': data.get('likes', {}).get('summary', {}).get('total_count', 0),
                    'shares': data.get('shares', {}).get('count', 0),
                    'comments': data.get('comments', {}).get('summary', {}).get('total_count', 0),
                })
        except Exception as e:
            print(f"FB engagement fetch error: {e}")
    return results


def _fetch_x_engagement(article_id):
    """Fetch X/Twitter engagement metrics using OAuth 1.0a."""
    results = []
    if not X_CONFIGURED:
        return results
    try:
        from requests_oauthlib import OAuth1Session
        oauth = OAuth1Session(X_API_KEY, client_secret=X_API_SECRET,
                              resource_owner_key=X_ACCESS_TOKEN,
                              resource_owner_secret=X_ACCESS_TOKEN_SECRET)
        for content_type, key_pattern in [('carousel', f'x_car_tweet_id_{article_id}'),
                                           ('narrated_video', f'x_narrated_tweet_id_{article_id}_%')]:
            if '%' in key_pattern:
                rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s", (key_pattern,))
            else:
                val = get_setting(key_pattern)
                rows = [(None, val)] if val else []
            for row in (rows or []):
                tweet_id = row[1]
                if not tweet_id:
                    continue
                resp = oauth.get(f"https://api.x.com/2/tweets/{tweet_id}",
                    params={"tweet.fields": "public_metrics"}, timeout=15)
                data = resp.json()
                metrics = data.get('data', {}).get('public_metrics', {})
                if metrics:
                    results.append({
                        'platform': 'x', 'content_type': content_type,
                        'post_id': tweet_id,
                        'likes': metrics.get('like_count', 0),
                        'views': metrics.get('impression_count', 0),
                        'shares': metrics.get('retweet_count', 0),
                        'comments': metrics.get('reply_count', 0),
                    })
    except Exception as e:
        print(f"X engagement fetch error: {e}")
    return results


def _fetch_threads_engagement(article_id):
    """Fetch Threads engagement metrics."""
    import requests as _req
    results = []
    if not THREADS_CONFIGURED:
        return results
    for content_type, key_pattern in [('carousel', f'threads_car_post_id_{article_id}'),
                                       ('narrated_video', f'threads_narrated_post_id_{article_id}_%')]:
        if '%' in key_pattern:
            rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s", (key_pattern,))
        else:
            val = get_setting(key_pattern)
            rows = [(None, val)] if val else []
        for row in (rows or []):
            post_id = row[1]
            if not post_id:
                continue
            try:
                resp = _req.get(f"{_THREADS_API_URL}/{post_id}",
                    params={"fields": "likes,replies,reposts,quotes,views",
                            "access_token": THREADS_ACCESS_TOKEN}, timeout=15)
                data = resp.json()
                if 'error' not in data:
                    results.append({
                        'platform': 'threads', 'content_type': content_type,
                        'post_id': post_id,
                        'likes': data.get('likes', 0),
                        'comments': data.get('replies', 0),
                        'shares': data.get('reposts', 0) + data.get('quotes', 0),
                        'views': data.get('views', 0),
                    })
            except Exception as e:
                print(f"Threads engagement fetch error: {e}")
    return results


def _fetch_youtube_engagement(article_id):
    """Fetch YouTube engagement metrics."""
    import requests as _req
    results = []
    yt_refresh = get_setting("youtube_refresh_token")
    if not yt_refresh:
        return results
    try:
        token_resp = _req.post("https://oauth2.googleapis.com/token", data={
            "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
            "refresh_token": yt_refresh,
            "grant_type": "refresh_token",
        }, timeout=15)
        access_token = token_resp.json().get("access_token", "")
        if not access_token:
            return results
    except Exception:
        return results

    rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s",
                     (f"youtube_video_id_{article_id}_%",))
    for row in (rows or []):
        video_id = row[1]
        if not video_id:
            continue
        try:
            resp = _req.get("https://www.googleapis.com/youtube/v3/videos",
                params={"part": "statistics", "id": video_id,
                        "access_token": access_token}, timeout=15)
            data = resp.json()
            items = data.get('items', [])
            if items:
                stats = items[0].get('statistics', {})
                results.append({
                    'platform': 'yt', 'content_type': 'narrated_video',
                    'post_id': video_id,
                    'likes': int(stats.get('likeCount', 0)),
                    'views': int(stats.get('viewCount', 0)),
                    'comments': int(stats.get('commentCount', 0)),
                })
        except Exception as e:
            print(f"YouTube engagement fetch error: {e}")
    return results


def _fetch_pinterest_engagement(article_id):
    """Fetch Pinterest engagement metrics."""
    import requests as _req
    results = []
    if not PINTEREST_CONFIGURED:
        return results
    for content_type, key_pattern in [('carousel', f'pinterest_car_pin_id_{article_id}'),
                                       ('narrated_video', f'pinterest_narrated_pin_id_{article_id}_%')]:
        if '%' in key_pattern:
            rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s", (key_pattern,))
        else:
            val = get_setting(key_pattern)
            rows = [(None, val)] if val else []
        for row in (rows or []):
            pin_id = row[1]
            if not pin_id:
                continue
            try:
                resp = _req.get(f"https://api.pinterest.com/v5/pins/{pin_id}",
                    headers={"Authorization": f"Bearer {PINTEREST_ACCESS_TOKEN}"}, timeout=15)
                data = resp.json()
                if 'code' not in data:
                    results.append({
                        'platform': 'pinterest', 'content_type': content_type,
                        'post_id': pin_id,
                        'saves': data.get('pin_metrics', {}).get('save', 0),
                        'comments': data.get('pin_metrics', {}).get('comment_count', 0) if 'pin_metrics' in data else 0,
                    })
            except Exception as e:
                print(f"Pinterest engagement fetch error: {e}")
    return results


def _fetch_bluesky_engagement(article_id):
    """Fetch Bluesky engagement metrics via AT Protocol."""
    import requests as _req
    results = []
    if not BLUESKY_CONFIGURED:
        return results
    try:
        auth_resp = _req.post("https://bsky.social/xrpc/com.atproto.server.createSession",
            json={"identifier": BLUESKY_HANDLE, "password": BLUESKY_APP_PASSWORD}, timeout=15)
        auth_data = auth_resp.json()
        access_jwt = auth_data.get("accessJwt", "")
        if not access_jwt:
            return results
    except Exception:
        return results

    for content_type, key_pattern in [('carousel', f'bluesky_car_post_uri_{article_id}'),
                                       ('narrated_video', f'bluesky_narrated_post_uri_{article_id}_%')]:
        if '%' in key_pattern:
            rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s", (key_pattern,))
        else:
            val = get_setting(key_pattern)
            rows = [(None, val)] if val else []
        for row in (rows or []):
            post_uri = row[1]
            if not post_uri:
                continue
            try:
                resp = _req.get("https://bsky.social/xrpc/app.bsky.feed.getPostThread",
                    params={"uri": post_uri, "depth": 0},
                    headers={"Authorization": f"Bearer {access_jwt}"}, timeout=15)
                data = resp.json()
                post = data.get('thread', {}).get('post', {})
                if post:
                    results.append({
                        'platform': 'bluesky', 'content_type': content_type,
                        'post_id': post_uri,
                        'likes': post.get('likeCount', 0),
                        'shares': post.get('repostCount', 0),
                        'comments': post.get('replyCount', 0),
                    })
            except Exception as e:
                print(f"Bluesky engagement fetch error: {e}")
    return results


# ── Comment fetchers for ML/NLP training data ──────────────────────

def _fetch_ig_comments(article_id):
    """Fetch comments from Instagram posts for this article."""
    import requests as _req
    for content_type in ('car', 'cine'):
        post_id = get_setting(f"ig_{content_type}_post_id_{article_id}")
        if not post_id:
            continue
        try:
            resp = _req.get(f"https://graph.instagram.com/v21.0/{post_id}/comments",
                           params={"fields": "id,text,username,timestamp", "access_token": IG_ACCESS_TOKEN},
                           timeout=15)
            data = resp.json()
            for c in data.get('data', []):
                execute("""INSERT IGNORE INTO post_comments
                    (article_id, platform, content_type, post_id, comment_id, comment_text,
                     comment_author, comment_timestamp, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (article_id, 'ig', content_type, post_id, c.get('id',''),
                     c.get('text',''), c.get('username',''), c.get('timestamp','')))
        except Exception:
            pass
    # Also narrated video posts
    for key_prefix in ('narrated_ig_media_id_',):
        rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s",
                         (f"{key_prefix}{article_id}%",))
        for row in (rows or []):
            post_id = row[1]
            if not post_id:
                continue
            try:
                resp = _req.get(f"https://graph.instagram.com/v21.0/{post_id}/comments",
                               params={"fields": "id,text,username,timestamp", "access_token": IG_ACCESS_TOKEN},
                               timeout=15)
                data = resp.json()
                for c in data.get('data', []):
                    execute("""INSERT IGNORE INTO post_comments
                        (article_id, platform, content_type, post_id, comment_id, comment_text,
                         comment_author, comment_timestamp, fetched_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                        (article_id, 'ig', 'narrated', post_id, c.get('id',''),
                         c.get('text',''), c.get('username',''), c.get('timestamp','')))
            except Exception:
                pass


def _fetch_fb_comments(article_id):
    """Fetch comments from Facebook posts for this article."""
    import requests as _req
    for content_type in ('car', 'cine'):
        post_id = get_setting(f"fb_{content_type}_media_id_{article_id}")
        if not post_id:
            continue
        try:
            resp = _req.get(f"https://graph.facebook.com/v21.0/{post_id}/comments",
                           params={"fields": "id,message,from,created_time", "access_token": FB_PAGE_ACCESS_TOKEN},
                           timeout=15)
            data = resp.json()
            for c in data.get('data', []):
                author = (c.get('from') or {}).get('name', '')
                execute("""INSERT IGNORE INTO post_comments
                    (article_id, platform, content_type, post_id, comment_id, comment_text,
                     comment_author, comment_timestamp, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (article_id, 'fb', content_type, post_id, c.get('id',''),
                     c.get('message',''), author, c.get('created_time','')))
        except Exception:
            pass
    # Narrated video posts
    rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s",
                     (f"narrated_fb_media_id_{article_id}%",))
    for row in (rows or []):
        post_id = row[1]
        if not post_id:
            continue
        try:
            resp = _req.get(f"https://graph.facebook.com/v21.0/{post_id}/comments",
                           params={"fields": "id,message,from,created_time", "access_token": FB_PAGE_ACCESS_TOKEN},
                           timeout=15)
            data = resp.json()
            for c in data.get('data', []):
                author = (c.get('from') or {}).get('name', '')
                execute("""INSERT IGNORE INTO post_comments
                    (article_id, platform, content_type, post_id, comment_id, comment_text,
                     comment_author, comment_timestamp, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (article_id, 'fb', 'narrated', post_id, c.get('id',''),
                     c.get('message',''), author, c.get('created_time','')))
        except Exception:
            pass


def _fetch_threads_comments(article_id):
    """Fetch comments (replies) from Threads posts for this article."""
    import requests as _req
    for content_type in ('car', 'cine'):
        post_id = get_setting(f"threads_{content_type}_post_id_{article_id}")
        if not post_id:
            continue
        try:
            resp = _req.get(f"{_THREADS_API_URL}/{post_id}/replies",
                           params={"fields": "id,text,username,timestamp", "access_token": THREADS_ACCESS_TOKEN},
                           timeout=15)
            data = resp.json()
            for c in data.get('data', []):
                execute("""INSERT IGNORE INTO post_comments
                    (article_id, platform, content_type, post_id, comment_id, comment_text,
                     comment_author, comment_timestamp, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (article_id, 'threads', content_type, post_id, c.get('id',''),
                     c.get('text',''), c.get('username',''), c.get('timestamp','')))
        except Exception:
            pass
    # Narrated video posts
    rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s",
                     (f"threads_narrated_post_id_{article_id}%",))
    for row in (rows or []):
        post_id = row[1]
        if not post_id:
            continue
        try:
            resp = _req.get(f"{_THREADS_API_URL}/{post_id}/replies",
                           params={"fields": "id,text,username,timestamp", "access_token": THREADS_ACCESS_TOKEN},
                           timeout=15)
            data = resp.json()
            for c in data.get('data', []):
                execute("""INSERT IGNORE INTO post_comments
                    (article_id, platform, content_type, post_id, comment_id, comment_text,
                     comment_author, comment_timestamp, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (article_id, 'threads', 'narrated', post_id, c.get('id',''),
                     c.get('text',''), c.get('username',''), c.get('timestamp','')))
        except Exception:
            pass


def _fetch_bluesky_comments(article_id):
    """Fetch replies from Bluesky posts for this article."""
    import requests as _req
    for content_type in ('car', 'cine'):
        rkey = get_setting(f"bluesky_{content_type}_post_id_{article_id}")
        if not rkey:
            continue
        uri = f"at://{BLUESKY_HANDLE}/app.bsky.feed.post/{rkey}"
        try:
            resp = _req.get("https://bsky.social/xrpc/app.bsky.feed.getPostThread",
                           params={"uri": uri, "depth": 1}, timeout=15)
            data = resp.json()
            thread = data.get('thread', {})
            for reply in thread.get('replies', []):
                post = reply.get('post', {})
                record = post.get('record', {})
                author = post.get('author', {})
                comment_id = post.get('uri', '')
                execute("""INSERT IGNORE INTO post_comments
                    (article_id, platform, content_type, post_id, comment_id, comment_text,
                     comment_author, comment_timestamp, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (article_id, 'bluesky', content_type, rkey, comment_id,
                     record.get('text',''), author.get('handle',''), record.get('createdAt','')))
        except Exception:
            pass
    # Narrated video posts
    rows = query_all("SELECT `key`, value FROM site_settings WHERE `key` LIKE %s",
                     (f"bluesky_narrated_post_id_{article_id}%",))
    for row in (rows or []):
        rkey = row[1]
        if not rkey:
            continue
        uri = f"at://{BLUESKY_HANDLE}/app.bsky.feed.post/{rkey}"
        try:
            resp = _req.get("https://bsky.social/xrpc/app.bsky.feed.getPostThread",
                           params={"uri": uri, "depth": 1}, timeout=15)
            data = resp.json()
            thread = data.get('thread', {})
            for reply in thread.get('replies', []):
                post = reply.get('post', {})
                record = post.get('record', {})
                author = post.get('author', {})
                comment_id = post.get('uri', '')
                execute("""INSERT IGNORE INTO post_comments
                    (article_id, platform, content_type, post_id, comment_id, comment_text,
                     comment_author, comment_timestamp, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (article_id, 'bluesky', 'narrated', rkey, comment_id,
                     record.get('text',''), author.get('handle',''), record.get('createdAt','')))
        except Exception:
            pass


def fetch_all_engagement(article_id):
    """Fetch engagement metrics from all configured platforms and store snapshots.

    Each platform fetch is isolated so one failure (e.g. expired YouTube token)
    cannot break the whole refresh. Per-platform errors are collected and
    returned in the special key '_errors' on the result list.
    """
    from datetime import datetime as _dt_now
    all_results = []
    errors = []  # list of {'platform': str, 'error': str}

    def _safe_fetch(platform_label, fetch_fn):
        try:
            res = fetch_fn(article_id)
            if res:
                all_results.extend(res)
        except Exception as _e:
            import traceback as _tb
            print(f"[Engagement] {platform_label} fetch failed: {_e}\n{_tb.format_exc()}", flush=True)
            errors.append({'platform': platform_label, 'error': str(_e)})

    if IG_USER_ID and IG_ACCESS_TOKEN:
        _safe_fetch('Instagram', _fetch_ig_engagement)
    if FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN:
        _safe_fetch('Facebook', _fetch_fb_engagement)
    if X_CONFIGURED:
        _safe_fetch('X', _fetch_x_engagement)
    if THREADS_CONFIGURED:
        _safe_fetch('Threads', _fetch_threads_engagement)
    if get_setting("youtube_refresh_token"):
        _safe_fetch('YouTube', _fetch_youtube_engagement)
    if PINTEREST_CONFIGURED:
        _safe_fetch('Pinterest', _fetch_pinterest_engagement)
    if BLUESKY_CONFIGURED:
        _safe_fetch('Bluesky', _fetch_bluesky_engagement)

    # Stash per-platform errors via function attribute so caller can read them
    fetch_all_engagement._last_errors = errors

    # Store snapshots
    now = _dt_now.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for r in all_results:
        execute("""INSERT INTO article_engagement
            (article_id, platform, content_type, post_id, likes, views, shares, comments, saves, clicks, impressions, reach, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (article_id, r.get('platform',''), r.get('content_type',''), r.get('post_id',''),
             r.get('likes',0), r.get('views',0), r.get('shares',0), r.get('comments',0),
             r.get('saves',0), r.get('clicks',0), r.get('impressions',0), r.get('reach',0), now))

    # Also store time-series snapshot for growth curve analysis
    for r in all_results:
        post_id = r.get('post_id', '')
        if not post_id:
            continue
        try:
            # Get posting time to calculate hours_since_post
            posted_row = query_one(
                "SELECT posted_at FROM posting_log WHERE post_id = %s AND platform = %s ORDER BY posted_at DESC LIMIT 1",
                (post_id, r.get('platform', '')))
            hours_since = 0
            if posted_row and posted_row[0]:
                from datetime import datetime as _dt_cls2
                delta = _dt_cls2.utcnow() - posted_row[0]
                hours_since = int(delta.total_seconds() / 3600)
            execute("""INSERT INTO post_metrics_history
                (article_id, platform, content_type, post_id, likes, views, shares, comments_count,
                 saves, clicks, impressions, reach, fetched_at, hours_since_post)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (article_id, r.get('platform',''), r.get('content_type',''), post_id,
                 r.get('likes',0), r.get('views',0), r.get('shares',0), r.get('comments',0),
                 r.get('saves',0), r.get('clicks',0), r.get('impressions',0), r.get('reach',0),
                 now, hours_since))
        except Exception:
            pass  # Don't fail engagement fetch if history insert fails

    return all_results


def fetch_follower_counts():
    """Snapshot current follower counts across all platforms."""
    import requests as _req
    from datetime import date as _date_cls, datetime as _dt_cls
    today = _date_cls.today()
    now = _dt_cls.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    platforms = []

    # Instagram
    if IG_USER_ID and IG_ACCESS_TOKEN:
        try:
            r = _req.get(f"https://graph.instagram.com/v21.0/{IG_USER_ID}",
                        params={"fields": "followers_count,follows_count,media_count", "access_token": IG_ACCESS_TOKEN},
                        timeout=15).json()
            platforms.append(('ig', r.get('followers_count',0), r.get('follows_count',0), r.get('media_count',0)))
        except Exception:
            pass

    # Facebook
    if FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN:
        try:
            r = _req.get(f"https://graph.facebook.com/v21.0/{FB_PAGE_ID}",
                        params={"fields": "fan_count,followers_count", "access_token": FB_PAGE_ACCESS_TOKEN},
                        timeout=15).json()
            platforms.append(('fb', r.get('followers_count', r.get('fan_count',0)), 0, 0))
        except Exception:
            pass

    # YouTube
    yt_token = get_setting("youtube_refresh_token")
    if yt_token:
        try:
            # Would need to refresh access token first, then call channels.list
            pass  # TODO: implement when YouTube OAuth is fully set up
        except Exception:
            pass

    # X — requires user lookup endpoint
    if X_CONFIGURED:
        try:
            # OAuth1 signed request to GET /2/users/me?user.fields=public_metrics
            pass  # X free tier may not support this
        except Exception:
            pass

    # Threads
    if THREADS_CONFIGURED:
        try:
            r = _req.get(f"https://graph.threads.net/v1.0/{THREADS_USER_ID}",
                        params={"fields": "followers_count", "access_token": THREADS_ACCESS_TOKEN},
                        timeout=15).json()
            platforms.append(('threads', r.get('followers_count',0), 0, 0))
        except Exception:
            pass

    # Bluesky
    if BLUESKY_CONFIGURED:
        try:
            r = _req.get("https://bsky.social/xrpc/app.bsky.actor.getProfile",
                        params={"actor": BLUESKY_HANDLE}, timeout=15).json()
            platforms.append(('bluesky', r.get('followersCount',0), r.get('followsCount',0), r.get('postsCount',0)))
        except Exception:
            pass

    for plat, followers, following, posts in platforms:
        execute("""INSERT INTO follower_snapshots
            (platform, follower_count, following_count, post_count, snapshot_date, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE follower_count=%s, following_count=%s, post_count=%s, fetched_at=%s""",
            (plat, followers, following, posts, today, now, followers, following, posts, now))

    return platforms


# ------------------------------------------------------------
# PAGES
# ------------------------------------------------------------
@app.route("/")
def home():
    catalogue_items = query_all(
        "SELECT id, title, description, price, image_url, status, category, slug "
        "FROM catalogue WHERE is_public = 1 ORDER BY id DESC"
    )
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


@app.route("/a/<code>")
def short_link_redirect(code):
    """301 redirect from short URL to full article with UTM params."""
    row = query_one("SELECT target_url FROM short_links WHERE code = %s", (code,))
    if not row:
        return render_template("404.html"), 404
    execute("UPDATE short_links SET click_count = click_count + 1 WHERE code = %s", (code,))
    from urllib.parse import urlparse, parse_qs
    params = parse_qs(urlparse(row[0]).query)
    if params.get('utm_source'):
        session['utm_source'] = params['utm_source'][0]
        session['utm_campaign'] = params.get('utm_campaign', [''])[0]
        session['utm_medium'] = params.get('utm_medium', [''])[0]
    return redirect(row[0], 301)


@app.route("/articles/<slug>")
def article_view(slug):
    # Store UTM params in session for purchase attribution
    if request.args.get('utm_source'):
        session['utm_source'] = request.args.get('utm_source', '')
        session['utm_campaign'] = request.args.get('utm_campaign', '')
        session['utm_medium'] = request.args.get('utm_medium', '')

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
    catalogue_items = query_all(
        "SELECT id, title, description, price, image_url, status, category, slug "
        "FROM catalogue WHERE is_public = 1 ORDER BY id DESC"
    )
    # Fetch gallery images for all items in one query
    galleries = {}
    if catalogue_items:
        ids = tuple(item[0] for item in catalogue_items)
        placeholders = ",".join(["%s"] * len(ids))
        gallery_rows = query_all(
            f"SELECT catalogue_id, image_url FROM catalogue_images "
            f"WHERE catalogue_id IN ({placeholders}) ORDER BY sort_order, id",
            ids,
        )
        for cid, url in gallery_rows:
            galleries.setdefault(cid, []).append(url)
    currency = get_active_currency()
    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        log_activity(current_user.id, "page_view", "Viewed catalogue", {"page": "/catalogue"})
    return render_template("catalogue.html",
                           catalogue_items=catalogue_items,
                           galleries=galleries,
                           user_currency=currency)


@app.route("/catalogue/<category>")
def catalogue_category(category):
    """Show all public items in a given category."""
    catalogue_items = query_all(
        "SELECT id, title, description, price, image_url, status, category, slug "
        "FROM catalogue WHERE is_public = 1 AND category = %s ORDER BY id DESC",
        (category,),
    )
    galleries = {}
    if catalogue_items:
        ids = tuple(item[0] for item in catalogue_items)
        placeholders = ",".join(["%s"] * len(ids))
        gallery_rows = query_all(
            f"SELECT catalogue_id, image_url FROM catalogue_images "
            f"WHERE catalogue_id IN ({placeholders}) ORDER BY sort_order, id",
            ids,
        )
        for cid, url in gallery_rows:
            galleries.setdefault(cid, []).append(url)
    currency = get_active_currency()
    return render_template("catalogue.html",
                           catalogue_items=catalogue_items,
                           galleries=galleries,
                           user_currency=currency,
                           active_category=category)


@app.route("/catalogue/<category>/<slug>")
def catalogue_item_detail(category, slug):
    """SEO-friendly detail page for a single catalogue item."""
    # Admins can preview private items; public visitors get 404
    is_admin_user = hasattr(current_user, "is_authenticated") and current_user.is_authenticated and is_admin()
    visibility_clause = "" if is_admin_user else " AND is_public = 1"
    row = query_one(
        "SELECT id, title, description, price, image_url, status, category, slug, is_public, gift_personas "
        "FROM catalogue WHERE category = %s AND slug = %s" + visibility_clause,
        (category, slug),
    )
    if not row:
        if _template_exists("404.html"):
            return render_template("404.html"), 404
        return "Not found", 404
    personas_raw = row[9] or ""
    item = {
        "id": row[0], "title": row[1], "description": row[2],
        "price": float(row[3]) if row[3] is not None else 0.0,
        "image_url": row[4],
        "status": row[5] or "available",
        "category": row[6] or "",
        "slug": row[7] or "",
        "is_public": bool(row[8]) if len(row) > 8 else True,
        "gift_personas": [p.strip() for p in personas_raw.split(",") if p.strip()],
    }
    gallery = query_all(
        "SELECT image_url FROM catalogue_images "
        "WHERE catalogue_id = %s ORDER BY sort_order, id",
        (item["id"],),
    )
    gallery_urls = [g[0] for g in gallery]
    specs = _fetch_catalogue_specs(item["id"])

    # Related items: same category, public, not this one, not sold, up to 4
    related = []
    if item.get("category"):
        related_rows = query_all(
            "SELECT id, title, price, image_url, status, category, slug "
            "FROM catalogue WHERE category = %s AND id != %s "
            "AND is_public = 1 ORDER BY id DESC LIMIT 4",
            (item["category"], item["id"]),
        )
        related = [
            {
                "id": r[0], "title": r[1],
                "price": float(r[2]) if r[2] is not None else 0.0,
                "image_url": r[3],
                "status": r[4] or "available",
                "category": r[5] or "",
                "slug": r[6] or "",
            }
            for r in related_rows
        ]

    # Delivery ETA: ~3 business days from now
    eta = _next_business_day(days=3)
    delivery_eta = eta.strftime("%A, %d %B").replace(" 0", " ")

    # JSON-LD Product schema for search engines
    product_url = (SITE_URL or "") + request.path
    json_ld = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": item["title"],
        "description": (item["description"] or item["title"])[:500],
        "sku": f"stamp-{item['id']}",
        "category": item["category"] or "Framed stamps",
        "image": [catalogue_img_url(item["image_url"])] + [catalogue_img_url(u) for u in gallery_urls] if item["image_url"] else [catalogue_img_url(u) for u in gallery_urls],
        "brand": {"@type": "Brand", "name": "Obelisk Stamps"},
        "offers": {
            "@type": "Offer",
            "url": product_url,
            "priceCurrency": "GBP",
            "price": f"{item['price']:.2f}",
            "availability": "https://schema.org/InStock" if item["status"] != "sold" else "https://schema.org/SoldOut",
            "itemCondition": "https://schema.org/NewCondition",
        },
    }

    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        log_activity(current_user.id, "page_view",
                     f"Viewed catalogue item: {item['title']}",
                     {"page": request.path, "catalogue_id": item["id"]})
    return render_template("catalogue_item.html",
                           item=item,
                           gallery=gallery_urls,
                           specs=specs,
                           related=related,
                           delivery_eta=delivery_eta,
                           json_ld=json.dumps(json_ld))


def _template_exists(name: str) -> bool:
    try:
        app.jinja_env.get_template(name)
        return True
    except Exception:
        return False


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

    # If the user is on a product page, include that product's details so the
    # assistant can answer specific questions about it ("Is this a good gift?",
    # "How big is it?", "What stamps are in it?") and help the buyer decide.
    product_block = ""
    product_context = data.get("product_context") or {}
    if isinstance(product_context, dict) and product_context.get("title"):
        lines = ["\n\nThe user is currently viewing this product on our site:"]
        if product_context.get("title"):
            lines.append(f"- Title: {product_context['title']}")
        if product_context.get("category"):
            lines.append(f"- Category: {product_context['category']}")
        if product_context.get("price_gbp") is not None:
            try:
                lines.append(f"- Price: GBP {float(product_context['price_gbp']):.2f}")
            except (TypeError, ValueError):
                pass
        if product_context.get("status"):
            lines.append(f"- Status: {product_context['status']}")
        if product_context.get("description"):
            desc = str(product_context['description'])[:600]
            lines.append(f"- Description: {desc}")
        specs_list = product_context.get("specs") or []
        if isinstance(specs_list, list) and specs_list:
            lines.append("- Specifications:")
            for s in specs_list[:20]:
                if isinstance(s, dict) and s.get("label") and s.get("value"):
                    lines.append(f"    - {s['label']}: {s['value']}")
        if product_context.get("url"):
            lines.append(f"- URL: {product_context['url']}")
        lines.append(
            "\nAnswer questions about this specific item helpfully and enthusiastically. "
            "If the user asks whether to buy it, help them decide by highlighting what's "
            "special about it, who it suits (gift, collector, decor), and the practical "
            "details (shipping, returns). Do not invent specs — only use what's listed above. "
            "If a detail isn't listed, say so honestly and suggest they contact us."
        )
        product_block = "\n".join(lines)

    system_message = CHATBOT_SYSTEM_PROMPT + product_block + context_block

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


@app.route("/ml-warmup", methods=["GET"])
def ml_warmup():
    """
    Fires a lightweight GET to the ML API in a background thread so the
    Cloud Run container warms up before the user clicks Identify.
    Returns immediately — callers poll /ml-ready for readiness.
    """
    import threading

    def _ping():
        try:
            http_requests.get(ML_API_URL, timeout=180)
        except Exception:
            pass

    t = threading.Thread(target=_ping, daemon=True)
    t.start()
    return jsonify({"started": True})


@app.route("/ml-ready", methods=["GET"])
def ml_ready():
    """
    Quick probe to check whether the ML API is reachable right now.
    Returns {ready: true/false} within a short timeout (5 s).
    """
    try:
        http_requests.get(ML_API_URL, timeout=5)
        return jsonify({"ready": True})
    except Exception:
        return jsonify({"ready": False})


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
            data = {"top_k": 3, "confidence": 0.1}

            response = http_requests.post(
                f"{ML_API_URL}/predict",
                files=files,
                data=data,
                timeout=180,
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


@app.route("/stamp-quest/capture", methods=["POST"])
def capture_unrecognized_stamp():
    """
    Save a stamp crop the ML API could not recognise with high confidence,
    so an admin can later enrich it (title, country, year, price, etc.) and
    promote it into postbeeld_stamps. The crop image is uploaded to GCS when
    available and falls back to a LONGBLOB in the DB.
    """
    import base64, uuid, json as _json

    data = request.get_json(silent=True) or {}
    img_b64 = data.get("image_base64")
    if not img_b64:
        return jsonify({"error": "image_base64 required"}), 400

    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception:
        return jsonify({"error": "invalid base64 image"}), 400

    if not img_bytes:
        return jsonify({"error": "empty image"}), 400

    # Upload to GCS under stamps/unrecognized/<uuid>.jpg (same top-level
    # prefix used elsewhere for stamp imagery). Fall back to blob storage.
    fname = f"{uuid.uuid4().hex}.jpg"
    gcs_url = upload_bytes_to_gcs(
        img_bytes, f"stamps/unrecognized/{fname}", content_type="image/jpeg"
    )
    image_blob = None if gcs_url else img_bytes

    bbox = data.get("bbox")
    bbox_json = _json.dumps(bbox) if bbox else None

    user_id = current_user.id if getattr(current_user, "is_authenticated", False) else None
    client_ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "")[:64]

    # Coerce year to int when possible
    year_val = data.get("best_match_year")
    try:
        year_val = int(year_val) if year_val not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        year_val = None

    try:
        row_id = execute(
            """INSERT INTO unrecognized_stamps
               (image_url, image_blob, detection_confidence, bbox_json,
                best_match_similarity, best_match_title, best_match_country,
                best_match_year, user_id, client_ip)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                gcs_url,
                image_blob,
                data.get("detection_confidence"),
                bbox_json,
                data.get("best_match_similarity"),
                (data.get("best_match_title") or None),
                (data.get("best_match_country") or None),
                year_val,
                user_id,
                client_ip,
            ),
        )
        return jsonify({"success": True, "id": row_id})
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
        """SELECT ci.id, ci.catalogue_id, ci.quantity, c.title, c.price, c.image_url, ci.gift_message
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
    item = query_one("SELECT id, title, status FROM catalogue WHERE id = %s", (catalogue_id,))
    if item and len(item) >= 3 and (item[2] or "available") == "sold":
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"success": False, "error": "This item has been sold."}), 400
        flash("This item has already been sold.", "warning")
        return redirect(request.referrer or url_for("catalogue"))
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

    gift_message = request.form.get("gift_message", "").strip() or None

    if existing:
        # Update quantity (and gift message if provided)
        if gift_message:
            execute(
                "UPDATE cart_items SET quantity = quantity + 1, gift_message = %s WHERE id = %s",
                (gift_message, existing[0]),
            )
        else:
            execute(
                "UPDATE cart_items SET quantity = quantity + 1 WHERE id = %s",
                (existing[0],),
            )
    else:
        # Insert new cart item
        execute(
            "INSERT INTO cart_items (user_id, catalogue_id, quantity, gift_message) VALUES (%s, %s, 1, %s)",
            (current_user.id, catalogue_id, gift_message),
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
                "utm_source": session.get("utm_source", ""),
                "utm_campaign": session.get("utm_campaign", ""),
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
            total_amount, currency, shipping_name, shipping_email, shipping_address,
            utm_source, utm_campaign)
           VALUES (%s, %s, %s, 'paid', %s, %s, %s, %s, %s, %s, %s)""",
        (
            user_id,
            session.id,
            session.payment_intent,
            total,
            order_currency,
            shipping.get("name"),
            session.customer_email,
            address_str,
            (session.metadata or {}).get("utm_source", "") or None,
            (session.metadata or {}).get("utm_campaign", "") or None,
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
        # Mark catalogue item as sold (each product is unique, 1 per listing)
        try:
            execute("UPDATE catalogue SET status = 'sold' WHERE id = %s", (catalogue_id,))
        except Exception as _e:
            # Status column may not yet be present in some envs — don't fail the order
            print(f"WARN: could not mark catalogue {catalogue_id} as sold: {_e}")

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
                           yt_connected=bool(get_setting("youtube_refresh_token")),
                           x_api_key_set=bool(X_API_KEY),
                           x_api_secret_set=bool(X_API_SECRET),
                           x_access_token_set=bool(X_ACCESS_TOKEN),
                           x_access_token_secret_set=bool(X_ACCESS_TOKEN_SECRET),
                           threads_user_id_set=bool(THREADS_USER_ID),
                           threads_token_set=bool(THREADS_ACCESS_TOKEN),
                           pinterest_token_set=bool(PINTEREST_ACCESS_TOKEN or get_setting("pinterest_access_token")),
                           pinterest_board_id_set=bool(PINTEREST_BOARD_ID),
                           pinterest_oauth_connected=bool(get_setting("pinterest_access_token")),
                           pinterest_client_id_set=bool(PINTEREST_CLIENT_ID),
                           tiktok_client_key_set=bool(TIKTOK_CLIENT_KEY),
                           tiktok_client_secret_set=bool(TIKTOK_CLIENT_SECRET),
                           tiktok_token_set=bool(TIKTOK_ACCESS_TOKEN),
                           linkedin_token_set=bool(LINKEDIN_ACCESS_TOKEN),
                           linkedin_org_id_set=bool(LINKEDIN_ORG_ID))


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
                           yt_configured=bool(get_setting("youtube_refresh_token")),
                           x_configured=X_CONFIGURED,
                           threads_configured=THREADS_CONFIGURED,
                           pinterest_configured=PINTEREST_CONFIGURED,
                           tiktok_configured=TIKTOK_CONFIGURED,
                           linkedin_configured=LINKEDIN_CONFIGURED,
                           bluesky_configured=BLUESKY_CONFIGURED,
                           reddit_configured=REDDIT_CONFIGURED,
                           telegram_configured=TELEGRAM_CONFIGURED,
                           vimeo_configured=VIMEO_CONFIGURED,
                           mastodon_configured=MASTODON_CONFIGURED,
                           vk_configured=VK_CONFIGURED,
                           tumblr_configured=TUMBLR_CONFIGURED)


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
    # Override with directly-stored captions if available
    cine_caption  = get_setting(f"ig_cine_caption_{article_id}") or cine_caption
    car_caption   = get_setting(f"ig_car_caption_{article_id}") or car_caption
    # Load platform carousel captions
    fb_car_caption        = get_setting(f"fb_car_caption_{article_id}") or ""
    fb_cine_caption       = get_setting(f"fb_cine_caption_{article_id}") or ""
    x_car_caption         = get_setting(f"x_car_caption_{article_id}") or ""
    threads_car_caption   = get_setting(f"threads_car_caption_{article_id}") or ""
    pinterest_car_caption = get_setting(f"pinterest_car_caption_{article_id}") or ""
    linkedin_car_caption  = get_setting(f"linkedin_car_caption_{article_id}") or ""
    bluesky_car_caption   = get_setting(f"bluesky_car_caption_{article_id}") or ""
    reddit_car_caption    = get_setting(f"reddit_car_caption_{article_id}") or ""
    telegram_car_caption  = get_setting(f"telegram_car_caption_{article_id}") or ""
    mastodon_car_caption  = get_setting(f"mastodon_car_caption_{article_id}") or ""
    vk_car_caption        = get_setting(f"vk_car_caption_{article_id}") or ""
    tumblr_car_caption    = get_setting(f"tumblr_car_caption_{article_id}") or ""

    return render_template("article_edit.html", article=article,
                           carousel_style=article_carousel_style,
                           cinemagraph_prompt=get_setting('cinemagraph_prompt', _CINE_DEFAULT_PROMPT),
                           cinemagraph_article_prompt=get_setting(f'cinemagraph_prompt_{article_id}', ''),
                           ig_caption_prompt=get_setting('ig_caption_prompt', _IG_CAPTION_DEFAULT_PROMPT),
                           ig_article_caption_prompt=get_setting(f'ig_caption_prompt_{article_id}', ''),
                           ig_configured=bool(IG_USER_ID and IG_ACCESS_TOKEN),
                           fb_configured=bool(FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN),
                           yt_configured=bool(get_setting("youtube_refresh_token")),
                           x_configured=X_CONFIGURED,
                           threads_configured=THREADS_CONFIGURED,
                           pinterest_configured=PINTEREST_CONFIGURED,
                           tiktok_configured=TIKTOK_CONFIGURED,
                           linkedin_configured=LINKEDIN_CONFIGURED,
                           bluesky_configured=BLUESKY_CONFIGURED,
                           reddit_configured=REDDIT_CONFIGURED,
                           telegram_configured=TELEGRAM_CONFIGURED,
                           vimeo_configured=VIMEO_CONFIGURED,
                           mastodon_configured=MASTODON_CONFIGURED,
                           vk_configured=VK_CONFIGURED,
                           tumblr_configured=TUMBLR_CONFIGURED,
                           ig_cine_caption=cine_caption,
                           ig_car_caption=car_caption,
                           fb_car_caption=fb_car_caption,
                           fb_cine_caption=fb_cine_caption,
                           x_car_caption=x_car_caption,
                           threads_car_caption=threads_car_caption,
                           pinterest_car_caption=pinterest_car_caption,
                           linkedin_car_caption=linkedin_car_caption,
                           bluesky_car_caption=bluesky_car_caption,
                           reddit_car_caption=reddit_car_caption,
                           telegram_car_caption=telegram_car_caption,
                           mastodon_car_caption=mastodon_car_caption,
                           vk_car_caption=vk_car_caption,
                           tumblr_car_caption=tumblr_car_caption)


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


# ------------------------------------------------------------
# ADMIN — CATALOGUE
# ------------------------------------------------------------
def _fetch_catalogue_gallery(catalogue_id):
    """Return list of {id, image_url} for a catalogue item's gallery images."""
    rows = query_all(
        "SELECT id, image_url FROM catalogue_images "
        "WHERE catalogue_id = %s ORDER BY sort_order, id",
        (catalogue_id,),
    )
    return [{"id": r[0], "image_url": r[1]} for r in rows]


def _fetch_catalogue_specs(catalogue_id):
    """Return list of {label, value} specs for a catalogue item."""
    rows = query_all(
        "SELECT label, value FROM catalogue_specs "
        "WHERE catalogue_id = %s ORDER BY sort_order, id",
        (catalogue_id,),
    )
    return [{"label": r[0], "value": r[1]} for r in rows]


def _replace_catalogue_specs(catalogue_id, labels, values):
    """Replace all specs for a catalogue item with the provided (label, value) pairs."""
    execute("DELETE FROM catalogue_specs WHERE catalogue_id = %s", (catalogue_id,))
    order = 0
    for lbl, val in zip(labels or [], values or []):
        lbl = (lbl or "").strip()[:100]
        val = (val or "").strip()[:500]
        if lbl and val:
            execute(
                "INSERT INTO catalogue_specs (catalogue_id, label, value, sort_order) "
                "VALUES (%s, %s, %s, %s)",
                (catalogue_id, lbl, val, order),
            )
            order += 1


def _next_business_day(start=None, days=3):
    """Return a date `days` business days in the future (skipping Sat/Sun)."""
    d = start or datetime.utcnow()
    added = 0
    while added < days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # 0=Mon .. 4=Fri
            added += 1
    return d


def _existing_categories():
    """List distinct non-empty categories, for the admin datalist."""
    rows = query_all(
        "SELECT DISTINCT category FROM catalogue "
        "WHERE category IS NOT NULL AND category != '' ORDER BY category"
    )
    return [r[0] for r in rows if r[0]]


@app.route("/admin/catalogue")
@login_required
@admin_required
def admin_catalogue():
    """List all catalogue items with status and image."""
    rows = query_all(
        "SELECT id, title, description, price, image_url, status, category, slug, is_public "
        "FROM catalogue ORDER BY id DESC"
    )
    items = [
        {
            "id": r[0], "title": r[1], "description": r[2],
            "price": float(r[3]) if r[3] is not None else 0.0,
            "image_url": r[4],
            "status": r[5] or "available",
            "category": r[6] or "",
            "slug": r[7] or "",
            "is_public": bool(r[8]) if len(r) > 8 else True,
        }
        for r in rows
    ]
    return render_template("admin.html", active_tab="catalogue", catalogue=items)


@app.route("/admin/catalogue/new")
@login_required
@admin_required
def admin_catalogue_new():
    return render_template("catalogue_edit.html", item=None, gallery=[], specs=[],
                           existing_categories=_existing_categories(),
                           gift_persona_presets=GIFT_PERSONA_PRESETS)


@app.route("/admin/catalogue/new", methods=["POST"])
@login_required
@admin_required
def admin_catalogue_create():
    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    price_raw = (request.form.get("price") or "").strip()
    category_raw = (request.form.get("category") or "").strip()
    slug_raw = (request.form.get("slug") or "").strip()

    if not title:
        flash("Title is required.", "danger")
        return redirect(url_for("admin_catalogue_new"))
    try:
        price = float(price_raw) if price_raw else 0.0
    except ValueError:
        flash("Price must be a number.", "danger")
        return redirect(url_for("admin_catalogue_new"))

    category = slugify(category_raw) or None
    slug = slugify(slug_raw) or (slugify(title) if category else None)

    # If category is set, slug is required (make sure it's unique in that category)
    if category and slug:
        existing = query_one(
            "SELECT id FROM catalogue WHERE category = %s AND slug = %s",
            (category, slug),
        )
        if existing:
            flash(f"The URL /catalogue/{category}/{slug} is already in use.", "danger")
            return redirect(url_for("admin_catalogue_new"))

    # Primary image (optional)
    primary_url = None
    primary_file = request.files.get("primary_image")
    if primary_file and primary_file.filename:
        primary_url = save_catalogue_image(primary_file)

    is_public = 1 if request.form.get("is_public") in ("1", "on", "true") else 0
    gift_personas_str = ",".join(p for p in request.form.getlist("gift_persona") if p.strip()) or None

    new_id = execute(
        "INSERT INTO catalogue (title, description, price, image_url, status, category, slug, is_public, gift_personas) "
        "VALUES (%s, %s, %s, %s, 'available', %s, %s, %s, %s)",
        (title, description, price, primary_url, category, slug, is_public, gift_personas_str),
    )

    # Gallery images (optional, multiple)
    gallery_files = request.files.getlist("gallery_images")
    for idx, gf in enumerate(gallery_files or []):
        if gf and gf.filename:
            url = save_catalogue_image(gf)
            if url:
                execute(
                    "INSERT INTO catalogue_images (catalogue_id, image_url, sort_order) "
                    "VALUES (%s, %s, %s)",
                    (new_id, url, idx),
                )

    # Specs (key/value pairs)
    spec_labels = request.form.getlist("spec_label[]")
    spec_values = request.form.getlist("spec_value[]")
    _replace_catalogue_specs(new_id, spec_labels, spec_values)

    flash("Catalogue item created.", "success")
    return redirect(url_for("admin_catalogue_edit", item_id=new_id))


@app.route("/admin/catalogue/<int:item_id>/edit")
@login_required
@admin_required
def admin_catalogue_edit(item_id):
    row = query_one(
        "SELECT id, title, description, price, image_url, status, category, slug, is_public, gift_personas "
        "FROM catalogue WHERE id = %s",
        (item_id,),
    )
    if not row:
        flash("Item not found.", "warning")
        return redirect(url_for("admin_catalogue"))
    personas_raw = row[9] or ""
    item = {
        "id": row[0], "title": row[1], "description": row[2],
        "price": float(row[3]) if row[3] is not None else 0.0,
        "image_url": row[4],
        "status": row[5] or "available",
        "category": row[6] or "",
        "slug": row[7] or "",
        "is_public": bool(row[8]) if len(row) > 8 else True,
        "gift_personas": [p.strip() for p in personas_raw.split(",") if p.strip()],
    }
    gallery = _fetch_catalogue_gallery(item_id)
    specs = _fetch_catalogue_specs(item_id)
    return render_template("catalogue_edit.html", item=item, gallery=gallery, specs=specs,
                           existing_categories=_existing_categories(),
                           gift_persona_presets=GIFT_PERSONA_PRESETS)


@app.route("/admin/catalogue/<int:item_id>/save", methods=["POST"])
@login_required
@admin_required
def admin_catalogue_save(item_id):
    existing = query_one("SELECT image_url FROM catalogue WHERE id = %s", (item_id,))
    if not existing:
        flash("Item not found.", "warning")
        return redirect(url_for("admin_catalogue"))

    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    price_raw = (request.form.get("price") or "").strip()
    category_raw = (request.form.get("category") or "").strip()
    slug_raw = (request.form.get("slug") or "").strip()

    if not title:
        flash("Title is required.", "danger")
        return redirect(url_for("admin_catalogue_edit", item_id=item_id))
    try:
        price = float(price_raw) if price_raw else 0.0
    except ValueError:
        flash("Price must be a number.", "danger")
        return redirect(url_for("admin_catalogue_edit", item_id=item_id))

    category = slugify(category_raw) or None
    slug = slugify(slug_raw) or (slugify(title) if category else None)

    # Uniqueness check (ignore the item itself)
    if category and slug:
        clash = query_one(
            "SELECT id FROM catalogue WHERE category = %s AND slug = %s AND id != %s",
            (category, slug, item_id),
        )
        if clash:
            flash(f"The URL /catalogue/{category}/{slug} is already in use.", "danger")
            return redirect(url_for("admin_catalogue_edit", item_id=item_id))

    # Optional replacement of primary image
    primary_url = existing[0]
    primary_file = request.files.get("primary_image")
    if primary_file and primary_file.filename:
        new_url = save_catalogue_image(primary_file)
        if new_url:
            primary_url = new_url

    is_public = 1 if request.form.get("is_public") in ("1", "on", "true") else 0
    gift_personas_str = ",".join(p for p in request.form.getlist("gift_persona") if p.strip()) or None

    execute(
        "UPDATE catalogue SET title = %s, description = %s, price = %s, image_url = %s, "
        "category = %s, slug = %s, is_public = %s, gift_personas = %s WHERE id = %s",
        (title, description, price, primary_url, category, slug, is_public, gift_personas_str, item_id),
    )

    # Replace specs with the submitted set (simple and atomic)
    spec_labels = request.form.getlist("spec_label[]")
    spec_values = request.form.getlist("spec_value[]")
    _replace_catalogue_specs(item_id, spec_labels, spec_values)

    # Append any new gallery images
    gallery_files = request.files.getlist("gallery_images")
    # Start sort_order after the current max
    max_order_row = query_one(
        "SELECT COALESCE(MAX(sort_order), -1) FROM catalogue_images WHERE catalogue_id = %s",
        (item_id,),
    )
    next_order = (max_order_row[0] if max_order_row else -1) + 1
    for gf in gallery_files or []:
        if gf and gf.filename:
            url = save_catalogue_image(gf)
            if url:
                execute(
                    "INSERT INTO catalogue_images (catalogue_id, image_url, sort_order) "
                    "VALUES (%s, %s, %s)",
                    (item_id, url, next_order),
                )
                next_order += 1

    flash("Item updated.", "success")
    return redirect(url_for("admin_catalogue_edit", item_id=item_id))


@app.route("/admin/catalogue/<int:item_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_catalogue_delete(item_id):
    try:
        execute("DELETE FROM catalogue WHERE id = %s", (item_id,))
        flash("Item removed from catalogue.", "success")
    except Exception as e:
        flash(f"Could not delete: {e}", "danger")
    return redirect(url_for("admin_catalogue"))


@app.route("/admin/catalogue/<int:item_id>/publish", methods=["POST"])
@login_required
@admin_required
def admin_catalogue_publish(item_id):
    execute("UPDATE catalogue SET is_public = 1 WHERE id = %s", (item_id,))
    flash("Item is now public.", "success")
    return redirect(request.referrer or url_for("admin_catalogue"))


@app.route("/admin/catalogue/<int:item_id>/unpublish", methods=["POST"])
@login_required
@admin_required
def admin_catalogue_unpublish(item_id):
    execute("UPDATE catalogue SET is_public = 0 WHERE id = %s", (item_id,))
    flash("Item is now private (hidden from the public catalogue).", "success")
    return redirect(request.referrer or url_for("admin_catalogue"))


@app.route("/admin/catalogue/<int:item_id>/mark-sold", methods=["POST"])
@login_required
@admin_required
def admin_catalogue_mark_sold(item_id):
    execute("UPDATE catalogue SET status = 'sold' WHERE id = %s", (item_id,))
    flash("Marked as sold.", "success")
    return redirect(request.referrer or url_for("admin_catalogue"))


@app.route("/admin/catalogue/<int:item_id>/mark-available", methods=["POST"])
@login_required
@admin_required
def admin_catalogue_mark_available(item_id):
    execute("UPDATE catalogue SET status = 'available' WHERE id = %s", (item_id,))
    flash("Marked as available.", "success")
    return redirect(request.referrer or url_for("admin_catalogue"))


@app.route("/admin/catalogue/images/<int:image_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_catalogue_image_delete(image_id):
    row = query_one("SELECT catalogue_id FROM catalogue_images WHERE id = %s", (image_id,))
    if not row:
        return jsonify({"error": "not found"}), 404
    execute("DELETE FROM catalogue_images WHERE id = %s", (image_id,))
    return jsonify({"success": True})


# ------------------------------------------------------------
# ADMIN: UNRECOGNIZED STAMPS (captured by /stamp-quest/capture)
# ------------------------------------------------------------
@app.route("/admin/unrecognized-stamps")
@login_required
@admin_required
def admin_unrecognized_stamps():
    show_reviewed = request.args.get("reviewed") == "1"
    if show_reviewed:
        rows = query_all(
            """SELECT id, image_url, detection_confidence, best_match_similarity,
                      best_match_title, best_match_country, best_match_year,
                      title, country, year, condition_text, price_value,
                      price_currency, notes, reviewed, created_at, user_id
               FROM unrecognized_stamps
               ORDER BY created_at DESC LIMIT 500"""
        )
    else:
        rows = query_all(
            """SELECT id, image_url, detection_confidence, best_match_similarity,
                      best_match_title, best_match_country, best_match_year,
                      title, country, year, condition_text, price_value,
                      price_currency, notes, reviewed, created_at, user_id
               FROM unrecognized_stamps
               WHERE reviewed = 0
               ORDER BY created_at DESC LIMIT 500"""
        )
    items = []
    for r in rows:
        items.append({
            "id": r[0],
            "image_url": r[1] or url_for("admin_unrecognized_image", item_id=r[0]),
            "detection_confidence": float(r[2]) if r[2] is not None else None,
            "best_match_similarity": float(r[3]) if r[3] is not None else None,
            "best_match_title": r[4],
            "best_match_country": r[5],
            "best_match_year": r[6],
            "title": r[7] or "",
            "country": r[8] or "",
            "year": r[9] or "",
            "condition_text": r[10] or "",
            "price_value": float(r[11]) if r[11] is not None else "",
            "price_currency": r[12] or "EUR",
            "notes": r[13] or "",
            "reviewed": bool(r[14]),
            "created_at": r[15],
            "user_id": r[16],
        })
    return render_template("admin.html", active_tab="unrecognized",
                           unrecognized=items, show_reviewed=show_reviewed)


@app.route("/admin/unrecognized-stamps/<int:item_id>/image")
@login_required
@admin_required
def admin_unrecognized_image(item_id):
    from flask import Response
    row = query_one("SELECT image_blob FROM unrecognized_stamps WHERE id = %s", (item_id,))
    if row and row[0]:
        return Response(row[0], mimetype="image/jpeg")
    return ("", 404)


@app.route("/admin/unrecognized-stamps/<int:item_id>/save", methods=["POST"])
@login_required
@admin_required
def admin_unrecognized_save(item_id):
    title = (request.form.get("title") or "").strip() or None
    country = (request.form.get("country") or "").strip() or None
    year_raw = (request.form.get("year") or "").strip()
    try:
        year = int(year_raw) if year_raw else None
    except ValueError:
        year = None
    condition_text = (request.form.get("condition_text") or "").strip() or None
    price_raw = (request.form.get("price_value") or "").strip()
    try:
        price_value = float(price_raw) if price_raw else None
    except ValueError:
        price_value = None
    price_currency = (request.form.get("price_currency") or "EUR").strip().upper()[:3]
    notes = (request.form.get("notes") or "").strip() or None

    execute(
        """UPDATE unrecognized_stamps
           SET title=%s, country=%s, year=%s, condition_text=%s,
               price_value=%s, price_currency=%s, notes=%s
           WHERE id=%s""",
        (title, country, year, condition_text, price_value, price_currency, notes, item_id),
    )
    flash("Saved.", "success")
    return redirect(url_for("admin_unrecognized_stamps"))


@app.route("/admin/unrecognized-stamps/<int:item_id>/mark-reviewed", methods=["POST"])
@login_required
@admin_required
def admin_unrecognized_mark_reviewed(item_id):
    execute(
        "UPDATE unrecognized_stamps SET reviewed=1, reviewed_at=NOW() WHERE id=%s",
        (item_id,),
    )
    flash("Marked as reviewed.", "success")
    return redirect(url_for("admin_unrecognized_stamps"))


@app.route("/admin/unrecognized-stamps/<int:item_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_unrecognized_delete(item_id):
    execute("DELETE FROM unrecognized_stamps WHERE id=%s", (item_id,))
    flash("Deleted.", "success")
    return redirect(url_for("admin_unrecognized_stamps"))


@app.route("/admin/articles/<int:article_id>/short-urls")
@login_required
@admin_required
def admin_article_short_urls(article_id):
    """Get or generate short URLs for all platforms for this article."""
    platforms = ['ig', 'fb', 'yt', 'x', 'threads', 'pinterest', 'tiktok', 'linkedin',
                 'bluesky', 'reddit', 'telegram', 'vimeo', 'mastodon', 'vk', 'tumblr']
    result = {}
    for p in platforms:
        try:
            result[p] = make_short_url(article_id, p)
        except Exception:
            result[p] = ''
    return jsonify(result)


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
    """Background thread: GPT script → OpenAI TTS → (optional Whisper for caption timing)
    → FFmpeg per-clip → FFmpeg concat → save MP4."""
    fmt           = cfg.get("format",        "vertical")
    voice         = cfg.get("voice",         "onyx")
    tts_model     = cfg.get("tts_model",     "tts-1")
    script_len    = cfg.get("script_len",    "medium")
    kb_speed      = cfg.get("kb_speed",      "slow")
    crf           = int(cfg.get("crf",       23))
    fps           = int(cfg.get("fps",       25))
    captions_on   = bool(cfg.get("captions", True))
    caption_style = cfg.get("caption_style", "tiktok")

    W, H         = (720, 1280) if fmt == "vertical" else (720, 720)
    render_fps   = 12   # Ken Burns on still images looks identical at 12fps vs 25fps
    speed_factor = float(cfg.get("speed", 1.35))
    if speed_factor < 0.5 or speed_factor > 3.0:
        speed_factor = 1.35  # Clamp to safe range
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

        # ── 2b. Whisper word-level timestamps for caption sync ─────────────────
        # word_timings[i] = list of {'word': str, 'start': float, 'end': float}
        # Empty list = fallback to estimated timing in _build_ass below.
        word_timings = [[] for _ in segments]
        if captions_on:
            execute("UPDATE articles SET video_narrated_status = %s WHERE id = %s",
                    ("running:captions", article_id))
            _log(f"Captions: on (style={caption_style}); transcribing TTS audio for word timing…")
            for i, seg_audio in enumerate(audio_paths):
                try:
                    with open(seg_audio, "rb") as fh:
                        tr = _openai_client.audio.transcriptions.create(
                            file=fh, model="whisper-1",
                            response_format="verbose_json",
                            timestamp_granularities=["word"],
                        )
                    words = getattr(tr, "words", None) or []
                    word_timings[i] = [
                        {"word": (w.get("word") if isinstance(w, dict) else w.word),
                         "start": float((w.get("start") if isinstance(w, dict) else w.start) or 0),
                         "end":   float((w.get("end")   if isinstance(w, dict) else w.end)   or 0)}
                        for w in words
                    ]
                    _log(f"Whisper {i+1}/{n_slides}: {len(word_timings[i])} word timestamps")
                except Exception as _we:
                    _log(f"Whisper {i+1} failed ({_we}); will use estimated timing for this clip")
                _flush_log()

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

        # ── 3b. ASS subtitle file builder (caption burn-in) ──────────────────
        def _ass_time(t):
            """Format seconds as ASS H:MM:SS.cs"""
            t = max(0.0, float(t))
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t - (h * 3600) - (m * 60)
            return f"{h}:{m:02d}:{s:05.2f}"

        def _build_ass_for_clip(idx, seg_text, words, audio_dur):
            """Generate an ASS subtitle file for one clip. Returns Path or None."""
            if not captions_on:
                return None
            # Decide font size and bottom margin based on canvas
            if fmt == "vertical":
                font_size_main = 56 if caption_style == "tiktok" else 40
                margin_v       = int(H * 0.22)   # ~22% from bottom — clear of platform UI
            else:
                font_size_main = 46 if caption_style == "tiktok" else 34
                margin_v       = int(H * 0.12)
            outline_w  = 4 if caption_style == "tiktok" else 2
            shadow_w   = 2 if caption_style == "tiktok" else 0
            bold_flag  = -1 if caption_style == "tiktok" else 0  # ASS: -1 = true
            primary    = "&H00FFFFFF"   # white text in BGRA hex (00 = opaque alpha)
            outline    = "&H00000000"   # black outline

            # Group into chunks of ~3 words (better readability than word-by-word)
            chunk_size = 3
            chunks = []  # list of {'text': str, 'start': float, 'end': float}

            if words and len(words) > 0:
                # Use real Whisper timings
                for j in range(0, len(words), chunk_size):
                    grp = words[j:j+chunk_size]
                    text = " ".join((w["word"] or "").strip() for w in grp).strip()
                    if not text:
                        continue
                    start = grp[0]["start"]
                    end   = grp[-1]["end"]
                    if end <= start:
                        end = start + 0.6
                    chunks.append({"text": text, "start": start, "end": end})
            else:
                # Fallback: distribute evenly across audio duration
                tokens = [t for t in seg_text.split() if t]
                if not tokens:
                    return None
                n_chunks = max(1, (len(tokens) + chunk_size - 1) // chunk_size)
                per_chunk = max(0.4, audio_dur / n_chunks)
                for j in range(n_chunks):
                    grp = tokens[j*chunk_size:(j+1)*chunk_size]
                    if not grp:
                        continue
                    chunks.append({
                        "text": " ".join(grp),
                        "start": j * per_chunk,
                        "end":   min(audio_dur, (j + 1) * per_chunk - 0.05),
                    })

            if not chunks:
                return None

            # Build ASS file
            ass_path = video_dir / f"captions_{idx+1}_{ts}.ass"
            temp_files.append(ass_path)
            header = (
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                "Collisions: Normal\n"
                f"PlayResX: {W}\n"
                f"PlayResY: {H}\n"
                "ScaledBorderAndShadow: yes\n"
                "WrapStyle: 2\n\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
                "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
                "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                f"Style: Main,Arial,{font_size_main},{primary},&H00FFFF00,{outline},&H64000000,"
                f"{bold_flag},0,0,0,100,100,0,0,1,{outline_w},{shadow_w},2,40,40,{margin_v},1\n\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            )
            events = []
            for ck in chunks:
                txt = ck["text"].replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")
                # tiktok style adds a soft pop-in fade
                fade = r"{\fad(80,60)}" if caption_style == "tiktok" else ""
                events.append(
                    f"Dialogue: 0,{_ass_time(ck['start'])},{_ass_time(ck['end'])},Main,,0,0,0,,{fade}{txt}"
                )
            ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
            return ass_path

        def _ffmpeg_subtitles_arg(ass_path):
            """Return the FFmpeg subtitles= filter argument for a given ASS path.
            Properly escapes Windows drive letters and backslashes for libavfilter."""
            if not ass_path:
                return ""
            p = str(ass_path.resolve())
            # libavfilter on Windows needs: drive colon escaped, backslashes doubled
            esc = p.replace("\\", "/").replace(":", r"\:")
            return f",subtitles='{esc}'"

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

            # Build subtitle filter for this clip (ASS file, or empty string)
            ass_path  = _build_ass_for_clip(i, segments[i], word_timings[i], dur_video)
            subs_filt = _ffmpeg_subtitles_arg(ass_path)

            if use_cine:
                # ── Cinemagraph: loop video + scale/crop to target size ───
                S = max(W, H)
                cine_filter = (
                    f"[0:v]fps={render_fps},"
                    f"scale={S}:{S}:flags=fast_bilinear,"
                    f"crop={W}:{H},"
                    f"setsar=1"
                    f"{subs_filt}"
                    f"[v]"
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
                    f"setsar=1"
                    f"{subs_filt}"
                    f"[v]"
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
    # UTM tracking: replace plain article URL with YouTube-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        desc = desc.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'yt', article_id))
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
        # Store published caption (title + description) for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"yt_narrated_title_{article_id}_{run_ts}", title, title))
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"yt_narrated_caption_{article_id}_{run_ts}", desc, desc))
        except Exception:
            pass
        _add_activity_log(article_id, "Posted to YouTube Shorts",
                          f"video_id={video_id}\nurl={yt_url}",
                          component="narrated")
        try:
            log_social_post(article_id, 'yt', 'narrated', video_id, yt_url, desc)
        except Exception:
            pass

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
    title_key  = f"yt_title_{article_id}_{run_ts}"
    desc_key   = f"yt_desc_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already uploading this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
            (status_key, "running:start", "running:start"))
    # Save title/description so they can be restored on page reload
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
            (title_key, title, title))
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
            (desc_key, desc, desc))
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
    run_ts      = request.args.get("ts", "0")
    status_key  = f"yt_status_{article_id}_{run_ts}"
    result_key  = f"yt_result_{article_id}_{run_ts}"
    title_key   = f"yt_title_{article_id}_{run_ts}"
    desc_key    = f"yt_desc_{article_id}_{run_ts}"
    history_key = f"yt_history_{article_id}_{run_ts}"
    status  = get_setting(status_key) or "idle"
    result  = get_setting(result_key) or ""
    title   = get_setting(title_key)  or ""
    desc    = get_setting(desc_key)   or ""
    history = json.loads(get_setting(history_key) or "[]")
    return jsonify({"status": status, "result": result,
                    "title": title, "description": desc, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-youtube-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_youtube_post(article_id):
    """Delete a YouTube video and archive the record."""
    import datetime as _dt
    data       = request.get_json() or {}
    run_ts     = str(data.get("run_ts", "0"))
    status_key  = f"yt_status_{article_id}_{run_ts}"
    result_key  = f"yt_result_{article_id}_{run_ts}"
    title_key   = f"yt_title_{article_id}_{run_ts}"
    desc_key    = f"yt_desc_{article_id}_{run_ts}"
    history_key = f"yt_history_{article_id}_{run_ts}"

    result = get_setting(result_key) or ""
    if not result.startswith("done:"):
        return jsonify({"error": "No YouTube post to delete"}), 400
    yt_url   = result.replace("done:", "")
    video_id = yt_url.rstrip("/").split("/")[-1]  # e.g. https://youtube.com/shorts/XXXXX

    # Try to delete from YouTube via Data API
    refresh_token = get_setting("youtube_refresh_token")
    if refresh_token:
        try:
            import requests as _req
            token_resp = _req.post("https://oauth2.googleapis.com/token", data={
                "client_id":     os.getenv("GOOGLE_CLIENT_ID", ""),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
                "refresh_token": refresh_token,
                "grant_type":    "refresh_token",
            }, timeout=10)
            access_token = token_resp.json().get("access_token", "")
            if access_token:
                del_resp = _req.delete(
                    f"https://www.googleapis.com/youtube/v3/videos?id={video_id}",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=15,
                )
                if del_resp.status_code not in (200, 204, 404):
                    return jsonify({"error": f"YouTube API error: {del_resp.text[:200]}"}), 400
        except Exception as exc:
            print(f"[YT] delete error: {exc}", flush=True)
            # Continue to archive even if API delete fails

    # Archive
    title   = get_setting(title_key) or ""
    desc    = get_setting(desc_key) or ""
    history = json.loads(get_setting(history_key) or "[]")
    history.append({
        "video_id": video_id, "url": yt_url, "title": title, "description": desc,
        "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (status_key, result_key, title_key, desc_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, "YouTube Post Deleted & Archived",
                      f"video_id={video_id}", component="narrated")
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/archive-youtube-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_youtube_post(article_id):
    """Archive a YouTube post record without deleting from YouTube."""
    import datetime as _dt
    data       = request.get_json() or {}
    run_ts     = str(data.get("run_ts", "0"))
    status_key  = f"yt_status_{article_id}_{run_ts}"
    result_key  = f"yt_result_{article_id}_{run_ts}"
    title_key   = f"yt_title_{article_id}_{run_ts}"
    desc_key    = f"yt_desc_{article_id}_{run_ts}"
    history_key = f"yt_history_{article_id}_{run_ts}"

    result = get_setting(result_key) or ""
    if not result.startswith("done:"):
        return jsonify({"error": "No YouTube post to archive"}), 400
    yt_url   = result.replace("done:", "")
    video_id = yt_url.rstrip("/").split("/")[-1]
    title    = get_setting(title_key) or ""
    desc     = get_setting(desc_key) or ""
    history  = json.loads(get_setting(history_key) or "[]")
    history.append({
        "video_id": video_id, "url": yt_url, "title": title, "description": desc,
        "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (status_key, result_key, title_key, desc_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, "YouTube Post Archived",
                      f"video_id={video_id}", component="narrated")
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/check-youtube-post")
@login_required
@admin_required
def admin_check_youtube_post(article_id):
    """Check if a YouTube video is still live."""
    import requests as _req
    run_ts     = request.args.get("ts", "0")
    result_key = f"yt_result_{article_id}_{run_ts}"
    result     = get_setting(result_key) or ""
    if not result.startswith("done:"):
        return jsonify({"error": "No YouTube post to check"}), 400
    yt_url   = result.replace("done:", "")
    video_id = yt_url.rstrip("/").split("/")[-1]
    try:
        # Use oEmbed endpoint (no auth required)
        resp = _req.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=10,
        )
        if resp.status_code == 200:
            return jsonify({"live": True})
        return jsonify({"live": False, "api_error": f"HTTP {resp.status_code}"})
    except Exception as exc:
        return jsonify({"error": f"Check failed: {exc}"}), 500


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
        # UTM tracking: replace plain article URL with IG-tagged version
        _art = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if _art and _art[0] and SITE_URL:
            caption = caption.replace(f"{SITE_URL}/articles/{_art[0]}", make_utm_url(_art[0], 'ig', article_id))

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
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"ig_{post_type}_caption_{article_id}", caption, caption))
        except Exception:
            pass
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
        try:
            log_social_post(article_id, 'ig', post_type, str(post_id), permalink, caption)
        except Exception:
            pass
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
        # UTM tracking: replace plain article URL with IG-tagged version
        _art = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if _art and _art[0] and SITE_URL:
            caption = caption.replace(f"{SITE_URL}/articles/{_art[0]}", make_utm_url(_art[0], 'ig', article_id))

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
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"ig_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "Posted Narrated Video to Instagram",
                          f"post_id={post_id}\npermalink={permalink}", component="narrated")
        try:
            log_social_post(article_id, 'ig', 'narrated', str(post_id), permalink, caption)
        except Exception:
            pass
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

        # Use tracked short URL so clicks can be measured per-platform
        log_component = (request.get_json() or {}).get("component") or "carousel"
        if log_component not in ("carousel", "cinemagraph", "narrated"):
            log_component = "carousel"
        try:
            article_link = make_short_url(article_id, "ig") if SITE_URL and slug else ""
        except Exception:
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

        # Remove any article link the model may have included itself, then append
        # the canonical tracked short URL so there's never a duplicate.
        if article_link and slug:
            import re as _re2
            # Strip any trailing "Want to read…" block the model added (with any URL)
            caption = _re2.sub(
                r'\n+Want to read the full article\?.*$',
                '', caption, flags=_re2.IGNORECASE | _re2.DOTALL
            ).rstrip()
            # Also strip bare article URLs the model may have appended
            caption = _re2.sub(
                r'\n+https?://[^\s]*' + _re2.escape(slug) + r'[^\s]*$',
                '', caption, flags=_re2.IGNORECASE | _re2.DOTALL
            ).rstrip()
            caption += f"\n\nWant to read the full article? Access the following link:\n{article_link}"
        elif article_link:
            caption += f"\n\nWant to read the full article? Access the following link:\n{article_link}"

        _add_activity_log(article_id, "Caption Generated (OpenAI)",
                          f"Prompt: {system_msg}\n\nGenerated caption:\n{caption}",
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


# Platforms with tight character limits where the IG caption (up to 2200 chars)
# typically does not fit and needs AI-rewriting rather than naive truncation.
_SHORTEN_PLATFORM_LIMITS = {
    'x': 280,
    'bluesky': 300,
    'threads': 500,
    'mastodon': 500,
    'pinterest': 800,
}

@app.route("/admin/articles/<int:article_id>/shorten-caption-for-platform", methods=["POST"])
@login_required
@admin_required
def admin_article_shorten_caption(article_id):
    """Use OpenAI to rewrite an Instagram caption so it fits a target platform's
    character limit. Strips any existing 'Want to read…' CTA from the source,
    asks GPT to compress, then re-appends the target platform's tracked short URL.
    """
    if not _openai_client:
        return jsonify({"error": "OpenAI is not configured on this server."}), 400

    body     = request.get_json(silent=True) or {}
    source   = (body.get("source") or "").strip()
    platform = (body.get("platform") or "").strip().lower()

    if not source:
        return jsonify({"error": "Source caption is empty."}), 400
    if platform not in _SHORTEN_PLATFORM_LIMITS:
        return jsonify({"error": f"Unsupported platform: {platform}"}), 400

    max_chars = _SHORTEN_PLATFORM_LIMITS[platform]

    try:
        # Resolve target short URL + CTA template
        slug_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        slug = slug_row[0] if slug_row else ""
        try:
            short_url = make_short_url(article_id, platform) if SITE_URL and slug else ""
        except Exception:
            short_url = f"{SITE_URL}/articles/{slug}" if SITE_URL and slug else ""

        # Strip any existing "Want to read…" CTA / bare article URLs from source
        import re as _re
        body_text = _re.sub(
            r'\n+Want to read the full article\?.*$',
            '', source, flags=_re.IGNORECASE | _re.DOTALL
        ).rstrip()
        body_text = _re.sub(
            r'\n+https?://[^\s]+$', '', body_text, flags=_re.IGNORECASE
        ).rstrip()

        # Decide whether to attach a CTA (only if there's room)
        cta_template = f"\n\n👉 {short_url}" if short_url else ""
        target_body_chars = max(80, max_chars - len(cta_template) - 5)

        system_msg = (
            "You rewrite social-media captions so they fit a strict character "
            "budget while keeping the voice, emojis, hashtags-relevance and key "
            "selling point. Drop or shorten hashtags first if needed. Never include "
            "any URL or 'read more' line — those will be appended separately. "
            f"Output MUST be at most {target_body_chars} characters. "
            "Return ONLY the caption text, no commentary, no quotes."
        )
        user_msg = (
            f"Target platform: {platform} ({max_chars} char total budget).\n"
            f"Body budget (excluding URL): {target_body_chars} characters.\n\n"
            f"Original caption:\n{body_text}"
        )

        response = _openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=600,
        )
        shortened = (response.choices[0].message.content or "").strip()
        # Hard-truncate as a safety net if model overshoots
        if len(shortened) > target_body_chars:
            shortened = shortened[:target_body_chars].rstrip()

        final = shortened + (cta_template if short_url else "")
        if len(final) > max_chars:
            final = final[:max_chars]

        _add_activity_log(
            article_id,
            f"Caption Shortened ({platform}, OpenAI)",
            f"Target: {max_chars} chars (body budget {target_body_chars}).\n\n"
            f"Source ({len(source)} chars):\n{source}\n\n"
            f"Result ({len(final)} chars):\n{final}",
            component="carousel",
        )
        return jsonify({"caption": final, "length": len(final), "limit": max_chars})

    except Exception as e:
        print(f"Caption shorten error: {e}", flush=True)
        try:
            _add_activity_log(article_id, "Caption Shorten Failed",
                              f"Platform: {platform}\nError: {e}", component="carousel")
        except Exception:
            pass
        return jsonify({"error": f"Caption shortening failed: {e}"}), 500


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
        # UTM tracking: replace plain article URL with FB-tagged version
        _art = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if _art and _art[0] and SITE_URL:
            caption = caption.replace(f"{SITE_URL}/articles/{_art[0]}", make_utm_url(_art[0], 'fb', article_id))

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
                              component="carousel")
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

            n = len(valid_urls)
            # Pre-compute consistent band height for punchline overlays
            carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

            _add_activity_log(article_id, f"Facebook Carousel Uploading Photos",
                              f"Uploading {n} photos to Facebook as unpublished…\ncaption length={len(caption)}",
                              component=component_label)

            photo_ids = []
            ts_fb = int(_time.time())
            for idx, img_url in enumerate(valid_urls):
                _set_status(f"running:compose:{idx+1}/{n}")
                # Download original image
                try:
                    if img_url.startswith("https://"):
                        resp_img = _req.get(img_url, timeout=20)
                        resp_img.raise_for_status()
                        img_bytes = resp_img.content
                    else:
                        local = resolve_image_to_local_path(img_url)
                        img_bytes = local.read_bytes() if local and local.exists() else b""
                except Exception as e:
                    _set_status("idle")
                    _set_result(f"error:Could not fetch image {idx+1}: {e}")
                    _add_activity_log(article_id, f"Facebook Carousel Post Failed",
                                      f"Could not fetch image {idx+1}: {e}",
                                      component=component_label)
                    return

                # Compose punchline overlay onto image
                punchline = punchlines[idx] if idx < len(punchlines) else ""
                jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n, band_top=carousel_band_top, hint_text="obelisk-stamps.com")
                gcs_obj = f"articles/{article_id}/facebook/composed_{idx+1}_{ts_fb}.jpg"
                composed_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
                upload_url = composed_url or img_url

                _set_status(f"running:upload:{idx+1}/{n}")
                api_url = f"{_IG_GRAPH_URL}/{page_id}/photos"
                resp = _req.post(
                    api_url,
                    data={
                        "url": upload_url,
                        "published": "false",
                        "access_token": access_token,
                    },
                    timeout=60,
                )
                try:
                    data = resp.json()
                except Exception:
                    _set_status("idle")
                    _set_result(f"error:Photo upload failed (slide {idx+1}): unexpected response (HTTP {resp.status_code})")
                    _add_activity_log(article_id, f"Facebook Carousel Post Failed",
                                      f"Photo upload failed at slide {idx+1}/{n} — non-JSON response.\n"
                                      f"image_url={upload_url}\n"
                                      f"API endpoint={api_url}\n"
                                      f"HTTP status={resp.status_code}\n"
                                      f"Raw response={resp.text[:1500]}",
                                      component=component_label)
                    return
                if "id" not in data:
                    err = data.get("error", {}).get("message", str(data))
                    _set_status("idle")
                    _set_result(f"error:Photo upload failed (slide {idx+1}): {err}")
                    _add_activity_log(article_id, f"Facebook Carousel Post Failed",
                                      f"Photo upload failed at slide {idx+1}/{n}.\n"
                                      f"image_url={upload_url}\n"
                                      f"API endpoint={api_url}\n"
                                      f"HTTP status={resp.status_code}\n"
                                      f"API response={json.dumps(data, indent=2)[:1500]}",
                                      component=component_label)
                    return
                photo_ids.append(data["id"])
                print(f"[FB] Uploaded photo {idx+1}/{n}: {data['id']}", flush=True)

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
            try:
                data = resp.json()
            except Exception:
                _set_status("idle")
                _set_result(f"error:Feed post failed — unexpected response (HTTP {resp.status_code})")
                _add_activity_log(article_id, f"Facebook Carousel Post Failed",
                                  f"Feed post creation failed — non-JSON response.\n"
                                  f"photo_ids={photo_ids}\n"
                                  f"API endpoint={api_url}\n"
                                  f"HTTP status={resp.status_code}\n"
                                  f"Raw response={resp.text[:1500]}",
                                  component=component_label)
                return
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

        # ── Save results ───────────────────────────────────────
        _set_kv(keys["media_id"], str(post_id))
        snapshot = {
            "caption": caption,
            "post_type": post_type,
            "posted_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image_urls": images,
            "video_urls": [],
        }
        _set_kv(keys["snapshot"], json.dumps(snapshot))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"fb_{post_type}_caption_{article_id}", caption, caption))
        except Exception:
            pass

        slides_info = f"{len(valid_urls)} photos"
        _add_activity_log(
            article_id,
            f"Facebook {component_label.title()} Post Published",
            f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
            f"post_id={post_id}\n{slides_info}\ncaption length={len(caption)}",
            component=component_label,
        )
        try:
            log_social_post(article_id, 'fb', post_type, str(post_id), permalink, caption)
        except Exception:
            pass

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[FB] Worker error: {e}\n{tb}", flush=True)
        _set_status("idle")
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Facebook Post Failed (Exception)",
                          f"Unhandled exception:\n{str(e)}\n\nTraceback:\n{tb[:2000]}",
                          component="carousel")


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
        # UTM tracking: replace plain article URL with FB-tagged version
        _art = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if _art and _art[0] and SITE_URL:
            caption = caption.replace(f"{SITE_URL}/articles/{_art[0]}", make_utm_url(_art[0], 'fb', article_id))

        _set_status("running:upload")

        _add_activity_log(
            article_id,
            "Facebook Narrated Video Uploading",
            f"Uploading narrated video to Facebook Page…\n"
            f"video_url={video_url}\nrun_ts={run_ts}\n"
            f"page_id={FB_PAGE_ID}\ncaption length={len(caption)}",
            component="narrated",
        )

        # Use file_url so Facebook fetches the video directly from GCS
        # (avoids Cloud Run request body size limits / HTTP 413)
        api_url = f"{_FB_VIDEO_URL}/{FB_PAGE_ID}/videos"
        resp = _req.post(
            api_url,
            data={
                "file_url": video_url,
                "description": caption,
                "access_token": FB_PAGE_ACCESS_TOKEN,
            },
            timeout=180,
        )
        raw = resp.text or ""
        try:
            data = resp.json()
        except Exception:
            _set_status("idle")
            _set_result(f"error:Unexpected response from Facebook (HTTP {resp.status_code})")
            _add_activity_log(article_id, "Facebook Narrated Video Post Failed",
                              f"Video upload failed — non-JSON response.\n"
                              f"video_url={video_url}\n"
                              f"API endpoint={api_url}\n"
                              f"HTTP status={resp.status_code}\n"
                              f"Raw response={raw[:1500]}",
                              component="narrated")
            return

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
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"fb_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass

        _add_activity_log(
            article_id,
            "Facebook Narrated Video Post Published",
            f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
            f"video_id={video_id}\nrun_ts={run_ts}\ncaption length={len(caption)}",
            component="narrated",
        )
        try:
            log_social_post(article_id, 'fb', 'narrated', str(video_id), permalink, caption)
        except Exception:
            pass
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

    status_key  = f"narrated_fb_status_{article_id}_{run_ts}"
    result_key  = f"narrated_fb_result_{article_id}_{run_ts}"
    caption_key = f"narrated_fb_caption_{article_id}_{run_ts}"
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (result_key, "", ""))
    # Save caption so it can be restored on page reload
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (caption_key, caption, caption))

    t = threading.Thread(target=_post_narrated_fb_worker,
                         args=(article_id, video_url, caption, run_ts), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-fb-status")
@login_required
@admin_required
def admin_narrated_fb_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"narrated_fb_status_{article_id}_{run_ts}"
    result_key  = f"narrated_fb_result_{article_id}_{run_ts}"
    caption_key = f"narrated_fb_caption_{article_id}_{run_ts}"
    status      = get_setting(status_key) or "idle"
    result      = get_setting(result_key) or ""
    caption     = get_setting(caption_key) or ""
    history     = json.loads(get_setting(f"narrated_fb_history_{article_id}_{run_ts}") or "[]")
    return jsonify({"status": status, "result": result, "caption": caption, "history": history})


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


# ============================================================
# X (TWITTER) POSTING
# ============================================================

def _post_narrated_x_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to X and post a tweet."""
    import requests as _req
    import tempfile as _tmp
    import os as _os
    import time as _time_mod
    import base64 as _b64

    status_key = f"x_narrated_status_{article_id}_{run_ts}"
    result_key = f"x_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    tmp_path = None
    try:
        # UTM tracking: replace plain article URL with X-tagged version
        _art = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if _art and _art[0] and SITE_URL:
            caption = caption.replace(f"{SITE_URL}/articles/{_art[0]}", make_utm_url(_art[0], 'x', article_id))

        auth = _x_auth()

        # 1. Download video to temp file
        _set_status("running:download")
        with _tmp.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_path = tmp_f.name
        with _req.get(video_url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        file_size = _os.path.getsize(tmp_path)
        print(f"[X] Downloaded video: {file_size} bytes", flush=True)

        # 2. INIT upload session
        _set_status("running:upload_init")
        resp = _req.post(_X_UPLOAD_URL, auth=auth, data={
            "command":        "INIT",
            "total_bytes":    str(file_size),
            "media_type":     "video/mp4",
            "media_category": "tweet_video",
        }, timeout=30)
        data = resp.json()
        if "media_id_string" not in data:
            raise Exception(f"INIT failed: {data}")
        media_id = data["media_id_string"]
        print(f"[X] INIT media_id={media_id}", flush=True)

        # 3. APPEND chunks (5 MB each)
        _set_status("running:upload_append")
        chunk_size = 5 * 1024 * 1024
        segment = 0
        with open(tmp_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                resp = _req.post(_X_UPLOAD_URL, auth=auth, data={
                    "command":       "APPEND",
                    "media_id":      media_id,
                    "segment_index": str(segment),
                }, files={"media": chunk}, timeout=60)
                if resp.status_code not in (200, 204):
                    raise Exception(f"APPEND segment {segment} failed: {resp.text[:300]}")
                segment += 1
        print(f"[X] APPEND done ({segment} segments)", flush=True)

        # 4. FINALIZE
        _set_status("running:upload_finalize")
        resp = _req.post(_X_UPLOAD_URL, auth=auth, data={
            "command":  "FINALIZE",
            "media_id": media_id,
        }, timeout=30)
        data = resp.json()
        if "media_id_string" not in data:
            raise Exception(f"FINALIZE failed: {data}")
        print(f"[X] FINALIZE ok", flush=True)

        # 5. Poll STATUS until processing complete
        _set_status("running:processing")
        for _ in range(60):
            _time_mod.sleep(5)
            resp = _req.get(_X_UPLOAD_URL, auth=auth, params={
                "command":  "STATUS",
                "media_id": media_id,
            }, timeout=30)
            info = resp.json().get("processing_info", {})
            state = info.get("state", "succeeded")
            print(f"[X] STATUS state={state}", flush=True)
            if state == "succeeded":
                break
            if state == "failed":
                raise Exception(f"X video processing failed: {info}")
        else:
            raise Exception("X video processing timed out after 5 minutes")

        # 6. Post tweet
        _set_status("running:tweeting")
        tweet_text = caption[:280] if caption else ""
        resp = _req.post(_X_TWEET_URL, auth=auth, json={
            "text":  tweet_text,
            "media": {"media_ids": [media_id]},
        }, timeout=30)
        tweet_data = resp.json()
        tweet_id = (tweet_data.get("data") or {}).get("id")
        if not tweet_id:
            raise Exception(f"Tweet creation failed: {tweet_data}")

        permalink = f"https://x.com/i/web/status/{tweet_id}"
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"x_narrated_tweet_id_{article_id}_{run_ts}", tweet_id, tweet_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"x_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "X Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"tweet_id={tweet_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'x', 'narrated', tweet_id, permalink, caption)
        except Exception:
            pass
        print(f"[X] Narrated video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[X] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "X Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            _os.unlink(tmp_path)


def _post_carousel_x_worker(article_id, caption, run_ts, component):
    """Background thread: compose carousel/cinemagraph images and post as X thread (4 per tweet)."""
    import requests as _req
    import time as _time_mod

    status_key = f"x_{component}_status_{article_id}"
    result_key = f"x_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        # UTM tracking: replace plain article URL with X-tagged version
        _art = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if _art and _art[0] and SITE_URL:
            caption = caption.replace(f"{SITE_URL}/articles/{_art[0]}", make_utm_url(_art[0], 'x', article_id))

        auth = _x_auth()

        # Fetch images from DB
        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

        # Compose all images with overlay and upload to X
        _set_status(f"running:compose:0/{n}")
        media_ids = []
        ts_x = int(_time_mod.time())
        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            gcs_obj    = f"articles/{article_id}/x/{component}_{idx+1}_{ts_x}.jpg"
            public_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")

            # Upload image to X
            _set_status(f"running:upload:{idx+1}/{n}")
            if public_url:
                img_dl = _req.get(public_url, timeout=30)
                upload_bytes = img_dl.content
            else:
                upload_bytes = jpeg_bytes

            resp = _req.post(_X_UPLOAD_URL, auth=auth,
                             files={"media": ("image.jpg", upload_bytes, "image/jpeg")},
                             timeout=60)
            mid_data = resp.json()
            mid = mid_data.get("media_id_string")
            if not mid:
                _set_result(f"error:Image upload failed at {idx+1}: {mid_data}")
                return
            media_ids.append(mid)
            print(f"[X] Uploaded image {idx+1}/{n}: media_id={mid}", flush=True)

        # Post as thread: 4 images per tweet, each replying to previous
        _set_status("running:tweeting")
        chunks = [media_ids[i:i+4] for i in range(0, len(media_ids), 4)]
        first_tweet_id = None
        permalink      = None
        reply_to       = None

        for chunk_idx, chunk in enumerate(chunks):
            tweet_body = {}
            if chunk_idx == 0:
                tweet_body["text"] = caption[:280] if caption else ""
            else:
                tweet_body["text"] = f"{chunk_idx+1}/{len(chunks)}"
            tweet_body["media"] = {"media_ids": chunk}
            if reply_to:
                tweet_body["reply"] = {"in_reply_to_tweet_id": reply_to}

            resp = _req.post(_X_TWEET_URL, auth=auth, json=tweet_body, timeout=30)
            tweet_data = resp.json()
            tweet_id   = (tweet_data.get("data") or {}).get("id")
            if not tweet_id:
                raise Exception(f"Tweet {chunk_idx+1} failed: {tweet_data}")
            if chunk_idx == 0:
                first_tweet_id = tweet_id
                permalink      = f"https://x.com/i/web/status/{tweet_id}"
            reply_to = tweet_id
            print(f"[X] Posted tweet {chunk_idx+1}/{len(chunks)}: {tweet_id}", flush=True)

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"x_{component}_tweet_id_{article_id}", first_tweet_id, first_tweet_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"x_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"X {component.title()} Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"tweet_id={first_tweet_id}\nimages={n}\nthreads={len(chunks)}",
                          component=component)
        try:
            log_social_post(article_id, 'x', component, first_tweet_id, permalink, caption)
        except Exception:
            pass
        print(f"[X] Carousel posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[X] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"X {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


# ── X: Post narrated video ──────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-x", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_x(article_id):
    if not X_CONFIGURED:
        return jsonify({"error": "X credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"x_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_x_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-x-status")
@login_required
@admin_required
def admin_narrated_x_status(article_id):
    run_ts     = request.args.get("ts", "0")
    status_key = f"x_narrated_status_{article_id}_{run_ts}"
    result_key = f"x_narrated_result_{article_id}_{run_ts}"
    history_key = f"x_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/check-narrated-x-post")
@login_required
@admin_required
def admin_check_narrated_x_post(article_id):
    """Check if narrated-video tweet is still live on X."""
    import requests as _req
    if not X_CONFIGURED:
        return jsonify({"error": "X credentials not configured."}), 400
    run_ts    = request.args.get("ts", "0")
    tweet_key = f"x_narrated_tweet_id_{article_id}_{run_ts}"
    tweet_id  = get_setting(tweet_key)
    if not tweet_id:
        return jsonify({"error": "No published post found for this article."}), 400
    try:
        resp = _req.get(f"{_X_TWEET_URL}/{tweet_id}", auth=_x_auth(), timeout=15)
        data = resp.json()
        print(f"[X] check-narrated-post-live: tweet_id={tweet_id} resp={data}", flush=True)
        tweet_data = data.get("data")
        if tweet_data and tweet_data.get("id"):
            _add_activity_log(article_id, "Check X Post Live (narrated)",
                              f"Post is live. tweet_id={tweet_id}", component="narrated")
            return jsonify({"live": True})
        # Check for errors array (tweet deleted / not found)
        api_err = ""
        errors = data.get("errors", [])
        if errors:
            api_err = errors[0].get("detail", errors[0].get("message", "Tweet not found"))
        _add_activity_log(article_id, "Check X Post Live (narrated)",
                          f"Post not found. tweet_id={tweet_id}, API: {api_err}",
                          component="narrated")
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        print(f"[X] check-narrated-post-live exception: {exc}", flush=True)
        return jsonify({"error": f"Failed to reach X API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/archive-narrated-x-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_x_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    tweet_key   = f"x_narrated_tweet_id_{article_id}_{run_ts}"
    result_key  = f"x_narrated_result_{article_id}_{run_ts}"
    status_key  = f"x_narrated_status_{article_id}_{run_ts}"
    history_key = f"x_narrated_history_{article_id}_{run_ts}"
    tweet_id = get_setting(tweet_key)
    if not tweet_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"tweet_id": tweet_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (tweet_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, "X Narrated Video Post Archived",
                      f"tweet_id={tweet_id} archived", component="narrated")
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-x-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_x_post(article_id):
    """Delete tweet from X and archive locally."""
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    tweet_key   = f"x_narrated_tweet_id_{article_id}_{run_ts}"
    result_key  = f"x_narrated_result_{article_id}_{run_ts}"
    status_key  = f"x_narrated_status_{article_id}_{run_ts}"
    history_key = f"x_narrated_history_{article_id}_{run_ts}"
    tweet_id = get_setting(tweet_key)
    if not tweet_id:
        return jsonify({"error": "No tweet record found"}), 400
    # Delete from X
    try:
        resp = _req.delete(f"{_X_TWEET_URL}/{tweet_id}", auth=_x_auth(), timeout=15)
        if resp.status_code not in (200, 204):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            # If already gone, continue to archive
            if resp.status_code != 404:
                return jsonify({"error": f"Could not delete tweet: {err}"}), 400
    except Exception as e:
        pass  # Archive anyway
    # Archive locally
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"tweet_id": tweet_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (tweet_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, "X Narrated Video Post Deleted & Archived",
                      f"tweet_id={tweet_id}", component="narrated")
    return jsonify({"ok": True, "history": history})


# ── X: Post carousel images ─────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-x", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_x(article_id):
    if not X_CONFIGURED:
        return jsonify({"error": "X credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")  # "car" or "cine"
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    status_key = f"x_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_x_worker,
                     args=(article_id, caption, run_ts, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-x-status")
@login_required
@admin_required
def admin_carousel_x_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"x_{component}_status_{article_id}"
    result_key  = f"x_{component}_result_{article_id}"
    history_key = f"x_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/check-carousel-x-post")
@login_required
@admin_required
def admin_check_carousel_x_post(article_id):
    """Check if carousel/cinemagraph tweet is still live on X."""
    import requests as _req
    if not X_CONFIGURED:
        return jsonify({"error": "X credentials not configured."}), 400
    component = request.args.get("type", "car")
    if component not in ("car", "cine"):
        component = "car"
    tweet_key = f"x_{component}_tweet_id_{article_id}"
    tweet_id  = get_setting(tweet_key)
    if not tweet_id:
        return jsonify({"error": "No published post found for this article."}), 400
    try:
        resp = _req.get(f"{_X_TWEET_URL}/{tweet_id}", auth=_x_auth(), timeout=15)
        data = resp.json()
        print(f"[X] check-{component}-post-live: tweet_id={tweet_id} resp={data}", flush=True)
        tweet_data = data.get("data")
        if tweet_data and tweet_data.get("id"):
            _add_activity_log(article_id, f"Check X Post Live ({component})",
                              f"Post is live. tweet_id={tweet_id}", component=component)
            return jsonify({"live": True})
        api_err = ""
        errors = data.get("errors", [])
        if errors:
            api_err = errors[0].get("detail", errors[0].get("message", "Tweet not found"))
        _add_activity_log(article_id, f"Check X Post Live ({component})",
                          f"Post not found. tweet_id={tweet_id}, API: {api_err}",
                          component=component)
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        print(f"[X] check-{component}-post-live exception: {exc}", flush=True)
        return jsonify({"error": f"Failed to reach X API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/archive-carousel-x-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_x_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    tweet_key   = f"x_{component}_tweet_id_{article_id}"
    result_key  = f"x_{component}_result_{article_id}"
    status_key  = f"x_{component}_status_{article_id}"
    history_key = f"x_{component}_history_{article_id}"
    tweet_id = get_setting(tweet_key)
    if not tweet_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"tweet_id": tweet_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (tweet_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, f"X {component.title()} Post Archived",
                      f"tweet_id={tweet_id}", component=component)
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-x-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_x_post(article_id):
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    tweet_key   = f"x_{component}_tweet_id_{article_id}"
    result_key  = f"x_{component}_result_{article_id}"
    status_key  = f"x_{component}_status_{article_id}"
    history_key = f"x_{component}_history_{article_id}"
    tweet_id = get_setting(tweet_key)
    if not tweet_id:
        return jsonify({"error": "No tweet record found"}), 400
    try:
        resp = _req.delete(f"{_X_TWEET_URL}/{tweet_id}", auth=_x_auth(), timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete tweet: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"tweet_id": tweet_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (tweet_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, f"X {component.title()} Post Deleted & Archived",
                      f"tweet_id={tweet_id}", component=component)
    return jsonify({"ok": True, "history": history})


# ── Threads: Workers & Routes ────────────────────────────────

def _post_carousel_threads_worker(article_id, caption, component):
    """Background thread: compose carousel images and post as Threads carousel."""
    import requests as _req
    import time as _time_mod

    status_key = f"threads_{component}_status_{article_id}"
    result_key = f"threads_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        # UTM tracking: replace plain article URL with Threads-tagged version
        _art = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if _art and _art[0] and SITE_URL:
            caption = caption.replace(f"{SITE_URL}/articles/{_art[0]}", make_utm_url(_art[0], 'threads', article_id))

        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

        # 1. Compose and upload images to GCS, then create Threads item containers
        _set_status(f"running:compose:0/{n}")
        item_ids = []
        ts_t = int(_time_mod.time())

        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            gcs_obj    = f"articles/{article_id}/threads/{component}_{idx+1}_{ts_t}.jpg"
            public_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
            if not public_url:
                _set_result(f"error:Image upload failed for slide {idx+1}")
                return

            # Create Threads item container (carousel child)
            _set_status(f"running:containers:{idx+1}/{n}")
            resp = _req.post(f"{_THREADS_API_URL}/{THREADS_USER_ID}/threads", params={
                "media_type":        "IMAGE",
                "image_url":         public_url,
                "is_carousel_item":  "true",
                "access_token":      THREADS_ACCESS_TOKEN,
            }, timeout=30)
            cdata = resp.json()
            cid   = cdata.get("id")
            if not cid:
                _set_result(f"error:Container creation failed at {idx+1}: {cdata}")
                return
            item_ids.append(cid)
            print(f"[Threads] Item container {idx+1}/{n}: {cid}", flush=True)

        # 2. Create carousel container
        _set_status("running:carousel")
        _time_mod.sleep(5)  # Let Threads process items
        resp = _req.post(f"{_THREADS_API_URL}/{THREADS_USER_ID}/threads", params={
            "media_type":    "CAROUSEL",
            "children":      ",".join(item_ids),
            "text":          caption[:500] if caption else "",
            "access_token":  THREADS_ACCESS_TOKEN,
        }, timeout=30)
        carousel_data = resp.json()
        carousel_id   = carousel_data.get("id")
        if not carousel_id:
            _set_result(f"error:Carousel container failed: {carousel_data}")
            return
        print(f"[Threads] Carousel container: {carousel_id}", flush=True)

        # 3. Poll until ready then publish
        _set_status("running:poll")
        for attempt in range(30):
            _time_mod.sleep(5)
            resp = _req.get(f"{_THREADS_API_URL}/{carousel_id}", params={
                "fields":       "status",
                "access_token": THREADS_ACCESS_TOKEN,
            }, timeout=15)
            status_data = resp.json()
            container_status = status_data.get("status", "")
            print(f"[Threads] Container status: {container_status}", flush=True)
            if container_status == "FINISHED":
                break
            if container_status == "ERROR":
                _set_result(f"error:Container processing failed: {status_data}")
                return
        else:
            _set_result("error:Container processing timed out")
            return

        _set_status("running:publish")
        resp = _req.post(f"{_THREADS_API_URL}/{THREADS_USER_ID}/threads_publish", params={
            "creation_id":   carousel_id,
            "access_token":  THREADS_ACCESS_TOKEN,
        }, timeout=30)
        pub_data = resp.json()
        post_id  = pub_data.get("id")
        if not post_id:
            _set_result(f"error:Publish failed: {pub_data}")
            return

        # Fetch permalink
        resp = _req.get(f"{_THREADS_API_URL}/{post_id}", params={
            "fields":       "id,permalink",
            "access_token": THREADS_ACCESS_TOKEN,
        }, timeout=15)
        pdata     = resp.json()
        permalink = pdata.get("permalink", f"https://www.threads.net/@/post/{post_id}")

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"threads_{component}_post_id_{article_id}", post_id, post_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"threads_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"Threads {component.title()} Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"post_id={post_id}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'threads', component, post_id, permalink, caption)
        except Exception:
            pass
        print(f"[Threads] Carousel posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Threads] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Threads {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_threads_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to Threads."""
    import requests as _req
    import time as _time_mod

    status_key = f"threads_narrated_status_{article_id}_{run_ts}"
    result_key = f"threads_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        # UTM tracking: replace plain article URL with Threads-tagged version
        _art = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if _art and _art[0] and SITE_URL:
            caption = caption.replace(f"{SITE_URL}/articles/{_art[0]}", make_utm_url(_art[0], 'threads', article_id))

        # 1. Create video container
        _set_status("running:container")
        resp = _req.post(f"{_THREADS_API_URL}/{THREADS_USER_ID}/threads", params={
            "media_type":    "VIDEO",
            "video_url":     video_url,
            "text":          caption[:500] if caption else "",
            "access_token":  THREADS_ACCESS_TOKEN,
        }, timeout=30)
        cdata        = resp.json()
        container_id = cdata.get("id")
        if not container_id:
            _set_result(f"error:Container creation failed: {cdata}")
            return
        print(f"[Threads] Video container: {container_id}", flush=True)

        # 2. Poll until FINISHED
        _set_status("running:poll")
        for attempt in range(60):
            _time_mod.sleep(5)
            resp = _req.get(f"{_THREADS_API_URL}/{container_id}", params={
                "fields":       "status,error_message",
                "access_token": THREADS_ACCESS_TOKEN,
            }, timeout=15)
            status_data = resp.json()
            container_status = status_data.get("status", "")
            print(f"[Threads] Video status: {container_status}", flush=True)
            if container_status == "FINISHED":
                break
            if container_status == "ERROR":
                err_msg = status_data.get("error_message", "Processing failed")
                _set_result(f"error:{err_msg}")
                return
        else:
            _set_result("error:Video processing timed out after 5 minutes")
            return

        # 3. Publish
        _set_status("running:publish")
        resp = _req.post(f"{_THREADS_API_URL}/{THREADS_USER_ID}/threads_publish", params={
            "creation_id":   container_id,
            "access_token":  THREADS_ACCESS_TOKEN,
        }, timeout=30)
        pub_data = resp.json()
        post_id  = pub_data.get("id")
        if not post_id:
            _set_result(f"error:Publish failed: {pub_data}")
            return

        # Fetch permalink
        resp = _req.get(f"{_THREADS_API_URL}/{post_id}", params={
            "fields":       "id,permalink",
            "access_token": THREADS_ACCESS_TOKEN,
        }, timeout=15)
        pdata     = resp.json()
        permalink = pdata.get("permalink", f"https://www.threads.net/@/post/{post_id}")

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"threads_narrated_post_id_{article_id}_{run_ts}", post_id, post_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"threads_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "Threads Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"post_id={post_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'threads', 'narrated', post_id, permalink, caption)
        except Exception:
            pass
        print(f"[Threads] Narrated video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Threads] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Threads Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── Threads: Post carousel ──────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-threads", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_threads(article_id):
    if not THREADS_CONFIGURED:
        return jsonify({"error": "Threads credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"threads_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_threads_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-threads-status")
@login_required
@admin_required
def admin_carousel_threads_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"threads_{component}_status_{article_id}"
    result_key  = f"threads_{component}_result_{article_id}"
    history_key = f"threads_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/check-carousel-threads-post")
@login_required
@admin_required
def admin_check_carousel_threads_post(article_id):
    """Check if carousel/cinemagraph Threads post is still live."""
    import requests as _req
    if not THREADS_CONFIGURED:
        return jsonify({"error": "Threads credentials not configured."}), 400
    component = request.args.get("type", "car")
    if component not in ("car", "cine"):
        component = "car"
    post_key = f"threads_{component}_post_id_{article_id}"
    post_id  = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No published post found for this article."}), 400
    try:
        resp = _req.get(f"{_THREADS_API_URL}/{post_id}", params={
            "fields": "id,permalink", "access_token": THREADS_ACCESS_TOKEN,
        }, timeout=15)
        data = resp.json()
        print(f"[Threads] check-{component}-post-live: post_id={post_id} resp={data}", flush=True)
        if data.get("id"):
            _add_activity_log(article_id, f"Check Threads Post Live ({component})",
                              f"Post is live. post_id={post_id}", component=component)
            return jsonify({"live": True, "permalink": data.get("permalink", "")})
        api_err = data.get("error", {}).get("message", "Post not found")
        _add_activity_log(article_id, f"Check Threads Post Live ({component})",
                          f"Post not found. post_id={post_id}, API: {api_err}",
                          component=component)
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        print(f"[Threads] check-{component}-post-live exception: {exc}", flush=True)
        return jsonify({"error": f"Failed to reach Threads API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/archive-carousel-threads-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_threads_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"threads_{component}_post_id_{article_id}"
    result_key  = f"threads_{component}_result_{article_id}"
    status_key  = f"threads_{component}_status_{article_id}"
    history_key = f"threads_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, f"Threads {component.title()} Post Archived",
                      f"post_id={post_id}", component=component)
    return jsonify({"ok": True, "history": history})


# ── Threads: Post narrated video ─────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-threads", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_threads(article_id):
    if not THREADS_CONFIGURED:
        return jsonify({"error": "Threads credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"threads_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_threads_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-threads-status")
@login_required
@admin_required
def admin_narrated_threads_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"threads_narrated_status_{article_id}_{run_ts}"
    result_key  = f"threads_narrated_result_{article_id}_{run_ts}"
    history_key = f"threads_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/check-narrated-threads-post")
@login_required
@admin_required
def admin_check_narrated_threads_post(article_id):
    """Check if narrated-video Threads post is still live."""
    import requests as _req
    if not THREADS_CONFIGURED:
        return jsonify({"error": "Threads credentials not configured."}), 400
    run_ts   = request.args.get("ts", "0")
    post_key = f"threads_narrated_post_id_{article_id}_{run_ts}"
    post_id  = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No published post found for this article."}), 400
    try:
        resp = _req.get(f"{_THREADS_API_URL}/{post_id}", params={
            "fields": "id,permalink", "access_token": THREADS_ACCESS_TOKEN,
        }, timeout=15)
        data = resp.json()
        print(f"[Threads] check-narrated-post-live: post_id={post_id} resp={data}", flush=True)
        if data.get("id"):
            _add_activity_log(article_id, "Check Threads Post Live (narrated)",
                              f"Post is live. post_id={post_id}", component="narrated")
            return jsonify({"live": True, "permalink": data.get("permalink", "")})
        api_err = data.get("error", {}).get("message", "Post not found")
        _add_activity_log(article_id, "Check Threads Post Live (narrated)",
                          f"Post not found. post_id={post_id}, API: {api_err}",
                          component="narrated")
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        print(f"[Threads] check-narrated-post-live exception: {exc}", flush=True)
        return jsonify({"error": f"Failed to reach Threads API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/archive-narrated-threads-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_threads_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"threads_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"threads_narrated_result_{article_id}_{run_ts}"
    status_key  = f"threads_narrated_status_{article_id}_{run_ts}"
    history_key = f"threads_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, "Threads Narrated Video Post Archived",
                      f"post_id={post_id} archived", component="narrated")
    return jsonify({"ok": True, "history": history})


# ── Pinterest: Workers & Routes ──────────────────────────────

def _refresh_pinterest_token():
    """Use the stored refresh token to get a new Pinterest access token. Returns new token or None."""
    import requests as _req, time as _time, base64 as _b64
    refresh_token = get_setting("pinterest_refresh_token")
    if not refresh_token or not PINTEREST_CLIENT_ID or not PINTEREST_CLIENT_SECRET:
        return None
    try:
        creds = _b64.b64encode(f"{PINTEREST_CLIENT_ID}:{PINTEREST_CLIENT_SECRET}".encode()).decode()
        resp = _req.post(
            "https://api.pinterest.com/v5/oauth/token",
            headers={"Authorization": f"Basic {creds}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=15,
        )
        data = resp.json()
        new_token = data.get("access_token")
        if new_token:
            expires_in = data.get("expires_in", 2592000)
            expires_at = str(_time.time() + expires_in)
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    ("pinterest_access_token", new_token, new_token))
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    ("pinterest_token_expires", expires_at, expires_at))
            if data.get("refresh_token"):
                execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                        ("pinterest_refresh_token", data["refresh_token"], data["refresh_token"]))
            print("[Pinterest] Token refreshed successfully", flush=True)
            return new_token
        print(f"[Pinterest] Token refresh failed: {data}", flush=True)
    except Exception as _e:
        print(f"[Pinterest] Token refresh error: {_e}", flush=True)
    return None


def _pinterest_headers():
    """Return Authorization headers for Pinterest API, preferring DB-stored OAuth token."""
    import time as _time
    # Prefer DB token (set via OAuth flow) — fall back to env var
    db_token = get_setting("pinterest_access_token")
    if db_token:
        expires = get_setting("pinterest_token_expires")
        if expires:
            try:
                if float(expires) < _time.time() + 300:   # refresh 5 min before expiry
                    db_token = _refresh_pinterest_token() or db_token
            except ValueError:
                pass
        token = db_token
    else:
        token = PINTEREST_ACCESS_TOKEN
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _post_carousel_pinterest_worker(article_id, caption, component):
    """Background thread: compose carousel images and post as Pinterest pin(s)."""
    # UTM tracking: replace plain article URL with Pinterest-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'pinterest', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"pinterest_{component}_status_{article_id}"
    result_key = f"pinterest_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        row = query_one(
            "SELECT carousel_images, carousel_punchlines, slug FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        slug       = row[2] or ""
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

        # 1. Compose and upload images to GCS
        _set_status(f"running:compose:0/{n}")
        ts_t = int(_time_mod.time())
        public_urls = []

        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            gcs_obj    = f"articles/{article_id}/pinterest/{component}_{idx+1}_{ts_t}.jpg"
            public_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
            if not public_url:
                _set_result(f"error:Image upload failed for slide {idx+1}")
                return
            public_urls.append(public_url)

        # Build article link
        article_link = f"{SITE_URL}/articles/{slug}" if SITE_URL and slug else ""

        pin_title = (caption[:100] if caption else "")
        pin_desc  = (caption[:800] if caption else "")

        # 2. Create Pinterest pin(s) — max 5 images per carousel pin
        _set_status("running:pin")
        first_pin_id = None

        # Split into chunks of 5 (Pinterest carousel supports 2-5 images)
        url_chunks = []
        for i in range(0, len(public_urls), 5):
            chunk = public_urls[i:i+5]
            url_chunks.append(chunk)

        for chunk_idx, chunk_urls in enumerate(url_chunks):
            if len(chunk_urls) >= 2:
                # Carousel pin (multiple_image_urls) — 2-5 images
                items = [{"title": "", "description": "", "link": article_link,
                          "image_url": url} for url in chunk_urls]
                payload = {
                    "board_id": PINTEREST_BOARD_ID,
                    "title": pin_title,
                    "description": pin_desc,
                    "media_source": {
                        "source_type": "multiple_image_urls",
                        "items": items,
                    },
                }
                if article_link:
                    payload["link"] = article_link
            else:
                # Single image pin
                payload = {
                    "board_id": PINTEREST_BOARD_ID,
                    "title": pin_title,
                    "description": pin_desc,
                    "media_source": {
                        "source_type": "image_url",
                        "url": chunk_urls[0],
                    },
                }
                if article_link:
                    payload["link"] = article_link

            resp = _req.post(f"{_PINTEREST_API_URL}/pins",
                             headers=_pinterest_headers(),
                             json=payload, timeout=30)
            pin_data = resp.json()
            pin_id   = pin_data.get("id")
            if not pin_id:
                _set_result(f"error:Pin creation failed (chunk {chunk_idx+1}): {pin_data}")
                return
            if chunk_idx == 0:
                first_pin_id = pin_id
            print(f"[Pinterest] Pin created {chunk_idx+1}/{len(url_chunks)}: {pin_id}", flush=True)

        permalink = f"https://www.pinterest.com/pin/{first_pin_id}/"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"pinterest_{component}_pin_id_{article_id}", first_pin_id, first_pin_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"pinterest_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"Pinterest {component.title()} Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"pin_id={first_pin_id}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'pinterest', component, first_pin_id, permalink, caption)
        except Exception:
            pass
        print(f"[Pinterest] Carousel posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Pinterest] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Pinterest {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_pinterest_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to Pinterest as a video pin."""
    # UTM tracking: replace plain article URL with Pinterest-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'pinterest', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"pinterest_narrated_status_{article_id}_{run_ts}"
    result_key = f"pinterest_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        # Get article slug for link
        row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        slug = row[0] if row else ""
        article_link = f"{SITE_URL}/articles/{slug}" if SITE_URL and slug else ""

        # 1. Register media upload with Pinterest
        _set_status("running:register")
        resp = _req.post(f"{_PINTEREST_API_URL}/media", headers=_pinterest_headers(),
                         json={"media_type": "video"}, timeout=30)
        media_data = resp.json()
        media_id       = media_data.get("media_id")
        upload_url     = media_data.get("upload_url")
        upload_params  = media_data.get("upload_parameters") or {}
        if not media_id or not upload_url:
            _set_result(f"error:Media registration failed: {media_data}")
            return
        print(f"[Pinterest] Media registered: {media_id}", flush=True)

        # 2. Download video
        _set_status("running:download")
        resp_vid = _req.get(video_url, timeout=120, stream=True)
        resp_vid.raise_for_status()
        video_bytes = resp_vid.content
        print(f"[Pinterest] Video downloaded: {len(video_bytes)} bytes", flush=True)

        # 3. Upload video to S3 via multipart form
        _set_status("running:upload")
        # upload_params are form fields that must be sent along with the file
        files_payload = {}
        form_fields = []
        for k, v in upload_params.items():
            form_fields.append((k, (None, v)))
        form_fields.append(("file", ("video.mp4", video_bytes, "video/mp4")))
        resp_upload = _req.post(upload_url, files=form_fields, timeout=300)
        if resp_upload.status_code not in (200, 201, 204):
            _set_result(f"error:Video upload failed (HTTP {resp_upload.status_code}): {resp_upload.text[:300]}")
            return
        print(f"[Pinterest] Video uploaded to S3: {resp_upload.status_code}", flush=True)

        # 4. Poll media status until succeeded
        _set_status("running:poll")
        for attempt in range(90):
            _time_mod.sleep(5)
            resp_status = _req.get(f"{_PINTEREST_API_URL}/media/{media_id}",
                                   headers=_pinterest_headers(), timeout=15)
            status_data = resp_status.json()
            media_status = status_data.get("status", "")
            print(f"[Pinterest] Media status ({attempt+1}): {media_status}", flush=True)
            if media_status == "succeeded":
                break
            if media_status == "failed":
                _set_result(f"error:Video processing failed: {status_data}")
                return
        else:
            _set_result("error:Video processing timed out after ~7.5 minutes")
            return

        # 5. Create video pin
        _set_status("running:pin")
        pin_title = (caption[:100] if caption else "")
        pin_desc  = (caption[:800] if caption else "")
        payload = {
            "board_id": PINTEREST_BOARD_ID,
            "title": pin_title,
            "description": pin_desc,
            "media_source": {
                "source_type": "video_id",
                "media_id": media_id,
            },
        }
        if article_link:
            payload["link"] = article_link

        resp_pin = _req.post(f"{_PINTEREST_API_URL}/pins",
                             headers=_pinterest_headers(),
                             json=payload, timeout=30)
        pin_data = resp_pin.json()
        pin_id   = pin_data.get("id")
        if not pin_id:
            _set_result(f"error:Pin creation failed: {pin_data}")
            return

        permalink = f"https://www.pinterest.com/pin/{pin_id}/"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"pinterest_narrated_pin_id_{article_id}_{run_ts}", pin_id, pin_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"pinterest_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "Pinterest Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"pin_id={pin_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'pinterest', 'narrated', pin_id, permalink, caption)
        except Exception:
            pass
        print(f"[Pinterest] Narrated video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Pinterest] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Pinterest Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── Pinterest: OAuth connect ─────────────────────────────────

@app.route("/admin/pinterest-connect")
@login_required
@admin_required
def admin_pinterest_connect():
    """Redirect admin to Pinterest OAuth to authorise posting access."""
    import urllib.parse as _urlparse
    if not PINTEREST_CLIENT_ID:
        flash("PINTEREST_CLIENT_ID not configured. Add it as a GitHub Secret first.", "danger")
        return redirect(url_for("admin_panel"))
    params = {
        "client_id":     PINTEREST_CLIENT_ID,
        "redirect_uri":  PINTEREST_REDIRECT_URI,
        "response_type": "code",
        "scope":         "pins:read,pins:write,boards:read,boards:write,user_accounts:read",
    }
    url = "https://www.pinterest.com/oauth/?" + _urlparse.urlencode(params)
    return redirect(url)


@app.route("/admin/pinterest-oauth-callback")
@login_required
@admin_required
def admin_pinterest_oauth_callback():
    """Exchange Pinterest auth code for access + refresh token and store in DB."""
    import requests as _req, time as _time, base64 as _b64
    code = request.args.get("code")
    if not code:
        flash("Pinterest authorisation failed — no code returned.", "danger")
        return redirect(url_for("admin_panel"))
    try:
        creds = _b64.b64encode(f"{PINTEREST_CLIENT_ID}:{PINTEREST_CLIENT_SECRET}".encode()).decode()
        resp = _req.post(
            "https://api.pinterest.com/v5/oauth/token",
            headers={"Authorization": f"Basic {creds}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type":   "authorization_code",
                "code":          code,
                "redirect_uri":  PINTEREST_REDIRECT_URI,
            },
            timeout=15,
        )
        data = resp.json()
        access_token  = data.get("access_token")
        refresh_token = data.get("refresh_token")
        if not access_token:
            flash(f"Pinterest auth failed: {data.get('message', data)}", "danger")
            return redirect(url_for("admin_panel"))
        expires_in = data.get("expires_in", 2592000)
        expires_at = str(_time.time() + expires_in)
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                ("pinterest_access_token", access_token, access_token))
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                ("pinterest_token_expires", expires_at, expires_at))
        if refresh_token:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    ("pinterest_refresh_token", refresh_token, refresh_token))
        flash("Pinterest account connected successfully!", "success")
    except Exception as _e:
        flash(f"Pinterest OAuth error: {_e}", "danger")
    return redirect(url_for("admin_panel"))


# ── Pinterest: Post carousel ────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-pinterest", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_pinterest(article_id):
    if not PINTEREST_CONFIGURED:
        return jsonify({"error": "Pinterest credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"pinterest_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_pinterest_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-pinterest-status")
@login_required
@admin_required
def admin_carousel_pinterest_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"pinterest_{component}_status_{article_id}"
    result_key  = f"pinterest_{component}_result_{article_id}"
    history_key = f"pinterest_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/check-carousel-pinterest-post")
@login_required
@admin_required
def admin_check_carousel_pinterest_post(article_id):
    """Check if carousel/cinemagraph Pinterest pin is still live."""
    import requests as _req
    if not PINTEREST_CONFIGURED:
        return jsonify({"error": "Pinterest credentials not configured."}), 400
    component = request.args.get("type", "car")
    if component not in ("car", "cine"):
        component = "car"
    pin_key = f"pinterest_{component}_pin_id_{article_id}"
    pin_id  = get_setting(pin_key)
    if not pin_id:
        return jsonify({"error": "No published pin found for this article."}), 400
    try:
        resp = _req.get(f"{_PINTEREST_API_URL}/pins/{pin_id}",
                        headers=_pinterest_headers(), timeout=15)
        data = resp.json()
        print(f"[Pinterest] check-{component}-pin-live: pin_id={pin_id} status={resp.status_code}", flush=True)
        if data.get("id"):
            _add_activity_log(article_id, f"Check Pinterest Pin Live ({component})",
                              f"Pin is live. pin_id={pin_id}", component=component)
            return jsonify({"live": True, "permalink": f"https://www.pinterest.com/pin/{pin_id}/"})
        api_err = data.get("message", "Pin not found")
        _add_activity_log(article_id, f"Check Pinterest Pin Live ({component})",
                          f"Pin not found. pin_id={pin_id}, API: {api_err}",
                          component=component)
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        print(f"[Pinterest] check-{component}-pin-live exception: {exc}", flush=True)
        return jsonify({"error": f"Failed to reach Pinterest API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/archive-carousel-pinterest-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_pinterest_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    pin_key     = f"pinterest_{component}_pin_id_{article_id}"
    result_key  = f"pinterest_{component}_result_{article_id}"
    status_key  = f"pinterest_{component}_status_{article_id}"
    history_key = f"pinterest_{component}_history_{article_id}"
    pin_id = get_setting(pin_key)
    if not pin_id:
        return jsonify({"error": "No pin to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"pin_id": pin_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (pin_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, f"Pinterest {component.title()} Pin Archived",
                      f"pin_id={pin_id}", component=component)
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-pinterest-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_pinterest_post(article_id):
    """Delete pin from Pinterest and archive locally."""
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    pin_key     = f"pinterest_{component}_pin_id_{article_id}"
    result_key  = f"pinterest_{component}_result_{article_id}"
    status_key  = f"pinterest_{component}_status_{article_id}"
    history_key = f"pinterest_{component}_history_{article_id}"
    pin_id = get_setting(pin_key)
    if not pin_id:
        return jsonify({"error": "No pin record found"}), 400
    # Delete from Pinterest
    try:
        resp = _req.delete(f"{_PINTEREST_API_URL}/pins/{pin_id}",
                           headers=_pinterest_headers(), timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete pin: {err}"}), 400
    except Exception:
        pass  # Archive anyway
    # Archive locally
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"pin_id": pin_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (pin_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, f"Pinterest {component.title()} Pin Deleted & Archived",
                      f"pin_id={pin_id}", component=component)
    return jsonify({"ok": True, "history": history})


# ── Pinterest: Post narrated video ──────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-pinterest", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_pinterest(article_id):
    if not PINTEREST_CONFIGURED:
        return jsonify({"error": "Pinterest credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"pinterest_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_pinterest_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-pinterest-status")
@login_required
@admin_required
def admin_narrated_pinterest_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"pinterest_narrated_status_{article_id}_{run_ts}"
    result_key  = f"pinterest_narrated_result_{article_id}_{run_ts}"
    history_key = f"pinterest_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/check-narrated-pinterest-post")
@login_required
@admin_required
def admin_check_narrated_pinterest_post(article_id):
    """Check if narrated-video Pinterest pin is still live."""
    import requests as _req
    if not PINTEREST_CONFIGURED:
        return jsonify({"error": "Pinterest credentials not configured."}), 400
    run_ts  = request.args.get("ts", "0")
    pin_key = f"pinterest_narrated_pin_id_{article_id}_{run_ts}"
    pin_id  = get_setting(pin_key)
    if not pin_id:
        return jsonify({"error": "No published pin found for this article."}), 400
    try:
        resp = _req.get(f"{_PINTEREST_API_URL}/pins/{pin_id}",
                        headers=_pinterest_headers(), timeout=15)
        data = resp.json()
        print(f"[Pinterest] check-narrated-pin-live: pin_id={pin_id} status={resp.status_code}", flush=True)
        if data.get("id"):
            _add_activity_log(article_id, "Check Pinterest Pin Live (narrated)",
                              f"Pin is live. pin_id={pin_id}", component="narrated")
            return jsonify({"live": True, "permalink": f"https://www.pinterest.com/pin/{pin_id}/"})
        api_err = data.get("message", "Pin not found")
        _add_activity_log(article_id, "Check Pinterest Pin Live (narrated)",
                          f"Pin not found. pin_id={pin_id}, API: {api_err}",
                          component="narrated")
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        print(f"[Pinterest] check-narrated-pin-live exception: {exc}", flush=True)
        return jsonify({"error": f"Failed to reach Pinterest API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/archive-narrated-pinterest-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_pinterest_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    pin_key     = f"pinterest_narrated_pin_id_{article_id}_{run_ts}"
    result_key  = f"pinterest_narrated_result_{article_id}_{run_ts}"
    status_key  = f"pinterest_narrated_status_{article_id}_{run_ts}"
    history_key = f"pinterest_narrated_history_{article_id}_{run_ts}"
    pin_id = get_setting(pin_key)
    if not pin_id:
        return jsonify({"error": "No pin to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"pin_id": pin_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (pin_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, "Pinterest Narrated Video Pin Archived",
                      f"pin_id={pin_id} archived", component="narrated")
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-pinterest-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_pinterest_post(article_id):
    """Delete narrated video pin from Pinterest and archive locally."""
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    pin_key     = f"pinterest_narrated_pin_id_{article_id}_{run_ts}"
    result_key  = f"pinterest_narrated_result_{article_id}_{run_ts}"
    status_key  = f"pinterest_narrated_status_{article_id}_{run_ts}"
    history_key = f"pinterest_narrated_history_{article_id}_{run_ts}"
    pin_id = get_setting(pin_key)
    if not pin_id:
        return jsonify({"error": "No pin record found"}), 400
    # Delete from Pinterest
    try:
        resp = _req.delete(f"{_PINTEREST_API_URL}/pins/{pin_id}",
                           headers=_pinterest_headers(), timeout=15)
        if resp.status_code not in (200, 204):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            if resp.status_code != 404:
                return jsonify({"error": f"Could not delete pin: {err}"}), 400
    except Exception:
        pass  # Archive anyway
    # Archive locally
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"pin_id": pin_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (pin_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, "Pinterest Narrated Video Pin Deleted & Archived",
                      f"pin_id={pin_id}", component="narrated")
    return jsonify({"ok": True, "history": history})


# ── TikTok: Workers & Routes ─────────────────────────────────

def _post_narrated_tiktok_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to TikTok via Content Posting API."""
    # UTM tracking: replace plain article URL with TikTok-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'tiktok', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"tiktok_narrated_status_{article_id}_{run_ts}"
    result_key = f"tiktok_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        headers = {
            "Authorization": f"Bearer {TIKTOK_ACCESS_TOKEN}",
            "Content-Type": "application/json; charset=UTF-8",
        }

        # 1. Initialize video upload (pull from URL)
        _set_status("running:init")
        resp = _req.post(f"{_TIKTOK_API_URL}/post/publish/video/init/", headers=headers, json={
            "post_info": {
                "title": caption[:150] if caption else "",
                "privacy_level": "PUBLIC_TO_EVERYONE",
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
            },
            "source_info": {
                "source": "PULL_FROM_URL",
                "video_url": video_url,
            },
        }, timeout=30)
        data = resp.json()
        print(f"[TikTok] init response: {data}", flush=True)

        if data.get("error", {}).get("code") != "ok":
            err_msg = data.get("error", {}).get("message", "Init failed")
            _set_result(f"error:{err_msg}")
            return

        publish_id = data.get("data", {}).get("publish_id")
        if not publish_id:
            _set_result(f"error:No publish_id returned: {data}")
            return

        # 2. Poll status until complete
        _set_status("running:processing")
        for attempt in range(60):
            _time_mod.sleep(5)
            resp = _req.post(f"{_TIKTOK_API_URL}/post/publish/status/fetch/", headers=headers, json={
                "publish_id": publish_id,
            }, timeout=15)
            sdata = resp.json()
            status_val = sdata.get("data", {}).get("status", "")
            print(f"[TikTok] status: {status_val}", flush=True)

            if status_val == "PUBLISH_COMPLETE":
                break
            if status_val in ("FAILED", "PUBLISH_CANCELLED"):
                fail_reason = sdata.get("data", {}).get("fail_reason", "Unknown error")
                _set_result(f"error:{fail_reason}")
                return
        else:
            _set_result("error:Video processing timed out after 5 minutes")
            return

        video_id = sdata.get("data", {}).get("video_id", publish_id)
        permalink = f"https://www.tiktok.com/@/video/{video_id}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"tiktok_narrated_video_id_{article_id}_{run_ts}", video_id, video_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"tiktok_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "TikTok Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"video_id={video_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'tiktok', 'narrated', video_id, permalink, caption)
        except Exception:
            pass
        print(f"[TikTok] Video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[TikTok] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "TikTok Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


@app.route("/admin/articles/<int:article_id>/post-narrated-to-tiktok", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_tiktok(article_id):
    if not TIKTOK_CONFIGURED:
        return jsonify({"error": "TikTok credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"tiktok_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_tiktok_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-tiktok-status")
@login_required
@admin_required
def admin_narrated_tiktok_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"tiktok_narrated_status_{article_id}_{run_ts}"
    result_key  = f"tiktok_narrated_result_{article_id}_{run_ts}"
    history_key = f"tiktok_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-narrated-tiktok-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_tiktok_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    video_key   = f"tiktok_narrated_video_id_{article_id}_{run_ts}"
    result_key  = f"tiktok_narrated_result_{article_id}_{run_ts}"
    status_key  = f"tiktok_narrated_status_{article_id}_{run_ts}"
    history_key = f"tiktok_narrated_history_{article_id}_{run_ts}"
    video_id = get_setting(video_key)
    if not video_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"video_id": video_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (video_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    _add_activity_log(article_id, "TikTok Narrated Video Post Archived",
                      f"video_id={video_id} archived", component="narrated")
    return jsonify({"ok": True, "history": history})


# ── LinkedIn: Workers & Routes ───────────────────────────────

def _post_carousel_linkedin_worker(article_id, caption, component):
    """Background thread: compose carousel images and post to LinkedIn."""
    # UTM tracking: replace plain article URL with LinkedIn-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'linkedin', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"linkedin_{component}_status_{article_id}"
    result_key = f"linkedin_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        headers = {
            "Authorization":            f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type":             "application/json",
            "LinkedIn-Version":         "202502",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        author = f"urn:li:organization:{LINKEDIN_ORG_ID}"

        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

        _set_status(f"running:compose:0/{n}")
        image_urls = []
        ts_l = int(_time_mod.time())
        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            gcs_obj    = f"articles/{article_id}/linkedin/{component}_{idx+1}_{ts_l}.jpg"
            public_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
            if public_url:
                image_urls.append(public_url)

        if not image_urls:
            _set_result("error:No images uploaded successfully")
            return

        # Register images with LinkedIn
        _set_status("running:upload")
        image_urns = []
        for idx, public_url in enumerate(image_urls):
            resp = _req.post(f"{_LINKEDIN_API_URL}/images?action=initializeUpload", headers=headers, json={
                "initializeUploadRequest": {"owner": author}
            }, timeout=30)
            init_data = resp.json()
            upload_url = init_data.get("value", {}).get("uploadUrl", "")
            image_urn  = init_data.get("value", {}).get("image", "")
            if not upload_url or not image_urn:
                _set_result(f"error:Image init failed at {idx+1}: {init_data}")
                return
            img_dl = _req.get(public_url, timeout=30)
            resp = _req.put(upload_url, headers={
                "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
                "Content-Type": "image/jpeg",
            }, data=img_dl.content, timeout=60)
            if resp.status_code not in (200, 201):
                _set_result(f"error:Image upload failed at {idx+1}")
                return
            image_urns.append(image_urn)
            print(f"[LinkedIn] Uploaded image {idx+1}/{len(image_urls)}: {image_urn}", flush=True)

        _set_status("running:publish")
        post_body = {
            "author": author,
            "commentary": caption[:3000] if caption else "",
            "visibility": "PUBLIC",
            "distribution": {"feedDistribution": "MAIN_FEED", "targetEntities": [], "thirdPartyDistributionChannels": []},
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }
        if len(image_urns) == 1:
            post_body["content"] = {"media": {"id": image_urns[0], "altText": caption[:200] if caption else ""}}
        else:
            post_body["content"] = {"multiImage": {"images": [{"id": urn, "altText": ""} for urn in image_urns]}}

        resp = _req.post(f"{_LINKEDIN_API_URL}/posts", headers=headers, json=post_body, timeout=30)
        if resp.status_code not in (200, 201):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:500]
            _set_result(f"error:Post creation failed: {err}")
            return

        post_id = resp.headers.get("x-restli-id", "")
        permalink = f"https://www.linkedin.com/feed/update/{post_id}/" if post_id else ""

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"linkedin_{component}_post_id_{article_id}", post_id, post_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"linkedin_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"LinkedIn {component.title()} Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\npost_id={post_id}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'linkedin', component, post_id, permalink, caption)
        except Exception:
            pass
        print(f"[LinkedIn] Post created: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[LinkedIn] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"LinkedIn {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_linkedin_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to LinkedIn."""
    # UTM tracking: replace plain article URL with LinkedIn-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'linkedin', article_id))
    import requests as _req
    import time as _time_mod
    import tempfile as _tmp
    import os as _os

    status_key = f"linkedin_narrated_status_{article_id}_{run_ts}"
    result_key = f"linkedin_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    tmp_path = None
    try:
        headers = {
            "Authorization":            f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "Content-Type":             "application/json",
            "LinkedIn-Version":         "202502",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        author = f"urn:li:organization:{LINKEDIN_ORG_ID}"

        _set_status("running:download")
        with _tmp.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_path = tmp_f.name
        with _req.get(video_url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        file_size = _os.path.getsize(tmp_path)

        _set_status("running:upload_init")
        resp = _req.post(f"{_LINKEDIN_API_URL}/videos?action=initializeUpload", headers=headers, json={
            "initializeUploadRequest": {"owner": author, "fileSizeBytes": file_size, "uploadCaptions": False, "uploadThumbnail": False}
        }, timeout=30)
        init_data = resp.json()
        video_urn  = init_data.get("value", {}).get("video", "")
        upload_instructions = init_data.get("value", {}).get("uploadInstructions", [])
        if not video_urn or not upload_instructions:
            _set_result(f"error:Video init failed: {init_data}")
            return

        _set_status("running:upload")
        with open(tmp_path, "rb") as f:
            for instr in upload_instructions:
                upload_url = instr.get("uploadUrl", "")
                if not upload_url:
                    continue
                chunk_data = f.read(file_size)
                resp = _req.put(upload_url, headers={
                    "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
                    "Content-Type": "application/octet-stream",
                }, data=chunk_data, timeout=120)
                if resp.status_code not in (200, 201):
                    _set_result(f"error:Video upload failed: {resp.status_code}")
                    return

        _set_status("running:finalize")
        _req.post(f"{_LINKEDIN_API_URL}/videos?action=finalizeUpload", headers=headers, json={
            "finalizeUploadRequest": {"video": video_urn, "uploadToken": "", "uploadedPartIds": []}
        }, timeout=30)

        _set_status("running:processing")
        for attempt in range(60):
            _time_mod.sleep(5)
            resp = _req.get(f"{_LINKEDIN_API_URL}/videos/{video_urn}", headers={
                "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}", "LinkedIn-Version": "202502",
            }, timeout=15)
            if resp.status_code == 200:
                v_status = resp.json().get("status", "")
                if v_status == "AVAILABLE":
                    break
                if v_status in ("PROCESSING_FAILED", "UPLOAD_FAILED"):
                    _set_result(f"error:Video processing failed: {v_status}")
                    return

        _set_status("running:publish")
        post_body = {
            "author": author,
            "commentary": caption[:3000] if caption else "",
            "visibility": "PUBLIC",
            "distribution": {"feedDistribution": "MAIN_FEED", "targetEntities": [], "thirdPartyDistributionChannels": []},
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
            "content": {"media": {"id": video_urn}},
        }
        resp = _req.post(f"{_LINKEDIN_API_URL}/posts", headers=headers, json=post_body, timeout=30)
        if resp.status_code not in (200, 201):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:500]
            _set_result(f"error:Post creation failed: {err}")
            return

        post_id = resp.headers.get("x-restli-id", "")
        permalink = f"https://www.linkedin.com/feed/update/{post_id}/" if post_id else ""

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"linkedin_narrated_post_id_{article_id}_{run_ts}", post_id, post_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"linkedin_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "LinkedIn Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\npost_id={post_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'linkedin', 'narrated', post_id, permalink, caption)
        except Exception:
            pass

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[LinkedIn] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "LinkedIn Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            _os.unlink(tmp_path)


# ── LinkedIn: Routes ─────────────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-linkedin", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_linkedin(article_id):
    if not LINKEDIN_CONFIGURED:
        return jsonify({"error": "LinkedIn credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"linkedin_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_linkedin_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-linkedin-status")
@login_required
@admin_required
def admin_carousel_linkedin_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"linkedin_{component}_status_{article_id}"
    result_key  = f"linkedin_{component}_result_{article_id}"
    history_key = f"linkedin_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/check-carousel-linkedin-post")
@login_required
@admin_required
def admin_check_carousel_linkedin_post(article_id):
    import requests as _req
    if not LINKEDIN_CONFIGURED:
        return jsonify({"error": "LinkedIn credentials not configured."}), 400
    component = request.args.get("type", "car")
    post_key  = f"linkedin_{component}_post_id_{article_id}"
    post_id   = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No published post found."}), 400
    try:
        resp = _req.get(f"{_LINKEDIN_API_URL}/posts/{post_id}", headers={
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "LinkedIn-Version": "202502", "X-Restli-Protocol-Version": "2.0.0",
        }, timeout=15)
        if resp.status_code == 200:
            return jsonify({"live": True})
        api_err = "Post not found"
        try:
            api_err = resp.json().get("message", api_err)
        except Exception:
            pass
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        return jsonify({"error": f"Failed to reach LinkedIn API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/archive-carousel-linkedin-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_linkedin_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"linkedin_{component}_post_id_{article_id}"
    result_key  = f"linkedin_{component}_result_{article_id}"
    status_key  = f"linkedin_{component}_status_{article_id}"
    history_key = f"linkedin_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-linkedin-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_linkedin_post(article_id):
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"linkedin_{component}_post_id_{article_id}"
    result_key  = f"linkedin_{component}_result_{article_id}"
    status_key  = f"linkedin_{component}_status_{article_id}"
    history_key = f"linkedin_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        resp = _req.delete(f"{_LINKEDIN_API_URL}/posts/{post_id}", headers={
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "LinkedIn-Version": "202502", "X-Restli-Protocol-Version": "2.0.0",
        }, timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete post: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/post-narrated-to-linkedin", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_linkedin(article_id):
    if not LINKEDIN_CONFIGURED:
        return jsonify({"error": "LinkedIn credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"linkedin_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_linkedin_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-linkedin-status")
@login_required
@admin_required
def admin_narrated_linkedin_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"linkedin_narrated_status_{article_id}_{run_ts}"
    result_key  = f"linkedin_narrated_result_{article_id}_{run_ts}"
    history_key = f"linkedin_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/check-narrated-linkedin-post")
@login_required
@admin_required
def admin_check_narrated_linkedin_post(article_id):
    import requests as _req
    if not LINKEDIN_CONFIGURED:
        return jsonify({"error": "LinkedIn credentials not configured."}), 400
    run_ts   = request.args.get("ts", "0")
    post_key = f"linkedin_narrated_post_id_{article_id}_{run_ts}"
    post_id  = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No published post found."}), 400
    try:
        resp = _req.get(f"{_LINKEDIN_API_URL}/posts/{post_id}", headers={
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "LinkedIn-Version": "202502", "X-Restli-Protocol-Version": "2.0.0",
        }, timeout=15)
        if resp.status_code == 200:
            return jsonify({"live": True})
        api_err = "Post not found"
        try:
            api_err = resp.json().get("message", api_err)
        except Exception:
            pass
        return jsonify({"live": False, "api_error": api_err})
    except Exception as exc:
        return jsonify({"error": f"Failed to reach LinkedIn API: {exc}"}), 500


@app.route("/admin/articles/<int:article_id>/archive-narrated-linkedin-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_linkedin_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"linkedin_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"linkedin_narrated_result_{article_id}_{run_ts}"
    status_key  = f"linkedin_narrated_status_{article_id}_{run_ts}"
    history_key = f"linkedin_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-linkedin-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_linkedin_post(article_id):
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"linkedin_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"linkedin_narrated_result_{article_id}_{run_ts}"
    status_key  = f"linkedin_narrated_status_{article_id}_{run_ts}"
    history_key = f"linkedin_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        resp = _req.delete(f"{_LINKEDIN_API_URL}/posts/{post_id}", headers={
            "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
            "LinkedIn-Version": "202502", "X-Restli-Protocol-Version": "2.0.0",
        }, timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete post: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ══════════════════════════════════════════════════════════════
# BLUESKY (AT Protocol): Workers & Routes
# ══════════════════════════════════════════════════════════════

def _bluesky_auth():
    """Authenticate with Bluesky and return (jwt, did)."""
    import requests as _req
    resp = _req.post("https://bsky.social/xrpc/com.atproto.server.createSession", json={
        "identifier": BLUESKY_HANDLE,
        "password":   BLUESKY_APP_PASSWORD,
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data["accessJwt"], data["did"]


def _post_carousel_bluesky_worker(article_id, caption, component):
    """Background thread: compose carousel images and post to Bluesky."""
    # UTM tracking: replace plain article URL with Bluesky-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'bluesky', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"bluesky_{component}_status_{article_id}"
    result_key = f"bluesky_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

        # Authenticate
        jwt, did = _bluesky_auth()

        # 1. Compose, upload images as blobs
        _set_status(f"running:compose:0/{n}")
        ts_t = int(_time_mod.time())
        blobs = []

        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            # Also upload to GCS for archival
            gcs_obj    = f"articles/{article_id}/bluesky/{component}_{idx+1}_{ts_t}.jpg"
            upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")

            # Upload blob to Bluesky
            _set_status(f"running:upload:{idx+1}/{n}")
            resp = _req.post("https://bsky.social/xrpc/com.atproto.repo.uploadBlob",
                             headers={"Authorization": f"Bearer {jwt}",
                                      "Content-Type": "image/jpeg"},
                             data=jpeg_bytes, timeout=30)
            resp.raise_for_status()
            blob_data = resp.json().get("blob")
            if not blob_data:
                _set_result(f"error:Blob upload failed at {idx+1}: {resp.json()}")
                return
            blobs.append(blob_data)
            print(f"[Bluesky] Blob uploaded {idx+1}/{n}", flush=True)

        # 2. Create post(s) — max 4 images per post, thread for overflow
        _set_status("running:publish")
        from datetime import datetime, timezone
        iso_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        first_uri = None
        first_cid = None
        first_rkey = None
        parent_uri = None
        parent_cid = None

        blob_chunks = [blobs[i:i+4] for i in range(0, len(blobs), 4)]
        for chunk_idx, chunk in enumerate(blob_chunks):
            images_embed = [{"alt": "", "image": b} for b in chunk]
            record = {
                "$type": "app.bsky.feed.post",
                "text": caption[:300] if caption and chunk_idx == 0 else "",
                "createdAt": iso_now,
                "embed": {
                    "$type": "app.bsky.embed.images",
                    "images": images_embed,
                },
            }
            if parent_uri and parent_cid:
                record["reply"] = {
                    "root":   {"uri": first_uri, "cid": first_cid},
                    "parent": {"uri": parent_uri, "cid": parent_cid},
                }
            resp = _req.post("https://bsky.social/xrpc/com.atproto.repo.createRecord", json={
                "repo": did,
                "collection": "app.bsky.feed.post",
                "record": record,
            }, headers={"Authorization": f"Bearer {jwt}"}, timeout=30)
            resp.raise_for_status()
            rec_data = resp.json()
            uri = rec_data.get("uri", "")
            cid = rec_data.get("cid", "")
            if chunk_idx == 0:
                first_uri = uri
                first_cid = cid
                first_rkey = uri.split("/")[-1] if "/" in uri else ""
            parent_uri = uri
            parent_cid = cid
            print(f"[Bluesky] Post created (chunk {chunk_idx+1}/{len(blob_chunks)}): {uri}", flush=True)

        permalink = f"https://bsky.app/profile/{BLUESKY_HANDLE}/post/{first_rkey}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"bluesky_{component}_post_id_{article_id}", first_rkey, first_rkey))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"bluesky_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"Bluesky {component.title()} Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"rkey={first_rkey}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'bluesky', component, first_rkey, permalink, caption)
        except Exception:
            pass
        print(f"[Bluesky] Carousel posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Bluesky] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Bluesky {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_bluesky_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to Bluesky."""
    # UTM tracking: replace plain article URL with Bluesky-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'bluesky', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"bluesky_narrated_status_{article_id}_{run_ts}"
    result_key = f"bluesky_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        jwt, did = _bluesky_auth()

        # 1. Download video
        _set_status("running:download")
        resp_dl = _req.get(video_url, timeout=120)
        resp_dl.raise_for_status()
        video_bytes = resp_dl.content

        # 2. Upload video to Bluesky video service
        _set_status("running:upload")
        resp = _req.post(
            f"https://video.bsky.app/xrpc/app.bsky.video.uploadVideo?did={did}&name=video.mp4",
            headers={"Authorization": f"Bearer {jwt}",
                     "Content-Type": "video/mp4"},
            data=video_bytes, timeout=120)
        resp.raise_for_status()
        job_data = resp.json()
        job_id = job_data.get("jobId", "")
        if not job_id:
            _set_result(f"error:Video upload failed: {job_data}")
            return
        print(f"[Bluesky] Video upload job: {job_id}", flush=True)

        # 3. Poll job status
        _set_status("running:poll")
        blob_ref = None
        for attempt in range(60):
            _time_mod.sleep(5)
            resp = _req.get("https://video.bsky.app/xrpc/app.bsky.video.getJobStatus",
                            params={"jobId": job_id},
                            headers={"Authorization": f"Bearer {jwt}"},
                            timeout=15)
            status_data = resp.json()
            job_state = status_data.get("jobStatus", {}).get("state", "")
            print(f"[Bluesky] Video job state: {job_state}", flush=True)
            if job_state == "JOB_STATE_COMPLETED":
                blob_ref = status_data.get("jobStatus", {}).get("blob")
                break
            if job_state == "JOB_STATE_FAILED":
                err_msg = status_data.get("jobStatus", {}).get("error", "Processing failed")
                _set_result(f"error:{err_msg}")
                return
        else:
            _set_result("error:Video processing timed out after 5 minutes")
            return

        if not blob_ref:
            _set_result("error:No blob reference returned")
            return

        # 4. Create post with video embed
        _set_status("running:publish")
        from datetime import datetime, timezone
        iso_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        record = {
            "$type": "app.bsky.feed.post",
            "text": caption[:300] if caption else "",
            "createdAt": iso_now,
            "embed": {
                "$type": "app.bsky.embed.video",
                "video": blob_ref,
            },
        }
        resp = _req.post("https://bsky.social/xrpc/com.atproto.repo.createRecord", json={
            "repo": did,
            "collection": "app.bsky.feed.post",
            "record": record,
        }, headers={"Authorization": f"Bearer {jwt}"}, timeout=30)
        resp.raise_for_status()
        rec_data = resp.json()
        uri  = rec_data.get("uri", "")
        rkey = uri.split("/")[-1] if "/" in uri else ""

        permalink = f"https://bsky.app/profile/{BLUESKY_HANDLE}/post/{rkey}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"bluesky_narrated_post_id_{article_id}_{run_ts}", rkey, rkey))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"bluesky_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "Bluesky Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"rkey={rkey}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'bluesky', 'narrated', rkey, permalink, caption)
        except Exception:
            pass
        print(f"[Bluesky] Narrated video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Bluesky] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Bluesky Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── Bluesky: Post carousel ──────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-bluesky", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_bluesky(article_id):
    if not BLUESKY_CONFIGURED:
        return jsonify({"error": "Bluesky credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"bluesky_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_bluesky_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-bluesky-status")
@login_required
@admin_required
def admin_carousel_bluesky_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"bluesky_{component}_status_{article_id}"
    result_key  = f"bluesky_{component}_result_{article_id}"
    history_key = f"bluesky_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-carousel-bluesky-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_bluesky_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"bluesky_{component}_post_id_{article_id}"
    result_key  = f"bluesky_{component}_result_{article_id}"
    status_key  = f"bluesky_{component}_status_{article_id}"
    history_key = f"bluesky_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-bluesky-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_bluesky_post(article_id):
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"bluesky_{component}_post_id_{article_id}"
    result_key  = f"bluesky_{component}_result_{article_id}"
    status_key  = f"bluesky_{component}_status_{article_id}"
    history_key = f"bluesky_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        jwt, did = _bluesky_auth()
        resp = _req.post("https://bsky.social/xrpc/com.atproto.repo.deleteRecord", json={
            "repo": did,
            "collection": "app.bsky.feed.post",
            "rkey": post_id,
        }, headers={"Authorization": f"Bearer {jwt}"}, timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete post: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ── Bluesky: Post narrated video ─────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-bluesky", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_bluesky(article_id):
    if not BLUESKY_CONFIGURED:
        return jsonify({"error": "Bluesky credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"bluesky_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_bluesky_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-bluesky-status")
@login_required
@admin_required
def admin_narrated_bluesky_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"bluesky_narrated_status_{article_id}_{run_ts}"
    result_key  = f"bluesky_narrated_result_{article_id}_{run_ts}"
    history_key = f"bluesky_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-narrated-bluesky-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_bluesky_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"bluesky_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"bluesky_narrated_result_{article_id}_{run_ts}"
    status_key  = f"bluesky_narrated_status_{article_id}_{run_ts}"
    history_key = f"bluesky_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-bluesky-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_bluesky_post(article_id):
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"bluesky_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"bluesky_narrated_result_{article_id}_{run_ts}"
    status_key  = f"bluesky_narrated_status_{article_id}_{run_ts}"
    history_key = f"bluesky_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        jwt, did = _bluesky_auth()
        resp = _req.post("https://bsky.social/xrpc/com.atproto.repo.deleteRecord", json={
            "repo": did,
            "collection": "app.bsky.feed.post",
            "rkey": post_id,
        }, headers={"Authorization": f"Bearer {jwt}"}, timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete post: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ══════════════════════════════════════════════════════════════
# REDDIT: Workers & Routes
# ══════════════════════════════════════════════════════════════

def _reddit_get_access_token():
    """Get a fresh Reddit access token using the refresh token."""
    import requests as _req
    resp = _req.post("https://www.reddit.com/api/v1/access_token",
                     auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
                     data={"grant_type": "refresh_token", "refresh_token": REDDIT_REFRESH_TOKEN},
                     headers={"User-Agent": "ObeliskStamps/1.0"},
                     timeout=15)
    resp.raise_for_status()
    return resp.json()["access_token"]


def _post_carousel_reddit_worker(article_id, caption, component):
    """Background thread: compose carousel images and post to Reddit as self post."""
    # UTM tracking: replace plain article URL with Reddit-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'reddit', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"reddit_{component}_status_{article_id}"
    result_key = f"reddit_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

        # 1. Compose and upload images to GCS
        _set_status(f"running:compose:0/{n}")
        ts_t = int(_time_mod.time())
        public_urls = []

        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            gcs_obj    = f"articles/{article_id}/reddit/{component}_{idx+1}_{ts_t}.jpg"
            public_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
            if not public_url:
                _set_result(f"error:Image upload failed for slide {idx+1}")
                return
            public_urls.append(public_url)

        # 2. Post to Reddit as self post with image links
        _set_status("running:publish")
        token = _reddit_get_access_token()
        title = caption[:300] if caption else "New post"
        body  = caption + "\n\n" if caption else ""
        for i, url in enumerate(public_urls):
            body += f"![Slide {i+1}]({url})\n\n"

        resp = _req.post("https://oauth.reddit.com/api/submit",
                         headers={"Authorization": f"Bearer {token}",
                                  "User-Agent": "ObeliskStamps/1.0"},
                         data={"kind": "self", "sr": REDDIT_SUBREDDIT,
                               "title": title, "text": body},
                         timeout=30)
        resp.raise_for_status()
        rdata = resp.json()
        post_url  = rdata.get("json", {}).get("data", {}).get("url", "")
        post_name = rdata.get("json", {}).get("data", {}).get("name", "")
        if not post_url:
            _set_result(f"error:Reddit submit failed: {rdata}")
            return

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"reddit_{component}_post_id_{article_id}", post_name, post_name))
        _set_result(f"done:{post_url}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"reddit_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"Reddit {component.title()} Posted",
                          f"<a href=\"{post_url}\" target=\"_blank\">{post_url}</a>\n"
                          f"fullname={post_name}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'reddit', component, post_name, post_url, caption)
        except Exception:
            pass
        print(f"[Reddit] Carousel posted: {post_url}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Reddit] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Reddit {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_reddit_worker(article_id, video_url, caption, run_ts):
    """Background thread: post narrated video to Reddit as link post."""
    # UTM tracking: replace plain article URL with Reddit-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'reddit', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"reddit_narrated_status_{article_id}_{run_ts}"
    result_key = f"reddit_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        _set_status("running:publish")
        token = _reddit_get_access_token()
        title = caption[:300] if caption else "New video"

        resp = _req.post("https://oauth.reddit.com/api/submit",
                         headers={"Authorization": f"Bearer {token}",
                                  "User-Agent": "ObeliskStamps/1.0"},
                         data={"kind": "link", "sr": REDDIT_SUBREDDIT,
                               "title": title, "url": video_url},
                         timeout=30)
        resp.raise_for_status()
        rdata = resp.json()
        post_url  = rdata.get("json", {}).get("data", {}).get("url", "")
        post_name = rdata.get("json", {}).get("data", {}).get("name", "")
        if not post_url:
            _set_result(f"error:Reddit submit failed: {rdata}")
            return

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"reddit_narrated_post_id_{article_id}_{run_ts}", post_name, post_name))
        _set_result(f"done:{post_url}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"reddit_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "Reddit Narrated Video Posted",
                          f"<a href=\"{post_url}\" target=\"_blank\">{post_url}</a>\n"
                          f"fullname={post_name}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'reddit', 'narrated', post_name, post_url, caption)
        except Exception:
            pass
        print(f"[Reddit] Narrated video posted: {post_url}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Reddit] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Reddit Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── Reddit: Post carousel ────────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-reddit", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_reddit(article_id):
    if not REDDIT_CONFIGURED:
        return jsonify({"error": "Reddit credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"reddit_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_reddit_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-reddit-status")
@login_required
@admin_required
def admin_carousel_reddit_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"reddit_{component}_status_{article_id}"
    result_key  = f"reddit_{component}_result_{article_id}"
    history_key = f"reddit_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-carousel-reddit-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_reddit_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"reddit_{component}_post_id_{article_id}"
    result_key  = f"reddit_{component}_result_{article_id}"
    status_key  = f"reddit_{component}_status_{article_id}"
    history_key = f"reddit_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-reddit-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_reddit_post(article_id):
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"reddit_{component}_post_id_{article_id}"
    result_key  = f"reddit_{component}_result_{article_id}"
    status_key  = f"reddit_{component}_status_{article_id}"
    history_key = f"reddit_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        token = _reddit_get_access_token()
        resp = _req.post("https://oauth.reddit.com/api/del",
                         headers={"Authorization": f"Bearer {token}",
                                  "User-Agent": "ObeliskStamps/1.0"},
                         data={"id": post_id}, timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete post: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ── Reddit: Post narrated video ──────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-reddit", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_reddit(article_id):
    if not REDDIT_CONFIGURED:
        return jsonify({"error": "Reddit credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"reddit_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_reddit_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-reddit-status")
@login_required
@admin_required
def admin_narrated_reddit_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"reddit_narrated_status_{article_id}_{run_ts}"
    result_key  = f"reddit_narrated_result_{article_id}_{run_ts}"
    history_key = f"reddit_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-narrated-reddit-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_reddit_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"reddit_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"reddit_narrated_result_{article_id}_{run_ts}"
    status_key  = f"reddit_narrated_status_{article_id}_{run_ts}"
    history_key = f"reddit_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-reddit-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_reddit_post(article_id):
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"reddit_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"reddit_narrated_result_{article_id}_{run_ts}"
    status_key  = f"reddit_narrated_status_{article_id}_{run_ts}"
    history_key = f"reddit_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        token = _reddit_get_access_token()
        resp = _req.post("https://oauth.reddit.com/api/del",
                         headers={"Authorization": f"Bearer {token}",
                                  "User-Agent": "ObeliskStamps/1.0"},
                         data={"id": post_id}, timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete post: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ══════════════════════════════════════════════════════════════
# TELEGRAM: Workers & Routes
# ══════════════════════════════════════════════════════════════

def _post_carousel_telegram_worker(article_id, caption, component):
    """Background thread: compose carousel images and post as Telegram media group."""
    # UTM tracking: replace plain article URL with Telegram-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'telegram', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"telegram_{component}_status_{article_id}"
    result_key = f"telegram_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])

        # 1. Compose and upload images to GCS
        _set_status(f"running:compose:0/{n}")
        ts_t = int(_time_mod.time())
        public_urls = []

        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            gcs_obj    = f"articles/{article_id}/telegram/{component}_{idx+1}_{ts_t}.jpg"
            public_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
            if not public_url:
                _set_result(f"error:Image upload failed for slide {idx+1}")
                return
            public_urls.append(public_url)

        # 2. Send as media group (max 10 photos)
        _set_status("running:publish")
        media = []
        for i, url in enumerate(public_urls):
            item = {"type": "photo", "media": url}
            if i == 0 and caption:
                item["caption"] = caption[:1024]
            media.append(item)

        resp = _req.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMediaGroup",
                         json={"chat_id": TELEGRAM_CHAT_ID, "media": media},
                         timeout=60)
        resp.raise_for_status()
        rdata = resp.json()
        if not rdata.get("ok"):
            _set_result(f"error:Telegram API error: {rdata.get('description', '')}")
            return

        # Store all message_ids for deletion
        results = rdata.get("result", [])
        msg_ids = [str(m.get("message_id", "")) for m in results if m.get("message_id")]
        msg_ids_str = json.dumps(msg_ids)
        first_msg_id = msg_ids[0] if msg_ids else ""

        # Telegram doesn't have public permalinks for bot messages in private chats
        permalink = f"telegram:chat={TELEGRAM_CHAT_ID}:msg={first_msg_id}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"telegram_{component}_post_id_{article_id}", msg_ids_str, msg_ids_str))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"telegram_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"Telegram {component.title()} Posted",
                          f"chat_id={TELEGRAM_CHAT_ID}\nmessage_ids={msg_ids_str}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'telegram', component, first_msg_id, permalink, caption)
        except Exception:
            pass
        print(f"[Telegram] Media group posted: {msg_ids_str}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Telegram] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Telegram {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_telegram_worker(article_id, video_url, caption, run_ts):
    """Background thread: post narrated video to Telegram."""
    # UTM tracking: replace plain article URL with Telegram-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'telegram', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"telegram_narrated_status_{article_id}_{run_ts}"
    result_key = f"telegram_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        _set_status("running:publish")
        resp = _req.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo",
                         json={"chat_id": TELEGRAM_CHAT_ID,
                               "video": video_url,
                               "caption": caption[:1024] if caption else ""},
                         timeout=120)
        resp.raise_for_status()
        rdata = resp.json()
        if not rdata.get("ok"):
            _set_result(f"error:Telegram API error: {rdata.get('description', '')}")
            return

        msg_id = str(rdata.get("result", {}).get("message_id", ""))
        permalink = f"telegram:chat={TELEGRAM_CHAT_ID}:msg={msg_id}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"telegram_narrated_post_id_{article_id}_{run_ts}", msg_id, msg_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"telegram_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "Telegram Narrated Video Posted",
                          f"chat_id={TELEGRAM_CHAT_ID}\nmessage_id={msg_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'telegram', 'narrated', msg_id, permalink, caption)
        except Exception:
            pass
        print(f"[Telegram] Narrated video posted: msg_id={msg_id}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Telegram] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Telegram Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── Telegram: Post carousel ──────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-telegram", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_telegram(article_id):
    if not TELEGRAM_CONFIGURED:
        return jsonify({"error": "Telegram credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"telegram_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_telegram_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-telegram-status")
@login_required
@admin_required
def admin_carousel_telegram_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"telegram_{component}_status_{article_id}"
    result_key  = f"telegram_{component}_result_{article_id}"
    history_key = f"telegram_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-carousel-telegram-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_telegram_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"telegram_{component}_post_id_{article_id}"
    result_key  = f"telegram_{component}_result_{article_id}"
    status_key  = f"telegram_{component}_status_{article_id}"
    history_key = f"telegram_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-telegram-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_telegram_post(article_id):
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"telegram_{component}_post_id_{article_id}"
    result_key  = f"telegram_{component}_result_{article_id}"
    status_key  = f"telegram_{component}_status_{article_id}"
    history_key = f"telegram_{component}_history_{article_id}"
    post_id_raw = get_setting(post_key)
    if not post_id_raw:
        return jsonify({"error": "No post record found"}), 400
    try:
        # post_id_raw is JSON array of message_ids for media groups
        msg_ids = json.loads(post_id_raw)
        if not isinstance(msg_ids, list):
            msg_ids = [str(post_id_raw)]
    except (json.JSONDecodeError, TypeError):
        msg_ids = [str(post_id_raw)]
    try:
        for mid in msg_ids:
            _req.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "message_id": int(mid)},
                      timeout=15)
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id_raw, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ── Telegram: Post narrated video ────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-telegram", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_telegram(article_id):
    if not TELEGRAM_CONFIGURED:
        return jsonify({"error": "Telegram credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"telegram_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_telegram_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-telegram-status")
@login_required
@admin_required
def admin_narrated_telegram_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"telegram_narrated_status_{article_id}_{run_ts}"
    result_key  = f"telegram_narrated_result_{article_id}_{run_ts}"
    history_key = f"telegram_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-narrated-telegram-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_telegram_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"telegram_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"telegram_narrated_result_{article_id}_{run_ts}"
    status_key  = f"telegram_narrated_status_{article_id}_{run_ts}"
    history_key = f"telegram_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-telegram-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_telegram_post(article_id):
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"telegram_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"telegram_narrated_result_{article_id}_{run_ts}"
    status_key  = f"telegram_narrated_status_{article_id}_{run_ts}"
    history_key = f"telegram_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        _req.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteMessage",
                  json={"chat_id": TELEGRAM_CHAT_ID, "message_id": int(post_id)},
                  timeout=15)
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ══════════════════════════════════════════════════════════════
# VIMEO: Workers & Routes (Narrated Video ONLY)
# ══════════════════════════════════════════════════════════════

def _post_narrated_vimeo_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to Vimeo via tus protocol."""
    # UTM tracking: replace plain article URL with Vimeo-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'vimeo', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"vimeo_narrated_status_{article_id}_{run_ts}"
    result_key = f"vimeo_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        # 1. Download video
        _set_status("running:download")
        resp_dl = _req.get(video_url, timeout=120)
        resp_dl.raise_for_status()
        video_bytes = resp_dl.content
        file_size = len(video_bytes)

        vimeo_headers = {
            "Authorization": f"Bearer {VIMEO_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.vimeo.*+json;version=3.4",
        }

        # 2. Create video on Vimeo
        _set_status("running:create")
        title = caption[:128] if caption else "Narrated Video"
        resp = _req.post("https://api.vimeo.com/me/videos", headers=vimeo_headers, json={
            "upload": {"approach": "tus", "size": str(file_size)},
            "name": title,
            "description": caption[:5000] if caption else "",
            "privacy": {"view": "anybody"},
        }, timeout=30)
        resp.raise_for_status()
        vdata = resp.json()
        upload_link = vdata.get("upload", {}).get("upload_link", "")
        video_uri   = vdata.get("uri", "")
        video_id    = video_uri.split("/")[-1] if "/" in video_uri else ""
        if not upload_link or not video_id:
            _set_result(f"error:Vimeo video creation failed: {vdata}")
            return
        print(f"[Vimeo] Video created: {video_uri}", flush=True)

        # 3. Upload via tus
        _set_status("running:upload")
        resp = _req.patch(upload_link, headers={
            "Content-Type": "application/offset+octet-stream",
            "Upload-Offset": "0",
            "Tus-Resumable": "1.0.0",
        }, data=video_bytes, timeout=300)
        if resp.status_code not in (200, 204):
            _set_result(f"error:Vimeo upload failed: {resp.status_code}")
            return
        print(f"[Vimeo] Upload complete", flush=True)

        # 4. Poll for transcode completion
        _set_status("running:transcode")
        for attempt in range(120):
            _time_mod.sleep(5)
            resp = _req.get(f"https://api.vimeo.com/videos/{video_id}?fields=transcode.status",
                            headers={"Authorization": f"Bearer {VIMEO_ACCESS_TOKEN}",
                                     "Accept": "application/vnd.vimeo.*+json;version=3.4"},
                            timeout=15)
            if resp.status_code == 200:
                t_status = resp.json().get("transcode", {}).get("status", "")
                print(f"[Vimeo] Transcode status: {t_status}", flush=True)
                if t_status == "complete":
                    break
                if t_status == "error":
                    _set_result("error:Vimeo transcode failed")
                    return
        else:
            _set_result("error:Vimeo transcode timed out")
            return

        permalink = f"https://vimeo.com/{video_id}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"vimeo_narrated_post_id_{article_id}_{run_ts}", video_id, video_id))
        _set_result(f"done:{permalink}")
        _add_activity_log(article_id, "Vimeo Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"video_id={video_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'vimeo', 'narrated', video_id, permalink, caption)
        except Exception:
            pass
        print(f"[Vimeo] Narrated video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Vimeo] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Vimeo Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── Vimeo: Post narrated video ───────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-vimeo", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_vimeo(article_id):
    if not VIMEO_CONFIGURED:
        return jsonify({"error": "Vimeo credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"vimeo_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_vimeo_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-vimeo-status")
@login_required
@admin_required
def admin_narrated_vimeo_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"vimeo_narrated_status_{article_id}_{run_ts}"
    result_key  = f"vimeo_narrated_result_{article_id}_{run_ts}"
    history_key = f"vimeo_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-narrated-vimeo-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_vimeo_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"vimeo_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"vimeo_narrated_result_{article_id}_{run_ts}"
    status_key  = f"vimeo_narrated_status_{article_id}_{run_ts}"
    history_key = f"vimeo_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-vimeo-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_vimeo_post(article_id):
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"vimeo_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"vimeo_narrated_result_{article_id}_{run_ts}"
    status_key  = f"vimeo_narrated_status_{article_id}_{run_ts}"
    history_key = f"vimeo_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        resp = _req.delete(f"https://api.vimeo.com/videos/{post_id}",
                           headers={"Authorization": f"Bearer {VIMEO_ACCESS_TOKEN}"},
                           timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete video: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ══════════════════════════════════════════════════════════════
# MASTODON: Workers & Routes
# ══════════════════════════════════════════════════════════════

def _post_carousel_mastodon_worker(article_id, caption, component):
    """Background thread: compose carousel images and post to Mastodon."""
    # UTM tracking: replace plain article URL with Mastodon-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'mastodon', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"mastodon_{component}_status_{article_id}"
    result_key = f"mastodon_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])
        masto_headers = {"Authorization": f"Bearer {MASTODON_ACCESS_TOKEN}"}

        # 1. Compose and upload images
        _set_status(f"running:compose:0/{n}")
        ts_t = int(_time_mod.time())
        media_ids = []

        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            # Also upload to GCS
            gcs_obj    = f"articles/{article_id}/mastodon/{component}_{idx+1}_{ts_t}.jpg"
            upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")

            # Upload to Mastodon
            _set_status(f"running:upload:{idx+1}/{n}")
            resp = _req.post(f"{MASTODON_INSTANCE_URL}/api/v2/media",
                             headers=masto_headers,
                             files={"file": (f"slide_{idx+1}.jpg", jpeg_bytes, "image/jpeg")},
                             timeout=60)
            resp.raise_for_status()
            mid = resp.json().get("id")
            if not mid:
                _set_result(f"error:Media upload failed at {idx+1}: {resp.json()}")
                return
            media_ids.append(mid)
            print(f"[Mastodon] Media uploaded {idx+1}/{n}: {mid}", flush=True)

        # 2. Create status(es) — max 4 media per status, thread for overflow
        _set_status("running:publish")
        first_status_id = None
        first_status_url = None
        parent_id = None

        id_chunks = [media_ids[i:i+4] for i in range(0, len(media_ids), 4)]
        for chunk_idx, chunk in enumerate(id_chunks):
            payload = {
                "status": caption[:500] if caption and chunk_idx == 0 else "",
                "media_ids": chunk,
                "visibility": "public",
            }
            if parent_id:
                payload["in_reply_to_id"] = parent_id

            resp = _req.post(f"{MASTODON_INSTANCE_URL}/api/v1/statuses",
                             headers={**masto_headers, "Content-Type": "application/json"},
                             json=payload, timeout=30)
            resp.raise_for_status()
            sdata = resp.json()
            sid = sdata.get("id", "")
            surl = sdata.get("url", "")
            if chunk_idx == 0:
                first_status_id = sid
                first_status_url = surl
            parent_id = sid
            print(f"[Mastodon] Status created (chunk {chunk_idx+1}): {surl}", flush=True)

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"mastodon_{component}_post_id_{article_id}", first_status_id, first_status_id))
        _set_result(f"done:{first_status_url}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"mastodon_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"Mastodon {component.title()} Posted",
                          f"<a href=\"{first_status_url}\" target=\"_blank\">{first_status_url}</a>\n"
                          f"status_id={first_status_id}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'mastodon', component, first_status_id, first_status_url, caption)
        except Exception:
            pass
        print(f"[Mastodon] Carousel posted: {first_status_url}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Mastodon] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Mastodon {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_mastodon_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to Mastodon."""
    # UTM tracking: replace plain article URL with Mastodon-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'mastodon', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"mastodon_narrated_status_{article_id}_{run_ts}"
    result_key = f"mastodon_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        masto_headers = {"Authorization": f"Bearer {MASTODON_ACCESS_TOKEN}"}

        # 1. Download video
        _set_status("running:download")
        resp_dl = _req.get(video_url, timeout=120)
        resp_dl.raise_for_status()
        video_bytes = resp_dl.content

        # 2. Upload video to Mastodon
        _set_status("running:upload")
        resp = _req.post(f"{MASTODON_INSTANCE_URL}/api/v2/media",
                         headers=masto_headers,
                         files={"file": ("video.mp4", video_bytes, "video/mp4")},
                         timeout=120)
        resp.raise_for_status()
        media_data = resp.json()
        media_id = media_data.get("id")
        if not media_id:
            _set_result(f"error:Media upload failed: {media_data}")
            return
        print(f"[Mastodon] Video uploaded: {media_id}", flush=True)

        # 3. Poll until media is processed (url becomes non-null)
        _set_status("running:poll")
        for attempt in range(60):
            _time_mod.sleep(5)
            resp = _req.get(f"{MASTODON_INSTANCE_URL}/api/v1/media/{media_id}",
                            headers=masto_headers, timeout=15)
            if resp.status_code == 200:
                mdata = resp.json()
                if mdata.get("url"):
                    break
            elif resp.status_code == 206:
                # Still processing
                continue
        else:
            _set_result("error:Video processing timed out")
            return

        # 4. Create status
        _set_status("running:publish")
        resp = _req.post(f"{MASTODON_INSTANCE_URL}/api/v1/statuses",
                         headers={**masto_headers, "Content-Type": "application/json"},
                         json={"status": caption[:500] if caption else "",
                               "media_ids": [media_id],
                               "visibility": "public"},
                         timeout=30)
        resp.raise_for_status()
        sdata = resp.json()
        status_id = sdata.get("id", "")
        status_url = sdata.get("url", "")

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"mastodon_narrated_post_id_{article_id}_{run_ts}", status_id, status_id))
        _set_result(f"done:{status_url}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"mastodon_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "Mastodon Narrated Video Posted",
                          f"<a href=\"{status_url}\" target=\"_blank\">{status_url}</a>\n"
                          f"status_id={status_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'mastodon', 'narrated', status_id, status_url, caption)
        except Exception:
            pass
        print(f"[Mastodon] Narrated video posted: {status_url}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Mastodon] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Mastodon Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── Mastodon: Post carousel ──────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-mastodon", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_mastodon(article_id):
    if not MASTODON_CONFIGURED:
        return jsonify({"error": "Mastodon credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"mastodon_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_mastodon_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-mastodon-status")
@login_required
@admin_required
def admin_carousel_mastodon_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"mastodon_{component}_status_{article_id}"
    result_key  = f"mastodon_{component}_result_{article_id}"
    history_key = f"mastodon_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-carousel-mastodon-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_mastodon_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"mastodon_{component}_post_id_{article_id}"
    result_key  = f"mastodon_{component}_result_{article_id}"
    status_key  = f"mastodon_{component}_status_{article_id}"
    history_key = f"mastodon_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-mastodon-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_mastodon_post(article_id):
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"mastodon_{component}_post_id_{article_id}"
    result_key  = f"mastodon_{component}_result_{article_id}"
    status_key  = f"mastodon_{component}_status_{article_id}"
    history_key = f"mastodon_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        resp = _req.delete(f"{MASTODON_INSTANCE_URL}/api/v1/statuses/{post_id}",
                           headers={"Authorization": f"Bearer {MASTODON_ACCESS_TOKEN}"},
                           timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete status: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ── Mastodon: Post narrated video ────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-mastodon", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_mastodon(article_id):
    if not MASTODON_CONFIGURED:
        return jsonify({"error": "Mastodon credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"mastodon_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_mastodon_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-mastodon-status")
@login_required
@admin_required
def admin_narrated_mastodon_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"mastodon_narrated_status_{article_id}_{run_ts}"
    result_key  = f"mastodon_narrated_result_{article_id}_{run_ts}"
    history_key = f"mastodon_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-narrated-mastodon-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_mastodon_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"mastodon_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"mastodon_narrated_result_{article_id}_{run_ts}"
    status_key  = f"mastodon_narrated_status_{article_id}_{run_ts}"
    history_key = f"mastodon_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-mastodon-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_mastodon_post(article_id):
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"mastodon_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"mastodon_narrated_result_{article_id}_{run_ts}"
    status_key  = f"mastodon_narrated_status_{article_id}_{run_ts}"
    history_key = f"mastodon_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        resp = _req.delete(f"{MASTODON_INSTANCE_URL}/api/v1/statuses/{post_id}",
                           headers={"Authorization": f"Bearer {MASTODON_ACCESS_TOKEN}"},
                           timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete status: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ══════════════════════════════════════════════════════════════
# VK (VKontakte): Workers & Routes
# ══════════════════════════════════════════════════════════════

def _post_carousel_vk_worker(article_id, caption, component):
    """Background thread: compose carousel images and post to VK wall."""
    # UTM tracking: replace plain article URL with VK-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'vk', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"vk_{component}_status_{article_id}"
    result_key = f"vk_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])
        group_id = str(abs(int(VK_OWNER_ID)))

        # 1. Get upload server
        _set_status(f"running:compose:0/{n}")
        resp = _req.get("https://api.vk.com/method/photos.getWallUploadServer", params={
            "access_token": VK_ACCESS_TOKEN, "v": "5.199", "group_id": group_id,
        }, timeout=15)
        resp.raise_for_status()
        upload_url = resp.json().get("response", {}).get("upload_url", "")
        if not upload_url:
            _set_result(f"error:Could not get VK upload server: {resp.json()}")
            return

        ts_t = int(_time_mod.time())
        attachments = []

        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            # Also GCS
            gcs_obj = f"articles/{article_id}/vk/{component}_{idx+1}_{ts_t}.jpg"
            upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")

            # Upload to VK
            _set_status(f"running:upload:{idx+1}/{n}")
            resp = _req.post(upload_url,
                             files={"photo": (f"slide_{idx+1}.jpg", jpeg_bytes, "image/jpeg")},
                             timeout=30)
            resp.raise_for_status()
            udata = resp.json()
            server = udata.get("server", "")
            photo  = udata.get("photo", "")
            hash_  = udata.get("hash", "")

            # Save photo
            resp = _req.get("https://api.vk.com/method/photos.saveWallPhoto", params={
                "access_token": VK_ACCESS_TOKEN, "v": "5.199", "group_id": group_id,
                "server": server, "photo": photo, "hash": hash_,
            }, timeout=15)
            resp.raise_for_status()
            saved = resp.json().get("response", [])
            if not saved:
                _set_result(f"error:VK photo save failed at {idx+1}: {resp.json()}")
                return
            oid = saved[0].get("owner_id")
            pid = saved[0].get("id")
            attachments.append(f"photo{oid}_{pid}")
            print(f"[VK] Photo saved {idx+1}/{n}: photo{oid}_{pid}", flush=True)

        # 2. Create wall post
        _set_status("running:publish")
        resp = _req.get("https://api.vk.com/method/wall.post", params={
            "access_token": VK_ACCESS_TOKEN, "v": "5.199",
            "owner_id": VK_OWNER_ID,
            "message": caption[:10000] if caption else "",
            "attachments": ",".join(attachments),
        }, timeout=30)
        resp.raise_for_status()
        wdata = resp.json()
        post_id = str(wdata.get("response", {}).get("post_id", ""))
        if not post_id:
            _set_result(f"error:VK wall.post failed: {wdata}")
            return

        permalink = f"https://vk.com/wall{VK_OWNER_ID}_{post_id}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"vk_{component}_post_id_{article_id}", post_id, post_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"vk_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"VK {component.title()} Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"post_id={post_id}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'vk', component, post_id, permalink, caption)
        except Exception:
            pass
        print(f"[VK] Carousel posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[VK] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"VK {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_vk_worker(article_id, video_url, caption, run_ts):
    """Background thread: upload narrated video to VK."""
    # UTM tracking: replace plain article URL with VK-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'vk', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"vk_narrated_status_{article_id}_{run_ts}"
    result_key = f"vk_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        group_id = str(abs(int(VK_OWNER_ID)))
        title = caption[:128] if caption else "Narrated Video"

        # 1. Create video entry
        _set_status("running:create")
        resp = _req.get("https://api.vk.com/method/video.save", params={
            "access_token": VK_ACCESS_TOKEN, "v": "5.199",
            "group_id": group_id,
            "name": title,
            "description": caption[:5000] if caption else "",
            "wallpost": "0",
        }, timeout=15)
        resp.raise_for_status()
        vdata = resp.json().get("response", {})
        vk_upload_url = vdata.get("upload_url", "")
        video_id      = str(vdata.get("video_id", ""))
        owner_id      = str(vdata.get("owner_id", ""))
        if not vk_upload_url or not video_id:
            _set_result(f"error:VK video.save failed: {resp.json()}")
            return

        # 2. Download video
        _set_status("running:download")
        resp_dl = _req.get(video_url, timeout=120)
        resp_dl.raise_for_status()
        video_bytes = resp_dl.content

        # 3. Upload video
        _set_status("running:upload")
        resp = _req.post(vk_upload_url,
                         files={"video_file": ("video.mp4", video_bytes, "video/mp4")},
                         timeout=300)
        resp.raise_for_status()

        permalink = f"https://vk.com/video{owner_id}_{video_id}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"vk_narrated_post_id_{article_id}_{run_ts}", video_id, video_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"vk_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "VK Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"video_id={video_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'vk', 'narrated', video_id, permalink, caption)
        except Exception:
            pass
        print(f"[VK] Narrated video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[VK] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "VK Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── VK: Post carousel ────────────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-vk", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_vk(article_id):
    if not VK_CONFIGURED:
        return jsonify({"error": "VK credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"vk_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_vk_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-vk-status")
@login_required
@admin_required
def admin_carousel_vk_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"vk_{component}_status_{article_id}"
    result_key  = f"vk_{component}_result_{article_id}"
    history_key = f"vk_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-carousel-vk-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_vk_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"vk_{component}_post_id_{article_id}"
    result_key  = f"vk_{component}_result_{article_id}"
    status_key  = f"vk_{component}_status_{article_id}"
    history_key = f"vk_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-vk-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_vk_post(article_id):
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"vk_{component}_post_id_{article_id}"
    result_key  = f"vk_{component}_result_{article_id}"
    status_key  = f"vk_{component}_status_{article_id}"
    history_key = f"vk_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        resp = _req.get("https://api.vk.com/method/wall.delete", params={
            "access_token": VK_ACCESS_TOKEN, "v": "5.199",
            "owner_id": VK_OWNER_ID, "post_id": post_id,
        }, timeout=15)
        rdata = resp.json()
        if rdata.get("error"):
            err_msg = rdata["error"].get("error_msg", "Unknown error")
            return jsonify({"error": f"Could not delete post: {err_msg}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ── VK: Post narrated video ──────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-vk", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_vk(article_id):
    if not VK_CONFIGURED:
        return jsonify({"error": "VK credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"vk_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_vk_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-vk-status")
@login_required
@admin_required
def admin_narrated_vk_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"vk_narrated_status_{article_id}_{run_ts}"
    result_key  = f"vk_narrated_result_{article_id}_{run_ts}"
    history_key = f"vk_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-narrated-vk-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_vk_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"vk_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"vk_narrated_result_{article_id}_{run_ts}"
    status_key  = f"vk_narrated_status_{article_id}_{run_ts}"
    history_key = f"vk_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-vk-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_vk_post(article_id):
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"vk_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"vk_narrated_result_{article_id}_{run_ts}"
    status_key  = f"vk_narrated_status_{article_id}_{run_ts}"
    history_key = f"vk_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        # VK video deletion — use video.delete
        resp = _req.get("https://api.vk.com/method/video.delete", params={
            "access_token": VK_ACCESS_TOKEN, "v": "5.199",
            "video_id": post_id, "owner_id": VK_OWNER_ID,
        }, timeout=15)
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ══════════════════════════════════════════════════════════════
# TUMBLR: Workers & Routes
# ══════════════════════════════════════════════════════════════

def _post_carousel_tumblr_worker(article_id, caption, component):
    """Background thread: compose carousel images and post to Tumblr via NPF."""
    # UTM tracking: replace plain article URL with Tumblr-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'tumblr', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"tumblr_{component}_status_{article_id}"
    result_key = f"tumblr_{component}_result_{article_id}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        row = query_one(
            "SELECT carousel_images, carousel_punchlines FROM articles WHERE id = %s",
            (article_id,)
        )
        if not row:
            _set_result("error:Article not found")
            return

        images     = (json.loads(row[0]) if row[0] else [])[:10]
        punchlines = (json.loads(row[1]) if row[1] else [])[:10]
        valid_imgs = [u for u in images if u]
        if not valid_imgs:
            _set_result("error:No images found")
            return

        n = len(valid_imgs)
        carousel_band_top = _compute_max_overlay_band_top(punchlines[:n])
        tumblr_headers = {
            "Authorization": f"Bearer {TUMBLR_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }

        # 1. Compose and upload images to GCS
        _set_status(f"running:compose:0/{n}")
        ts_t = int(_time_mod.time())
        public_urls = []

        for idx, img_url in enumerate(valid_imgs):
            _set_status(f"running:compose:{idx+1}/{n}")
            try:
                if img_url.startswith("https://"):
                    resp_img = _req.get(img_url, timeout=20)
                    resp_img.raise_for_status()
                    img_bytes = resp_img.content
                else:
                    local = resolve_image_to_local_path(img_url)
                    img_bytes = local.read_bytes() if local and local.exists() else b""
            except Exception as e:
                _set_result(f"error:Could not fetch image {idx+1}: {e}")
                return

            punchline  = punchlines[idx] if idx < len(punchlines) else ""
            jpeg_bytes = compose_carousel_slide(img_bytes, punchline, idx, n,
                                                band_top=carousel_band_top,
                                                hint_text="obelisk-stamps.com")
            gcs_obj    = f"articles/{article_id}/tumblr/{component}_{idx+1}_{ts_t}.jpg"
            public_url = upload_bytes_to_gcs(jpeg_bytes, gcs_obj, content_type="image/jpeg")
            if not public_url:
                _set_result(f"error:Image upload failed for slide {idx+1}")
                return
            public_urls.append(public_url)

        # 2. Create Tumblr post (NPF format)
        _set_status("running:publish")
        content_blocks = []
        if caption:
            content_blocks.append({"type": "text", "text": caption[:4096]})
        for url in public_urls:
            content_blocks.append({"type": "image", "media": [{"url": url}]})

        resp = _req.post(f"https://api.tumblr.com/v2/blog/{TUMBLR_BLOG_NAME}/posts",
                         headers=tumblr_headers,
                         json={"content": content_blocks, "state": "published"},
                         timeout=30)
        resp.raise_for_status()
        tdata = resp.json()
        post_id = str(tdata.get("response", {}).get("id", ""))
        if not post_id:
            _set_result(f"error:Tumblr post failed: {tdata}")
            return

        permalink = f"https://{TUMBLR_BLOG_NAME}.tumblr.com/post/{post_id}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"tumblr_{component}_post_id_{article_id}", post_id, post_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"tumblr_{component}_caption_{article_id}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, f"Tumblr {component.title()} Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"post_id={post_id}\nimages={n}",
                          component=component)
        try:
            log_social_post(article_id, 'tumblr', component, post_id, permalink, caption)
        except Exception:
            pass
        print(f"[Tumblr] Carousel posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Tumblr] Carousel worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, f"Tumblr {component.title()} Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component=component)


def _post_narrated_tumblr_worker(article_id, video_url, caption, run_ts):
    """Background thread: post narrated video to Tumblr via NPF."""
    # UTM tracking: replace plain article URL with Tumblr-tagged version
    _art_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
    if _art_row and _art_row[0] and SITE_URL:
        caption = caption.replace(f"{SITE_URL}/articles/{_art_row[0]}", make_utm_url(_art_row[0], 'tumblr', article_id))
    import requests as _req
    import time as _time_mod

    status_key = f"tumblr_narrated_status_{article_id}_{run_ts}"
    result_key = f"tumblr_narrated_result_{article_id}_{run_ts}"

    def _set_status(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (status_key, v, v))

    def _set_result(v):
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s", (result_key, v, v))
        execute("DELETE FROM site_settings WHERE `key` = %s", (status_key,))

    try:
        tumblr_headers = {
            "Authorization": f"Bearer {TUMBLR_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }

        _set_status("running:publish")
        content_blocks = []
        if caption:
            content_blocks.append({"type": "text", "text": caption[:4096]})
        content_blocks.append({"type": "video", "url": video_url})

        resp = _req.post(f"https://api.tumblr.com/v2/blog/{TUMBLR_BLOG_NAME}/posts",
                         headers=tumblr_headers,
                         json={"content": content_blocks, "state": "published"},
                         timeout=30)
        resp.raise_for_status()
        tdata = resp.json()
        post_id = str(tdata.get("response", {}).get("id", ""))
        if not post_id:
            _set_result(f"error:Tumblr post failed: {tdata}")
            return

        permalink = f"https://{TUMBLR_BLOG_NAME}.tumblr.com/post/{post_id}"

        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                (f"tumblr_narrated_post_id_{article_id}_{run_ts}", post_id, post_id))
        _set_result(f"done:{permalink}")
        # Store published caption for page reload
        try:
            execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE value = %s",
                    (f"tumblr_narrated_caption_{article_id}_{run_ts}", caption, caption))
        except Exception:
            pass
        _add_activity_log(article_id, "Tumblr Narrated Video Posted",
                          f"<a href=\"{permalink}\" target=\"_blank\">{permalink}</a>\n"
                          f"post_id={post_id}\nrun_ts={run_ts}",
                          component="narrated")
        try:
            log_social_post(article_id, 'tumblr', 'narrated', post_id, permalink, caption)
        except Exception:
            pass
        print(f"[Tumblr] Narrated video posted: {permalink}", flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Tumblr] Narrated worker error: {e}\n{tb}", flush=True)
        _set_result(f"error:{e}")
        _add_activity_log(article_id, "Tumblr Narrated Video Post Failed",
                          f"Exception: {e}\n\n{tb[:2000]}", component="narrated")


# ── Tumblr: Post carousel ────────────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-carousel-to-tumblr", methods=["POST"])
@login_required
@admin_required
def admin_post_carousel_to_tumblr(article_id):
    if not TUMBLR_CONFIGURED:
        return jsonify({"error": "Tumblr credentials not configured."}), 400
    data      = request.get_json() or {}
    component = data.get("type", "car")
    caption   = data.get("caption", "")
    status_key = f"tumblr_{component}_status_{article_id}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_carousel_tumblr_worker,
                     args=(article_id, caption, component), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/carousel-tumblr-status")
@login_required
@admin_required
def admin_carousel_tumblr_status(article_id):
    component   = request.args.get("type", "car")
    status_key  = f"tumblr_{component}_status_{article_id}"
    result_key  = f"tumblr_{component}_result_{article_id}"
    history_key = f"tumblr_{component}_history_{article_id}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-carousel-tumblr-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_carousel_tumblr_post(article_id):
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"tumblr_{component}_post_id_{article_id}"
    result_key  = f"tumblr_{component}_result_{article_id}"
    status_key  = f"tumblr_{component}_status_{article_id}"
    history_key = f"tumblr_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-carousel-tumblr-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_carousel_tumblr_post(article_id):
    import requests as _req
    import datetime as _dt
    data      = request.get_json() or {}
    component = data.get("type", "car")
    post_key    = f"tumblr_{component}_post_id_{article_id}"
    result_key  = f"tumblr_{component}_result_{article_id}"
    status_key  = f"tumblr_{component}_status_{article_id}"
    history_key = f"tumblr_{component}_history_{article_id}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        resp = _req.delete(f"https://api.tumblr.com/v2/blog/{TUMBLR_BLOG_NAME}/post/{post_id}",
                           headers={"Authorization": f"Bearer {TUMBLR_ACCESS_TOKEN}"},
                           timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete post: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ── Tumblr: Post narrated video ──────────────────────────────

@app.route("/admin/articles/<int:article_id>/post-narrated-to-tumblr", methods=["POST"])
@login_required
@admin_required
def admin_post_narrated_to_tumblr(article_id):
    if not TUMBLR_CONFIGURED:
        return jsonify({"error": "Tumblr credentials not configured."}), 400
    data      = request.get_json() or {}
    video_url = data.get("video_url", "")
    caption   = data.get("caption", "")
    run_ts    = str(data.get("run_ts", "0"))
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    status_key = f"tumblr_narrated_status_{article_id}_{run_ts}"
    if (get_setting(status_key) or "").startswith("running"):
        return jsonify({"error": "Already posting this video."}), 409
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (status_key, "running:0", "running:0"))
    threading.Thread(target=_post_narrated_tumblr_worker,
                     args=(article_id, video_url, caption, run_ts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/admin/articles/<int:article_id>/narrated-tumblr-status")
@login_required
@admin_required
def admin_narrated_tumblr_status(article_id):
    run_ts      = request.args.get("ts", "0")
    status_key  = f"tumblr_narrated_status_{article_id}_{run_ts}"
    result_key  = f"tumblr_narrated_result_{article_id}_{run_ts}"
    history_key = f"tumblr_narrated_history_{article_id}_{run_ts}"
    return jsonify({
        "status":  get_setting(status_key) or "idle",
        "result":  get_setting(result_key) or "",
        "history": json.loads(get_setting(history_key) or "[]"),
    })


@app.route("/admin/articles/<int:article_id>/archive-narrated-tumblr-post", methods=["POST"])
@login_required
@admin_required
def admin_archive_narrated_tumblr_post(article_id):
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"tumblr_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"tumblr_narrated_result_{article_id}_{run_ts}"
    status_key  = f"tumblr_narrated_status_{article_id}_{run_ts}"
    history_key = f"tumblr_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post to archive"}), 400
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


@app.route("/admin/articles/<int:article_id>/delete-narrated-tumblr-post", methods=["POST"])
@login_required
@admin_required
def admin_delete_narrated_tumblr_post(article_id):
    import requests as _req
    import datetime as _dt
    data    = request.get_json() or {}
    run_ts  = str(data.get("run_ts", "0"))
    post_key    = f"tumblr_narrated_post_id_{article_id}_{run_ts}"
    result_key  = f"tumblr_narrated_result_{article_id}_{run_ts}"
    status_key  = f"tumblr_narrated_status_{article_id}_{run_ts}"
    history_key = f"tumblr_narrated_history_{article_id}_{run_ts}"
    post_id = get_setting(post_key)
    if not post_id:
        return jsonify({"error": "No post record found"}), 400
    try:
        resp = _req.delete(f"https://api.tumblr.com/v2/blog/{TUMBLR_BLOG_NAME}/post/{post_id}",
                           headers={"Authorization": f"Bearer {TUMBLR_ACCESS_TOKEN}"},
                           timeout=15)
        if resp.status_code not in (200, 204, 404):
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:300]
            return jsonify({"error": f"Could not delete post: {err}"}), 400
    except Exception:
        pass
    result    = get_setting(result_key) or ""
    permalink = result.replace("done:", "") if result.startswith("done:") else ""
    history   = json.loads(get_setting(history_key) or "[]")
    history.append({"post_id": post_id, "permalink": permalink,
                    "archived_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    hist_val = json.dumps(history)
    execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s", (history_key, hist_val, hist_val))
    for k in (post_key, result_key, status_key):
        execute("DELETE FROM site_settings WHERE `key` = %s", (k,))
    return jsonify({"ok": True, "history": history})


# ------------------------------------------------------------
# ENGAGEMENT & ANALYTICS
# ------------------------------------------------------------

@app.route("/admin/articles/<int:article_id>/refresh-engagement", methods=["POST"])
@login_required
@admin_required
def admin_refresh_engagement(article_id):
    """Fetch latest engagement metrics from all platform APIs.

    Always returns JSON, even on partial or total failure, so the frontend can
    render a meaningful error instead of choking on an HTML error page.
    """
    try:
        results = fetch_all_engagement(article_id)
        platform_errors = getattr(fetch_all_engagement, '_last_errors', []) or []
    except Exception as _e:
        import traceback as _tb
        print(f"[Engagement] fetch_all_engagement crashed: {_e}\n{_tb.format_exc()}", flush=True)
        return jsonify({'ok': False, 'error': f'Engagement fetch crashed: {_e}',
                        'totals': {}, 'platforms': {}, 'details': [],
                        'platform_errors': []}), 200

    # Comment fetches are best-effort — log per-platform errors but don't abort
    comment_errors = []
    for label, configured, fn in [
        ('Instagram', bool(IG_USER_ID and IG_ACCESS_TOKEN), _fetch_ig_comments),
        ('Facebook',  bool(FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN), _fetch_fb_comments),
        ('Threads',   bool(THREADS_CONFIGURED), _fetch_threads_comments),
        ('Bluesky',   bool(BLUESKY_CONFIGURED), _fetch_bluesky_comments),
    ]:
        if not configured:
            continue
        try:
            fn(article_id)
        except Exception as _ce:
            print(f"[Engagement] {label} comments fetch failed: {_ce}", flush=True)
            comment_errors.append({'platform': label, 'error': str(_ce)})

    METRIC_KEYS = ['likes', 'views', 'shares', 'comments', 'saves', 'clicks', 'impressions', 'reach']
    totals = {k: 0 for k in METRIC_KEYS}
    platforms = {}
    by_content_type = {}  # {(platform, content_type): {metrics...}}

    for r in results:
        for k in METRIC_KEYS:
            totals[k] += r.get(k, 0)
        p = r.get('platform', '')
        if not p:
            continue
        ct = r.get('content_type', '')
        if p not in platforms:
            platforms[p] = {k: 0 for k in METRIC_KEYS}
            platforms[p]['content_types'] = []
            platforms[p]['link_clicks'] = 0
        for k in METRIC_KEYS:
            platforms[p][k] += r.get(k, 0)
        if ct and ct not in platforms[p]['content_types']:
            platforms[p]['content_types'].append(ct)

        # Per content-type breakdown
        ct_key = f"{p}::{ct}"
        if ct_key not in by_content_type:
            by_content_type[ct_key] = {'platform': p, 'content_type': ct,
                                        **{k: 0 for k in METRIC_KEYS}}
        for k in METRIC_KEYS:
            by_content_type[ct_key][k] += r.get(k, 0)

    # Short-link clicks per platform (the user's "clicks on the links" metric)
    link_clicks_total = 0
    try:
        link_rows = query_all(
            "SELECT platform, click_count FROM short_links WHERE article_id = %s",
            (article_id,))
        for lr in (link_rows or []):
            plat, clicks = lr[0], int(lr[1] or 0)
            link_clicks_total += clicks
            if plat not in platforms:
                platforms[plat] = {k: 0 for k in METRIC_KEYS}
                platforms[plat]['content_types'] = []
                platforms[plat]['link_clicks'] = 0
            platforms[plat]['link_clicks'] = clicks
    except Exception as _le:
        print(f"[Engagement] short_links lookup failed: {_le}", flush=True)

    # UTM-attributed orders/revenue per platform
    revenue_by_platform = {}
    orders_by_platform = {}
    revenue_total = 0.0
    orders_total = 0
    try:
        slug_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if slug_row and slug_row[0]:
            order_rows = query_all(
                "SELECT utm_source, COUNT(*), COALESCE(SUM(total_amount), 0) "
                "FROM orders WHERE utm_campaign = %s AND utm_source IS NOT NULL "
                "GROUP BY utm_source",
                (slug_row[0],))
            for orow in (order_rows or []):
                src = (orow[0] or '').lower()
                cnt = int(orow[1] or 0)
                rev = float(orow[2] or 0)
                if not src:
                    continue
                orders_by_platform[src] = cnt
                revenue_by_platform[src] = rev
                orders_total += cnt
                revenue_total += rev
    except Exception as _re:
        print(f"[Engagement] orders attribution lookup failed: {_re}", flush=True)

    totals['link_clicks'] = link_clicks_total
    totals['orders'] = orders_total
    totals['revenue'] = round(revenue_total, 2)

    # Filter out platforms that aren't actually being used.
    # A platform is "active" if any of: it has rows in posting_log for this
    # article, has any non-zero engagement metric, has link clicks, or has
    # attributed orders.
    try:
        plog_rows = query_all(
            "SELECT DISTINCT platform FROM posting_log WHERE article_id = %s",
            (article_id,))
        published_platforms = {row[0] for row in (plog_rows or []) if row[0]}
    except Exception:
        published_platforms = set()

    def _platform_is_active(p, metrics):
        if p in published_platforms:
            return True
        if (metrics.get('link_clicks') or 0) > 0:
            return True
        if any((metrics.get(k) or 0) > 0 for k in METRIC_KEYS):
            return True
        if (orders_by_platform.get(p) or 0) > 0:
            return True
        return False

    platforms = {p: m for p, m in platforms.items() if _platform_is_active(p, m)}
    by_ct_filtered = [r for r in by_content_type.values() if r['platform'] in platforms]

    return jsonify({'ok': True,
                    'totals': totals,
                    'platforms': platforms,
                    'details': results,
                    'by_content_type': by_ct_filtered,
                    'revenue_by_platform': revenue_by_platform,
                    'orders_by_platform': orders_by_platform,
                    'platform_errors': platform_errors + comment_errors,
                    'fetched_at': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})


@app.route("/admin/fetch-follower-counts", methods=["POST"])
@login_required
@admin_required
def admin_fetch_follower_counts():
    """Snapshot current follower counts across all platforms."""
    try:
        results = fetch_follower_counts()
        return jsonify({"ok": True, "platforms": [{"platform": p, "followers": f, "following": fo, "posts": po} for p, f, fo, po in results]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/admin/articles/<int:article_id>/engagement")
@login_required
@admin_required
def admin_get_engagement(article_id):
    """Get latest engagement snapshot for an article (incl. link clicks & orders)."""
    rows = query_all("""
        SELECT e1.* FROM article_engagement e1
        INNER JOIN (
            SELECT platform, content_type, MAX(fetched_at) as max_fetched
            FROM article_engagement WHERE article_id = %s
            GROUP BY platform, content_type
        ) e2 ON e1.platform = e2.platform AND e1.content_type = e2.content_type
            AND e1.fetched_at = e2.max_fetched
        WHERE e1.article_id = %s
        ORDER BY e1.platform, e1.content_type
    """, (article_id, article_id))

    METRIC_KEYS = ['likes', 'views', 'shares', 'comments', 'saves', 'clicks', 'impressions', 'reach']
    _eng_cols = ['id','article_id','platform','content_type','post_id','permalink',
                 'likes','views','shares','comments','saves','clicks','impressions','reach',
                 'fetched_at','created_at']
    totals = {k: 0 for k in METRIC_KEYS}
    platforms = {}
    by_content_type = {}
    details = []
    fetched_at = None
    for r in rows or []:
        d = {_eng_cols[i]: r[i] for i in range(min(len(_eng_cols), len(r)))}
        details.append(d)
        for k in METRIC_KEYS:
            totals[k] += d.get(k, 0) or 0
        p = d.get('platform') or ''
        ct = d.get('content_type') or ''
        if not p:
            continue
        if p not in platforms:
            platforms[p] = {k: 0 for k in METRIC_KEYS}
            platforms[p]['content_types'] = []
            platforms[p]['link_clicks'] = 0
        for k in METRIC_KEYS:
            platforms[p][k] += d.get(k, 0) or 0
        if ct and ct not in platforms[p]['content_types']:
            platforms[p]['content_types'].append(ct)
        ct_key = f"{p}::{ct}"
        if ct_key not in by_content_type:
            by_content_type[ct_key] = {'platform': p, 'content_type': ct,
                                        **{k: 0 for k in METRIC_KEYS}}
        for k in METRIC_KEYS:
            by_content_type[ct_key][k] += d.get(k, 0) or 0
        if d.get('fetched_at'):
            fa = d['fetched_at']
            fetched_at = fa.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(fa, 'strftime') else str(fa)

    # Short-link clicks per platform
    link_clicks_total = 0
    try:
        link_rows = query_all(
            "SELECT platform, click_count FROM short_links WHERE article_id = %s",
            (article_id,))
        for lr in (link_rows or []):
            plat, clicks = lr[0], int(lr[1] or 0)
            link_clicks_total += clicks
            if plat not in platforms:
                platforms[plat] = {k: 0 for k in METRIC_KEYS}
                platforms[plat]['content_types'] = []
                platforms[plat]['link_clicks'] = 0
            platforms[plat]['link_clicks'] = clicks
    except Exception as _le:
        print(f"[Engagement] short_links lookup failed: {_le}", flush=True)

    # UTM-attributed orders
    revenue_by_platform = {}
    orders_by_platform = {}
    revenue_total = 0.0
    orders_total = 0
    try:
        slug_row = query_one("SELECT slug FROM articles WHERE id = %s", (article_id,))
        if slug_row and slug_row[0]:
            order_rows = query_all(
                "SELECT utm_source, COUNT(*), COALESCE(SUM(total_amount), 0) "
                "FROM orders WHERE utm_campaign = %s AND utm_source IS NOT NULL "
                "GROUP BY utm_source",
                (slug_row[0],))
            for orow in (order_rows or []):
                src = (orow[0] or '').lower()
                if not src:
                    continue
                cnt = int(orow[1] or 0)
                rev = float(orow[2] or 0)
                orders_by_platform[src] = cnt
                revenue_by_platform[src] = rev
                orders_total += cnt
                revenue_total += rev
    except Exception as _re:
        print(f"[Engagement] orders attribution lookup failed: {_re}", flush=True)

    totals['link_clicks'] = link_clicks_total
    totals['orders'] = orders_total
    totals['revenue'] = round(revenue_total, 2)

    # Filter out platforms that aren't actually being used.
    try:
        plog_rows = query_all(
            "SELECT DISTINCT platform FROM posting_log WHERE article_id = %s",
            (article_id,))
        published_platforms = {row[0] for row in (plog_rows or []) if row[0]}
    except Exception:
        published_platforms = set()

    def _platform_is_active(p, metrics):
        if p in published_platforms:
            return True
        if (metrics.get('link_clicks') or 0) > 0:
            return True
        if any((metrics.get(k) or 0) > 0 for k in METRIC_KEYS):
            return True
        if (orders_by_platform.get(p) or 0) > 0:
            return True
        return False

    platforms = {p: m for p, m in platforms.items() if _platform_is_active(p, m)}
    by_ct_filtered = [r for r in by_content_type.values() if r['platform'] in platforms]

    return jsonify({
        'totals': totals,
        'platforms': platforms,
        'details': details,
        'by_content_type': by_ct_filtered,
        'revenue_by_platform': revenue_by_platform,
        'orders_by_platform': orders_by_platform,
        'fetched_at': fetched_at,
    })


@app.route("/admin/engagement/poll-all", methods=["POST"])
@login_required
@admin_required
def admin_engagement_poll_all():
    """Fetch engagement for all published articles."""
    articles = query_all("SELECT id, title FROM articles WHERE is_published = 1")
    results = {}
    for art in (articles or []):
        art_id, art_title = art[0], art[1]
        try:
            data = fetch_all_engagement(art_id)
            results[art_id] = {'title': art_title, 'metrics_count': len(data)}
        except Exception as e:
            results[art_id] = {'title': art_title, 'error': str(e)}
    return jsonify({'ok': True, 'articles': results})


@app.route("/admin/analytics")
@login_required
@admin_required
def admin_analytics():
    """Analytics dashboard page."""
    class _Row:
        """Simple wrapper to allow attribute access on query tuples."""
        def __init__(self, **kw):
            self.__dict__.update(kw)

    try:
        raw_articles = query_all("""
            SELECT a.id, a.title, a.slug, a.is_published, a.created_at,
                COALESCE(e.total_likes, 0) as total_likes,
                COALESCE(e.total_views, 0) as total_views,
                COALESCE(e.total_shares, 0) as total_shares,
                COALESCE(e.total_comments, 0) as total_comments,
                e.last_fetched
            FROM articles a
            LEFT JOIN (
                SELECT article_id,
                    SUM(likes) as total_likes, SUM(views) as total_views,
                    SUM(shares) as total_shares, SUM(comments) as total_comments,
                    MAX(fetched_at) as last_fetched
                FROM article_engagement
                GROUP BY article_id
            ) e ON a.id = e.article_id
            WHERE a.is_published = 1
            ORDER BY COALESCE(e.total_views, 0) DESC
        """)
    except Exception:
        raw_articles = query_all("SELECT id, title, slug, is_published, created_at, 0,0,0,0, NULL FROM articles WHERE is_published = 1 ORDER BY id DESC")
    articles = [_Row(id=r[0], title=r[1], slug=r[2], status=r[3], created_at=r[4],
                     total_likes=r[5], total_views=r[6], total_shares=r[7],
                     total_comments=r[8], last_fetched=r[9]) for r in (raw_articles or []) if r]

    try:
        raw_platform = query_all("""
            SELECT platform,
                SUM(likes) as total_likes, SUM(views) as total_views,
                SUM(shares) as total_shares, SUM(comments) as total_comments
            FROM article_engagement
            GROUP BY platform
            ORDER BY total_views DESC
        """)
    except Exception:
        raw_platform = []
    platform_totals = [_Row(platform=r[0], total_likes=r[1], total_views=r[2],
                            total_shares=r[3], total_comments=r[4]) for r in (raw_platform or [])]

    return render_template("admin_analytics.html",
                           articles=articles,
                           platform_totals=platform_totals,
                           ga_measurement_id=GA_MEASUREMENT_ID)


# ------------------------------------------------------------
# ARTICLE PIPELINE
# ------------------------------------------------------------

@app.route("/admin/article-pipeline")
@login_required
@admin_required
def admin_article_pipeline():
    rows = query_all(
        "SELECT id, title, subtitle, description, target_date, status, article_id, generation_error, batch_id, created_at "
        "FROM article_queue ORDER BY target_date ASC, id ASC"
    )
    queue = []
    for r in rows:
        queue.append({
            "id": r[0], "title": r[1], "subtitle": r[2], "description": r[3],
            "target_date": str(r[4]) if r[4] else "", "status": r[5],
            "article_id": r[6], "generation_error": r[7], "batch_id": r[8],
            "created_at": str(r[9]) if r[9] else ""
        })
    _default_idea_prompt = (
        "You are an expert philatelist, historian, and content strategist for Obelisk Stamps, "
        "a premium online store selling handcrafted framed stamp displays.\n\n"
        "Research and generate article ideas based on real stories, curiosities, and fascinating events "
        "from the world of stamps and postal history. Every country has issued stamps that reflect their "
        "culture, politics, technology, and identity — find the most compelling stories behind them.\n\n"
        "Include a mix of:\n"
        "- Famous stamp errors and printing mistakes that became legendary (e.g., Inverted Jenny, Treskilling Yellow)\n"
        "- Rare stamps with dramatic ownership histories and record-breaking auction sales\n"
        "- How countries used stamps as propaganda, cultural promotion, or political statements\n"
        "- Unusual stamp materials and formats (foil, silk, scented, 3D, shaped)\n"
        "- Wartime postal stories, espionage, and stamps used in psychological operations\n"
        "- The origins of postal systems and how they shaped global communication\n"
        "- Stamp collecting tips, investment insights, and how to spot valuable finds\n"
        "- Stories from all continents and eras — not just Western/European stamps\n\n"
        "Each idea should be based on real historical events or verifiable facts. "
        "Write titles that are click-worthy and SEO-friendly. "
        "The articles will ultimately drive traffic to an e-commerce store selling framed stamp displays, "
        "so favour topics that make readers appreciate the beauty and value of stamps."
    )
    idea_prompt = get_setting("pipeline_idea_prompt", _default_idea_prompt)
    return render_template("admin_pipeline.html", queue=queue, idea_prompt=idea_prompt)


@app.route("/admin/article-pipeline/generate-ideas", methods=["POST"])
@login_required
@admin_required
def admin_pipeline_generate_ideas():
    if not _openai_client:
        return jsonify({"error": "OpenAI not configured"}), 400
    try:
        return _pipeline_generate_ideas_inner()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Pipeline] generate-ideas error: {tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500

def _pipeline_generate_ideas_inner():
    data = request.get_json() or {}
    articles_per_day = int(data.get("articles_per_day", 2))
    start_date = data.get("start_date", "")
    end_date = data.get("end_date", "")

    from datetime import datetime, timedelta
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end - start).days + 1
    total_articles = total_days * articles_per_day

    target_dates = []
    current = start
    while current <= end:
        for _ in range(articles_per_day):
            target_dates.append(current)
        current += timedelta(days=1)

    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Use custom prompt from form, or fall back to stored/default
    custom_prompt = data.get("custom_prompt", "").strip()
    if custom_prompt:
        # Save as the new default for next time
        execute("INSERT INTO site_settings (`key`, value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE value = %s",
                ("pipeline_idea_prompt", custom_prompt, custom_prompt))

    all_ideas = []
    chunk_size = 30
    for chunk_start in range(0, total_articles, chunk_size):
        chunk_n = min(chunk_size, total_articles - chunk_start)
        previous_titles = "\n".join([i["title"] for i in all_ideas[-50:]]) if all_ideas else "None yet"

        # Build system prompt from custom prompt + structural instructions
        base_prompt = custom_prompt or (
            "You are an expert philatelist and content strategist for Obelisk Stamps, "
            "a premium online store selling handcrafted framed stamp displays.\n\n"
            "Generate unique article ideas about stamps, postal history, "
            "stamp collecting, philately, famous stamps, stamp design, postal systems, and related topics.\n\n"
            "Each article should be educational, engaging, and appeal to both new and experienced stamp collectors."
        )

        system_prompt = (
            base_prompt + "\n\n"
            "Generate exactly " + str(chunk_n) + " unique article ideas.\n\n"
            "AVOID these previously generated titles:\n" + previous_titles + "\n\n"
            "For each article provide:\n"
            "- title: An engaging, SEO-friendly title (50-80 chars)\n"
            "- subtitle: A complementary subtitle (40-60 chars)\n"
            "- description: 2-3 sentences describing the article's angle and what it would cover\n\n"
            "Output ONLY a valid JSON array of objects with keys \"title\", \"subtitle\", \"description\". "
            "No markdown, no code blocks, just the JSON array."
        )

        try:
            response = _openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {chunk_n} unique stamp article ideas."}
                ],
                max_tokens=4000,
                temperature=0.9
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            ideas = json.loads(text)
        except Exception as e:
            return jsonify({"error": f"OpenAI error: {str(e)}"}), 500

        for idx, idea in enumerate(ideas):
            date_idx = chunk_start + idx
            if date_idx >= len(target_dates):
                break
            td = target_dates[date_idx]
            execute(
                "INSERT INTO article_queue (title, subtitle, description, target_date, status, batch_id, prompt_used) "
                "VALUES (%s, %s, %s, %s, 'pending', %s, %s)",
                (idea.get("title", "Untitled"), idea.get("subtitle", ""),
                 idea.get("description", ""), td.strftime("%Y-%m-%d"), batch_id,
                 base_prompt[:2000])
            )
            all_ideas.append(idea)

    return jsonify({"ok": True, "count": len(all_ideas), "batch_id": batch_id})


@app.route("/admin/article-pipeline/<int:queue_id>/status", methods=["POST"])
@login_required
@admin_required
def admin_pipeline_update_status(queue_id):
    data = request.get_json() or {}
    new_status = data.get("status", "")
    if new_status not in ("accepted", "rejected", "pending"):
        return jsonify({"error": "Invalid status"}), 400
    execute("UPDATE article_queue SET status = %s WHERE id = %s", (new_status, queue_id))

    if new_status == "accepted":
        threading.Thread(target=_generate_article_from_queue, args=(queue_id,), daemon=True).start()

    return jsonify({"ok": True})


def _generate_article_from_queue(queue_id):
    """Background thread: generate article content + carousel from queue idea."""
    import time as _time

    row = query_one(
        "SELECT id, title, subtitle, description FROM article_queue WHERE id = %s", (queue_id,))
    if not row:
        return

    qid, title, subtitle, description = row
    execute("UPDATE article_queue SET status = 'generating' WHERE id = %s", (qid,))

    try:
        if not _openai_client:
            raise Exception("OpenAI not configured")

        system_prompt = (
            "You are a professional writer for Obelisk Stamps, a premium online store selling "
            "handcrafted framed stamp displays. Write a comprehensive, educational, and engaging article.\n\n"
            "Requirements:\n"
            "- Write 2000-4000 words of well-structured HTML content\n"
            "- Use <h2> and <h3> for section headings (NOT <h1>)\n"
            "- Use <p> for paragraphs, <ul>/<ol> for lists where appropriate\n"
            "- Include historical facts, interesting anecdotes, and collector tips where relevant\n"
            "- Write in an authoritative but accessible tone\n"
            "- SEO-friendly: naturally incorporate relevant keywords\n"
            "- Do NOT include the article title in the content (it's rendered separately)\n"
            "- End with a brief conclusion\n\n"
            "Also generate:\n"
            "- excerpt: 2-3 sentence summary for previews (max 200 chars)\n"
            "- slug: URL-friendly version of the title (lowercase, hyphens, no special chars, max 60 chars)\n\n"
            "Output ONLY valid JSON with keys: \"content\", \"excerpt\", \"slug\". No markdown wrapping."
        )

        user_msg = f"Title: {title}\nSubtitle: {subtitle}\nDescription: {description}"

        response = _openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=8000,
            temperature=0.7
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        result = json.loads(text)
        content = result.get("content", "")
        excerpt = result.get("excerpt", "")
        slug = result.get("slug", "")

        if not slug:
            slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:60]

        existing = query_one("SELECT id FROM articles WHERE slug = %s", (slug,))
        if existing:
            slug = slug + "-" + str(qid)

        execute(
            "INSERT INTO articles (slug, title, subtitle, content, excerpt, is_published) "
            "VALUES (%s, %s, %s, %s, %s, FALSE)",
            (slug, title, subtitle or "", content, excerpt)
        )
        article_id = query_one("SELECT LAST_INSERT_ID()")[0]

        execute("UPDATE article_queue SET status = 'ready', article_id = %s WHERE id = %s",
                (article_id, qid))

        try:
            _generate_carousel_for_article_sync(article_id)
        except Exception as ce:
            print(f"[Pipeline] Carousel generation failed for article {article_id}: {ce}")

    except Exception as e:
        execute("UPDATE article_queue SET status = 'failed', generation_error = %s WHERE id = %s",
                (str(e)[:500], qid))


def _generate_carousel_for_article_sync(article_id):
    """Synchronous carousel generation for pipeline use. Generates storyboard + DALL-E images."""
    import time as _time

    row = query_one(
        "SELECT title, content, slug, carousel_prompts, carousel_images, carousel_punchlines, "
        "carousel_style, carousel_created_at FROM articles WHERE id = %s", (article_id,))
    if not row:
        return

    title = row[0] or ""
    content = row[1] or ""
    existing_prompts = json.loads(row[3] or "[]")
    existing_images = json.loads(row[4] or "[]")
    existing_punchlines = json.loads(row[5] or "[]")
    existing_created = json.loads(row[7] or "[]")

    plain = re.sub(r"<[^>]+>", " ", content)
    plain = re.sub(r"\s+", " ", plain).strip()[:6000]

    target_count = 10
    empty_slots = [i for i in range(target_count) if i >= len(existing_images) or not existing_images[i]]
    if not empty_slots:
        return

    n_needed = len(empty_slots)
    active_style = get_setting('carousel_style', CAROUSEL_STYLE_SUFFIX)

    # GPT storyboard
    storyboard_system = (
        "You are a visual storyteller and Instagram content strategist. "
        f"Your job is to break an article into exactly {n_needed} sequential scenes "
        "that narrate the article's story from beginning to end.\n\n"
        "For each scene, produce TWO things:\n"
        "1. A detailed DALL-E 3 image prompt that captures that section's key message, moment, or concept.\n"
        "2. A short punchline (max 15 words) -- a compelling caption that gives context to the image.\n\n"
        "Rules:\n"
        "- Each prompt must be specific to its section of the article (not generic)\n"
        "- Prompts progress chronologically through the article\n"
        "- Include concrete visual details: setting, subjects, actions, mood, colours\n"
        f"- All {n_needed} prompts must share the same art style for visual cohesion\n"
        "- Punchlines should be intriguing, concise, and encourage swiping\n"
        f"- Output ONLY a JSON array of {n_needed} objects, each with keys \"prompt\" and \"punchline\", no other text"
    )

    try:
        storyboard_resp = _openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": storyboard_system},
                {"role": "user", "content": (
                    f"Article title: {title}\n\nFull article content:\n{plain[:3000]}\n\n"
                    f"Generate a JSON array of exactly {n_needed} objects."
                )}
            ],
            max_tokens=3000,
            temperature=0.8
        )
        storyboard_text = storyboard_resp.choices[0].message.content.strip()
        if storyboard_text.startswith("```"):
            storyboard_text = re.sub(r'^```(?:json)?\s*', '', storyboard_text)
            storyboard_text = re.sub(r'\s*```$', '', storyboard_text)
        scenes = json.loads(storyboard_text.strip())
    except Exception as e:
        print(f"[Pipeline] Storyboard failed: {e}")
        return

    while len(existing_prompts) < target_count:
        existing_prompts.append("")
    while len(existing_images) < target_count:
        existing_images.append("")
    while len(existing_punchlines) < target_count:
        existing_punchlines.append("")
    while len(existing_created) < target_count:
        existing_created.append("")

    scene_idx = 0
    for slot in empty_slots:
        if scene_idx >= len(scenes):
            break
        scene = scenes[scene_idx]
        prompt = scene.get("prompt", "")
        punchline = scene.get("punchline", "")
        scene_idx += 1

        try:
            dalle_resp = _openai_client.images.generate(
                model="dall-e-3",
                prompt=f"{prompt}. {active_style}",
                size="1024x1024",
                quality="standard",
                style="vivid",
                n=1
            )
            image_url = dalle_resp.data[0].url

            dl = http_requests.get(image_url, timeout=60)
            dl.raise_for_status()

            local_filename = f"image_{slot + 1}.png"
            gcs_object_name = f"articles/{article_id}/carousel/{local_filename}"
            carousel_dir = os.path.join("static", "articles", str(article_id), "carousel")
            os.makedirs(carousel_dir, exist_ok=True)
            with open(os.path.join(carousel_dir, local_filename), "wb") as f:
                f.write(dl.content)

            gcs_url = upload_bytes_to_gcs(dl.content, gcs_object_name, content_type="image/png")
            if gcs_url:
                existing_images[slot] = gcs_url
            else:
                existing_images[slot] = f"articles/{article_id}/carousel/{local_filename}"

            existing_prompts[slot] = prompt
            existing_punchlines[slot] = punchline
            existing_created[slot] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            execute(
                "UPDATE articles SET carousel_prompts=%s, carousel_images=%s, carousel_punchlines=%s, "
                "carousel_style=%s, carousel_created_at=%s WHERE id=%s",
                (json.dumps(existing_prompts), json.dumps(existing_images), json.dumps(existing_punchlines),
                 active_style, json.dumps(existing_created), article_id)
            )

            _time.sleep(12)

        except Exception as e:
            print(f"[Pipeline] DALL-E failed for slot {slot}: {e}")
            _time.sleep(5)
            continue

    print(f"[Pipeline] Carousel done for article {article_id}: {sum(1 for i in existing_images[:10] if i)} images")


@app.route("/admin/article-pipeline/bulk-action", methods=["POST"])
@login_required
@admin_required
def admin_pipeline_bulk_action():
    data = request.get_json() or {}
    action = data.get("action", "")
    ids = data.get("ids", [])

    if action == "accept_all":
        threading.Thread(target=_bulk_accept_queue, args=(ids,), daemon=True).start()
        return jsonify({"ok": True, "message": f"Processing {len(ids)} items"})
    elif action == "reject_all":
        for qid in ids:
            execute("UPDATE article_queue SET status = 'rejected' WHERE id = %s AND status = 'pending'", (qid,))
        return jsonify({"ok": True})
    return jsonify({"error": "Unknown action"}), 400


def _bulk_accept_queue(queue_ids):
    import time as _time
    for qid in queue_ids:
        try:
            row = query_one("SELECT status FROM article_queue WHERE id = %s", (qid,))
            if row and row[0] == 'pending':
                execute("UPDATE article_queue SET status = 'accepted' WHERE id = %s", (qid,))
                _generate_article_from_queue(qid)
                _time.sleep(2)
        except Exception as e:
            execute("UPDATE article_queue SET status = 'failed', generation_error = %s WHERE id = %s",
                    (str(e)[:500], qid))


@app.route("/admin/article-pipeline/progress")
@login_required
@admin_required
def admin_pipeline_progress():
    rows = query_all(
        "SELECT id, title, status, article_id, generation_error FROM article_queue "
        "WHERE status IN ('accepted','generating') ORDER BY id ASC"
    )
    items = [{"id": r[0], "title": r[1], "status": r[2], "article_id": r[3], "error": r[4]} for r in rows]
    return jsonify({"items": items})


@app.route("/admin/article-pipeline/<int:queue_id>", methods=["DELETE"])
@login_required
@admin_required
def admin_pipeline_delete(queue_id):
    execute("DELETE FROM article_queue WHERE id = %s", (queue_id,))
    return jsonify({"ok": True})


@app.route("/admin/article-pipeline/<int:queue_id>/edit", methods=["POST"])
@login_required
@admin_required
def admin_pipeline_edit(queue_id):
    data = request.get_json() or {}
    title = data.get("title")
    subtitle = data.get("subtitle")
    description = data.get("description")
    target_date = data.get("target_date")
    updates = []
    params = []
    if title is not None:
        updates.append("title = %s")
        params.append(title)
    if subtitle is not None:
        updates.append("subtitle = %s")
        params.append(subtitle)
    if description is not None:
        updates.append("description = %s")
        params.append(description)
    if target_date is not None:
        updates.append("target_date = %s")
        params.append(target_date)
    if updates:
        params.append(queue_id)
        execute(f"UPDATE article_queue SET {', '.join(updates)} WHERE id = %s", tuple(params))
    return jsonify({"ok": True})


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
