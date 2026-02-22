"""
Main Flask app for Obelisk Stamps.

This application calls a separate ML API service for stamp recognition.
The ML API handles model loading and inference, keeping the website lightweight.
"""

import os
import json
import time
import base64
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
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_from_directory, session
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
# PATHS + UPLOADS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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


# ------------------------------------------------------------
# AUTH (Flask-Login)
# ------------------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id, username, email, name, active, picture=None, currency="GBP", country=None):
        self.id = id
        self.username = username
        self.email = email
        self.name = name
        self.active = active
        self.picture = picture  # Profile picture as bytes (BLOB)
        self.currency = currency or "GBP"
        self.country = country


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
        "SELECT id, username, email, name, active, picture, currency, country FROM users WHERE id = %s",
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


@app.route("/catalogue")
def catalogue():
    catalogue_items = query_all("SELECT * FROM catalogue")
    currency = get_active_currency()
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
        "SELECT id, username, email, name, active, picture, currency, country FROM users WHERE email = %s",
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
    """Update user's preferred currency."""
    data = request.get_json()
    currency = data.get("currency", "").upper()
    if currency not in ("EUR", "GBP", "USD"):
        return jsonify({"error": "Invalid currency"}), 400
    execute(
        "UPDATE users SET currency = %s WHERE id = %s",
        (currency, current_user.id),
    )
    current_user.currency = currency
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
