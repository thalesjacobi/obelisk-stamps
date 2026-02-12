"""
Main Flask app for Obelisk Stamps.

This version calls a separate ML API service for stamp recognition.
The ML API handles model loading and inference, keeping the website lightweight.
"""

import os
import json
from pathlib import Path
from typing import Optional

import mysql.connector
import requests as http_requests
from mysql.connector import Error
from oauthlib.oauth2 import WebApplicationClient

from dotenv import load_dotenv
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message

# ------------------------------------------------------------
# ENV + APP
# ------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# ------------------------------------------------------------
# ML API Configuration
# ------------------------------------------------------------
ML_API_URL = os.getenv("ML_API_URL", "http://localhost:8081")

# ------------------------------------------------------------
# MAIL
# ------------------------------------------------------------
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", "587"))
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")

mail = Mail(app)

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
    def __init__(self, id, username, email, name, active):
        self.id = id
        self.username = username
        self.email = email
        self.name = name
        self.active = active


@login_manager.user_loader
def load_user(user_id):
    user_data = query_one(
        "SELECT id, username, email, name, active FROM users WHERE id = %s",
        (user_id,),
    )
    if user_data:
        return User(*user_data)
    return None


# ------------------------------------------------------------
# PAGES
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/catalogue")
def catalogue():
    catalogue_items = query_all("SELECT * FROM catalogue")
    return render_template("catalogue.html", catalogue_items=catalogue_items)


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        try:
            msg = Message(
                "New Contact Request",
                sender=email,
                recipients=[os.getenv("CONTACT_TO_EMAIL", "thalesjacobi@gmail.com")],
            )
            msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
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


# ------------------------------------------------------------
# LOGIN VIA GOOGLE
# ------------------------------------------------------------
@app.route("/login")
def login():
    if client is None:
        flash("Google login is not configured. Set GOOGLE_CLIENT_ID/SECRET.", "danger")
        return redirect(url_for("home"))

    google_provider_cfg = http_requests.get(GOOGLE_DISCOVERY_URL).json()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=url_for("login_callback", _external=True),
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)


@app.route("/login/callback")
def login_callback():
    if client is None:
        flash("Google login is not configured.", "danger")
        return redirect(url_for("home"))

    code = request.args.get("code")
    google_provider_cfg = http_requests.get(GOOGLE_DISCOVERY_URL).json()
    token_endpoint = google_provider_cfg["token_endpoint"]

    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code,
    )

    token_response = http_requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    client.parse_request_body_response(json.dumps(token_response.json()))
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]

    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = http_requests.get(uri, headers=headers, data=body)

    user_info = userinfo_response.json()
    username = user_info["email"].split("@")[0]
    email = user_info["email"]
    name = user_info.get("name", username)

    user_data = query_one(
        "SELECT id, username, email, name, active FROM users WHERE email = %s",
        (email,),
    )

    if user_data:
        user = User(*user_data)
        if not user.active:
            flash("Your account is inactive. Please contact support.", "danger")
            return redirect(url_for("home"))
    else:
        user_id = execute(
            "INSERT INTO users (username, email, name, active, date_created) VALUES (%s, %s, %s, %s, NOW())",
            (username, email, name, 1),
        )
        user = User(user_id, username, email, name, 1)

    login_user(user)
    return redirect(url_for("account"))


@app.route("/account")
@login_required
def account():
    return f"Welcome, {current_user.name}! This is your account page."


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
    """Convert price to target currency."""
    fx_rates = {"EUR": 1.0, "USD": 1.08, "GBP": 0.86}

    if not price_value or not price_currency:
        return None

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
