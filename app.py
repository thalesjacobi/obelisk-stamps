"""
Main Flask app for Obelisk Stamps.

This version fixes ML asset loading + predict route wiring:
- Uses one consistent set of ML globals: model/ref_embeddings/ref_rows
- Loads assets using absolute paths relative to this file
- Registers custom Keras layer(s) before loading the .keras model
- Predict route hard-guards against None to avoid crashes
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional

import mysql.connector
import numpy as np
import requests
import tensorflow as tf
from keras import models as keras_models
from mysql.connector import Error
from oauthlib.oauth2 import WebApplicationClient

from dotenv import load_dotenv
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message

# IMPORTANT: Ensure custom layers are registered BEFORE loading the model
import ml.model_utils  # noqa: F401

from ml.inference import predict_stamp_value_from_image

# ------------------------------------------------------------
# ENV + APP
# ------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

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

# ML artifact paths (can be overridden via env vars, but default to project folders)
MODEL_PATH = Path(os.getenv("STAMP_MODEL_PATH", str(BASE_DIR / "models" / "stamp_embed.keras")))
INDEX_PATH = Path(os.getenv("STAMP_INDEX_PATH", str(BASE_DIR / "indexes" / "ref_embeddings.npy")))
ROWS_PATH = Path(os.getenv("STAMP_ROWS_PATH", str(BASE_DIR / "indexes" / "ref_rows.pkl")))

# ------------------------------------------------------------
# ML GLOBALS
# ------------------------------------------------------------
model: Optional[tf.keras.Model] = None
ref_embeddings: Optional[np.ndarray] = None
ref_rows: Optional[list] = None
ml_load_error: Optional[str] = None


def load_ml_assets() -> bool:
    """
    Load model + reference embeddings + rows into globals.
    Uses absolute paths so it works regardless of where you run `python app.py` from.
    """
    global model, ref_embeddings, ref_rows, ml_load_error
    ml_load_error = None

    expected = [MODEL_PATH, INDEX_PATH, ROWS_PATH]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        model = None
        ref_embeddings = None
        ref_rows = None
        ml_load_error = f"Missing ML files: {missing}"
        print(f"[ML] {ml_load_error}")
        return False

    try:
        model = keras_models.load_model(str(MODEL_PATH))
    except Exception as e:
        model = None
        ref_embeddings = None
        ref_rows = None
        ml_load_error = f"Model load failed: {repr(e)}"
        print(f"[ML] {ml_load_error}")
        return False

    try:
        ref_embeddings = np.load(str(INDEX_PATH))
        with open(str(ROWS_PATH), "rb") as f:
            ref_rows = pickle.load(f)
    except Exception as e:
        model = None
        ref_embeddings = None
        ref_rows = None
        ml_load_error = f"Index load failed: {repr(e)}"
        print(f"[ML] {ml_load_error}")
        return False

    # Sanity checks (fail fast with useful errors)
    if not isinstance(ref_embeddings, np.ndarray):
        ml_load_error = f"ref_embeddings is not a numpy array (got {type(ref_embeddings)})"
        print(f"[ML] {ml_load_error}")
        model = None
        ref_embeddings = None
        ref_rows = None
        return False

    if ref_embeddings.ndim != 2:
        ml_load_error = f"ref_embeddings must be 2D, got shape {ref_embeddings.shape}"
        print(f"[ML] {ml_load_error}")
        model = None
        ref_embeddings = None
        ref_rows = None
        return False

    if ref_rows is None or not isinstance(ref_rows, list):
        ml_load_error = f"ref_rows is not a list (got {type(ref_rows)})"
        print(f"[ML] {ml_load_error}")
        model = None
        ref_embeddings = None
        ref_rows = None
        return False

    if ref_embeddings.shape[0] != len(ref_rows):
        ml_load_error = f"ref_embeddings rows ({ref_embeddings.shape[0]}) != ref_rows ({len(ref_rows)})"
        print(f"[ML] {ml_load_error}")
        model = None
        ref_embeddings = None
        ref_rows = None
        return False

    print(f"[ML] Loaded model: {MODEL_PATH}")
    print(f"[ML] Loaded embeddings: {ref_embeddings.shape} from {INDEX_PATH}")
    print(f"[ML] Loaded rows: {len(ref_rows)} from {ROWS_PATH}")
    return True


# Load ML once on startup
load_ml_assets()

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

    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
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
    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
    token_endpoint = google_provider_cfg["token_endpoint"]

    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code,
    )

    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    client.parse_request_body_response(json.dumps(token_response.json()))
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]

    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

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
# ML PREDICT ROUTE
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Hard guard: never let None reach inference.py
    if model is None or ref_embeddings is None or ref_rows is None:
        return jsonify(
            {
                "error": "ML assets not loaded. Train model and build index first.",
                "expected_files": [str(MODEL_PATH), str(INDEX_PATH), str(ROWS_PATH)],
                "details": ml_load_error,
            }
        ), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Use multipart field name 'image'."}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))

    currency = request.form.get("currency", "EUR").upper()

    try:
        results = predict_stamp_value_from_image(
            model=model,
            ref_embeddings=ref_embeddings,
            ref_rows=ref_rows,
            query_image_path=str(save_path),
            return_currency=currency,
        )
    except Exception as e:
        return jsonify({"error": "Inference failed", "details": repr(e)}), 500

    return jsonify(results[:3])


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
