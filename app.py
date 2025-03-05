"""This is the main file of the website."""

import os
import json
import mysql.connector
from flask import Flask, render_template, request, flash, redirect, url_for, session
from flask_mail import Mail, Message
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
#import smtplib
from oauthlib.oauth2 import WebApplicationClient
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

# Flask-Mail Setup
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'thalesjacobi@gmail.com'
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

mail = Mail(app)

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# Database Connection
db = mysql.connector.connect(
    host="mysql-28241c40-thalesjacobi-7f99.d.aivencloud.com",
    port="16042",
    user="avnadmin",
    password=os.getenv('DB_PASSWORD'),
    database="defaultdb"
)

cursor = db.cursor()

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Mock User Model
class User(UserMixin):
    def __init__(self, id, username, email, name, active):
        self.id = id
        self.username = username
        self.email = email
        self.name = name
        self.active = active

@login_manager.user_loader
def load_user(user_id):
    cursor.execute("SELECT id, username, email, name, active FROM users WHERE id = %s", (user_id,))
    user_data = cursor.fetchone()
    if user_data:
        return User(*user_data)
    return None

#-------------------------------------------------------
# Templates

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/catalogue')
def catalogue():
    cursor.execute("SELECT * FROM catalogue")
    catalogue_items = cursor.fetchall()
    return render_template('catalogue.html', catalogue_items=catalogue_items)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        try:
            msg = Message("New Contact Request",
                          sender=email,
                          recipients=["thalesjacobi@gmail.com"])
            msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
            mail.send(msg)
            flash(f"Thank you, {name}. Your message has been sent!", "success")
        except smtplib.SMTPException as e:
            flash("Email sending failed. Please try again later.", "danger")
            print(f"Error sending email: {e}")
    
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    """Privacy Policy Page"""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Terms and Conditions Page"""
    return render_template('terms.html')

@app.route('/login')
def login():
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
    name = user_info["name"]

    cursor.execute("SELECT id, username, email, name, active FROM users WHERE email = %s", (email,))
    user_data = cursor.fetchone()

    if user_data:
        user = User(*user_data)
        if not user.active:
            flash("Your account is inactive. Please contact support.", "danger")
            return redirect(url_for("home"))
    else:
        cursor.execute("INSERT INTO users (username, email, name, active, date_created) VALUES (%s, %s, %s, %s, NOW())", (username, email, name, 1))
        db.commit()
        user_id = cursor.lastrowid
        user = User(user_id, username, email, name, 1)

    login_user(user)
    return redirect(url_for("account"))

@app.route("/account")
@login_required
def account():
    return f"Welcome, {current_user.name}! This is your account page."

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Use PORT from environment or default to 8080
    app.run(host='0.0.0.0', port=port, debug=True)  # Start the Flask app
