"""This is the main file of the website."""

import mysql.connector
import os
from flask import Flask, render_template, request, flash, redirect, url_for, session
from flask_mail import Mail, Message
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import smtplib
from oauthlib.oauth2 import WebApplicationClient
import requests
import json
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


@app.route('/')
def home():
    """Home page of the website"""
    return render_template('home.html')

@app.route('/about')
def about():
    """About page of the website"""
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact page of the website"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Here, you would handle the form submission, e.g., save to database or send email
        flash(f"Thank you, {name}. Your message has been sent!", "success")

    return render_template('contact.html')

@app.route('/catalogue')
def catalogue():
    """Catalogue page of the website"""
    return render_template('catalogue.html')

@app.route('/privacy')
def privacy():
    """Privacy page of the website"""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Terms page of the website"""
    return render_template('terms.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
