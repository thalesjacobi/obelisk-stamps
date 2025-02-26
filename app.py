"""This is the main file of the website."""

from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

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
