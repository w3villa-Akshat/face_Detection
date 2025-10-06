from flask import Flask, render_template, request, redirect, url_for, flash
from register import register  # import your function
import cv2
import time


app = Flask(__name__)
app.secret_key = 'supersecretkey'  # needed for flash messages
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "POST":
        name = request.form["name"]
        # Call your function directly
    
        reg_name = register(name)
        if reg_name == name:
            flash(f"User {name} registered successfully!", "success")
        else:
            flash(f"Failed to register user {name}.", "danger")
            return redirect(url_for("register_page"))
    return render_template("register.html")

if __name__ == "__main__":
    app.run(debug=True)
