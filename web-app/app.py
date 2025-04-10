import os
import time
import random
import logging  
from flask import Flask, render_template, request, jsonify, make_response
import requests
from requests.exceptions import RequestException
from pymongo import MongoClient
from bson.objectid import ObjectId

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
client = MongoClient(MONGO_URI)
db = client["rps_database"]
collection = db["stats"]

def generate_stats_doc():
    statistics = {
        "Rock": {"wins": 0, "losses": 0, "ties": 0, "total": 0},
        "Paper": {"wins": 0, "losses": 0, "ties": 0, "total": 0},
        "Scissors": {"wins": 0, "losses": 0, "ties": 0, "total": 0},
        "Totals": {"wins": 0, "losses": 0, "ties": 0},
    }
    _id = str(collection.insert_one(statistics).inserted_id)
    return _id

def retry_request(url, files, retries=5, delay=2, timeout=10):
    for attempt in range(retries):
        try:
            resp = requests.post(url, files=files, timeout=timeout)
            resp.raise_for_status()
            return resp
        except RequestException as error:
            logging.warning("Retry attempt %d failed: %s", attempt + 1, str(error))
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error("All retry attempts failed.")
                return None
    return None

@app.route("/")
def home():
    resp = make_response(render_template("index.html"))
    if "db_object_id" not in request.cookies:
        resp.set_cookie("db_object_id", generate_stats_doc())
    return resp


@app.route("/index")
def index():
    resp = make_response(render_template("index.html"))
    if "db_object_id" not in request.cookies:
        resp.set_cookie("db_object_id", generate_stats_doc())
    return resp


@app.route("/result", methods=["POST"])
def result():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        file = request.files["image"]
        mlClientUrl = os.getenv(
            "ML_CLIENT_URL", "http://machine-learning-client:5000"
        )
        resp = retry_request(f"{mlClientUrl}/predict", files={"image": file})
        if not resp:
            return render_template(
                "result.html",
                user="Unknown",
                ai=random.choice(["Rock", "Paper", "Scissors"]),
                result="No valid prediction. Please try again.",
            )
        userGesture = resp.json().get("gesture", "Unknown")
        if userGesture == "Unknown":
            return render_template(
                "result.html",
                user="Unknown",
                ai=random.choice(["Rock", "Paper", "Scissors"]),
                result="Gesture not recognized. Please try again.",
            )
    except RequestException as error:
        return jsonify({"error": "Error communicating with ML client"}), 500
    aiGesture = random.choice(["Rock", "Paper", "Scissors"])
    winner = determine_winner(userGesture, aiGesture)
    _id = request.cookies.get("db_object_id")
    if winner == "AI wins!":
        res = "losses"
    elif winner == "It's a tie!":
        res = "ties"
    else:
        res = "wins"
    collection.update_one(
        {"_id": ObjectId(_id)},
        {
            "$inc": {
                "Totals" + "." + res: 1,
                userGesture + "." + res: 1,
                userGesture + "." + "total": 1,
            }
        },
        upsert=False,
    )
    return render_template(
        "result.html", user=userGesture, ai=aiGesture, result=winner
    )


def determine_winner(user, aiChoice):
    winningCases = {
        "Rock": "Scissors",
        "Paper": "Rock",
        "Scissors": "Paper",
    }
    if user == aiChoice:
        return "It's a tie!"
    if aiChoice == winningCases.get(user):
        return "You win!"
    return "AI wins!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
