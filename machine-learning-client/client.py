import os
import logging
from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from inference_sdk import InferenceHTTPClient
from requests.exceptions import RequestException 

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
client = MongoClient(MONGO_URI)
db = client["rps_database"]
collection = db["predictions"]

API_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:9001")
API_KEY = os.getenv("ROBOFLOW_API_KEY")  
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
inference_client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            logging.error("No image file provided")
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files["image"]
        temp_path = f"./temp/{image_file.filename}"
        os.makedirs("./temp", exist_ok=True)
        image_file.save(temp_path)

        result = inference_client.infer(temp_path, model_id=MODEL_ID)

        prediction_data = result.get("predictions", [{}])[0]
        gesture = prediction_data.get("class", "Unknown")
        confidence = prediction_data.get("confidence", 0) if gesture != "Unknown" else 0

        record = {
            "gesture": gesture,
            "prediction_score": confidence,
            "image_metadata": {"filename": image_file.filename},
        }
        collection.insert_one(record)

        return jsonify({"gesture": gesture, "confidence": confidence})
    except (RequestException, PyMongoError, FileNotFoundError) as error:
        return jsonify({"error": f"Prediction error: {str(error)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
