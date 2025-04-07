import os
from pymongo import MongoClient
from datetime import datetime

# Get connection string from environment variable
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")  # Fallback to local if not set
client = MongoClient(MONGO_URI)
db = client.emotion_db  # Database name

def save_emotion(user_id: str, emotion: str):
    db.emotions.insert_one({
        "user_id": user_id,
        "emotion": emotion,
        "timestamp": datetime.now()
    })

def get_emotions(user_id: str):
    return list(db.emotions.find({"user_id": user_id}))