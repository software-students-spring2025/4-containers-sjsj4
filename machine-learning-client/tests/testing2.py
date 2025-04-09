import base64
import io
from datetime import datetime
import numpy as np
from PIL import Image
import pymongo
from deepface import DeepFace

class MoodDetector:
    """Handles emotion detection using DeepFace and MongoDB logging."""

    def __init__(self, mongo_uri: str = "mongodb://mongodb:27017/"):
        self.client = pymongo.MongoClient(mongo_uri)
        self.collection = self.client["mood_db"]["mood_data"]

    def _decode_image(self, base64_image: str) -> np.ndarray:
        """Convert base64 image string to numpy array."""
        if ',' in base64_image: # Remove data URL prefix if present
            base64_image = base64_image.split(',')[1]
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
            return np.array(image.convert('RGB'))

    def analyze_frame(self, base64_image: str) -> str:
        """
        Analyze an image and return dominant emotion.
        Stores result in MongoDB with timestamp.
        """
        try:
            img_np = self._decode_image(base64_image)

            # DeepFace analysis (disable unnecessary actions for speed)
            result = DeepFace.analyze(
                img_np,
                actions=['emotion'],
                detector_backend='opencv', # Faster than default
                enforce_detection=False # Return neutral if no face
            )

            dominant_emotion = result[0]['dominant_emotion']

            # Minimal MongoDB logging
            self.collection.insert_one({
                "emotion": dominant_emotion,
                "timestamp": datetime.utcnow()
            })

            return dominant_emotion

        except Exception as e:
            # Log errors for debugging
            self.collection.insert_one({
                "error": str(e),
                "timestamp": datetime.utcnow()
            })
            return "neutral" # Fallback