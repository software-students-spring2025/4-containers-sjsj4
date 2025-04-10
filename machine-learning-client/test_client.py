import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from requests.exceptions import RequestException
from pymongo.errors import PyMongoError
from client import app
import os
os.environ["MODEL_ID"] = "fake_model_id"


@pytest.fixture
def client():
    """Creates and yields a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


# Happy path-> the prediction works
@patch("client.inference_client")
@patch("client.collection")
def test_successful_prediction(mock_db, mock_model, client):
    mock_model.infer.return_value = {"predictions": [{"class": "Rock", "confidence": 0.95}]}
    mock_db.insert_one.return_value = MagicMock()

    image_data = {"image": (BytesIO(b"fake image bytes"), "sample.jpg")}
    response = client.post("/predict", content_type="multipart/form-data", data=image_data)

    assert response.status_code == 200
    response_json = response.get_json()
    assert response_json["gesture"] == "Rock"
    assert response_json["confidence"] == 0.95
    mock_model.infer.assert_called_once()
    mock_db.insert_one.assert_called_once()


# No image sent
def test_missing_image_file(client):
    response = client.post("/predict", content_type="multipart/form-data", data={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "No image file provided"


# infrence service throws error
@patch("client.inference_client")
def test_inference_service_error(mock_model, client):
    mock_model.infer.side_effect = RequestException("Inference failure")

    image_data = {"image": (BytesIO(b"fake image bytes"), "sample.jpg")}
    response = client.post("/predict", content_type="multipart/form-data", data=image_data)

    assert response.status_code == 500
    assert "Prediction error" in response.get_json()["error"]


# db insertion fails
@patch("client.inference_client")
@patch("client.collection")
def test_database_insertion_failure(mock_db, mock_model, client):
    mock_model.infer.return_value = {"predictions": [{"class": "Paper", "confidence": 0.85}]}
    mock_db.insert_one.side_effect = PyMongoError("DB insert failure")

    image_data = {"image": (BytesIO(b"fake image bytes"), "sample.jpg")}
    response = client.post("/predict", content_type="multipart/form-data", data=image_data)

    assert response.status_code == 500
    assert "Prediction error" in response.get_json()["error"]