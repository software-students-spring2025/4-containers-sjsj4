"""Test suite for the ML client."""

import os
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from requests.exceptions import RequestException
from pymongo.errors import PyMongoError

from client import app

# Set environment variable for testing
os.environ["MODEL_ID"] = "fake_model_id"


@pytest.fixture
def flask_client():
    """Creates and yields a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


# prediction should work 
@patch("client.inference_client")
@patch("client.collection")
def test_successful_prediction(mock_db, mock_model, flask_client):
    mock_model.infer.return_value = {"predictions": [{"class": "Rock", "confidence": 0.95}]}
    mock_db.insert_one.return_value = MagicMock()

    image_data = {"image": (BytesIO(b"fake image bytes"), "sample.jpg")}
    response = flask_client.post("/predict", content_type="multipart/form-data", data=image_data)

    assert response.status_code == 200
    response_json = response.get_json()
    assert response_json["gesture"] == "Rock"
    assert response_json["confidence"] == 0.95
    mock_model.infer.assert_called_once()
    mock_db.insert_one.assert_called_once()


# No image sent
def test_missing_image_file(flask_client):
    response = flask_client.post("/predict", content_type="multipart/form-data", data={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "No image file provided"


# Inference service throws error
@patch("client.inference_client")
def test_inference_service_error(mock_model, flask_client):
    mock_model.infer.side_effect = RequestException("Inference failure")

    image_data = {"image": (BytesIO(b"fake image bytes"), "sample.jpg")}
    response = flask_client.post("/predict", content_type="multipart/form-data", data=image_data)

    assert response.status_code == 500
    assert "Prediction error" in response.get_json()["error"]


# DB insertion fails
@patch("client.inference_client")
@patch("client.collection")
def test_database_insertion_failure(mock_db, mock_model, flask_client):
    mock_model.infer.return_value = {"predictions": [{"class": "Paper", "confidence": 0.85}]}
    mock_db.insert_one.side_effect = PyMongoError("DB insert failure")

    image_data = {"image": (BytesIO(b"fake image bytes"), "sample.jpg")}
    response = flask_client.post("/predict", content_type="multipart/form-data", data=image_data)

    assert response.status_code == 500
    assert "Prediction error" in response.get_json()["error"]


# File saving failure
@patch("client.inference_client")
@patch("client.collection")
@patch("os.makedirs")
def test_file_saving_error(mock_os, mock_db, mock_model, flask_client):
    mock_model.infer.return_value = {"predictions": [{"class": "Scissors", "confidence": 0.90}]}

    with patch("werkzeug.datastructures.FileStorage.save", side_effect=FileNotFoundError("Save error")):
        image_data = {"image": (BytesIO(b"fake image bytes"), "sample.jpg")}
        response = flask_client.post("/predict", content_type="multipart/form-data", data=image_data)

        assert response.status_code == 500
        assert "Prediction error" in response.get_json()["error"]


# Model returns invalid response missing c key
@patch("client.inference_client")
@patch("client.collection")
def test_invalid_model_response(mock_db, mock_model, flask_client):
    mock_model.infer.return_value = {"predictions": [{"confidence": 0.80}]}

    image_data = {"image": (BytesIO(b"fake image bytes"), "sample.jpg")}
    response = flask_client.post("/predict", content_type="multipart/form-data", data=image_data)

    assert response.status_code == 200
    response_json = response.get_json()
    assert response_json["gesture"] == "Unknown"
    assert response_json["confidence"] == 0


# Large image upload
@patch("client.inference_client")
@patch("client.collection")
def test_large_image_upload(mock_db, mock_model, flask_client):
    mock_model.infer.return_value = {"predictions": [{"class": "Rock", "confidence": 0.92}]}
    mock_db.insert_one.return_value = MagicMock()

    large_image = BytesIO(b"x" * (5 * 1024 * 1024))  # 5MB dummy image
    image_data = {"image": (large_image, "large_image.jpg")}
    response = flask_client.post("/predict", content_type="multipart/form-data", data=image_data)

    assert response.status_code == 200
    response_json = response.get_json()
    assert response_json["gesture"] == "Rock"
