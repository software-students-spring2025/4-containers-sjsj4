from unittest.mock import patch, MagicMock
from io import BytesIO
from flask.testing import FlaskClient
import pytest
from bson.objectid import ObjectId
import requests
from app import app, generate_stats_doc, retry_request


@pytest.fixture(name="flask_client")
def flask_client_fixture():
    app.config["TESTING"] = True
    with app.test_client() as temp_client:
        yield temp_client


@patch("app.collection")
def test_generate_stats_doc(mockCollection):
    mockId = ObjectId()
    mockCollection.insert_one.return_value.inserted_id = mockId
    _id = generate_stats_doc()
    mockCollection.insert_one.assert_called_once()
    assert _id == str(mockId)

@patch("app.requests.post")
def test_retry_request_success_on_first_try(mockPost):
    mockResponse = MagicMock()
    mockResponse.raise_for_status.return_value = None
    mockPost.return_value = mockResponse
    url = "http://example.com/predict"
    files = {"image": MagicMock()}
    response = retry_request(url, files)
    assert response == mockResponse
    mockPost.assert_called_once()


@patch("app.requests.post")
def test_retry_request_success_after_retries(mockPost):
    mockResponse = MagicMock()
    mockResponse.raise_for_status.side_effect = [
        requests.exceptions.HTTPError("Connection error"),
        requests.exceptions.HTTPError("Timeout"),
        None,
        ]
    mockPost.return_value = mockResponse

    url = "http://example.com/predict"
    files = {"image": MagicMock()}

    response = retry_request(url, files, retries=3, delay=0)
    assert response == mockResponse
    assert mockPost.call_count == 3
    assert mockResponse.raise_for_status.call_count == 3


@patch("app.requests.post")
def test_retry_request_all_failures(mockPost):
    mockResponse = MagicMock()
    mockResponse.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Connection error"
    )
    mockPost.return_value = mockResponse

    url = "http://example.com/predict"
    files = {"image": MagicMock()}

    response = retry_request(url, files, retries=3, delay=0)
    assert response is None
    assert mockPost.call_count == 3
    assert mockResponse.raise_for_status.call_count == 3


@patch("app.generate_stats_doc", return_value=str(ObjectId()))
def test_home_route(mock_generate_stats_doc, flask_client):
    response = flask_client.get("/")
    _ = mock_generate_stats_doc
    assert response.status_code == 200
    assert "db_object_id" in response.headers["Set-Cookie"]


@patch("app.generate_stats_doc", return_value=str(ObjectId()))
def test_index_route(mock_generate_stats_doc, flask_client: FlaskClient):
    _ = mock_generate_stats_doc
    response = flask_client.get("/index")
    assert response.status_code == 200


@patch("app.retry_request")
@patch("app.collection.update_one")
def test_result_route_success(
    mock_retry_request, mock_update_one, flask_client: FlaskClient
):
    mockResponse = MagicMock()
    mockResponse.json.return_value = {"gesture": "Scissors"}
    mock_retry_request.return_value = mockResponse

    mockId = ObjectId()
    flask_client.set_cookie("db_object_id", str(mockId))

    data = {"image": (BytesIO(b"fake image data"), "test_image.jpg")}
    response = flask_client.post(
        "/result", data=data, content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"AI wins!" in response.data
    _ = mock_update_one


@patch("app.retry_request")
def test_result_route_unknown_gesture(mock_retry_request, flask_client: FlaskClient):
    mockResponse = MagicMock()
    mockResponse.json.return_value = {"gesture": "Unknown"}
    mock_retry_request.return_value = mockResponse

    mockId = ObjectId()
    flask_client.set_cookie("db_object_id", str(mockId))

    data = {"image": (BytesIO(b"fake image data"), "test_image.jpg")}
    response = flask_client.post(
        "/result", data=data, content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Gesture not recognized" in response.data


@patch("app.retry_request")
def test_result_route_no_image(mock_retry_request, flask_client: FlaskClient):
    response = flask_client.post("/result", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"No image file provided" in response.data
    mock_retry_request.assert_not_called()


@patch("app.generate_stats_doc", return_value=str(ObjectId()))
def test_home_route_creates_new_stats_doc(mock_generate_stats_doc, flask_client):
    response = flask_client.get("/")
    assert response.status_code == 200
    assert "db_object_id" in response.headers.get("Set-Cookie", "")
    mock_generate_stats_doc.assert_called_once()

    
