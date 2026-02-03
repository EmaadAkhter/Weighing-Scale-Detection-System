import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_health_check(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Scale Detection API"}

def test_detect_image_endpoint(api_client, sample_image_bytes):
    # Test image output format
    response = api_client.post(
        "/detect?format=image",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_detect_json_endpoint(api_client, sample_image_bytes):
    # Test json output format
    response = api_client.post(
        "/detect?format=json",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "detected" in data
    assert "prediction" in data

def test_invalid_file_type(api_client):
    # Test with a text file
    response = api_client.post(
        "/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_file_too_large(api_client, monkeypatch):
    # Mock settings to have small max size
    from app.config import settings
    monkeypatch.setattr(settings, "max_file_size_mb", 0) # 0MB limit
    
    response = api_client.post(
        "/detect",
        files={"file": ("test.jpg", b"small data", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]

def test_not_found_endpoint(api_client):
    response = api_client.get("/non-existent")
    assert response.status_code == 404
