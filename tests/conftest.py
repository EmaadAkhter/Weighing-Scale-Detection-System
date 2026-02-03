import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def api_client():
    return TestClient(app)

@pytest.fixture
def sample_image_bytes():
    import cv2
    import numpy as np
    # Create a simple red square image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [0, 0, 255]
    _, encoded_img = cv2.imencode(".jpg", img)
    return encoded_img.tobytes()
