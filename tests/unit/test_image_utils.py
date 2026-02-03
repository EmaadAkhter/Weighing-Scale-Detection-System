import numpy as np
from app.utils.image import read_image_from_bytes, encode_image_to_bytes

def test_read_image_from_bytes(sample_image_bytes):
    img = read_image_from_bytes(sample_image_bytes)
    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 100, 3)
    # Check if the center pixel is close to red (BGR)
    # JPEG compression might slightly alter values
    pixel = img[50, 50]
    assert pixel[0] < 10  # Blue
    assert pixel[1] < 10  # Green
    assert pixel[2] > 240 # Red

def test_encode_image_to_bytes():
    # Create a dummy image
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :] = [255, 0, 0] # Blue
    
    encoded = encode_image_to_bytes(img, ext=".png")
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
    
    # Verify we can decode it back
    decoded = read_image_from_bytes(encoded)
    assert decoded.shape == (50, 50, 3)
    assert np.all(decoded == [255, 0, 0])
