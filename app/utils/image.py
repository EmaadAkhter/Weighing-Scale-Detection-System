
import cv2
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

def read_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Decodes an image from bytes (standard upload format).
    
    Args:
        data (bytes): Raw image bytes.
        
    Returns:
        np.ndarray: Decoded image in BGR format.
    """
    try:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise ValueError("Invalid image data")

def encode_image_to_bytes(img: np.ndarray, ext: str = ".jpg") -> bytes:
    """
    Encodes an OpenCV image to bytes.
    
    Args:
        img (np.ndarray): Image array (BGR).
        ext (str): File extension for encoding (e.g., '.jpg', '.png').
        
    Returns:
        bytes: Encoded image bytes.
    """
    try:
        _, encoded_img = cv2.imencode(ext, img)
        return encoded_img.tobytes()
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise ValueError("Failed to encode image")
