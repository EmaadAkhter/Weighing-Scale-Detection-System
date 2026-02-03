import threading
from ultralytics import YOLO
from pathlib import Path
import logging
from app.config import settings

# Configure logger
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Thread-safe singleton class to handle model loading.
    Ensures the model is loaded only once and reused across the application.
    """
    _instance = None
    _model = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_path: str = None):
        """
        Loads the YOLO model with thread-safety.
        
        Args:
            model_path (str): Path to the .pt model file. Defaults to config settings.
        
        Returns:
            YOLO: The loaded YOLO model or None if loading fails.
        """
        path = model_path or settings.model_path
        
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        logger.info(f"Loading model from {path}...")
                        self._model = YOLO(path)
                        logger.info("Model loaded successfully.")
                    except Exception as e:
                        logger.error(f"Failed to load model from {path}: {e}")
                        self._model = None
        
        return self._model

    def get_model(self):
        """
        Returns the loaded model instance. If not loaded, attempts to load it.
        """
        if self._model is None:
            self.load_model()
        return self._model
