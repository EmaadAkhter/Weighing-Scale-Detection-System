
from ultralytics import YOLO
from pathlib import Path
import logging

# Configure logger
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Singleton class to handle model loading.
    Ensures the model is loaded only once and reused.
    """
    _instance = None
    _model = None
    _model_path = "runs/detect/train/weights/best.pt"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_path: str = None):
        """
        Loads the YOLO model.
        
        Args:
            model_path (str): Path to the .pt model file. Defaults to internal default.
        
        Returns:
            YOLO: The loaded YOLO model or None if loading fails.
        """
        path = model_path or self._model_path
        
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
