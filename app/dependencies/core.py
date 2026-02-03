
from app.model.loader import ModelLoader

# Global instance
model_loader = ModelLoader()

def get_model():
    """
    Dependency to provide the YOLO model.
    """
    return model_loader.get_model()
