from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    """
    Application configuration with environment variable support.
    
    Environment variables can be set in .env file or system environment.
    Example: MODEL_PATH=/path/to/model.pt
    """
    
    # Model Configuration
    model_path: str = "runs/detect/train/weights/best.pt"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8010
    
    # Detection Parameters
    confidence_threshold: float = 0.1
    iou_threshold: float = 0.5
    detection_strategy: Literal['best', 'all'] = 'best'
    top_n_detections: int = 1
    
    # File Upload Limits
    max_file_size_mb: int = 10
    allowed_file_types: list[str] = ["image/jpeg", "image/jpg", "image/png"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
