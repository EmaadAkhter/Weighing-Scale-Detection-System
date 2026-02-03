
from fastapi import FastAPI
from app.routes import detection
from app.model.loader import ModelLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    logger.info("Initializing application...")
    try:
        ModelLoader().load_model() # Preload model
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
    
    yield
    
    logger.info("Shutting down application...")

# Initialize FastAPI App
app = FastAPI(
    title="Weighing Scale Detection API",
    description="API for detecting weighing scales in images using YOLOv8.",
    version="1.0.0",
    lifespan=lifespan
)

# Include Routers
app.include_router(detection.router)

@app.get("/", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "service": "Scale Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8010, reload=True)
