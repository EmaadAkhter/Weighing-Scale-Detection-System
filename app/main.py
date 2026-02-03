
from fastapi import FastAPI
from app.routes import detection
from app.model.loader import ModelLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Weighing Scale Detection API",
    description="API for detecting weighing scales in images using YOLOv8.",
    version="1.0.0"
)

# Include Routers
app.include_router(detection.router)

@app.on_event("startup")
async def startup_event():
    """
    Initialize resources on startup.
    """
    logger.info("Initializing application...")
    ModelLoader().load_model() # Preload model

@app.get("/", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "service": "Scale Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8010, reload=True)
