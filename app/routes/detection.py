from enum import Enum
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import Response
from ultralytics import YOLO
import logging

from app.dependencies.core import get_model
from app.utils.image import read_image_from_bytes, encode_image_to_bytes
from app.utils.detection import filter_detections, get_detection_info
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

class OutputFormat(str, Enum):
    """Output format for detection results"""
    IMAGE = "image"
    JSON = "json"

@router.post("/detect", tags=["Detection"])
async def detect_object(
    file: UploadFile = File(...),
    format: OutputFormat = Query(OutputFormat.IMAGE, description="Output format: image or json"),
    confidence: float = Query(
        settings.confidence_threshold,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections"
    ),
    iou: float = Query(
        settings.iou_threshold,
        ge=0.0,
        le=1.0,
        description="IOU threshold for NMS"
    ),
    model: YOLO = Depends(get_model)
):
    """
    Detect weighing scales in an uploaded image.
    
    Args:
        file: Image file (JPEG/PNG, max 10MB)
        format: Output format - 'image' returns annotated image, 'json' returns coordinates
        confidence: Minimum confidence threshold (0.0-1.0)
        iou: IOU threshold for Non-Maximum Suppression (0.0-1.0)
    
    Returns:
        - format=image: Annotated JPEG image
        - format=json: JSON with detection info {detected: bool, prediction: {...}}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if file.content_type not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_file_types)}"
        )
    
    try:
        # Read file with size limit
        contents = await file.read()
        max_size = settings.max_file_size_mb * 1024 * 1024
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Decode image
        img = read_image_from_bytes(contents)
        
        # Run inference with configurable thresholds
        results = model.predict(img, conf=confidence, iou=iou)
        result = results[0]
        
        # Apply detection filtering strategy
        result = filter_detections(
            result,
            strategy=settings.detection_strategy,
            top_n=settings.top_n_detections
        )
        
        # Return JSON if requested
        if format == OutputFormat.JSON:
            detection_info = get_detection_info(result)
            return {
                "detected": detection_info is not None,
                "prediction": detection_info
            }
        
        # Draw annotations and return image
        annotated_img = result.plot()
        encoded_img_bytes = encode_image_to_bytes(annotated_img)
        
        return Response(content=encoded_img_bytes, media_type="image/jpeg")

    except HTTPException:
        # Re-raise HTTP exceptions (400, 503)
        raise
    except Exception as e:
        # Log and return 500 for unexpected errors
        logger.error(f"Unexpected error during detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
