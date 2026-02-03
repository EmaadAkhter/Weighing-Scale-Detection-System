
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import Response
from ultralytics import YOLO
import logging

from app.dependencies.core import get_model
from app.utils.image import read_image_from_bytes, encode_image_to_bytes

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/detect", tags=["Detection"])
async def detect_object(
    file: UploadFile = File(...),
    format: str = "image",
    model: YOLO = Depends(get_model)
):
    """
    Endpoint to detect objects in an uploaded image.
    Params:
        format: "image" (default) or "json"
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and decode image
        contents = await file.read()
        img = read_image_from_bytes(contents)
        
        # Run inference
        results = model.predict(img)
        result = results[0]
        
        # Strategy: Keep only the highest confidence prediction
        best_box = None
        if len(result.boxes) > 0:
            max_idx = result.boxes.conf.argmax()
            result.boxes = result.boxes[max_idx.item()]
            
            # Extract box info for JSON
            box = result.boxes[0]
            best_box = {
                "class": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xywh.tolist()[0] # [x, y, w, h]
            }
            
        if format == "json":
            return {"detected": best_box is not None, "prediction": best_box}

        # Draw annotations
        annotated_img = result.plot()
        
        # Encode back to JPEG
        encoded_img_bytes = encode_image_to_bytes(annotated_img)
        
        return Response(content=encoded_img_bytes, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
