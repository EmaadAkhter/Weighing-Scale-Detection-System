import argparse
import sys
from pathlib import Path
import cv2
import logging

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from app.utils.detection import filter_detections, get_detection_info
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_and_save(model_path: str, image_path: str, output_dir: str, confidence: float = None, iou: float = None):
    """
    Runs detection on an image and saves the result.
    
    Args:
        model_path: Path to the model weights
        image_path: Path to the input image
        output_dir: Directory to save the output
        confidence: Confidence threshold (uses config default if None)
        iou: IOU threshold for NMS (uses config default if None)
    """
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return

    try:
        # Load model
        model = YOLO(model_path)
        
        # Use config defaults if not specified
        conf_threshold = confidence if confidence is not None else settings.confidence_threshold
        iou_threshold = iou if iou is not None else settings.iou_threshold
        
        # Run inference with configurable thresholds
        results = model.predict(source=image_path, save=False, conf=conf_threshold, iou=iou_threshold) 
        result = results[0]
        
        # Apply shared detection filtering
        result = filter_detections(
            result,
            strategy=settings.detection_strategy,
            top_n=settings.top_n_detections
        )
        
        # Extract and print detection info for system tests
        detection_info = get_detection_info(result)
        if detection_info:
            print(f"DETECTION_RESULT: {detection_info}")
        else:
            print("DETECTION_RESULT: None")

        # Save annotated image
        output_path_dir = Path(output_dir)
        output_path_dir.mkdir(parents=True, exist_ok=True)
        
        im_array = result.plot()
        original_filename = Path(image_path).name
        output_file = output_path_dir / f"annotated_{original_filename}"
        
        cv2.imwrite(str(output_file), im_array)
        logger.info(f"Saved annotated image to: {output_file}")
            
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="CLI for Weighing Scale Detection")
    
    parser.add_argument("--model", type=str, default=settings.model_path, help="Path to model weights")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="output", help="Output directory for annotated image")
    parser.add_argument("--confidence", type=float, default=None, help=f"Confidence threshold (default: {settings.confidence_threshold})")
    parser.add_argument("--iou", type=float, default=None, help=f"IOU threshold for NMS (default: {settings.iou_threshold})")
    
    args = parser.parse_args()
    
    detect_and_save(args.model, args.image, args.output, args.confidence, args.iou)

if __name__ == "__main__":
    main()
