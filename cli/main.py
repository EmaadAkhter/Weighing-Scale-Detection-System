
import argparse
from pathlib import Path
import cv2
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_and_save(model_path: str, image_path: str, output_dir: str):
    """
    Runs detection on an image and saves the result.
    
    Args:
        model_path (str): Path to the model weights.
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output.
    """
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return

    try:
        # Load model
        model = YOLO(model_path)
        
        # Run inference
        results = model.predict(source=image_path, save=False) 
        result = results[0]
        
        # Keep only the highest confidence prediction
        if len(result.boxes) > 0:
            max_idx = result.boxes.conf.argmax()
            result.boxes = result.boxes[max_idx.item()]
            
            # Print result for capture
            box = result.boxes[0]
            print(f"DETECTION_RESULT: {{'class': {int(box.cls)}, 'confidence': {float(box.conf)}, 'bbox': {box.xywh.tolist()[0]}}}")
        else:
            print("DETECTION_RESULT: None")

        output_path_dir = Path(output_dir)
        output_path_dir.mkdir(parents=True, exist_ok=True)
        
        # Save annotated image
        im_array = result.plot()
        original_filename = Path(image_path).name
        output_file = output_path_dir / f"annotated_{original_filename}"
        
        cv2.imwrite(str(output_file), im_array)
        logger.info(f"Saved annotated image to: {output_file}")
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")

def main():
    parser = argparse.ArgumentParser(description="CLI for Weighing Scale Detection")
    
    default_model = "runs/detect/train/weights/best.pt"
    
    parser.add_argument("--model", type=str, default=default_model, help="Path to model weights")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="output", help="Output directory for annotated image")
    
    args = parser.parse_args()
    
    detect_and_save(args.model, args.image, args.output)

if __name__ == "__main__":
    main()
