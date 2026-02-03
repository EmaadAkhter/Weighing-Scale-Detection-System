import argparse
from ultralytics import YOLO
from pathlib import Path

def test_model(model_path, data_yaml):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print("\n--- Validation Set Evaluation ---")
    # Validate on the 'val' split defined in data.yaml
    val_results = model.val(data=data_yaml, split='val')
    print(f"Validation mAP50: {val_results.box.map50}")
    print(f"Validation mAP50-95: {val_results.box.map}")

    print("\n--- Test Set Evaluation ---")
    # Validate on the 'test' split defined in data.yaml
    test_results = model.val(data=data_yaml, split='test')
    print(f"Test mAP50: {test_results.box.map50}")
    print(f"Test mAP50-95: {test_results.box.map}")
    
    return val_results, test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained YOLOv8 model")
    
    # Default path assumes the standard runs/detect/train structure from the previous step
    # We'll check for 'best.pt' in the most recent run or generic path
    default_model = "runs/detect/train/weights/best.pt" 
    
    parser.add_argument("--model", type=str, default=default_model, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="./v5-display-l.coco/data.yaml", help="Path to data.yaml")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please check the path or run training first.")
    else:
        test_model(args.model, args.data)
