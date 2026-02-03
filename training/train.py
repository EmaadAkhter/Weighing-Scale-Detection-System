#!/usr/bin/env python3
"""
Training script for Weighing Scale Detection using YOLOv8 (Ultralytics).
Replaces previous Detectron2 implementation.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def train_yolo(data_dir, output_dir, epochs=30, batch_size=32, device='cpu'):
    # Handle data path: if directory provided, look for data.yaml inside
    data_path = Path(data_dir)
    if data_path.is_dir():
        data_yaml = data_path / "data.yaml"
    else:
        data_yaml = data_path
        
    if not data_yaml.exists():
        raise FileNotFoundError(f"Could not find data config at {data_yaml}")

    print(f"Using data config: {data_yaml}")
    
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    print(f"Starting training on device: {device}")
    
    # Separate project (parent dir) and name (subdir) from output_dir
    output_path = Path(output_dir)
    project = str(output_path.parent)
    name = output_path.name

    # Train the model
    # Note: Ultralytics handles 'mps' automatically if passed as device
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        plots=True
    )
    
    print(f"Training complete. Results saved to {output_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Weighing Scale Detection")
    parser.add_argument("--data_dir", type=str, default="./v5-display-l.coco", help="Path to dataset directory (containing data.yaml) or path to data.yaml")
    parser.add_argument("--output_dir", type=str, default="runs/detect/train", help="Directory where results will be saved")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="mps", help="Device to train on (cpu, mps, cuda)")
    
    args = parser.parse_args()
    
    train_yolo(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.device)
