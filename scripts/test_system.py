
import subprocess
import requests
import time
import os
import signal
import json
import random
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Config
DATASET_DIR = Path("v5-display-l.coco")
API_URL = "http://127.0.0.1:8010"
CLI_SCRIPT = "cli/main.py"
OUTPUT_DIR = "test_output"

# IoU Threshold for considering a detection "correct"
IOU_THRESHOLD = 0.5 

def compute_iou(box1, box2):
    """
    Computes IoU between two boxes in [x_center, y_center, width, height] format.
    """
    # Convert to [x1, y1, x2, y2]
    def to_xyxy(b):
        xc, yc, w, h = b
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2
        return [x1, y1, x2, y2]

    b1 = to_xyxy(box1)
    b2 = to_xyxy(box2)

    # Intersection
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Union
    b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = b1_area + b2_area - inter_area
    
    if union_area == 0: return 0
    return inter_area / union_area

def load_ground_truth(subset):
    """
    Loads ground truth annotations for a subset (test/valid).
    Returns a list of dicts: {'image_path': Path, 'bbox': [x,y,w,h]}
    Only returns images that HAVE annotations (category_id=1).
    """
    ann_file = DATASET_DIR / subset / "_annotations.coco.json"
    img_dir = DATASET_DIR / subset / "images"
    
    if not ann_file.exists():
        logger.error(f"Annotation file not found: {ann_file}")
        return []

    with open(ann_file) as f:
        data = json.load(f)
        
    images = {item['id']: item for item in data['images']}
    samples = []
    
    for ann in data['annotations']:
        # We only care about weighing_scale (id=1 usually, check your data)
        # Using implicit assumption or mapped id. 
        # In this dataset user said category 1 is weighing_scale
        if ann.get('category_id') != 1: 
            continue
            
        img_info = images[ann['image_id']]
        img_path = img_dir / img_info['file_name']
        
        if img_path.exists():
            # COCO bbox is [x_min, y_min, width, height]
            # YOLO expects [x_center, y_center, width, height]
            # But wait, our API returns [x_center, y_center, w, h] because YOLO output is usually that.
            # Let's standardize on xywh (center based) for IoU calculation logic above?
            # Actually, compute_iou helper handles conversion if we define what it expects.
            # Let's convert COCO [top-left] to [center] here to match API output.
            
            coco_box = ann['bbox'] # x,y,w,h (top-left)
            w, h = coco_box[2], coco_box[3]
            xc = coco_box[0] + w/2
            yc = coco_box[1] + h/2
            
            samples.append({
                'image_path': img_path,
                'bbox': [xc, yc, w, h],
                'id': ann['id']
            })
            
    return samples

def run_tests():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Select samples
    test_samples = load_ground_truth("test")
    valid_samples = load_ground_truth("valid")
    
    all_samples = test_samples + valid_samples
    if not all_samples:
        logger.error("No samples found.")
        return

    # Pick 3 random samples
    selected_samples = random.sample(all_samples, min(3, len(all_samples)))
    
    logger.info(f"Selected {len(selected_samples)} samples for verification.")
    
    # ------------------
    # Start API Server
    # ------------------
    logger.info("Starting API Server...")
    python_executable = sys.executable
    uvicorn_executable = str(Path(python_executable).parent / "uvicorn")
    
    server_process = subprocess.Popen(
        [uvicorn_executable, "app.main:app", "--port", "8010", "--host", "127.0.0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(5) # warmup

    passed_count = 0
    
    try:
        for i, sample in enumerate(selected_samples):
            img_path = sample['image_path']
            gt_bbox = sample['bbox']
            
            logger.info(f"\n--- Test Case {i+1}: {img_path.name} ---")
            
            # 1. Test CLI
            # We capture stdout looking for "DETECTION_RESULT: {...}"
            cmd = [python_executable, CLI_SCRIPT, str(img_path), "--output", OUTPUT_DIR]
            cli_res = subprocess.run(cmd, capture_output=True, text=True)
            
            cli_bbox = None
            for line in cli_res.stdout.splitlines():
                if "DETECTION_RESULT:" in line:
                    val = line.split("DETECTION_RESULT:")[1].strip()
                    if val != "None":
                        # safe eval? it's our own string
                        import ast
                        data = ast.literal_eval(val)
                        cli_bbox = data['bbox']
            
            # 2. Test API
            with open(img_path, "rb") as f:
                # pass format=json
                resp = requests.post(f"{API_URL}/detect?format=json", files={"file": f})
            
            api_bbox = None
            if resp.status_code == 200:
                data = resp.json()
                if data.get("detected"):
                    api_bbox = data['prediction']['bbox']
            
            # Verification
            logger.info(f"Ground Truth (xywh): {gt_bbox}")
            
            iou_cli = 0.0
            iou_api = 0.0
            
            if cli_bbox:
                iou_cli = compute_iou(gt_bbox, cli_bbox)
                logger.info(f"CLI Prediction: {cli_bbox} | IoU: {iou_cli:.2f}")
            else:
                logger.warning("CLI detected nothing.")
                
            if api_bbox:
                iou_api = compute_iou(gt_bbox, api_bbox)
                logger.info(f"API Prediction: {api_bbox} | IoU: {iou_api:.2f}")
            else:
                logger.warning("API detected nothing.")
            
            # Check pass
            if iou_cli > IOU_THRESHOLD and iou_api > IOU_THRESHOLD:
                logger.info("✅ Result: CORRECT")
                passed_count += 1
            else:
                logger.error("❌ Result: INCORRECT (Low IoU or Missed)")

    finally:
        server_process.terminate()
        server_process.wait()

    logger.info(f"\nTotal Passed: {passed_count}/{len(selected_samples)}")
    if passed_count == len(selected_samples):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
