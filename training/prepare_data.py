import json
import os
import shutil
from pathlib import Path

def setup_directories(base_path, subset):
    subset_path = base_path / subset
    images_dir = subset_path / "images"
    labels_dir = subset_path / "labels"
    
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    return subset_path, images_dir, labels_dir

def convert_bbox(size, box):
    # box: [x_min, y_min, width, height]
    # size: [width, height]
    # return: [x_center, y_center, width, height] normalized
    
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return [x, y, w, h]

def process_subset(base_path, subset):
    print(f"Processing {subset}...")
    subset_path, images_dir, labels_dir = setup_directories(base_path, subset)
    
    coco_file = subset_path / "_annotations.coco.json"
    if not coco_file.exists():
        print(f"No annotation file found for {subset}, skipping.")
        return

    with open(coco_file) as f:
        data = json.load(f)
    
    # Create image map
    images = {item['id']: item for item in data['images']}
    
    # Process images and move them
    for img_id, img_info in images.items():
        file_name = img_info['file_name']
        src_file = subset_path / file_name
        dst_file = images_dir / file_name
        
        # Move image if it exists in the root of subset dir
        if src_file.exists():
            shutil.move(str(src_file), str(dst_file))
        
    # Process annotations
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_info = images[img_id]
        file_name = img_info['file_name']
        
        # YOLO label annotation file name
        label_file = labels_dir / (Path(file_name).stem + ".txt")
        
        # Category mapping: map id 1 (weighing_scale) to 0
        cat_id = ann['category_id']
        if cat_id == 1:
            class_id = 0
        else:
            continue # Skip other categories if any
            
        bbox = convert_bbox((img_info['width'], img_info['height']), ann['bbox'])
        
        with open(label_file, "a") as f:
            f.write(f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n")

    print(f"Finished {subset}.")

def create_data_yaml(base_path):
    yaml_content = f"""
path: {base_path.absolute()} # dataset root dir
train: train/images
val: valid/images
test: test/images

# Classes
names:
  0: weighing_scale
"""
    with open(base_path / "data.yaml", "w") as f:
        f.write(yaml_content)
    print("Created data.yaml")

if __name__ == "__main__":
    dataset_root = Path("./My First Project.v5-display-l.coco")
    
    for subset in ["train", "valid", "test"]:
        process_subset(dataset_root, subset)
        
    create_data_yaml(dataset_root)
