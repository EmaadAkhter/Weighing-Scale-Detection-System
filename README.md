# Weighing Scale Detection System

A robust, production-ready Object Detection system capable of identifying weighing scales in images. Built with **YOLOv8** and wrapped in a modular **FastAPI** service.

## Key Features

*   **State-of-the-Art Detection**: Powered by YOLOv8n (Nano) for an optimal balance of speed and accuracy.
*   **Production API**: Fast, async REST API built with FastAPI, supporting image upload and JSON result formats.
*   **CLI Tool**: Simple command-line interface for batch processing or local testing.
*   **End-to-End Testing**: Automated system-level tests verifying accuracy (IoU) against Ground Truth.
*   **Modular Architecture**: Clean separation of concerns (Model, Routes, Utils, Training).

---

## Technical Strategy & Decisions

### 1. Why YOLOv8 over Detectron2?
I initially considered Detectron2 (Faster R-CNN) but switched to **YOLOv8** for the following reasons:
*   **Efficiency**: YOLO (You Only Look Once) is a single-stage detector, making it significantly faster for real-time inference (API usage) compared to two-stage detectors like Faster R-CNN.
*   **Deployment**: The Ultralytics ecosystem provides seamless export options (ONNX, CoreML, TFLite), making future edge deployment easier.
*   **Performance**: On my specific *Weighing Scale* dataset, YOLOv8n achieved **98% mAP50** within just 30 epochs, proving it is highly capable for this task without the overhead of larger models.

### 2. Dataset Management
*   **Format**: Converted COCO JSON annotations to standard YOLO TXT format [class x_center y_center width height].
*   **Split**: utilized a Train/Val/Test split to ensure robust evaluation.
*   **Training**: Trained for 30 epochs with default augmentation to prevent overfitting on the small dataset.
*   **Download**: [Access Dataset Here](https://drive.google.com/drive/folders/1BWuMFP8JNYG5OT_zTo1JkrP_RXd7R5Dh?usp=sharing)

---

## Model Performance

The model was trained for 30 epochs on an M-series Mac (MPS acceleration).

### Metrics
| Metric | Score | Description |
| :--- | :--- | :--- |
| **mAP50** | **0.995** | Near perfect detection presence. |
| **mAP50-95** | **0.512** | High precision in bounding box alignment. |
| **Precision** | **0.995** | Low false positive rate. |
| **Recall** | **1.000** | Successfully found all objects in validation. |

### Training Graphs
#### Overall Results (Loss, Precision, Recall, mAP)
![Results](runs/detect/train/results.png)

#### Confusion Matrix
![Confusion Matrix](runs/detect/train/confusion_matrix.png)

#### F1 Curve
![F1 Curve](runs/detect/train/BoxF1_curve.png)

---

## Installation & Usage

### Option 1: Docker (Recommended for Production)

**Quick Start:**
```bash
# Build and run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop service
docker-compose down
```

**Manual Docker Build:**
```bash
# Build image
docker build -t scale-detection:latest .

# Run container
docker run -d -p 8010:8010 --name scale-detection-api scale-detection:latest
```

*   **API Docs**: `http://localhost:8010/docs`
*   **Health Check**: `http://localhost:8010/`

### Option 2: Local Development Setup

**1. Setup Environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Run the Web Service (API)**
Start the FastAPI server:
```bash
uvicorn app.main:app --port 8010 --reload
```
*   **Docs**: `http://localhost:8010/docs`
*   **Endpoint**: `POST /detect`
    *   `file`: Image file
    *   `format`: "image" (returns annotated jpeg) or "json" (returns coordinates)

**3. Run the CLI**
Detect objects in a local image:
```bash
python cli/main.py path/to/image.jpg --output ./output
```

**4. Run System Tests**
Verify the entire pipeline (Accuracy > 0.5 IoU required):
```bash
python scripts/test_system.py
```

**5. Training (Reproduce Results)**
```bash
python training/train.py
```

---

## Project Structure

```text
.
├── app/                          # Web Service (FastAPI)
│   ├── __init__.py
│   ├── main.py                   # FastAPI Application Entry Point
│   ├── dependencies/             # Dependency Injection
│   │   ├── __init__.py
│   │   └── core.py               # Model Dependency Provider
│   ├── model/                    # Model Management
│   │   ├── __init__.py
│   │   └── loader.py             # YOLOv8 Model Loader (Singleton)
│   ├── routes/                   # API Endpoints
│   │   ├── __init__.py
│   │   └── detection.py          # /detect Endpoint (Image & JSON)
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── image.py              # Image Processing Functions
│
├── cli/                          # Command Line Interface
│   ├── __init__.py
│   └── main.py                   # CLI Detection Tool
│
├── training/                     # Training & Evaluation
│   ├── train.py                  # YOLOv8 Training Script
│   ├── test.py                   # Model Evaluation Script
│   └── prepare_data.py           # COCO to YOLO Converter
│
├── scripts/                      # Testing & Verification
│   ├── __init__.py
│   └── test_system.py            # End-to-End System Tests (IoU Validation)
│
├── runs/                         # Training Artifacts (Auto-generated)
│   └── detect/
│       ├── train/                # Training Run Outputs
│       │   ├── weights/
│       │   │   ├── best.pt       # Best Model Weights
│       │   │   └── last.pt       # Last Epoch Weights
│       │   ├── results.png       # Training Metrics Graph
│       │   ├── results.csv       # Training Metrics CSV
│       │   ├── confusion_matrix.png
│       │   ├── BoxF1_curve.png
│       │   ├── BoxPR_curve.png
│       │   └── args.yaml         # Training Configuration
│       ├── val/                  # Validation Run 1
│       └── val2/                 # Validation Run 2
│
├── test_output/                  # Test Results (Auto-generated)
│   ├── annotated_*.jpg           # CLI Annotated Images
│   └── api_result.jpg            # API Test Result
│
├── v5-display-l.coco/            # Dataset (YOLO Format)
│   ├── data.yaml                 # Dataset Configuration
│   ├── train/
│   │   ├── images/               # Training Images
│   │   └── labels/               # Training Labels (.txt)
│   ├── valid/
│   │   ├── images/               # Validation Images
│   │   └── labels/               # Validation Labels (.txt)
│   └── test/
│       ├── images/               # Test Images
│       └── labels/               # Test Labels (.txt)
│
├── requirements.txt              # Python Dependencies
├── .gitignore                    # Git Ignore Rules
├── README.md                     # Project Documentation
└── yolov8n.pt                    # Pre-trained YOLOv8 Nano Model
```
