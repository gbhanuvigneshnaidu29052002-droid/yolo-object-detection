# YOLOv8 Object Detection & Counting (YOLO DETCTION)

This repository contains the implementation of **Block 2 - Object Detection** for the Image Processing & Computer Vision course. The project involves fine-tuning a YOLOv8 network on a custom annotated dataset to detect objects (traffic cones) and perform real-time instance counting.

---

## 📂 Project Structure

```text
YOLO DETCTION/
├── My First Project.yolov8/            # Main project environment containing scripts
│   ├── 11/                             # Folder with core training/detection scripts
│   │   ├── detect_and_count.py        # Counts and locates cones in test images
│   │   ├── evaluate_metrics.py        # Custom precision/recall metrics calculation
│   │   ├── split_dataset.py           # Prepares training/validation splits
│   │   ├── train.py                   # Core YOLOv8 training script
│   │   ├── train_and_detect.py        # Combined pipeline training & detection runner
│   │   ├── yolo_pipeline_fixed.py     # Cleaned-up object detection pipeline
│   │   └── your_script_name.py        # Command line arguments parser for inference
│   ├── anti.py                         # Deepseek/Gemini pipeline interface script
│   ├── deepseek_pipeline.py            # Custom testing/metrics processing pipeline
│   ├── T1.py / T1deep.py               # Custom inference scripts
│   ├── data.yaml                       # YOLOv8 dataset configuration (links image paths)
│   ├── yolo26n.pt / yolov8n.pt         # Pretrained YOLOv8 weights (baseline models)
│   └── README.roboflow.txt             # Original dataset source documentation
├── 02 Object Detection.pdf             # Lecture theory slides
├── Object Detection Image Proce...pptx # Final presentation slide deck
├── .gitignore                          # Tailored ignore config (excludes large zip/datasets)
└── README.md                           # This documentation
```

*(Note: The massive directories like `cone_runs/` (training checkpoints) and `My First Project.coco/` (unprocessed datasets) are omitted from Git history to follow Git best practices.)*

---

## 📊 Practical Report Details

### A. Short Summary
- **Goal**: Fine-tune a YOLOv8 model to detect and count traffic cones under different lighting conditions, distances, and angles.
- **Approach**: Used the **YOLOv8 nano (yolov8n)** baseline. Annotated bounding boxes in YOLO format and trained for 25 epochs.
- **Result**: High mean Average Precision (mAP@0.5) achieved, enabling accurate counting on real-time webcams or video feeds.

### B. Data Collection & Bounding Boxes
- **Bounding Boxes**: Encoded in standard YOLO format: `[class_id, x_center, y_center, width, height]` normalized to $[0, 1]$.
- **Dataset Size**: Configured via `data.yaml` linking training, validation, and test paths.

### C. Evaluation & Quantitative Metrics
- **mAP@0.5**: Calculates mean Average Precision using Intersection over Union (IoU) threshold of 0.5.
- **Precision/Recall**: Precision measures the percentage of correct predictions, while recall measures how many of the actual cones were successfully detected.

---

## 🚀 Execution Instructions

### Prerequisite
Use the root environment (`TASKS\.venv`) to run all scripts.

### 1. Train the YOLOv8 Detector
To start YOLOv8 training on your dataset:
```bash
cd "My First Project.yolov8\11"
python train.py
```

### 2. Run Inference and Counting
To run inference on unseen images and count the objects:
```bash
python detect_and_count.py
```
*(This script will output predictions with boxes drawn around detected objects and print the count)*

### 3. Run Pipeline with GUI/Command Arguments
For generalized configuration:
```bash
python yolo_pipeline_fixed.py
```
