# Real-Time Traffic Cone Object Detection & Instance Counting via YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org)
[![Tests Passing](https://img.shields.io/badge/Tests-27%2F27%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Author:** Bhanu Vignesh Naidu Ganeshna  
**Course:** Image Processing & Computer Vision (Practical Project)  
**Repository Type:** Standalone Production Package  

---

## 📌 Executive Summary & Practical Report

### A. Short Summary
* **Goal:** Build an end-to-end real-time object detection and instance counting pipeline to detect and track multi-colored traffic cones (`blue_cone`, `green_cone`, `orange_cone`) in dynamic field environments for autonomous navigation and robotic safety.
* **Approach:** Annotations converted to YOLO coordinate format, trained using single-stage YOLOv8 architecture (`yolov8n` baseline vs. fine-tuned `improved_model`), incorporating mosaic augmentations and Non-Maximum Suppression (NMS) post-processing.
* **Main Result:** Achieved an exceptional **0.9831 mAP@0.5**, **0.9472 Precision**, **0.8115 Recall**, and **0.7868 mAP@0.5:0.95** with sub-18ms inference latency on an RTX 4050 GPU.

---

## 📊 Model Performance & Benchmark Comparison

| Model Variant | Parameters | GFLOPs | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 | Inference Speed | Target Application |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Baseline YOLOv8n | 3.15M | 8.7 | 0.8420 | 0.8110 | 0.8262 | 0.8540 | 0.6120 | 11.4 ms | Lightweight edge devices |
| **Improved Model (Ours)** | **3.01M** | **8.1** | **0.9472** | **0.8115** | **0.8741** | **0.9831** | **0.7868** | **18.0 ms** | Autonomous Navigation / ROS 2 Robotics |

---

## 📐 Mathematical Formulation

### 1. Mean Average Precision (mAP@0.5)
```math
\text{mAP@0.5} = \frac{1}{C} \sum_{c=1}^{C} \int_{0}^{1} P_c(R_c) \, dR_c
```

### 2. Intersection over Union (IoU)
```math
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
```

### 3. Instance Counting Mean Absolute Error (MAE)
```math
\text{MAE}_{\text{count}} = \frac{1}{N} \sum_{i=1}^{N} |N_{\text{pred}, i} - N_{\text{true}, i}|
```

### 4. Harmonic Mean F1-Score
```math
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

---

## 📈 Visual Assets & Training Curves

### 1. Training & Validation Loss / Metric Convergence
![Training & Validation Curves](cone_runs/improved_model/results.png)

---

### 2. Precision-Recall (PR) Curve
![Precision-Recall Curve](cone_runs/improved_model/BoxPR_curve.png)

---

### 3. Confusion Matrix & Validation Predictions

| Normalized Confusion Matrix | Qualitative Validation Detections |
| :---: | :---: |
| ![Normalized Confusion Matrix](cone_runs/improved_model/confusion_matrix_normalized.png) | ![Validation Predictions](cone_runs/improved_model/val_batch0_pred.jpg) |

---

## 🏗️ Repository Architecture

```text
yolo-object-detection/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md             # Standardized bug reporting form
│   │   ├── feature_request.md        # Robotics & tracking feature proposals
│   │   └── config.yml                # Issue configuration & discussion links
│   ├── pull_request_template.md      # PR checklist & verification standard
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI workflow (Python 3.10)
├── cone_runs/
│   └── improved_model/               # Training logs, curves & confusion matrices
├── src/
│   ├── __init__.py                   # Core package exports
│   ├── detector.py                   # TrafficConeDetector class & counting logic
│   ├── evaluator.py                  # IoU, counting MAE, and metrics computation
│   └── trainer.py                    # Training pipeline orchestration
├── tests/
│   ├── __init__.py
│   └── test_cone_detection.py        # Automated 27-test unit test suite
├── CODE_OF_CONDUCT.md                # Contributor Covenant v2.1
├── CONTRIBUTING.md                   # Contribution & development guide
├── LICENSE                           # MIT License
├── README.md                         # Comprehensive project documentation
├── SECURITY.md                       # Security policy & threat modeling
├── main.py                           # CLI entrypoint for train/eval/predict/metrics
├── requirements.txt                  # Production dependencies
└── setup.py                          # Setuptools package configuration
```

---

## 🚀 Quickstart & Usage Instructions

### 1. Setup Environment
```bash
git clone https://github.com/gbhanuvigneshnaidu29052002-droid/yolo-object-detection.git
cd yolo-object-detection

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
python3 -m unittest discover -s tests -v
```

### 3. Inspect Validation Metrics from Training Logs
```bash
python main.py --mode metrics --csv cone_runs/improved_model/results.csv
```

### 4. Run Model Evaluation
```bash
python main.py --mode eval --model cone_runs/improved_model/weights/best.pt --data "My First Project.yolov8/data.yaml"
```

### 5. Run Inference & Multi-Class Instance Counting
```bash
python main.py --mode predict --model cone_runs/improved_model/weights/best.pt --source path/to/images --conf 0.25
```

### 6. Train Custom YOLOv8 Model
```bash
python main.py --mode train --data "My First Project.yolov8/data.yaml" --epochs 30 --batch 16 --model yolov8n.pt
```

---

## 🔮 Future Work & Robotics Expansion

1. **ROS 2 Topic Integration**:
   - Wrap the detector in a ROS 2 node publishing `/traffic_cone/detections` (`vision_msgs/Detection2DArray`) for autonomous rover path planning.
2. **3D Lidar & Monocular Fusion**:
   - Fuse 2D YOLO cone bounding boxes with monocular depth maps to calculate 3D spatial coordinates $(X, Y, Z)$ for real-time obstacle avoidance.
3. **Edge Optimization**:
   - Quantize fine-tuned YOLOv8 checkpoints to TensorRT FP16 / INT8 engines for sub-5ms latency on NVIDIA Jetson Orin.

---

## 🤝 Contributing & Community Standards

We welcome contributions! Please review our community guidelines before participating:

- **[Code of Conduct](CODE_OF_CONDUCT.md)**: Details our standards of behavior and reporting process.
- **[Contributing Guide](CONTRIBUTING.md)**: Architectural overview, PR submission guidelines, and test requirements.
- **[Security Policy](SECURITY.md)**: Supported versions, vulnerability reporting, and model checkpoint safety.

---

### 📝 Declaration of Original Work

I confirm that this project was designed, implemented, and documented by me for the Image Processing & Computer Vision coursework.

**Author:** Bhanu Vignesh Naidu Ganeshna  
**License:** [MIT License](LICENSE)
