# Contributing to YOLO Traffic Cone Object Detection & Counting

Thank you for your interest in contributing to **Real-Time Traffic Cone Object Detection & Instance Counting via YOLOv8**! We welcome contributions from computer vision researchers, autonomous driving engineers, robotics developers, and open-source practitioners.

Please read through this guide before submitting issues or pull requests.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainer.

---

## Areas of Contribution

We encourage contributions across several computer vision and autonomous robotics domains:

- **Robotics & ROS 2 Perception**: Implement native ROS 2 nodes publishing `vision_msgs/Detection2DArray` and PointCloud-fused 3D cone coordinates for rover path planning.
- **Edge Acceleration & Quantization**: Add export scripts and deployment benchmarks for TensorRT FP16/INT8, OpenVINO, and ONNX Runtime targeting embedded platforms (NVIDIA Jetson, Raspberry Pi).
- **Multi-Object Tracking (MOT)**: Integrate real-time tracking algorithms such as ByteTrack, BoT-SORT, or DeepSORT to maintain consistent cone track IDs across dynamic video sequences.
- **Dataset Expansion & Robustness**: Add synthetic domain randomization (rain, fog, direct sunlight, low-light night conditions) and benchmark multi-class cone detection across varied track environments.
- **Instance Counting Algorithms**: Enhance multi-class instance counting with density estimation maps or spatial region-of-interest (ROI) filtering.

---

## Reporting Issues & Bugs

Before opening a new issue, please check existing [GitHub Issues](https://github.com/gbhanuvigneshnaidu29052002-droid/yolo-object-detection/issues) to avoid duplicates.

When reporting a bug:
1. Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
2. Specify your environment details:
   - OS: Linux (Ubuntu) / Windows / macOS
   - Python Version: Python 3.10+
   - PyTorch Version: PyTorch 2.x
   - Ultralytics Version: YOLOv8 8.x+
   - Hardware: CPU or GPU (CUDA version, GPU model)
3. Include minimal reproduction steps and code snippets.
4. Attach full error tracebacks and terminal outputs.

---

## Development Workflow

### 1. Fork & Clone Repository
```bash
git clone https://github.com/<your-username>/yolo-object-detection.git
cd yolo-object-detection
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Run Automated Tests
Before submitting changes, ensure all unit tests pass cleanly:
```bash
python3 -m unittest discover -s tests -v
```

### 5. Commit & Push Changes
Follow conventional commit messages:
- `feat: Add ROS 2 Detection2DArray publisher node`
- `fix: Handle empty bounding box tensors in instance counter`
- `perf: Add TensorRT INT8 calibration pipeline`
- `docs: Expand mAP and MAE mathematical formulation in README`
- `test: Add edge-case test for overlapping bounding boxes`

```bash
git add .
git commit -m "feat: Describe your change cleanly"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request
Submit your PR against the `main` branch using our [Pull Request Template](.github/pull_request_template.md).

---

## Code Style & Guidelines

- **Ultralytics & PyTorch Idioms**: Write clean, modular components adhering to PyTorch and YOLO design standards.
- **Device Agnostic**: Ensure code runs seamlessly on both CPU and CUDA devices (`device="cuda" if torch.cuda.is_available() else "cpu"`).
- **Type Annotations**: Use Python typing (`Optional`, `Dict`, `List`, `Tuple`) across all public functions and class methods.
- **Reproducibility**: Set seeds when adding training or evaluation routines.
- **Documentation**: Keep comments and docstrings clear and informative.

---

## Recognition

Contributors who have meaningful pull requests merged will be acknowledged in the project documentation. Thank you for contributing!
