"""
YOLO Model Training Pipeline
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
from typing import Optional, Any

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


def train_yolo(
    data_yaml: str,
    model_variant: str = 'yolov8n.pt',
    epochs: int = 30,
    batch: int = 16,
    imgsz: int = 640,
    project: str = 'cone_runs',
    name: str = 'cone_detector',
    exist_ok: bool = True,
    device: Optional[str] = None
) -> Any:
    """
    Train a YOLO model variant with specified training arguments.

    Args:
        data_yaml: Path to data.yaml dataset specification.
        model_variant: Base YOLO architecture checkpoint or yaml config.
        epochs: Number of training epochs.
        batch: Batch size.
        imgsz: Image resolution size.
        project: Save directory project root.
        name: Experiment subfolder name.
        exist_ok: Whether to overwrite existing experiment folder.
        device: Computational device ('cpu', '0', 'cuda', etc.).

    Returns:
        Ultralytics training results object.
    """
    if YOLO is None:
        raise ImportError("ultralytics is required for training. Install with `pip install ultralytics`.")

    data_yaml_abs = os.path.abspath(data_yaml)
    if not os.path.exists(data_yaml_abs):
        raise FileNotFoundError(f"Dataset config YAML not found: {data_yaml_abs}")

    print(f"🚀 Initializing YOLOv8 Model Training ({model_variant})...")
    print(f"   - Dataset Config: {data_yaml_abs}")
    print(f"   - Epochs: {epochs} | Batch: {batch} | ImgSz: {imgsz}")

    model = YOLO(model_variant)
    train_kwargs = {
        "data": data_yaml_abs,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
        "name": name,
        "exist_ok": exist_ok
    }
    if device is not None:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)
    print("✅ Training complete.")
    return results
