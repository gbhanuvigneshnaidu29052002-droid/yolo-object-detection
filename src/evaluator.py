"""
Evaluation and Metrics Computation Module for YOLO Object Detection & Counting
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


def calculate_iou(box1: Union[List[float], np.ndarray], box2: Union[List[float], np.ndarray]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes in [x1, y1, x2, y2] format.

    Args:
        box1: Bounding box coordinates [x1, y1, x2, y2].
        box2: Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        IoU value in range [0.0, 1.0].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # Coordinates of intersection rectangle
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
    area2 = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def calculate_pairwise_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise IoU matrix between two sets of boxes.

    Args:
        boxes1: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes2: Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        Matrix of shape (N, M) with pairwise IoU scores.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=float)

    boxes1 = np.asarray(boxes1, dtype=float)
    boxes2 = np.asarray(boxes2, dtype=float)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0.0, a_max=None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return iou


def compute_counting_mae(
    ground_truth_counts: List[int],
    predicted_counts: List[int]
) -> float:
    """
    Calculate Mean Absolute Error (MAE) for instance counting:
        MAE_count = (1 / N) * sum(|N_pred_i - N_true_i|)

    Args:
        ground_truth_counts: List of ground-truth counts across N images.
        predicted_counts: List of predicted counts across N images.

    Returns:
        MAE as a float.
    """
    if len(ground_truth_counts) == 0 or len(predicted_counts) == 0:
        return 0.0

    if len(ground_truth_counts) != len(predicted_counts):
        raise ValueError(
            f"Counts length mismatch: {len(ground_truth_counts)} GT vs {len(predicted_counts)} Pred."
        )

    gt = np.array(ground_truth_counts, dtype=float)
    pred = np.array(predicted_counts, dtype=float)
    return float(np.mean(np.abs(pred - gt)))


def compute_f1_score(precision: float, recall: float) -> float:
    """
    Compute harmonic mean F1-Score from precision and recall.

    Args:
        precision: Precision value in [0.0, 1.0].
        recall: Recall value in [0.0, 1.0].

    Returns:
        F1-score in [0.0, 1.0].
    """
    if (precision + recall) <= 0.0:
        return 0.0
    return float(2.0 * (precision * recall) / (precision + recall))


def parse_yolo_labels(
    label_path: str,
    img_width: int = 640,
    img_height: int = 640
) -> List[Dict[str, Any]]:
    """
    Parse YOLO normalized annotation file (class, x_center, y_center, width, height)
    and convert to absolute pixel coordinates [x1, y1, x2, y2].

    Args:
        label_path: Path to text file containing YOLO formatted labels.
        img_width: Reference image width in pixels.
        img_height: Reference image height in pixels.

    Returns:
        List of parsed annotations with class_id, norm_coords, and pixel_xyxy.
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])

            x1 = max(0.0, (xc - w / 2.0) * img_width)
            y1 = max(0.0, (yc - h / 2.0) * img_height)
            x2 = min(float(img_width), (xc + w / 2.0) * img_width)
            y2 = min(float(img_height), (yc + h / 2.0) * img_height)

            annotations.append({
                "class_id": cls_id,
                "norm_coords": [xc, yc, w, h],
                "box_xyxy": [x1, y1, x2, y2]
            })

    return annotations


def parse_training_csv(csv_path: str) -> Dict[str, Any]:
    """
    Parse YOLO training results.csv file to extract final epoch metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training results CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        return {}

    header = [h.strip() for h in lines[0].split(",")]
    last_row = [float(v.strip()) if v.strip().replace(".", "", 1).replace("-", "", 1).isdigit() else v.strip()
                for v in lines[-1].split(",")]

    record = dict(zip(header, last_row))
    p = float(record.get("metrics/precision(B)", 0.0))
    r = float(record.get("metrics/recall(B)", 0.0))
    map50 = float(record.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(record.get("metrics/mAP50-95(B)", 0.0))

    return {
        "precision": p,
        "recall": r,
        "map50": map50,
        "map50_95": map50_95,
        "f1_score": compute_f1_score(p, r),
        "raw": record
    }


def evaluate_yolo(model_path: str, data_yaml: str) -> Any:
    """
    Execute validation using YOLO model and print standardized metrics summary.
    """
    if YOLO is None:
        raise ImportError("ultralytics is required for YOLO evaluation. Install with `pip install ultralytics`.")

    print(f"📊 Evaluating YOLO Model: {model_path} on {data_yaml}...")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.p.mean()) if hasattr(metrics.box.p, "mean") else float(metrics.box.p)
    recall = float(metrics.box.r.mean()) if hasattr(metrics.box.r, "mean") else float(metrics.box.r)
    f1 = compute_f1_score(precision, recall)

    print("=" * 60)
    print("🏆 YOLO OBJECT DETECTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"  - Precision:     {precision:.4f}")
    print(f"  - Recall:        {recall:.4f}")
    print(f"  - F1-Score:      {f1:.4f}")
    print(f"  - mAP@0.5:       {map50:.4f}")
    print(f"  - mAP@0.5:0.95:  {map50_95:.4f}")
    print("=" * 60)

    return metrics
