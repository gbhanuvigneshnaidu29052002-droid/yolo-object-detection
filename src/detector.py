"""
Traffic Cone Detector & Instance Counter Module
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


class TrafficConeDetector:
    """
    Traffic Cone Detection and Instance Counting pipeline based on YOLOv8.
    Supports multi-class detection for 'blue_cone', 'green_cone', and 'orange_cone'.
    """

    DEFAULT_CLASSES = ['blue_cone', 'green_cone', 'orange_cone']

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.class_names = class_names or list(self.DEFAULT_CLASSES)
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy loader for YOLO model instance."""
        if self._model is None:
            if YOLO is None:
                raise ImportError("ultralytics is required for YOLO inference. Install with `pip install ultralytics`.")
            self._model = YOLO(self.model_path)
            if self.device:
                self._model.to(self.device)
        return self._model

    def count_instances_from_boxes(self, class_ids: List[int], names: Optional[Dict[int, str]] = None) -> Dict[str, int]:
        """
        Count instances per class given a list of class IDs.

        Args:
            class_ids: List of integer class labels.
            names: Optional mapping from class ID to class name.

        Returns:
            Dictionary mapping class name to detected count.
        """
        counts = {name: 0 for name in self.class_names}
        for cid in class_ids:
            if names and cid in names:
                cname = names[cid]
            elif 0 <= cid < len(self.class_names):
                cname = self.class_names[cid]
            else:
                cname = f"class_{cid}"

            counts[cname] = counts.get(cname, 0) + 1
        return counts

    def predict(
        self,
        source: Any,
        conf: Optional[float] = None,
        save: bool = False,
        imgsz: int = 640
    ) -> List[Any]:
        """
        Run inference on image/directory source.

        Args:
            source: Image file path, numpy array, PIL Image, or directory path.
            conf: Confidence threshold override.
            save: Whether to save predicted images with bounding box overlays.
            imgsz: Input resolution.

        Returns:
            List of ultralytics Results objects.
        """
        confidence = conf if conf is not None else self.conf_threshold
        return self.model.predict(source=source, conf=confidence, save=save, imgsz=imgsz)

    def detect_and_count(
        self,
        source: Any,
        conf: Optional[float] = None,
        save: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run inference on source and return structured detection results with instance counts.

        Returns:
            List of dictionaries containing path, total_count, counts_per_class, and detections.
        """
        results = self.predict(source=source, conf=conf, save=save)
        summary = []

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                summary.append({
                    "path": getattr(r, "path", ""),
                    "total_count": 0,
                    "counts_per_class": {name: 0 for name in self.class_names},
                    "detections": []
                })
                continue

            cls_arr = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls, dtype=int)
            conf_arr = boxes.conf.cpu().numpy().astype(float) if hasattr(boxes.conf, "cpu") else np.array(boxes.conf, dtype=float)
            xyxy_arr = boxes.xyxy.cpu().numpy().astype(float) if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy, dtype=float)

            names_map = getattr(r, "names", {i: n for i, n in enumerate(self.class_names)})
            counts = self.count_instances_from_boxes(cls_arr.tolist(), names_map)

            detections = []
            for cid, score, box in zip(cls_arr, conf_arr, xyxy_arr):
                cname = names_map.get(cid, self.class_names[cid] if 0 <= cid < len(self.class_names) else f"class_{cid}")
                detections.append({
                    "class_id": int(cid),
                    "class_name": cname,
                    "confidence": float(score),
                    "box_xyxy": box.tolist()
                })

            summary.append({
                "path": getattr(r, "path", ""),
                "total_count": len(detections),
                "counts_per_class": counts,
                "detections": detections
            })

        return summary


def predict_and_count(model_path: str, source: str, conf: float = 0.25, save: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function matching CLI invocation for running predictions and printing instance counts.
    """
    print(f"🔍 Running Prediction & Instance Counting on: {source}...")
    detector = TrafficConeDetector(model_path=model_path, conf_threshold=conf)
    summaries = detector.detect_and_count(source=source, conf=conf, save=save)

    for item in summaries:
        print(f"📸 Image {item['path']}: Total Objects Detected = {item['total_count']}")
        for cname, count in item["counts_per_class"].items():
            print(f"   - {cname}: {count}")

    return summaries
