"""
Unit Test Suite for YOLO Traffic Cone Detection, Counting & Evaluation
Author: Bhanu Vignesh Naidu Ganeshna
"""

import unittest
import os
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch

from src.evaluator import (
    calculate_iou,
    calculate_pairwise_iou,
    compute_counting_mae,
    compute_f1_score,
    parse_yolo_labels,
    parse_training_csv
)
from src.detector import TrafficConeDetector
from main import build_parser


class TestIoUComputation(unittest.TestCase):
    """Unit tests for single-box and pairwise IoU bounding box calculations."""

    def test_identical_boxes_iou(self):
        box = [10.0, 10.0, 50.0, 50.0]
        iou = calculate_iou(box, box)
        self.assertAlmostEqual(iou, 1.0, places=5)

    def test_disjoint_boxes_iou(self):
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [20.0, 20.0, 30.0, 30.0]
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)

    def test_partial_overlap_iou(self):
        # box1 area = 100 (0 to 10), box2 area = 100 (5 to 15)
        # intersection = 5*10 = 50, union = 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3 ~ 0.33333
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [5.0, 0.0, 15.0, 10.0]
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=4)

    def test_zero_area_box(self):
        box1 = [10.0, 10.0, 10.0, 10.0]
        box2 = [10.0, 10.0, 20.0, 20.0]
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)

    def test_pairwise_iou_matrix(self):
        boxes1 = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 20.0, 20.0]
        ])
        boxes2 = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [50.0, 50.0, 60.0, 60.0]
        ])
        matrix = calculate_pairwise_iou(boxes1, boxes2)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertAlmostEqual(matrix[0, 0], 1.0, places=5)
        self.assertEqual(matrix[0, 1], 0.0)
        self.assertEqual(matrix[1, 0], 0.0)
        self.assertEqual(matrix[1, 1], 0.0)

    def test_pairwise_iou_empty(self):
        matrix = calculate_pairwise_iou(np.empty((0, 4)), np.empty((0, 4)))
        self.assertEqual(matrix.shape, (0, 0))


class TestCountingMAE(unittest.TestCase):
    """Unit tests for Instance Counting Mean Absolute Error calculation."""

    def test_zero_error_mae(self):
        gt = [3, 5, 2, 8]
        pred = [3, 5, 2, 8]
        mae = compute_counting_mae(gt, pred)
        self.assertEqual(mae, 0.0)

    def test_known_mae(self):
        gt = [4, 6, 2, 5]
        pred = [3, 7, 2, 9]  # diffs: |3-4|=1, |7-6|=1, |2-2|=0, |9-5|=4 -> sum=6, avg=1.5
        mae = compute_counting_mae(gt, pred)
        self.assertAlmostEqual(mae, 1.5, places=5)

    def test_mismatched_length_raises_error(self):
        gt = [1, 2, 3]
        pred = [1, 2]
        with self.assertRaises(ValueError):
            compute_counting_mae(gt, pred)

    def test_empty_counts_returns_zero(self):
        self.assertEqual(compute_counting_mae([], []), 0.0)


class TestF1Score(unittest.TestCase):
    """Unit tests for F1 Score harmonic mean computation."""

    def test_f1_score_calculation(self):
        p, r = 0.9472, 0.8115
        f1 = compute_f1_score(p, r)
        expected = 2.0 * (p * r) / (p + r)
        self.assertAlmostEqual(f1, expected, places=5)

    def test_f1_score_zero_division(self):
        f1 = compute_f1_score(0.0, 0.0)
        self.assertEqual(f1, 0.0)

    def test_f1_score_perfect(self):
        f1 = compute_f1_score(1.0, 1.0)
        self.assertEqual(f1, 1.0)


class TestYOLOLabelParsing(unittest.TestCase):
    """Unit tests for reading and converting YOLO formatted label text files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.label_file = os.path.join(self.temp_dir.name, "sample.txt")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_valid_labels(self):
        # 2 objects: class 0 and class 2 with normalized center-xywh
        content = "0 0.5 0.5 0.2 0.4\n2 0.8 0.8 0.1 0.2\n"
        with open(self.label_file, "w", encoding="utf-8") as f:
            f.write(content)

        parsed = parse_yolo_labels(self.label_file, img_width=1000, img_height=1000)
        self.assertEqual(len(parsed), 2)

        # Obj 1: xc=500, yc=500, w=200, h=400 -> x1=400, y1=300, x2=600, y2=700
        obj1 = parsed[0]
        self.assertEqual(obj1["class_id"], 0)
        self.assertEqual(obj1["box_xyxy"], [400.0, 300.0, 600.0, 700.0])

        obj2 = parsed[1]
        self.assertEqual(obj2["class_id"], 2)

    def test_parse_nonexistent_file(self):
        res = parse_yolo_labels("/non/existent/path/label.txt")
        self.assertEqual(res, [])

    def test_parse_malformed_lines_ignored(self):
        content = "corrupt line\n0 0.5 0.5 0.2 0.4\n1 0.2\n"
        with open(self.label_file, "w", encoding="utf-8") as f:
            f.write(content)

        parsed = parse_yolo_labels(self.label_file, img_width=100, img_height=100)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["class_id"], 0)


class TestResultsCSVParser(unittest.TestCase):
    """Unit tests for parsing YOLO training results.csv logs."""

    def test_parse_existing_training_csv(self):
        csv_path = "cone_runs/improved_model/results.csv"
        if os.path.exists(csv_path):
            metrics = parse_training_csv(csv_path)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("map50", metrics)
            self.assertIn("map50_95", metrics)
            self.assertIn("f1_score", metrics)
            self.assertGreater(metrics["map50"], 0.90)

    def test_nonexistent_csv_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            parse_training_csv("/invalid/path/results.csv")


class TestTrafficConeDetector(unittest.TestCase):
    """Unit tests for TrafficConeDetector instance counting and class mapping."""

    def setUp(self):
        self.detector = TrafficConeDetector(
            model_path="dummy.pt",
            class_names=['blue_cone', 'green_cone', 'orange_cone'],
            conf_threshold=0.3
        )

    def test_count_instances_from_boxes(self):
        class_ids = [0, 0, 1, 2, 2, 2]
        counts = self.detector.count_instances_from_boxes(class_ids)
        self.assertEqual(counts["blue_cone"], 2)
        self.assertEqual(counts["green_cone"], 1)
        self.assertEqual(counts["orange_cone"], 3)

    def test_count_instances_custom_mapping(self):
        custom_names = {0: "blue", 1: "green", 2: "orange"}
        counts = self.detector.count_instances_from_boxes([0, 1], names=custom_names)
        self.assertEqual(counts["blue"], 1)
        self.assertEqual(counts["green"], 1)

    def test_detect_and_count_with_mocked_yolo(self):
        # Mock YOLO model inference
        mock_box = MagicMock()
        mock_box.__len__.return_value = 2
        mock_box.cls = np.array([0, 2])
        mock_box.conf = np.array([0.92, 0.88])
        mock_box.xyxy = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])

        mock_result = MagicMock()
        mock_result.path = "test_image.jpg"
        mock_result.boxes = mock_box
        mock_result.names = {0: "blue_cone", 1: "green_cone", 2: "orange_cone"}

        with patch.object(self.detector, 'predict', return_value=[mock_result]):
            summary = self.detector.detect_and_count(source="test_image.jpg")
            self.assertEqual(len(summary), 1)
            item = summary[0]
            self.assertEqual(item["total_count"], 2)
            self.assertEqual(item["counts_per_class"]["blue_cone"], 1)
            self.assertEqual(item["counts_per_class"]["orange_cone"], 1)
            self.assertEqual(item["counts_per_class"]["green_cone"], 0)
            self.assertEqual(len(item["detections"]), 2)

    def test_detect_and_count_empty_boxes(self):
        mock_result = MagicMock()
        mock_result.path = "empty_image.jpg"
        mock_result.boxes = None

        with patch.object(self.detector, 'predict', return_value=[mock_result]):
            summary = self.detector.detect_and_count(source="empty_image.jpg")
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["total_count"], 0)
            self.assertEqual(summary[0]["detections"], [])


class TestCLIArgumentParsing(unittest.TestCase):
    """Unit tests verifying CLI parser options and default behaviors."""

    def setUp(self):
        self.parser = build_parser()

    def test_default_arguments(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.mode, "eval")
        self.assertEqual(args.model, "yolov8n.pt")
        self.assertEqual(args.conf, 0.25)
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.batch, 16)

    def test_custom_train_arguments(self):
        cmd = ["--mode", "train", "--epochs", "50", "--batch", "8", "--device", "cpu"]
        args = self.parser.parse_args(cmd)
        self.assertEqual(args.mode, "train")
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.batch, 8)
        self.assertEqual(args.device, "cpu")

    def test_predict_mode_arguments(self):
        cmd = ["--mode", "predict", "--source", "custom/images", "--conf", "0.45"]
        args = self.parser.parse_args(cmd)
        self.assertEqual(args.mode, "predict")
        self.assertEqual(args.source, "custom/images")
        self.assertEqual(args.conf, 0.45)


class TestYOLOTrainer(unittest.TestCase):
    """Unit tests for YOLO training pipeline validation."""

    def test_trainer_nonexistent_yaml_raises(self):
        from src.trainer import train_yolo
        with self.assertRaises(FileNotFoundError):
            train_yolo(data_yaml="/non/existent/data.yaml")

    def test_trainer_mocked_model(self):
        from src.trainer import train_yolo
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.train.return_value = {"status": "trained"}

        with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
            with patch("src.trainer.YOLO", return_value=mock_yolo_instance):
                res = train_yolo(data_yaml=tmp.name, epochs=1, batch=2, imgsz=320)
                self.assertEqual(res, {"status": "trained"})
                mock_yolo_instance.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
