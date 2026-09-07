"""
YOLOv8 Traffic Cone Object Detection & Object Counter CLI Entrypoint
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import sys
import argparse

# Add package root to sys.path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    TrafficConeDetector,
    train_yolo,
    evaluate_yolo,
    predict_and_count,
    parse_training_csv
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Real-Time Traffic Cone Detection & Automated Instance Counting CLI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["train", "eval", "predict", "metrics"],
        help="Pipeline operational mode: 'train', 'eval', 'predict', or 'metrics'"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="My First Project.yolov8/data.yaml",
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO checkpoint weights (.pt file)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="My First Project.yolov8/valid/images",
        help="Path to image or directory for object detection and instance counting"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for bounding box predictions"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for model training"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="cone_runs",
        help="Target folder for training runs"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="cone_detector",
        help="Run identifier name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computational device ('cpu', 'cuda', '0')"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="cone_runs/improved_model/results.csv",
        help="Path to results.csv log for metrics reporting"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train_yolo(
            data_yaml=args.data,
            model_variant=args.model,
            epochs=args.epochs,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device
        )
    elif args.mode == "eval":
        evaluate_yolo(args.model, args.data)
    elif args.mode == "predict":
        predict_and_count(args.model, args.source, conf=args.conf)
    elif args.mode == "metrics":
        print(f"📊 Reading Validation Metrics from CSV: {args.csv}")
        metrics = parse_training_csv(args.csv)
        print("=" * 60)
        print("📈 YOLO TRAINING & VALIDATION LOG METRICS")
        print("=" * 60)
        print(f"  - Precision (B):   {metrics.get('precision', 0.0):.4f}")
        print(f"  - Recall (B):      {metrics.get('recall', 0.0):.4f}")
        print(f"  - F1-Score:        {metrics.get('f1_score', 0.0):.4f}")
        print(f"  - mAP@0.5 (B):     {metrics.get('map50', 0.0):.4f}")
        print(f"  - mAP@0.5:0.95 (B):{metrics.get('map50_95', 0.0):.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
