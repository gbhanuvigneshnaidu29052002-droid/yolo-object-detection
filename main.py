"""
YOLOv8 Traffic Cone Object Detection & Object Counter CLI Entrypoint
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import argparse
import json
from ultralytics import YOLO

def train_yolo(data_yaml, model_variant='yolov8n.pt', epochs=30, batch=16, project='cone_runs', name='cone_detector'):
    print(f"🚀 Initializing YOLOv8 Model Training ({model_variant})...")
    model = YOLO(model_variant)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True
    )
    print("✅ Training complete.")
    return results

def evaluate_yolo(model_path, data_yaml):
    print(f"📊 Evaluating YOLO Model: {model_path}...")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.p.mean()
    recall = metrics.box.r.mean()
    
    print("=" * 60)
    print("🏆 YOLO OBJECT DETECTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"  - Precision:   {precision:.4f}")
    print(f"  - Recall:      {recall:.4f}")
    print(f"  - mAP@0.5:     {map50:.4f}")
    print(f"  - mAP@0.5:0.95:{map50_95:.4f}")
    print("=" * 60)
    return metrics

def predict_and_count(model_path, source):
    print(f"🔍 Running Prediction & Instance Counting on: {source}...")
    model = YOLO(model_path)
    results = model.predict(source=source, save=True, conf=0.25)
    
    for r in results:
        boxes = r.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        names = r.names
        
        counts = {}
        for c in class_ids:
            cname = names[c]
            counts[cname] = counts.get(cname, 0) + 1
            
        print(f"📸 Image {r.path}: Total Objects Detected = {len(boxes)}")
        for cname, count in counts.items():
            print(f"   - {cname}: {count}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection & Counting CLI")
    parser.add_argument("--mode", type=str, default="eval", choices=["train", "eval", "predict"])
    parser.add_argument("--data", type=str, default="My First Project.yolov8/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--source", type=str, default="My First Project.yolov8/valid/images")
    args = parser.parse_args()

    if args.mode == "train":
        train_yolo(args.data, model_variant=args.model)
    elif args.mode == "eval":
        evaluate_yolo(args.model, args.data)
    elif args.mode == "predict":
        predict_and_count(args.model, args.source)

if __name__ == "__main__":
    main()
