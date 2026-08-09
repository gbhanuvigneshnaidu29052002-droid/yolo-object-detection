import os
import sys
import torch
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import cv2
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def check_paths(data_yaml):
    """Validate all required paths exist."""
    if not Path(data_yaml).exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        sys.exit(1)
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    for split in ['train', 'val', 'test']:
        path = data.get(split)
        if path and not Path(path).exists():
            print(f"⚠️ Warning: {split} path '{path}' does not exist. Check data.yaml")
    return data

def calculate_f1(precision, recall):
    """Calculate F1 score from precision and recall."""
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on custom dataset')
    parser.add_argument('--data', type=str,
                        default=r'C:\Users\gbhan\Desktop\YOLO DETCTION\My First Project.yolov8\data.yaml',
                        help='path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold for inference')
    parser.add_argument('--project', type=str,
                        default=r'C:\Users\gbhan\Desktop\YOLO DETCTION\cone_runs',
                        help='project output directory')
    parser.add_argument('--name', type=str, default='cone_detector', help='experiment name')
    parser.add_argument('--pretrained', type=str, default='yolov8n.pt',
                        help='pretrained model (yolov8n.pt, yolov8s.pt, etc.)')
    args = parser.parse_args()

    # Convert paths
    DATA_YAML = args.data
    PROJECT_DIR = Path(args.project)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # Check dataset and get class names
    data_cfg = check_paths(DATA_YAML)
    class_names = data_cfg.get('names', ['class0', 'class1', 'class2'])
    num_classes = len(class_names)
    print(f"\n📋 Classes: {class_names}")
    print(f"📊 Number of classes: {num_classes}")

    # Device
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\n✅ Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")
    if device == 0:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ========================
    # 1. TRAIN
    # ========================
    print("\n🚀 Starting training...")
    model = YOLO(args.pretrained)

    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(PROJECT_DIR),
        name=args.name,
        exist_ok=True,
        patience=20,
        device=device,
        workers=0,               # Windows safe
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )

    # ========================
    # 2. BEST MODEL PATH
    # ========================
    best_model_path = PROJECT_DIR / args.name / 'weights' / 'best.pt'
    if not best_model_path.exists():
        print(f"❌ best.pt not found at {best_model_path}")
        sys.exit(1)
    print(f"\n🏆 Best model saved to: {best_model_path}")
    
    best_model = YOLO(str(best_model_path))

    # ========================
    # 3. VALIDATION METRICS (on val set)
    # ========================
    print("\n" + "="*60)
    print("📊 VALIDATION SET METRICS")
    print("="*60)
    
    val_results = best_model.val(
        data=DATA_YAML,
        split='val',
        conf=args.conf,
        iou=0.5,
        device=device,
        workers=0,
        plots=True,
        save_json=True
    )
    
    val_metrics = val_results.results_dict
    val_precision = val_metrics.get('metrics/precision(B)', 0)
    val_recall = val_metrics.get('metrics/recall(B)', 0)
    val_map50 = val_metrics.get('metrics/mAP50(B)', 0)
    val_map5095 = val_metrics.get('metrics/mAP50-95(B)', 0)
    val_f1 = calculate_f1(val_precision, val_recall)
    
    print(f"Precision (B)      : {val_precision:.4f}")
    print(f"Recall (B)         : {val_recall:.4f}")
    print(f"mAP@0.5 (B)        : {val_map50:.4f}")
    print(f"mAP@0.5:0.95 (B)   : {val_map5095:.4f}")
    print(f"F1 Score           : {val_f1:.4f}")

    # ========================
    # 4. TEST SET METRICS (if test labels exist)
    # ========================
    test_path = data_cfg.get('test')
    test_labels_exist = False
    if test_path and Path(test_path).exists():
        # Check if labels directory exists
        test_image_dir = Path(test_path)
        test_label_dir = test_image_dir.parent / 'labels' if 'images' in str(test_image_dir) else Path(str(test_image_dir).replace('images', 'labels'))
        if test_label_dir.exists() and any(test_label_dir.glob('*.txt')):
            test_labels_exist = True
    
    print("\n" + "="*60)
    if test_labels_exist:
        print("📊 TEST SET METRICS (with ground truth)")
        print("="*60)
        
        test_results = best_model.val(
            data=DATA_YAML,
            split='test',
            conf=args.conf,
            iou=0.5,
            device=device,
            workers=0,
            plots=True,
            save_json=True,
            name='test_evaluation'
        )
        
        test_metrics = test_results.results_dict
        test_precision = test_metrics.get('metrics/precision(B)', 0)
        test_recall = test_metrics.get('metrics/recall(B)', 0)
        test_map50 = test_metrics.get('metrics/mAP50(B)', 0)
        test_map5095 = test_metrics.get('metrics/mAP50-95(B)', 0)
        test_f1 = calculate_f1(test_precision, test_recall)
        
        print(f"Precision (B)      : {test_precision:.4f}")
        print(f"Recall (B)         : {test_recall:.4f}")
        print(f"mAP@0.5 (B)        : {test_map50:.4f}")
        print(f"mAP@0.5:0.95 (B)   : {test_map5095:.4f}")
        print(f"F1 Score           : {test_f1:.4f}")
    else:
        print("📊 TEST SET METRICS")
        print("="*60)
        print("⚠️ No labels found in test set. Cannot compute test metrics.")
        print("   Running inference only (without ground truth evaluation).")
        test_precision = test_recall = test_map50 = test_map5095 = test_f1 = None

    # ========================
    # 5. INFERENCE ON TEST IMAGES (visual + CSV)
    # ========================
    print("\n" + "="*60)
    print("🔍 INFERENCE ON TEST IMAGES")
    print("="*60)
    
    test_image_dir = test_path if test_path and Path(test_path).exists() else None
    if not test_image_dir:
        print("❌ Test image directory not found. Skipping inference.")
    else:
        output_dir = PROJECT_DIR / 'test_predictions'
        output_dir.mkdir(exist_ok=True)
        
        detections = []
        results_pred = best_model.predict(
            source=str(test_image_dir),
            conf=args.conf,
            iou=0.45,
            save=False,
            stream=True,
            device=device,
            workers=0
        )
        
        for result in results_pred:
            img = result.orig_img.copy()
            img_path = result.path
            img_name = Path(img_path).name
            
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    class_name = result.names[cls_id]
                    
                    detections.append([img_name, class_name, round(conf, 4), x1, y1, x2, y2])
                    
                    # Draw on image
                    label = f"{class_name} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img, label, (x1, max(y1-10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            out_path = output_dir / img_name
            cv2.imwrite(str(out_path), img)
        
        # Save CSV
        csv_path = PROJECT_DIR / 'test_detections.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            writer.writerows(detections)
        
        print(f"✅ Annotated test images saved to: {output_dir}")
        print(f"✅ Detection CSV saved to: {csv_path}")
        print(f"   Total detections: {len(detections)}")

    # ========================
    # 6. FIND BEST CONFIDENCE THRESHOLD FROM F1 CURVE
    # ========================
    print("\n" + "="*60)
    print("📈 OPTIMAL CONFIDENCE THRESHOLD")
    print("="*60)
    
    # Read the F1 curve data from results.csv or use default
    results_csv_path = PROJECT_DIR / args.name / 'results.csv'
    if results_csv_path.exists():
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
        
        # Get best validation metrics from training
        best_map50_epoch = df['metrics/mAP50(B)'].idxmax() + 1
        best_map50 = df['metrics/mAP50(B)'].max()
        best_precision_epoch = df['metrics/precision(B)'].idxmax() + 1
        best_precision = df['metrics/precision(B)'].max()
        best_recall = df['metrics/recall(B)'].max()
        
        print(f"Best mAP@0.5: {best_map50:.4f} (epoch {best_map50_epoch})")
        print(f"Best Precision: {best_precision:.4f} (epoch {best_precision_epoch})")
        print(f"Best Recall: {best_recall:.4f}")
    else:
        print("⚠️ results.csv not found. Using validation metrics from current model.")
        print(f"Current Precision: {val_precision:.4f}")
        print(f"Current Recall: {val_recall:.4f}")
        print(f"Current mAP@0.5: {val_map50:.4f}")
    
    print("\n💡 Tip: Check 'BoxF1_curve.png' in the output folder to find the")
    print("   optimal confidence threshold for your model (where F1 peaks).")

    # ========================
    # 7. SAVE ALL METRICS FOR REPORT
    # ========================
    print("\n" + "="*60)
    print("💾 SAVING METRICS FOR REPORT")
    print("="*60)
    
    report_metrics = {
        'dataset_classes': ', '.join(class_names),
        'num_classes': num_classes,
        'validation_precision': val_precision,
        'validation_recall': val_recall,
        'validation_map50': val_map50,
        'validation_map5095': val_map5095,
        'validation_f1': val_f1,
        'test_precision': test_precision if test_labels_exist else 'N/A (no labels)',
        'test_recall': test_recall if test_labels_exist else 'N/A (no labels)',
        'test_map50': test_map50 if test_labels_exist else 'N/A (no labels)',
        'test_map5095': test_map5095 if test_labels_exist else 'N/A (no labels)',
        'test_f1': test_f1 if test_labels_exist else 'N/A (no labels)',
        'best_model_path': str(best_model_path),
        'confidence_threshold_used': args.conf
    }
    
    metrics_csv = PROJECT_DIR / 'report_metrics.csv'
    pd.DataFrame([report_metrics]).to_csv(metrics_csv, index=False)
    print(f"✅ Metrics saved to: {metrics_csv}")
    
    # Print summary for report
    print("\n" + "="*60)
    print("📋 SUMMARY FOR YOUR REPORT (Section E2)")
    print("="*60)
    print("\n| Experiment | mAP@0.5 | Precision | Recall | F1 Score |")
    print("|------------|---------|-----------|--------|----------|")
    print(f"| Validation | {val_map50:.4f} | {val_precision:.4f} | {val_recall:.4f} | {val_f1:.4f} |")
    if test_labels_exist:
        print(f"| Test       | {test_map50:.4f} | {test_precision:.4f} | {test_recall:.4f} | {test_f1:.4f} |")
    else:
        print("| Test       | N/A (no labels) | N/A | N/A | N/A |")
    
    print("\n" + "="*60)
    print("🎉 All done!")
    print("="*60)
    print("\n📁 Output files:")
    print(f"   - Best model: {best_model_path}")
    print(f"   - Validation plots: {PROJECT_DIR / args.name}/")
    print(f"   - Test predictions: {PROJECT_DIR / 'test_predictions'}/")
    print(f"   - Detection CSV: {PROJECT_DIR / 'test_detections.csv'}")
    print(f"   - Metrics for report: {metrics_csv}")

if __name__ == '__main__':
    main()