"""
YOLOv8 Object Detection Pipeline - FIXED VERSION
Properly copies all files to organized output folder
"""

import os
import sys
import torch
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import cv2
import csv
import pandas as pd
import json

class YOLOPipeline:
    """Interactive YOLOv8 object detection pipeline."""
    
    def __init__(self):
        self.setup_paths()
        self.setup_device()
        
    def setup_paths(self):
        """Create organized folder structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(f"yolo_output_{timestamp}")
        
        self.folders = {
            'weights': self.base_dir / "01_weights",
            'metrics': self.base_dir / "02_metrics",
            'plots': self.base_dir / "03_plots",
            'predictions': self.base_dir / "04_predictions",
            'csv': self.base_dir / "05_csv_outputs",
            'config': self.base_dir / "06_config"
        }
        
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 All results will be saved to: {self.base_dir}")
        
    def setup_device(self):
        """Configure computation device."""
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"\n🖥️  Device: {'GPU (CUDA)' if self.device == 0 else 'CPU'}")
        if self.device == 0:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    def get_user_choice(self):
        """Ask user what to do."""
        print("\n" + "="*60)
        print("🎯 YOLOv8 OBJECT DETECTION PIPELINE")
        print("="*60)
        print("\nWhat would you like to do?")
        print("  1. Train + Test (Complete pipeline)")
        print("  2. Test only (using existing model)")
        print("  3. Inference only (draw boxes, no metrics)")
        print("  4. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    def get_data_config(self):
        """Get data.yaml path from user."""
        default_path = r"..\..\My First Project.yolov8\data.yaml"
        print(f"\n📂 Data configuration file path:")
        print(f"   Default: {default_path}")
        
        use_default = input("Use default path? (y/n): ").strip().lower()
        
        if use_default == 'y':
            data_path = default_path
        else:
            data_path = input("Enter full path to data.yaml: ").strip()
        
        if not Path(data_path).exists():
            print(f"❌ File not found: {data_path}")
            sys.exit(1)
        
        with open(data_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
        
        self.class_names = self.data_cfg.get('names', [])
        print(f"\n✅ Loaded {len(self.class_names)} classes: {self.class_names}")
        
        shutil.copy(data_path, self.folders['config'] / "data.yaml")
        
        return data_path
    
    def get_model_path(self):
        """Get path to existing model."""
        print("\n" + "="*60)
        print("📂 LOAD EXISTING MODEL")
        print("="*60)
        
        # Look for existing model
        default_model = r"..\..\cone_runs\cone_detector\weights\best.pt"
        
        # Also check recent runs
        recent_runs = list(Path(".").glob("yolo_output_*"))
        if recent_runs:
            latest = max(recent_runs, key=lambda p: p.stat().st_ctime)
            recent_model = latest / "01_weights" / "best.pt"
            if recent_model.exists():
                print(f"Found recent model: {recent_model}")
                use_recent = input("Use this model? (y/n): ").strip().lower()
                if use_recent == 'y':
                    return str(recent_model)
        
        print(f"Default: {default_model}")
        use_default = input("Use default model? (y/n): ").strip().lower()
        
        if use_default == 'y':
            model_path = default_model
        else:
            model_path = input("Enter path to model (.pt file): ").strip()
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        # Copy model to organized folder
        shutil.copy(model_path, self.folders['weights'] / "best.pt")
        print(f"✅ Model copied to: {self.folders['weights'] / 'best.pt'}")
        
        return str(self.folders['weights'] / "best.pt")
    
    def evaluate_and_copy(self, model, data_path, split_name, conf_threshold):
        """Evaluate model and copy all results to organized folder."""
        print(f"\n📊 {split_name.upper()} SET EVALUATION")
        print("-"*40)
        
        results = model.val(
            data=data_path,
            split=split_name,
            conf=conf_threshold,
            iou=0.5,
            device=self.device,
            workers=0,
            plots=True,
            save_json=True,
        )
        
        metrics = {
            'precision': results.results_dict.get('metrics/precision(B)', 0),
            'recall': results.results_dict.get('metrics/recall(B)', 0),
            'map50': results.results_dict.get('metrics/mAP50(B)', 0),
            'map5095': results.results_dict.get('metrics/mAP50-95(B)', 0),
        }
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   mAP@0.5:   {metrics['map50']:.4f}")
        print(f"   mAP@0.5:0.95: {metrics['map5095']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        
        # Copy all plots from the most recent runs folder
        runs_dir = Path(r"..\..\My First Project.yolov8\runs\detect")
        if runs_dir.exists():
            # Get the most recent folder for this split
            split_folders = [f for f in runs_dir.iterdir() if f.is_dir() and split_name in f.name.lower()]
            if split_folders:
                latest_folder = max(split_folders, key=lambda f: f.stat().st_ctime)
                print(f"   Copying plots from: {latest_folder}")
                
                # Copy all PNG files
                for png_file in latest_folder.glob("*.png"):
                    dest_name = f"{split_name}_{png_file.name}"
                    shutil.copy(png_file, self.folders['plots'] / dest_name)
                    print(f"      Copied: {dest_name}")
        
        return metrics
    
    def run_inference_and_copy(self, model, test_path, conf_threshold):
        """Run inference and save annotated images."""
        print("\n" + "="*60)
        print("🔍 INFERENCE ON TEST IMAGES")
        print("="*60)
        
        if not test_path or not Path(test_path).exists():
            print("❌ Test image path not found")
            return None
        
        detections = []
        results = model.predict(
            source=test_path,
            conf=conf_threshold,
            iou=0.45,
            save=False,
            stream=True,
            device=self.device,
            workers=0
        )
        
        for result in results:
            img = result.orig_img.copy()
            img_name = Path(result.path).name
            
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    class_name = result.names[cls_id]
                    
                    detections.append([img_name, class_name, round(conf, 4), x1, y1, x2, y2])
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(img, label, (x1, max(y1-10, 20)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            output_path = self.folders['predictions'] / img_name
            cv2.imwrite(str(output_path), img)
        
        # Save CSV
        csv_path = self.folders['csv'] / 'detections.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            writer.writerows(detections)
        
        print(f"\n✅ Saved to {self.folders['predictions']}:")
        print(f"   - {len(detections)} detections")
        print(f"   - CSV: {csv_path}")
        
        return detections
    
    def save_metrics(self, val_metrics=None, test_metrics=None):
        """Save all metrics to CSV and JSON."""
        # Save CSV
        if val_metrics or test_metrics:
            report_data = []
            if val_metrics:
                report_data.append(['Validation', 
                                   val_metrics['precision'], 
                                   val_metrics['recall'], 
                                   val_metrics['map50'], 
                                   val_metrics['map5095'], 
                                   val_metrics['f1']])
            if test_metrics:
                report_data.append(['Test', 
                                   test_metrics['precision'], 
                                   test_metrics['recall'], 
                                   test_metrics['map50'], 
                                   test_metrics['map5095'], 
                                   test_metrics['f1']])
            
            df = pd.DataFrame(report_data, 
                             columns=['Dataset', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'F1 Score'])
            csv_path = self.folders['csv'] / 'report_metrics.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Metrics saved to: {csv_path}")
        
        # Save JSON
        all_data = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        json_path = self.folders['metrics'] / 'all_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=2)
    
    def print_summary(self, val_metrics, test_metrics):
        """Print final summary."""
        print("\n" + "="*60)
        print("📋 FINAL SUMMARY FOR REPORT")
        print("="*60)
        
        print("\n| Metric | Validation | Test |")
        print("|--------|------------|------|")
        if val_metrics:
            print(f"| Precision | {val_metrics['precision']:.4f} | ", end="")
            print(f"{test_metrics['precision']:.4f} |" if test_metrics else "N/A |")
            print(f"| Recall | {val_metrics['recall']:.4f} | ", end="")
            print(f"{test_metrics['recall']:.4f} |" if test_metrics else "N/A |")
            print(f"| mAP@0.5 | {val_metrics['map50']:.4f} | ", end="")
            print(f"{test_metrics['map50']:.4f} |" if test_metrics else "N/A |")
            print(f"| mAP@0.5:0.95 | {val_metrics['map5095']:.4f} | ", end="")
            print(f"{test_metrics['map5095']:.4f} |" if test_metrics else "N/A |")
            print(f"| F1 Score | {val_metrics['f1']:.4f} | ", end="")
            print(f"{test_metrics['f1']:.4f} |" if test_metrics else "N/A |")
    
    def run(self):
        """Main pipeline execution."""
        choice = self.get_user_choice()
        
        if choice == '4':
            print("\n👋 Exiting...")
            return
        
        data_path = self.get_data_config()
        test_path = self.data_cfg.get('test')
        
        val_metrics = None
        test_metrics = None
        
        if choice == '2':  # Test only
            model_path = self.get_model_path()
            model = YOLO(model_path)
            conf = float(input("\nConfidence threshold (default: 0.5): ").strip() or 0.5)
            
            val_metrics = self.evaluate_and_copy(model, data_path, 'val', conf)
            
            # Check if test has labels
            test_label_dir = None
            if test_path:
                test_label_dir = Path(test_path).parent / 'labels' if 'images' in str(test_path) else Path(str(test_path).replace('images', 'labels'))
            has_labels = test_label_dir and test_label_dir.exists() and any(test_label_dir.glob('*.txt'))
            
            if has_labels:
                test_metrics = self.evaluate_and_copy(model, data_path, 'test', conf)
            else:
                print("\n⚠️ No labels in test set - running inference only")
                self.run_inference_and_copy(model, test_path, conf)
            
            self.save_metrics(val_metrics, test_metrics)
            self.print_summary(val_metrics, test_metrics)
        
        elif choice == '3':  # Inference only
            model_path = self.get_model_path()
            model = YOLO(model_path)
            conf = float(input("\nConfidence threshold (default: 0.5): ").strip() or 0.5)
            
            if test_path and Path(test_path).exists():
                self.run_inference_and_copy(model, test_path, conf)
            else:
                print("❌ No test images found.")
        
        print("\n" + "="*60)
        print(f"🎉 Done! All files saved in: {self.base_dir}")
        print("="*60)


if __name__ == '__main__':
    pipeline = YOLOPipeline()
    pipeline.run()