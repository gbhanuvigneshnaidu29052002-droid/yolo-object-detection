"""
YOLOv8 Object Detection Pipeline - Interactive Version (FIXED)
---------------------------------------------------------------
Asks user what to do: Train, Test, or Both
Organizes all results in a single folder with subfolders
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
        # Create main output folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(f"yolo_output_{timestamp}")
        
        # Create all subfolders
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
        print("  2. Train only (then exit)")
        print("  3. Test only (using existing model)")
        print("  4. Inference only (draw boxes, no metrics)")
        print("  5. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
    
    def get_data_config(self):
        """Get data.yaml path from user."""
        default_path = r"C:\Users\gbhan\Desktop\YOLO DETCTION\My First Project.yolov8\data.yaml"
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
        
        # Save config to output folder
        shutil.copy(data_path, self.folders['config'] / "data.yaml")
        
        return data_path
    
    def get_training_params(self):
        """Get training parameters from user."""
        print("\n" + "="*60)
        print("📊 TRAINING PARAMETERS")
        print("="*60)
        print("Press Enter to use default values\n")
        
        epochs = input(f"Epochs (default: 50): ").strip()
        epochs = int(epochs) if epochs else 50
        
        imgsz = input(f"Image size (default: 640): ").strip()
        imgsz = int(imgsz) if imgsz else 640
        
        batch = input(f"Batch size (default: 16): ").strip()
        batch = int(batch) if batch else 16
        
        conf = input(f"Confidence threshold (default: 0.5): ").strip()
        conf = float(conf) if conf else 0.5
        
        return epochs, imgsz, batch, conf
    
    def get_model_path(self):
        """Get path to existing model for testing."""
        print("\n" + "="*60)
        print("📂 LOAD EXISTING MODEL")
        print("="*60)
        
        # Try to find the most recent model in the current directory
        yolo_outputs = list(Path(".").glob("yolo_output_*"))
        if yolo_outputs:
            latest_output = max(yolo_outputs, key=lambda p: p.stat().st_ctime)
            suggested_model = latest_output / "01_weights" / "best.pt"
            if suggested_model.exists():
                print(f"Found recent model: {suggested_model}")
                use_suggested = input("Use this model? (y/n): ").strip().lower()
                if use_suggested == 'y':
                    return str(suggested_model)
        
        default_model = r"C:\Users\gbhan\Desktop\YOLO DETCTION\cone_runs\cone_detector\weights\best.pt"
        print(f"Default: {default_model}")
        
        use_default = input("Use default model? (y/n): ").strip().lower()
        
        if use_default == 'y':
            model_path = default_model
        else:
            model_path = input("Enter path to model (.pt file): ").strip()
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        print(f"✅ Model loaded: {model_path}")
        return model_path
    
    def train_model(self, data_path, epochs, imgsz, batch):
        """Train YOLO model."""
        print("\n" + "="*60)
        print("🚀 TRAINING PHASE")
        print("="*60)
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch}")
        
        # Create a training output directory inside base_dir
        train_output_dir = self.base_dir / "training_output"
        train_output_dir.mkdir(exist_ok=True)
        
        # Initialize model
        model = YOLO('yolov8n.pt')
        
        # Train - let Ultralytics save where it wants
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(self.base_dir),
            name='training_output',
            exist_ok=True,
            patience=20,
            device=self.device,
            workers=0,  # Windows safe
            pretrained=True,
            lr0=0.01,
            fliplr=0.5,
            mosaic=1.0
        )
        
        # Find where Ultralytics saved the best model
        # It could be in different locations depending on the version
        possible_paths = [
            self.base_dir / "training_output" / "weights" / "best.pt",
            self.base_dir / "training_output" / "best.pt",
            Path("runs") / "detect" / "train" / "weights" / "best.pt",
            Path("runs") / "detect" / "training_output" / "weights" / "best.pt",
        ]
        
        best_model_found = None
        for path in possible_paths:
            if path.exists():
                best_model_found = path
                break
        
        # Also search recursively in the base_dir
        if not best_model_found:
            for pt_file in self.base_dir.rglob("best.pt"):
                best_model_found = pt_file
                break
        
        if best_model_found:
            # Copy to our organized weights folder
            shutil.copy(best_model_found, self.folders['weights'] / "best.pt")
            print(f"\n✅ Best model copied to: {self.folders['weights'] / 'best.pt'}")
            
            # Copy plots
            for png_file in self.base_dir.rglob("*.png"):
                if "confusion_matrix" in png_file.name or "results" in png_file.name or "F1" in png_file.name or "PR" in png_file.name:
                    shutil.copy(png_file, self.folders['plots'] / png_file.name)
            
            return YOLO(str(self.folders['weights'] / "best.pt"))
        else:
            print(f"❌ Could not find best.pt in {self.base_dir}")
            print("   Training may have failed or saved elsewhere.")
            sys.exit(1)
    
    def evaluate_model(self, model, data_path, split_name, conf_threshold):
        """Evaluate model on validation or test set."""
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
        
        # Copy any new plots
        for png_file in Path(".").rglob("*.png"):
            if "confusion_matrix" in png_file.name or "results" in png_file.name:
                try:
                    shutil.copy(png_file, self.folders['plots'] / f"{split_name}_{png_file.name}")
                except:
                    pass
        
        return metrics
    
    def run_inference(self, model, test_path, conf_threshold):
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
                    
                    # Draw bounding box
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
        
        print(f"\n✅ Saved:")
        print(f"   - Annotated images: {self.folders['predictions']} ({len(detections)} detections)")
        print(f"   - Detections CSV: {csv_path}")
        
        return detections
    
    def save_all_metrics(self, val_metrics=None, test_metrics=None, training_params=None):
        """Save all metrics to files."""
        all_data = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'training_parameters': training_params,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        # Save JSON
        json_path = self.folders['metrics'] / 'all_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        # Save CSV for report
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
        
        if choice == '5':
            print("\n👋 Exiting...")
            return
        
        # Get data configuration
        data_path = self.get_data_config()
        test_path = self.data_cfg.get('test')
        
        val_metrics = None
        test_metrics = None
        model = None
        
        # Case 1: Train + Test
        if choice == '1':
            epochs, imgsz, batch, conf = self.get_training_params()
            model = self.train_model(data_path, epochs, imgsz, batch)
            val_metrics = self.evaluate_model(model, data_path, 'val', conf)
            
            # Check if test set has labels
            test_label_dir = None
            if test_path:
                test_label_dir = Path(test_path).parent / 'labels' if 'images' in str(test_path) else Path(str(test_path).replace('images', 'labels'))
            has_test_labels = test_label_dir and test_label_dir.exists() and any(test_label_dir.glob('*.txt'))
            
            if has_test_labels:
                test_metrics = self.evaluate_model(model, data_path, 'test', conf)
            else:
                print("\n⚠️ No labels in test set - running inference only")
                self.run_inference(model, test_path, conf)
            
            self.save_all_metrics(val_metrics, test_metrics, 
                                 {'epochs': epochs, 'imgsz': imgsz, 'batch': batch, 'conf': conf})
            self.print_summary(val_metrics, test_metrics)
        
        # Case 2: Train only
        elif choice == '2':
            epochs, imgsz, batch, conf = self.get_training_params()
            model = self.train_model(data_path, epochs, imgsz, batch)
            val_metrics = self.evaluate_model(model, data_path, 'val', conf)
            self.save_all_metrics(val_metrics, None, 
                                 {'epochs': epochs, 'imgsz': imgsz, 'batch': batch, 'conf': conf})
            print("\n✅ Training completed. Model saved.")
        
        # Case 3: Test only (using existing model)
        elif choice == '3':
            model_path = self.get_model_path()
            model = YOLO(model_path)
            conf = float(input("\nConfidence threshold (default: 0.5): ").strip() or 0.5)
            
            val_metrics = self.evaluate_model(model, data_path, 'val', conf)
            
            test_label_dir = None
            if test_path:
                test_label_dir = Path(test_path).parent / 'labels' if 'images' in str(test_path) else Path(str(test_path).replace('images', 'labels'))
            has_test_labels = test_label_dir and test_label_dir.exists() and any(test_label_dir.glob('*.txt'))
            
            if has_test_labels:
                test_metrics = self.evaluate_model(model, data_path, 'test', conf)
            else:
                print("\n⚠️ No labels in test set - running inference only")
                self.run_inference(model, test_path, conf)
            
            self.save_all_metrics(val_metrics, test_metrics, None)
            self.print_summary(val_metrics, test_metrics)
        
        # Case 4: Inference only (no metrics)
        elif choice == '4':
            model_path = self.get_model_path()
            model = YOLO(model_path)
            conf = float(input("\nConfidence threshold (default: 0.5): ").strip() or 0.5)
            
            if test_path and Path(test_path).exists():
                self.run_inference(model, test_path, conf)
                print("\n✅ Inference completed. Check the predictions folder.")
            else:
                print("❌ No test images found.")
        
        print("\n" + "="*60)
        print(f"🎉 Done! All files saved in: {self.base_dir}")
        print("="*60)


if __name__ == '__main__':
    pipeline = YOLOPipeline()
    pipeline.run()