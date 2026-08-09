"""
YOLOv8 Object Detection Pipeline - WITH FULL LOCATION TRACKING (FIXED)
Shows you exactly where every file is saved at each step
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

class YOLOPipelineWithTracking:
    def __init__(self):
        # Create main output folder with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(f"YOLO_Results_{self.timestamp}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders
        self.weights_dir = self.base_dir / "01_Weights"
        self.metrics_dir = self.base_dir / "02_Metrics"
        self.plots_dir = self.base_dir / "03_Plots_Graphs"
        self.matrices_dir = self.base_dir / "04_Confusion_Matrices"
        self.predictions_dir = self.base_dir / "05_Predictions"
        self.logs_dir = self.base_dir / "06_Logs"
        
        for d in [self.weights_dir, self.metrics_dir, self.plots_dir, 
                  self.matrices_dir, self.predictions_dir, self.logs_dir]:
            d.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("🎯 YOLOv8 OBJECT DETECTION PIPELINE")
        print("="*70)
        print(f"\n📁 MAIN OUTPUT FOLDER:")
        print(f"   {self.base_dir.absolute()}")
        print("\n📂 SUBFOLDERS CREATED:")
        print(f"   01_Weights:           {self.weights_dir.absolute()}")
        print(f"   02_Metrics:           {self.metrics_dir.absolute()}")
        print(f"   03_Plots_Graphs:      {self.plots_dir.absolute()}")
        print(f"   04_Confusion_Matrices: {self.matrices_dir.absolute()}")
        print(f"   05_Predictions:       {self.predictions_dir.absolute()}")
        print(f"   06_Logs:              {self.logs_dir.absolute()}")
        
        self.setup_device()
    
    def setup_device(self):
        """Setup GPU/CPU device"""
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"\n🖥️  DEVICE: {'GPU (CUDA)' if self.device == 0 else 'CPU'}")
        if self.device == 0:
            print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def print_location(self, step_name, file_path):
        """Print file location for tracking"""
        print(f"\n📍 {step_name}")
        print(f"   Path: {Path(file_path).absolute()}")
    
    def get_user_choice(self):
        """Ask user what to do"""
        print("\n" + "="*70)
        print("WHAT WOULD YOU LIKE TO DO?")
        print("="*70)
        print("   [1] TRAIN NEW MODEL (from scratch)")
        print("   [2] TEST WITH EXISTING MODEL")
        print("   [3] EXIT")
        
        while True:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                return choice
            print("❌ Invalid choice. Enter 1, 2, or 3.")
    
    def get_dataset_path(self):
        """Get dataset configuration"""
        default = r"..\..\My First Project.yolov8\data.yaml"
        print(f"\n📂 DATASET CONFIGURATION")
        print(f"   Default: {default}")
        
        use_default = input("\nUse default path? (y/n): ").strip().lower()
        
        if use_default == 'y':
            data_path = default
        else:
            data_path = input("Enter full path to data.yaml: ").strip()
        
        if not Path(data_path).exists():
            print(f"❌ File not found: {data_path}")
            sys.exit(1)
        
        # Load dataset info
        with open(data_path, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
        
        self.class_names = self.data_cfg.get('names', [])
        print(f"\n✅ DATASET LOADED")
        print(f"   Classes ({len(self.class_names)}): {', '.join(self.class_names)}")
        print(f"   Train path: {self.data_cfg.get('train', 'N/A')}")
        print(f"   Val path: {self.data_cfg.get('val', 'N/A')}")
        print(f"   Test path: {self.data_cfg.get('test', 'N/A')}")
        
        # Save copy to output folder
        shutil.copy(data_path, self.base_dir / "data.yaml")
        self.print_location("Dataset config saved", self.base_dir / "data.yaml")
        
        return data_path
    
    def get_training_params(self):
        """Get training parameters from user"""
        print("\n" + "="*70)
        print("TRAINING PARAMETERS")
        print("="*70)
        print("Press Enter to use default values\n")
        
        epochs = input("Epochs (default: 50): ").strip()
        epochs = int(epochs) if epochs else 50
        
        imgsz = input("Image size (default: 640): ").strip()
        imgsz = int(imgsz) if imgsz else 640
        
        batch = input("Batch size (default: 16): ").strip()
        batch = int(batch) if batch else 16
        
        return epochs, imgsz, batch
    
    def train_model(self, data_path, epochs, imgsz, batch):
        """Train YOLO model and track all outputs"""
        print("\n" + "="*70)
        print("🚀 TRAINING PHASE")
        print("="*70)
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch}")
        print(f"   Device: {'GPU' if self.device == 0 else 'CPU'}")
        print("\n⏳ Training started... This may take several minutes.\n")
        
        # Initialize model
        model = YOLO('yolov8n.pt')
        print("   ✅ Pretrained YOLOv8n model loaded")
        
        # Train
        model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(self.base_dir),
            name='training',
            exist_ok=True,
            device=self.device,
            workers=0,
            verbose=True
        )
        
        # LOCATION 1: Training output folder
        training_dir = self.base_dir / "training"
        self.print_location("Training output folder", training_dir)
        
        # LOCATION 2: Training results CSV
        results_csv = training_dir / "results.csv"
        if results_csv.exists():
            shutil.copy(results_csv, self.logs_dir / "training_results.csv")
            self.print_location("Training results (CSV)", self.logs_dir / "training_results.csv")
        
        # LOCATION 3: Best model weights
        best_model_original = training_dir / "weights" / "best.pt"
        if best_model_original.exists():
            # Copy to our weights folder
            shutil.copy(best_model_original, self.weights_dir / "best.pt")
            self.print_location("Best model weights (COPY THIS PATH)", self.weights_dir / "best.pt")
            print(f"\n   💡 IMPORTANT: Use this path for future testing!")
            print(f"   {self.weights_dir / 'best.pt'}")
            
            # Also save the path to a text file for reference
            weights_path_file = self.base_dir / "MODEL_WEIGHTS_PATH.txt"
            with open(weights_path_file, 'w') as f:
                f.write(str(self.weights_dir / "best.pt"))
            self.print_location("Weights path saved for reference", weights_path_file)
            
            return YOLO(str(self.weights_dir / "best.pt"))
        else:
            print("❌ Training failed - best.pt not found")
            sys.exit(1)
    
    def get_existing_model_path(self):
        """Get path to existing model - with automatic detection"""
        print("\n" + "="*70)
        print("LOAD EXISTING MODEL")
        print("="*70)
        
        # FIRST: Look for recently trained model in our output folders
        recent_outputs = list(Path(".").glob("YOLO_Results_*"))
        if recent_outputs:
            latest = max(recent_outputs, key=lambda p: p.stat().st_ctime)
            weights_path_file = latest / "MODEL_WEIGHTS_PATH.txt"
            
            if weights_path_file.exists():
                with open(weights_path_file, 'r') as f:
                    saved_path = f.read().strip()
                if Path(saved_path).exists():
                    print(f"\n🔍 FOUND RECENTLY TRAINED MODEL:")
                    print(f"   {saved_path}")
                    use_recent = input("\nUse this model? (y/n): ").strip().lower()
                    if use_recent == 'y':
                        return saved_path
            
            # Also check weights folder directly
            recent_weights = latest / "01_Weights" / "best.pt"
            if recent_weights.exists():
                print(f"\n🔍 FOUND RECENT WEIGHTS:")
                print(f"   {recent_weights}")
                use_recent = input("Use this model? (y/n): ").strip().lower()
                if use_recent == 'y':
                    return str(recent_weights)
        
        # SECOND: Look for default trained model location
        default_model = r"..\..\cone_runs\cone_detector\weights\best.pt"
        if Path(default_model).exists():
            print(f"\n📂 DEFAULT MODEL LOCATION:")
            print(f"   {default_model}")
            use_default = input("Use this model? (y/n): ").strip().lower()
            if use_default == 'y':
                return default_model
        
        # THIRD: Ask user manually
        print("\n📂 ENTER MODEL PATH MANUALLY")
        model_path = input("Full path to .pt file: ").strip()
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        return model_path
    
    def evaluate_and_save(self, model, data_path, split_name, conf):
        """Evaluate model and save all outputs"""
        print(f"\n📊 {split_name.upper()} SET EVALUATION")
        print("-"*50)
        
        results = model.val(
            data=data_path,
            split=split_name,
            conf=conf,
            iou=0.5,
            device=self.device,
            workers=0,
            plots=True,
        )
        
        # Extract metrics
        metrics = {
            'precision': results.results_dict.get('metrics/precision(B)', 0),
            'recall': results.results_dict.get('metrics/recall(B)', 0),
            'map50': results.results_dict.get('metrics/mAP50(B)', 0),
            'map5095': results.results_dict.get('metrics/mAP50-95(B)', 0),
        }
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        print(f"\n   {'Metric':<15} {'Score':<10}")
        print(f"   {'-'*25}")
        print(f"   {'Precision':<15} {metrics['precision']:.4f}")
        print(f"   {'Recall':<15} {metrics['recall']:.4f}")
        print(f"   {'mAP@0.5':<15} {metrics['map50']:.4f}")
        print(f"   {'mAP@0.5:0.95':<15} {metrics['map5095']:.4f}")
        print(f"   {'F1 Score':<15} {metrics['f1']:.4f}")
        
        # Find and copy YOLO output files
        runs_dir = Path(r"..\..\My First Project.yolov8\runs\detect")
        if runs_dir.exists():
            for folder in runs_dir.iterdir():
                if folder.is_dir() and split_name in folder.name.lower():
                    # Copy plots
                    for png_file in folder.glob("*.png"):
                        if "confusion" in png_file.name.lower():
                            dest = self.matrices_dir / f"{split_name}_{png_file.name}"
                        else:
                            dest = self.plots_dir / f"{split_name}_{png_file.name}"
                        shutil.copy(png_file, dest)
                        self.print_location(f"{split_name} {png_file.name}", dest)
        
        return metrics
    
    def run_inference(self, model, test_path, conf):
        """Run inference on test images"""
        print(f"\n🔍 INFERENCE ON TEST IMAGES")
        print("-"*50)
        
        if not test_path or not Path(test_path).exists():
            print(f"❌ Test path not found: {test_path}")
            return
        
        detections = []
        results = model.predict(
            source=test_path,
            conf=conf,
            iou=0.45,
            stream=True,
            device=self.device,
            workers=0
        )
        
        for result in results:
            img = result.orig_img.copy()
            img_name = Path(result.path).name
            
            if result.boxes:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    class_name = result.names[cls_id]
                    
                    detections.append([img_name, class_name, round(conf_val, 4), x1, y1, x2, y2])
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf_val:.2f}"
                    cv2.putText(img, label, (x1, max(y1-10, 20)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save annotated image
            output_path = self.predictions_dir / img_name
            cv2.imwrite(str(output_path), img)
        
        # Save detections CSV
        csv_path = self.metrics_dir / 'test_detections.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            writer.writerows(detections)
        
        self.print_location(f"Annotated test images folder", self.predictions_dir)
        self.print_location(f"Detections CSV", csv_path)
        print(f"\n   Total detections: {len(detections)}")
    
    def save_final_results(self, val_metrics, test_metrics, training_params=None):
        """Save all final results"""
        print("\n" + "="*70)
        print("💾 SAVING FINAL RESULTS")
        print("="*70)
        
        # Create results table
        results_data = []
        if val_metrics:
            results_data.append(['Validation', 
                               val_metrics['precision'], 
                               val_metrics['recall'], 
                               val_metrics['map50'], 
                               val_metrics['map5095'], 
                               val_metrics['f1']])
        if test_metrics:
            results_data.append(['Test', 
                               test_metrics['precision'], 
                               test_metrics['recall'], 
                               test_metrics['map50'], 
                               test_metrics['map5095'], 
                               test_metrics['f1']])
        
        # Save CSV
        df = pd.DataFrame(results_data, 
                         columns=['Dataset', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'F1 Score'])
        csv_path = self.metrics_dir / 'final_results.csv'
        df.to_csv(csv_path, index=False)
        self.print_location("Final results CSV", csv_path)
        
        # Save JSON
        all_data = {
            'timestamp': self.timestamp,
            'device': str(self.device),
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'training_parameters': training_params,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
        }
        json_path = self.metrics_dir / 'complete_results.json'
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        self.print_location("Complete results JSON", json_path)
        
        # Save summary text file
        summary_path = self.base_dir / 'README_Results.txt'
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("YOLOv8 OBJECT DETECTION - COMPLETE RESULTS\n")
            f.write(f"Date: {self.timestamp}\n")
            f.write("="*70 + "\n\n")
            
            f.write("📁 OUTPUT FOLDER LOCATIONS:\n")
            f.write(f"   Main folder: {self.base_dir.absolute()}\n")
            f.write(f"   Weights: {self.weights_dir.absolute()}\n")
            f.write(f"   Metrics: {self.metrics_dir.absolute()}\n")
            f.write(f"   Plots: {self.plots_dir.absolute()}\n")
            f.write(f"   Confusion Matrices: {self.matrices_dir.absolute()}\n")
            f.write(f"   Predictions: {self.predictions_dir.absolute()}\n")
            f.write(f"   Logs: {self.logs_dir.absolute()}\n\n")
            
            f.write("📊 RESULTS:\n")
            f.write("-"*40 + "\n")
            if val_metrics:
                f.write(f"Validation mAP@0.5: {val_metrics['map50']:.4f}\n")
                f.write(f"Validation Precision: {val_metrics['precision']:.4f}\n")
                f.write(f"Validation Recall: {val_metrics['recall']:.4f}\n")
                f.write(f"Validation F1 Score: {val_metrics['f1']:.4f}\n\n")
            if test_metrics:
                f.write(f"Test mAP@0.5: {test_metrics['map50']:.4f}\n")
                f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
                f.write(f"Test Recall: {test_metrics['recall']:.4f}\n")
                f.write(f"Test F1 Score: {test_metrics['f1']:.4f}\n")
        
        self.print_location("Summary README", summary_path)
    
    def print_final_summary(self, val_metrics, test_metrics):
        """Print final summary with all locations (FIXED)"""
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📊 RESULTS SUMMARY:")
        print("-"*50)
        print(f"{'Metric':<15} {'Validation':<15} {'Test':<15}")
        print("-"*45)
        
        if val_metrics:
            val_precision = val_metrics.get('precision', 0)
            val_recall = val_metrics.get('recall', 0)
            val_map50 = val_metrics.get('map50', 0)
            val_f1 = val_metrics.get('f1', 0)
            
            if test_metrics:
                test_precision = test_metrics.get('precision', 0)
                test_recall = test_metrics.get('recall', 0)
                test_map50 = test_metrics.get('map50', 0)
                test_f1 = test_metrics.get('f1', 0)
                
                print(f"{'Precision':<15} {val_precision:.4f}{'':<10} {test_precision:.4f}")
                print(f"{'Recall':<15} {val_recall:.4f}{'':<10} {test_recall:.4f}")
                print(f"{'mAP@0.5':<15} {val_map50:.4f}{'':<10} {test_map50:.4f}")
                print(f"{'F1 Score':<15} {val_f1:.4f}{'':<10} {test_f1:.4f}")
            else:
                print(f"{'Precision':<15} {val_precision:.4f}{'':<10} {'N/A'}")
                print(f"{'Recall':<15} {val_recall:.4f}{'':<10} {'N/A'}")
                print(f"{'mAP@0.5':<15} {val_map50:.4f}{'':<10} {'N/A'}")
                print(f"{'F1 Score':<15} {val_f1:.4f}{'':<10} {'N/A'}")
        
        print("\n📁 ALL FILES SAVED IN:")
        print(f"   {self.base_dir.absolute()}")
        
        print("\n📂 QUICK ACCESS TO KEY FILES:")
        print(f"   Model weights:  {self.weights_dir / 'best.pt'}")
        print(f"   Final metrics:  {self.metrics_dir / 'final_results.csv'}")
        print(f"   Predictions:    {self.predictions_dir}")
        print(f"   README:         {self.base_dir / 'README_Results.txt'}")
        
        print("\n💡 TIP: Save the model weights path for future use:")
        print(f"   {self.weights_dir / 'best.pt'}")
        
        print("\n" + "="*70)
    
    def run(self):
        """Main pipeline execution"""
        choice = self.get_user_choice()
        
        if choice == '3':
            print("\n👋 Exiting...")
            return
        
        # Get dataset
        data_path = self.get_dataset_path()
        test_path = self.data_cfg.get('test')
        
        val_metrics = None
        test_metrics = None
        model = None
        
        if choice == '1':  # Train new model
            epochs, imgsz, batch = self.get_training_params()
            conf = float(input("\nConfidence threshold (default: 0.5): ").strip() or 0.5)
            
            # Train
            model = self.train_model(data_path, epochs, imgsz, batch)
            
            # Evaluate on validation
            val_metrics = self.evaluate_and_save(model, data_path, 'val', conf)
            
            # Evaluate on test (if labels exist)
            if test_path:
                test_label_dir = Path(test_path).parent / 'labels'
                if test_label_dir.exists() and any(test_label_dir.glob('*.txt')):
                    test_metrics = self.evaluate_and_save(model, data_path, 'test', conf)
            
            # Save results
            self.save_final_results(val_metrics, test_metrics, 
                                   {'epochs': epochs, 'imgsz': imgsz, 'batch': batch, 'conf': conf})
            
        elif choice == '2':  # Test with existing model
            model_path = self.get_existing_model_path()
            conf = float(input("\nConfidence threshold (default: 0.5): ").strip() or 0.5)
            
            print(f"\n✅ Loading model from: {model_path}")
            model = YOLO(model_path)
            
            # Copy model to our folder for reference
            shutil.copy(model_path, self.weights_dir / "best.pt")
            self.print_location("Model copied to output folder", self.weights_dir / "best.pt")
            
            # Evaluate on validation
            val_metrics = self.evaluate_and_save(model, data_path, 'val', conf)
            
            # Evaluate on test (if labels exist)
            if test_path:
                test_label_dir = Path(test_path).parent / 'labels'
                if test_label_dir.exists() and any(test_label_dir.glob('*.txt')):
                    test_metrics = self.evaluate_and_save(model, data_path, 'test', conf)
            
            # Save results
            self.save_final_results(val_metrics, test_metrics, None)
        
        # Run inference on test images (always do this)
        if model and test_path and Path(test_path).exists():
            self.run_inference(model, test_path, conf)
        
        # Print final summary with all locations
        self.print_final_summary(val_metrics, test_metrics)


if __name__ == '__main__':
    pipeline = YOLOPipelineWithTracking()
    pipeline.run()