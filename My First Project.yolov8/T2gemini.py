import os
import torch
import yaml
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import cv2

# ---------- CONFIG ----------
PROJECT_ROOT = Path(r"..\..\My First Project.yolov8")
DATA_YAML = PROJECT_ROOT / "data.yaml"
OUTPUT_BASE = PROJECT_ROOT / "deepseek_final_optimization"

# --- THE "SMALL DATASET" SURVIVAL PARAMETERS ---
SMALL_DATA_HYPS = {
    'hsv_h': 0.02, 'hsv_s': 0.9, 'hsv_v': 0.5, 
    'degrees': 180.0,    
    'translate': 0.3,    
    'scale': 0.6,        
    'shear': 2.0,        
    'flipud': 0.5, 'fliplr': 0.5,
    'mosaic': 1.0,       
    'mixup': 0.2,        
    'copy_paste': 0.3,   
    'label_smoothing': 0.15, 
    'dropout': 0.1       
}

def get_next_run_number():
    OUTPUT_BASE.mkdir(exist_ok=True)
    existing = [d for d in OUTPUT_BASE.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not existing: return 1
    return max([int(d.name.split("_")[1]) for d in existing]) + 1

def main():
    run_num = get_next_run_number()
    run_dir = OUTPUT_BASE / f"run_{run_num:03d}"
    
    # Pre-create necessary directories
    weights_dir = run_dir / "weights"
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    predictions_dir = run_dir / "predictions"
    
    for d in [weights_dir, metrics_dir, plots_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 RUNNING OPTIMIZED PIPELINE #{run_num}")

    # 1. LOAD MODEL
    model = YOLO("yolov8n.pt") 

    # 2. OPTIMIZED TRAINING
    # project and name here determine where results.jpg etc. go
    model.train(
        data=str(DATA_YAML),
        epochs=150,            
        patience=30,           
        batch=8,               
        imgsz=640,             
        project=str(run_dir),
        name="train_job",
        device=device,
        close_mosaic=15,       
        **SMALL_DATA_HYPS
    )

    # 3. EXPORT BEST WEIGHTS
    best = run_dir / "train_job" / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, weights_dir / "best_optimized.pt")

    # 4. ORGANIZE PLOTS
    train_out = run_dir / "train_job"
    if train_out.exists():
        for ext in ["*.png", "*.jpg", "*.csv"]:
            for f in train_out.glob(ext):
                shutil.copy(str(f), str(plots_dir / f.name))

    # 5. TEST PREDICTIONS
    with open(DATA_YAML) as f:
        test_data = yaml.safe_load(f)
        test_path = test_data.get('test')

    if test_path and os.path.exists(test_path):
        print("\n🔍 Generating visual proof on test set...")
        # Force save to a temp folder we control
        temp_preds = run_dir / "temp_preds"
        model.predict(
            source=test_path, 
            conf=0.5, 
            save=True, 
            project=str(run_dir), 
            name="temp_preds"
        )
        
        # Move images to the final 'predictions' folder
        if temp_preds.exists():
            for img_file in temp_preds.glob("*.*"):
                shutil.copy(img_file, predictions_dir / img_file.name)
            # Safe removal
            shutil.rmtree(temp_preds)

    print(f"\n✅ DONE. Everything saved in: {run_dir.absolute()}")

if __name__ == "__main__":
    main()