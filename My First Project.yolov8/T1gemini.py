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
PROJECT_ROOT = Path(r"C:\Users\gbhan\Desktop\YOLO DETCTION\My First Project.yolov8")
DATA_YAML = PROJECT_ROOT / "data.yaml"
OUTPUT_BASE = PROJECT_ROOT / "deepseek_robust"

# --- THE MENTOR'S OVERRIDES (Augmentation to kill overfitting) ---
ROBUST_HYPS = {
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, # Color/Light jitter
    'degrees': 180.0,    # Random rotation
    'translate': 0.2,    # Kill center-bias
    'scale': 0.5,        # Small/Large variations
    'shear': 2.0,        # Perspective change
    'flipud': 0.5, 'fliplr': 0.5,
    'mosaic': 1.0,       # 4-image mix
    'mixup': 0.1,        # Occlusion handling
    'label_smoothing': 0.1 # Stop hallucinating floor patterns
}

def get_next_run():
    OUTPUT_BASE.mkdir(exist_ok=True)
    existing = [d for d in OUTPUT_BASE.iterdir() if d.is_dir() and d.name.startswith("run_")]
    num = max([int(d.name.split("_")[1]) for d in existing]) + 1 if existing else 1
    run_dir = OUTPUT_BASE / f"run_{num:03d}"
    for sub in ["weights", "metrics", "plots", "predictions"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir

def main():
    run_dir = get_next_run()
    device = 0 if torch.cuda.is_available() else "cpu"
    
    # 1. INITIALIZE MODEL
    model = YOLO("yolov8n.pt") 

    # 2. ROBUST TRAINING (Includes automatic generation of results.jpg, curves, etc.)
    print(f"\n🚀 Training started. Results saving to: {run_dir}")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        patience=20, # Early stopping: stops before it starts "memorizing"
        batch=16,
        imgsz=640,
        project=str(run_dir),
        name="train_results",
        device=device,
        close_mosaic=10,
        **ROBUST_HYPS
    )

    # 3. EXPORT BEST WEIGHTS
    best_weights = run_dir / "train_results" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, run_dir / "weights" / "best_robust.pt")

    # 4. VALIDATION & METRIC COLLECTION
    print("\n📊 Evaluating on Validation Set...")
    val_results = model.val(conf=0.25) # Low conf here to see the "truth" in confusion matrix

    # Move generated plots to your organized folders
    for f in (run_dir / "train_results").glob("*.png"):
        shutil.copy(f, run_dir / "plots" / f.name)
    for f in (run_dir / "train_results").glob("*.jpg"):
        shutil.copy(f, run_dir / "plots" / f.name)

    # 5. PREDICTION ON TEST IMAGES (Visualizing the boxes)
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
        test_path = data_cfg.get('test')

    if test_path and os.path.exists(test_path):
        print("\n🔍 Drawing boxes on test images...")
        results = model.predict(source=test_path, save=True, project=str(run_dir), name="preds", conf=0.5)
        
        # Clean up prediction images into your folder
        pred_source = run_dir / "preds"
        for img in pred_source.glob("*.*"):
            shutil.copy(img, run_dir / "predictions" / img.name)
        shutil.rmtree(pred_source)

    print(f"\n✅ COMPLETE. Check {run_dir}/plots for your new, honest curves.")

if __name__ == "__main__":
    main()