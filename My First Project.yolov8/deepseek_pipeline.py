"""
YOLOv8 pipeline – stores each run in numbered folder inside 'deepseek'
"""

import os, torch, yaml, shutil, csv, json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import cv2
import pandas as pd

# ---------- CONFIG ----------
PROJECT_ROOT = Path(r"C:\Users\gbhan\Desktop\YOLO DETCTION\My First Project.yolov8")
DATA_YAML = PROJECT_ROOT / "data.yaml"
OUTPUT_BASE = PROJECT_ROOT / "deepseek"
# ----------------------------

def get_next_run_number():
    OUTPUT_BASE.mkdir(exist_ok=True)
    existing = [d for d in OUTPUT_BASE.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not existing:
        return 1
    numbers = [int(d.name.split("_")[1]) for d in existing]
    return max(numbers) + 1

def save_location(step, path):
    print(f"\n📍 {step}\n   {path.absolute()}")

def main():
    print("\n🔍 YOLO Pipeline – deepseek folder")
    choice = input("Train (1) or Test only (2)? ").strip()
    if choice not in ["1","2"]:
        print("Exit")
        return

    # Create run folder
    run_num = get_next_run_number()
    run_dir = OUTPUT_BASE / f"run_{run_num:03d}"
    for sub in ["01_weights","02_metrics","03_plots","04_confusion","05_predictions"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "01_weights"
    metrics_dir = run_dir / "02_metrics"
    plots_dir = run_dir / "03_plots"
    conf_dir = run_dir / "04_confusion"
    pred_dir = run_dir / "05_predictions"

    print(f"\n📁 All results → {run_dir.absolute()}")

    # Device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {'GPU' if device==0 else 'CPU'}")

    # Load dataset info
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    classes = data_cfg.get("names", [])
    test_path = data_cfg.get("test")
    print(f"Classes: {classes}")

    # Confidence threshold
    conf = float(input("Confidence threshold (default 0.5): ").strip() or 0.5)

    if choice == "1":  # Train
        epochs = int(input("Epochs (default 50): ").strip() or 50)
        imgsz = int(input("Image size (default 640): ").strip() or 640)
        batch = int(input("Batch size (default 16): ").strip() or 16)

        model = YOLO("yolov8n.pt")
        model.train(data=str(DATA_YAML), epochs=epochs, imgsz=imgsz, batch=batch,
                    project=str(run_dir), name="train", exist_ok=True,
                    device=device, workers=0, verbose=True)

        # Copy best model
        src_best = run_dir / "train" / "weights" / "best.pt"
        if src_best.exists():
            shutil.copy(src_best, weights_dir / "best.pt")
            save_location("Trained weights", weights_dir / "best.pt")
        else:
            print("❌ Training failed")
            return
        model = YOLO(str(weights_dir / "best.pt"))
    else:  # Test only
        # Try to find latest weights in deepseek runs
        latest_weights = None
        for r in sorted(OUTPUT_BASE.glob("run_*"), reverse=True):
            w = r / "01_weights" / "best.pt"
            if w.exists():
                latest_weights = w
                break
        if latest_weights:
            print(f"\n🔍 Found recent weights: {latest_weights}")
            use = input("Use these? (y/n): ").strip().lower()
            if use == "y":
                model = YOLO(str(latest_weights))
                shutil.copy(latest_weights, weights_dir / "best.pt")
            else:
                path = input("Enter full path to .pt file: ").strip()
                model = YOLO(path)
                shutil.copy(path, weights_dir / "best.pt")
        else:
            path = input("Enter path to .pt file: ").strip()
            model = YOLO(path)
            shutil.copy(path, weights_dir / "best.pt")
        save_location("Loaded weights", weights_dir / "best.pt")

    # ---------- Validation ----------
    print("\n📊 Validation evaluation")
    val_res = model.val(data=str(DATA_YAML), split="val", conf=conf, iou=0.5,
                        device=device, workers=0, plots=True)
    val_metrics = {
        "precision": val_res.results_dict.get("metrics/precision(B)",0),
        "recall": val_res.results_dict.get("metrics/recall(B)",0),
        "map50": val_res.results_dict.get("metrics/mAP50(B)",0),
        "map5095": val_res.results_dict.get("metrics/mAP50-95(B)",0),
    }
    val_metrics["f1"] = 2*val_metrics["precision"]*val_metrics["recall"]/(val_metrics["precision"]+val_metrics["recall"]+1e-6)

    print(f"   Precision: {val_metrics['precision']:.4f}")
    print(f"   Recall:    {val_metrics['recall']:.4f}")
    print(f"   mAP@0.5:   {val_metrics['map50']:.4f}")

    # ---------- Test (if labels exist) ----------
    test_metrics = None
    if test_path and Path(test_path).exists():
        label_dir = Path(test_path).parent / "labels"
        if label_dir.exists() and any(label_dir.glob("*.txt")):
            print("\n📊 Test evaluation (with ground truth)")
            test_res = model.val(data=str(DATA_YAML), split="test", conf=conf, iou=0.5,
                                 device=device, workers=0, plots=True)
            test_metrics = {
                "precision": test_res.results_dict.get("metrics/precision(B)",0),
                "recall": test_res.results_dict.get("metrics/recall(B)",0),
                "map50": test_res.results_dict.get("metrics/mAP50(B)",0),
                "map5095": test_res.results_dict.get("metrics/mAP50-95(B)",0),
            }
            test_metrics["f1"] = 2*test_metrics["precision"]*test_metrics["recall"]/(test_metrics["precision"]+test_metrics["recall"]+1e-6)
            print(f"   Precision: {test_metrics['precision']:.4f}")
            print(f"   Recall:    {test_metrics['recall']:.4f}")
            print(f"   mAP@0.5:   {test_metrics['map50']:.4f}")

    # ---------- Copy plots and confusion matrices ----------
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    if runs_dir.exists():
        for folder in runs_dir.iterdir():
            if folder.is_dir():
                for png in folder.glob("*.png"):
                    if "confusion" in png.name.lower():
                        shutil.copy(png, conf_dir / png.name)
                    else:
                        shutil.copy(png, plots_dir / png.name)
        save_location("Plots & confusion matrices", plots_dir)
        save_location("Confusion matrices", conf_dir)

    # ---------- Inference on test images (always) ----------
    if test_path and Path(test_path).exists():
        print("\n🔍 Inference on test images (drawing boxes)")
        detections = []
        results = model.predict(source=test_path, conf=conf, iou=0.45,
                                stream=True, device=device, workers=0)
        for r in results:
            img = r.orig_img.copy()
            img_name = Path(r.path).name
            if r.boxes:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    c = float(box.conf[0])
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append([img_name, r.names[cls], round(c,4), x1,y1,x2,y2])
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, f"{r.names[cls]} {c:.2f}", (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imwrite(str(pred_dir / img_name), img)
        # Save detections CSV
        csv_path = metrics_dir / "test_detections.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["image","class","confidence","x1","y1","x2","y2"])
            w.writerows(detections)
        save_location("Prediction images", pred_dir)
        print(f"   Detections CSV: {csv_path}")

    # ---------- Save final metrics ----------
    results_df = pd.DataFrame([
        ["Validation", val_metrics["precision"], val_metrics["recall"],
         val_metrics["map50"], val_metrics["map5095"], val_metrics["f1"]],
        *([["Test", test_metrics["precision"], test_metrics["recall"],
            test_metrics["map50"], test_metrics["map5095"], test_metrics["f1"]]] if test_metrics else [])
    ], columns=["Dataset","Precision","Recall","mAP@0.5","mAP@0.5:0.95","F1 Score"])
    results_df.to_csv(metrics_dir / "final_results.csv", index=False)

    # README
    readme = run_dir / "README.txt"
    readme.write_text(f"""
============================================================
YOLO DETECTION RUN #{run_num:03d}
Date: {datetime.now()}
============================================================

📁 Folders:
  weights     : {weights_dir.absolute()}
  metrics     : {metrics_dir.absolute()}
  plots       : {plots_dir.absolute()}
  confusion   : {conf_dir.absolute()}
  predictions : {pred_dir.absolute()}

📊 Results:
  Validation mAP@0.5: {val_metrics['map50']:.4f}
  Validation Precision: {val_metrics['precision']:.4f}
  Validation Recall: {val_metrics['recall']:.4f}
""")
    print(f"\n✅ All saved in: {run_dir.absolute()}")
    print("="*60)

if __name__ == "__main__":
    main()