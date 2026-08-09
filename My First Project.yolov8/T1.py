from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import torch
import cv2
import os

# ======================================================
# CONFIG – EDIT THESE 3 PATHS ONLY
# ======================================================
ROOT = r"..\..\My First Project.yolov8"
PROJECT_DIR = Path(r"..\..\My First Project.yolov8\perplexity outputs")
MODEL_NAME = "yolov8n.pt"  # or "yolov8s.pt" if GPU has VRAM

DATA_YAML = os.path.join(ROOT, "data.yaml")

# folders inside ROOT (Roboflow export)
TRAIN_IMAGES = os.path.join(ROOT, "train", "images")
VAL_IMAGES   = os.path.join(ROOT, "valid", "images")
TEST_IMAGES  = os.path.join(ROOT, "test", "images")

TRAIN_NAME = "cone_detector"
BEST_WEIGHTS = PROJECT_DIR / TRAIN_NAME / "weights" / "best.pt"

EPOCHS = 50
IMG_SIZE = 640
CONF_THRESHOLD = 0.4   # favour recall over a few false positives


# ======================================================
# HELPERS
# ======================================================
def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def train_model():
    device = get_device()
    print(f"\n[TRAIN] Using device: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(PROJECT_DIR),
        name=TRAIN_NAME,
        exist_ok=True,
        patience=20,
        device=device,
        workers=0,      # Windows
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    run_dir = PROJECT_DIR / TRAIN_NAME
    print("\n[TRAIN] Finished.")
    print("       Metrics & loss curves:   ", run_dir / "results.png")
    print("       Extra plots (PR, F1, CM):", run_dir)


def validate_model(split="val"):
    if not BEST_WEIGHTS.exists():
        print("\n[VAL] ERROR: best.pt not found at", BEST_WEIGHTS)
        return

    device = get_device()
    print(f"\n[VAL] Using device: {'GPU' if device == 0 else 'CPU'}")
    print(f"[VAL] Split: {split}")

    model = YOLO(str(BEST_WEIGHTS))

    metrics = model.val(
        data=DATA_YAML,
        split=split,          # "val" or "test"
        conf=CONF_THRESHOLD,
        device=device,
        workers=0,
    )

    precision = metrics.results_dict.get("metrics/precision(B)", 0)
    recall    = metrics.results_dict.get("metrics/recall(B)", 0)
    map50     = metrics.results_dict.get("metrics/mAP50(B)", 0)
    map5095   = metrics.results_dict.get("metrics/mAP50-95(B)", 0)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n===== VALIDATION METRICS =====")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"mAP@0.50       : {map50:.4f}")
    print(f"mAP@0.50:0.95  : {map5095:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Approx. accuracy-style summary (use mAP50): {map50*100:.2f}%")

    run_dir = PROJECT_DIR / TRAIN_NAME
    print("\n[VAL] Plots (confusion_matrix, PR/F1 curves, etc.) are saved in:")
    print("      ", run_dir)


def predict_on_folder(source, out_name):
    if not BEST_WEIGHTS.exists():
        print("\n[PRED] ERROR: best.pt not found at", BEST_WEIGHTS)
        return

    device = get_device()
    print(f"\n[PRED] Using device: {'GPU' if device == 0 else 'CPU'}")
    print(f"[PRED] Source folder: {source}")

    model = YOLO(str(BEST_WEIGHTS))

    output_dir = PROJECT_DIR / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=source,
        conf=CONF_THRESHOLD,
        save=False,
        stream=True,
        device=device,
        workers=0,
    )

    summary_lines = []

    for r in results:
        img = r.orig_img.copy()
        image_path = r.path
        image_name = os.path.basename(image_path)

        class_counter = Counter()
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_name = r.names[cls_id]
            class_counter[cls_name] += 1
            label = f"{cls_name} {conf*100:.1f}%"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        summary_lines.append(
            f"{image_name}: " +
            ", ".join(f"{k}={v}" for k, v in class_counter.items())
        )

        cv2.imwrite(str(output_dir / image_name), img)

    summary_file = PROJECT_DIR / f"{out_name}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n[PRED] Annotated images saved in: {output_dir}")
    print(f"[PRED] Per-image counts saved in: {summary_file}")


# ======================================================
# MAIN MENU
# ======================================================
def main():
    while True:
        print("\n================ YOLO CONE PIPELINE ================")
        print("1) Train model from pretrained YOLO")
        print("2) Validate best.pt on VALIDATION set")
        print("3) Validate best.pt on TEST set")
        print("4) Run predictions on TEST set (save images + summary)")
        print("5) Exit")
        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            validate_model(split="val")
        elif choice == "3":
            validate_model(split="test")
        elif choice == "4":
            predict_on_folder(TEST_IMAGES, "test_predictions")
        elif choice == "5":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()