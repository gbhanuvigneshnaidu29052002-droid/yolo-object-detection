from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import torch
import cv2
import os

def main():
    import shutil
    from datetime import datetime

    # =========================
    # CONFIG
    # =========================
    DATA_YAML = r"..\..\My First Project.yolov8\data.yaml"
    PROJECT_DIR = Path(r"..\..\cone_runs")
    MODEL_NAME = "yolov8n.pt"

    EPOCHS = 50
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.7

    TEST_SOURCE = r"..\..\My First Project.yolov8\valid\images"

    # =========================
    # DEVICE CHECK
    # =========================
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {'GPU' if device == 0 else 'CPU'}")

    # Generate timestamped run folder to separate different training sessions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"cone_detector_{timestamp}"

    # =========================
    # 1. TRAIN
    # =========================
    model = YOLO(MODEL_NAME)

    train_results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=True,
        patience=20,
        device=device,
        workers=0,          # << IMPORTANT FOR WINDOWS
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1
    )

    # =========================
    # 2. VALIDATE
    # =========================
    best_model_path = PROJECT_DIR / run_name / "weights" / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(f"best.pt not found at {best_model_path} — training may have failed.")

    best_model = YOLO(str(best_model_path))

    metrics = best_model.val(
        data=DATA_YAML,
        conf=CONF_THRESHOLD,
        device=device,
        workers=0          # << IMPORTANT FOR WINDOWS
    )

    precision = metrics.results_dict.get("metrics/precision(B)", 0)
    recall = metrics.results_dict.get("metrics/recall(B)", 0)
    map50 = metrics.results_dict.get("metrics/mAP50(B)", 0)
    map5095 = metrics.results_dict.get("metrics/mAP50-95(B)", 0)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n===== VALIDATION METRICS =====")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"mAP@0.50       : {map50:.4f}")
    print(f"mAP@0.50:0.95  : {map5095:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Approx. accuracy-style summary (use mAP50): {map50*100:.2f}%")

    # =========================
    # 3. PREDICT
    # =========================
    output_dir = PROJECT_DIR / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = best_model.predict(
        source=TEST_SOURCE,
        conf=CONF_THRESHOLD,
        save=False,
        stream=True,
        device=device,
        workers=0          # << IMPORTANT FOR WINDOWS
    )

    print("\n===== IMAGE-WISE DETECTIONS =====")

    for result in results:
        img = result.orig_img.copy()
        image_path = result.path
        image_name = os.path.basename(image_path)

        class_counter = Counter()
        boxes = result.boxes
        names = result.names

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                class_name = names[cls_id]
                class_counter[class_name] += 1

                label = f"{class_name} {conf*100:.1f}%"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(f"\nImage: {image_name}")
            print(f"Total objects detected: {sum(class_counter.values())}")
            for cls_name, count in class_counter.items():
                print(f"  {cls_name}: {count}")

        else:
            print(f"\nImage: {image_name}")
            print("No objects detected.")

        save_path = output_dir / image_name
        cv2.imwrite(str(save_path), img)

    # Copy best weights to the static directory so other scripts can load the latest run by default
    static_weights_dir = PROJECT_DIR / "cone_detector" / "weights"
    static_weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model_path, static_weights_dir / "best.pt")

    print(f"\nAnnotated prediction images saved in: {output_dir}")
    print(f"Best trained model saved in: {best_model_path}")
    print(f"Latest weights copied to default static folder: {static_weights_dir / 'best.pt'}")


if __name__ == "__main__":
    main()