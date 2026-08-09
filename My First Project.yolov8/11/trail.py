from ultralytics import YOLO
from pathlib import Path
import cv2
import csv

# Paths
TEST_IMAGES = r"..\..\My First Project.yolov8\test\images"
MODEL_PATH = r"..\..\cone_runs\cone_detector\weights\best.pt"
OUTPUT_DIR = r"..\..\test_predictions_no_labels"

# Load model
model = YOLO(MODEL_PATH)
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# Run inference (NO labels needed)
print("🔍 Running inference without labels...")
print("   Model will draw bounding boxes independently\n")

detections = []
results = model.predict(
    source=TEST_IMAGES,
    conf=0.5,          # Confidence threshold
    iou=0.45,
    save=False,
    stream=True,
    device=0           # Use GPU if available
)

for result in results:
    img = result.orig_img.copy()
    img_name = Path(result.path).name
    
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        print(f"📷 {img_name}: Found {len(boxes)} cone(s)")
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            class_name = result.names[cls_id]
            
            # Save to CSV
            detections.append([img_name, class_name, round(conf, 4), x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, max(y1-10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print(f"📷 {img_name}: No cones detected")
    
    # Save annotated image
    output_path = output_dir / img_name
    cv2.imwrite(str(output_path), img)

# Save CSV
csv_path = output_dir / 'detections.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    writer.writerows(detections)

print(f"\n✅ Done!")
print(f"📁 Annotated images saved to: {output_dir}")
print(f"📊 Total detections: {len(detections)}")
print(f"📄 CSV saved to: {csv_path}")