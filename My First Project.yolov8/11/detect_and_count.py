# detect_and_count.py
import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

# ==================== CONFIGURATION ====================
MODEL_PATH = r"runs/detect/train/weights/best.pt"   # update if different
IMAGE_PATH = r"path/to/your/mix_image.jpg"          # single image
# Or process all images in a folder:
FOLDER_PATH = r"..\..\My First Project.yolov8\mix_images"  # create this folder and put mix images there
CONF_THRESHOLD = 0.25                               # minimum confidence to keep a detection
# =======================================================

# Class names (must match order in data.yaml)
CLASS_NAMES = ['blue cone', 'green cone', 'orange cone']

# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
model = YOLO(MODEL_PATH)

def process_image(img_path):
    """Run inference on a single image, display bounding boxes, and return counts."""
    results = model(img_path, conf=CONF_THRESHOLD)[0]   # first image result
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None

    # Count objects per class
    counts = defaultdict(int)
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            counts[cls_id] += 1
            # Draw bounding box and label
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print counts
    print(f"\n--- Results for {os.path.basename(img_path)} ---")
    total = 0
    for cls_id, name in enumerate(CLASS_NAMES):
        c = counts[cls_id]
        total += c
        print(f"{name}: {c}")
    print(f"Total objects: {total}")

    # Show the image (press any key to close)
    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return counts

def process_folder(folder_path):
    """Process all images in a folder and print summary."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    if not image_files:
        print(f"No images found in {folder_path}")
        return

    all_counts = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        counts = process_image(img_path)
        if counts:
            all_counts.append(counts)

    # Optional: print overall statistics
    print("\n===== OVERALL STATISTICS =====")
    total_by_class = defaultdict(int)
    for counts in all_counts:
        for cls_id, c in counts.items():
            total_by_class[cls_id] += c
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"Total {name}: {total_by_class[cls_id]}")
    print(f"Total images processed: {len(all_counts)}")

if __name__ == "__main__":
    if os.path.isdir(FOLDER_PATH):
        process_folder(FOLDER_PATH)
    elif os.path.isfile(IMAGE_PATH):
        process_image(IMAGE_PATH)
    else:
        print(f"Neither file nor folder found: {IMAGE_PATH} / {FOLDER_PATH}")