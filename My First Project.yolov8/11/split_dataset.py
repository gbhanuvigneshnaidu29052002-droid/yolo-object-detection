import os
import random
import shutil
from pathlib import Path

# ----------------------
# CONFIG
# ----------------------
ROOT = Path(r"C:\Users\gbhan\Desktop\YOLO DETCTION\My First Project.yolov8")
TRAIN_DIR = ROOT / "train"
IMAGES_DIR = TRAIN_DIR / "images"
LABELS_DIR = TRAIN_DIR / "labels"

VAL_IMAGES_DIR  = ROOT / "valid" / "images"
VAL_LABELS_DIR  = ROOT / "valid" / "labels"
TEST_IMAGES_DIR = ROOT / "test" / "images"
TEST_LABELS_DIR = ROOT / "test" / "labels"

TRAIN_RATIO = 0.75   # 75% train
VAL_RATIO   = 0.15   # 15% val
# remaining 10% -> test
SEED = 42

random.seed(SEED)

VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TEST_LABELS_DIR.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in image_exts]

paired_files = []
missing_labels = []

for img_path in image_files:
    label_path = LABELS_DIR / f"{img_path.stem}.txt"
    if label_path.exists():
        paired_files.append((img_path, label_path))
    else:
        missing_labels.append(img_path.name)

if not paired_files:
    raise ValueError("No image-label pairs found. Check train/images and train/labels paths.")

random.shuffle(paired_files)

n = len(paired_files)
train_end = int(n * TRAIN_RATIO)
val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

train_pairs = paired_files[:train_end]
val_pairs   = paired_files[train_end:val_end]
test_pairs  = paired_files[val_end:]

def move_pairs(pairs, img_dst_dir, lbl_dst_dir):
    for img_path, label_path in pairs:
        shutil.move(str(img_path), str(img_dst_dir / img_path.name))
        shutil.move(str(label_path), str(lbl_dst_dir / label_path.name))

# move val and test pairs; train pairs stay where they are
move_pairs(val_pairs, VAL_IMAGES_DIR, VAL_LABELS_DIR)
move_pairs(test_pairs, TEST_IMAGES_DIR, TEST_LABELS_DIR)

data_yaml = ROOT / "data.yaml"
if not data_yaml.exists():
    data_yaml = ROOT / "data.yml"

yaml_text = f"""train: train/images
val: valid/images
test: test/images

nc: 3
names: ['blue_cone', 'green_cone', 'orange_cone']
"""

data_yaml.write_text(yaml_text, encoding="utf-8")

print(f"Total paired images: {n}")
print(f"Train images: {len(train_pairs)}")
print(f"Val images:   {len(val_pairs)}")
print(f"Test images:  {len(test_pairs)}")
if missing_labels:
    print("Images skipped because label missing:")
    for name in missing_labels:
        print(" -", name)
print(f"Updated YAML: {data_yaml}")