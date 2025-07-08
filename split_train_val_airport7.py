#!/usr/bin/env python3
"""
Split airport7 dataset: move 20% of images from train/ to val/ for each class.
"""
import os
import random
from pathlib import Path

random.seed(42)

TRAIN_DIR = Path("data/airport7/train")
VAL_DIR = Path("data/airport7/val")
VAL_SPLIT = 0.2

classes = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()]

print(
    "Splitting airport7 dataset: moving 20% of images from train/ to val/ for each class."
)
print("=" * 60)

for class_name in classes:
    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name
    val_class_dir.mkdir(parents=True, exist_ok=True)

    images = [f for f in train_class_dir.iterdir() if f.suffix.lower() == ".jpg"]
    n_val = max(1, int(len(images) * VAL_SPLIT))
    val_images = random.sample(images, n_val)

    for img_path in val_images:
        img_path.rename(val_class_dir / img_path.name)

    print(f"{class_name}: moved {n_val} images to validation set.")

print("\n✅ Split complete. Validation images are now in data/airport7/val/")
