import os
import shutil

from icrawler.builtin import BingImageCrawler
from sklearn.model_selection import train_test_split

AIRPORT7_CLASSES = [
    "black_kite",
    "brahminy_kite",
    "cormorant",
    "stork",
    "egret",
    "pigeon",
    "crow",
]

IMAGES_PER_CLASS = 100
TRAIN_RATIO = 0.8  #80% train, 20% val

BASE_DIR = "data/airport7"
RAW_DIR = os.path.join(BASE_DIR, "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


def download_images_for_class(class_name, num_images=IMAGES_PER_CLASS):
    save_dir = os.path.join(RAW_DIR, class_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading images for {class_name}...")
    crawler = BingImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=class_name + " bird", max_num=num_images)
    print(f"Downloaded images for {class_name} to {save_dir}")


def split_and_organize():
    for class_name in AIRPORT7_CLASSES:
        raw_class_dir = os.path.join(RAW_DIR, class_name)
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        images = [
            f
            for f in os.listdir(raw_class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        train_imgs, val_imgs = train_test_split(
            images, train_size=TRAIN_RATIO, random_state=42
        )
        for img in train_imgs:
            shutil.copy(
                os.path.join(raw_class_dir, img), os.path.join(train_class_dir, img)
            )
        for img in val_imgs:
            shutil.copy(
                os.path.join(raw_class_dir, img), os.path.join(val_class_dir, img)
            )
        print(
            f"Organized {len(train_imgs)} train and {len(val_imgs)} val images for {class_name}"
        )


def main():
    for class_name in AIRPORT7_CLASSES:
        download_images_for_class(class_name)
    split_and_organize()
    print(
        "\n✅ Dataset expansion complete! You can now retrain your classifier with more data."
    )


if __name__ == "__main__":
    main()
