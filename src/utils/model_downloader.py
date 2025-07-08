"""
Model Downloader Utility
Downloads YOLOv8 weights and (optionally) classification model weights
"""

import os
import requests

def download_yolov8_weights(model_name='yolov8x.pt', dest_dir='models'):
    """Download YOLOv8 weights if not present"""
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, model_name)
    if os.path.exists(dest_path):
        print(f"YOLOv8 weights already exist: {dest_path}")
        return dest_path
    url = f'https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}'
    print(f"Downloading {model_name} from {url} ...")
    r = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded YOLOv8 weights to {dest_path}")
    return dest_path

def download_classification_weights(model_name='bird_classifier.pth', dest_dir='models'):
    """Placeholder for downloading classification model weights"""
    print(f"[INFO] Please manually add your classification model weights as {os.path.join(dest_dir, model_name)}")
    return os.path.join(dest_dir, model_name)

if __name__ == '__main__':
    download_yolov8_weights()
    download_classification_weights() 