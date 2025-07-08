"""
Bird Detection Module using YOLOv8
Handles real-time bird detection in video streams
Optimized for airport bird detection and classification
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class BirdDetection:
    """Data class for storing bird detection results"""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    species: str
    size_category: str
    frame_id: int
    timestamp: datetime
    center_point: Tuple[int, int]
    area: float


class BirdDetector:
    """
    Main bird detection class using YOLOv8
    Handles detection, tracking, and result processing
    Optimized for airport bird detection
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the bird detector

        Args:
            model_path: Path to custom YOLOv8 model (optional)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.airport_bird_classes = self._load_airport_bird_classes()

        # Bird class names for detection (size-based categories)
        self.bird_classes = {
            0: "small_bird",  #Sparrows, finches, small songbirds
            1: "medium_bird",  #Pigeons, crows, medium raptors
            2: "large_bird",  #Eagles, vultures, large waterfowl
            3: "raptor",  #Hawks, kites, eagles
            4: "waterfowl",  #Ducks, geese, herons, cormorants, storks, egrets
        }

        self._load_model(model_path)

    def _load_airport_bird_classes(self) -> Dict:
        """Load airport-specific bird class mappings"""
        classes_path = Path("data/classes/airport_birds.json")
        if classes_path.exists():
            with open(classes_path, "r") as f:
                return json.load(f)
        else:
            return {
                "kites": {
                    "black_kite": "Milvus migrans",
                    "brahminy_kite": "Haliastur indus",
                    "red_kite": "Milvus milvus",
                },
                "raptors": {
                    "eagle": "Aquila spp",
                    "hawk": "Accipiter spp",
                    "falcon": "Falco spp",
                    "vulture": "Gyps spp",
                },
                "waterfowl": {
                    "cormorant": "Phalacrocorax spp",
                    "stork": "Ciconia spp",
                    "egret": "Egretta spp",
                    "heron": "Ardea spp",
                    "duck": "Anas spp",
                    "goose": "Branta spp",
                },
                "small_birds": {
                    "sparrow": "Passer spp",
                    "finch": "Fringilla spp",
                    "starling": "Sturnus vulgaris",
                    "pigeon": "Columba spp",
                },
            }

    def _load_model(self, model_path: str = None):
        """Load YOLOv8 model for bird detection"""
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Loaded custom model from {model_path}")
            else:
                #Use pre-trained YOLOv8 model optimized for bird detection
                model_path = "models/yolov8x.pt"
                if not os.path.exists(model_path):
                    print("Downloading YOLOv8x model...")
                    self.model = YOLO("yolov8x.pt")
                    os.makedirs("models", exist_ok=True)
                    torch.save(self.model.state_dict(), model_path)
                else:
                    self.model = YOLO(model_path)
                print(f"Loaded YOLOv8x model from {model_path}")

            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _classify_bird_size(
        self, area: float, image_height: int, image_width: int
    ) -> str:
        """Classify bird size based on detection area"""
        image_area = image_height * image_width
        relative_area = area / image_area

        if relative_area < 0.01:  #Less than 1% of image
            return "small_bird"
        elif relative_area < 0.05:  #1-5% of image
            return "medium_bird"
        else:  #More than 5% of image
            return "large_bird"

    def _estimate_species(
        self, class_id: int, area: float, bbox: Tuple[int, int, int, int]
    ) -> str:
        """Estimate bird species based on detection characteristics"""

        size_category = self._classify_bird_size(
            area, 640, 640
        )  

        #Map size categories to likely species
        species_mapping = {
            "small_bird": "sparrow",
            "medium_bird": "pigeon",
            "large_bird": "eagle",
            "raptor": "kite",
            "waterfowl": "cormorant",
        }

        return species_mapping.get(
            self.bird_classes.get(class_id, "unknown"), "unknown"
        )

    def detect_birds_in_frame(
        self, frame: np.ndarray, frame_id: int = 0
    ) -> List[BirdDetection]:
        """
        Detect birds in a single frame

        Args:
            frame: Input frame as numpy array
            frame_id: Frame identifier

        Returns:
            List of BirdDetection objects
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        results = self.model(frame, verbose=False)

        detections = []
        timestamp = datetime.now()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    #Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    #Filter by confidence threshold
                    if confidence < self.confidence_threshold:
                        continue

                    #Filter for bird-like objects and flying objects
                    #COCO classes: bird=14, airplane=4, kite=33, person=0(for testing)
                    bird_related_classes = [
                        14,
                        4,
                        33,
                        0,
                    ]  #bird, airplane, kite, person

                    if class_id in bird_related_classes or class_id < len(
                        self.bird_classes
                    ):
                        #Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        #Calculate center point and area
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)

                        #Map COCO classes to our bird classes
                        if class_id in bird_related_classes:
                            if class_id == 14:  #bird
                                mapped_class_id = 0  #small_bird
                            elif class_id == 4:  #airplane (could be large bird)
                                mapped_class_id = 2  #large_bird
                            elif class_id == 33:  #kite
                                mapped_class_id = 3  #raptor
                            elif class_id == 0:  #person (for testing)
                                mapped_class_id = 1  #medium_bird
                            else:
                                mapped_class_id = 1  #medium_bird
                        else:
                            mapped_class_id = class_id

                        class_name = self.bird_classes.get(
                            mapped_class_id, "unknown_bird"
                        )
                        species = self._estimate_species(
                            mapped_class_id, area, (x1, y1, x2, y2)
                        )
                        size_category = self._classify_bird_size(
                            area, frame.shape[0], frame.shape[1]
                        )

                        detection = BirdDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(confidence),
                            class_id=mapped_class_id,
                            class_name=class_name,
                            species=species,
                            size_category=size_category,
                            frame_id=frame_id,
                            timestamp=timestamp,
                            center_point=(center_x, center_y),
                            area=area,
                        )

                        detections.append(detection)

        return detections

    def detect_birds_in_video(
        self, video_path: str, max_frames: int = None
    ) -> List[List[BirdDetection]]:
        """
        Detect birds in a video file

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None for all)

        Returns:
            List of frame detections
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        all_detections = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            detections = self.detect_birds_in_frame(frame, frame_count)
            all_detections.append(detections)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        return all_detections

    def detect_birds_in_stream(self, stream_source: int = 0):
        """
        Detect birds in real-time video stream

        Args:
            stream_source: Camera source (0 for default webcam)

        Yields:
            Tuple of (frame, detections) for each frame
        """
        cap = cv2.VideoCapture(stream_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {stream_source}")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_birds_in_frame(frame, frame_count)
            yield frame, detections
            frame_count += 1

        cap.release()

    def draw_detections(
        self, frame: np.ndarray, detections: List[BirdDetection]
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on frame

        Args:
            frame: Input frame
            detections: List of bird detections

        Returns:
            Frame with detections drawn
        """
        frame_copy = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            colors = {
                "small_bird": (0, 255, 0),  #Green
                "medium_bird": (255, 255, 0),  #Yellow
                "large_bird": (255, 0, 0),  #Red
                "raptor": (0, 0, 255),  #Blue
                "waterfowl": (255, 0, 255),  #Magenta
            }

            color = colors.get(detection.class_name, (255, 255, 255))

            #Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            #Draw label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            #Draw center point
            cv2.circle(frame_copy, detection.center_point, 3, color, -1)

        return frame_copy

    def get_detection_statistics(self, detections: List[List[BirdDetection]]) -> Dict:
        """
        Calculate statistics from detections

        Args:
            detections: List of frame detections

        Returns:
            Dictionary with statistics
        """
        total_detections = sum(len(frame_dets) for frame_dets in detections)
        total_frames = len(detections)

        class_counts = {}
        confidence_scores = []

        for frame_dets in detections:
            for detection in frame_dets:
                class_counts[detection.class_name] = (
                    class_counts.get(detection.class_name, 0) + 1
                )
                confidence_scores.append(detection.confidence)

        return {
            "total_detections": total_detections,
            "total_frames": total_frames,
            "detections_per_frame": (
                total_detections / total_frames if total_frames > 0 else 0
            ),
            "class_distribution": class_counts,
            "average_confidence": (
                np.mean(confidence_scores) if confidence_scores else 0
            ),
            "max_confidence": np.max(confidence_scores) if confidence_scores else 0,
            "min_confidence": np.min(confidence_scores) if confidence_scores else 0,
        }
