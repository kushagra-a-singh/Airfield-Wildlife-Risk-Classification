"""
Enhanced Bird Detector Module
Multi-model ensemble detection system for bird detection in video streams
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not available, using fallback detection")


@dataclass
class BirdDetection:
    """Data class for bird detection results"""

    id: int
    bbox: Tuple[int, int, int, int]  #x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    timestamp: float


class EnhancedBirdDetector:
    """Enhanced bird detection using multiple models and ensemble techniques"""

    def __init__(self, use_ensemble: bool = True, confidence_threshold: float = 0.5):
        """Initialize the enhanced bird detector"""
        self.use_ensemble = use_ensemble
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #Performance optimization settings
        self.frame_skip = 2  #Process every Nth frame for speed
        self.last_processed_frame = -1
        self.cached_detections = {}  #Cache detections for skipped frames
        self.enable_preprocessing = True  #Can be disabled for speed
        self.enable_enhancement = False  #Disabled for speed by default
        self.target_fps = 15  #Target FPS for optimization

        self.models = {}
        self._load_models()

        #Initialize tracking and statistics
        self.detection_history = {}
        self.processing_times = []
        self.detection_counts = []
        self.next_detection_id = 0

        # Bird detection parameters
        self.bird_classes = {0: "bird"}  #COCO dataset bird class
        self.min_bird_size = 20
        self.max_bird_size = 500
        self.aspect_ratio_range = (0.5, 3.0)  #Expected bird aspect ratios

    def _load_models(self):
        """Load YOLOv8x model for best accuracy detection"""
        try:
            model_path = "models/yolov8x.pt"

            if Path(model_path).exists():
                model = YOLO(model_path)
                model.to(self.device)
                self.models["yolov8x"] = model
                print(f"✅ Loaded YOLOv8x model: {model_path}")
            else:
                print(f"❌ YOLOv8x model not found at: {model_path}")
                self._load_smaller_model()

        except Exception as e:
            print(f"Error loading YOLOv8x model: {e}")
            self._load_smaller_model()

    def _load_smaller_model(self):
        """Load a smaller YOLO model for better speed"""
        try:
            model = YOLO("yolov8n.pt") 
            model.to(self.device)
            self.models["yolov8n"] = model
            print(f"✅ Loaded YOLOv8n model for speed optimization")
        except Exception as e:
            print(f"Error loading smaller model: {e}")
            self._load_fallback_detector()

    def _load_fallback_detector(self):
        """Load fallback detection method using OpenCV"""
        cascade_path = "data/models/bird_cascade.xml"
        try:
            self.models["cascade"] = cv2.CascadeClassifier(cascade_path)
        except:
            print("Warning: Cascade classifier not available")

    def detect_birds_in_frame(
        self, frame: np.ndarray, frame_id: int
    ) -> List[BirdDetection]:
        """Detect birds in a single frame using YOLO (bird class only)"""
        start_time = time.time()

        #Frame skipping for performance
        if frame_id % self.frame_skip != 0:
            cached_key = frame_id - (frame_id % self.frame_skip)
            if cached_key in self.cached_detections:
                return self.cached_detections[cached_key]
            else:
                return []

        #Preprocessing with scale tracking
        processed_frame, scale, (orig_w, orig_h) = (
            self._preprocess_frame_fast_with_scale(frame)
            if self.enable_preprocessing
            else (frame, 1.0, frame.shape[1::-1])
        )

        #Run YOLO detection (bird class only)
        detections = []
        if "yolov8x" in self.models:
            yolo_model = self.models["yolov8x"]
            results = yolo_model(processed_frame)
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    #Only keep 'bird' class (COCO class 14 or custom class 0)
                    if hasattr(yolo_model, "names"):
                        class_name = yolo_model.names[class_id]
                        if class_name.lower() != "bird":
                            continue
                    else:
                        class_name = "bird"
                    #Map box coordinates back to original image size
                    x1, y1, x2, y2 = [
                        int(coord.item() / scale) for coord in box.xyxy[0]
                    ]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(orig_w, x2)
                    y2 = min(orig_h, y2)
                    if conf >= self.confidence_threshold:
                        detection = BirdDetection(
                            id=self.next_detection_id,
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=class_id,
                            class_name=class_name,
                            frame_id=frame_id,
                            timestamp=time.time(),
                        )
                        detections.append(detection)
                        self.next_detection_id += 1
        else:
            pass

        filtered_detections = self._filter_detections_fast(
            detections, (orig_h, orig_w, 3)
        )

        #Update tracking (reduced frequency)
        if frame_id % 5 == 0:
            self._update_detection_tracking(filtered_detections, frame_id)

        #Cache results for frame skipping
        self.cached_detections[frame_id] = filtered_detections
        if len(self.cached_detections) > 10:
            old_keys = [k for k in self.cached_detections.keys() if k < frame_id - 20]
            for k in old_keys:
                del self.cached_detections[k]

        #Update performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.detection_counts.append(len(filtered_detections))

        return filtered_detections

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection"""
        if frame.shape[0] > 640 or frame.shape[1] > 640:
            scale = min(640 / frame.shape[1], 640 / frame.shape[0])
            new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            frame = cv2.resize(frame, new_size)

        frame = self._enhance_image(frame)

        return frame

    def _enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """Apply image enhancement for better bird detection"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        #Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        #Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        #Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return enhanced

    def _preprocess_frame_fast(self, frame: np.ndarray) -> np.ndarray:
        """Fast preprocessing for better performance"""
        #Smaller resize for faster processing
        target_size = 416 if self.enable_preprocessing else 640  #Smaller for speed
        if frame.shape[0] > target_size or frame.shape[1] > target_size:
            scale = min(target_size / frame.shape[1], target_size / frame.shape[0])
            new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        #Optional enhancement (disabled by default for speed)
        if self.enable_enhancement:
            frame = self._enhance_image(frame)

        return frame

    def _preprocess_frame_fast_with_scale(self, frame: np.ndarray):
        """Fast preprocessing for better performance, returns scale and original size"""
        target_size = 416 if self.enable_preprocessing else 640  #Smaller for speed
        h, w = frame.shape[:2]
        if h > target_size or w > target_size:
            scale = min(target_size / w, target_size / h)
            new_size = (int(w * scale), int(h * scale))
            resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
            return resized, scale, (w, h)
        return frame, 1.0, (w, h)

    def _ensemble_detect(self, frame: np.ndarray, frame_id: int) -> List[BirdDetection]:
        """Run detection using YOLO model with optimized settings"""
        all_detections = []

        for model_name, model in self.models.items():
            try:
                if model_name in ["yolov8x", "yolov8n"]:
                    results = model(
                        frame,
                        conf=self.confidence_threshold,
                        verbose=False,
                        iou=0.5,  
                        max_det=10,  
                        agnostic_nms=True, 
                        half=(
                            True if self.device == "cuda" else False
                        ),  
                    )

                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                class_id = int(box.cls[0].cpu().numpy())

                                #Check if it's a bird detection
                                if self._is_bird_detection(class_id, confidence):
                                    detection = BirdDetection(
                                        id=self.next_detection_id,
                                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                                        confidence=float(confidence),
                                        class_id=class_id,
                                        class_name=self.bird_classes.get(
                                            class_id, "bird"
                                        ),
                                        frame_id=frame_id,
                                        timestamp=time.time(),
                                    )
                                    all_detections.append(detection)
                                    self.next_detection_id += 1

                elif model_name == "cascade":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    bird_rects = model.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    for x, y, w, h in bird_rects:
                        detection = BirdDetection(
                            id=self.next_detection_id,
                            bbox=(x, y, x + w, y + h),
                            confidence=0.7,  #Default confidence for cascade
                            class_id=0,
                            class_name="bird",
                            frame_id=frame_id,
                            timestamp=time.time(),
                        )
                        all_detections.append(detection)
                        self.next_detection_id += 1

            except Exception as e:
                print(f"Error in {model_name} detection: {e}")
                continue

        return all_detections

    def _single_model_detect(
        self, frame: np.ndarray, frame_id: int
    ) -> List[BirdDetection]:
        """Run detection using a single model"""
        if not self.models:
            return []

        model_name, model = list(self.models.items())[0]
        return self._ensemble_detect(frame, frame_id)

    def _is_bird_detection(self, class_id: int, confidence: float) -> bool:
        """Check if detection is likely a bird"""
        if class_id in self.bird_classes:
            return True

        return confidence > self.confidence_threshold

    def _filter_detections(
        self, detections: List[BirdDetection], frame_shape: Tuple[int, int, int]
    ) -> List[BirdDetection]:
        """Filter and validate detections"""
        filtered = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            width = x2 - x1
            height = y2 - y1

            if width < self.min_bird_size or height < self.min_bird_size:
                continue

            if width > self.max_bird_size or height > self.max_bird_size:
                continue

            aspect_ratio = width / height
            if not (
                self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]
            ):
                continue

            #Check if detection is within frame bounds
            if x1 < 0 or y1 < 0 or x2 > frame_shape[1] or y2 > frame_shape[0]:
                continue

            #Check confidence threshold
            if detection.confidence < self.confidence_threshold:
                continue

            filtered.append(detection)

        return filtered

    def _filter_detections_fast(
        self, detections: List[BirdDetection], frame_shape: Tuple[int, int, int]
    ) -> List[BirdDetection]:
        """Fast filtering of detections"""
        filtered = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            width = x2 - x1
            height = y2 - y1

            if width < self.min_bird_size or height < self.min_bird_size:
                continue

            if width > self.max_bird_size or height > self.max_bird_size:
                continue

            if detection.confidence < self.confidence_threshold:
                continue

            if x1 < 0 or y1 < 0 or x2 > frame_shape[1] or y2 > frame_shape[0]:
                continue

            filtered.append(detection)

        return filtered

    def _update_detection_tracking(
        self, detections: List[BirdDetection], frame_id: int
    ):
        """Update detection tracking for trajectory analysis"""
        for detection in detections:
            if detection.id not in self.detection_history:
                self.detection_history[detection.id] = []

            self.detection_history[detection.id].append(
                {
                    "frame_id": frame_id,
                    "bbox": detection.bbox,
                    "confidence": detection.confidence,
                    "timestamp": detection.timestamp,
                }
            )

            if len(self.detection_history[detection.id]) > 30:
                self.detection_history[detection.id] = self.detection_history[
                    detection.id
                ][-30:]

    def detect_birds_in_video(
        self, video_path: str, max_frames: Optional[int] = None
    ) -> List[List[BirdDetection]]:
        """Detect birds in video file"""
        cap = cv2.VideoCapture(video_path)
        all_detections = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_id >= max_frames:
                break

            detections = self.detect_birds_in_frame(frame, frame_id)
            all_detections.append(detections)

            frame_id += 1

        cap.release()
        return all_detections

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics with FPS optimization"""
        if not self.processing_times:
            return {}

        avg_time = np.mean(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        #Calculate FPS with frame skipping
        effective_fps = fps * self.frame_skip

        return {
            "total_frames_processed": len(self.processing_times),
            "average_processing_time": avg_time,
            "average_detections_per_frame": (
                np.mean(self.detection_counts) if self.detection_counts else 0
            ),
            "total_detections": (
                sum(self.detection_counts) if self.detection_counts else 0
            ),
            "detection_rate": (
                len([c for c in self.detection_counts if c > 0])
                / len(self.detection_counts)
                if self.detection_counts
                else 0
            ),
            "fps": fps,
            "effective_fps": effective_fps,
            "frame_skip": self.frame_skip,
            "target_fps": self.target_fps,
            "performance_ratio": (
                effective_fps / self.target_fps if self.target_fps > 0 else 0
            ),
        }

    def optimize_for_fps(self, target_fps: float = 15.0):
        """Dynamically optimize settings for target FPS"""
        self.target_fps = target_fps

        #Get current performance
        stats = self.get_detection_statistics()
        current_fps = stats.get("effective_fps", 0)

        if current_fps < target_fps * 0.8:  #Below 80% of target
            #Increase frame skip
            self.frame_skip = min(self.frame_skip + 1, 5)
            print(f"Optimizing: Increased frame skip to {self.frame_skip}")

        elif current_fps > target_fps * 1.2:  #Above 120% of target
            #Decrease frame skip for better quality
            self.frame_skip = max(self.frame_skip - 1, 1)
            print(f"Optimizing: Decreased frame skip to {self.frame_skip}")

        #Disable preprocessing if still too slow
        if current_fps < target_fps * 0.6:
            self.enable_preprocessing = False
            print("Optimizing: Disabled preprocessing for speed")

        #Disable enhancement if still too slow
        if current_fps < target_fps * 0.4:
            self.enable_enhancement = False
            print("Optimizing: Disabled enhancement for speed")

    def set_performance_mode(self, mode: str = "balanced"):
        """Set performance mode for different use cases"""
        if mode == "fast":
            self.frame_skip = 2  
            self.enable_preprocessing = False
            self.enable_enhancement = False
            self.confidence_threshold = 0.4  
            print("Performance mode: FAST (optimized speed)")

        elif mode == "balanced":
            self.frame_skip = 2
            self.enable_preprocessing = True
            self.enable_enhancement = False
            self.confidence_threshold = 0.5
            print("Performance mode: BALANCED (speed + accuracy)")

        elif mode == "accurate":
            self.frame_skip = 1
            self.enable_preprocessing = True
            self.enable_enhancement = True
            self.confidence_threshold = 0.4
            print("Performance mode: ACCURATE (max accuracy)")

        elif mode == "ultra_fast":
            self.frame_skip = 3
            self.enable_preprocessing = False
            self.enable_enhancement = False
            self.confidence_threshold = 0.3  
            print("Performance mode: ULTRA FAST (max speed, lower accuracy)")

    def get_trajectory_analysis(self) -> Dict[str, Any]:
        """Get trajectory analysis for detected birds"""
        trajectories = {}

        for detection_id, history in self.detection_history.items():
            if len(history) < 2:
                continue

            positions = [
                ((h["bbox"][0] + h["bbox"][2]) // 2, (h["bbox"][1] + h["bbox"][3]) // 2)
                for h in history
            ]

            if len(positions) >= 2:
                movement = np.sqrt(
                    (positions[-1][0] - positions[0][0]) ** 2
                    + (positions[-1][1] - positions[0][1]) ** 2
                )

                trajectories[detection_id] = {
                    "frames_tracked": len(history),
                    "total_movement": movement,
                    "average_confidence": np.mean([h["confidence"] for h in history]),
                    "trajectory_length": len(positions),
                }

        return trajectories

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[BirdDetection],
        species_labels: dict = None,
    ) -> np.ndarray:
        """Draw bounding boxes for all detected birds, with optional species labels."""
        out_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(out_frame.shape[1], x2)
            y2 = min(out_frame.shape[0], y2)
            label = "bird"
            if species_labels and det.id in species_labels:
                label = f"{species_labels[det.id]}"
            conf_text = f"{label} {det.confidence:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out_frame,
                conf_text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return out_frame

    def cleanup(self):
        """Cleanup resources"""
        self.models.clear()
        self.detection_history.clear()
        self.processing_times.clear()
        self.detection_counts.clear()
