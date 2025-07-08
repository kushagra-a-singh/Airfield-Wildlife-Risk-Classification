"""
Video Processor Utility Module
Enhanced video processing and visualization for bird detection system
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.classification.species_classifier import BirdClassification
from src.detection.enhanced_bird_detector import BirdDetection
from src.risk_assessment.risk_calculator import BirdRiskAssessment


@dataclass
class VisualizationConfig:
    """Configuration for video visualization"""

    show_bbox: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_risk: bool = True
    show_trajectory: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 1
    save_frames: bool = False
    save_bird_regions: bool = False
    output_dir: str = "output"


class VideoProcessor:
    """Enhanced video processing and visualization utilities"""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()

        if self.config.save_frames or self.config.save_bird_regions:
            os.makedirs(self.config.output_dir, exist_ok=True)
            os.makedirs(f"{self.config.output_dir}/frames", exist_ok=True)
            os.makedirs(f"{self.config.output_dir}/bird_regions", exist_ok=True)

        self.risk_colors = {
            "low": (0, 255, 0),  #Green
            "moderate": (0, 255, 255),  #Yellow
            "high": (0, 0, 255),  #Red
        }

        self.species_colors = {
            "black_kite": (255, 0, 0),  #Blue
            "brahminy_kite": (255, 0, 255),  # agenta
            "cormorant": (0, 255, 255),  #Cyan
            "stork": (255, 255, 0),  #Yellow
            "egret": (128, 128, 128),  #Gray
        }

        #Trajectory tracking
        self.trajectory_history = {}
        self.max_trajectory_length = 20

    def draw_enhanced_detections(
        self,
        frame: np.ndarray,
        detections: List[BirdDetection],
        classifications: List[BirdClassification],
        risks: List[BirdRiskAssessment],
        frame_id: int = 0,
    ) -> np.ndarray:
        """Draw enhanced detections with comprehensive information overlay"""
        frame_copy = frame.copy()

        for i, (detection, classification, risk) in enumerate(
            zip(detections, classifications, risks)
        ):
            #Get bounding box coordinates
            x1, y1, x2, y2 = detection.bbox
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            #Create unique detection ID using frame_id and detection index
            detection_id = f"{frame_id}_{i}"

            #Update trajectory
            self._update_trajectory(detection_id, bbox_center)

            #Draw bounding box
            if self.config.show_bbox:
                color = self._get_detection_color(risk, classification)
                cv2.rectangle(
                    frame_copy, (x1, y1), (x2, y2), color, self.config.bbox_thickness
                )

            #Draw labels and information
            if self.config.show_labels:
                self._draw_detection_labels(
                    frame_copy, detection, classification, risk, bbox_center
                )

            #Draw trajectory
            if self.config.show_trajectory:
                self._draw_trajectory(frame_copy, detection_id)

            #Save individual bird region if enabled
            if self.config.save_bird_regions:
                self._save_bird_region(
                    frame_copy, detection, classification, frame_id, i
                )

        #Save processed frame if enabled
        if self.config.save_frames:
            self._save_processed_frame(frame_copy, frame_id)

        return frame_copy

    def _save_processed_frame(self, frame: np.ndarray, frame_id: int):
        """Save processed frame with detections"""
        try:
            frame_path = f"{self.config.output_dir}/frames/frame_{frame_id:04d}.jpg"
            cv2.imwrite(frame_path, frame)
        except Exception as e:
            print(f"Error saving frame {frame_id}: {e}")

    def _save_bird_region(
        self,
        frame: np.ndarray,
        detection: BirdDetection,
        classification: BirdClassification,
        frame_id: int,
        detection_id: int,
    ):
        """Save individual bird region image"""
        try:
            x1, y1, x2, y2 = detection.bbox

            #Extract bird region with padding
            padding = 20
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(frame.shape[1], x2 + padding)
            y2_pad = min(frame.shape[0], y2 + padding)

            bird_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]

            if bird_region.size > 0:
                #Create filename with species and confidence
                species = classification.species.replace("_", "")
                confidence = detection.confidence
                region_path = f"{self.config.output_dir}/bird_regions/frame_{frame_id:04d}_det_{detection_id}_{species}_{confidence:.2f}.jpg"
                cv2.imwrite(region_path, bird_region)

        except Exception as e:
            print(f"Error saving bird region: {e}")

    def _get_detection_color(
        self, risk: BirdRiskAssessment, classification: BirdClassification
    ) -> Tuple[int, int, int]:
        """Get color for detection based on risk and species"""
        #Primary color based on risk level
        risk_color = self.risk_colors.get(risk.risk_level, (128, 128, 128))

        #Blend with species color if available
        species_color = self.species_colors.get(classification.species, (128, 128, 128))

        #Blend colors (70% risk, 30% species)
        blended_color = tuple(
            int(0.7 * rc + 0.3 * sc) for rc, sc in zip(risk_color, species_color)
        )

        return blended_color

    def _draw_detection_labels(
        self,
        frame: np.ndarray,
        detection: BirdDetection,
        classification: BirdClassification,
        risk: BirdRiskAssessment,
        bbox_center: Tuple[int, int],
    ):
        """Draw comprehensive labels for each detection"""
        x1, y1, x2, y2 = detection.bbox
        label_y = y1 - 10 if y1 > 30 else y2 + 20

        labels = []

        if self.config.show_confidence:
            labels.append(f"Conf: {detection.confidence:.2f}")

        labels.append(f"Species: {classification.species}")
        labels.append(f"Size: {classification.size_category}")

        if self.config.show_risk:
            labels.append(f"Risk: {risk.risk_level} ({risk.risk_score:.1f})")

        labels.append(f"Behavior: {classification.behavior}")
        labels.append(f"Altitude: {classification.altitude_estimate:.0f}m")

        #Draw labels with background
        for i, label in enumerate(labels):
            label_pos = (x1, label_y + i * 20)

            #Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.font_thickness,
            )

            #Draw background rectangle
            bg_rect = (
                label_pos[0] - 2,
                label_pos[1] - text_height - 2,
                label_pos[0] + text_width + 2,
                label_pos[1] + baseline + 2,
            )
            cv2.rectangle(
                frame, (bg_rect[0], bg_rect[1]), (bg_rect[2], bg_rect[3]), (0, 0, 0), -1
            )

            #Draw text
            cv2.putText(
                frame,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                (255, 255, 255),
                self.config.font_thickness,
            )

    def _update_trajectory(self, detection_id: str, center: Tuple[int, int]):
        """Update trajectory history for a detection"""
        if detection_id not in self.trajectory_history:
            self.trajectory_history[detection_id] = []

        self.trajectory_history[detection_id].append(center)

        #Keep only recent trajectory points
        if len(self.trajectory_history[detection_id]) > self.max_trajectory_length:
            self.trajectory_history[detection_id] = self.trajectory_history[
                detection_id
            ][-self.max_trajectory_length :]

    def _draw_trajectory(self, frame: np.ndarray, detection_id: str):
        """Draw trajectory for a detection"""
        if (
            detection_id not in self.trajectory_history
            or len(self.trajectory_history[detection_id]) < 2
        ):
            return

        trajectory = self.trajectory_history[detection_id]

        #Draw trajectory lines with fading effect
        for i in range(1, len(trajectory)):
            #Calculate alpha based on recency
            alpha = i / len(trajectory)
            color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))

            cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)

        #Draw current position
        cv2.circle(frame, trajectory[-1], 3, (255, 255, 255), -1)

    def add_frame_info_overlay(
        self,
        frame: np.ndarray,
        frame_id: int,
        detections_count: int,
        risks_count: int,
        avg_risk: float,
    ) -> np.ndarray:
        """Add frame information overlay"""
        #Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        #Add text information
        info_lines = [
            f"Frame: {frame_id}",
            f"Detections: {detections_count}",
            f"Risks: {risks_count}",
            f"Avg Risk: {avg_risk:.2f}",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return frame

    def create_risk_heatmap(
        self, frame: np.ndarray, risks: List[BirdRiskAssessment]
    ) -> np.ndarray:
        """Create risk heatmap overlay"""
        heatmap = np.zeros_like(frame)

        for risk in risks:
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            radius = 50

            #Create gradient circle
            y, x = np.ogrid[: frame.shape[0], : frame.shape[1]]
            mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2

            #Color based on risk level
            if risk.risk_level == "high":
                color = (0, 0, 255)  #Red
            elif risk.risk_level == "moderate":
                color = (0, 255, 255)  #Yellow
            else:
                color = (0, 255, 0)  #Green

            heatmap[mask] = color

        #Blend with original frame
        blended = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        return blended

    def extract_bird_region(
        self, frame: np.ndarray, detection: BirdDetection, padding: int = 20
    ) -> np.ndarray:
        """Extract bird region from frame with padding"""
        x1, y1, x2, y2 = detection.bbox

        #Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        return frame[y1:y2, x1:x2]

    def resize_frame_for_processing(
        self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """Resize frame for consistent processing"""
        return cv2.resize(frame, target_size)

    def apply_image_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """Apply image enhancement for better detection"""
        #Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        #Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        #Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def create_detection_summary_frame(
        self,
        detections: List[BirdDetection],
        classifications: List[BirdClassification],
        risks: List[BirdRiskAssessment],
        frame_size: Tuple[int, int] = (800, 600),
    ) -> np.ndarray:
        """Create a summary frame showing all detections"""
        summary_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        cv2.putText(
            summary_frame,
            "Bird Detection Summary",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        #Add detection information
        y_offset = 80
        for i, (detection, classification, risk) in enumerate(
            zip(detections, classifications, risks)
        ):
            info_text = f"{i+1}. {classification.species} - Risk: {risk.risk_level} ({risk.risk_score:.1f})"
            cv2.putText(
                summary_frame,
                info_text,
                (20, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self._get_detection_color(risk, classification),
                2,
            )

        return summary_frame
