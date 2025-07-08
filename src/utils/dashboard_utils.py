"""
Dashboard Utilities Module
Enhanced dashboard functionality and system overlay for bird detection system
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.enhanced_bird_detector import BirdDetection
from src.risk_assessment.risk_calculator import BirdRiskAssessment


@dataclass
class SystemOverlayConfig:
    """Configuration for system overlay"""

    show_frame_info: bool = True
    show_risk_summary: bool = True
    show_species_count: bool = True
    show_system_stats: bool = True
    overlay_opacity: float = 0.8
    font_scale: float = 0.6
    font_thickness: int = 1


class DashboardUtils:
    """Enhanced dashboard utilities and system overlay functionality"""

    def __init__(self, config: Optional[SystemOverlayConfig] = None):
        self.config = config or SystemOverlayConfig()

        #Color schemes for different information types
        self.info_colors = {
            "frame_info": (255, 255, 255),  #White
            "risk_high": (0, 0, 255),  #Red
            "risk_moderate": (0, 255, 255),  #Yellow
            "risk_low": (0, 255, 0),  #Green
            "system_info": (255, 255, 0),  #Cyan
            "species_info": (255, 0, 255),  #Magenta
        }

    def add_system_overlay(
        self,
        frame: np.ndarray,
        frame_id: int,
        detections: List[BirdDetection],
        risks: List[BirdRiskAssessment],
        system_stats: Dict[str, Any],
    ) -> np.ndarray:
        """Add comprehensive system information overlay to frame"""
        frame_copy = frame.copy()

        overlay_height = 200
        overlay = frame_copy[:overlay_height, :].copy()
        cv2.rectangle(
            overlay, (0, 0), (frame_copy.shape[1], overlay_height), (0, 0, 0), -1
        )
        cv2.addWeighted(
            overlay,
            1 - self.config.overlay_opacity,
            frame_copy[:overlay_height, :],
            self.config.overlay_opacity,
            0,
            frame_copy[:overlay_height, :],
        )

        if self.config.show_frame_info:
            self._add_frame_info(frame_copy, frame_id, len(detections))

        if self.config.show_risk_summary:
            self._add_risk_summary(frame_copy, risks)

        if self.config.show_species_count:
            self._add_species_count(frame_copy, risks)

        if self.config.show_system_stats:
            self._add_system_stats(frame_copy, system_stats)

        return frame_copy

    def _add_frame_info(self, frame: np.ndarray, frame_id: int, detections_count: int):
        """Add frame information to overlay"""
        info_lines = [
            f"Frame: {frame_id}",
            f"Detections: {detections_count}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.info_colors["frame_info"],
                self.config.font_thickness,
            )

    def _add_risk_summary(self, frame: np.ndarray, risks: List[BirdRiskAssessment]):
        """Add risk summary to overlay"""
        if not risks:
            return

        risk_levels = {"low": 0, "moderate": 0, "high": 0}
        avg_risk = 0

        for risk in risks:
            risk_levels[risk.risk_level] += 1
            avg_risk += risk.risk_score

        avg_risk /= len(risks)

        x_offset = frame.shape[1] - 300
        y_offset = 25

        cv2.putText(
            frame,
            "Risk Summary:",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.info_colors["risk_high"],
            self.config.font_thickness,
        )

        y_offset += 25
        for level, count in risk_levels.items():
            if count > 0:
                color = self.info_colors[f"risk_{level}"]
                cv2.putText(
                    frame,
                    f"{level.title()}: {count}",
                    (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    color,
                    self.config.font_thickness,
                )
                y_offset += 20

        cv2.putText(
            frame,
            f"Avg Risk: {avg_risk:.1f}",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.info_colors["system_info"],
            self.config.font_thickness,
        )

    def _add_species_count(self, frame: np.ndarray, risks: List[BirdRiskAssessment]):
        """Add species count to overlay"""
        if not risks:
            return

        species_count = {}
        for risk in risks:
            species = risk.species
            species_count[species] = species_count.get(species, 0) + 1

        x_offset = frame.shape[1] // 2 - 150
        y_offset = 25

        cv2.putText(
            frame,
            "Species Detected:",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.info_colors["species_info"],
            self.config.font_thickness,
        )

        y_offset += 25
        for species, count in species_count.items():
            cv2.putText(
                frame,
                f"{species}: {count}",
                (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.info_colors["species_info"],
                self.config.font_thickness,
            )
            y_offset += 20

    def _add_system_stats(self, frame: np.ndarray, system_stats: Dict[str, Any]):
        """Add system statistics to overlay"""
        y_offset = 150

        stats_lines = [
            f"Total Frames: {system_stats.get('total_frames_processed', 0)}",
            f"Total Detections: {system_stats.get('total_detections', 0)}",
            f"High Risk Events: {system_stats.get('total_high_risk_events', 0)}",
        ]

        for i, line in enumerate(stats_lines):
            cv2.putText(
                frame,
                line,
                (10, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.info_colors["system_info"],
                1,
            )

    def create_risk_alert_overlay(
        self,
        frame: np.ndarray,
        risks: List[BirdRiskAssessment],
        alert_threshold: float = 8.0,
    ) -> np.ndarray:
        """Create alert overlay for high-risk situations"""
        high_risk_count = sum(1 for risk in risks if risk.risk_score > alert_threshold)

        if high_risk_count == 0:
            return frame

        overlay = frame.copy()

        border_thickness = 10
        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], frame.shape[0]),
            (0, 0, 255),
            border_thickness,
        )

        alert_text = f"ALERT: {high_risk_count} HIGH RISK DETECTIONS"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]

        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 100

        cv2.rectangle(
            overlay,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            overlay,
            alert_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

        return overlay

    def create_performance_metrics_frame(
        self, system_stats: Dict[str, Any], frame_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        """Create a frame showing performance metrics"""
        metrics_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        cv2.putText(
            metrics_frame,
            "System Performance Metrics",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        uptime = datetime.now() - system_stats.get("system_uptime", datetime.now())
        fps = system_stats.get("total_frames_processed", 0) / max(
            1, uptime.total_seconds()
        )
        detection_rate = system_stats.get("total_detections", 0) / max(
            1, system_stats.get("total_frames_processed", 1)
        )

        metrics = [
            f"Uptime: {int(uptime.total_seconds())} seconds",
            f"FPS: {fps:.2f}",
            f"Detection Rate: {detection_rate:.2f} detections/frame",
            f"Total Detections: {system_stats.get('total_detections', 0)}",
            f"High Risk Events: {system_stats.get('total_high_risk_events', 0)}",
            f"Species Detected: {len(system_stats.get('bird_species_detected', set()))}",
        ]

        y_offset = 80
        for metric in metrics:
            cv2.putText(
                metrics_frame,
                metric,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += 40

        return metrics_frame

    def create_risk_distribution_chart(
        self, system_stats: Dict[str, Any], frame_size: Tuple[int, int] = (600, 400)
    ) -> np.ndarray:
        """Create a simple risk distribution chart"""
        chart_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        #Get risk distribution
        risk_levels = system_stats.get(
            "risk_levels", {"low": 0, "moderate": 0, "high": 0}
        )
        total = sum(risk_levels.values())

        if total == 0:
            return chart_frame

        #Calculate percentages
        percentages = {
            level: count / total * 100 for level, count in risk_levels.items()
        }

        #Draw chart
        chart_width = 400
        chart_height = 200
        chart_x = (frame_size[0] - chart_width) // 2
        chart_y = 100

        #Draw bars
        bar_width = chart_width // 3
        colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]  #Green, Yellow, Red

        for i, (level, percentage) in enumerate(percentages.items()):
            bar_height = int((percentage / 100) * chart_height)
            bar_x = chart_x + i * bar_width
            bar_y = chart_y + chart_height - bar_height

            cv2.rectangle(
                chart_frame,
                (bar_x, bar_y),
                (bar_x + bar_width - 10, chart_y + chart_height),
                colors[i],
                -1,
            )

            #Add label
            label = f"{level.title()}: {percentage:.1f}%"
            cv2.putText(
                chart_frame,
                label,
                (bar_x, chart_y + chart_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.putText(
            chart_frame,
            "Risk Level Distribution",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return chart_frame

    def create_species_summary_frame(
        self, system_stats: Dict[str, Any], frame_size: Tuple[int, int] = (800, 600)
    ) -> np.ndarray:
        """Create a frame showing species summary"""
        summary_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        # Add title
        cv2.putText(
            summary_frame,
            "Detected Species Summary",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        #Get species information
        species_detected = system_stats.get("bird_species_detected", set())
        behavior_patterns = system_stats.get("behavior_patterns", {})
        size_categories = system_stats.get("size_categories", {})

        #Add species list
        y_offset = 80
        cv2.putText(
            summary_frame,
            "Species Detected:",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        y_offset += 30
        for species in sorted(species_detected):
            cv2.putText(
                summary_frame,
                f"• {species}",
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 25

        #Add behavior patterns
        y_offset += 20
        cv2.putText(
            summary_frame,
            "Behavior Patterns:",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        y_offset += 30
        for behavior, count in behavior_patterns.items():
            cv2.putText(
                summary_frame,
                f"• {behavior}: {count}",
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 25

        #Add size distribution
        y_offset += 20
        cv2.putText(
            summary_frame,
            "Size Distribution:",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )

        y_offset += 30
        for size, count in size_categories.items():
            cv2.putText(
                summary_frame,
                f"• {size.title()}: {count}",
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 25

        return summary_frame
