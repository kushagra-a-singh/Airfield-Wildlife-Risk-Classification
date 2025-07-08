"""
Utils Module for Bird Detection System
Contains utility classes for video processing and dashboard functionality
"""

from .dashboard_utils import DashboardUtils, SystemOverlayConfig
from .video_processor import VideoProcessor, VisualizationConfig

__all__ = [
    "VideoProcessor",
    "VisualizationConfig",
    "DashboardUtils",
    "SystemOverlayConfig",
]
