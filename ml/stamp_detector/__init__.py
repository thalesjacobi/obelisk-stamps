"""
Stamp Detector Module

A reusable YOLO-based stamp detection system for identifying and extracting
individual stamps from multi-stamp images.

Usage:
    from ml.stamp_detector import StampDetector

    detector = StampDetector()
    stamps = detector.detect_and_crop("path/to/image.jpg")

    # stamps is a list of cropped PIL Images or saved file paths
"""

from .detector import StampDetector

__all__ = ["StampDetector"]
__version__ = "1.0.0"
