"""Region of Interest (ROI) detection module."""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class ROI:
    """Region of interest bounding box."""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    
    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop image to this ROI."""
        h, w = image.shape[:2]
        x0 = max(0, min(self.x, w - 1))
        y0 = max(0, min(self.y, h - 1))
        x1 = max(x0 + 1, min(self.x + self.w, w))
        y1 = max(y0 + 1, min(self.y + self.h, h))
        return image[y0:y1, x0:x1]


class ROIDetector:
    """Detect regions of interest containing instruction text."""
    
    def __init__(self, config: Dict[str, Any]):
        self.mode = config.get("mode", "fixed")
        self.fixed_config = config.get("fixed", {})
        self.adaptive_config = config.get("adaptive", {})
    
    def detect(self, image: np.ndarray) -> List[ROI]:
        """Detect ROIs in image."""
        if self.mode == "fixed":
            return self._detect_fixed(image)
        elif self.mode == "adaptive":
            return self._detect_adaptive(image)
        else:
            raise ValueError(f"Unknown ROI mode: {self.mode}")
    
    def _detect_fixed(self, image: np.ndarray) -> List[ROI]:
        """Use fixed normalized coordinates."""
        h, w = image.shape[:2]
        
        norm_x = self.fixed_config.get("x", 0.52)
        norm_y = self.fixed_config.get("y", 0.03)
        norm_w = self.fixed_config.get("w", 0.45)
        norm_h = self.fixed_config.get("h", 0.30)
        
        roi = ROI(
            x=int(norm_x * w),
            y=int(norm_y * h),
            w=int(norm_w * w),
            h=int(norm_h * h),
            confidence=1.0
        )
        
        return [roi]
    
    def _detect_adaptive(self, image: np.ndarray) -> List[ROI]:
        """Use learned model to detect instruction panel."""
        raise NotImplementedError("Adaptive ROI detection not yet implemented. Use 'fixed' mode.")
