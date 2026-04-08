"""Visualization utilities for debugging."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from ..core.roi_detector import ROI
from ..core.ocr_engine import OCRResult


class Visualizer:
    """Visualize pipeline outputs for debugging."""
    
    @staticmethod
    def draw_roi(image: np.ndarray, roi: ROI, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """Draw ROI bounding box on image."""
        result = image.copy()
        cv2.rectangle(result, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, thickness)
        
        label = f"ROI (conf: {roi.confidence:.2f})"
        cv2.putText(result, label, (roi.x, roi.y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    @staticmethod
    def draw_ocr_results(image: np.ndarray, results: List[OCRResult], 
                        color=(255, 0, 0), thickness=2) -> np.ndarray:
        """Draw OCR bounding boxes and text on image."""
        result = image.copy()
        
        for ocr_result in results:
            bbox = ocr_result.bbox
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(result, [pts], True, color, thickness)
            
            x, y = int(bbox[0][0]), int(bbox[0][1])
            label = f"{ocr_result.text} ({ocr_result.confidence:.2f})"
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    @staticmethod
    def save_debug_frame(image: np.ndarray, output_dir: Path, 
                        frame_idx: int, stage: str):
        """Save debug frame to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"frame_{frame_idx:04d}_{stage}.jpg"
        cv2.imwrite(str(output_dir / filename), image)
    
    @staticmethod
    def create_side_by_side(images: List[np.ndarray], labels: List[str]) -> np.ndarray:
        """Create side-by-side comparison of images."""
        max_height = max(img.shape[0] for img in images)
        
        resized = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            scale = max_height / img.shape[0]
            new_w = int(img.shape[1] * scale)
            resized.append(cv2.resize(img, (new_w, max_height)))
        
        result = np.hstack(resized)
        
        x_offset = 0
        for i, (img, label) in enumerate(zip(resized, labels)):
            cv2.putText(result, label, (x_offset + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            x_offset += img.shape[1]
        
        return result
