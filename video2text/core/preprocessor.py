"""Image preprocessing for OCR optimization."""

import cv2
import numpy as np
from typing import Dict, Any, Optional


class Preprocessor:
    """Preprocess images to improve OCR accuracy on ultrasound UI."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grayscale = config.get("grayscale", True)
        
        self.clahe_enabled = config.get("clahe", {}).get("enabled", True)
        self.clahe_clip = config.get("clahe", {}).get("clip_limit", 2.0)
        self.clahe_grid = tuple(config.get("clahe", {}).get("tile_grid_size", [8, 8]))
        
        self.denoise_enabled = config.get("denoise", {}).get("enabled", True)
        self.denoise_method = config.get("denoise", {}).get("method", "gaussian")
        self.denoise_kernel = config.get("denoise", {}).get("kernel_size", 3)
        
        self.upscale_enabled = config.get("upscale", {}).get("enabled", True)
        self.upscale_factor = config.get("upscale", {}).get("factor", 1.5)
        
        self.sharpen_enabled = config.get("sharpen", {}).get("enabled", False)
        self.sharpen_kernel = config.get("sharpen", {}).get("kernel", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        self.invert_enabled = config.get("invert", {}).get("enabled", True)
        self.adaptive_threshold = config.get("adaptive_threshold", {}).get("enabled", False)
        
        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to image."""
        result = image.copy()
        
        if self.grayscale and len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Invert for white-on-black text (Tesseract works better with black-on-white)
        if self.invert_enabled and len(result.shape) == 2:
            mean_intensity = np.mean(result)
            if mean_intensity < 128:  # Dark background detected
                result = cv2.bitwise_not(result)
        
        if self.clahe_enabled and len(result.shape) == 2:
            result = self.clahe.apply(result)
        
        if self.denoise_enabled:
            if self.denoise_method == "gaussian":
                result = cv2.GaussianBlur(result, (self.denoise_kernel, self.denoise_kernel), 0)
            elif self.denoise_method == "bilateral":
                result = cv2.bilateralFilter(result, self.denoise_kernel, 75, 75)
        
        # Adaptive thresholding for better text separation
        if self.adaptive_threshold and len(result.shape) == 2:
            result = cv2.adaptiveThreshold(
                result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        if self.upscale_enabled and self.upscale_factor != 1.0:
            result = cv2.resize(
                result, 
                None, 
                fx=self.upscale_factor, 
                fy=self.upscale_factor, 
                interpolation=cv2.INTER_CUBIC
            )
        
        if self.sharpen_enabled:
            kernel = np.array(self.sharpen_kernel, dtype=np.float32)
            result = cv2.filter2D(result, -1, kernel)
        
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
