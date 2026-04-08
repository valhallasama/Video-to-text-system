"""OCR engine wrapper for text recognition."""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path


@dataclass
class OCRResult:
    """Single OCR detection result."""
    text: str
    confidence: float
    bbox: List[List[int]]


class OCREngine:
    """Wrapper for multiple OCR engines (Tesseract, PaddleOCR, EasyOCR)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.engine_type = config.get("engine", "tesseract")
        self.lang = config.get("lang", "en")
        self.min_confidence = config.get("min_confidence", 0.6)
        self.use_gpu = config.get("use_gpu", False)
        self.single_pass = config.get("single_pass", False)
        self.psm_mode = config.get("psm_mode", 6)
        self.char_whitelist = config.get("char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -.")
        
        self._init_engine()
    
    def _init_engine(self):
        """Initialize OCR engine."""
        if self.engine_type == "tesseract":
            try:
                result = subprocess.run(['tesseract', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError("Tesseract not found")
            except FileNotFoundError:
                raise RuntimeError("Tesseract not installed. Run: sudo apt-get install tesseract-ocr")
        
        elif self.engine_type == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(lang=self.lang, use_angle_cls=True)
            except ImportError:
                raise RuntimeError("PaddleOCR not installed. Run: pip install paddleocr")
        
        elif self.engine_type == "easyocr":
            try:
                import easyocr
                self.ocr = easyocr.Reader([self.lang], gpu=self.use_gpu)
                print(f"EasyOCR initialized with GPU={'enabled' if self.use_gpu else 'disabled'}")
            except ImportError:
                raise RuntimeError("EasyOCR not installed. Run: pip install easyocr")
        
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine_type}")
    
    def recognize(self, image: np.ndarray) -> List[OCRResult]:
        """Run OCR on image and return results."""
        if self.engine_type == "tesseract":
            return self._recognize_tesseract(image)
        elif self.engine_type == "paddleocr":
            return self._recognize_paddle(image)
        elif self.engine_type == "easyocr":
            return self._recognize_easyocr(image)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine_type}")
    
    def _recognize_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """Run Tesseract OCR recognition with single or multiple PSM modes."""
        # Use single PSM mode for speed or multiple for accuracy
        if self.single_pass:
            psm_modes = [str(self.psm_mode)]  # Single mode: 3x faster
        else:
            psm_modes = ['6', '11', '3']  # Multiple modes: better accuracy
        
        all_results = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            img_path = tmpdir_path / "input.png"
            
            cv2.imwrite(str(img_path), image)
            
            for psm in psm_modes:
                output_base = tmpdir_path / f"output_{psm}"
                
                cmd = [
                    'tesseract',
                    str(img_path),
                    str(output_base),
                    '--psm', psm,
                    '-c', f'tessedit_char_whitelist={self.char_whitelist}',
                    'tsv'
                ]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    continue
                
                tsv_path = Path(str(output_base) + '.tsv')
                if tsv_path.exists():
                    results = self._parse_tesseract_tsv(tsv_path)
                    all_results.extend(results)
            
            # Deduplicate results, keeping highest confidence
            unique_results = {}
            for result in all_results:
                key = result.text.lower()
                if key not in unique_results or result.confidence > unique_results[key].confidence:
                    unique_results[key] = result
            
            return list(unique_results.values())
    
    def _parse_tesseract_tsv(self, tsv_path: Path) -> List[OCRResult]:
        """Parse Tesseract TSV output."""
        ocr_results: List[OCRResult] = []
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return ocr_results
        
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 12:
                continue
            
            try:
                level = int(parts[0])
                if level != 5:
                    continue
                
                left = int(parts[6])
                top = int(parts[7])
                width = int(parts[8])
                height = int(parts[9])
                conf = float(parts[10])
                text = parts[11] if len(parts) > 11 else ""
                
                if text.strip() and conf >= self.min_confidence:
                    bbox = [
                        [left, top],
                        [left + width, top],
                        [left + width, top + height],
                        [left, top + height]
                    ]
                    
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=conf / 100.0,
                        bbox=bbox
                    ))
            except (ValueError, IndexError):
                continue
        
        return ocr_results
    
    def _recognize_paddle(self, image: np.ndarray) -> List[OCRResult]:
        """Run PaddleOCR recognition."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = self.ocr.predict(rgb)
        
        ocr_results: List[OCRResult] = []
        
        if not result or not result[0]:
            return ocr_results
        
        for line in result[0]:
            bbox, (text, conf) = line
            
            if conf >= self.min_confidence:
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=float(conf),
                    bbox=bbox
                ))
        
        return ocr_results
    
    def _recognize_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Run EasyOCR recognition."""
        result = self.ocr.readtext(image)
        
        ocr_results: List[OCRResult] = []
        
        for bbox, text, conf in result:
            if conf >= self.min_confidence:
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=float(conf),
                    bbox=bbox
                ))
        
        return ocr_results
    
    def get_best_result(self, results: List[OCRResult]) -> Optional[OCRResult]:
        """Get highest confidence result."""
        if not results:
            return None
        return max(results, key=lambda r: r.confidence)
