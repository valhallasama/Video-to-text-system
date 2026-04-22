#!/usr/bin/env python3
"""
Extract dataset from ultrasound video files.

This script processes ultrasound video files frame-by-frame and extracts:
- Ultrasound fan area images (the triangular imaging region)
- Instruction text via OCR
- Quality bar score
- Frame numbers for alignment

Output structure:
- frames/frame_XXXXX.png - Ultrasound fan images
- dataset.csv - Frame number, instruction text, quality score
"""

import cv2
import numpy as np
import csv
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video2text.core.preprocessor import Preprocessor
from video2text.core.ocr_engine import OCREngine
from video2text.core.postprocessor import TextPostprocessor


class QualityBarAnalyzer:
    """Analyzer for the vertical quality bar indicator."""
    
    def analyze_quality_bar(self, roi):
        """Analyze quality bar and return percentage score."""
        if roi is None or roi.size == 0:
            return 0
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Find actual bar boundaries (ignore black padding)
        bar_start = 0
        bar_end = height - 1
        
        for y in range(height):
            if np.mean(gray[y, :]) > 30:
                bar_start = y
                break
        
        for y in range(height - 1, -1, -1):
            if np.mean(gray[y, :]) > 30:
                bar_end = y
                break
        
        bar_height = bar_end - bar_start + 1
        if bar_height <= 0:
            return 0
        
        # Count bright rows (light gray, white, or green)
        bright_threshold = 60
        bright_rows = 0
        
        for y in range(bar_start, bar_end + 1):
            row_brightness = np.mean(gray[y, :])
            if row_brightness > bright_threshold:
                bright_rows += 1
        
        quality_percentage = int((bright_rows / bar_height) * 100)
        quality_percentage += 1  # Compensate for excluding green overflow
        
        return min(99, max(0, quality_percentage))


class DatasetExtractor:
    """Extract dataset from ultrasound video."""
    
    def __init__(self, output_dir, save_original_frames=False):
        """Initialize extractor."""
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Option to save original full frames
        self.save_original_frames = save_original_frames
        if save_original_frames:
            self.original_frames_dir = self.output_dir / "original_frames"
            self.original_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR components
        preprocess_config = {
            'resize': {'enabled': False},
            'grayscale': {'enabled': True},
            'denoise': {'enabled': True, 'method': 'bilateral'},
            'threshold': {'enabled': True, 'method': 'adaptive'}
        }
        self.preprocessor = Preprocessor(preprocess_config)
        
        ocr_config = {
            'engine': 'tesseract',
            'lang': 'eng',
            'min_confidence': 0.6,
            'psm_mode': 6
        }
        self.ocr_engine = OCREngine(ocr_config)
        
        postprocess_config = {
            'remove_special_chars': True,
            'normalize_whitespace': True,
            'lowercase': True,
            'spell_check': True,
            'spell_correction': {
                'enabled': True,
                'use_medical_dictionary': True
            },
            'app_vocabulary': {
                'enabled': True,
                'strict_mode': False
            },
            'blacklist': ["|||", "___", "...", "===", "protocol", "views", "guidance", 
                         "short", "shift", "shor", "ss a", "ss", "a a", "- -"]
        }
        self.postprocessor = TextPostprocessor(postprocess_config)
        self.quality_analyzer = QualityBarAnalyzer()
        
        # ROI configurations (normalized coordinates for 1920x1080)
        self.roi_configs = {
            'ultrasound_fan': {
                'x': 0.0850,
                'y': 0.1250,
                'w': 0.4800,
                'h': 0.7600,
            },
            'instruction': {
                'x': 0.5750,
                'y': 0.1300,
                'w': 0.3550,
                'h': 0.0600,
            },
            'quality_bar': {
                'x': 0.5795,
                'y': 0.2586,
                'w': 0.0150,
                'h': 0.2938,
            }
        }
    
    def get_roi_bbox(self, frame_w, frame_h, roi_config):
        """Convert normalized ROI to pixel coordinates."""
        x = int(roi_config['x'] * frame_w)
        y = int(roi_config['y'] * frame_h)
        w = int(roi_config['w'] * frame_w)
        h = int(roi_config['h'] * frame_h)
        return x, y, w, h
    
    def extract_ultrasound_fan(self, frame):
        """Extract the ultrasound fan/imaging area."""
        h, w = frame.shape[:2]
        roi_config = self.roi_configs['ultrasound_fan']
        x, y, roi_w, roi_h = self.get_roi_bbox(w, h, roi_config)
        
        roi = frame[y:y+roi_h, x:x+roi_w]
        return roi
    
    def extract_instruction(self, frame):
        """Extract instruction text from frame."""
        h, w = frame.shape[:2]
        roi_config = self.roi_configs['instruction']
        x, y, roi_w, roi_h = self.get_roi_bbox(w, h, roi_config)
        
        roi = frame[y:y+roi_h, x:x+roi_w]
        
        if roi.size == 0:
            return ""
        
        # Custom preprocessing for instruction text (matching realtime_ocr.py)
        # Simple preprocessing: grayscale + upscale + moderate contrast
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Upscale 2x for better OCR
        preprocessed = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # Moderate contrast enhancement - not too aggressive
        preprocessed = cv2.convertScaleAbs(preprocessed, alpha=1.3, beta=15)
        
        # OCR
        ocr_results = self.ocr_engine.recognize(preprocessed)
        
        if not ocr_results:
            return ""
        
        # Postprocess each word individually (like realtime_ocr.py)
        words = []
        for result in ocr_results:
            cleaned = self.postprocessor.process(result.text)
            if cleaned:
                words.append(cleaned)
        
        # Filter out results with too few valid words (likely noise in empty areas)
        # If we only got 1-2 short words, it's probably OCR noise
        if len(words) <= 2 and all(len(w) <= 3 for w in words):
            return ""
        
        if not words:
            return ""
        
        # Combine words
        text = ' '.join(words)
        return text.strip()
    
    def extract_quality(self, frame):
        """Extract quality score from quality bar."""
        h, w = frame.shape[:2]
        roi_config = self.roi_configs['quality_bar']
        x, y, roi_w, roi_h = self.get_roi_bbox(w, h, roi_config)
        
        roi = frame[y:y+roi_h, x:x+roi_w]
        
        if roi.size == 0:
            return 0
        
        return self.quality_analyzer.analyze_quality_bar(roi)
    
    def process_video(self, video_path, sample_rate=1):
        """Process video and extract dataset.
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1 = all frames)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return
        
        print(f"Processing video: {video_path}")
        print(f"Output directory: {self.output_dir}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print(f"Sample rate: every {sample_rate} frame(s)")
        
        # Prepare CSV output
        csv_path = self.output_dir / "dataset.csv"
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_number', 'instruction_text', 'quality_score', 'image_path'])
        
        frame_idx = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Sample frames
                if frame_idx % sample_rate != 0:
                    frame_idx += 1
                    continue
                
                # Extract data
                ultrasound_img = self.extract_ultrasound_fan(frame)
                instruction_text = self.extract_instruction(frame)
                quality_score = self.extract_quality(frame)
                
                # Save ultrasound image
                img_filename = f"frame_{frame_idx:06d}.png"
                img_path = self.frames_dir / img_filename
                cv2.imwrite(str(img_path), ultrasound_img)
                
                # Save original full frame if requested
                if self.save_original_frames:
                    # Add frame number annotation
                    original_annotated = frame.copy()
                    cv2.putText(original_annotated, f"Frame: {frame_idx}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    original_filename = f"original_{frame_idx:06d}.png"
                    original_path = self.original_frames_dir / original_filename
                    cv2.imwrite(str(original_path), original_annotated)
                
                # Write to CSV
                csv_writer.writerow([
                    frame_idx,
                    instruction_text,
                    quality_score,
                    f"frames/{img_filename}"
                ])
                
                processed_count += 1
                
                # Progress update
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} frames (frame {frame_idx}/{total_frames})")
                
                frame_idx += 1
        
        finally:
            cap.release()
            csv_file.close()
        
        print(f"\n✅ Dataset extraction complete!")
        print(f"   Processed: {processed_count} frames")
        print(f"   Images: {self.frames_dir}")
        print(f"   CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract dataset from ultrasound video')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('-o', '--output', type=str, default='dataset_output',
                        help='Output directory (default: dataset_output)')
    parser.add_argument('-s', '--sample-rate', type=int, default=1,
                        help='Process every Nth frame (default: 1 = all frames)')
    parser.add_argument('--save-original', action='store_true',
                        help='Save original full frames with frame number annotations')
    
    args = parser.parse_args()
    
    extractor = DatasetExtractor(args.output, save_original_frames=args.save_original)
    extractor.process_video(args.video, args.sample_rate)


if __name__ == '__main__':
    main()
