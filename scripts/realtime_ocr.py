#!/usr/bin/env python3
"""Real-time OCR on live GE ultrasound video with dual ROI support."""

import cv2
import argparse
import sys
import time
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video2text.core.preprocessor import Preprocessor
from video2text.core.ocr_engine import OCREngine
from video2text.core.postprocessor import TextPostprocessor


class DigitTemplateRecognizer:
    """Template matching recognizer for 7-segment digits."""
    
    def __init__(self, template_dir):
        """Load digit templates."""
        self.templates = {}
        self.template_size = (40, 60)
        
        template_path = Path(template_dir)
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
        
        for digit in range(10):
            template_file = template_path / f"{digit}.png"
            if template_file.exists():
                tmpl = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    tmpl = cv2.resize(tmpl, self.template_size, interpolation=cv2.INTER_AREA)
                    self.templates[str(digit)] = tmpl
        
        if len(self.templates) != 10:
            missing = set('0123456789') - set(self.templates.keys())
            raise ValueError(f"Missing templates for digits: {missing}")
    
    def match_digit(self, digit_img):
        """Match a single digit image against templates."""
        if digit_img is None or digit_img.size == 0:
            return None, 0.0
        
        digit_resized = cv2.resize(digit_img, self.template_size, interpolation=cv2.INTER_AREA)
        
        best_score = -1
        best_digit = None
        
        for digit_str, template in self.templates.items():
            result = cv2.matchTemplate(digit_resized, template, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            
            if score > best_score:
                best_score = score
                best_digit = digit_str
        
        return best_digit, best_score
    
    def recognize_score(self, roi):
        """Recognize two-digit score from ROI."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        
        h, w = thresh.shape
        digit_w = w // 2
        
        digit1_img = thresh[:, :digit_w]
        digit2_img = thresh[:, digit_w:]
        
        d1, conf1 = self.match_digit(digit1_img)
        d2, conf2 = self.match_digit(digit2_img)
        
        if d1 is None or d2 is None:
            return None, 0.0
        
        score = d1 + d2
        avg_confidence = (conf1 + conf2) / 2.0
        
        return score, avg_confidence


class RealtimeOCR:
    """Real-time OCR processor for GE ultrasound video."""
    
    def __init__(self, config_path, device_index=0, interval=1.0):
        """
        Initialize real-time OCR.
        
        Args:
            config_path: Path to config YAML
            device_index: Video device index
            interval: OCR processing interval in seconds
        """
        self.device_index = device_index
        self.interval = interval
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize pipeline components
        self.preprocessor = Preprocessor(self.config['preprocessing'])
        self.ocr_engine = OCREngine(self.config['ocr'])
        self.postprocessor = TextPostprocessor(self.config['postprocessing'])
        
        # Initialize template matching for score detection
        template_dir = Path(__file__).resolve().parents[1] / "templates"
        try:
            self.score_recognizer = DigitTemplateRecognizer(template_dir)
            print("✅ Template matching score recognizer loaded")
        except Exception as e:
            print(f"⚠️  Warning: Could not load score templates: {e}")
            print("   Score detection will be disabled")
            self.score_recognizer = None
        
        # ROI configurations for 1920x1080 video
        # These are normalized [0,1] coordinates
        self.roi_configs = {
            'instruction': {
                'name': 'Instruction Text',
                'x': 0.59,      # Top right area
                'y': 0.045,
                'w': 0.40,
                'h': 0.06,      # Smaller height for 1080p
                'color': (0, 255, 0)  # Green
            },
            'score': {
                'name': 'Score',
                'x': 0.2361,    # Bottom left - score display
                'y': 0.9345,
                'w': 0.0103,
                'h': 0.0141,
                'color': (0, 255, 255)  # Yellow
            }
        }
        
        # State
        self.cap = None
        self.last_ocr_time = 0
        self.frame_count = 0
        self.last_results = {
            'instruction': '',
            'score': ''
        }
        self.last_confidences = {
            'instruction': 0.0,
            'score': 0.0
        }
        self.last_preprocessed_instruction = None  # For debug display
        
        # Logging
        self.log_file = Path(__file__).resolve().parents[1] / "realtime_ocr_log.txt"
        self.init_log()
    
    def init_log(self):
        """Initialize log file."""
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REAL-TIME OCR LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log_result(self, roi_name, text, confidence):
        """Log OCR result to file and console."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_line = f"[{timestamp}] {roi_name:15s}: {text:50s} (conf: {confidence:.1%})"
        
        print(log_line)
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
    
    def get_roi_bbox(self, frame_width, frame_height, roi_config):
        """Convert normalized ROI to pixel coordinates."""
        x = int(roi_config['x'] * frame_width)
        y = int(roi_config['y'] * frame_height)
        w = int(roi_config['w'] * frame_width)
        h = int(roi_config['h'] * frame_height)
        return x, y, w, h
    
    def process_roi(self, frame, roi_name, roi_config):
        """Process a single ROI and return OCR text."""
        h, w = frame.shape[:2]
        
        # Get ROI bounding box
        x, y, roi_w, roi_h = self.get_roi_bbox(w, h, roi_config)
        
        # Crop ROI
        roi = frame[y:y+roi_h, x:x+roi_w]
        
        if roi.size == 0:
            return "", 0.0
        
        # Use minimal preprocessing for instruction text to preserve low-contrast text
        # The default preprocessing (CLAHE, denoise, sharpen) makes faint text too faint
        if roi_name == 'instruction':
            # Simple preprocessing: grayscale + upscale + moderate contrast
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Upscale 2x for better OCR
            preprocessed = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            # Moderate contrast enhancement - not too aggressive
            preprocessed = cv2.convertScaleAbs(preprocessed, alpha=1.3, beta=15)
        else:
            # Use standard preprocessing for other ROIs
            preprocessed = self.preprocessor.process(roi)
        
        # Save preprocessed instruction for debug display
        if roi_name == 'instruction':
            self.last_preprocessed_instruction = preprocessed.copy()
        
        # OCR
        ocr_results = self.ocr_engine.recognize(preprocessed)
        
        if not ocr_results:
            return "", 0.0
        
        # Postprocess and combine words
        words = []
        confidences = []
        
        for result in ocr_results:
            cleaned = self.postprocessor.process(result.text)
            if cleaned:
                words.append(cleaned)
                confidences.append(result.confidence)
        
        if not words:
            return "", 0.0
        
        # Combine words
        text = ' '.join(words)
        
        # Apply context-aware phrase corrections for instruction text
        if roi_name == 'instruction':
            # Pattern: "their l side" → "their left side"
            text = text.replace('their l side', 'their left side')
            text = text.replace('their l slide', 'their left side')
            
            # Pattern: "their r side" → "their right side"
            text = text.replace('their r side', 'their right side')
            text = text.replace('their r slide', 'their right side')
            
            # Pattern: "slide" at end → "side" (common misreading)
            if text.endswith(' slide'):
                text = text[:-5] + 'side'
            
            # Pattern: "lying on l" → "lying on left"
            text = text.replace('lying on l ', 'lying on left ')
            text = text.replace('lying on r ', 'lying on right ')
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return text, avg_confidence
    
    def draw_rois(self, frame):
        """Draw ROI boxes on frame."""
        h, w = frame.shape[:2]
        
        for roi_name, roi_config in self.roi_configs.items():
            x, y, roi_w, roi_h = self.get_roi_bbox(w, h, roi_config)
            color = roi_config['color']
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + roi_w, y + roi_h), color, 2)
            
            # Draw label
            label = roi_config['name']
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_results(self, frame):
        """Draw OCR results on frame."""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (630, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw results
        y_offset = 35
        
        # Instruction text
        instruction = self.last_results['instruction']
        if instruction:
            cv2.putText(frame, f"Instruction: {instruction}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Instruction: (detecting...)", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        y_offset += 25
        
        # Score
        score = self.last_results['score']
        if score:
            cv2.putText(frame, f"Score: {score}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Score: (detecting...)", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        y_offset += 25
        
        # Status
        time_since_ocr = time.time() - self.last_ocr_time
        next_ocr = max(0, self.interval - time_since_ocr)
        cv2.putText(frame, f"Next OCR in: {next_ocr:.1f}s", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Run real-time OCR loop."""
        print(f"Opening video device: /dev/video{self.device_index}")
        
        self.cap = cv2.VideoCapture(self.device_index)
        
        if not self.cap.isOpened():
            print(f"❌ Error: Cannot open /dev/video{self.device_index}")
            return False
        
        # Set higher resolution for better OCR
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Get actual video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Video source opened")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   OCR interval: {self.interval}s")
        print()
        print("="*80)
        print("REAL-TIME OCR OUTPUT")
        print("="*80)
        print()
        print("Controls:")
        print("  • Press 'q' or ESC to quit")
        print("  • Press 'r' to toggle ROI display")
        print("  • Press 'd' to toggle debug window")
        print("  • Press 's' to save snapshot")
        print()
        
        # Create window
        window_name = "Real-time OCR - GE Ultrasound"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # State
        show_rois = True
        show_debug = True  # Show debug window by default
        snapshot_count = 0
        snapshot_dir = Path(__file__).resolve().parents[1] / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        
        # Create debug window
        debug_window = "Debug: Preprocessed Instruction Text"
        cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️  Warning: Cannot read frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # Process OCR at specified interval
                if current_time - self.last_ocr_time >= self.interval:
                    print(f"\n--- Frame {self.frame_count} ---")
                    
                    # Process instruction text with OCR
                    if 'instruction' in self.roi_configs:
                        roi_config = self.roi_configs['instruction']
                        text, confidence = self.process_roi(frame, 'instruction', roi_config)
                        self.last_results['instruction'] = text
                        self.last_confidences['instruction'] = confidence
                        self.log_result(roi_config['name'], text, confidence)
                    
                    # Process score with template matching
                    if 'score' in self.roi_configs and self.score_recognizer:
                        roi_config = self.roi_configs['score']
                        h, w = frame.shape[:2]
                        x, y, roi_w, roi_h = self.get_roi_bbox(w, h, roi_config)
                        roi = frame[y:y+roi_h, x:x+roi_w]
                        
                        if roi.size > 0:
                            score, confidence = self.score_recognizer.recognize_score(roi)
                            if score and confidence > 0.7:
                                self.last_results['score'] = score
                                self.last_confidences['score'] = confidence
                                self.log_result(roi_config['name'], score, confidence)
                    
                    self.last_ocr_time = current_time
                
                # Draw visualization
                display_frame = frame.copy()
                
                if show_rois:
                    self.draw_rois(display_frame)
                
                self.draw_results(display_frame)
                
                # Show frame
                cv2.imshow(window_name, display_frame)
                
                # Show debug window with preprocessed instruction text
                if show_debug and self.last_preprocessed_instruction is not None:
                    debug_img = self.last_preprocessed_instruction.copy()
                    
                    # Convert to color if grayscale
                    if len(debug_img.shape) == 2:
                        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                    
                    # Enlarge for visibility
                    debug_img = cv2.resize(debug_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                    
                    # Add info text
                    info_text = f"Instruction: {self.last_results['instruction']}"
                    conf_text = f"Confidence: {self.last_confidences['instruction']:.1%}"
                    
                    cv2.putText(debug_img, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(debug_img, conf_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(debug_img, "Press 'd' to hide", (10, debug_img.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                    
                    cv2.imshow(debug_window, debug_img)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nExiting...")
                    break
                
                elif key == ord('r'):  # Toggle ROI display
                    show_rois = not show_rois
                
                elif key == ord('d'):  # Toggle debug window
                    show_debug = not show_debug
                    if not show_debug:
                        cv2.destroyWindow(debug_window)
                    else:
                        cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)
                
                elif key == ord('s'):  # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    snapshot_path = snapshot_dir / f"snapshot_{timestamp}.png"
                    cv2.imwrite(str(snapshot_path), frame)
                    snapshot_count += 1
                    print(f"📸 Snapshot saved: {snapshot_path}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            print()
            print("="*80)
            print("SESSION SUMMARY")
            print("="*80)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Snapshots saved: {snapshot_count}")
            print(f"Log file: {self.log_file}")
            print("="*80)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Real-time OCR on GE ultrasound video"
    )
    parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='Video device index (default: 0)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=1.0,
        help='OCR processing interval in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(__file__).resolve().parents[1] / args.config
    
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Run real-time OCR
    ocr = RealtimeOCR(
        config_path=config_path,
        device_index=args.device,
        interval=args.interval
    )
    
    success = ocr.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
