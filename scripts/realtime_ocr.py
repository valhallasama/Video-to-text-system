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


class QualityBarAnalyzer:
    """Analyzer for vertical quality bar with gray/white/green segments."""
    
    def __init__(self):
        """Initialize quality bar analyzer."""
        pass
    
    def analyze_quality_bar(self, roi):
        """Analyze quality bar and return percentage score.
        
        The bar has a dark gray background. As quality increases, bright segments appear from bottom:
        - Light gray: low quality (bright, not dark background)
        - White: medium quality (brighter)
        - Green: high quality (appears at ~99%)
        
        Quality score = percentage of bar with BRIGHT colored segments (not dark background)
        
        Returns:
            quality_percentage: 0-100 score based on bright segment height
        """
        if roi is None or roi.size == 0:
            return 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check if frame is essentially black (screen transition, no video)
        # If mean brightness is very low, it's a black frame
        overall_brightness = np.mean(gray)
        if overall_brightness < 20:  # Nearly black frame
            return 0
        
        # Get bar dimensions
        height, width = gray.shape
        
        # Find actual bar boundaries (ignore black padding at top/bottom)
        # Detect where the bar (dark gray or brighter) starts and ends
        bar_start = 0
        bar_end = height - 1
        
        for y in range(height):
            if np.mean(gray[y, :]) > 30:  # Found bar (not black)
                bar_start = y
                break
        
        for y in range(height - 1, -1, -1):
            if np.mean(gray[y, :]) > 30:  # Found bar (not black)
                bar_end = y
                break
        
        # The bar structure from bottom to top:
        # - Light gray filled portion (quality indicator)
        # - Dark gray background (unfilled)
        # - Black separator line (marks 99% point)
        # - Green overflow area (above 99%)
        
        # For now, use the entire bar height as reference
        # The quality is measured from bottom upward
        bar_height = bar_end - bar_start + 1
        if bar_height <= 0:
            return 0
        
        # Count rows with bright pixels (light gray, white, or green) from bottom up
        # Dark gray background has low brightness (~40-50)
        # Light gray segments have brightness (~60-95)
        # White/green segments have higher brightness (>100)
        bright_threshold = 60
        
        bright_rows = 0
        for y in range(bar_start, bar_end + 1):
            row_brightness = np.mean(gray[y, :])
            if row_brightness > bright_threshold:
                bright_rows += 1
        
        # Convert to percentage based on bar height
        quality_percentage = int((bright_rows / bar_height) * 100)
        
        # Add 1 to compensate for excluding green overflow area
        quality_percentage += 1
        
        # Cap at 99% maximum (never show 100%)
        return min(99, max(0, quality_percentage))


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
    
    def __init__(self, config_path, device_index=0, interval=1.0, save_dataset=False, dataset_dir="realtime_dataset", robot_pose_mqtt_url=None):
        """
        Initialize real-time OCR.
        
        Args:
            config_path: Path to config YAML
            device_index: Video device index
            interval: OCR processing interval in seconds
            save_dataset: If True, save frames as dataset
            dataset_dir: Directory to save dataset
            robot_pose_mqtt_url: MQTT broker URL (e.g., mqtt://192.168.1.100:1883)
        """
        self.device_index = device_index
        self.interval = interval
        self.start_time = datetime.now()
        
        # Dataset generation settings
        self.save_dataset = save_dataset
        self.dataset_dir = Path(dataset_dir) if save_dataset else None
        self.dataset_frame_count = 0
        self.robot_pose_client = None
        
        if self.save_dataset:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_frames_dir = self.dataset_dir / "frames"
            self.dataset_frames_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_original_frames_dir = self.dataset_dir / "original_frames"
            self.dataset_original_frames_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_csv_path = self.dataset_dir / "dataset.csv"
            
            # Initialize robot pose MQTT client if URL provided
            if robot_pose_mqtt_url:
                try:
                    # Parse MQTT URL: mqtt://host:port
                    if robot_pose_mqtt_url.startswith('mqtt://'):
                        mqtt_url = robot_pose_mqtt_url[7:]  # Remove mqtt://
                        if ':' in mqtt_url:
                            broker_host, broker_port = mqtt_url.split(':')
                            broker_port = int(broker_port)
                        else:
                            broker_host = mqtt_url
                            broker_port = 1883
                        
                        from video2text.utils.robot_pose_mqtt_client import RobotPoseMQTTClient
                        self.robot_pose_client = RobotPoseMQTTClient(
                            broker_host=broker_host,
                            broker_port=broker_port
                        )
                    else:
                        print(f"⚠️  Invalid MQTT URL format: {robot_pose_mqtt_url}")
                        print(f"   Expected format: mqtt://host:port (e.g., mqtt://192.168.1.100:1883)")
                except ImportError as e:
                    print(f"⚠️  Cannot import RobotPoseMQTTClient: {e}")
                    print(f"   Robot coordinates will not be recorded.")
                except Exception as e:
                    print(f"⚠️  Failed to initialize MQTT client: {e}")
                    print(f"   Robot coordinates will not be recorded.")
            
            # Create CSV with header
            header = "frame_number,instruction_text,quality_score,image_path"
            if self.robot_pose_client and self.robot_pose_client.is_connected():
                header += ",robot_x,robot_y,robot_z,robot_qx,robot_qy,robot_qz,robot_qw,roll_deg,pitch_deg,yaw_deg"
            header += "\n"
            
            with open(self.dataset_csv_path, 'w') as f:
                f.write(header)
        
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
            print("   Using OCR-based score detection instead")
            self.score_recognizer = None
        
        # Initialize quality bar analyzer
        self.quality_bar_analyzer = QualityBarAnalyzer()
        
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
            },
            'quality_bar': {
                'name': 'Quality',
                'x': 0.5995,     # Left side - quality bar (moved right)
                'y': 0.1836,      # Start near top
                'w': 0.010,     # Narrow vertical bar
                'h': 0.3288,      # Most of screen height
                'color': (255, 0, 255)  # Magenta
            }
        }
        
        # State
        self.cap = None
        self.last_ocr_time = 0
        self.frame_count = 0
        self.last_results = {
            'instruction': '',
            'score': '',
            'quality': 0
        }
        self.last_confidences = {
            'instruction': 0.0,
            'score': 0.0,
            'quality': 100.0
        }
        self.last_preprocessed_instruction = None  # For debug display
        self.last_preprocessed_score = None  # For debug display
        self.last_quality_bar_roi = None  # For debug display
        
        # Change detection - track previous values to detect changes
        self.previous_results = {
            'instruction': '',
            'score': '',
            'quality': 0
        }
        
        # Logging - temporary file during runtime
        self.log_dir = Path(__file__).resolve().parents[1] / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.temp_log_file = self.log_dir / "temp_current_session.txt"
        self.final_log_file = None
        self.init_log()
    
    def init_log(self):
        """Initialize log file."""
        with open(self.temp_log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REAL-TIME OCR LOG\n")
            f.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log_result(self, roi_name, text, confidence):
        """Log OCR result to file and console."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_line = f"[{timestamp}] {roi_name:15s}: {text:50s} (conf: {confidence:.1%})"
        
        print(log_line)
        
        with open(self.temp_log_file, 'a') as f:
            f.write(log_line + "\n")
    
    def finalize_log(self):
        """Rename log file with timestamp when session ends."""
        end_time = datetime.now()
        timestamp_str = end_time.strftime('%Y%m%d_%H%M%S')
        self.final_log_file = self.log_dir / f"test-{timestamp_str}.txt"
        
        # Add session summary to log
        with open(self.temp_log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("SESSION ENDED\n")
            f.write(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = end_time - self.start_time
            f.write(f"Duration: {duration}\n")
            f.write("="*80 + "\n")
        
        # Rename to final timestamped name
        self.temp_log_file.rename(self.final_log_file)
        print(f"\n📝 Log saved: {self.final_log_file}")
    
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
        elif roi_name == 'score':
            # Score preprocessing: grayscale + heavy upscale for small digits
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Upscale 10x for very small score area
            preprocessed = cv2.resize(gray, None, fx=10.0, fy=10.0, interpolation=cv2.INTER_CUBIC)
            # High contrast for digit recognition
            preprocessed = cv2.convertScaleAbs(preprocessed, alpha=1.5, beta=20)
        else:
            # Use standard preprocessing for other ROIs
            preprocessed = self.preprocessor.process(roi)
        
        # Save preprocessed images for debug display
        if roi_name == 'instruction':
            self.last_preprocessed_instruction = preprocessed.copy()
        elif roi_name == 'score':
            self.last_preprocessed_score = preprocessed.copy()
        
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
    
    def draw_results(self, frame, show_overlay=True):
        """Draw OCR results on frame."""
        if not show_overlay:
            return
        
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
        print("  • Press 'w' to start/stop recording (toggle)")
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
        snapshot_dir = Path("snapshots")
        snapshot_dir.mkdir(exist_ok=True)
        
        # Toggle for overlay display
        show_overlay = True
        
        # Recording state (for 'w' key toggle)
        is_recording = self.save_dataset  # Start recording if --save-dataset flag is set
        
        # Create debug windows
        debug_window_instruction = "Debug: Preprocessed Instruction Text"
        debug_window_quality = "Debug: Quality Bar"
        cv2.namedWindow(debug_window_instruction, cv2.WINDOW_NORMAL)
        cv2.namedWindow(debug_window_quality, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️  Warning: Cannot read frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # Process OCR at specified interval (still throttle to avoid excessive CPU usage)
                if current_time - self.last_ocr_time >= self.interval:
                    # Track if any changes detected
                    changes_detected = False
                    
                    # Process instruction text with OCR
                    if 'instruction' in self.roi_configs:
                        roi_config = self.roi_configs['instruction']
                        text, confidence = self.process_roi(frame, 'instruction', roi_config)
                        
                        # Check if text changed
                        if text != self.previous_results['instruction']:
                            if not changes_detected:
                                print(f"\n--- Frame {self.frame_count} ---")
                                changes_detected = True
                            self.log_result(roi_config['name'], text, confidence)
                            self.previous_results['instruction'] = text
                        
                        self.last_results['instruction'] = text
                        self.last_confidences['instruction'] = confidence
                    
                    # Process score with template matching or OCR fallback
                    if 'score' in self.roi_configs:
                        roi_config = self.roi_configs['score']
                        
                        if self.score_recognizer:
                            # Use template matching if available
                            h, w = frame.shape[:2]
                            x, y, roi_w, roi_h = self.get_roi_bbox(w, h, roi_config)
                            roi = frame[y:y+roi_h, x:x+roi_w]
                            
                            score = ''
                            confidence = 0.0
                            
                            if roi.size > 0:
                                score, confidence = self.score_recognizer.recognize_score(roi)
                                if not score or confidence <= 0.7:
                                    score = ''
                                    confidence = 0.0
                            
                            # Check if score changed (including empty state)
                            if score != self.previous_results['score']:
                                if not changes_detected:
                                    print(f"\n--- Frame {self.frame_count} ---")
                                    changes_detected = True
                                self.log_result(roi_config['name'], score, confidence)
                                self.previous_results['score'] = score
                            
                            self.last_results['score'] = score
                            self.last_confidences['score'] = confidence
                        else:
                            # Fallback to OCR-based score detection
                            score_text, score_conf = self.process_roi(frame, 'score', roi_config)
                            
                            # Always output score (not just on change)
                            if not changes_detected:
                                print(f"\n--- Frame {self.frame_count} ---")
                                changes_detected = True
                            self.log_result(roi_config['name'], score_text, score_conf)
                            self.previous_results['score'] = score_text
                            
                            self.last_results['score'] = score_text
                            self.last_confidences['score'] = score_conf
                    
                    # Process quality bar
                    if 'quality_bar' in self.roi_configs:
                        roi_config = self.roi_configs['quality_bar']
                        h, w = frame.shape[:2]
                        x, y, roi_w, roi_h = self.get_roi_bbox(w, h, roi_config)
                        roi = frame[y:y+roi_h, x:x+roi_w]
                        
                        if roi.size > 0:
                            # Save ROI for debug display
                            self.last_quality_bar_roi = roi.copy()
                            
                            quality_score = self.quality_bar_analyzer.analyze_quality_bar(roi)
                            
                            # Check if quality changed (with 2% threshold to avoid noise)
                            if abs(quality_score - self.previous_results['quality']) >= 2:
                                if not changes_detected:
                                    print(f"\n--- Frame {self.frame_count} ---")
                                    changes_detected = True
                                self.log_result(roi_config['name'], f"{quality_score}%", 100.0)
                                self.previous_results['quality'] = quality_score
                            
                            self.last_results['quality'] = quality_score
                            self.last_confidences['quality'] = 100.0
                    
                    self.last_ocr_time = current_time
                
                # Save dataset if enabled and recording is active
                if self.save_dataset and is_recording:
                    # Calculate quality for THIS specific frame (real-time per frame)
                    h, w = frame.shape[:2]
                    
                    # Get quality bar ROI for this frame
                    if 'quality_bar' in self.roi_configs:
                        roi_config = self.roi_configs['quality_bar']
                        qb_x, qb_y, qb_w, qb_h = self.get_roi_bbox(w, h, roi_config)
                        quality_bar_roi = frame[qb_y:qb_y+qb_h, qb_x:qb_x+qb_w]
                        
                        if quality_bar_roi.size > 0:
                            # Calculate quality for this exact frame
                            frame_quality_score = self.quality_bar_analyzer.analyze_quality_bar(quality_bar_roi)
                        else:
                            frame_quality_score = self.last_results['quality']
                    else:
                        frame_quality_score = self.last_results['quality']
                    
                    # Extract ultrasound fan ROI
                    fan_x = int(0.0850 * w)
                    fan_y = int(0.1250 * h)
                    fan_w = int(0.4800 * w)
                    fan_h = int(0.7600 * h)
                    ultrasound_roi = frame[fan_y:fan_y+fan_h, fan_x:fan_x+fan_w]
                    
                    # Save ultrasound ROI frame
                    frame_filename = f"frame_{self.dataset_frame_count:06d}.png"
                    frame_path = self.dataset_frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), ultrasound_roi)
                    
                    # Save original full frame with annotations
                    original_annotated = frame.copy()
                    
                    # Draw ROI boxes on original frame
                    for roi_name, roi_config in self.roi_configs.items():
                        rx, ry, rw, rh = self.get_roi_bbox(w, h, roi_config)
                        color = roi_config['color']
                        cv2.rectangle(original_annotated, (rx, ry), (rx + rw, ry + rh), color, 2)
                        label = f"{roi_config['name']}"
                        cv2.putText(original_annotated, label, (rx, ry - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add frame info
                    cv2.putText(original_annotated, f"Frame: {self.dataset_frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(original_annotated, f"Quality: {frame_quality_score}%", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    
                    original_filename = f"original_{self.dataset_frame_count:06d}.png"
                    original_path = self.dataset_original_frames_dir / original_filename
                    cv2.imwrite(str(original_path), original_annotated)
                    
                    # Append to CSV with frame-specific quality and robot pose
                    instruction_text = self.last_results['instruction'].replace(',', ';') if self.last_results['instruction'] else ''
                    
                    # Build CSV row
                    csv_row = f"{self.dataset_frame_count},{instruction_text},{frame_quality_score},frames/{frame_filename}"
                    
                    # Add robot pose if available
                    if self.robot_pose_client and self.robot_pose_client.is_connected():
                        robot_pose_str = self.robot_pose_client.get_pose_string()
                        if robot_pose_str:
                            csv_row += f",{robot_pose_str}"
                    
                    csv_row += "\n"
                    
                    with open(self.dataset_csv_path, 'a') as f:
                        f.write(csv_row)
                    
                    self.dataset_frame_count += 1
                
                # Draw visualization
                display_frame = frame.copy()
                
                if show_rois:
                    self.draw_rois(display_frame)
                
                self.draw_results(display_frame, show_overlay=show_overlay)
                
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
                    
                    cv2.imshow(debug_window_instruction, debug_img)
                
                # Show debug window with quality bar ROI
                if show_debug and self.last_quality_bar_roi is not None:
                    quality_debug_img = self.last_quality_bar_roi.copy()
                    
                    # Enlarge the narrow bar for visibility (scale width more than height)
                    h, w = quality_debug_img.shape[:2]
                    quality_debug_img = cv2.resize(quality_debug_img, (w * 10, h * 2), 
                                                   interpolation=cv2.INTER_LINEAR)
                    
                    # Add padding for text
                    padded = cv2.copyMakeBorder(quality_debug_img, 80, 30, 10, 10,
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    
                    # Add info text
                    quality_text = f"Quality Score: {self.last_results['quality']}%"
                    
                    cv2.putText(padded, quality_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    cv2.putText(padded, "Press 'd' to hide", (10, padded.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                    
                    cv2.imshow(debug_window_quality, padded)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nExiting...")
                    break
                
                elif key == ord('r'):  # Toggle ROI display
                    show_rois = not show_rois
                
                elif key == ord('d'):  # Toggle debug windows
                    show_debug = not show_debug
                    if not show_debug:
                        cv2.destroyWindow(debug_window_instruction)
                        cv2.destroyWindow(debug_window_quality)
                    else:
                        cv2.namedWindow(debug_window_instruction, cv2.WINDOW_NORMAL)
                        cv2.namedWindow(debug_window_quality, cv2.WINDOW_NORMAL)
                
                elif key == ord('o'):  # Toggle overlay
                    show_overlay = not show_overlay
                
                elif key == ord('s'):  # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    snapshot_path = snapshot_dir / f"snapshot_{timestamp}.png"
                    cv2.imwrite(str(snapshot_path), frame)
                    snapshot_count += 1
                    print(f"📸 Snapshot saved: {snapshot_path}")
                
                elif key == ord('w'):  # Toggle recording
                    if self.save_dataset:
                        is_recording = not is_recording
                        if is_recording:
                            print(f"🔴 Recording STARTED (frame {self.dataset_frame_count})")
                        else:
                            print(f"⏸️  Recording PAUSED (total frames: {self.dataset_frame_count})")
                    else:
                        print("⚠️  Dataset saving not enabled. Use --save-dataset flag.")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Finalize and save log with timestamp
            self.finalize_log()
            
            print()
            print("="*80)
            print("SESSION SUMMARY")
            print("="*80)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Snapshots saved: {snapshot_count}")
            print(f"Log file: {self.final_log_file}")
            if self.save_dataset:
                print(f"Dataset frames saved: {self.dataset_frame_count}")
                print(f"Dataset directory: {self.dataset_dir}")
                print(f"Dataset CSV: {self.dataset_csv_path}")
            print("="*80)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Real-time OCR on GE ultrasound video"
    )
    parser.add_argument('--device', type=str, default='/dev/video0',
                       help='Video device path')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='OCR processing interval in seconds')
    parser.add_argument('--save-dataset', action='store_true',
                       help='Save frames as dataset during real-time capture')
    parser.add_argument('--dataset-dir', type=str, default='realtime_dataset',
                       help='Directory to save dataset (default: realtime_dataset)')
    parser.add_argument('--robot-pose-mqtt', type=str, default=None,
                       help='MQTT broker URL for robot pose (e.g., mqtt://192.168.1.100:1883)')
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
    
    # Parse device argument - support both device number (16) and full path (/dev/video16)
    device_str = args.device
    if device_str.startswith('/dev/video'):
        # Full path provided - extract device number
        device_index = int(device_str.replace('/dev/video', ''))
    else:
        # Just device number provided
        try:
            device_index = int(device_str)
        except ValueError:
            print(f"❌ Error: Invalid device: {device_str}")
            print("   Use device number (e.g., 16) or full path (e.g., /dev/video16)")
            sys.exit(1)
    
    # Run real-time OCR
    ocr = RealtimeOCR(
        config_path=config_path,
        device_index=device_index,
        interval=args.interval,
        save_dataset=args.save_dataset,
        dataset_dir=args.dataset_dir,
        robot_pose_mqtt_url=args.robot_pose_mqtt
    )
    
    success = ocr.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
