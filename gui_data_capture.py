#!/usr/bin/env python3
"""
GUI Data Capture Application
User interface with start/stop buttons to control robot training dataset capture.
"""

import os
# Set OpenCV to use headless backend before importing
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import time
import yaml
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# Import cv2 with headless backend
import cv2

# Now import PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QFileDialog, QGroupBox, QGridLayout, QTextEdit)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from video2text.core.preprocessor import Preprocessor
from video2text.core.ocr_engine import OCREngine
from video2text.core.postprocessor import TextPostprocessor
from video2text.utils.robot_pose_mqtt_client import RobotPoseMQTTClient


class QualityBarAnalyzer:
    """Analyzer for vertical quality bar with gray/white/green segments."""
    
    def analyze_quality_bar(self, roi):
        """Analyze quality bar and return percentage score."""
        if roi is None or roi.size == 0:
            return 0
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        overall_brightness = np.mean(gray)
        if overall_brightness < 20:
            return 0
        
        height, width = gray.shape
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
        
        bright_threshold = 60
        bright_rows = 0
        for y in range(bar_start, bar_end + 1):
            row_brightness = np.mean(gray[y, :])
            if row_brightness > bright_threshold:
                bright_rows += 1
        
        quality_percentage = int((bright_rows / bar_height) * 100)
        quality_percentage += 1
        
        return min(quality_percentage, 99)


class DataCaptureWorker(QThread):
    """Worker thread for data capture to avoid blocking GUI."""
    
    frame_ready = pyqtSignal(np.ndarray, str, int)  # frame, instruction, quality
    status_update = pyqtSignal(str)
    frame_saved = pyqtSignal(int)  # frame number
    
    def __init__(self, device_index, config_path, dataset_dir, robot_pose_url=None):
        super().__init__()
        self.device_index = device_index
        self.config_path = config_path
        self.dataset_dir = Path(dataset_dir)
        self.robot_pose_url = robot_pose_url
        
        self.is_capturing = False
        self.is_running = True
        self.frame_count = 0
        
        # Initialize components
        self.cap = None
        self.ocr_engine = None
        self.postprocessor = None
        self.quality_analyzer = QualityBarAnalyzer()
        self.robot_pose_client = None
        
        # ROI coordinates (from config)
        self.instruction_roi = None
        self.quality_roi = None
        self.ultrasound_roi = None
        
    def initialize(self):
        """Initialize video capture and OCR components."""
        try:
            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get ROI coordinates
            self.instruction_roi = config['roi']['instruction']
            self.quality_roi = config['roi']['quality']
            self.ultrasound_roi = config['roi']['ultrasound']
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                self.status_update.emit(f"❌ Failed to open video device {self.device_index}")
                return False
            
            # Initialize OCR
            self.ocr_engine = OCREngine(config['ocr'])
            self.postprocessor = TextPostprocessor(config['postprocessor'])
            
            # Initialize robot pose client if URL provided
            if self.robot_pose_url:
                try:
                    self.robot_pose_client = RobotPoseMQTTClient(self.robot_pose_url)
                    self.robot_pose_client.connect()
                    self.status_update.emit(f"✅ Connected to robot pose MQTT: {self.robot_pose_url}")
                except Exception as e:
                    self.status_update.emit(f"⚠️ Robot pose connection failed: {e}")
                    self.robot_pose_client = None
            
            self.status_update.emit("✅ Initialization complete")
            return True
            
        except Exception as e:
            self.status_update.emit(f"❌ Initialization error: {e}")
            return False
    
    def start_capture(self):
        """Start capturing data."""
        if not self.is_capturing:
            # Create dataset directories
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / 'frames').mkdir(exist_ok=True)
            (self.dataset_dir / 'original_frames').mkdir(exist_ok=True)
            
            # Create CSV file with header
            csv_path = self.dataset_dir / 'dataset.csv'
            if not csv_path.exists():
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'frame_number', 'instruction_text', 'quality_score', 'image_path',
                        'robot_x', 'robot_y', 'robot_z',
                        'robot_qx', 'robot_qy', 'robot_qz', 'robot_qw',
                        'roll_deg', 'pitch_deg', 'yaw_deg'
                    ])
                self.frame_count = 0
            else:
                # Count existing frames
                with open(csv_path, 'r') as f:
                    self.frame_count = sum(1 for _ in f) - 1  # Subtract header
            
            self.is_capturing = True
            self.status_update.emit(f"🔴 Recording started (frame {self.frame_count})")
    
    def stop_capture(self):
        """Stop capturing data."""
        if self.is_capturing:
            self.is_capturing = False
            self.status_update.emit(f"⏹️ Recording stopped (total frames: {self.frame_count})")
    
    def run(self):
        """Main worker loop."""
        if not self.initialize():
            return
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Extract ROIs
            x1, y1, x2, y2 = self.instruction_roi
            instruction_roi = frame[y1:y2, x1:x2]
            
            x1, y1, x2, y2 = self.quality_roi
            quality_roi = frame[y1:y2, x1:x2]
            
            x1, y1, x2, y2 = self.ultrasound_roi
            ultrasound_roi = frame[y1:y2, x1:x2]
            
            # Process instruction text
            instruction_text = self.process_instruction(instruction_roi)
            
            # Analyze quality
            quality_score = self.quality_analyzer.analyze_quality_bar(quality_roi)
            
            # Emit frame for display
            self.frame_ready.emit(frame.copy(), instruction_text, quality_score)
            
            # Save frame if capturing
            if self.is_capturing:
                self.save_frame(frame, ultrasound_roi, instruction_text, quality_score)
            
            time.sleep(0.01)  # ~100 FPS max
    
    def process_instruction(self, roi):
        """Process instruction ROI with OCR."""
        try:
            # Custom preprocessing for instruction text
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(upscaled)
            
            # OCR
            results = self.ocr_engine.recognize(enhanced)
            
            # Process each word individually
            words = []
            for result in results:
                text = result[1][0] if isinstance(result[1], tuple) else result[1]
                processed = self.postprocessor.process(text)
                if processed and len(processed) >= 2:
                    words.append(processed)
            
            return ' '.join(words) if words else ''
            
        except Exception as e:
            return ''
    
    def save_frame(self, frame, ultrasound_roi, instruction_text, quality_score):
        """Save frame data to dataset."""
        try:
            # Save ultrasound ROI
            frame_filename = f"frame_{self.frame_count:06d}.png"
            frame_path = self.dataset_dir / 'frames' / frame_filename
            cv2.imwrite(str(frame_path), ultrasound_roi)
            
            # Save original frame with annotations
            original_filename = f"original_{self.frame_count:06d}.png"
            original_path = self.dataset_dir / 'original_frames' / original_filename
            annotated_frame = self.draw_annotations(frame.copy(), instruction_text, quality_score)
            cv2.imwrite(str(original_path), annotated_frame)
            
            # Get robot pose
            robot_pose = self.get_robot_pose()
            
            # Append to CSV
            csv_path = self.dataset_dir / 'dataset.csv'
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.frame_count,
                    instruction_text,
                    quality_score,
                    f"frames/{frame_filename}",
                    robot_pose['x'], robot_pose['y'], robot_pose['z'],
                    robot_pose['qx'], robot_pose['qy'], robot_pose['qz'], robot_pose['qw'],
                    robot_pose['roll'], robot_pose['pitch'], robot_pose['yaw']
                ])
            
            self.frame_saved.emit(self.frame_count)
            self.frame_count += 1
            
        except Exception as e:
            self.status_update.emit(f"❌ Save error: {e}")
    
    def get_robot_pose(self):
        """Get current robot pose from MQTT client."""
        if self.robot_pose_client:
            pose = self.robot_pose_client.get_current_pose()
            if pose:
                return pose
        
        # Return zeros if no robot pose available
        return {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    
    def draw_annotations(self, frame, instruction_text, quality_score):
        """Draw annotations on frame."""
        # Draw ROI boxes
        for roi_name, coords in [
            ('Instruction', self.instruction_roi),
            ('Quality', self.quality_roi),
            ('Ultrasound', self.ultrasound_roi)
        ]:
            x1, y1, x2, y2 = coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, roi_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
        
        # Draw frame number and quality
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Quality: {quality_score}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def stop(self):
        """Stop the worker thread."""
        self.is_running = False
        self.stop_capture()
        if self.cap:
            self.cap.release()
        if self.robot_pose_client:
            self.robot_pose_client.disconnect()


class DataCaptureGUI(QMainWindow):
    """Main GUI window for data capture control."""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Robot Training Dataset Capture")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Video display
        left_panel = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.video_label)
        
        # Info display
        info_layout = QHBoxLayout()
        self.instruction_label = QLabel("Instruction: --")
        self.instruction_label.setFont(QFont("Arial", 12))
        self.quality_label = QLabel("Quality: --")
        self.quality_label.setFont(QFont("Arial", 12, QFont.Bold))
        info_layout.addWidget(self.instruction_label)
        info_layout.addStretch()
        info_layout.addWidget(self.quality_label)
        left_panel.addLayout(info_layout)
        
        main_layout.addLayout(left_panel, 3)
        
        # Right panel: Controls
        right_panel = QVBoxLayout()
        
        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout()
        
        config_layout.addWidget(QLabel("Video Device:"), 0, 0)
        self.device_input = QLineEdit("16")
        config_layout.addWidget(self.device_input, 0, 1)
        
        config_layout.addWidget(QLabel("Config File:"), 1, 0)
        self.config_input = QLineEdit("config/roi_config.yaml")
        config_layout.addWidget(self.config_input, 1, 1)
        self.config_browse_btn = QPushButton("Browse")
        self.config_browse_btn.clicked.connect(self.browse_config)
        config_layout.addWidget(self.config_browse_btn, 1, 2)
        
        config_layout.addWidget(QLabel("Dataset Dir:"), 2, 0)
        self.dataset_input = QLineEdit("robot_training_dataset")
        config_layout.addWidget(self.dataset_input, 2, 1)
        self.dataset_browse_btn = QPushButton("Browse")
        self.dataset_browse_btn.clicked.connect(self.browse_dataset)
        config_layout.addWidget(self.dataset_browse_btn, 2, 2)
        
        config_layout.addWidget(QLabel("Robot MQTT:"), 3, 0)
        self.mqtt_input = QLineEdit("mqtt://192.168.56.2:1883")
        config_layout.addWidget(self.mqtt_input, 3, 1, 1, 2)
        
        config_group.setLayout(config_layout)
        right_panel.addWidget(config_group)
        
        # Control buttons
        control_group = QGroupBox("Recording Control")
        control_layout = QVBoxLayout()
        
        self.init_btn = QPushButton("Initialize System")
        self.init_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;")
        self.init_btn.clicked.connect(self.initialize_system)
        control_layout.addWidget(self.init_btn)
        
        self.start_btn = QPushButton("🔴 Start Recording")
        self.start_btn.setStyleSheet("background-color: #f44336; color: white; font-size: 16px; padding: 15px;")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_recording)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop Recording")
        self.stop_btn.setStyleSheet("background-color: #FF9800; color: white; font-size: 16px; padding: 15px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        control_layout.addWidget(self.stop_btn)
        
        control_group.setLayout(control_layout)
        right_panel.addWidget(control_group)
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        status_layout.addWidget(self.status_text)
        
        self.frame_count_label = QLabel("Frames Captured: 0")
        self.frame_count_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(self.frame_count_label)
        
        status_group.setLayout(status_layout)
        right_panel.addWidget(status_group)
        
        right_panel.addStretch()
        
        main_layout.addLayout(right_panel, 1)
        
        self.add_status("Ready to initialize...")
    
    def browse_config(self):
        """Browse for config file."""
        filename, _ = QFileDialog.getOpenFileName(self, "Select Config File", "", "YAML Files (*.yaml *.yml)")
        if filename:
            self.config_input.setText(filename)
    
    def browse_dataset(self):
        """Browse for dataset directory."""
        dirname = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dirname:
            self.dataset_input.setText(dirname)
    
    def initialize_system(self):
        """Initialize the data capture system."""
        try:
            device = int(self.device_input.text())
            config_path = self.config_input.text()
            dataset_dir = self.dataset_input.text()
            mqtt_url = self.mqtt_input.text() if self.mqtt_input.text() else None
            
            self.add_status("Initializing system...")
            
            # Create worker thread
            self.worker = DataCaptureWorker(device, config_path, dataset_dir, mqtt_url)
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.status_update.connect(self.add_status)
            self.worker.frame_saved.connect(self.update_frame_count)
            
            # Start worker
            self.worker.start()
            
            # Enable controls
            self.start_btn.setEnabled(True)
            self.init_btn.setEnabled(False)
            
        except Exception as e:
            self.add_status(f"❌ Initialization failed: {e}")
    
    def start_recording(self):
        """Start recording data."""
        if self.worker:
            self.worker.start_capture()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
    
    def stop_recording(self):
        """Stop recording data."""
        if self.worker:
            self.worker.stop_capture()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def update_frame(self, frame, instruction, quality):
        """Update video display."""
        # Convert frame to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update info
        self.instruction_label.setText(f"Instruction: {instruction if instruction else '--'}")
        
        # Color code quality
        if quality >= 90:
            color = "green"
        elif quality >= 50:
            color = "orange"
        else:
            color = "red"
        self.quality_label.setText(f"Quality: {quality}%")
        self.quality_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def update_frame_count(self, count):
        """Update frame count display."""
        self.frame_count_label.setText(f"Frames Captured: {count}")
    
    def add_status(self, message):
        """Add message to status log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    gui = DataCaptureGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
