#!/usr/bin/env python3
"""
Visualize and adjust ROI areas on ultrasound video.

This script displays the ROI boxes for instruction text, quality bar, and ultrasound fan
on video frames. You can interactively adjust the ROI positions and sizes.

Controls:
- Arrow keys: Move selected ROI
- +/-: Increase/decrease ROI size
- 1/2/3: Select ROI (1=instruction, 2=quality, 3=ultrasound)
- s: Save current ROI configuration
- q: Quit
- Space: Pause/Resume
"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path


class ROIVisualizer:
    """Visualize and adjust ROI areas on video."""
    
    def __init__(self, video_path):
        """Initialize visualizer."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # ROI configurations (normalized 0-1 coordinates)
        self.roi_configs = {
            'instruction': {
                'name': 'Instruction Text',
                'x': 0.59,
                'y': 0.045,
                'w': 0.40,
                'h': 0.06,
                'color': (0, 255, 0),  # Green
                'key': '1'
            },
            'quality_bar': {
                'name': 'Quality Bar',
                'x': 0.5995,
                'y': 0.1836,
                'w': 0.010,
                'h': 0.3288,
                'color': (255, 0, 255),  # Magenta
                'key': '2'
            },
            'ultrasound_fan': {
                'name': 'Ultrasound Fan',
                'x': 0.05,
                'y': 0.15,
                'w': 0.50,
                'h': 0.70,
                'color': (0, 255, 255),  # Cyan
                'key': '3'
            }
        }
        
        self.selected_roi = 'instruction'
        self.paused = False
        self.current_frame_idx = 0
        
    def get_roi_bbox(self, roi_config):
        """Convert normalized ROI to pixel coordinates."""
        x = int(roi_config['x'] * self.width)
        y = int(roi_config['y'] * self.height)
        w = int(roi_config['w'] * self.width)
        h = int(roi_config['h'] * self.height)
        return x, y, w, h
    
    def draw_rois(self, frame):
        """Draw all ROI boxes on frame."""
        display = frame.copy()
        
        for roi_name, roi_config in self.roi_configs.items():
            x, y, w, h = self.get_roi_bbox(roi_config)
            color = roi_config['color']
            
            # Thicker border for selected ROI
            thickness = 3 if roi_name == self.selected_roi else 2
            
            # Draw rectangle
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            label = f"{roi_config['name']} [{roi_config['key']}]"
            if roi_name == self.selected_roi:
                label += " *SELECTED*"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
            cv2.putText(display, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Show coordinates
            coord_text = f"x:{roi_config['x']:.3f} y:{roi_config['y']:.3f} w:{roi_config['w']:.3f} h:{roi_config['h']:.3f}"
            cv2.putText(display, coord_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "Arrow keys: Move ROI",
            "+/-: Resize width",
            "w/s: Resize height",
            "1/2/3: Select ROI",
            "Space: Pause/Resume",
            "c: Save config",
            "q: Quit"
        ]
        
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(display, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame info
        frame_info = f"Frame: {self.current_frame_idx}/{self.total_frames}"
        cv2.putText(display, frame_info, (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display
    
    def adjust_roi(self, key):
        """Adjust selected ROI based on key press."""
        roi = self.roi_configs[self.selected_roi]
        step = 0.005  # Adjustment step
        
        # Arrow keys - move position
        if key == 82 or key == 0:  # Up
            roi['y'] = max(0, roi['y'] - step)
        elif key == 84 or key == 1:  # Down
            roi['y'] = min(1 - roi['h'], roi['y'] + step)
        elif key == 81 or key == 2:  # Left
            roi['x'] = max(0, roi['x'] - step)
        elif key == 83 or key == 3:  # Right
            roi['x'] = min(1 - roi['w'], roi['x'] + step)
        
        # +/- keys - adjust width
        elif key == ord('+') or key == ord('='):
            roi['w'] = min(1 - roi['x'], roi['w'] + step)
        elif key == ord('-') or key == ord('_'):
            roi['w'] = max(0.01, roi['w'] - step)
        
        # w/s keys - adjust height
        elif key == ord('w') or key == ord('W'):
            roi['h'] = min(1 - roi['y'], roi['h'] + step)
        elif key == ord('s') or key == ord('S'):
            roi['h'] = max(0.01, roi['h'] - step)
    
    def save_config(self, output_path='roi_config.json'):
        """Save current ROI configuration to JSON file."""
        config = {}
        for roi_name, roi_config in self.roi_configs.items():
            config[roi_name] = {
                'x': roi_config['x'],
                'y': roi_config['y'],
                'w': roi_config['w'],
                'h': roi_config['h']
            }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ ROI configuration saved to: {output_path}")
        print("\nCopy these values to extract_dataset.py:")
        print("=" * 60)
        for roi_name, roi_config in self.roi_configs.items():
            print(f"'{roi_name}': {{")
            print(f"    'x': {roi_config['x']:.4f},")
            print(f"    'y': {roi_config['y']:.4f},")
            print(f"    'w': {roi_config['w']:.4f},")
            print(f"    'h': {roi_config['h']:.4f},")
            print("},")
        print("=" * 60)
    
    def run(self, start_frame=0):
        """Run the visualizer."""
        print(f"Video: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print("\nControls:")
        print("  Arrow keys: Move selected ROI")
        print("  +/-: Adjust width")
        print("  w/s: Adjust height")
        print("  1/2/3: Select ROI (1=instruction, 2=quality, 3=ultrasound)")
        print("  Space: Pause/Resume")
        print("  c: Save configuration")
        print("  q: Quit")
        print()
        
        # Jump to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.current_frame_idx = start_frame
        
        window_name = "ROI Visualizer - Adjust ROI Areas"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                
                if not ret:
                    # Loop back to start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    continue
                
                self.current_frame_idx += 1
                self.current_frame = frame
            
            # Draw ROIs
            display = self.draw_rois(self.current_frame)
            cv2.imshow(window_name, display)
            
            # Handle key press (always use small delay to allow quit to work)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord(' '):  # Space - pause/resume
                self.paused = not self.paused
                print(f"{'Paused' if self.paused else 'Resumed'}")
            elif key == ord('1'):
                self.selected_roi = 'instruction'
                print(f"Selected: {self.roi_configs['instruction']['name']}")
            elif key == ord('2'):
                self.selected_roi = 'quality_bar'
                print(f"Selected: {self.roi_configs['quality_bar']['name']}")
            elif key == ord('3'):
                self.selected_roi = 'ultrasound_fan'
                print(f"Selected: {self.roi_configs['ultrasound_fan']['name']}")
            elif key == ord('c') or key == ord('C'):
                self.save_config()
            elif key != 255:  # Any other key
                self.adjust_roi(key)
        
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize and adjust ROI areas')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('-f', '--frame', type=int, default=500,
                       help='Start frame number (default: 500)')
    
    args = parser.parse_args()
    
    visualizer = ROIVisualizer(args.video)
    visualizer.run(start_frame=args.frame)


if __name__ == '__main__':
    main()
