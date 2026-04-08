#!/usr/bin/env python3
"""Create digit templates from the score display for template matching."""

import cv2
import numpy as np
from pathlib import Path

def main():
    """Capture score display and create digit templates."""
    
    print("Opening /dev/video0...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open video device")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video: {width}x{height}\n")
    print("=" * 70)
    print("DIGIT TEMPLATE CREATOR")
    print("=" * 70)
    print()
    print("This will help you create templates for digits 0-9")
    print("from the score display on your ultrasound system.")
    print()
    print("Instructions:")
    print("1. Adjust the score on your ultrasound to show different digits")
    print("2. Press the number key (0-9) to save that digit as a template")
    print("3. Press 'q' when done")
    print()
    print("Example: If score shows '12', press '1' then '2'")
    print()
    
    # ROI for score
    roi_x = int(0.2361 * width)
    roi_y = int(0.9345 * height)
    roi_w = int(0.0103 * width)  # Narrower to fit just the two digits
    roi_h = int(0.0141 * height)
    
    # Create templates directory
    template_dir = Path(__file__).resolve().parents[1] / "templates"
    template_dir.mkdir(exist_ok=True)
    
    saved_templates = set()
    
    cv2.namedWindow("Template Creator", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Extract ROI
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Simple preprocessing - clean binary
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Use lower threshold to capture white digits on dark background
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # Digits should already be white on black, check if inversion needed
        if np.mean(thresh) > 127:  # If mostly white, invert
            thresh = cv2.bitwise_not(thresh)
        
        # Split into two digits
        h, w = thresh.shape
        digit_w = w // 2
        
        digit1 = thresh[:, :digit_w]
        digit2 = thresh[:, digit_w:]
        
        # Create display
        display = frame.copy()
        
        # Draw ROI
        cv2.rectangle(display, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
        
        # Show enlarged binary digits
        digit1_large = cv2.resize(digit1, None, fx=15, fy=15, interpolation=cv2.INTER_NEAREST)
        digit2_large = cv2.resize(digit2, None, fx=15, fy=15, interpolation=cv2.INTER_NEAREST)
        
        # Convert to color for display
        digit1_color = cv2.cvtColor(digit1_large, cv2.COLOR_GRAY2BGR)
        digit2_color = cv2.cvtColor(digit2_large, cv2.COLOR_GRAY2BGR)
        
        # Place digits on display
        h1, w1 = digit1_large.shape
        h2, w2 = digit2_large.shape
        
        y_offset = 100
        x_offset1 = 50
        x_offset2 = x_offset1 + w1 + 50
        
        if y_offset + max(h1, h2) < height:
            display[y_offset:y_offset+h1, x_offset1:x_offset1+w1] = digit1_color
            display[y_offset:y_offset+h2, x_offset2:x_offset2+w2] = digit2_color
            
            cv2.putText(display, "Digit 1", (x_offset1, y_offset-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Digit 2", (x_offset2, y_offset-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Show saved templates
        info_y = 50
        cv2.putText(display, f"Saved templates: {sorted(saved_templates)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, "Press 0-9 to save digit template, 'q' to quit", (10, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Template Creator", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('9'):
            digit_num = chr(key)
            
            # Ask which digit to save (1 or 2)
            print(f"\nSave digit '{digit_num}' - which position?")
            print("  Press '1' for left digit")
            print("  Press '2' for right digit")
            print("  Press any other key to cancel")
            
            # Wait for position selection
            while True:
                pos_key = cv2.waitKey(0) & 0xFF
                
                if pos_key == ord('1'):
                    # Save left digit
                    template_path = template_dir / f"{digit_num}.png"
                    cv2.imwrite(str(template_path), digit1)
                    saved_templates.add(digit_num)
                    print(f"✅ Saved template for digit '{digit_num}' (left position)")
                    break
                elif pos_key == ord('2'):
                    # Save right digit
                    template_path = template_dir / f"{digit_num}.png"
                    cv2.imwrite(str(template_path), digit2)
                    saved_templates.add(digit_num)
                    print(f"✅ Saved template for digit '{digit_num}' (right position)")
                    break
                else:
                    print("Cancelled")
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("TEMPLATE CREATION SUMMARY")
    print("=" * 70)
    print(f"Saved templates: {sorted(saved_templates)}")
    print(f"Missing templates: {set('0123456789') - saved_templates}")
    print(f"Template directory: {template_dir}")
    print()
    
    if len(saved_templates) == 10:
        print("✅ All digit templates created!")
        print("   You can now use template matching for score detection.")
    else:
        print("⚠️  Not all templates created yet.")
        print("   Run this script again to create missing templates.")

if __name__ == "__main__":
    main()
