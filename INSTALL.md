# Installation Guide - Video-to-Text Ultrasound Guidance System

Complete step-by-step installation instructions for deploying this system on a new PC.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Hardware Setup](#hardware-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Operating System
- **Ubuntu 20.04 LTS or later** (recommended)
- **Ubuntu 22.04 LTS** (tested)
- Other Linux distributions may work but are not officially supported

### Hardware
- **CPU**: Intel i5 or better (i7+ recommended for real-time processing)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space for installation + dataset storage
- **USB Video Capture Device**: Compatible with V4L2 (Video4Linux2)
  - Example: Elgato Cam Link, Magewell USB Capture, or similar
  - Must support MJPEG format at 1920x1080 resolution

### Software Dependencies
- Python 3.8 - 3.10 (Python 3.10 recommended)
- Tesseract OCR 4.0+
- OpenCV 4.8+
- Video4Linux2 (v4l2) drivers

---

## Installation Steps

### Step 1: System Preparation

Update your system packages:
```bash
sudo apt update
sudo apt upgrade -y
```

### Step 2: Install System Dependencies

Install required system packages:
```bash
# Install Python and development tools
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install Tesseract OCR
sudo apt install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev

# Install video and image processing libraries
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y v4l-utils ffmpeg

# Install build tools (required for some Python packages)
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Install Git (if not already installed)
sudo apt install -y git
```

Verify Tesseract installation:
```bash
tesseract --version
# Expected output: tesseract 4.x.x or later
```

### Step 3: Clone the Repository

Clone the project from GitHub:
```bash
cd ~
mkdir -p cam_ws
cd cam_ws

# Clone the repository
git clone https://github.com/valhallasama/Video-to-text-system.git
cd Video-to-text-system

# Checkout the latest branch (if needed)
git checkout 15-04-2026
```

Or if you're copying from another PC:
```bash
# On source PC, create a tarball
cd /home/edler/cam_ws
tar -czf Video-to-text-system-main.tar.gz Video-to-text-system-main/

# Transfer to new PC (via USB, scp, or network share)
# On new PC:
cd ~/cam_ws
tar -xzf Video-to-text-system-main.tar.gz
cd Video-to-text-system-main
```

### Step 4: Create Python Virtual Environment

Create and activate a virtual environment:
```bash
cd ~/cam_ws/Video-to-text-system-main

# Create virtual environment
python3 -m venv .venv_video2text

# Activate virtual environment
source .venv_video2text/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**Important:** Always activate the virtual environment before running scripts:
```bash
source .venv_video2text/bin/activate
```

### Step 5: Install Python Dependencies

Install all required Python packages:
```bash
# Make sure virtual environment is activated
source .venv_video2text/bin/activate

# Install dependencies
pip install -r requirements.txt

# This will install:
# - opencv-python (computer vision)
# - paddlepaddle (deep learning framework)
# - paddleocr (OCR engine)
# - numpy (numerical computing)
# - pyyaml (configuration files)
# - pillow (image processing)
# - shapely (geometric operations)
```

**Note:** PaddlePaddle installation may take 5-10 minutes depending on your internet speed.

### Step 6: Verify Installation

Test that all dependencies are installed correctly:
```bash
# Activate virtual environment
source .venv_video2text/bin/activate

# Test imports
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import paddle; print(f'PaddlePaddle: {paddle.__version__}')"
python3 -c "import paddleocr; print('PaddleOCR: OK')"
python3 -c "import yaml; print('PyYAML: OK')"

# Test Tesseract
python3 -c "import pytesseract; print('Tesseract: OK')"
```

Expected output (versions may vary):
```
OpenCV: 4.8.0
NumPy: 1.24.3
PaddlePaddle: 2.5.2
PaddleOCR: OK
PyYAML: OK
Tesseract: OK
```

### Step 7: Make Scripts Executable

Set execute permissions on shell scripts:
```bash
chmod +x run_realtime_ocr.sh
chmod +x extract_dataset.sh
chmod +x visualize_rois.sh
```

---

## Hardware Setup

### Connect Video Capture Device

1. **Connect USB Video Capture Device**
   - Plug USB capture device into your PC
   - Connect GE ultrasound HDMI output to capture device input

2. **Verify Video Device**
   ```bash
   # List all video devices
   ls -la /dev/video*
   
   # Expected output:
   # /dev/video0
   # /dev/video1
   # /dev/video14
   # /dev/video15
   # /dev/video16  <- Your capture device (number may vary)
   # /dev/video17
   ```

3. **Identify Your Capture Device**
   ```bash
   # Check device capabilities
   v4l2-ctl --list-devices
   
   # Example output:
   # USB Video: USB Video (usb-0000:00:14.0-1):
   #     /dev/video16
   #     /dev/video17
   ```

4. **Test Video Stream**
   ```bash
   # Test with ffplay (if installed)
   ffplay /dev/video16
   
   # Or test with Python
   python3 -c "
   import cv2
   cap = cv2.VideoCapture(16)
   if cap.isOpened():
       print('✅ Video device /dev/video16 is working!')
       ret, frame = cap.read()
       if ret:
           print(f'   Resolution: {frame.shape[1]}x{frame.shape[0]}')
   else:
       print('❌ Cannot open /dev/video16')
   cap.release()
   "
   ```

5. **Set Correct Device Number**
   - Note your device number (e.g., 16)
   - Use this number when running scripts

---

## Verification

### Test Real-time OCR

Run the real-time OCR script to verify everything works:

```bash
# Activate virtual environment
source .venv_video2text/bin/activate

# Run real-time OCR (replace 16 with your device number)
./run_realtime_ocr.sh 16
```

**Expected behavior:**
- Window opens showing ultrasound video feed
- Green box around instruction text area
- Magenta box around quality bar
- Cyan box around ultrasound fan
- OCR output printed to terminal when text/quality changes

**Controls:**
- Press `r` to toggle ROI boxes
- Press `d` to toggle debug window
- Press `o` to toggle overlay
- Press `q` to quit

### Test Dataset Extraction

Extract a small dataset from a video file:

```bash
# Activate virtual environment
source .venv_video2text/bin/activate

# Extract dataset (replace with your video path)
./extract_dataset.sh "/path/to/test_video.webm" test_dataset 50 --save-original
```

**Expected output:**
```
test_dataset/
├── frames/              # Ultrasound ROI images
├── original_frames/     # Full frames with annotations
└── dataset.csv         # Metadata
```

### Test ROI Visualization

Verify ROI positions on your video:

```bash
# Activate virtual environment
source .venv_video2text/bin/activate

# Visualize ROIs (replace with your video path)
./visualize_rois.sh "/path/to/test_video.webm" 500
```

**Expected behavior:**
- Video plays with colored ROI boxes
- Arrow keys move selected ROI
- +/- keys adjust ROI size
- Press `c` to save configuration

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**
```bash
# Make sure virtual environment is activated
source .venv_video2text/bin/activate

# Reinstall opencv-python
pip install --force-reinstall opencv-python
```

### Issue: "Cannot open /dev/videoX"

**Possible causes:**
1. **Wrong device number**
   ```bash
   # List all devices
   ls -la /dev/video*
   
   # Try different device numbers
   ./run_realtime_ocr.sh 0
   ./run_realtime_ocr.sh 14
   ./run_realtime_ocr.sh 16
   ```

2. **Permission denied**
   ```bash
   # Add your user to video group
   sudo usermod -a -G video $USER
   
   # Log out and log back in for changes to take effect
   ```

3. **Device in use**
   ```bash
   # Check if another process is using the device
   fuser /dev/video16
   
   # Kill the process if needed
   sudo fuser -k /dev/video16
   ```

### Issue: "Tesseract not found"

**Solution:**
```bash
# Install Tesseract
sudo apt install -y tesseract-ocr tesseract-ocr-eng

# Verify installation
tesseract --version
which tesseract
```

### Issue: Poor OCR Results

**Solutions:**

1. **Adjust ROI positions**
   ```bash
   ./visualize_rois.sh "/path/to/video.webm" 500
   # Use arrow keys to adjust ROI boxes
   # Press 'c' to save configuration
   ```

2. **Check video quality**
   - Ensure HDMI connection is secure
   - Verify capture device supports 1920x1080 resolution
   - Check that ultrasound display is not scaled/cropped

3. **Verify ROI coordinates**
   - Instruction text should be fully within green box
   - Quality bar should be fully within magenta box
   - No overlap with other UI elements

### Issue: "PaddlePaddle installation failed"

**Solution:**
```bash
# Install CPU-only version (smaller, faster install)
pip install paddlepaddle==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Or use official PyPI
pip install paddlepaddle==2.5.2
```

### Issue: Virtual environment activation fails

**Solution:**
```bash
# Recreate virtual environment
rm -rf .venv_video2text
python3 -m venv .venv_video2text
source .venv_video2text/bin/activate
pip install -r requirements.txt
```

### Issue: Scripts not executable

**Solution:**
```bash
chmod +x run_realtime_ocr.sh
chmod +x extract_dataset.sh
chmod +x visualize_rois.sh
```

---

## Post-Installation Configuration

### Adjust ROI Coordinates

If the default ROI positions don't match your ultrasound layout:

1. **Use ROI Visualizer**
   ```bash
   ./visualize_rois.sh "/path/to/video.webm" 500
   ```

2. **Adjust positions**
   - Press `1` to select instruction ROI (green)
   - Press `2` to select quality bar ROI (magenta)
   - Press `3` to select ultrasound fan ROI (cyan)
   - Use arrow keys to move
   - Use `+/-` to adjust width
   - Use `w/s` to adjust height

3. **Save configuration**
   - Press `c` to save to `roi_config.json`
   - Configuration will be printed to terminal

4. **Update scripts**
   - Copy the printed coordinates
   - Update `scripts/realtime_ocr.py` and `scripts/extract_dataset.py`
   - Or use the saved `roi_config.json` file

### Create Digit Templates (Optional)

For improved score recognition:

```bash
# Activate virtual environment
source .venv_video2text/bin/activate

# Run template creator
python3 scripts/create_digit_templates.py --device 16

# When each digit (0-9) appears clearly on screen, press the corresponding key
# Templates will be saved to templates/ directory
```

---

## Quick Reference

### Daily Usage

```bash
# 1. Navigate to project directory
cd ~/cam_ws/Video-to-text-system-main

# 2. Activate virtual environment
source .venv_video2text/bin/activate

# 3. Run real-time OCR
./run_realtime_ocr.sh 16

# 4. Extract dataset from video
./extract_dataset.sh "/path/to/video.webm" output_dir 50 --save-original
```

### Common Commands

```bash
# List video devices
ls -la /dev/video*

# Test video device
python3 -c "import cv2; cap = cv2.VideoCapture(16); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"

# Check Tesseract version
tesseract --version

# Activate virtual environment
source .venv_video2text/bin/activate

# Deactivate virtual environment
deactivate
```

---

## System Architecture

For detailed information about how the system works, see [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).

For usage examples and quick start guide, see [README.md](README.md).

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [README.md](README.md) for usage examples
3. Verify all dependencies are installed correctly
4. Check system logs: `logs/` directory
5. Test with a known-good video file first

---

## Summary Checklist

- [ ] Ubuntu 20.04+ installed
- [ ] Python 3.8-3.10 installed
- [ ] Tesseract OCR installed
- [ ] System dependencies installed (opencv, v4l-utils, etc.)
- [ ] Repository cloned or copied
- [ ] Virtual environment created and activated
- [ ] Python dependencies installed (requirements.txt)
- [ ] Scripts made executable
- [ ] USB video capture device connected
- [ ] Video device identified (/dev/videoX)
- [ ] Real-time OCR tested successfully
- [ ] ROI positions verified and adjusted if needed

**Installation complete!** You're ready to use the Video-to-Text Ultrasound Guidance System.
