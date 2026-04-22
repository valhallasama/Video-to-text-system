# Quick Start Guide - Video-to-Text System

## ✅ Installation Complete!

All dependencies have been installed and the system is ready to use.

## Running the Real-time OCR

### Basic Usage

```bash
cd /home/edler/cam_ws/Video-to-text-system-main
./run_realtime_ocr.sh
```

This will use `/dev/video0` by default with 1.0 second OCR interval.

### Specify Video Device

If your ultrasound video is on a different device (e.g., `/dev/video15`):

```bash
./run_realtime_ocr.sh 15
```

### Adjust OCR Interval

To process OCR every 0.5 seconds instead of 1.0:

```bash
./run_realtime_ocr.sh 15 0.5
```

## Controls During Runtime

- **`q` or `ESC`** - Quit the application
- **`r`** - Toggle ROI (Region of Interest) display boxes
- **`d`** - Toggle debug window showing preprocessed text
- **`s`** - Save a snapshot of the current frame

## Output Files

- **`realtime_ocr_log.txt`** - Timestamped log of all OCR detections
- **`snapshots/`** - Directory containing saved snapshots (created when you press 's')

## Configuration

Edit `config/default.yaml` to adjust:
- ROI coordinates (normalized 0-1 relative to frame size)
- Frame sampling rate
- OCR parameters (engine, confidence thresholds)
- Preprocessing settings (CLAHE, denoising, sharpening)
- Temporal aggregation (smoothing across frames)

## Troubleshooting

### Camera not opening
- Check available video devices: `ls /dev/video*`
- Try different device numbers: `./run_realtime_ocr.sh 0`, `./run_realtime_ocr.sh 1`, etc.

### Low OCR accuracy
- Adjust ROI coordinates in `config/default.yaml` to match your UI layout
- Increase preprocessing contrast: edit `preprocessing.clahe.clip_limit`
- Lower confidence threshold: edit `ocr.min_confidence`

### Performance issues
- Increase OCR interval: `./run_realtime_ocr.sh 15 2.0` (process every 2 seconds)
- Disable debug window: press `d` during runtime
- Set `single_pass: true` in config for 3x faster OCR (slightly lower accuracy)

## Manual Activation (if needed)

If you prefer to run commands manually:

```bash
cd /home/edler/cam_ws/Video-to-text-system-main
source .venv_video2text/bin/activate
python scripts/realtime_ocr.py --device 15 --interval 1.0 --config config/default.yaml
```

## What's Installed

- ✅ Tesseract OCR 4.1.1
- ✅ Python virtual environment (`.venv_video2text`)
- ✅ OpenCV 4.13.0
- ✅ PaddleOCR 3.4.0 (optional, can use Tesseract instead)
- ✅ NumPy, PyYAML, Pillow, Shapely
- ✅ All project dependencies

## Next Steps

1. Connect your ultrasound video source
2. Find the correct `/dev/videoN` device number
3. Run: `./run_realtime_ocr.sh N`
4. Adjust ROI boxes in config if needed for your specific UI layout
