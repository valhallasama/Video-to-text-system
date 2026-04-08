# Video-to-Text Ultrasound Guidance System

Automatically extract probe guidance instructions from GE ultrasound UI videos.

## Architecture

```
Video Stream → Frame Extraction → Preprocessing → ROI Detection → 
Text Detection → OCR Recognition → Text Postprocessing → 
Temporal Aggregation → Structured Instructions
```

## Features

- **Frame Extraction**: Adaptive sampling (5-10 FPS)
- **Preprocessing**: CLAHE, denoising, upscaling for low-contrast UI text
- **ROI Detection**: Fixed or adaptive region detection
- **Text Detection**: Optional CRAFT/DBNet integration
- **OCR**: PaddleOCR with confidence scoring
- **Temporal Smoothing**: Majority voting across frames
- **Structured Output**: Parse text into robot commands

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Offline Processing

```bash
python scripts/extract_instructions.py \
  --video /path/to/ultrasound_video.mp4 \
  --config config/default.yaml \
  --output results.json
```

### Real-time Processing

```bash
python scripts/realtime_capture.py \
  --source 0 \
  --config config/default.yaml
```

### Visualization/Debug

```bash
python scripts/visualize_pipeline.py \
  --video /path/to/video.mp4 \
  --output debug_frames/
```

## Configuration

Edit `config/default.yaml` to adjust:
- ROI coordinates (normalized 0-1)
- Frame sampling rate
- OCR parameters
- Temporal aggregation settings
- Instruction parsing rules

## Project Structure

```
video2text/
├── core/
│   ├── frame_extractor.py      # Video frame sampling
│   ├── preprocessor.py          # Image enhancement
│   ├── roi_detector.py          # Region of interest detection
│   ├── text_detector.py         # Text localization (optional)
│   ├── ocr_engine.py            # PaddleOCR wrapper
│   ├── postprocessor.py         # Text cleaning
│   ├── temporal_aggregator.py   # Frame-level smoothing
│   └── instruction_parser.py    # Text → structured commands
├── pipeline.py                  # Main orchestration
└── utils/
    ├── config.py                # Configuration loading
    └── visualization.py         # Debug visualization
```
