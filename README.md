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
- **Preprocessing**: Custom upscaling + contrast enhancement for instruction text
- **ROI Detection**: Fixed region detection with adjustable coordinates
- **OCR**: Tesseract with vocabulary constraint and spell correction
- **Temporal Smoothing**: Majority voting across frames
- **Quality Bar Analysis**: Real-time quality score extraction
- **Dataset Extraction**: Batch processing for ML training data

## Installation

```bash
pip install -r requirements.txt
```

## Available Scripts

### 1. GUI Data Capture (`gui_data_capture.py`) ⭐ NEW

**User-friendly GUI application with start/stop buttons to control dataset capture.**

**Launch GUI:**
```bash
./run_gui_capture.sh
```

**Features:**
- ✅ Live video preview with ROI overlays
- ✅ Start/Stop recording with buttons (no keyboard required)
- ✅ Real-time instruction and quality score display
- ✅ Frame counter and status log
- ✅ Automatic robot pose synchronization via MQTT
- ✅ Configurable video device, dataset directory, and MQTT URL

**Keyboard Controls:**
- **`w`** - Start/Stop recording (toggle) ⭐ NEW
- **`q`** or **ESC** - Quit application
- **`r`** - Toggle ROI display
- **`d`** - Toggle debug window
- **`s`** - Save snapshot

**Usage:**
1. Run with `--save-dataset` flag to enable recording mode
2. Press **`w`** to start recording
3. Press **`w`** again to pause recording
4. Press **`w`** again to resume recording
5. Frames are saved to `robot_training_dataset/` while recording is active

**Configuration:**
- **Video Device:** Device index (e.g., `16` for `/dev/video16`)
- **Config File:** Path to ROI configuration YAML
- **Dataset Dir:** Output directory for captured data
- **Robot MQTT:** MQTT broker URL (e.g., `mqtt://192.168.56.2:1883`)

---

### 2. Real-time OCR (`realtime_ocr.py`)

Extract instruction text and quality scores from live ultrasound video feed (command-line version).

**Basic Usage:**
```bash
./run_realtime_ocr.sh
./run_realtime_ocr.sh 16
```

**With Custom Device:**
```bash
python3 scripts/realtime_ocr.py --device /dev/video16 --interval 1.0
```

**Arguments:**
- `--config`: Path to config YAML (default: `config/default.yaml`)
- `--device`: Video device number or path (e.g., `2` or `/dev/video2`)
- `--interval`: OCR processing interval in seconds (default: `1.0`)
- `--save-dataset`: Enable dataset recording mode
- `--dataset-dir`: Output directory (default: `realtime_dataset`)
- `--robot-pose-mqtt`: MQTT broker URL for robot pose

**Keyboard Controls:**
- **`w`** - Start/Stop recording (toggle) - only works with `--save-dataset` flag
- **`q`** or **ESC** - Quit
- **`r`** - Toggle ROI display
- **`d`** - Toggle debug window
- **`s`** - Save snapshot

**Example with Recording:**
```bash
python3 scripts/realtime_ocr.py \
    --device 2 \
    --save-dataset \
    --dataset-dir robot_training_dataset \
    --robot-pose-mqtt mqtt://192.168.56.2:1883
```
Then press **`w`** to start/stop recording as needed.

# Real-time OCR with dataset generation
python3 scripts/realtime_ocr.py --device /dev/video16 --save-dataset

# Custom dataset directory
python3 scripts/realtime_ocr.py --device /dev/video16 --save-dataset --dataset-dir my_live_dataset

# With custom interval
python3 scripts/realtime_ocr.py --device /dev/video16 --interval 0.5 --save-dataset

# With robot pose recording via MQTT
python3 scripts/realtime_ocr.py --device /dev/video16 --save-dataset \
    --robot-pose-mqtt mqtt://192.168.1.100:1883

**Features:**
- Real-time instruction text extraction
- Quality bar score monitoring (0-99%)
- Change-driven output (only logs when text/score changes)
- Debug windows showing ROI areas and preprocessed images
- Timestamped log file saved on exit

**Example Output:**
```
--- Frame 150 ---
Instruction Text: make slow circular sweeps until moving anatomy appears
  Confidence: 85.3%
Quality: 45%
```

---

### 2. Dataset Extraction (`extract_dataset.py`)

Extract training dataset from ultrasound video files.

**Basic Usage:**
```bash
./extract_dataset.sh "/path/to/video.webm"
```

**With Custom Options:**
```bash
./extract_dataset.sh "/path/to/video.webm" my_dataset 50 --save-original
```

**Direct Python Usage:**
```bash
python3 scripts/extract_dataset.py "/path/to/video.webm" \
  -o dataset_output \
  -s 50 \
  --save-original
```

**Arguments:**
- `video`: Path to video file (required)
- `-o, --output`: Output directory (default: `dataset_output`)
- `-s, --sample-rate`: Process every Nth frame (default: `1`)
- `--save-original`: Save original full frames with frame numbers

**Output Structure:**
```
dataset_output/
├── frames/                    # Ultrasound ROI images (960x756)
│   ├── frame_000000.png
│   ├── frame_000050.png
│   └── ...
├── original_frames/           # Full frames with annotations (1920x1080)
│   ├── original_000000.png
│   └── ...
└── dataset.csv               # Frame metadata
```

**CSV Columns:**
- `frame_number`: Original frame index
- `instruction_text`: OCR-extracted instruction
- `quality_score`: Quality bar percentage (0-99)
- `image_path`: Relative path to ultrasound image

**Examples:**
```bash
# Extract all frames
./extract_dataset.sh video.webm full_dataset 1

# Extract ~20 samples from 3634 frame video
./extract_dataset.sh video.webm samples_20 180 --save-original

# Extract every 10th frame without original frames
./extract_dataset.sh video.webm dataset_10 10
```

---

### 3. ROI Visualizer (`visualize_rois.py`)

Interactive tool to visualize and adjust ROI areas on video.

**Basic Usage:**
```bash
./visualize_rois.sh "/path/to/video.webm"
```

**With Custom Start Frame:**
```bash
./visualize_rois.sh "/path/to/video.webm" 1000
```

**Direct Python Usage:**
```bash
python3 scripts/visualize_rois.py "/path/to/video.webm" -f 500
```

**Arguments:**
- `video`: Path to video file (required)
- `-f, --frame`: Start frame number (default: `500`)

**Interactive Controls:**
- **Arrow keys**: Move selected ROI position
- **+/-**: Adjust ROI width
- **w/s**: Adjust ROI height
- **1**: Select Instruction Text ROI (green)
- **2**: Select Quality Bar ROI (magenta)
- **3**: Select Ultrasound Fan ROI (cyan)
- **Space**: Pause/Resume video
- **c**: Save current ROI configuration to `roi_config.json`
- **q**: Quit

**Features:**
- Real-time ROI visualization with color-coded boxes
- Live coordinate display
- Save adjusted ROI values to JSON
- Auto-loop video for continuous viewing

**Example Workflow:**
```bash
# 1. Visualize ROIs on your video
./visualize_rois.sh video.webm 500

# 2. Adjust ROI positions using arrow keys and +/-/w/s
# 3. Press 'c' to save configuration
# 4. Copy printed values to extract_dataset.py or realtime_ocr.py
```

---

### 4. Create Digit Templates (`create_digit_templates.py`)

Create template images for digit recognition (0-9) from live video.

**Basic Usage:**
```bash
python3 scripts/create_digit_templates.py --device /dev/video16
```

**Arguments:**
- `--device`: Video device path (default: `/dev/video0`)
- `--config`: Path to config YAML (default: `config/default.yaml`)

**Interactive Controls:**
- **0-9 keys**: Save current digit image as template
- **Space**: Pause/Resume
- **q**: Quit

**Output:**
Creates template images in `templates/` directory:
```
templates/
├── 0.png
├── 1.png
├── 2.png
├── ...
└── 9.png
```

**Workflow:**
1. Position ultrasound probe to display score
2. Press number key (0-9) when corresponding digit is clearly visible
3. Repeat for all digits
4. Templates are used by `realtime_ocr.py` for template matching

**Example:**
```bash
# Create templates from video device
python3 scripts/create_digit_templates.py --device /dev/video16

# When score shows "0", press '0' key
# When score shows "1", press '1' key
# ... continue for all digits
```

## Configuration

### ROI Coordinates (Normalized 0-1)

Current ROI settings in `extract_dataset.py` and `visualize_rois.py`:

```python
'instruction': {
    'x': 0.5750,  # Top right area
    'y': 0.1300,
    'w': 0.3550,
    'h': 0.0600,
}

'quality_bar': {
    'x': 0.5795,  # Right side vertical bar
    'y': 0.2586,
    'w': 0.0150,
    'h': 0.2938,
}

'ultrasound_fan': {
    'x': 0.0850,  # Left side imaging area
    'y': 0.1250,
    'w': 0.4800,
    'h': 0.7600,
}
```

Use `visualize_rois.sh` to adjust these coordinates for your video layout.

### OCR Settings

Edit `config/default.yaml` to adjust:
- OCR engine (Tesseract/EasyOCR)
- Confidence thresholds
- Vocabulary constraint settings
- Spell correction options
- Temporal aggregation parameters

## Project Structure

```
Video-to-text-system-main/
├── scripts/
│   ├── realtime_ocr.py          # Real-time OCR from video device
│   ├── extract_dataset.py       # Batch dataset extraction
│   ├── visualize_rois.py        # Interactive ROI adjustment
│   └── create_digit_templates.py # Template creation for digits
├── video2text/
│   └── core/
│       ├── preprocessor.py      # Image enhancement
│       ├── ocr_engine.py        # Tesseract/PaddleOCR wrapper
│       ├── postprocessor.py     # Text cleaning + vocabulary
│       ├── app_vocabulary.py    # APP-specific vocabulary
│       └── spell_corrector.py   # OCR error corrections
├── config/
│   └── default.yaml             # Pipeline configuration
├── templates/                   # Digit templates for matching
├── run_realtime_ocr.sh         # Quick launch script
├── extract_dataset.sh          # Dataset extraction wrapper
├── visualize_rois.sh           # ROI visualizer wrapper
└── README.md
```

## Quick Start Examples

```bash
# 1. Real-time OCR monitoring
./run_realtime_ocr.sh

# 2. Extract dataset from video (every 50th frame)
./extract_dataset.sh "/path/to/video.webm" my_dataset 50 --save-original

# 3. Adjust ROI positions
./visualize_rois.sh "/path/to/video.webm" 500

# 4. Create digit templates
python3 scripts/create_digit_templates.py --device /dev/video16
```

## Troubleshooting

### Poor OCR Results
- Use `visualize_rois.sh` to verify ROI positions
- Check that instruction text is fully within green ROI box
- Adjust ROI coordinates if needed

### Quality Score Inaccurate
- Verify quality bar is within magenta ROI box
- Ensure ROI captures only the vertical bar (not surrounding UI)
- Adjust `y`, `h` values to match bar position

### Video Device Not Found
- List available devices: `ls /dev/video*`
- Update `--device` parameter to correct device path

## Robot Pose Integration (Optional)

Record robot 6D coordinates alongside ultrasound images for training robot control models.

**See:** [ROBOT_POSE_MQTT.md](ROBOT_POSE_MQTT.md) for complete setup guide.

**Quick Setup:**
1. Install MQTT broker on ROS PC: `sudo apt install mosquitto`
2. Run publisher on ROS PC: `python3 robot_pose_mqtt_publisher.py`
3. Run dataset generation with MQTT: `--robot-pose-mqtt mqtt://192.168.1.100:1883`

**Dataset includes:**
- Ultrasound images + instruction text + quality scores
- Robot Cartesian position (x, y, z)
- Robot orientation (quaternion)
- Joint angles (6 joints)

---

## Dataset Format

The extracted dataset is suitable for:
- Instruction text recognition training
- Quality assessment model training
- Ultrasound image analysis
- Multi-modal learning (image + text + score)
- **Robot control model training** (with robot pose data)

CSV format allows easy loading with pandas:
```python
import pandas as pd
df = pd.read_csv('dataset_output/dataset.csv')
print(df.head())
```
