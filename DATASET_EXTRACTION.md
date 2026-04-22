# Dataset Extraction from Ultrasound Videos

This tool processes ultrasound video files frame-by-frame and extracts:
- **Ultrasound fan area images** (the triangular imaging region)
- **Instruction text** (via OCR)
- **Quality bar score** (0-99%)
- **Frame numbers** (for alignment)

## Quick Start

### Extract from a video file

```bash
# Process all frames
./extract_dataset.sh /path/to/video.webm

# Process every 10th frame (faster, smaller dataset)
./extract_dataset.sh /path/to/video.webm my_dataset 10

# Custom output directory
./extract_dataset.sh /path/to/video.webm custom_output_dir
```

### Example with your video

```bash
./extract_dataset.sh "/home/edler/cam_ws/Screencast from 04-13-2026 03:40:02 PM.webm"
```

## Output Structure

```
dataset_output/
├── frames/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
└── dataset.csv
```

### CSV Format

The `dataset.csv` file contains:

| Column | Description |
|--------|-------------|
| `frame_number` | Original frame number from video |
| `instruction_text` | OCR-extracted instruction text |
| `quality_score` | Quality bar score (0-99%) |
| `image_path` | Relative path to ultrasound image |

Example:
```csv
frame_number,instruction_text,quality_score,image_path
0,make slow circular sweeps until moving anatomy appears,12,frames/frame_000000.png
1,make slow circular sweeps until moving anatomy appears,12,frames/frame_000001.png
2,slide down,34,frames/frame_000002.png
```

## Advanced Usage

### Python API

```python
from scripts.extract_dataset import DatasetExtractor

# Initialize extractor
extractor = DatasetExtractor(output_dir='my_dataset')

# Process video
extractor.process_video(
    video_path='video.webm',
    sample_rate=1  # Process every frame
)
```

### Adjust ROI Coordinates

If the ultrasound fan area is not correctly captured, edit the ROI coordinates in `scripts/extract_dataset.py`:

```python
self.roi_configs = {
    'ultrasound_fan': {
        'x': 0.05,   # Left edge (0-1 normalized)
        'y': 0.15,   # Top edge
        'w': 0.50,   # Width
        'h': 0.70,   # Height
    },
    # ... other ROIs
}
```

## Use Cases

### 1. Training Data for ML Models

Extract ultrasound images with quality scores for training image quality assessment models:

```bash
./extract_dataset.sh training_video.webm training_data 5
```

### 2. Instruction Text Analysis

Analyze instruction text patterns across frames:

```python
import pandas as pd

df = pd.read_csv('dataset_output/dataset.csv')
print(df['instruction_text'].value_counts())
```

### 3. Quality Score Distribution

Analyze quality score distribution:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_output/dataset.csv')
df['quality_score'].hist(bins=20)
plt.xlabel('Quality Score')
plt.ylabel('Frequency')
plt.show()
```

## Tips

- **Sample rate**: Use `sample_rate=10` or higher for faster processing and smaller datasets
- **Disk space**: Each frame is saved as PNG (~50-200KB), so a 1000-frame video at full rate = ~50-200MB
- **Processing time**: ~10-30 frames/second depending on your system
- **Video formats**: Supports .webm, .mp4, .avi, and other OpenCV-compatible formats

## Troubleshooting

### Issue: Ultrasound fan area not captured correctly

**Solution**: Adjust the `ultrasound_fan` ROI coordinates in `scripts/extract_dataset.py`

### Issue: OCR not detecting text

**Solution**: Check that Tesseract is installed and the instruction text ROI is correctly positioned

### Issue: Quality score always 0

**Solution**: Verify the quality bar ROI coordinates match your video layout

## Notes

- The tool uses the same ROI configurations as the real-time OCR system
- Quality scores are capped at 99% (never 100%)
- Empty instruction text is saved as empty string in CSV
- Frame numbers correspond to the original video frame indices
