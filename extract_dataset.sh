#!/bin/bash
##########################################
# Dataset Extraction from Ultrasound Video
##########################################

# Check if video file is provided
if [ -z "$1" ]; then
    echo "Usage: ./extract_dataset.sh <video_file> [output_dir] [sample_rate] [--save-original]"
    echo ""
    echo "Arguments:"
    echo "  video_file      - Path to video file (e.g., video.webm)"
    echo "  output_dir      - Output directory (default: dataset_output)"
    echo "  sample_rate     - Process every Nth frame (default: 1 = all frames)"
    echo "  --save-original - Save original full frames with frame numbers"
    echo ""
    echo "Examples:"
    echo "  ./extract_dataset.sh video.webm"
    echo "  ./extract_dataset.sh video.webm my_dataset"
    echo "  ./extract_dataset.sh video.webm my_dataset 10"
    echo "  ./extract_dataset.sh video.webm my_dataset 10 --save-original"
    exit 1
fi

VIDEO_FILE="$1"
OUTPUT_DIR="${2:-dataset_output}"
SAMPLE_RATE="${3:-1}"
SAVE_ORIGINAL=""

# Check for --save-original flag
if [ "$4" = "--save-original" ] || [ "$3" = "--save-original" ]; then
    SAVE_ORIGINAL="--save-original"
    if [ "$3" = "--save-original" ]; then
        SAMPLE_RATE="1"
    fi
fi

echo "=========================================="
echo "Ultrasound Dataset Extraction"
echo "=========================================="
echo "Video: $VIDEO_FILE"
echo "Output: $OUTPUT_DIR"
echo "Sample rate: every $SAMPLE_RATE frame(s)"
if [ -n "$SAVE_ORIGINAL" ]; then
    echo "Save original frames: YES"
fi
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run extraction
python3 scripts/extract_dataset.py "$VIDEO_FILE" -o "$OUTPUT_DIR" -s "$SAMPLE_RATE" $SAVE_ORIGINAL
