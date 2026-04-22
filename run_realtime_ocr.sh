#!/bin/bash
# Launch script for Video-to-Text Real-time OCR
# Usage: ./run_realtime_ocr.sh [device_number] [interval]
#
# Examples:
#   ./run_realtime_ocr.sh           # Use default device 0, interval 1.0s
#   ./run_realtime_ocr.sh 15        # Use /dev/video15, interval 1.0s
#   ./run_realtime_ocr.sh 15 0.5    # Use /dev/video15, interval 0.5s

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
DEVICE=${1:-0}
INTERVAL=${2:-1.0}

echo "=========================================="
echo "Video-to-Text Real-time OCR"
echo "=========================================="
echo "Device: /dev/video${DEVICE}"
echo "OCR Interval: ${INTERVAL}s"
echo "Config: config/default.yaml"
echo ""
echo "Controls:"
echo "  • Press 'q' or ESC to quit"
echo "  • Press 'r' to toggle ROI display"
echo "  • Press 'd' to toggle debug window"
echo "  • Press 's' to save snapshot"
echo "=========================================="
echo ""

# Activate virtual environment and run
cd "${SCRIPT_DIR}"
source .venv_video2text/bin/activate
python scripts/realtime_ocr.py --device ${DEVICE} --interval ${INTERVAL} --config config/default.yaml
