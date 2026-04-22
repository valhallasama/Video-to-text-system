#!/bin/bash
# Launch GUI data capture application

# Set PYTHONPATH to include virtual environment packages
export PYTHONPATH="$(pwd)/.venv_video2text/lib/python3.10/site-packages:$PYTHONPATH"

# Remove OpenCV's Qt plugin path from environment
export QT_QPA_PLATFORM_PLUGIN_PATH=""

# Run GUI with system Python (has PyQt5)
python3 gui_data_capture.py
