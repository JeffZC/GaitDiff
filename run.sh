#!/bin/bash
# Simple script to run GaitDiff application

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if requirements are installed
echo "Checking dependencies..."
$PYTHON_CMD -c "import PySide6; import cv2; import mediapipe; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Run the application
echo "Starting GaitDiff..."
$PYTHON_CMD -m gaitdiff
