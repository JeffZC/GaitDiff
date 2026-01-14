# GaitDiff - Gait Analysis Application
# Dockerfile for running the PySide6 GUI application

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

# Install system dependencies for OpenCV, Qt/PySide6, and X11
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Qt/PySide6 dependencies
    libxcb1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libegl1 \
    libfontconfig1 \
    libfreetype6 \
    # X11 utilities
    x11-utils \
    # Video codecs
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for model cache
RUN mkdir -p /root/.gaitdiff/models

# Create runs directory for output
RUN mkdir -p /app/runs

# Expose no ports (GUI app)
# Entry point
CMD ["python", "-m", "gaitdiff"]
