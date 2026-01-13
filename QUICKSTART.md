# GaitDiff - Quick Start Guide

## Installation (3 steps)

1. **Clone the repository:**
```bash
git clone https://github.com/JeffZC/GaitDiff.git
cd GaitDiff
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python -m gaitdiff
```

Or use the convenience script:
```bash
./run.sh
```

## First Use

1. **On first run**, the application will download the MediaPipe pose detection model (~30MB) to `~/.gaitdiff/models/`
   - This only happens once
   - Requires internet connection

2. **Select videos:**
   - Click "Select Video A" and choose your first gait video
   - Click "Select Video B" and choose your second gait video

3. **Review videos:**
   - Use Play/Pause buttons to review each video
   - Toggle "Pose Overlay" to see detected body landmarks

4. **Analyze:**
   - Click "Analyze" button
   - Analysis runs in background (takes ~30 seconds for two videos)
   - Results appear in the table below

5. **View results:**
   - ROM (Range of Motion) for each joint
   - Comparison between Video A and Video B
   - Results automatically saved to `runs/<timestamp>/results.json`

## Example Videos

The application works with common video formats:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)

For best results:
- Use videos where the person is clearly visible
- Side view works best for gait analysis
- Ensure good lighting
- 640x480 or higher resolution recommended

## Supported Joints

The application analyzes:
- ✓ Left Knee
- ✓ Right Knee
- ✓ Left Hip
- ✓ Right Hip

More joints can be added by modifying `gaitdiff/core/pose.py`.

## Troubleshooting

### "No module named 'PySide6'"
```bash
pip install -r requirements.txt
```

### "Cannot open display"
The application requires a desktop environment (GUI). It cannot run on:
- SSH sessions without X11 forwarding
- Headless servers
- WSL without WSLg

### Model download fails
If automatic download fails:
1. Manually download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
2. Save to: `~/.gaitdiff/models/pose_landmarker_lite.task`

### "No pose detected"
- Ensure person is fully visible in frame
- Try adjusting video lighting
- Check if video has clear side view
- Lower confidence threshold in `gaitdiff/core/pose.py`

## Advanced Usage

### Programmatic API

```python
from gaitdiff.core.analyzer import GaitAnalyzer

analyzer = GaitAnalyzer()
results = analyzer.analyze_comparison(
    "video_a.mp4",
    "video_b.mp4",
    num_samples=30
)
analyzer.save_results(results)
```

See `examples.py` for more usage patterns.

### Custom Analysis

Modify number of frames sampled:
```python
results = analyzer.analyze_video(video_path, num_samples=50)
```

### Batch Processing

Process multiple videos:
```bash
python examples.py
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Check [UI_LAYOUT.md](UI_LAYOUT.md) for UI reference
- See [examples.py](examples.py) for API examples
- Modify pose detection parameters in `gaitdiff/core/pose.py`
- Extend analysis metrics in `gaitdiff/core/analyzer.py`

## System Requirements

- Python 3.8 or higher
- 2GB RAM minimum
- Display/GUI environment
- ~100MB disk space (including model)
- Internet connection (first run only)

## Platform Support

✓ Linux (Ubuntu, Fedora, etc.)
✓ macOS 10.14+
✓ Windows 10/11

## Getting Help

- Check README.md for general information
- Review ARCHITECTURE.md for technical details
- Open an issue on GitHub for bugs
- See examples.py for code samples

## License

MIT - See LICENSE file
