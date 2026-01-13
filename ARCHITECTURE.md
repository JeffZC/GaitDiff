"""
GaitDiff - Gait Analysis Application
=====================================

This document provides technical details about the GaitDiff application architecture.

## Architecture Overview

GaitDiff follows a clean separation between core analysis logic and GUI presentation:

```
gaitdiff/
├── __init__.py          # Package initialization
├── __main__.py          # Application entry point
├── core/                # Core analysis logic (local-first)
│   ├── __init__.py
│   ├── video.py         # Video processing utilities
│   ├── pose.py          # Pose detection and angle computation
│   └── analyzer.py      # Gait analysis engine
└── gui/                 # GUI components (PySide6)
    ├── __init__.py
    ├── video_player.py  # Video player widget with pose overlay
    └── main_window.py   # Main application window
```

## Core Modules

### video.py - Video Processing
- `VideoReader`: Handles video file reading and frame extraction
- Methods:
  - `read_frame(frame_number)`: Read specific frame
  - `sample_frames(num_samples)`: Uniformly sample frames across video
  - `get_current_position()`: Get current playback position

### pose.py - Pose Detection
- `PoseDetector`: MediaPipe-based pose detection
- `calculate_angle(a, b, c)`: Calculate angle between three points
- `extract_joint_angles(results)`: Extract knee and hip angles from landmarks
- `compute_rom(angle_history)`: Compute Range of Motion statistics

### analyzer.py - Gait Analysis
- `GaitAnalyzer`: Main analysis engine
- Methods:
  - `analyze_video(video_path, num_samples)`: Analyze single video
  - `analyze_comparison(video_a, video_b)`: Compare two videos
  - `save_results(results, run_id)`: Save results to JSON

## GUI Modules

### video_player.py - Video Player Widget
- `VideoPlayer`: Custom widget for video playback
- Features:
  - Play/pause controls
  - Pose overlay toggle
  - Frame-by-frame display with OpenCV

### main_window.py - Main Window
- `MainWindow`: Application main window
- `AnalysisWorker`: Background thread for analysis
- Features:
  - Dual video player setup
  - Analysis controls
  - Results table
  - LLM chat panel (placeholder)

## Data Flow

1. User selects Video A and Video B
2. Videos are loaded into `VideoPlayer` widgets
3. User clicks "Analyze"
4. `AnalysisWorker` thread starts:
   - Samples frames from both videos
   - Runs pose detection on each frame
   - Computes joint angles
   - Calculates ROM for each joint
   - Compares metrics between videos
5. Results are displayed in table and saved to JSON

## Results Format

Results are saved to `runs/<run_id>/results.json`:

```json
{
  "video_a": {
    "video_path": "path/to/video_a.mp4",
    "timestamp": "2024-01-13T12:00:00",
    "angle_data": {
      "left_knee": [120.5, 125.3, ...],
      "right_knee": [118.2, 122.1, ...],
      ...
    },
    "rom": {
      "left_knee": {
        "min": 100.2,
        "max": 150.8,
        "range": 50.6,
        "mean": 125.5
      },
      ...
    }
  },
  "video_b": { ... },
  "comparison": {
    "left_knee": {
      "range_diff": 5.2,
      "mean_diff": 3.1,
      "video_a_range": 50.6,
      "video_b_range": 55.8,
      ...
    },
    ...
  }
}
```

## MediaPipe Model

The application uses MediaPipe Pose Landmarker for pose detection. On first run,
it downloads the model (~30MB) to `~/.gaitdiff/models/`.

Model: pose_landmarker_lite.task (float16)
Source: Google MediaPipe Models

## Extending the Application

### Adding New Metrics
1. Update `extract_joint_angles()` in `pose.py` to extract new angles
2. Update `analyze_video()` in `analyzer.py` to include new metrics
3. Update results table in `main_window.py` to display new data

### LLM Integration
The chat panel in `main_window.py` currently echoes messages. To integrate an LLM:
1. Add LLM client library to `requirements.txt`
2. Update `_send_chat_message()` to call LLM API
3. Pass analysis results as context for insights

### Custom Visualizations
Extend `draw_landmarks()` in `pose.py` to add:
- Angle annotations on video
- Trajectory paths
- Heatmaps

## Performance Considerations

- Frame sampling: Default 30 frames per video (configurable)
- Background analysis: Runs in separate thread to keep UI responsive
- Video caching: Frames are processed on-demand to minimize memory usage
- Model loading: PoseDetector initialized once and reused

## Testing

Core functionality can be tested without GUI:

```python
from gaitdiff.core.analyzer import GaitAnalyzer

analyzer = GaitAnalyzer()
results = analyzer.analyze_comparison("video_a.mp4", "video_b.mp4")
analyzer.save_results(results)
```

## Dependencies

- PySide6: Qt-based GUI framework
- OpenCV: Video I/O and image processing
- MediaPipe: Pose detection and landmark tracking
- NumPy: Numerical computations

## License

MIT License - See LICENSE file for details
