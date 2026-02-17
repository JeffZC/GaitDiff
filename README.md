# GaitDiff

A local-first Python desktop GUI application for gait analysis and comparison using computer vision.

## Features

- **Side-by-Side Video Comparison**: Load and play two videos simultaneously
- **Pose Detection**: Real-time pose overlay using MediaPipe
- **Gait Analysis**: Automatic computation of joint angles (knee/hip) and Range of Motion (ROM)
- **Results Export**: Analysis results saved to JSON in `runs/<run_id>/results.json`
- **LLM Chat Panel**: Placeholder chat interface (echo functionality)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JeffZC/GaitDiff.git
cd GaitDiff
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python -m gaitdiff
```

### Workflow

1. Click "Select Video A" to choose the first video
2. Click "Select Video B" to choose the second video
3. Use play/pause controls to review videos
4. Toggle "Pose Overlay" to visualize detected poses
5. Click "Analyze" to run gait analysis (runs in background)
6. View results in the metrics table
7. Results are automatically saved to `runs/<timestamp>/results.json`

### LLM Chat Panel

The chat panel uses Azure OpenAI with Azure AD credentials (no API key in code). Endpoint and deployment can be stored in the OS keyring under `SERVICE="gaitdiff"` or provided as environment variables. Keys are never committed.

Keyring setup:
```bash
python -c "import keyring; keyring.set_password('gaitdiff','AZURE_OPENAI_ENDPOINT','https://ai-gait-us1.openai.azure.com/')"
python -c "import keyring; keyring.set_password('gaitdiff','AZURE_OPENAI_DEPLOYMENT','gpt-5-mini')"
```

Environment variable fallback:
```bash
set AZURE_OPENAI_ENDPOINT=https://ai-gait-us1.openai.azure.com/
set AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
```

## Architecture

```
gaitdiff/
├── core/              # Core analysis logic (local-first)
│   ├── video.py       # Video processing utilities
│   ├── pose.py        # Pose detection and angle computation
│   └── analyzer.py    # Gait analysis engine
└── gui/               # GUI components (PySide6)
    ├── video_player.py   # Video player widget
    └── main_window.py    # Main application window
```

## Technologies

- **PySide6**: Cross-platform GUI framework
- **OpenCV**: Video processing and computer vision
- **MediaPipe**: Pose detection and landmark tracking
- **NumPy**: Numerical computations

## License

MIT
