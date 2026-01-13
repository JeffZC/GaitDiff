# GaitDiff Application UI Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GaitDiff - Gait Analysis Tool                                         ─ □ × │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐        │
│  │ Video A: gait_normal.mp4     │  │ Video B: gait_test.mp4       │        │
│  └──────────────────────────────┘  └──────────────────────────────┘        │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐        │
│  │   [Select Video A]           │  │   [Select Video B]           │        │
│  └──────────────────────────────┘  └──────────────────────────────┘        │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Video Player A                                │  │
│  │                                                                        │  │
│  │                      ┌──────────────────┐                             │  │
│  │                      │                  │                             │  │
│  │                      │  Video Frame A   │                             │  │
│  │                      │   640 x 480      │                             │  │
│  │                      │                  │                             │  │
│  │                      └──────────────────┘                             │  │
│  │                                                                        │  │
│  │           [Play]  [Toggle Pose Overlay]                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Video Player B                                │  │
│  │                                                                        │  │
│  │                      ┌──────────────────┐                             │  │
│  │                      │                  │                             │  │
│  │                      │  Video Frame B   │                             │  │
│  │                      │   640 x 480      │                             │  │
│  │                      │                  │                             │  │
│  │                      └──────────────────┘                             │  │
│  │                                                                        │  │
│  │           [Play]  [Toggle Pose Overlay]                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│                          [Analyze Gait]                                      │
│                                                                              │
│  ┌────────────────────────────────┬──────────────────────────────────────┐  │
│  │  Analysis Results              │  LLM Chat (Placeholder)              │  │
│  ├────────────────────────────────┼──────────────────────────────────────┤  │
│  │ Joint      │ Video A │ Video B │                                      │  │
│  │            │  ROM    │  ROM    │  Chat messages will appear here...   │  │
│  │────────────┼─────────┼─────────┤                                      │  │
│  │ Left Knee  │  85.2°  │  90.5°  │                                      │  │
│  │ Right Knee │  87.1°  │  88.3°  │                                      │  │
│  │ Left Hip   │  45.8°  │  48.2°  │                                      │  │
│  │ Right Hip  │  46.3°  │  47.9°  │                                      │  │
│  │            │         │         │                                      │  │
│  │                                 │  ┌─────────────────────────────┐    │  │
│  │                                 │  │ Type a message...           │    │  │
│  │                                 │  └─────────────────────────────┘    │  │
│  │                                 │           [Send]                     │  │
│  └────────────────────────────────┴──────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## UI Components

### Top Section - Video Selection
- Two file selection buttons for Video A and Video B
- Display selected filenames
- Browse functionality via standard file dialog

### Middle Section - Video Players (Side-by-Side)
- Two synchronized video display areas
- Each player has:
  - Play/Pause button
  - Toggle Pose Overlay button
  - Video frame display (640x480, scalable)
  - Support for .mp4, .avi, .mov formats

### Analysis Button
- Large "Analyze Gait" button
- Runs analysis in background thread
- Progress shown in status bar
- Disables during analysis to prevent double-clicks

### Bottom Section (Split View)
Left Panel - Analysis Results Table:
- Displays joint angles and ROM metrics
- Columns: Joint, Video A ROM, Video B ROM, Difference
- Rows: Left Knee, Right Knee, Left Hip, Right Hip
- Auto-updates after analysis completes

Right Panel - LLM Chat (Placeholder):
- Text display area for chat history
- Input text box for user messages
- Send button
- Currently echoes messages (placeholder for future LLM integration)

## Features Implemented

✓ Dual video player with independent controls
✓ Real-time pose overlay using MediaPipe
✓ Background gait analysis (non-blocking UI)
✓ Automatic frame sampling (30 frames per video)
✓ Joint angle computation (knee, hip)
✓ Range of Motion (ROM) calculation
✓ Results comparison table
✓ JSON export to runs/<timestamp>/results.json
✓ LLM chat panel placeholder with echo functionality

## Keyboard Shortcuts
- Space: Play/Pause current video
- P: Toggle pose overlay
- A: Run analysis (when both videos loaded)

## Technical Details
- Framework: PySide6 (Qt6)
- Video Processing: OpenCV
- Pose Detection: MediaPipe Pose Landmarker
- Threading: QThread for background analysis
- Local-First: All processing done locally, no cloud required
