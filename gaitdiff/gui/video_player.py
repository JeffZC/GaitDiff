"""Video player widget with pose overlay"""
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

from ..core.video import VideoReader
from ..core.pose import PoseDetector


class VideoPlayer(QWidget):
    """Video player widget with play/pause and pose overlay"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_reader = None
        self.pose_detector = PoseDetector()
        self.current_frame = 0
        self.is_playing = False
        self.show_pose = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self._toggle_play_pause)
        self.play_pause_btn.setEnabled(False)
        controls_layout.addWidget(self.play_pause_btn)
        
        self.pose_toggle_btn = QPushButton("Toggle Pose Overlay")
        self.pose_toggle_btn.clicked.connect(self._toggle_pose)
        self.pose_toggle_btn.setEnabled(False)
        controls_layout.addWidget(self.pose_toggle_btn)
        
        layout.addLayout(controls_layout)
    
    def load_video(self, video_path: str):
        """Load a video file"""
        if self.video_reader:
            self.video_reader.release()
        
        self.video_reader = VideoReader(video_path)
        self.current_frame = 0
        self.is_playing = False
        self.play_pause_btn.setText("Play")
        self.play_pause_btn.setEnabled(True)
        self.pose_toggle_btn.setEnabled(True)
        
        # Display first frame
        self._display_frame(0)
    
    def _toggle_play_pause(self):
        """Toggle play/pause"""
        if not self.video_reader:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_pause_btn.setText("Pause")
            fps = self.video_reader.fps if self.video_reader.fps > 0 else 30
            self.timer.start(int(1000 / fps))
        else:
            self.play_pause_btn.setText("Play")
            self.timer.stop()
    
    def _toggle_pose(self):
        """Toggle pose overlay"""
        self.show_pose = not self.show_pose
        if not self.is_playing:
            self._display_frame(self.current_frame)
    
    def _update_frame(self):
        """Update to next frame"""
        if not self.video_reader:
            return
        
        self.current_frame += 1
        if self.current_frame >= self.video_reader.frame_count:
            self.current_frame = 0
        
        self._display_frame(self.current_frame)
    
    def _display_frame(self, frame_number: int):
        """Display a specific frame"""
        if not self.video_reader:
            return
        
        frame = self.video_reader.read_frame(frame_number)
        if frame is None:
            return
        
        # Apply pose overlay if enabled
        if self.show_pose:
            pose_results = self.pose_detector.detect(frame)
            frame = self.pose_detector.draw_landmarks(frame, pose_results)
        
        # Convert to QPixmap and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        self.timer.stop()
        self.play_pause_btn.setText("Play")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if self.video_reader:
            self.video_reader.release()
        self.pose_detector.release()
