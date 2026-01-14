"""Video player widget with pose overlay and advanced controls"""
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QSlider, QComboBox, QCheckBox, QGroupBox, QSpinBox
)

from ..core.video import VideoReader
from ..core.pose import PoseDetector


class VideoPlayer(QWidget):
    """Video player widget with play/pause, pose overlay, and advanced controls"""
    
    # Signal emitted when frame changes (for syncing players)
    frame_changed = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_reader = None
        self.pose_detector = PoseDetector()
        self.current_frame = 0
        self.is_playing = False
        self.show_pose = False
        self.black_and_white = False
        self.brightness = 0
        self.contrast = 1.0
        self.playback_speed = 1.0
        self.zoom_level = 1.0
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(560, 420)
        self.video_label.setStyleSheet("QLabel { background-color: #1a1a1a; border-radius: 4px; }")
        layout.addWidget(self.video_label)
        
        # Frame slider
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)
        slider_layout.addWidget(self.frame_slider)
        
        # Frame counter
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFixedWidth(80)
        self.frame_label.setAlignment(Qt.AlignCenter)
        slider_layout.addWidget(self.frame_label)
        layout.addLayout(slider_layout)
        
        # Playback controls row
        playback_layout = QHBoxLayout()
        
        # Previous frame button
        self.prev_btn = QPushButton("◀◀")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self._prev_frame)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setToolTip("Previous Frame")
        playback_layout.addWidget(self.prev_btn)
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setFixedWidth(50)
        self.play_pause_btn.clicked.connect(self._toggle_play_pause)
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setToolTip("Play/Pause")
        playback_layout.addWidget(self.play_pause_btn)
        
        # Next frame button
        self.next_btn = QPushButton("▶▶")
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self._next_frame)
        self.next_btn.setEnabled(False)
        self.next_btn.setToolTip("Next Frame")
        playback_layout.addWidget(self.next_btn)
        
        playback_layout.addStretch()
        
        # Speed control
        speed_label = QLabel("Speed:")
        playback_layout.addWidget(speed_label)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "1.5x", "2x"])
        self.speed_combo.setCurrentIndex(2)  # Default 1x
        self.speed_combo.currentTextChanged.connect(self._on_speed_change)
        self.speed_combo.setFixedWidth(70)
        playback_layout.addWidget(self.speed_combo)
        
        layout.addLayout(playback_layout)
        
        # Visual controls row
        visual_layout = QHBoxLayout()
        
        # Pose overlay toggle
        self.pose_checkbox = QCheckBox("Pose Overlay")
        self.pose_checkbox.setChecked(False)
        self.pose_checkbox.stateChanged.connect(self._toggle_pose)
        self.pose_checkbox.setEnabled(False)
        visual_layout.addWidget(self.pose_checkbox)
        
        # Black & white toggle
        self.bw_checkbox = QCheckBox("B&&W")
        self.bw_checkbox.setChecked(False)
        self.bw_checkbox.stateChanged.connect(self._toggle_bw)
        self.bw_checkbox.setEnabled(False)
        self.bw_checkbox.setToolTip("Toggle Black and White")
        visual_layout.addWidget(self.bw_checkbox)
        
        visual_layout.addStretch()
        
        # Brightness control
        bright_label = QLabel("Bright:")
        visual_layout.addWidget(bright_label)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setFixedWidth(80)
        self.brightness_slider.valueChanged.connect(self._on_brightness_change)
        self.brightness_slider.setEnabled(False)
        self.brightness_slider.setToolTip("Adjust Brightness")
        visual_layout.addWidget(self.brightness_slider)
        
        # Contrast control
        contrast_label = QLabel("Contrast:")
        visual_layout.addWidget(contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(50)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedWidth(80)
        self.contrast_slider.valueChanged.connect(self._on_contrast_change)
        self.contrast_slider.setEnabled(False)
        self.contrast_slider.setToolTip("Adjust Contrast")
        visual_layout.addWidget(self.contrast_slider)
        
        # Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedWidth(50)
        self.reset_btn.clicked.connect(self._reset_adjustments)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setToolTip("Reset all adjustments")
        visual_layout.addWidget(self.reset_btn)
        
        layout.addLayout(visual_layout)
    
    def load_video(self, video_path: str):
        """Load a video file"""
        if self.video_reader:
            self.video_reader.release()
        
        self.video_reader = VideoReader(video_path)
        self.current_frame = 0
        self.is_playing = False
        self.play_pause_btn.setText("▶")
        
        # Enable all controls
        self.play_pause_btn.setEnabled(True)
        self.pose_checkbox.setEnabled(True)
        self.bw_checkbox.setEnabled(True)
        self.frame_slider.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.brightness_slider.setEnabled(True)
        self.contrast_slider.setEnabled(True)
        self.reset_btn.setEnabled(True)
        
        # Setup frame slider
        self.frame_slider.setMaximum(self.video_reader.frame_count - 1)
        self.frame_slider.setValue(0)
        self._update_frame_label()
        
        # Display first frame
        self._display_frame(0)
    
    def _on_slider_change(self, value):
        """Handle slider value change"""
        if self.video_reader and not self.frame_slider.isSliderDown():
            self.current_frame = value
            self._display_frame(value)
            self._update_frame_label()
            self.frame_changed.emit(value)
    
    def _on_slider_pressed(self):
        """Handle slider press - pause if playing"""
        self._was_playing = self.is_playing
        if self.is_playing:
            self.timer.stop()
    
    def _on_slider_released(self):
        """Handle slider release - update frame and resume if was playing"""
        self.current_frame = self.frame_slider.value()
        self._display_frame(self.current_frame)
        self._update_frame_label()
        self.frame_changed.emit(self.current_frame)
        
        if self._was_playing:
            self._start_timer()
    
    def _update_frame_label(self):
        """Update the frame counter label"""
        if self.video_reader:
            self.frame_label.setText(f"{self.current_frame} / {self.video_reader.frame_count - 1}")
    
    def _prev_frame(self):
        """Go to previous frame"""
        if self.video_reader and self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.setValue(self.current_frame)
            self._display_frame(self.current_frame)
            self._update_frame_label()
    
    def _next_frame(self):
        """Go to next frame"""
        if self.video_reader and self.current_frame < self.video_reader.frame_count - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self._display_frame(self.current_frame)
            self._update_frame_label()
    
    def _on_speed_change(self, speed_text):
        """Handle playback speed change"""
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "1.5x": 1.5, "2x": 2.0}
        self.playback_speed = speed_map.get(speed_text, 1.0)
        
        # Update timer if playing
        if self.is_playing:
            self._start_timer()
    
    def _toggle_bw(self, state):
        """Toggle black and white mode"""
        self.black_and_white = state == Qt.Checked
        if not self.is_playing:
            self._display_frame(self.current_frame)
    
    def _on_brightness_change(self, value):
        """Handle brightness slider change"""
        self.brightness = value
        if not self.is_playing:
            self._display_frame(self.current_frame)
    
    def _on_contrast_change(self, value):
        """Handle contrast slider change"""
        self.contrast = value / 100.0
        if not self.is_playing:
            self._display_frame(self.current_frame)
    
    def _reset_adjustments(self):
        """Reset all visual adjustments"""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.bw_checkbox.setChecked(False)
        self.brightness = 0
        self.contrast = 1.0
        self.black_and_white = False
        if not self.is_playing:
            self._display_frame(self.current_frame)
    
    def _start_timer(self):
        """Start the playback timer with current speed"""
        if self.video_reader:
            fps = self.video_reader.fps if self.video_reader.fps > 0 else 30
            interval = int(1000 / (fps * self.playback_speed))
            self.timer.start(interval)
    
    def _toggle_play_pause(self):
        """Toggle play/pause"""
        if not self.video_reader:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_pause_btn.setText("⏸")
            self._start_timer()
        else:
            self.play_pause_btn.setText("▶")
            self.timer.stop()
    
    def _toggle_pose(self, state):
        """Toggle pose overlay"""
        self.show_pose = state == Qt.Checked
        if not self.is_playing:
            self._display_frame(self.current_frame)
    
    def _update_frame(self):
        """Update to next frame"""
        if not self.video_reader:
            return
        
        self.current_frame += 1
        if self.current_frame >= self.video_reader.frame_count:
            self.current_frame = 0
        
        # Update slider without triggering change event
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        
        self._display_frame(self.current_frame)
        self._update_frame_label()
    
    def _apply_adjustments(self, frame: np.ndarray) -> np.ndarray:
        """Apply brightness, contrast, and B&W adjustments to frame"""
        adjusted = frame.copy()
        
        # Apply brightness and contrast
        if self.brightness != 0 or self.contrast != 1.0:
            adjusted = cv2.convertScaleAbs(adjusted, alpha=self.contrast, beta=self.brightness)
        
        # Apply black and white
        if self.black_and_white:
            gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
            adjusted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return adjusted
    
    def _display_frame(self, frame_number: int):
        """Display a specific frame"""
        if not self.video_reader:
            return
        
        frame = self.video_reader.read_frame(frame_number)
        if frame is None:
            return
        
        # Apply visual adjustments
        frame = self._apply_adjustments(frame)
        
        # Apply pose overlay if enabled
        if self.show_pose:
            pose_results = self.pose_detector.detect(frame)
            frame = self.pose_detector.draw_landmarks(frame, pose_results)
        
        # Convert to QPixmap and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def seek_to_frame(self, frame_number: int):
        """Seek to a specific frame (for external sync)"""
        if self.video_reader and 0 <= frame_number < self.video_reader.frame_count:
            self.current_frame = frame_number
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_number)
            self.frame_slider.blockSignals(False)
            self._display_frame(frame_number)
            self._update_frame_label()
    
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        self.timer.stop()
        self.play_pause_btn.setText("▶")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if self.video_reader:
            self.video_reader.release()
        self.pose_detector.release()
