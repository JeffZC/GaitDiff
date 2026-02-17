"""
Pose Editor Window - Full featured pose editing window (PySide6)
Opens from GaitDiff with shared state for seamless integration.
"""

import sys
import cv2
import pandas as pd
import numpy as np
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, 
    QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QSlider,
    QScrollArea, QGroupBox, QComboBox, QLineEdit,
    QProgressDialog, QMessageBox
)
from PySide6.QtCore import Qt, QPoint, QSize, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QCursor, QIcon, QKeySequence, QShortcut

from .plot_utils import create_plot_widget, calculate_ankle_angle
from .mediapipe_utils import get_pose_landmarks_from_frame, process_video_with_mediapipe
from .pose_format_utils import (
    load_pose_data, save_pose_data, SUPPORTED_FORMATS, 
    process_mediapipe_to_rr21, get_keypoint_connections
)
from .shared_state import get_shared_state


class KeypointCommand:
    """Command class for undo/redo operations"""
    def __init__(self, editor, frame_idx, point_idx, old_x, old_y, new_x, new_y):
        self.editor = editor
        self.frame_idx = frame_idx
        self.point_idx = point_idx
        self.old_x = old_x
        self.old_y = old_y
        self.new_x = new_x
        self.new_y = new_y
        
    def undo(self):
        """Restore the previous state"""
        if self.frame_idx != self.editor.current_frame_idx:
            self.editor.current_frame_idx = self.frame_idx
            self.editor.frame_slider.setValue(self.frame_idx)
        
        if self.point_idx != self.editor.selected_point:
            self.editor.selected_point = self.point_idx
            self.editor.keypoint_dropdown.blockSignals(True)
            self.editor.keypoint_dropdown.setCurrentIndex(self.point_idx)
            self.editor.keypoint_dropdown.blockSignals(False)
        
        self.editor.pose_data.iloc[self.frame_idx, self.point_idx * 2] = self.old_x
        self.editor.pose_data.iloc[self.frame_idx, self.point_idx * 2 + 1] = self.old_y
        
        if self.editor.current_pose is not None:
            self.editor.current_pose[self.point_idx] = [self.old_x, self.old_y]
        
        self.editor._needs_redraw = True
        self.editor.update_coordinate_inputs()
        self.editor.display_frame()
        self.editor.update_plot()
        
    def redo(self):
        """Apply the change again"""
        if self.frame_idx != self.editor.current_frame_idx:
            self.editor.current_frame_idx = self.frame_idx
            self.editor.frame_slider.setValue(self.frame_idx)
        
        if self.point_idx != self.editor.selected_point:
            self.editor.selected_point = self.point_idx
            self.editor.keypoint_dropdown.blockSignals(True)
            self.editor.keypoint_dropdown.setCurrentIndex(self.point_idx)
            self.editor.keypoint_dropdown.blockSignals(False)
        
        self.editor.pose_data.iloc[self.frame_idx, self.point_idx * 2] = self.new_x
        self.editor.pose_data.iloc[self.frame_idx, self.point_idx * 2 + 1] = self.new_y
        
        if self.editor.current_pose is not None:
            self.editor.current_pose[self.point_idx] = [self.new_x, self.new_y]
        
        self.editor._needs_redraw = True
        self.editor.update_coordinate_inputs()
        self.editor.display_frame()
        self.editor.update_plot()


class MediaPipeDetectionCommand:
    """Command class for MediaPipe pose detection undo/redo"""
    def __init__(self, editor, frame_idx, old_pose_data, new_pose_data):
        self.editor = editor
        self.frame_idx = frame_idx
        self.old_pose_data = old_pose_data.copy() if old_pose_data is not None else None
        self.new_pose_data = new_pose_data.copy()
        
    def undo(self):
        if self.frame_idx != self.editor.current_frame_idx:
            self.editor.current_frame_idx = self.frame_idx
            self.editor.frame_slider.setValue(self.frame_idx)
        
        if self.old_pose_data is not None:
            self.editor.pose_data.iloc[self.frame_idx] = self.old_pose_data
        
        self.editor.current_pose = self.editor.pose_data.iloc[self.frame_idx].values.reshape(-1, 2)
        self.editor._needs_redraw = True
        self.editor.update_coordinate_inputs()
        self.editor.display_frame()
        self.editor.update_plot()
        
    def redo(self):
        if self.frame_idx != self.editor.current_frame_idx:
            self.editor.current_frame_idx = self.frame_idx
            self.editor.frame_slider.setValue(self.frame_idx)
        
        self.editor.pose_data.iloc[self.frame_idx] = self.new_pose_data
        self.editor.current_pose = self.editor.pose_data.iloc[self.frame_idx].values.reshape(-1, 2)
        self.editor._needs_redraw = True
        self.editor.update_coordinate_inputs()
        self.editor.display_frame()
        self.editor.update_plot()


class PoseEditorWindow(QMainWindow):
    """
    Full-featured pose editor window.
    Can be opened standalone or from GaitDiff with shared state.
    """
    
    # Signal emitted when editor is closed (for syncing back to GaitDiff)
    editor_closed = Signal(str)  # video_id
    pose_data_updated = Signal(str)  # video_id
    
    def __init__(self, video_id: str = None, parent=None):
        super().__init__(parent)
        
        self.video_id = video_id  # 'A' or 'B' or None for standalone
        self.shared_state = get_shared_state() if video_id else None
        
        self.setWindowTitle(f"Pose Editor{f' - Video {video_id}' if video_id else ''}")
        self.setGeometry(100, 100, 1280, 800)

        # Initialize variables
        self.video_path = None
        self.pose_data = None
        self.current_frame = None
        self.current_pose = None
        self.selected_point = None
        self.zoom_level = 1.0
        self.current_frame_idx = 0
        self.max_zoom_level = 5.0
        self.min_zoom_level = 0.5
        self.zoom_center = QPoint(0, 0)
        self.dragging = False
        self.black_and_white = False
        self.playing = False
        self.play_speed = 30
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.advance_frame)
        self.rotation_angle = 0
        self.cap = None
        
        # Undo/redo
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50
        
        # Keypoint names
        self.keypoint_names = SUPPORTED_FORMATS["rr21"]
        
        # Performance optimization
        self.last_update_time = 0
        self.update_interval_ms = 10
        self._cached_frame = None
        self._cached_frame_idx = -1
        self._needs_redraw = True
        self._base_pixmap = None

        self.initUI()
        
        # Load from shared state if available
        if self.shared_state and video_id:
            self._load_from_shared_state()
    
    def _load_from_shared_state(self):
        """Load video and pose data from shared state"""
        if not self.shared_state or not self.video_id:
            return
            
        video_path = self.shared_state.get_video_path(self.video_id)
        if video_path:
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_slider.setMaximum(total_frames - 1)
                
                # Get FPS
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    self.play_speed = fps
                
                # Load pose data
                pose_data = self.shared_state.get_pose_data(self.video_id)
                if pose_data is not None:
                    self.pose_data = pose_data
                    self.keypoint_dropdown.clear()
                    self.keypoint_dropdown.addItems(self.keypoint_names)
                
                # Set current frame
                frame_idx = self.shared_state.get_current_frame(self.video_id)
                self.current_frame_idx = frame_idx
                self.frame_slider.setValue(frame_idx)
                
                # Load selected keypoint
                selected = self.shared_state.get_selected_keypoint(self.video_id)
                if selected is not None:
                    self.selected_point = selected
                    self.keypoint_dropdown.setCurrentIndex(selected)
                
                self.update_frame()
    
    def initUI(self):
        """Initialize the user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
    
        # Left panel for video and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
    
        # Scroll area for video
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    
        self.video_container = QWidget()
        self.video_layout = QVBoxLayout(self.video_container)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.installEventFilter(self)
        self.label.setMinimumSize(640, 480)
        self.video_layout.addWidget(self.label)
    
        self.scroll_area.setWidget(self.video_container)
        left_layout.addWidget(self.scroll_area)
    
        # Plot widget
        self.plot_widget, self.keypoint_plot = create_plot_widget()
        self.keypoint_plot.frame_callback = self.set_frame_from_plot
        left_layout.addWidget(self.plot_widget)
    
        # Navigation controls
        self.nav_controls = QHBoxLayout()
        
        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(35, 35)
        self.play_button.clicked.connect(self.toggle_playback)
        
        self.prev_frame_button = QPushButton("←")
        self.prev_frame_button.setFixedWidth(30)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        self.frame_slider.sliderPressed.connect(self.on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self.on_slider_released)
        
        self.next_frame_button = QPushButton("→")
        self.next_frame_button.setFixedWidth(30)
        self.next_frame_button.clicked.connect(self.next_frame)
        
        self.frame_counter = QLabel("Frame: 0/0")
        self.zoom_label = QLabel("Zoom: 100%")

        self.nav_controls.addWidget(self.play_button)
        self.nav_controls.addWidget(self.prev_frame_button)
        self.nav_controls.addWidget(self.frame_slider)
        self.nav_controls.addWidget(self.next_frame_button)
        self.nav_controls.addWidget(self.frame_counter)
        self.nav_controls.addWidget(self.zoom_label)
        left_layout.addLayout(self.nav_controls)
    
        # Right panel for controls
        right_panel = QWidget()
        right_panel.setFixedWidth(400)
        right_layout = QVBoxLayout(right_panel)
    
        # Video controls
        video_group = QGroupBox("Video Controls")
        video_layout = QVBoxLayout()

        self.load_video_button = QPushButton("📂 Load Video")
        self.load_video_button.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_video_button)

        self.rotate_button = QPushButton("Rotate View (90°)")
        self.rotate_button.clicked.connect(self.rotate_video)
        video_layout.addWidget(self.rotate_button)

        self.bw_button = QPushButton("Toggle B&W")
        self.bw_button.clicked.connect(self.toggle_black_and_white)
        video_layout.addWidget(self.bw_button)

        zoom_layout = QHBoxLayout()
        self.zoom_out_button = QPushButton("➖ Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_in_button = QPushButton("➕ Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.reset_zoom_button = QPushButton("🔄 Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.reset_zoom_button)
        video_layout.addLayout(zoom_layout)
        
        video_group.setLayout(video_layout)
        right_layout.addWidget(video_group)
    
        # Keypoint controls
        keypoint_group = QGroupBox("Keypoint Editing")
        keypoint_layout = QVBoxLayout()

        kp_select_layout = QHBoxLayout()
        kp_select_layout.addWidget(QLabel("Keypoint:"))
        self.keypoint_dropdown = QComboBox()
        self.keypoint_dropdown.addItems(self.keypoint_names)
        self.keypoint_dropdown.currentIndexChanged.connect(self.on_keypoint_selected)
        kp_select_layout.addWidget(self.keypoint_dropdown)
        keypoint_layout.addLayout(kp_select_layout)

        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("X:"))
        self.x_coord_input = QLineEdit()
        self.x_coord_input.setFixedWidth(80)
        coord_layout.addWidget(self.x_coord_input)
        coord_layout.addWidget(QLabel("Y:"))
        self.y_coord_input = QLineEdit()
        self.y_coord_input.setFixedWidth(80)
        coord_layout.addWidget(self.y_coord_input)
        keypoint_layout.addLayout(coord_layout)

        self.confirm_button = QPushButton("✓ Update Coords")
        self.confirm_button.clicked.connect(self.update_keypoint_coordinates)
        keypoint_layout.addWidget(self.confirm_button)

        self.x_coord_input.returnPressed.connect(self.update_keypoint_coordinates)
        self.y_coord_input.returnPressed.connect(self.update_keypoint_coordinates)

        keypoint_group.setLayout(keypoint_layout)
        right_layout.addWidget(keypoint_group)
    
        # History controls
        history_group = QGroupBox("Edit History")
        history_layout = QHBoxLayout()
        self.undo_button = QPushButton("↩ Undo")
        self.undo_button.clicked.connect(self.undo_last_command)
        self.undo_button.setEnabled(False)
        self.redo_button = QPushButton("↪ Redo")
        self.redo_button.clicked.connect(self.redo_last_command)
        self.redo_button.setEnabled(False)
        history_layout.addWidget(self.undo_button)
        history_layout.addWidget(self.redo_button)
        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group)
        
        # Pose detection controls
        pose_group = QGroupBox("Pose Detection")
        pose_layout = QVBoxLayout()

        self.load_pose_button = QPushButton("📄 Load Pose CSV")
        self.load_pose_button.clicked.connect(self.load_pose)
        pose_layout.addWidget(self.load_pose_button)

        self.detect_current_button = QPushButton("Detect Current Frame")
        self.detect_current_button.clicked.connect(self.detect_pose_current_frame)
        pose_layout.addWidget(self.detect_current_button)

        self.detect_video_button = QPushButton("Detect Entire Video")
        self.detect_video_button.clicked.connect(self.detect_pose_video)
        pose_layout.addWidget(self.detect_video_button)

        self.save_button = QPushButton("💾 Save Pose CSV")
        self.save_button.clicked.connect(self.save_pose)
        pose_layout.addWidget(self.save_button)

        pose_group.setLayout(pose_layout)
        right_layout.addWidget(pose_group)
        
        # Sync button for shared state
        if self.shared_state and self.video_id:
            sync_group = QGroupBox("GaitDiff Sync")
            sync_layout = QVBoxLayout()
            
            self.sync_button = QPushButton("🔄 Sync to GaitDiff")
            self.sync_button.clicked.connect(self.sync_to_gaitdiff)
            sync_layout.addWidget(self.sync_button)
            
            sync_group.setLayout(sync_layout)
            right_layout.addWidget(sync_group)
        
        right_layout.addStretch()
    
        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=4)
        main_layout.addWidget(right_panel, stretch=1)

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.prev_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.next_frame)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo_last_command)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.redo_last_command)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_pose)

    def set_frame_from_plot(self, frame_idx):
        """Called when user clicks on the plot to jump to a frame"""
        if 0 <= frame_idx <= self.frame_slider.maximum():
            self.current_frame_idx = frame_idx
            self.frame_slider.setValue(frame_idx)

    def display_frame(self):
        """Display the current frame with pose overlay"""
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        
        if self.black_and_white:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
        # Draw pose overlay
        if self.pose_data is not None and self.current_frame_idx < len(self.pose_data):
            self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
            
            # Draw skeleton connections
            connections = get_keypoint_connections("rr21")
            for (idx1, idx2) in connections:
                if idx1 < len(self.current_pose) and idx2 < len(self.current_pose):
                    pt1 = self.current_pose[idx1]
                    pt2 = self.current_pose[idx2]
                    if (pt1[0] > 0 or pt1[1] > 0) and (pt2[0] > 0 or pt2[1] > 0):
                        cv2.line(frame, 
                                (int(pt1[0]), int(pt1[1])),
                                (int(pt2[0]), int(pt2[1])),
                                (0, 255, 0), 2)
            
            # Draw keypoints
            for i, point in enumerate(self.current_pose):
                if point[0] == 0 and point[1] == 0:
                    continue
                radius = 8 if i == self.selected_point else 5
                color = (255, 0, 0) if i == self.selected_point else (0, 255, 0)
                cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)
        
        # Apply rotation
        if self.rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convert to QPixmap
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(q_img)
        
        # Apply zoom
        scaled_width = max(50, int(self._base_pixmap.width() * self.zoom_level))  # Minimum 50px
        scaled_height = max(50, int(self._base_pixmap.height() * self.zoom_level))  # Minimum 50px

        transformation = Qt.TransformationMode.FastTransformation if self.dragging else Qt.TransformationMode.SmoothTransformation
        scaled_pixmap = self._base_pixmap.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            transformation
        )

        self.label.setPixmap(scaled_pixmap)
        self.label.setFixedSize(scaled_width, scaled_height)

    def update_coordinate_inputs(self):
        """Update the coordinate input fields"""
        if self.selected_point is not None and self.current_pose is not None:
            if 0 <= self.selected_point < len(self.current_pose):
                point = self.current_pose[self.selected_point]
                self.x_coord_input.setText(str(int(point[0])))
                self.y_coord_input.setText(str(int(point[1])))
        else:
            self.x_coord_input.clear()
            self.y_coord_input.clear()

    def load_video(self):
        """Load a video file"""
        if self.cap is not None:
            self.cap.release()
            
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", 
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if video_path:
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.setMaximum(total_frames - 1)
            self.current_frame_idx = 0
            
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.play_speed = fps
            
            self._cached_frame_idx = -1
            self._needs_redraw = True
            
            # Update shared state
            if self.shared_state and self.video_id:
                self.shared_state.set_video_path(self.video_id, video_path)
                self.shared_state.set_frame_count(self.video_id, total_frames)
            
            self.update_frame()

    def load_pose(self):
        """Load pose data from a CSV file"""
        pose_path, _ = QFileDialog.getOpenFileName(
            self, "Open Pose Data", "",
            "Pose Files (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if not pose_path:
            return
            
        expected_frame_count = None
        if self.cap is not None and self.cap.isOpened():
            expected_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pose_data, format_name, keypoint_names, success, message = load_pose_data(
            pose_path, expected_frame_count=expected_frame_count, force_import=False
        )
        
        if not success and "Frame count mismatch" in message:
            reply = QMessageBox.question(
                self, "Frame Count Mismatch",
                f"{message}\n\nForce import anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                pose_data, format_name, keypoint_names, success, message = load_pose_data(
                    pose_path, expected_frame_count=expected_frame_count, force_import=True
                )
        
        if not success:
            QMessageBox.warning(self, "Load Failed", message)
            return
        
        pose_data = pose_data.fillna(0)
        self.pose_data = pose_data
        self.keypoint_names = keypoint_names if keypoint_names else SUPPORTED_FORMATS["rr21"]
        
        self.keypoint_dropdown.blockSignals(True)
        self.keypoint_dropdown.clear()
        self.keypoint_dropdown.addItems(self.keypoint_names)
        self.keypoint_dropdown.blockSignals(False)
        
        self.selected_point = None
        
        # Update shared state
        if self.shared_state and self.video_id:
            self.shared_state.set_pose_data(self.video_id, pose_data)
        
        if self.cap is not None and self.cap.isOpened():
            self.update_frame()
            
        QMessageBox.information(
            self, "Pose Data Loaded",
            f"Loaded {len(self.pose_data)} frames in {format_name} format."
        )

    def save_pose(self):
        """Save pose data to a CSV file"""
        if self.pose_data is None:
            QMessageBox.warning(self, "No Data", "No pose data to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Pose Data", "",
            "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if file_path:
            if save_pose_data(self.pose_data, file_path, "rr21"):
                QMessageBox.information(self, "Saved", "Pose data saved successfully.")
            else:
                QMessageBox.warning(self, "Save Failed", "Failed to save pose data.")

    def update_frame(self):
        """Update the current frame display"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")
                self._needs_redraw = True
                self.display_frame()
                self.update_coordinate_inputs()
                
                if not self.playing:
                    self.update_plot()
                    
                # Update shared state
                if self.shared_state and self.video_id:
                    self.shared_state.set_current_frame(self.video_id, self.current_frame_idx)

    def on_frame_change(self, value):
        """Handle frame slider changes"""
        self.current_frame_idx = value
        self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")
        
        if self.frame_slider.isSliderDown():
            self.preview_frame_at_position(value)
        else:
            self.update_frame()
            self.update_plot()

    def preview_frame_at_position(self, frame_idx):
        """Fast preview while dragging slider"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                self.current_frame = frame
                frame_preview = frame.copy()
                
                if self.black_and_white:
                    frame_preview = cv2.cvtColor(frame_preview, cv2.COLOR_BGR2GRAY)
                    frame_preview = cv2.cvtColor(frame_preview, cv2.COLOR_GRAY2RGB)
                
                if self.pose_data is not None and frame_idx < len(self.pose_data):
                    self.current_pose = self.pose_data.iloc[frame_idx].values.reshape(-1, 2)
                    for point in self.current_pose:
                        if point[0] > 0 or point[1] > 0:
                            cv2.circle(frame_preview, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                
                frame_rgb = cv2.cvtColor(frame_preview, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                scaled_width = int(pixmap.width() * self.zoom_level)
                scaled_height = int(pixmap.height() * self.zoom_level)
                scaled_pixmap = pixmap.scaled(
                    scaled_width, scaled_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
                
                self.label.setPixmap(scaled_pixmap)
                self.label.setFixedSize(scaled_width, scaled_height)
                self.update_coordinate_inputs()

    def on_slider_pressed(self):
        if self.playing:
            self.pause_playback()
            self._was_playing = True
        else:
            self._was_playing = False
        self._slider_start_frame = self.current_frame_idx

    def on_slider_released(self):
        if getattr(self, '_was_playing', False):
            self.start_playback()
        if self._slider_start_frame != self.current_frame_idx:
            self._needs_redraw = True
            self.update_frame()
            self.update_plot()

    def next_frame(self):
        if self.cap is not None and self.current_frame_idx < self.frame_slider.maximum():
            if self.playing:
                self.pause_playback()
            self.current_frame_idx += 1
            self.frame_slider.setValue(self.current_frame_idx)
    
    def prev_frame(self):
        if self.cap is not None and self.current_frame_idx > 0:
            if self.playing:
                self.pause_playback()
            self.current_frame_idx -= 1
            self.frame_slider.setValue(self.current_frame_idx)

    def on_keypoint_selected(self, index):
        if 0 <= index < len(self.keypoint_names):
            self.selected_point = index
            self._needs_redraw = True
            self.update_coordinate_inputs()
            self.display_frame()
            self.update_plot()
            
            if self.shared_state and self.video_id:
                self.shared_state.set_selected_keypoint(self.video_id, index)
        else:
            self.selected_point = None
            self._needs_redraw = True
            self.update_coordinate_inputs()
            self.display_frame()
            self.update_plot()

    def eventFilter(self, source, event):
        """Handle mouse events on the video label"""
        if source is self.label:
            if event.type() == event.Type.MouseButtonPress:
                pos = event.pos()
                scaled_pos = QPoint(int(pos.x() / self.zoom_level), int(pos.y() / self.zoom_level))
                new_selected_point = self.get_selected_point(scaled_pos)
                
                if event.button() == Qt.MouseButton.LeftButton:
                    if new_selected_point is not None:
                        self.selected_point = new_selected_point
                        self.keypoint_dropdown.blockSignals(True)
                        self.keypoint_dropdown.setCurrentIndex(self.selected_point)
                        self.keypoint_dropdown.blockSignals(False)
                        self.dragging = True
                        self.update_coordinate_inputs()
                        self._needs_redraw = True
                        self.display_frame()
                        return True
                        
                elif event.button() == Qt.MouseButton.RightButton:
                    self.selected_point = None
                    self.dragging = False
                    self.keypoint_dropdown.blockSignals(True)
                    self.keypoint_dropdown.setCurrentIndex(-1)
                    self.keypoint_dropdown.blockSignals(False)
                    self.update_coordinate_inputs()
                    self._needs_redraw = True
                    self.display_frame()
                    self.update_plot()
                    return True
                    
            elif event.type() == event.Type.MouseMove and self.dragging and self.selected_point is not None:
                pos = event.pos()
                scaled_pos = QPoint(int(pos.x() / self.zoom_level), int(pos.y() / self.zoom_level))
                current_time = time.time() * 1000
                if current_time - self.last_update_time > self.update_interval_ms:
                    self.move_point(scaled_pos)
                    self.last_update_time = current_time
                self.display_frame()
                return True
                
            elif event.type() == event.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self.dragging:
                    self.dragging = False
                    
                    if hasattr(self, '_drag_start_pos'):
                        start_x, start_y = self._drag_start_pos
                        current_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
                        current_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
                        
                        if abs(start_x - current_x) > 0 or abs(start_y - current_y) > 0:
                            self.create_move_command(
                                self.selected_point,
                                start_x, start_y,
                                current_x, current_y
                            )
                        delattr(self, '_drag_start_pos')
                    
                    self.update_plot()
                    return True
        
        return super().eventFilter(source, event)

    def get_selected_point(self, pos):
        """Find which keypoint is at the given position"""
        if self.current_pose is not None:
            for i, point in enumerate(self.current_pose):
                detect_radius = 15 / self.zoom_level
                if np.linalg.norm(np.array([point[0], point[1]]) - np.array([pos.x(), pos.y()])) < detect_radius:
                    return i
        return None

    def move_point(self, pos):
        """Move the selected keypoint to a new position"""
        if self.selected_point is not None and self.pose_data is not None:
            x, y = pos.x(), pos.y()
            if x < 0 or y < 0:
                return
                
            current_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
            current_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
            
            if abs(x - current_x) < 1 and abs(y - current_y) < 1:
                return
            
            if not hasattr(self, '_drag_start_pos'):
                self._drag_start_pos = (current_x, current_y)
            
            self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2] = x
            self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1] = y
            self.current_pose[self.selected_point] = [x, y]
            
            self.x_coord_input.setText(str(int(x)))
            self.y_coord_input.setText(str(int(y)))
            
            self._needs_redraw = True
            
            # Update shared state
            if self.shared_state and self.video_id:
                self.shared_state.update_keypoint(self.video_id, self.current_frame_idx, self.selected_point, x, y)

    def update_keypoint_coordinates(self):
        """Update keypoint from text input"""
        if self.selected_point is not None and self.pose_data is not None:
            try:
                new_x = int(self.x_coord_input.text())
                new_y = int(self.y_coord_input.text())
            except ValueError:
                return
            
            if new_x < 0 or new_y < 0:
                return
            
            old_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
            old_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
            
            if old_x != new_x or old_y != new_y:
                self.create_move_command(self.selected_point, old_x, old_y, new_x, new_y)
                
                self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2] = new_x
                self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1] = new_y
                
                if self.current_pose is not None and self.selected_point < len(self.current_pose):
                    self.current_pose[self.selected_point] = [new_x, new_y]
                
                self._needs_redraw = True
                self.display_frame()
                self.update_plot()
                
                if self.shared_state and self.video_id:
                    self.shared_state.update_keypoint(self.video_id, self.current_frame_idx, self.selected_point, new_x, new_y)

    def update_plot(self):
        """Update the trajectory plot"""
        if self.dragging:
            return
            
        if hasattr(self, 'keypoint_plot') and self.pose_data is not None and self.selected_point is not None:
            ankle_angles = {'left': [], 'right': []}
            
            try:
                kp_names = self.keypoint_names
                l_knee_idx = kp_names.index("LEFT_KNEE") if "LEFT_KNEE" in kp_names else None
                l_ankle_idx = kp_names.index("LEFT_ANKLE") if "LEFT_ANKLE" in kp_names else None
                l_foot_idx = kp_names.index("LEFT_FOOT") if "LEFT_FOOT" in kp_names else None
                
                r_knee_idx = kp_names.index("RIGHT_KNEE") if "RIGHT_KNEE" in kp_names else None
                r_ankle_idx = kp_names.index("RIGHT_ANKLE") if "RIGHT_ANKLE" in kp_names else None
                r_foot_idx = kp_names.index("RIGHT_FOOT") if "RIGHT_FOOT" in kp_names else None

                have_left = all(idx is not None for idx in [l_knee_idx, l_ankle_idx, l_foot_idx])
                have_right = all(idx is not None for idx in [r_knee_idx, r_ankle_idx, r_foot_idx])

                ankle_angles['left'] = [None] * len(self.pose_data)
                ankle_angles['right'] = [None] * len(self.pose_data)
                
                for frame_idx in range(len(self.pose_data)):
                    frame_pose = self.pose_data.iloc[frame_idx].values.reshape(-1, 2)
                    
                    if have_left:
                        l_angle = calculate_ankle_angle(
                            frame_pose[l_knee_idx], frame_pose[l_ankle_idx], frame_pose[l_foot_idx]
                        )
                        ankle_angles['left'][frame_idx] = l_angle
                    
                    if have_right:
                        r_angle = calculate_ankle_angle(
                            frame_pose[r_knee_idx], frame_pose[r_ankle_idx], frame_pose[r_foot_idx]
                        )
                        ankle_angles['right'][frame_idx] = r_angle
            
            except Exception as e:
                print(f"Error calculating ankle angles: {e}")
            
            total_frames = len(self.pose_data)
            self.keypoint_plot.plot_keypoint_trajectory(
                self.pose_data, self.selected_point, self.current_frame_idx, total_frames, ankle_angles
            )
            
        elif hasattr(self, 'keypoint_plot'):
            self.keypoint_plot.clear_plot()

    # ===== Playback =====

    def toggle_playback(self):
        if self.cap is None:
            return
        if self.playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        self.playing = True
        self.play_button.setText("⏸")
        frame_interval = int(1000 / self.play_speed)
        self.play_timer.start(frame_interval)
        self.frame_slider.setEnabled(False)
        self.prev_frame_button.setEnabled(False)
        self.next_frame_button.setEnabled(False)

    def pause_playback(self):
        self.playing = False
        self.play_button.setText("▶")
        self.play_timer.stop()
        self.frame_slider.setEnabled(True)
        self.prev_frame_button.setEnabled(True)
        self.next_frame_button.setEnabled(True)
        self.update_plot()
    
    def advance_frame(self):
        if self.current_frame_idx >= self.frame_slider.maximum():
            self.pause_playback()
            return
        self.current_frame_idx += 1
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_idx)
        self.frame_slider.blockSignals(False)
        self.update_frame()

    # ===== Undo/Redo =====

    def add_command(self, command):
        self.undo_stack.append(command)
        self.redo_stack = []
        self.redo_button.setEnabled(False)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.undo_button.setEnabled(True)

    def undo_last_command(self):
        if self.undo_stack:
            command = self.undo_stack.pop()
            self.redo_stack.append(command)
            command.undo()
            self.redo_button.setEnabled(True)
            self.undo_button.setEnabled(len(self.undo_stack) > 0)

    def redo_last_command(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            self.undo_stack.append(command)
            command.redo()
            self.undo_button.setEnabled(True)
            self.redo_button.setEnabled(len(self.redo_stack) > 0)

    def create_move_command(self, point_idx, old_x, old_y, new_x, new_y):
        if old_x != new_x or old_y != new_y:
            command = KeypointCommand(
                self, self.current_frame_idx, point_idx, old_x, old_y, new_x, new_y
            )
            self.add_command(command)

    # ===== Pose Detection =====

    def detect_pose_current_frame(self):
        """Detect pose on current frame"""
        if self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "Please load a video first.")
            return
        
        try:
            landmarks_list, annotated_frame = get_pose_landmarks_from_frame(self.current_frame)
            
            if not landmarks_list:
                QMessageBox.warning(self, "No Pose", "Could not detect pose in this frame.")
                return
            
            rr21_landmarks = process_mediapipe_to_rr21(landmarks_list)
            
            old_pose_data = None
            if self.pose_data is not None and self.current_frame_idx < len(self.pose_data):
                old_pose_data = self.pose_data.iloc[self.current_frame_idx].copy()
            
            if self.pose_data is None:
                column_names = []
                for name in SUPPORTED_FORMATS["rr21"]:
                    column_names.extend([f'{name}_X', f'{name}_Y'])
                
                if self.cap:
                    num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.pose_data = pd.DataFrame(np.zeros((num_frames, len(column_names))), columns=column_names)
                else:
                    self.pose_data = pd.DataFrame([np.zeros(len(column_names))], columns=column_names)
                
                self.keypoint_names = SUPPORTED_FORMATS["rr21"]
                self.keypoint_dropdown.blockSignals(True)
                self.keypoint_dropdown.clear()
                self.keypoint_dropdown.addItems(self.keypoint_names)
                self.keypoint_dropdown.blockSignals(False)
            
            for i in range(0, len(rr21_landmarks), 2):
                if i+1 < len(rr21_landmarks) and i//2 < len(self.pose_data.columns)//2:
                    self.pose_data.iloc[self.current_frame_idx, i] = rr21_landmarks[i]
                    self.pose_data.iloc[self.current_frame_idx, i+1] = rr21_landmarks[i+1]
            
            new_pose_data = self.pose_data.iloc[self.current_frame_idx].copy()
            command = MediaPipeDetectionCommand(self, self.current_frame_idx, old_pose_data, new_pose_data)
            self.add_command(command)
            
            self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
            self._needs_redraw = True
            self.display_frame()
            self.update_coordinate_inputs()
            self.update_plot()
            
            if self.shared_state and self.video_id:
                self.shared_state.set_pose_data(self.video_id, self.pose_data)
            
            QMessageBox.information(self, "Success", "Pose detected!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during detection: {str(e)}")

    def detect_pose_video(self):
        """Detect pose on entire video"""
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        
        try:
            progress = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            def progress_callback(percent):
                progress.setValue(percent)
                QApplication.processEvents()
                return progress.wasCanceled()
            
            new_pose_data, success = process_video_with_mediapipe(self.video_path, progress_callback)
            
            progress.close()
            
            if not success:
                if progress.wasCanceled():
                    QMessageBox.information(self, "Canceled", "Processing canceled.")
                else:
                    QMessageBox.warning(self, "Failed", "Could not process video.")
                return
            
            self.pose_data = new_pose_data
            self.keypoint_names = SUPPORTED_FORMATS["rr21"]
            
            self.keypoint_dropdown.blockSignals(True)
            self.keypoint_dropdown.clear()
            self.keypoint_dropdown.addItems(self.keypoint_names)
            self.keypoint_dropdown.blockSignals(False)
            
            if self.current_frame_idx < len(self.pose_data):
                self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
            
            self._needs_redraw = True
            self.display_frame()
            self.update_coordinate_inputs()
            self.update_plot()
            
            if self.shared_state and self.video_id:
                self.shared_state.set_pose_data(self.video_id, self.pose_data)
            
            QMessageBox.information(self, "Success", "Video processed successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error: {str(e)}")

    # ===== View Controls =====

    def rotate_video(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self._needs_redraw = True
        self.display_frame()

    def toggle_black_and_white(self):
        self.black_and_white = not self.black_and_white
        self._needs_redraw = True
        self.display_frame()

    def zoom_in(self):
        if self.zoom_level < self.max_zoom_level:
            old_zoom = self.zoom_level
            self.zoom_level = min(self.max_zoom_level, round(self.zoom_level * 1.2, 1))
            if self.zoom_level != old_zoom:
                self.zoom_label.setText(f"Zoom: {int(self.zoom_level * 100)}%")
                self.display_frame()
    
    def zoom_out(self):
        if self.zoom_level > self.min_zoom_level:
            old_zoom = self.zoom_level
            self.zoom_level = max(self.min_zoom_level, round(self.zoom_level / 1.2, 1))
            if self.zoom_level != old_zoom:
                self.zoom_label.setText(f"Zoom: {int(self.zoom_level * 100)}%")
                self.display_frame()

    def reset_zoom(self):
        """Reset zoom to 100%"""
        if self.zoom_level != 1.0:
            self.zoom_level = 1.0
            self.zoom_label.setText("Zoom: 100%")
            self.display_frame()

    # ===== Sync =====

    def sync_to_gaitdiff(self):
        """Sync current pose data back to GaitDiff"""
        if self.shared_state and self.video_id and self.pose_data is not None:
            self.shared_state.set_pose_data(self.video_id, self.pose_data)
            self.pose_data_updated.emit(self.video_id)
            QMessageBox.information(self, "Synced", "Pose data synced to GaitDiff!")

    def closeEvent(self, event):
        if hasattr(self, 'play_timer') and self.play_timer.isActive():
            self.play_timer.stop()
        if self.cap is not None:
            self.cap.release()
        if self.video_id:
            self.editor_closed.emit(self.video_id)
        super().closeEvent(event)


def main():
    """Standalone entry point for pose editor"""
    app = QApplication(sys.argv)
    editor = PoseEditorWindow()
    editor.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
