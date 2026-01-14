"""Main application window"""
import os
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTableWidget,
    QTableWidgetItem, QTextEdit, QSplitter, QMessageBox,
    QHeaderView
)

from .video_player import VideoPlayer
from ..core.analyzer import GaitAnalyzer


class AnalysisWorker(QThread):
    """Background worker for running analysis"""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, video_a_path, video_b_path, analyzer):
        super().__init__()
        self.video_a_path = video_a_path
        self.video_b_path = video_b_path
        self.analyzer = analyzer
    
    def run(self):
        try:
            self.progress.emit("Starting analysis...")
            results = self.analyzer.analyze_comparison(
                self.video_a_path, 
                self.video_b_path,
                num_samples=30
            )
            self.progress.emit("Analysis complete!")
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.video_a_path = None
        self.video_b_path = None
        self.analyzer = GaitAnalyzer()
        self.analysis_worker = None
        self.last_results = None
        
        self.setWindowTitle("GaitDiff - Gait Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top section: Video controls
        video_controls_layout = QHBoxLayout()
        
        # Video A controls
        video_a_layout = QVBoxLayout()
        self.video_a_label = QLabel("Video A: Not selected")
        self.select_video_a_btn = QPushButton("Select Video A")
        self.select_video_a_btn.clicked.connect(self._select_video_a)
        video_a_layout.addWidget(self.video_a_label)
        video_a_layout.addWidget(self.select_video_a_btn)
        video_controls_layout.addLayout(video_a_layout)
        
        # Video B controls
        video_b_layout = QVBoxLayout()
        self.video_b_label = QLabel("Video B: Not selected")
        self.select_video_b_btn = QPushButton("Select Video B")
        self.select_video_b_btn.clicked.connect(self._select_video_b)
        video_b_layout.addWidget(self.video_b_label)
        video_b_layout.addWidget(self.select_video_b_btn)
        video_controls_layout.addLayout(video_b_layout)
        
        main_layout.addLayout(video_controls_layout)
        
        # Video players
        video_layout = QHBoxLayout()
        
        self.player_a = VideoPlayer()
        video_layout.addWidget(self.player_a)
        
        self.player_b = VideoPlayer()
        video_layout.addWidget(self.player_b)
        
        main_layout.addLayout(video_layout)
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._run_analysis)
        main_layout.addWidget(self.analyze_btn)
        
        # Bottom section: Results and LLM chat
        splitter = QSplitter(Qt.Horizontal)
        
        # Results section
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("Gait Analysis Results"))
        
        # Gait metrics table (Step Time, Walking Speed, Step Length, Cadence)
        self.results_table = QTableWidget()
        self.results_table.setRowCount(3)
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "", "Walking Speed\n(m/s)", "Cadence\n(steps/min)", "Step Length\n(cm)", "Step Time\n(s)"
        ])
        self.results_table.setVerticalHeaderLabels(["Video A", "Video B", "Difference"])
        
        # Set row labels in first column
        self.results_table.setItem(0, 0, QTableWidgetItem("Video A"))
        self.results_table.setItem(1, 0, QTableWidgetItem("Video B"))
        self.results_table.setItem(2, 0, QTableWidgetItem("Difference"))
        
        # Style the table - make columns stretch to fill available space
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)  # Hide vertical header since we have row labels
        self.results_table.setMinimumHeight(120)
        
        results_layout.addWidget(self.results_table)
        
        # Joint info section (placeholder for future detailed metrics)
        rom_label = QLabel("Joint Info (placeholder)")
        results_layout.addWidget(rom_label)
        
        self.rom_table = QTableWidget()
        self.rom_table.setRowCount(3)
        self.rom_table.setColumnCount(5)
        self.rom_table.setHorizontalHeaderLabels([
            "", "Left Knee", "Right Knee", "Left Hip", "Right Hip"
        ])
        self.rom_table.setItem(0, 0, QTableWidgetItem("Video A"))
        self.rom_table.setItem(1, 0, QTableWidgetItem("Video B"))
        self.rom_table.setItem(2, 0, QTableWidgetItem("Difference"))
        self.rom_table.setAlternatingRowColors(True)
        self.rom_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rom_table.verticalHeader().setVisible(False)
        self.rom_table.setMinimumHeight(120)
        
        results_layout.addWidget(self.rom_table)
        
        splitter.addWidget(results_widget)
        
        # LLM chat panel (placeholder)
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.addWidget(QLabel("LLM Chat (Placeholder)"))
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Chat messages will appear here...")
        chat_layout.addWidget(self.chat_display)
        
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(80)
        self.chat_input.setPlaceholderText("Type a message and press Enter...")
        chat_layout.addWidget(self.chat_input)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_chat_message)
        chat_layout.addWidget(send_btn)
        
        splitter.addWidget(chat_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
    
    def _select_video_a(self):
        """Select video A"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video A",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if file_path:
            self.video_a_path = file_path
            self.video_a_label.setText(f"Video A: {os.path.basename(file_path)}")
            self.player_a.load_video(file_path)
            self._update_analyze_button()
    
    def _select_video_b(self):
        """Select video B"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video B",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if file_path:
            self.video_b_path = file_path
            self.video_b_label.setText(f"Video B: {os.path.basename(file_path)}")
            self.player_b.load_video(file_path)
            self._update_analyze_button()
    
    def _update_analyze_button(self):
        """Enable analyze button if both videos are selected"""
        self.analyze_btn.setEnabled(
            self.video_a_path is not None and self.video_b_path is not None
        )
    
    def _run_analysis(self):
        """Run gait analysis in background"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            QMessageBox.warning(self, "Analysis Running", "An analysis is already in progress.")
            return
        
        # Stop video playback
        self.player_a.stop()
        self.player_b.stop()
        
        # Disable controls
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")
        
        # Create and start worker thread
        self.analysis_worker = AnalysisWorker(
            self.video_a_path,
            self.video_b_path,
            self.analyzer
        )
        self.analysis_worker.finished.connect(self._on_analysis_complete)
        self.analysis_worker.error.connect(self._on_analysis_error)
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.start()
    
    def _on_analysis_complete(self, results):
        """Handle analysis completion"""
        self.last_results = results
        
        # Save results
        self.analyzer.save_results(results)
        
        # Update table
        self._update_results_table(results)
        
        # Re-enable controls
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        
        QMessageBox.information(self, "Analysis Complete", "Gait analysis completed successfully!")
    
    def _on_analysis_error(self, error_msg):
        """Handle analysis error"""
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze")
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {error_msg}")
    
    def _on_analysis_progress(self, message):
        """Handle analysis progress updates"""
        self.statusBar().showMessage(message)
    
    def _update_results_table(self, results):
        """Update the results tables with analysis data"""
        # Update gait metrics table
        gait_comparison = results.get('gait_comparison', {})
        
        # Order matches new headers: Walking Speed, Cadence, Step Length, Step Time
        metrics = ['walking_speed', 'cadence', 'step_length', 'step_time']
        
        for col, metric in enumerate(metrics, start=1):
            metric_data = gait_comparison.get(metric, {})
            
            # Video A value
            video_a_val = metric_data.get('video_a', 0)
            item_a = QTableWidgetItem(f"{video_a_val:.2f}")
            item_a.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(0, col, item_a)
            
            # Video B value
            video_b_val = metric_data.get('video_b', 0)
            item_b = QTableWidgetItem(f"{video_b_val:.2f}")
            item_b.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(1, col, item_b)
            
            # Difference with color coding
            diff = metric_data.get('difference', 0)
            diff_item = QTableWidgetItem(f"{diff:+.2f}")
            diff_item.setTextAlignment(Qt.AlignCenter)
            
            # Color code: green for improvement, red for decline (context-dependent)
            if diff > 0:
                diff_item.setBackground(Qt.green)
            elif diff < 0:
                diff_item.setBackground(Qt.red)
            
            self.results_table.setItem(2, col, diff_item)
        
        self.results_table.resizeColumnsToContents()
        
        # Update ROM table
        comparison = results.get('comparison', {})
        joints = ['left_knee', 'right_knee', 'left_hip', 'right_hip']
        
        for col, joint in enumerate(joints, start=1):
            joint_data = comparison.get(joint, {})
            
            # Video A ROM
            video_a_range = joint_data.get('video_a_range', 0)
            item_a = QTableWidgetItem(f"{video_a_range:.1f}°")
            item_a.setTextAlignment(Qt.AlignCenter)
            self.rom_table.setItem(0, col, item_a)
            
            # Video B ROM
            video_b_range = joint_data.get('video_b_range', 0)
            item_b = QTableWidgetItem(f"{video_b_range:.1f}°")
            item_b.setTextAlignment(Qt.AlignCenter)
            self.rom_table.setItem(1, col, item_b)
            
            # Difference
            range_diff = joint_data.get('range_diff', 0)
            diff_item = QTableWidgetItem(f"{range_diff:+.1f}°")
            diff_item.setTextAlignment(Qt.AlignCenter)
            self.rom_table.setItem(2, col, diff_item)
        
        self.rom_table.resizeColumnsToContents()
    
    def _send_chat_message(self):
        """Send chat message (echo placeholder)"""
        message = self.chat_input.toPlainText().strip()
        if message:
            # Display user message
            self.chat_display.append(f"<b>You:</b> {message}")
            
            # Echo response (placeholder)
            self.chat_display.append(f"<b>LLM:</b> Echo: {message}")
            
            # Clear input
            self.chat_input.clear()
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop any running analysis
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.terminate()
            self.analysis_worker.wait()
        
        # Cleanup video players
        self.player_a.cleanup()
        self.player_b.cleanup()
        
        # Cleanup analyzer
        self.analyzer.release()
        
        event.accept()
