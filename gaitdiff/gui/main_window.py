"""Main application window"""
import os
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTableWidget,
    QTableWidgetItem, QTextEdit, QSplitter, QMessageBox
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
        
        # Results table
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("Analysis Results"))
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Joint", "Video A ROM", "Video B ROM", "Difference"
        ])
        results_layout.addWidget(self.results_table)
        
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
        """Update the results table with analysis data"""
        comparison = results.get('comparison', {})
        
        joints = ['left_knee', 'right_knee', 'left_hip', 'right_hip']
        self.results_table.setRowCount(len(joints))
        
        for i, joint in enumerate(joints):
            joint_data = comparison.get(joint, {})
            
            # Joint name
            self.results_table.setItem(i, 0, QTableWidgetItem(joint.replace('_', ' ').title()))
            
            # Video A ROM
            video_a_range = joint_data.get('video_a_range', 0)
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{video_a_range:.2f}°"))
            
            # Video B ROM
            video_b_range = joint_data.get('video_b_range', 0)
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{video_b_range:.2f}°"))
            
            # Difference
            range_diff = joint_data.get('range_diff', 0)
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{range_diff:+.2f}°"))
        
        self.results_table.resizeColumnsToContents()
    
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
