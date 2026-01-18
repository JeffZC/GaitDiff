"""Main application window"""
import os
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTableWidget,
    QTableWidgetItem, QTextEdit, QSplitter, QMessageBox,
    QHeaderView, QLineEdit, QTabWidget
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
        
        # Main horizontal splitter: Content | AI Sidebar
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Main content area
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(8, 8, 4, 8)
        
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
        
        # Results section with tabs
        self.results_tabs = QTabWidget()
        
        # Tab 1: Gait Metrics
        gait_tab = QWidget()
        gait_layout = QVBoxLayout(gait_tab)
        gait_layout.setContentsMargins(8, 8, 8, 8)
        
        self.results_table = QTableWidget()
        self.results_table.setRowCount(4)
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "", "Walking Speed\n(m/s)", "Cadence\n(steps/min)", "Step Length\n(cm)", "Step Time\n(s)"
        ])
        self.results_table.setItem(0, 0, QTableWidgetItem("Video A"))
        self.results_table.setItem(1, 0, QTableWidgetItem("Video B"))
        self.results_table.setItem(2, 0, QTableWidgetItem("Difference"))
        self.results_table.setItem(3, 0, QTableWidgetItem("% Change"))
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        gait_layout.addWidget(self.results_table)
        
        self.results_tabs.addTab(gait_tab, "📊 Gait Metrics")
        
        # Tab 2: Joint ROM
        rom_tab = QWidget()
        rom_layout = QVBoxLayout(rom_tab)
        rom_layout.setContentsMargins(8, 8, 8, 8)
        
        self.rom_table = QTableWidget()
        self.rom_table.setRowCount(4)
        self.rom_table.setColumnCount(5)
        self.rom_table.setHorizontalHeaderLabels([
            "", "Left Knee", "Right Knee", "Left Hip", "Right Hip"
        ])
        self.rom_table.setItem(0, 0, QTableWidgetItem("Video A"))
        self.rom_table.setItem(1, 0, QTableWidgetItem("Video B"))
        self.rom_table.setItem(2, 0, QTableWidgetItem("Difference"))
        self.rom_table.setItem(3, 0, QTableWidgetItem("% Change"))
        self.rom_table.setAlternatingRowColors(True)
        self.rom_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rom_table.verticalHeader().setVisible(False)
        rom_layout.addWidget(self.rom_table)
        
        self.results_tabs.addTab(rom_tab, "🦵 Joint ROM")
        
        # Tab 3: AI Summary (placeholder for LLM-generated content)
        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        ai_layout.setContentsMargins(8, 8, 8, 8)
        
        self.ai_summary_display = QTextEdit()
        self.ai_summary_display.setReadOnly(True)
        self.ai_summary_display.setPlaceholderText(
            "This tab will display AI assisted analysis summaries..."
        )
        ai_layout.addWidget(self.ai_summary_display)
        
        self.results_tabs.addTab(ai_tab, "🤖 AI Summary")
        
        main_layout.addWidget(self.results_tabs)
        
        main_splitter.addWidget(content_widget)
        
        # Right: AI Insights sidebar (full height) - VS Code dark theme style
        sidebar_widget = QWidget()
        sidebar_widget.setMinimumWidth(280)
        sidebar_widget.setMaximumWidth(400)
        sidebar_widget.setStyleSheet("background-color: #252526; border-left: 1px solid #3c3c3c;")
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        
        # Sidebar header
        sidebar_header = QLabel("🤖 AI Insights")
        sidebar_header.setStyleSheet("font-weight: bold; font-size: 16px; padding-bottom: 8px; background: transparent; color: #cccccc;")
        sidebar_layout.addWidget(sidebar_header)
        
        # Insights display
        self.insights_display = QTextEdit()
        self.insights_display.setReadOnly(True)
        self.insights_display.setPlaceholderText("Run analysis to see LLM output...")
        self.insights_display.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3c3c3c; border-radius: 4px; padding: 8px; color: #d4d4d4;")
        sidebar_layout.addWidget(self.insights_display, stretch=1)
        
        # Chat input section
        chat_label = QLabel("Ask a question:")
        chat_label.setStyleSheet("font-weight: bold; margin-top: 12px; background: transparent; color: #cccccc;")
        sidebar_layout.addWidget(chat_label)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("e.g., Did asymmetry change?")
        self.chat_input.setStyleSheet("padding: 8px; border: 1px solid #3c3c3c; border-radius: 4px; background-color: #3c3c3c; color: #d4d4d4;")
        self.chat_input.returnPressed.connect(self._send_chat_message)
        sidebar_layout.addWidget(self.chat_input)
        
        send_btn = QPushButton("Ask")
        send_btn.setStyleSheet("padding: 8px; margin-top: 4px; background-color: #0e639c; color: white; border: none; border-radius: 4px;")
        send_btn.clicked.connect(self._send_chat_message)
        sidebar_layout.addWidget(send_btn)
        
        main_splitter.addWidget(sidebar_widget)
        main_splitter.setStretchFactor(0, 4)  # Content gets more space
        main_splitter.setStretchFactor(1, 1)  # Sidebar is narrower
        
        # Set the splitter as central layout
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(main_splitter)
        
        container = QWidget()
        container.setLayout(outer_layout)
        self.setCentralWidget(container)
    
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
            item_a.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(0, col, item_a)
            
            # Video B value
            video_b_val = metric_data.get('video_b', 0)
            item_b = QTableWidgetItem(f"{video_b_val:.2f}")
            item_b.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_table.setItem(1, col, item_b)
            
            # Difference with color coding
            diff = metric_data.get('difference', 0)
            diff_item = QTableWidgetItem(f"{diff:+.2f}")
            diff_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Color code: green for improvement, red for decline (context-dependent)
            if diff > 0:
                diff_item.setBackground(QColor(144, 238, 144))  # Light green
            elif diff < 0:
                diff_item.setBackground(QColor(255, 182, 182))  # Light red
            
            self.results_table.setItem(2, col, diff_item)
            
            # Percent Change
            pct_change = metric_data.get('percent_change', 0)
            pct_item = QTableWidgetItem(f"{pct_change:+.1f}%")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if pct_change > 0:
                pct_item.setBackground(QColor(144, 238, 144))  # Light green
            elif pct_change < 0:
                pct_item.setBackground(QColor(255, 182, 182))  # Light red
            self.results_table.setItem(3, col, pct_item)
        
        self.results_table.resizeColumnsToContents()
        
        # Update ROM table
        comparison = results.get('comparison', {})
        joints = ['left_knee', 'right_knee', 'left_hip', 'right_hip']
        
        for col, joint in enumerate(joints, start=1):
            joint_data = comparison.get(joint, {})
            
            # Video A ROM
            video_a_range = joint_data.get('video_a_range', 0)
            item_a = QTableWidgetItem(f"{video_a_range:.1f}°")
            item_a.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rom_table.setItem(0, col, item_a)
            
            # Video B ROM
            video_b_range = joint_data.get('video_b_range', 0)
            item_b = QTableWidgetItem(f"{video_b_range:.1f}°")
            item_b.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rom_table.setItem(1, col, item_b)
            
            # Difference
            range_diff = joint_data.get('range_diff', 0)
            diff_item = QTableWidgetItem(f"{range_diff:+.1f}°")
            diff_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rom_table.setItem(2, col, diff_item)
            
            # Percent Change
            if video_a_range != 0:
                pct_change = ((video_b_range - video_a_range) / video_a_range) * 100
            else:
                pct_change = 0
            pct_item = QTableWidgetItem(f"{pct_change:+.1f}%")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if pct_change > 0:
                pct_item.setBackground(QColor(144, 238, 144))  # Light green
            elif pct_change < 0:
                pct_item.setBackground(QColor(255, 182, 182))  # Light red
            self.rom_table.setItem(3, col, pct_item)
        
        self.rom_table.resizeColumnsToContents()
        
        # Generate AI insights
        self._generate_insights(results)
    
    def _generate_insights(self, results):
        """Generate AI-style insights from analysis results"""
        gait = results.get('gait_comparison', {})
        comparison = results.get('comparison', {})
        
        lines = []
        lines.append("<h3>📊 Analysis Complete</h3>")
        
        # Walking Speed insight
        speed = gait.get('walking_speed', {})
        speed_a, speed_b = speed.get('video_a', 0), speed.get('video_b', 0)
        speed_diff = speed.get('difference', 0)
        if speed_diff > 0:
            lines.append(f"✅ <b>Walking Speed:</b> Improved from {speed_a:.2f} to {speed_b:.2f} m/s")
        elif speed_diff < 0:
            lines.append(f"⚠️ <b>Walking Speed:</b> Decreased from {speed_a:.2f} to {speed_b:.2f} m/s")
        else:
            lines.append(f"➖ <b>Walking Speed:</b> Stable at {speed_a:.2f} m/s")
        
        # Cadence insight
        cadence = gait.get('cadence', {})
        cad_diff = cadence.get('difference', 0)
        if abs(cad_diff) > 5:
            direction = "Higher" if cad_diff > 0 else "Lower"
            lines.append(f"{'✅' if cad_diff > 0 else '⚠️'} <b>Cadence:</b> {direction} by {abs(cad_diff):.1f} steps/min")
        else:
            lines.append("➖ <b>Cadence:</b> Consistent stepping rhythm")
        
        # Step Length insight
        step = gait.get('step_length', {})
        step_diff = step.get('difference', 0)
        if abs(step_diff) > 0.5:
            direction = "Longer" if step_diff > 0 else "Shorter"
            lines.append(f"{'✅' if step_diff > 0 else '⚠️'} <b>Step Length:</b> {direction} strides in Video B")
        else:
            lines.append("➖ <b>Step Length:</b> Similar stride patterns")
        
        # Overall summary
        lines.append("<br><b>💡 Tip:</b> Ask me questions about the results below!")
        
        self.insights_display.setHtml("<br>".join(lines))
    
    def _send_chat_message(self):
        """Handle chat input for follow-up questions"""
        message = self.chat_input.text().strip()
        if not message:
            return
        
        # Generate response based on context
        response = self._get_chat_response(message)
        
        # Append to insights display
        self.insights_display.append(f"<br><b>You:</b> {message}")
        self.insights_display.append(f"<b>LLM:</b> {response}")
        
        # Clear input
        self.chat_input.clear()
    
    def _get_chat_response(self, message: str) -> str:
        """Generate contextual response (placeholder for real LLM)"""
        msg = message.lower()
        
        if not self.last_results:
            return "Please run an analysis first to get insights."
        
        gait = self.last_results.get('gait_comparison', {})
        
        if 'speed' in msg or 'fast' in msg or 'slow' in msg:
            s = gait.get('walking_speed', {})
            return f"Video A: {s.get('video_a', 0):.2f} m/s, Video B: {s.get('video_b', 0):.2f} m/s. Normal range is 1.2-1.4 m/s."
        
        elif 'cadence' in msg or 'steps' in msg:
            c = gait.get('cadence', {})
            return f"Video A: {c.get('video_a', 0):.1f}, Video B: {c.get('video_b', 0):.1f} steps/min. Normal is 100-120."
        
        elif 'improve' in msg or 'better' in msg or 'tip' in msg:
            return "Focus on posture, increase step length gradually, and maintain consistent cadence. Consider physical therapy if needed."
        
        elif 'normal' in msg or 'range' in msg:
            return "Normal values: Speed 1.2-1.4 m/s, Cadence 100-120 steps/min, Step length 60-80cm."
        
        else:
            return "I can help with: walking speed, cadence, step length, or improvement tips. What would you like to know?"
    
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
