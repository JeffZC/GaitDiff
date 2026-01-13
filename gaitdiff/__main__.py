"""Main entry point for GaitDiff application"""
import sys
from PySide6.QtWidgets import QApplication

from gaitdiff.gui.main_window import MainWindow


def main():
    """Run the GaitDiff application"""
    app = QApplication(sys.argv)
    app.setApplicationName("GaitDiff")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
