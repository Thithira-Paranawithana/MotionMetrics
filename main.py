"""Main application entry point."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from gui.main_window import BiomechanicalAnalyzerGUI

def main():
    """Main application function."""
    app = QApplication(sys.argv)
    app.setApplicationName("Motion Metrics")
    app.setApplicationVersion("1.0")

    # Create and show main window
    window = BiomechanicalAnalyzerGUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
