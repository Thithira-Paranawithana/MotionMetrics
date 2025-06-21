"""GUI styling and themes."""

MAIN_STYLE = """
QMainWindow {
    background-color: #f0f0f0;
}

QLabel {
    font-size: 12px;
    color: #333333;
}

QLabel#title {
    font-size: 24px;
    font-weight: bold;
    color: #2E86AB;
    padding: 10px;
}

QLabel#subtitle {
    font-size: 16px;
    font-weight: bold;
    color: #A23B72;
    padding: 5px;
}

QPushButton {
    background-color: #2E86AB;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1E5F7A;
}

QPushButton:pressed {
    background-color: #0F3A4A;
}

QPushButton:disabled {
    background-color: #cccccc;
    color: #666666;
}

QComboBox {
    padding: 8px;
    border: 2px solid #2E86AB;
    border-radius: 5px;
    background-color: white;
    font-size: 12px;
}

QComboBox:focus {
    border-color: #A23B72;
}

QLineEdit {
    padding: 8px;
    border: 2px solid #2E86AB;
    border-radius: 5px;
    background-color: white;
    font-size: 12px;
}

QLineEdit:focus {
    border-color: #A23B72;
}

QTextEdit {
    border: 2px solid #2E86AB;
    border-radius: 5px;
    background-color: white;
    font-size: 11px;
    padding: 10px;
}

QProgressBar {
    border: 2px solid #2E86AB;
    border-radius: 5px;
    text-align: center;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #2E86AB;
    border-radius: 3px;
}

QGroupBox {
    font-weight: bold;
    border: 2px solid #2E86AB;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    color: #2E86AB;
}

QTabWidget::pane {
    border: 2px solid #2E86AB;
    border-radius: 5px;
}

QTabBar::tab {
    background-color: #e0e0e0;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}

QTabBar::tab:selected {
    background-color: #2E86AB;
    color: white;
}

QTabBar::tab:hover {
    background-color: #A23B72;
    color: white;
}
"""
