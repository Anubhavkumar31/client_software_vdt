# widgets/loading.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout 
from PyQt6.QtCore import Qt
from config.styles import DialogStyles
import time
from PyQt6.QtCore import QTimer

class ModernLoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading Pipe Data")
        self.setModal(True)
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Styling
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f0f0, stop:1 #e0e0e0);
                border: 2px solid #3498db;
                border-radius: 10px;
            }
            QLabel {
                color: #2c3e50;
                font-family: 'Segoe UI', Arial;
            }
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background-color: #ecf0f1;
                text-align: center;
                font-weight: bold;
                color: #2c3e50;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🔄 Loading Pipe Data")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        layout.addWidget(self.status_label)

        # Time info layout
        time_layout = QHBoxLayout()
        self.elapsed_label = QLabel("Elapsed: 0s")
        self.remaining_label = QLabel("Remaining: --")
        self.elapsed_label.setStyleSheet("font-size: 10px; color: #95a5a6;")
        self.remaining_label.setStyleSheet("font-size: 10px; color: #95a5a6;")

        time_layout.addWidget(self.elapsed_label)
        time_layout.addStretch()
        time_layout.addWidget(self.remaining_label)
        layout.addLayout(time_layout)

        # Timer for elapsed time
        self.start_time = time.time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        self.timer.start(100)  # Update every 100ms

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def update_time_estimate(self, remaining_seconds):
        if remaining_seconds and remaining_seconds > 0:
            self.remaining_label.setText(f"Remaining: {remaining_seconds:.1f}s")
        else:
            self.remaining_label.setText("Estimating…")


    def update_elapsed_time(self):
        elapsed = time.time() - self.start_time
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)