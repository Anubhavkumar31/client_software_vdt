from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
import sys

class HeaderWidget(QWidget):
    def __init__(self, client_name, pipeline_name, pipeline_section, icon_path,
                 size=(600, 150), border_margin=20):
        super().__init__()

        # --- Size variable ---
        self.setFixedSize(*size)

        # --- Outer transparent widget ---
        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(border_margin, border_margin, border_margin, border_margin)

        # --- Inner frame with border ---
        inner_frame = QFrame()
        inner_frame.setStyleSheet("""
            QFrame {
                border: 1px solid black;
                border-radius: 8px;
                background-color: white;
            }
            QLabel {
                font-family: Times New Roman;
                font-size: 12pt;
                color: #000;
                border: none;
                background-color: transparent;
            }
        """)

        # --- Left layout (text info) ---
        text_layout = QVBoxLayout()
        text_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        lbl_client = QLabel(f"<b>Client Name:</b> {client_name}")
        lbl_pipeline = QLabel(f"<b>Pipeline Name:</b> {pipeline_name}")
        lbl_section = QLabel(f"<b>Pipeline Section:</b> {pipeline_section}")

        for lbl in [lbl_client, lbl_pipeline, lbl_section]:
            text_layout.addWidget(lbl)

        # --- Right layout (PNG icon) ---
        icon_label = QLabel()
        pixmap = QPixmap(icon_path).scaled(
            100, 100,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # --- Inner layout (bordered box) ---
        inner_layout = QHBoxLayout(inner_frame)
        inner_layout.setContentsMargins(15, 15, 15, 15)  # padding inside border
        inner_layout.addLayout(text_layout, stretch=2)
        inner_layout.addWidget(icon_label, stretch=1)

        # --- Add bordered frame to outer layout ---
        outer_layout.addWidget(inner_frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = HeaderWidget(
        client_name="ABC Corp",
        pipeline_name="Pipeline A",
        pipeline_section="Section 1",
        icon_path=r"F:\work_new\client_software\PIE_dv_new\ui\icons\vdt-logo.png",  # your PNG file
        size=(800, 250),       # overall widget size
        border_margin=10       # control white space before border
    )
    widget.show()

    sys.exit(app.exec())














from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QFrame
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
import sys

class CommentTemplate(QWidget):
    def __init__(self, title="Comments", size=(600, 250), margin=20, border_gap=20):
        super().__init__()

        # --- Size variable ---
        self.setFixedSize(*size)

        # --- Outer layout (adds gap between window and border) ---
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(border_gap, border_gap, border_gap, border_gap)

        # --- Inner frame with border ---
        inner_frame = QFrame()
        inner_frame.setStyleSheet("""
            QFrame {
                border: 1px solid black;
                background-color: white;
            }
            QLabel {
                font-family: Arial;
                font-size: 12pt;
                font-weight: bold;
                color: #000;
                border: none;
                background-color: transparent;
            }
        """)

        # --- Layout inside border ---
        inner_layout = QVBoxLayout(inner_frame)
        inner_layout.setContentsMargins(margin, margin, margin, margin)

        # --- Title ---
        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        # --- Blank space for comments ---
        blank_space = QWidget()
        blank_space.setStyleSheet("background-color: white; border: none;")

        # --- Add widgets ---
        inner_layout.addWidget(lbl_title)
        inner_layout.addWidget(blank_space, stretch=1)

        # --- Add bordered frame to outer layout ---
        outer_layout.addWidget(inner_frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = CommentTemplate(
        title="Comments",
        size=(800, 300),   # overall window size
        margin=20,         # inside spacing (between text and border)
        border_gap=10      # gap between window edge and border
    )
    widget.show()

    sys.exit(app.exec())
