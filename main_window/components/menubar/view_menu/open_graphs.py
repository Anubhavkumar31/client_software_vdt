import os
import sys
import importlib.util

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QScrollArea, QFrame

from main_window.components.helper_func import _close_graphs_view
from ui.graphs_ui import GraphApp


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def force_scroll_update(scroll_area, graphs_widget):
    """Force the scroll area to update its dimensions"""
    try:
        # Set a large fixed size to force scrollbars
        graphs_widget.setMinimumSize(QSize(1200, 1000))

        # Update the scroll area
        scroll_area.updateGeometry()
        scroll_area.repaint()

        # Ensure scrollbars are visible
        scroll_area.ensureVisible(0, 0, 100, 100)

        print("Scroll update completed")  # Debug message

    except Exception as e:
        print(f"Error updating scroll: {e}")


def open_graphs(self):
    try:
        if self.pipe_tally is None:
            self.open_Error("Pipe tally not loaded yet.")
            return
        if self._central_graphs is not None and self.centralWidget() is self._central_graphs:
            return
        if self._central_original is None:
            self._central_original = self.centralWidget()

        ui_file_path = resource_path(os.path.join("ui", "graphs_ui.py"))
        if not os.path.exists(ui_file_path):
            self.open_Error(f"Graphs UI file not found at:\n{ui_file_path}")
            return

        spec = importlib.util.spec_from_file_location("graphs_ui", ui_file_path)
        graphs_ui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graphs_ui)

        # Main container
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        # Header section (fixed, not scrollable)
        header = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setIcon(QIcon("ui/icons/arrow_left.svg"))
        back_btn.setIconSize(QSize(16, 16))
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #000000;
                border: 1.5px solid #000000;
                border-radius: 8px;
                padding: 5px 14px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f2f2f2;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #f9f9f9;
                color: #aaaaaa;
                border: 1.5px solid #cccccc;
            }
        """)
        back_btn.clicked.connect(lambda: _close_graphs_view(self))
        title = QLabel("Graphs")
        title.setStyleSheet("font-weight: 600; font-size: 14pt;")
        header.addWidget(back_btn)
        header.addSpacing(12)
        header.addWidget(title)
        header.addStretch(1)
        v.addLayout(header)

        # Create the scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow widget to resize
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll_area.setFrameShape(QFrame.Shape.Box)

        # Create the graphs widget
        graphs_widget = graphs_ui.GraphApp(dataframe=self.pipe_tally, project_root=self.project_root)

        # Set a large minimum size to ensure scrollbars appear
        graphs_widget.setMinimumSize(QSize(1200, 1000))

        # Set the graphs widget as the scroll area's widget
        scroll_area.setWidget(graphs_widget)

        # Style the scroll area for better visibility
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #ffffff;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 14px;
                border-radius: 7px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #b0b0b0;
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #909090;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #f0f0f0;
                height: 14px;
                border-radius: 7px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #b0b0b0;
                border-radius: 7px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #909090;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
                width: 0px;
            }
        """)

        # Add scroll area to main layout with stretch
        v.addWidget(scroll_area, stretch=1)

        self._graphs_widget = graphs_widget
        self._central_graphs = container

        if self._central_original is not None and self._central_original.parent() is self:
            self.takeCentralWidget()
        self.setCentralWidget(container)

        # Force scroll area update after window is shown
        # IMPORTANT: Use the standalone function, NOT self._force_scroll_update
        QTimer.singleShot(200, lambda: force_scroll_update(scroll_area, graphs_widget))

    except Exception as e:
        try:
            if self.centralWidget() is None and self._central_original is not None:
                self.setCentralWidget(self._central_original)
        except Exception:
            pass
        self.open_Error(f"Unable to open graphs inline: {e}")