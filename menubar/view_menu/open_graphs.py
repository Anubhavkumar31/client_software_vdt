import os
import sys
import importlib.util

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QLabel

from main_window.components.helper_func import _close_graphs_view
from ui.graphs_ui import GraphApp

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

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

        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        header = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setIcon(QIcon("ui/icons/arrow_left.svg"))  # replace with your arrow icon path
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
        back_btn.clicked.connect(lambda :_close_graphs_view(self))
        title = QLabel("Graphs")
        title.setStyleSheet("font-weight: 600; font-size: 14pt;")
        header.addWidget(back_btn);
        header.addSpacing(12);
        header.addWidget(title);
        header.addStretch(1)
        v.addLayout(header)

        graphs_widget = graphs_ui.GraphApp(dataframe=self.pipe_tally, project_root=self.project_root)
        v.addWidget(graphs_widget, stretch=1)

        self._graphs_widget = graphs_widget
        self._central_graphs = container

        if self._central_original is not None and self._central_original.parent() is self:
            self.takeCentralWidget()
        self.setCentralWidget(container)
    except Exception as e:
        try:
            if self.centralWidget() is None and self._central_original is not None:
                self.setCentralWidget(self._central_original)
        except Exception:
            pass
        self.open_Error(f"Unable to open graphs inline: {e}")