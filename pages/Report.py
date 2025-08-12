import sys
import os
import plotly.graph_objects as go
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QHBoxLayout, QComboBox, QTabWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
from pages.backend.plotting_utils import plot_erf, plot_depth, plot_orientation
from pages.backend.utils import get_app_dir


class GraphTab(QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)

    def update_graph(self):
        pass  # To be implemented by subclasses

    def load_html_file(self, file_path):
        if not os.path.isfile(file_path):
            print(f"Error: File {file_path} not found.")
            return

        absolute_file_path = os.path.abspath(file_path)
        url = QUrl.fromLocalFile(absolute_file_path)
        self.web_view.setUrl(url)

    def load_html_content(self, html_str: str):
        """Optionally load HTML directly as a string (useful for inline image display)."""
        self.web_view.setHtml(html_str)


class Main1Tab(GraphTab):
    def __init__(self, data, parent=None):
        self.init_done = False
        super().__init__(data, parent)

        self.internal_checkbox = QCheckBox("Internal")
        self.external_checkbox = QCheckBox("External")
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.internal_checkbox)
        checkbox_layout.addWidget(self.external_checkbox)
        self.layout.addLayout(checkbox_layout)

        # Set checked BEFORE calling update_graph
        self.internal_checkbox.setChecked(True)
        self.external_checkbox.setChecked(True)

        self.internal_checkbox.stateChanged.connect(self.update_graph)
        self.external_checkbox.stateChanged.connect(self.update_graph)

        self.init_done = True
        self.update_graph()


    def update_graph(self):
        if self.data is None or self.data.empty:
            return

        if not getattr(self, "init_done", False):
            return  # Silently skip if not ready

        internal = self.internal_checkbox.isChecked()
        external = self.external_checkbox.isChecked()

        if internal and external:
            view = "Both"
        elif internal:
            view = "Internal"
        elif external:
            view = "External"
        else:
            return

        fig, path = plot_erf(self.data.copy(), view=view, return_fig=True)
        self.web_view.load(QUrl.fromLocalFile(path))

class Main2Tab(GraphTab):
    def __init__(self, data, parent=None):
        super().__init__(data, parent)

        self.internal_checkbox = QCheckBox("Internal")
        self.external_checkbox = QCheckBox("External")

        # ✅ Set default to checked
        self.internal_checkbox.setChecked(True)
        self.external_checkbox.setChecked(True)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.internal_checkbox)
        checkbox_layout.addWidget(self.external_checkbox)
        self.layout.addLayout(checkbox_layout)

        self.internal_checkbox.stateChanged.connect(self.update_graph)
        self.external_checkbox.stateChanged.connect(self.update_graph)

        self.init_done = True
        self.update_graph()

    def update_graph(self):
        if self.data is None or self.data.empty:
            return

        # Determine view based on checkboxes
        if self.internal_checkbox.isChecked() and self.external_checkbox.isChecked():
            view = "Both"
        elif self.internal_checkbox.isChecked():
            view = "Internal"
        elif self.external_checkbox.isChecked():
            view = "External"
        else:
            self.web_view.setHtml("<h3>No Surface Location selected</h3>")
            return

        print("🧪 Plotting Depth % view:", view)

        try:
            # Generate the plot and get the path to the saved image
            _, image_path = plot_depth(self.data, view=view, return_fig=True)

            # Create a temporary HTML to embed the image
            html_content = f"""
            <html>
                <head><style>body {{ margin: 0; }}</style></head>
                <body><img src="file:///{image_path}" width="100%" height="100%"></body>
            </html>
            """

            # Save to a temporary HTML file
            temp_html_path = os.path.join(tempfile.gettempdir(), "depth_plot.html")
            with open(temp_html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            self.load_html_file(temp_html_path)

        except Exception as e:
            print("❌ Error plotting depth:", e)
            import traceback
            print(traceback.format_exc())

class Main3Tab(GraphTab):
    def __init__(self, data, parent=None):
        super().__init__(data, parent)

        self.internal_checkbox = QCheckBox("Internal")
        self.external_checkbox = QCheckBox("External")

        # ✅ Check both boxes by default so the graph shows on startup
        self.internal_checkbox.setChecked(True)
        self.external_checkbox.setChecked(True)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.internal_checkbox)
        checkbox_layout.addWidget(self.external_checkbox)
        self.layout.addLayout(checkbox_layout)

        self.internal_checkbox.stateChanged.connect(self.update_graph)
        self.external_checkbox.stateChanged.connect(self.update_graph)

        self.init_done = True
        self.update_graph()

    def update_graph(self):
        if self.data is None or self.data.empty:
            return

        # Determine view mode
        if self.internal_checkbox.isChecked() and self.external_checkbox.isChecked():
            view = "Both"
        elif self.internal_checkbox.isChecked():
            view = "Internal"
        elif self.external_checkbox.isChecked():
            view = "External"
        else:
            self.web_view.setHtml("<h3 style='color:red;'>Please select at least one Surface Location</h3>")
            return


        try:
            _, html_path = plot_orientation(self.data, view=view, return_fig=True)
            self.load_html_file(html_path)
        except Exception as e:
            print(f"❌ Error plotting orientation: {e}")
            import traceback
            print(traceback.format_exc())




class Report(QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("Report")
        self.resize(1285, 913)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(1285, 913))
        self.setMaximumSize(QtCore.QSize(1285, 913))

        self.verticalLayout = QVBoxLayout(self)
        self.setLayout(self.verticalLayout)

        self.tabWidget = QTabWidget()
        self.verticalLayout.addWidget(self.tabWidget)

        self.tab1 = Main1Tab(self.data)
        self.tab2 = Main2Tab(self.data)
        self.tab3 = Main3Tab(self.data)

        self.tabWidget.addTab(self.tab1, "ERF")
        self.tabWidget.addTab(self.tab2, "Depth %")
        self.tabWidget.addTab(self.tab3, "Orientation")

        self.tabWidget.setCurrentIndex(0)
