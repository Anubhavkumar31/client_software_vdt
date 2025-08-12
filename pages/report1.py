import sys
import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QHBoxLayout, QComboBox, QTabWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl


def get_app_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

class GraphTab(QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)
        
        self.update_graph()

    def update_graph(self):
        pass  # To be implemented by subclasses

    def load_html_file(self, file_path):
        if not os.path.isfile(file_path):
            print(f"Error: File {file_path} not found.")
            return

        absolute_file_path = os.path.abspath(file_path)
        url = QUrl.fromLocalFile(absolute_file_path)
        self.web_view.setUrl(url)


class Main01Tab(GraphTab):
    def __init__(self, data, parent=None):
        super().__init__(data, parent)

        self.internal_checkbox = QCheckBox("Internal")
        self.external_checkbox = QCheckBox("External")
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.internal_checkbox)
        checkbox_layout.addWidget(self.external_checkbox)
        self.layout.addLayout(checkbox_layout)

        self.internal_checkbox.stateChanged.connect(self.update_graph)
        self.external_checkbox.stateChanged.connect(self.update_graph)

        self.init_done = True

    def update_graph(self):
        if self.data is None or self.data.empty:
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data[r"Abs. Distance (m)"],
            y=self.data[r"ERF (ASME B31G)"],
            mode='lines+markers',
            name='ERF'
        ))
        fig.update_layout(
            title="ERF (ASME B31G) vs Abs. Distance (m)",
            xaxis_title="Abs. Distance (m)",
            yaxis_title="ERF (ASME B31G)"
        )

        # Use correct base path
        base_dir = get_app_dir()
        html_file = base_dir / "backend" / "files" / "ERF_Abs.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

        print("📄 Saving to:", html_file)  # DEBUG OUTPUT
        fig.write_html(str(html_file))
        self.load_html_file(str(html_file))

class Main02Tab(GraphTab):
    def __init__(self, data, parent=None):
        super().__init__(data, parent)

        self.internal_checkbox = QCheckBox("Internal")
        self.external_checkbox = QCheckBox("External")
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.internal_checkbox)
        checkbox_layout.addWidget(self.external_checkbox)
        self.layout.addLayout(checkbox_layout)

        self.internal_checkbox.stateChanged.connect(self.update_graph)
        self.external_checkbox.stateChanged.connect(self.update_graph)

        self.init_done = True

    def update_graph(self):
        if self.data is None or self.data.empty:
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data[r"Abs. Distance (m)"],
            y=self.data[r"Depth %"],
            mode='lines+markers',
            name='Depth %'
        ))
        fig.update_layout(
            title="Depth % vs Abs. Distance (m)",
            xaxis_title="Abs. Distance (m)",
            yaxis_title="Depth %"
        )

        base_dir = get_app_dir()
        html_file = base_dir / "backend" / "files" / "Depth_Percent.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)
        print("📄 Saving to:", html_file)
        fig.write_html(str(html_file))
        self.load_html_file(str(html_file))

class Main03Tab(GraphTab):
    def __init__(self, data, parent=None):
        super().__init__(data, parent)

        self.internal_checkbox = QCheckBox("Internal")
        self.external_checkbox = QCheckBox("External")
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.internal_checkbox)
        checkbox_layout.addWidget(self.external_checkbox)
        self.layout.addLayout(checkbox_layout)

        self.internal_checkbox.stateChanged.connect(self.update_graph)
        self.external_checkbox.stateChanged.connect(self.update_graph)

        self.init_done = True

    def update_graph(self):
        if self.data is None or self.data.empty:
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data[r"Abs. Distance (m)"],
            y=self.data[r"Orientation o' clock"],
            mode='lines+markers',
            name='Orientation'
        ))
        fig.update_layout(
            title="Orientation vs Abs. Distance (m)",
            xaxis_title="Abs. Distance (m)",
            yaxis_title="Orientation o'clock"
        )

        # ✅ Use PyInstaller-safe path
        base_dir = get_app_dir()
        html_file = base_dir / "backend" / "files" / "Ori_Abs.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)

        print("📄 Saving to:", html_file)
        fig.write_html(str(html_file))
        self.load_html_file(str(html_file))



class Report01(QWidget):
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

