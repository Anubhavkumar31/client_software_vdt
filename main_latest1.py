

import sys, traceback
from PyQt6.QtWidgets import QMessageBox

def global_error_handler(exctype, value, tb):
    # Format error
    formatted = "".join(traceback.format_exception(exctype, value, tb))

    # 🔥 Print to Pycharm terminal
    print("\n" + "=" * 80)
    print("🔥 ERROR OCCURRED:")
    print(formatted)
    print("=" * 80)

    # 🔥 Show in Message Box
    msg = QMessageBox()
    msg.setWindowTitle("Application Error")
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText("An error occurred!")
    msg.setDetailedText(formatted)   # <-- opens on clicking "Show Details"
    msg.exec()

# Install hook
sys.excepthook = global_error_handler

import tempfile, uuid, runpy
import os

# from Data_Gen.DataGenApp import ScriptRunnerApp
from main_window.main_window import MyMainWindow

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-logging --log-level=3 --disable-features=AccessibilityAriaVirtualContent"

# main.py
import sys
import os
import time
import subprocess
import re
from glob import glob
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import Qt

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
except ImportError:
    # Some builds moved QWebEnginePage to a separate submodule
    from PyQt6.QtWebEngineCore import QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import plotly.graph_objects as go

# PyQt6 Core
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import (
    Qt, QSortFilterProxyModel, QThread, pyqtSignal,
    QTimer, QUrl, QEvent, QEventLoop,QSize
)
# PyQt6 GUI
from PyQt6.QtGui import (
    QStandardItemModel, QStandardItem, QMovie, QPixmap, QImage, QAction, QIcon,
    QCursor
)
# PyQt6 Widgets
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QHeaderView, QInputDialog,
    QSpacerItem, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem,
    QStatusBar, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox,
    QDialog, QTextEdit, QPushButton, QSplitter, QStackedWidget,
    QTabBar, QFrame, QHBoxLayout as _QHBoxLayout, QSplitterHandle, QComboBox,
    QAbstractItemView, QAbstractScrollArea, QProgressBar
)
# PyQt6 WebEngine
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QScrollArea, QScrollBar
from PyQt6.QtGui import QPalette, QColor

# Project imports (leave as-is)
from reportlab.pdfgen import canvas  # noqa
from pages.customPlot import CPlot_Frame as customPlot
from pages.telemetryPlot import TPlot_Frame as telePlot
from pages.anamolyPlot import ADPlot_Frame as adPlot
from pages.about import About_Dialog
from pages.adminPanel import Admin_Panel
from pages.erf1 import ERF1App as ERF

#hahahahahaasdasdasdasd
from pages.XYZ import XYZ  # noqa
from pages.metrics import Metric_Dialog  # noqa
from pages.cluster import Cluster_Dialog
from pages.assessMethod import Assess_Dialog
from pages.errorBox import Error_Dialog  # noqa
from pages.report1 import Report01, Main01Tab, Main02Tab, Main03Tab
from pages.Report import Report, Main1Tab, Main2Tab, Main3Tab
from backend.line_plot import PlotWindow
from backend.heatmap import HeatmapWindow as hm, pre_process, pre_process2  # noqa
from ui.graphs_ui import GraphApp



# --- Lightweight DataFrame model (no per-cell Qt items) ---
from PyQt6.QtCore import QAbstractTableModel, QVariant

class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, _parent=None):
        return 0 if self._df is None else len(self._df)

    def columnCount(self, _parent=None):
        return 0 if self._df is None else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if pd.isna(val):
                return ""
            # cheap formatting for floats
            if isinstance(val, float):
                return f"{val:.6g}"
            return str(val)

        return QVariant()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        elif role == Qt.ItemDataRole.FontRole:
            # Make headers bold
            from PyQt6.QtGui import QFont
            font = QFont()
            font.setBold(True)
            return font
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

        return QVariant()

    def flags(self, index):
        """Make all items non-editable"""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable



def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def _dump_tally_to_temp(df):
    import pickle
    p = os.path.join(tempfile.gettempdir(), f"pipe_tally_{uuid.uuid4().hex}.pkl")
    with open(p, "wb") as f: pickle.dump(df, f)
    return p


base_dir = os.path.dirname(__file__)
ui_path = os.path.join(base_dir, "ui", "landing.ui")
SplashScreen, SplashWindow = uic.loadUiType(ui_path)
ui_path_main = os.path.join(base_dir, "ui", "main_window.ui")
Form, Window = uic.loadUiType(ui_path_main)


# SCROLLBAR_STYLE = """
# QScrollBar:vertical {
#     background: #2b2b2b;
#     width: 14px;
# }
# QScrollBar::handle:vertical {
#     background: #555;
#     min-height: 20px;
# }
# QScrollBar::handle:vertical:hover {
#     background: #777;
# }
# QScrollBar:horizontal {
#     background: #2b2b2b;
#     height: 14px;
# }
# QScrollBar::handle:horizontal {
#     background: #555;
#     min-width: 20px;
# }
# QScrollBar::handle:horizontal:hover {
#     background: #777;
# }
# """

def setup_table_scroll(table):
    from PyQt6.QtWidgets import QHeaderView, QAbstractItemView, QAbstractScrollArea
    from PyQt6.QtCore import Qt

    # Show scrollbars when needed (or keep AlwaysOn if you prefer)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

    # per-pixel scrolling for smooth behavior
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    # don't let the view auto-adjust its size to contents (prevents hiding scrollbars)
    table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)

    # Configure horizontal header: interactive sizing and a large default width so total width > viewport
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

    # <- Increase default section size to force horizontal overflow.
    # Set this to a higher value if you have many columns (try 220 - 320).
    header.setDefaultSectionSize(380)

    # Configure vertical header (row height)
    vheader = table.verticalHeader()
    vheader.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    vheader.setDefaultSectionSize(40)

    # Set slower scroll speed
    table.verticalScrollBar().setSingleStep(15)

#
# class PipeLoaderWorker(QThread):
#     # Signals for communication
#     progress_updated = pyqtSignal(int, str)  # progress %, message
#     data_loaded = pyqtSignal(object)  # pandas DataFrame
#     assets_loaded = pyqtSignal(dict)  # asset paths dictionary
#     table_data_ready = pyqtSignal(object)  # processed table data
#     error_occurred = pyqtSignal(str)  # error message
#     time_estimate = pyqtSignal(float)  # estimated time remaining
#
#     def __init__(self, pkl_path, project_root, pipe_idx):
#         super().__init__()
#         self.pkl_path = pkl_path
#         self.project_root = project_root
#         self.pipe_idx = pipe_idx
#         self.start_time = None
#
#     def run(self):
#         try:
#             self.start_time = time.time()
#             total_steps = 6
#
#             # Step 1: Load pickle data
#             self.progress_updated.emit(10, "Loading pipe data...")
#             df = pd.read_pickle(self.pkl_path)
#             self.data_loaded.emit(df)
#             self._update_time_estimate(1, total_steps)
#             print(f"Loaded pickle with {len(df)} rows")
#
#             # Step 2: Find pipe directory
#             self.progress_updated.emit(25, "Locating asset files...")
#             pipe_dir = self._find_pipe_directory()
#             self._update_time_estimate(2, total_steps)
#
#             # Step 3: Load HTML assets
#             self.progress_updated.emit(40, "Loading chart assets...")
#             assets = self._load_html_assets(pipe_dir)
#             self.assets_loaded.emit(assets)
#             self._update_time_estimate(3, total_steps)
#
#             # Step 4: Load pipe tally data
#             self.progress_updated.emit(60, "Processing pipe tally...")
#             table_data = self._load_pipe_tally_data(pipe_dir)
#             self._update_time_estimate(4, total_steps)
#
#             # Step 5: Process table data
#             self.progress_updated.emit(80, "Preparing table data...")
#             if table_data is not None:
#                 processed_data = self._process_table_data(table_data)
#                 self.table_data_ready.emit(processed_data)
#             else:
#                 self.table_data_ready.emit(None)
#             self._update_time_estimate(5, total_steps)
#
#             # Step 6: Complete
#             self.progress_updated.emit(100, "Loading complete!")
#             self._update_time_estimate(6, total_steps)
#
#         except Exception as e:
#             self.error_occurred.emit(str(e))
#
#     def _update_time_estimate(self, current_step, total_steps):
#         elapsed = time.time() - self.start_time
#         if current_step > 0:
#             avg_time_per_step = elapsed / current_step
#             remaining_steps = total_steps - current_step
#             estimated_remaining = avg_time_per_step * remaining_steps
#             self.time_estimate.emit(estimated_remaining)
#
#     def _find_pipe_directory(self):
#         # Look for pipe directories inside pipes_data subfolder
#         pipes_data_dir = os.path.join(self.project_root, "pipes_data")
#         if not os.path.isdir(pipes_data_dir):
#             print(f"[Warning] pipes_data directory not found in {self.project_root}")
#             return None
#
#         candidates = [
#             os.path.join(pipes_data_dir, f"pipe_{self.pipe_idx}"),
#             os.path.join(pipes_data_dir, f"pipe-{self.pipe_idx}"),
#             os.path.join(pipes_data_dir, f"Pipe_{self.pipe_idx}"),
#         ]
#         return next((d for d in candidates if os.path.isdir(d)), None)
#
#
#     def _load_html_assets(self, pipedir):
#         if not pipedir:
#             return {}
#
#         def pickone(*patterns, exclude=None):
#             exclude = exclude or []
#             hits = []
#             for pat in patterns:
#                 hits.extend(glob(os.path.join(pipedir, pat)))
#             hits = [h for h in hits if not any(ex in os.path.basename(h).lower() for ex in (exclude or []))]
#             exact = [h for h in hits if re.search(rf"{re.escape(str(self.pipe_idx))}", os.path.basename(h))]
#             return exact[0] if exact else (hits[0] if hits else None)
#
#         return {
#             "hmap": pickone("heatmap*.html", exclude=["raw", "box"]),
#             "hmap_r": pickone("heatmap_raw*.html", "raw_heatmap*.html"),
#             "heatmap_box": pickone("heatmap_box*.html", "box_heatmap*.html"),
#             "lplot": pickone("lineplot*.html", "line*.html", exclude=["raw"]),
#             "lplot_r": pickone("lineplot_raw*.html", "line_raw*.html"),
#             "pipe3d": pickone("pipe_3d*.html", "pipe3d*.html"),
#             "prox_linechart": pickone("proximity_linechart*.html", "proximitylinechart*.html"),
#             "hallsensor_heatmap": pickone("hallsensor_heatmap*.html"),
#             "proximity_heatmap": pickone("proximity_heatmap*.html")
#         }
#
#     def _load_pipe_tally_data(self, pipe_dir):
#         if not pipe_dir:
#             return None
#
#         def pick_one(patterns, exclude=None):
#             exclude = exclude or []
#             hits = []
#             for pat in patterns:
#                 hits.extend(glob(os.path.join(pipe_dir, pat)))
#             hits = [h for h in hits if not any(ex in os.path.basename(h).lower() for ex in (exclude or []))]
#             exact = [h for h in hits if re.search(rf'{re.escape(str(self.pipe_idx))}\b', os.path.basename(h))]
#             return exact[0] if exact else (hits[0] if hits else None)
#
#         pipe_tally_csv = pick_one([f"*PipeTally{self.pipe_idx}.csv", f"*PipeTally{self.pipe_idx}.xlsx"])
#         if pipe_tally_csv:
#             try:
#                 if pipe_tally_csv.lower().endswith(".csv"):
#                     df = pd.read_csv(pipe_tally_csv)
#                 else:
#                     df = pd.read_excel(pipe_tally_csv)
#                 return df
#             except Exception:
#                 pass
#
#         # Fallback to defects.csv
#         ds_csv = pick_one(["*defectS*.csv", "*defects*.csv"])
#         if ds_csv:
#             try:
#                 return pd.read_csv(ds_csv)
#             except Exception:
#                 pass
#
#         return None
#
#     # def _process_table_data(self, df):
#     #     if df is None or df.empty:
#     #         return None
#
#     #     # Check if this is a PipeTally file (has Feature Type column) or defects.csv
#     #     if "Feature Type" in df.columns:
#     #         # Filter Metal Loss defects
#     #         original_count = len(df)
#     #         df = df[df["Feature Type"].astype(str).str.strip().str.lower() == "metal loss"]
#
#     #         if df.empty:
#     #             return None
#
#     #         # Round numeric columns
#     #         numeric_columns = [
#     #             'Depth %', 'Depth (mm)', 'ERF (ASME B31G)', 'Psafe (ASME B31G) Barg',
#     #             'Abs. Distance (m)', 'Distance to U/S GW(m)', 'Length (mm)',
#     #             'Width (mm)', 'WT (mm)', 'Pipe Length (mm)'
#     #         ]
#     #         for col in numeric_columns:
#     #             if col in df.columns:
#     #                 df[col] = pd.to_numeric(df[col], errors='coerce').round(3)
#
#     #     return df
#     def _process_table_data(self, df):
#         """Return full PipeTally data without filtering or skipping."""
#         if df is None or df.empty:
#             return None
#
#         # Just round numeric columns, no filtering
#         numeric_columns = [
#             'Depth %', 'Depth (mm)', 'ERF (ASME B31G)', 'Psafe (ASME B31G) Barg',
#             'Abs. Distance (m)', 'Distance to U/S GW(m)', 'Length (mm)',
#             'Width (mm)', 'WT (mm)', 'Pipe Length (mm)'
#         ]
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').round(3)
#
#         return df
#
#
#
# class ModernLoadingDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Loading Pipe Data")
#         self.setModal(True)
#         self.setFixedSize(400, 200)
#         self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
#
#         # Styling
#         self.setStyleSheet("""
#             QDialog {
#                 background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
#                     stop:0 #f0f0f0, stop:1 #e0e0e0);
#                 border: 2px solid #3498db;
#                 border-radius: 10px;
#             }
#             QLabel {
#                 color: #2c3e50;
#                 font-family: 'Segoe UI', Arial;
#             }
#             QProgressBar {
#                 border: 2px solid #bdc3c7;
#                 border-radius: 8px;
#                 background-color: #ecf0f1;
#                 text-align: center;
#                 font-weight: bold;
#                 color: #2c3e50;
#             }
#             QProgressBar::chunk {
#                 background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
#                     stop:0 #3498db, stop:1 #2980b9);
#                 border-radius: 6px;
#             }
#         """)
#
#         layout = QVBoxLayout(self)
#         layout.setSpacing(15)
#         layout.setContentsMargins(20, 20, 20, 20)
#
#         # Title
#         title = QLabel("🔄 Loading Pipe Data")
#         title.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
#         layout.addWidget(title)
#
#         # Progress bar
#         self.progress_bar = QProgressBar()
#         self.progress_bar.setRange(0, 100)
#         self.progress_bar.setTextVisible(True)
#         layout.addWidget(self.progress_bar)
#
#         # Status label
#         self.status_label = QLabel("Initializing...")
#         self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.status_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
#         layout.addWidget(self.status_label)
#
#         # Time info layout
#         time_layout = QHBoxLayout()
#         self.elapsed_label = QLabel("Elapsed: 0s")
#         self.remaining_label = QLabel("Remaining: --")
#         self.elapsed_label.setStyleSheet("font-size: 10px; color: #95a5a6;")
#         self.remaining_label.setStyleSheet("font-size: 10px; color: #95a5a6;")
#
#         time_layout.addWidget(self.elapsed_label)
#         time_layout.addStretch()
#         time_layout.addWidget(self.remaining_label)
#         layout.addLayout(time_layout)
#
#         # Timer for elapsed time
#         self.start_time = time.time()
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_elapsed_time)
#         self.timer.start(100)  # Update every 100ms
#
#     def update_progress(self, value, message):
#         self.progress_bar.setValue(value)
#         self.status_label.setText(message)
#
#     def update_time_estimate(self, remaining_seconds):
#         if remaining_seconds and remaining_seconds > 0:
#             self.remaining_label.setText(f"Remaining: {remaining_seconds:.1f}s")
#         else:
#             self.remaining_label.setText("Estimating…")
#
#
#     def update_elapsed_time(self):
#         elapsed = time.time() - self.start_time
#         self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")
#
#     def closeEvent(self, event):
#         self.timer.stop()
#         super().closeEvent(event)
#
#
# class MidBarHandle(QSplitterHandle):
#     def __init__(self, orientation, parent, tabbar: QTabBar):
#         super().__init__(orientation, parent)
#         self.setObjectName("MidBarHandle")
#         self.setCursor(Qt.CursorShape.SplitVCursor)
#
#         self.frame = QFrame(self)
#         self.frame.setObjectName("MidBarFrame")
#         self.frame.setFrameShape(QFrame.Shape.NoFrame)
#         self.frame.setCursor(Qt.CursorShape.SplitVCursor)
#
#         self.tabbar = tabbar
#         self.tabbar.setParent(self.frame)
#         self.tabbar.setDrawBase(False)
#         self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)
#
#         self.tabbar.setMouseTracking(True)
#         self.tabbar.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
#
#         lay = _QHBoxLayout(self.frame)
#         lay.setContentsMargins(8, 4, 8, 4)
#         lay.addWidget(self.tabbar)
#
#         self.tabbar.installEventFilter(self)
#
#     def resizeEvent(self, ev):
#         super().resizeEvent(ev)
#         self.frame.setGeometry(0, 0, self.width(), self.height())
#
#     def eventFilter(self, obj, ev):
#         if obj is self.tabbar:
#             t = ev.type()
#             p = None
#             if t in (QEvent.Type.MouseMove, QEvent.Type.HoverMove):
#                 if hasattr(ev, "position"):
#                     p = ev.position().toPoint()
#                 elif hasattr(ev, "pos"):
#                     p = ev.pos()
#             elif t in (QEvent.Type.Enter, QEvent.Type.HoverEnter):
#                 p = self.tabbar.mapFromGlobal(QCursor.pos())
#             elif t in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
#                 self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)
#                 return False
#
#             if p is not None:
#                 idx = self.tabbar.tabAt(p)
#                 if idx != -1 and self.tabbar.isTabEnabled(idx):
#                     self.tabbar.setCursor(Qt.CursorShape.PointingHandCursor)
#                 else:
#                     self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)
#             return False
#
#         return QSplitterHandle.eventFilter(self, obj, ev)


# class MidBarSplitter(QSplitter):
#     def __init__(self, parent=None, tabbar: Optional[QTabBar] = None):
#         super().__init__(Qt.Orientation.Vertical, parent)
#         self._tabbar = tabbar
#
#     def createHandle(self):
#         return MidBarHandle(self.orientation(), self, self._tabbar)


class SplashScreenWidget(QtWidgets.QWidget, SplashScreen):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)


class MainApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.splash = None
        self.main_window = None

    def show_splash_screen(self):
        self.splash = SplashScreenWidget()
        self.splash.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        label = self.splash.findChild(QLabel, 'label')
        if label:
            gif_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "VDT_ani.gif")
            self.movie = QMovie(gif_path)
            label.setMovie(self.movie)
            self.movie.start()
        self.splash.show()

    def close_splash_screen(self):
        if self.splash:
            self.splash.close()

    def show_main_window(self):
        self.main_window = MyMainWindow()
        self.main_window.show()

    def start(self):
        self.show_splash_screen()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.initialize_app)
        self.timer.start(1200)

    def initialize_app(self):
        self.close_splash_screen()
        self.show_main_window()

# class ColumnFilterDialog(QDialog):
#     def __init__(self, *, headers: list[str], checked: set[str], locked: set[str], parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Select Columns")
#         self.setModal(True)
#         self.resize(420, 520)
#
#         self._locked = set(locked)
#         # only show headers that are NOT locked
#         visible_headers = [h for h in headers if h not in self._locked]
#
#         # widgets
#         from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLineEdit, QListView, QPushButton, QLabel
#         from PyQt6.QtGui import QStandardItemModel, QStandardItem
#         from PyQt6.QtCore import Qt, QSortFilterProxyModel
#
#         lay = QVBoxLayout(self)
#
#         # search
#         self.search = QLineEdit(self)
#         self.search.setPlaceholderText("Search columns…")
#         lay.addWidget(self.search)
#
#         # list (checkable)
#         self.model = QStandardItemModel(self)
#         for name in visible_headers:
#             it = QStandardItem(name)
#             it.setCheckable(True)
#             it.setCheckState(Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked)
#             it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
#             self.model.appendRow(it)
#
#         self.proxy = QSortFilterProxyModel(self)
#         self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
#         self.proxy.setFilterKeyColumn(0)
#         self.proxy.setSourceModel(self.model)
#
#         self.view = QListView(self)
#         self.view.setModel(self.proxy)
#         self.view.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
#         lay.addWidget(self.view, 1)
#
#         # quick actions
#         row = QHBoxLayout()
#         self.btnAll = QPushButton("Select All")
#         self.btnNone = QPushButton("Select None")
#         row.addWidget(self.btnAll)
#         row.addWidget(self.btnNone)
#         row.addStretch(1)
#         lay.addLayout(row)
#
#         # footer
#         foot = QHBoxLayout()
#         self.info = QLabel("")  # shows e.g. "12 selected"
#         foot.addWidget(self.info)
#         foot.addStretch(1)
#         self.btnCancel = QPushButton("Cancel")
#         self.btnApply = QPushButton("Apply")
#         foot.addWidget(self.btnCancel)
#         foot.addWidget(self.btnApply)
#         lay.addLayout(foot)
#
#         # wire up
#         self.search.textChanged.connect(self.proxy.setFilterFixedString)
#         self.btnAll.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
#         self.btnNone.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
#         self.btnCancel.clicked.connect(self.reject)
#         self.btnApply.clicked.connect(self.accept)
#
#         self._update_info()
#         self.model.itemChanged.connect(lambda *_: self._update_info())
#
#     def _set_all(self, state: Qt.CheckState):
#         for r in range(self.model.rowCount()):
#             self.model.item(r).setCheckState(state)
#         self._update_info()
#
#     def _update_info(self):
#         total = self.model.rowCount()
#         sel = sum(1 for r in range(total) if self.model.item(r).checkState() == Qt.CheckState.Checked)
#         self.info.setText(f"{sel} / {total} visible columns selected")
#
#     def selected_names(self) -> set[str]:
#         """Return the names selected in the dialog (locked not included, they’re enforced by caller)."""
#         out = set()
#         for r in range(self.model.rowCount()):
#             it = self.model.item(r)
#             if it.checkState() == Qt.CheckState.Checked:
#                 out.add(it.text())
#         return out


# class ConsoleRelayPage(QWebEnginePage):
#     """Catches JS console messages to ferry Plotly relayout/hover to Python."""
#     relayout_json = pyqtSignal(dict)    # emits on plotly_relayout
#     hover_json    = pyqtSignal(dict)    # (optional) emits on plotly_hover
#
#     def javaScriptConsoleMessage(self, level, msg, line, source):
#         if msg.startswith("RANGE:"):
#             import json
#             try:
#                 payload = json.loads(msg[6:])
#                 self.relayout_json.emit(payload)
#             except Exception:
#                 pass
#         elif msg.startswith("HOVER:"):
#             import json
#             try:
#                 payload = json.loads(msg[6:])
#                 self.hover_json.emit(payload)
#             except Exception:
#                 pass
#         # still let base handle logging
#         return super().javaScriptConsoleMessage(level, msg, line, source)
    
# class SyncPlotlyView(QWebEngineView):
#     """
#     A webview that, after the Plotly HTML loads, injects small JS hooks that:
#       - listen for plotly_relayout and emit to Python
#       - expose a JS function to apply ranges from Python
#     """
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._page = ConsoleRelayPage(self)
#         self.setPage(self._page)
#         self._installed = False
#         self._busy = False
#         self.loadFinished.connect(self._install_hooks_if_needed)
#
#     @property
#     def relay(self) -> ConsoleRelayPage:
#         return self._page
#
#     def _install_hooks_if_needed(self, ok: bool):
#         if not ok or self._installed:
#             return
#
#         js = r"""
#         (function(){
#           if (window.__pie_hooks_installed) return;
#           window.__pie_hooks_installed = true;
#
#           function getGraph(){
#             let g = document.querySelector('.js-plotly-plot');
#             if (!g) g = document.querySelector('div[data-plotly]');
#             if (!g) {
#               const cand = Array.from(document.querySelectorAll('div'));
#               g = cand.find(d => d && d._fullLayout);
#             }
#             return g;
#           }
#
#           function emitRange(){
#             const g = getGraph();
#             if (!g || !window.Plotly) return;
#             const x = g.layout?.xaxis?.range;
#             const y = g.layout?.yaxis?.range;
#             if (x && y) {
#               try {
#                 console.log('RANGE:' + JSON.stringify({'xaxis.range':x, 'yaxis.range':y}));
#               } catch(e){}
#             }
#           }
#
#           function install(){
#             const g = getGraph();
#             if (!g || !window.Plotly) { setTimeout(install, 200); return; }
#
#             // Catch all interactions that change zoom/pan
#             g.on('plotly_relayout', emitRange);
#             g.on('plotly_doubleclick', emitRange);
#             g.on('plotly_afterplot', emitRange);
#             g.on('plotly_redraw', emitRange);
#             g.on('plotly_autosize', emitRange);
#             g.on('plotly_restyle', emitRange);
#
#             //  Support mouse wheel zoom
#             g.addEventListener('wheel', () => setTimeout(emitRange, 200));
#
#             // 🔹 Support laptop touchpad pinch / scroll gestures
#             g.addEventListener('gesturechange', () => setTimeout(emitRange, 200));
#             g.addEventListener('touchmove', () => setTimeout(emitRange, 200));
#
#             // 🔹 Function called from Python to apply the other heatmap's range
#             window.__pie_applyRelayout = function(payload){
#               try {
#                 const g2 = getGraph();
#                 if (g2 && window.Plotly) Plotly.relayout(g2, payload);
#               } catch(err){}
#             };
#           }
#
#           install();
#         })();
#         """
#         self.page().runJavaScript(js)
#         self._installed = True
#
#
#     def apply_relayout(self, payload: dict):
#         """Apply ranges from the other view (with a feedback guard)."""
#         if self._busy:
#             return
#         self._busy = True
#         self.page().runJavaScript(
#             f"window.__pie_applyRelayout({payload!r});",
#             lambda _=None: self._clear_busy()
#         )
#
#     def _clear_busy(self):
#         from PyQt6.QtCore import QTimer
#         QTimer.singleShot(0, lambda: setattr(self, "_busy", False))






if __name__ == "__main__":
    # Handle special modes in the frozen EXE so it doesn't relaunch the main UI
    if "--run-digsheet-abs" in sys.argv:
        i = sys.argv.index("--run-digsheet-abs")
        tally_pkl = sys.argv[i+1]
        abs_val = sys.argv[i+2]
        project_root = sys.argv[i+3] if len(sys.argv) > i+3 else None

        dig_py_abs = resource_path(os.path.join("dig", "digsheet_abs.py"))

        # 👇 Pass along all arguments, including project_root
        sys.argv = [dig_py_abs, tally_pkl, abs_val]
        if project_root:
            sys.argv.append(project_root)

        runpy.run_path(dig_py_abs, run_name="__main__")
        sys.exit(0)



    if "--run-digsheet" in sys.argv:
        i = sys.argv.index("--run-digsheet")
        tally_pkl = sys.argv[i+1]
        project_root = sys.argv[i+2] if len(sys.argv) > i+2 else None

        dig_py = resource_path(os.path.join("dig", "dig_sheet.py"))
        sys.argv = [dig_py, tally_pkl]
        if project_root:
            sys.argv.append(project_root)

        runpy.run_path(dig_py, run_name="__main__")
        sys.exit(0)


    app = MainApp(sys.argv)
    app.start()
    sys.exit(app.exec())