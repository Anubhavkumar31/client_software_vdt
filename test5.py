
import tempfile, uuid, runpy
import os
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import plotly.graph_objects as go

# PyQt6 Core
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import (
    Qt, QSortFilterProxyModel, QThread, pyqtSignal,
    QTimer, QUrl, QEvent, QEventLoop
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
from pages.ERF import ERF
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
from Data_Gen.DataGenApp import ScriptRunnerApp  # noqa


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


SCROLLBAR_STYLE = """
QScrollBar:vertical {
    background: #2b2b2b;
    width: 14px;
}
QScrollBar::handle:vertical {
    background: #555;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #777;
}
QScrollBar:horizontal {
    background: #2b2b2b;
    height: 14px;
}
QScrollBar::handle:horizontal {
    background: #555;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #777;
}
"""

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
    table.verticalScrollBar().setSingleStep(2)


class PipeLoaderWorker(QThread):
    # Signals for communication
    progress_updated = pyqtSignal(int, str)  # progress %, message
    data_loaded = pyqtSignal(object)  # pandas DataFrame
    assets_loaded = pyqtSignal(dict)  # asset paths dictionary
    table_data_ready = pyqtSignal(object)  # processed table data
    error_occurred = pyqtSignal(str)  # error message
    time_estimate = pyqtSignal(float)  # estimated time remaining
    
    def __init__(self, pkl_path, project_root, pipe_idx):
        super().__init__()
        self.pkl_path = pkl_path
        self.project_root = project_root
        self.pipe_idx = pipe_idx
        self.start_time = None
        
    def run(self):
        try:
            self.start_time = time.time()
            total_steps = 6
            
            # Step 1: Load pickle data
            self.progress_updated.emit(10, "Loading pipe data...")
            df = pd.read_pickle(self.pkl_path)
            self.data_loaded.emit(df)
            self._update_time_estimate(1, total_steps)
            print(f"Loaded pickle with {len(df)} rows")
            
            # Step 2: Find pipe directory
            self.progress_updated.emit(25, "Locating asset files...")
            pipe_dir = self._find_pipe_directory()
            self._update_time_estimate(2, total_steps)
            
            # Step 3: Load HTML assets
            self.progress_updated.emit(40, "Loading chart assets...")
            assets = self._load_html_assets(pipe_dir)
            self.assets_loaded.emit(assets)
            self._update_time_estimate(3, total_steps)
            
            # Step 4: Load pipe tally data
            self.progress_updated.emit(60, "Processing pipe tally...")
            table_data = self._load_pipe_tally_data(pipe_dir)
            self._update_time_estimate(4, total_steps)
            
            # Step 5: Process table data
            self.progress_updated.emit(80, "Preparing table data...")
            if table_data is not None:
                processed_data = self._process_table_data(table_data)
                self.table_data_ready.emit(processed_data)
            else:
                self.table_data_ready.emit(None)
            self._update_time_estimate(5, total_steps)
            
            # Step 6: Complete
            self.progress_updated.emit(100, "Loading complete!")
            self._update_time_estimate(6, total_steps)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _update_time_estimate(self, current_step, total_steps):
        elapsed = time.time() - self.start_time
        if current_step > 0:
            avg_time_per_step = elapsed / current_step
            remaining_steps = total_steps - current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            self.time_estimate.emit(estimated_remaining)
    
    def _find_pipe_directory(self):
        candidates = [
            os.path.join(self.project_root, f"pipe_{self.pipe_idx}"),
            os.path.join(self.project_root, f"pipe-{self.pipe_idx}"),
            os.path.join(self.project_root, f"Pipe_{self.pipe_idx}"),
        ]
        return next((d for d in candidates if os.path.isdir(d)), None)
    
    def _load_html_assets(self, pipe_dir):
        if not pipe_dir:
            return {}
        
        def pick_one(patterns, exclude=None):
            exclude = exclude or []
            hits = []
            for pat in patterns:
                hits.extend(glob(os.path.join(pipe_dir, pat)))
            hits = [h for h in hits if not any(ex in os.path.basename(h).lower() for ex in (exclude or []))]
            exact = [h for h in hits if re.search(rf'{re.escape(str(self.pipe_idx))}\b', os.path.basename(h))]
            return exact[0] if exact else (hits[0] if hits else None)
        
        return {
            'hmap': pick_one(["*heatmap*.html"], exclude=["raw", "box"]),
            'hmap_r': pick_one(["*heatmap*raw*.html", "*raw*heatmap*.html"]),
            'heatmap_box': pick_one(["*heatmap*box*.html", "*box*heatmap*.html"]),
            'lplot': pick_one(["*lineplot*.html", "*line*.html"], exclude=["raw"]),
            'lplot_r': pick_one(["*lineplot*raw*.html", "*line*raw*.html"]),
            'pipe3d': pick_one(["*pipe3d*.html", "pipe3d*.html"]),
            'prox_linechart': pick_one(["proximity_linechart*.html", "*proximity_linechart*.html"])
        }
    
    def _load_pipe_tally_data(self, pipe_dir):
        if not pipe_dir:
            return None
        
        def pick_one(patterns, exclude=None):
            exclude = exclude or []
            hits = []
            for pat in patterns:
                hits.extend(glob(os.path.join(pipe_dir, pat)))
            hits = [h for h in hits if not any(ex in os.path.basename(h).lower() for ex in (exclude or []))]
            exact = [h for h in hits if re.search(rf'{re.escape(str(self.pipe_idx))}\b', os.path.basename(h))]
            return exact[0] if exact else (hits[0] if hits else None)
        
        pipe_tally_csv = pick_one([f"*PipeTally{self.pipe_idx}.csv", f"*PipeTally{self.pipe_idx}.xlsx"])
        if pipe_tally_csv:
            try:
                if pipe_tally_csv.lower().endswith(".csv"):
                    df = pd.read_csv(pipe_tally_csv)
                else:
                    df = pd.read_excel(pipe_tally_csv)
                return df
            except Exception:
                pass
        
        # Fallback to defects.csv
        ds_csv = pick_one(["*defectS*.csv", "*defects*.csv"])
        if ds_csv:
            try:
                return pd.read_csv(ds_csv)
            except Exception:
                pass
        
        return None
    
    def _process_table_data(self, df):
        if df is None or df.empty:
            return None
        
        # Check if this is a PipeTally file (has Feature Type column) or defects.csv
        if "Feature Type" in df.columns:
            # Filter Metal Loss defects
            original_count = len(df)
            df = df[df["Feature Type"].astype(str).str.strip().str.lower() == "metal loss"]
            
            if df.empty:
                return None
            
            # Round numeric columns
            numeric_columns = [
                'Depth %', 'Depth (mm)', 'ERF (ASME B31G)', 'Psafe (ASME B31G) Barg',
                'Abs. Distance (m)', 'Distance to U/S GW(m)', 'Length (mm)', 
                'Width (mm)', 'WT (mm)', 'Pipe Length (mm)'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').round(3)
        
        return df


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


class MidBarHandle(QSplitterHandle):
    def __init__(self, orientation, parent, tabbar: QTabBar):
        super().__init__(orientation, parent)
        self.setObjectName("MidBarHandle")
        self.setCursor(Qt.CursorShape.SplitVCursor)

        self.frame = QFrame(self)
        self.frame.setObjectName("MidBarFrame")
        self.frame.setFrameShape(QFrame.Shape.NoFrame)
        self.frame.setCursor(Qt.CursorShape.SplitVCursor)

        self.tabbar = tabbar
        self.tabbar.setParent(self.frame)
        self.tabbar.setDrawBase(False)
        self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)

        self.tabbar.setMouseTracking(True)
        self.tabbar.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        lay = _QHBoxLayout(self.frame)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.addWidget(self.tabbar)

        self.tabbar.installEventFilter(self)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.frame.setGeometry(0, 0, self.width(), self.height())

    def eventFilter(self, obj, ev):
        if obj is self.tabbar:
            t = ev.type()
            p = None
            if t in (QEvent.Type.MouseMove, QEvent.Type.HoverMove):
                if hasattr(ev, "position"):
                    p = ev.position().toPoint()
                elif hasattr(ev, "pos"):
                    p = ev.pos()
            elif t in (QEvent.Type.Enter, QEvent.Type.HoverEnter):
                p = self.tabbar.mapFromGlobal(QCursor.pos())
            elif t in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)
                return False

            if p is not None:
                idx = self.tabbar.tabAt(p)
                if idx != -1 and self.tabbar.isTabEnabled(idx):
                    self.tabbar.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)
            return False

        return QSplitterHandle.eventFilter(self, obj, ev)


class MidBarSplitter(QSplitter):
    def __init__(self, parent=None, tabbar: Optional[QTabBar] = None):
        super().__init__(Qt.Orientation.Vertical, parent)
        self._tabbar = tabbar

    def createHandle(self):
        return MidBarHandle(self.orientation(), self, self._tabbar)


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

class MyMainWindow(QMainWindow):
    REQUIRED_TALLY_COLS = [
        r"Abs. Distance (m)", r"Depth %", r"Type",
        r"ERF (ASME B31G)", r"Orientation o' clock"
    ]

    def __init__(self):
        super().__init__()
        self.ui = Form()
        self.ui.setupUi(self)
        self.child_windows = {}

        self._central_original = self.centralWidget()
        self._central_graphs = None
        self._graphs_widget = None

        self.project_is_open = False
        self.project_root = None
        self.pkl_files = []
        self.curr_data = None
        self.header_list = []
        self.pipe_tally = None
        self.prox_linechart = None

        self.hmap = None
        self.hmap_r = None
        self.lplot = None
        self.lplot_r = None
        self.pipe3d = None
        self.heatmap_box = None
        self._hscroll_ready = False  # gate to avoid big first jump
        self._hscroll_ready_main = False  # gate for main web view scrollbar
        # --- Splitter limits (pixels) ---
        self._min_top_h     = 220   # top pane (charts) must be at least this tall
        self._min_bottom_h  = 250   # bottom pane (tables/proximity) must be at least this tall
        self._max_top_h     = None  # or set e.g. 900
        self._max_bottom_h  = None  # or set e.g. 900
        self._right_margin_px = 300
        self._hscroll_ready_table = False  # gate for table scrollbar... # guard state
        self._reverting_tab = False
        self._last_allowed_tab_index = 0
        self._ui_ready = False  # set true after first layout/show

        # ✅ Initialize "No Defects Found" label
        self._no_defects_label = None

        # Threading setup
        self.loader_worker = None
        self.loading_dialog = None

        self.ui.comboBoxPipe.setEditable(True)
        self.ui.comboBoxPipe.clear()
        self.ui.comboBoxPipe.addItem("-Pipe-")
        self.ui.comboBoxPipe.setMaxVisibleItems(12)
        self.ui.comboBoxPipe.completer().setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.PopupCompletion
        )
        self.ui.comboBoxPipe.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        self.model = QStandardItemModel(self)
        self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.model)
        self.ui.tableView.setModel(self.proxy_model)

        # after other attrs like self.prox_linechart = None
        self._scroll_scale = 3  # try 5–10; higher => gentler/longer scroll
        setup_table_scroll(self.ui.tableView)

        # Digsheet button (ABS-based)
        self.btnDigsheetAbs = QPushButton("Digsheet")
        self.btnDigsheetAbs.setToolTip("Select an Absolute Distance cell in the defect table (on Heatmap/3D) to enable.")
        self.btnDigsheetAbs.setEnabled(False)
        try:
            _parent = self.ui.comboBoxPipe.parentWidget()
            _lay = _parent.layout()
            if _lay is not None:
                pos = _lay.indexOf(self.ui.comboBoxPipe)
                if pos != -1:
                    _lay.insertWidget(pos + 1, self.btnDigsheetAbs)
                else:
                    _lay.addWidget(self.btnDigsheetAbs)
            else:
                self.btnDigsheetAbs.setParent(_parent)
        except Exception:
            self.statusBar().addPermanentWidget(self.btnDigsheetAbs)
        self.btnDigsheetAbs.clicked.connect(self.open_digsheet_by_abs_from_selection)

        # Global event filter for disabled-button popups + tabbar clicks
        QtWidgets.QApplication.instance().installEventFilter(self)

        # Resizable splitter with tabbar-handle
        self.mid_tabbar = QTabBar()
        for i in range(self.ui.tabWidgetM.count()):
            self.mid_tabbar.addTab(self.ui.tabWidgetM.tabText(i))
        self.mid_tabbar.setExpanding(False)
        self.mid_tabbar.currentChanged.connect(lambda i: self.ui.tabWidgetM.setCurrentIndex(i))
        self.ui.tabWidgetM.currentChanged.connect(lambda i: self.mid_tabbar.setCurrentIndex(i))
        self.mid_tabbar.installEventFilter(self)  # intercept clicks on the mid tab bar
        self.ui.tabWidgetM.hide()
        self._build_splitter()

        # --- hook table signals so the button can update when user selects a row ---
        tw = self.ui.tableWidgetDefect
        tw.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        setup_table_scroll(tw)
        # update the button state whenever selection changes or a cell is clicked
        try:
            tw.itemSelectionChanged.disconnect()
        except Exception:
            pass
        tw.itemSelectionChanged.connect(self.update_digsheet_button_state)

        try:
            tw.cellClicked.disconnect()
        except Exception:
            pass
        tw.cellClicked.connect(lambda *_: self.update_digsheet_button_state())

        # ✅ Setup "No Defects Found" label after table is configured
        self._setup_no_defects_label()
        self._setup_create_project_label()
        self._show_create_project_message()

        self._setup_table_styling()

        self.canvas = PlotWindow(self, width=5, height=4, dpi=100)  # noqa

        self.setStatusBar(QStatusBar(self))
        self.current_message = 'App running'
        self.statusBar().showMessage(f'           Status:      {self.current_message}')
        right_container = QWidget()
        rl = QHBoxLayout(right_container); rl.setContentsMargins(0, 0, 0, 0)
        self.right_status_label = QLabel('0.0s    '); rl.addWidget(self.right_status_label)
        self.statusBar().addPermanentWidget(right_container)
        self.timer = QTimer(); self.timer.timeout.connect(self._tick)
        self._t0 = None

        self.setup_actions()
        self._connect_guarded_graph_controls()

        self.ui.comboBoxPipe.currentIndexChanged.connect(self.on_combo_index_changed)
        # replace direct tab switcher with guarded handler
        try:
            self.ui.tabWidgetM.currentChanged.disconnect()
        except Exception:
            pass
        self.ui.tabWidgetM.currentChanged.connect(self._on_middle_tab_changed)

        # initial UI state
        self._toggle_plot_ui(False)
        self._update_project_actions()  # Create enabled, Close disabled

        self.setStyleSheet("QMainWindow { background-color: #FFFFFF; color: #000000; }")
        self.showMaximized()

        # mark UI ready on next tick (prevents popup at startup)
        QTimer.singleShot(0, lambda: setattr(self, "_ui_ready", True))

        try:
            excel_path = resource_path("14inch Petrofac pipetally.xlsx")
            if os.path.exists(excel_path) and self.pipe_tally is None:
                self.pipe_tally = pd.read_excel(excel_path)
        except Exception:
            pass

        self._show_watermark()

    def _setup_no_defects_label(self):
        """Create and setup the 'No Defects Found' label with absolute positioning"""
        # Create a container widget to control sizing
        self._no_defects_container = QWidget()
        self._no_defects_container.setMaximumSize(500, 200)
        self._no_defects_container.setMinimumSize(400, 150)
        
        # Set size policy to prevent expansion
        self._no_defects_container.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed
        )
        
        # Create the layout for the container
        container_layout = QVBoxLayout(self._no_defects_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the actual label
        self._no_defects_label = QLabel("No Defects Found in this Pipe")
        self._no_defects_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_defects_label.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                color: #666666;
                font-weight: bold;
                background-color: #f8f8f8;
                border: 2px dashed #cccccc;
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
            }
        """)
        
        container_layout.addWidget(self._no_defects_label)
        self._no_defects_container.hide()
        
        # Add to parent WITHOUT layout management
        table_parent = self.ui.tableWidgetDefect.parentWidget()
        if table_parent:
            self._no_defects_container.setParent(table_parent)
            # Position at specific coordinates (x=100, y=50)
            self._no_defects_container.move(500, 50)  # ← TWEAK THESE VALUES

    def _setup_table_styling(self):
        """Setup bold headers and row numbers for tables"""
        # Style for tableView (pandas model)
        if hasattr(self.ui, 'tableView'):
            # Set header style
            self.ui.tableView.horizontalHeader().setStyleSheet("""
                QHeaderView::section {
                    font-weight: bold;
                    background-color: #f0f0f0;
                    border: 1px solid #d0d0d0;
                    padding: 5px;
                    text-align: center;
                }
            """)
            self.ui.tableView.verticalHeader().setStyleSheet("""
                QHeaderView::section {
                    font-weight: bold;
                    background-color: #f0f0f0;
                    border: 1px solid #d0d0d0;
                    padding: 5px;
                    text-align: center;
                    min-width: 40px;
                }
            """)
        
        # Style for tableWidgetDefect
        if hasattr(self.ui, 'tableWidgetDefect'):
            self.ui.tableWidgetDefect.horizontalHeader().setStyleSheet("""
                QHeaderView::section {
                    font-weight: bold;
                    background-color: #f0f0f0;
                    border: 1px solid #d0d0d0;
                    padding: 5px;
                    text-align: center;
                }
            """)
            self.ui.tableWidgetDefect.verticalHeader().setStyleSheet("""
                QHeaderView::section {
                    font-weight: bold;
                    background-color: #f0f0f0;
                    border: 1px solid #d0d0d0;
                    padding: 5px;
                    text-align: center;
                    min-width: 40px;
                }
            """)


    # ✅ Helper methods for showing/hiding message vs table
    def _show_no_defects_message(self):
        try:
            if hasattr(self, '_no_defects_container'):
                self._no_defects_container.show()
            if hasattr(self.ui, 'tableWidgetDefect'):
                self.ui.tableWidgetDefect.hide()
            if hasattr(self, 'table_scrollbar'):
                self.table_scrollbar.hide()   
        except Exception as e:
            print(f"Error showing no defects message: {e}")

    def _show_defects_table(self):
        try:
            if hasattr(self, '_no_defects_container') and self._no_defects_container:
                self._no_defects_container.hide()
            if hasattr(self, '_create_proj_container') and self._create_proj_container:
                self._create_proj_container.hide()

            if hasattr(self.ui, 'tableWidgetDefect'):
                self.ui.tableWidgetDefect.show()
            if hasattr(self, 'table_scrollbar'):
                self.table_scrollbar.show()   

            print("📊 Displaying defects table")
        except Exception as e:
            print(f"Error showing defects table: {e}")

    # ---------- action enable/disable toggler ----------
    # def _update_project_actions(self):
    #     a = self.ui
    #     act_create = getattr(a, "action_Create_Proj", None)
    #     act_close  = getattr(a, "action_Close_Proj", None)
    #     if isinstance(act_create, QAction):
    #         act_create.setEnabled(not self.project_is_open)
    #     if isinstance(act_close, QAction):
    #         act_close.setEnabled(self.project_is_open)

    def _update_project_actions(self):
        a = self.ui
        act_create = getattr(a, "action_Create_Proj", None)
        act_close = getattr(a, "action_Close_Proj", None)
        act_graphs = getattr(a, "action_graphs", None)

        if isinstance(act_create, QAction):
            act_create.setEnabled(not self.project_is_open)
        if isinstance(act_close, QAction):
            act_close.setEnabled(self.project_is_open)
        if isinstance(act_graphs, QAction):
            act_graphs.setEnabled(self.project_is_open)

    # ---------------------------------------------------

    # ---------- guarded connections for heatmap/line/3D ----------
    def _connect_guarded_graph_controls(self):
        a = self.ui
        # QActions from menu/toolbar
        action_map = [
            ("actionHeatmap", "Heatmap"),
            ("action_LineChart", "LineChart"),
            ("action_3D_Graph", "3D"),
        ]
        for aname, tab in action_map:
            act = getattr(a, aname, None)
            if isinstance(act, QAction):
                try: act.triggered.disconnect()
                except Exception: pass
                act.triggered.connect(lambda _=False, t=tab: self._guarded_open_tab(t))

        # Buttons / toolbuttons
        widget_map = [
            ("btnHeatmap", "Heatmap"),
            ("toolButtonHeatmap", "Heatmap"),
            ("btnLinechart", "LineChart"),
            ("toolButtonLine", "LineChart"),
            ("btn3D", "3D"),
            ("toolButton3D", "3D"),
        ]
        for wname, tab in widget_map:
            w = getattr(a, wname, None)
            if w is not None and hasattr(w, "clicked"):
                try: w.clicked.disconnect()
                except Exception: pass
                w.clicked.connect(lambda _=False, t=tab: self._guarded_open_tab(t))

    def _guarded_open_tab(self, tab_name: str):
        if not self.project_is_open:
            if self._ui_ready:
                self._project_required_popup()
            return
        wanted = {
            "Heatmap": {"Heatmap"},
            "LineChart": {"LineChart", "Line Chart", "Line Plot"},
            "3D": {"3D Graph", "3D"},
        }.get(tab_name, {tab_name})

        tw = self.ui.tabWidgetM
        for i in range(tw.count()):
            if tw.tabText(i) in wanted:
                tw.setCurrentIndex(i)
                self.tab_switcher2()
                return
        QMessageBox.information(self, "Tab not found", f"Could not locate tab: {tab_name}")

    def _make_topbar_row(
        self,
        object_name: str,
        parent_vbox: QVBoxLayout,
        bar_h: int = 14,
        *,
        left_px: int | None = None,     # ← fixed left spacer (px). None = expanding
        right_px: int | None = None,    # ← fixed right spacer (px). None = expanding
        pad_left: int = 8,              # tiny inner padding (optional)
        pad_right: int = 8
    ) -> QScrollBar:
        row_frame = QFrame()
        row_frame.setObjectName(object_name + "_container")
        row_frame.setFixedHeight(bar_h)
        row_frame.setStyleSheet("QFrame{margin:0;padding:0;border:0;background:transparent;}")

        row = QHBoxLayout(row_frame)
        row.setContentsMargins(pad_left, 0, pad_right, 0)
        row.setSpacing(0)

        # Left spacer
        if left_px is None:
            left_sp = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        else:
            left_sp = QSpacerItem(left_px, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        # Right spacer
        if right_px is None:
            right_sp = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        else:
            right_sp = QSpacerItem(right_px, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        bar = QScrollBar(Qt.Orientation.Horizontal)
        bar.setObjectName(object_name)
        bar.setFixedHeight(bar_h)
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row.addItem(left_sp)
        row.addWidget(bar)
        row.addItem(right_sp)

        parent_vbox.addWidget(row_frame)
        return bar

    def _build_splitter(self):
        # Create main web view with scrollbar
        self.main_web_page = QWidget()
        main_web_layout = QVBoxLayout(self.main_web_page)
        main_web_layout.setContentsMargins(0, 0, 0, 0)
        main_web_layout.setSpacing(0)

        # Main chart pane (heatmap / 3D / line main)
        self.main_web_scroll_area = QScrollArea()
        self.main_web_scroll_area.setWidgetResizable(False)
        self.main_web_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.main_web_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self.web_view = QWebEngineView()
        self.web_view.setFixedSize(2500, 600)
        self.main_web_scroll_area.setWidget(self.web_view)

        # 1) scrollable view
        main_web_layout.addWidget(self.main_web_scroll_area)

        # 2) tight top bar (MAIN)
        self.main_top_scrollbar = self._make_topbar_row("mainTopBar", main_web_layout, bar_h=10, left_px=1300, right_px=570)

        # Setup scrollbar sync for main web view
        main_inner_hbar = self.main_web_scroll_area.horizontalScrollBar()
        VIRTUAL_MAX = 2000

        def _eff_main_bounds():
            imin, imax = main_inner_hbar.minimum(), main_inner_hbar.maximum()
            eff_max = max(imin, imax - self._right_margin_px)
            return imin, eff_max

        def _map_main_top_to_inner(v_top: int) -> int:
            imin, eff_max = _eff_main_bounds()
            rng = max(1, eff_max - imin)
            return int(round(imin + (v_top / VIRTUAL_MAX) * rng))

        def _map_main_inner_to_top(v_inner: int) -> int:
            imin, eff_max = _eff_main_bounds()
            rng = max(1, eff_max - imin)
            return int(round(((v_inner - imin) / rng) * VIRTUAL_MAX))

        def _apply_main_fixed_range():
            self.main_top_scrollbar.blockSignals(True)
            self.main_top_scrollbar.setRange(0, VIRTUAL_MAX)
            self.main_top_scrollbar.setPageStep(100)
            self.main_top_scrollbar.setSingleStep(10)
            self.main_top_scrollbar.setValue(_map_main_inner_to_top(main_inner_hbar.value()))
            self.main_top_scrollbar.blockSignals(False)

        def _on_main_top_changed(v):
            if not self._hscroll_ready_main:
                return
            main_inner_hbar.setValue(_map_main_top_to_inner(v))

        def _on_main_inner_changed(v):
            if not self._hscroll_ready_main:
                return
            self.main_top_scrollbar.blockSignals(True)
            self.main_top_scrollbar.setValue(_map_main_inner_to_top(v))
            self.main_top_scrollbar.blockSignals(False)

        self.main_top_scrollbar.valueChanged.connect(_on_main_top_changed)
        main_inner_hbar.valueChanged.connect(_on_main_inner_changed)

        def _on_main_inner_range_changed(_min, _max):
            if _max > _min:
                self._hscroll_ready_main = True
                _apply_main_fixed_range()

        main_inner_hbar.rangeChanged.connect(_on_main_inner_range_changed)

        # Bottom stack setup
        self.bottom_stack = QStackedWidget()
        self.bottom_stack.setContentsMargins(0, 0, 0, 0)
        self.bottom_stack.currentChanged.connect(lambda idx: self._arm_topbar() if idx == 2 else None)

        # ---------------------------
        # Defect table page (bottom)
        # ---------------------------
        self.defect_table_page = QWidget()
        defect_layout = QVBoxLayout(self.defect_table_page)
        defect_layout.setContentsMargins(0, 0, 0, 0)
        defect_layout.setSpacing(0)

        # Re-parent table into this page
        old_parent_def = self.ui.tableWidgetDefect.parentWidget()
        if old_parent_def and old_parent_def.layout():
            try:
                old_parent_def.layout().removeWidget(self.ui.tableWidgetDefect)
            except Exception:
                pass
        self.ui.tableWidgetDefect.setParent(self.defect_table_page)
        self.ui.tableWidgetDefect.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Tight top bar (TABLE)
        self.table_scrollbar = self._make_topbar_row("tableTopBar", defect_layout, bar_h=10, left_px=1300, right_px=570)
        self.table_scrollbar.hide()
        # Table directly under the bar
        defect_layout.addWidget(self.ui.tableWidgetDefect)

        # Sync custom table bar with internal table hbar
        self._setup_table_scrollbar_sync()

        # ---------------------------
        # Data table page (model view)
        # ---------------------------
        self.data_table_page = QWidget()
        tl = QVBoxLayout(self.data_table_page)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        old_parent_data = self.ui.tableView.parentWidget()
        if old_parent_data and old_parent_data.layout():
            try:
                old_parent_data.layout().removeWidget(self.ui.tableView)
            except Exception:
                pass
        self.ui.tableView.setParent(None)
        self.ui.tableView.setVisible(True)
        tl.addWidget(self.ui.tableView)

        # ---------------------------
        # Proximity line chart page (bottom)
        # ---------------------------
        self.web_page = QWidget()
        web_layout = QVBoxLayout(self.web_page)
        web_layout.setContentsMargins(0, 0, 0, 0)
        web_layout.setSpacing(0)

        # Tight top bar (PROX)
        self.top_scrollbar = self._make_topbar_row("proxTopBar", web_layout, bar_h=10, left_px=1300, right_px=570)

        # Scroll area without bottom horizontal bar
        self.web_scroll_area = QScrollArea()
        self.web_scroll_area.setWidgetResizable(False)
        self.web_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.web_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self.web_view2 = QWebEngineView()
        self.web_view2.setFixedSize(2500, 600)
        self.web_scroll_area.setWidget(self.web_view2)

        web_layout.addWidget(self.web_scroll_area)

        # Apply scrollbar theme to bars/areas
        self._apply_scrollbar_theme("#6AA2FF")

        # Sync top scrollbar with scroll area for proximity line chart
        inner_hbar = self.web_scroll_area.horizontalScrollBar()

        def _eff_prox_bounds():
            imin, imax = inner_hbar.minimum(), inner_hbar.maximum()
            eff_max = max(imin, imax - self._right_margin_px)
            return imin, eff_max

        def _map_top_to_inner(v_top: int) -> int:
            imin, eff_max = _eff_prox_bounds()
            rng = max(1, eff_max - imin)
            return int(round(imin + (v_top / VIRTUAL_MAX) * rng))

        def _map_inner_to_top(v_inner: int) -> int:
            imin, eff_max = _eff_prox_bounds()
            rng = max(1, eff_max - imin)
            return int(round(((v_inner - imin) / rng) * VIRTUAL_MAX))

        def _apply_fixed_range():
            self.top_scrollbar.blockSignals(True)
            self.top_scrollbar.setRange(0, VIRTUAL_MAX)
            self.top_scrollbar.setPageStep(100)
            self.top_scrollbar.setSingleStep(10)
            self.top_scrollbar.setValue(_map_inner_to_top(inner_hbar.value()))
            self.top_scrollbar.blockSignals(False)

        def _on_top_changed(v):
            if not self._hscroll_ready:
                return
            inner_hbar.setValue(_map_top_to_inner(v))

        def _on_inner_changed(v):
            if not self._hscroll_ready:
                return
            self.top_scrollbar.blockSignals(True)
            self.top_scrollbar.setValue(_map_inner_to_top(v))
            self.top_scrollbar.blockSignals(False)

        self.top_scrollbar.valueChanged.connect(_on_top_changed)
        inner_hbar.valueChanged.connect(_on_inner_changed)

        def _on_inner_range_changed(_min, _max):
            if _max > _min:
                self._hscroll_ready = True
                _apply_fixed_range()

        inner_hbar.rangeChanged.connect(_on_inner_range_changed)

        # nudge once to ensure a rangeChanged after layout
        QTimer.singleShot(0, lambda: inner_hbar.setValue(inner_hbar.value()))
        QTimer.singleShot(0, lambda: main_inner_hbar.setValue(main_inner_hbar.value()))

        # Assemble bottom pages
        self.bottom_stack.addWidget(self.defect_table_page)
        self.bottom_stack.addWidget(self.data_table_page)
        self.bottom_stack.addWidget(self.web_page)

        # Splitter with mid tabbar
        self.splitter = MidBarSplitter(self, tabbar=self.mid_tabbar)
        self.splitter.addWidget(self.main_web_page)
        self.splitter.addWidget(self.bottom_stack)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(40)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStyleSheet("""
            QSplitter::handle#MidBarHandle { background: #16181c; }
            #MidBarFrame { background: #16181c; }
            QTabBar::tab { color: #d8d8d8; padding: 6px 14px; margin: 0px; border: 0; background: transparent; }
            QTabBar::tab:selected { color: white; font-weight: 600; }
        """)
        self.ui.verticalLayoutGraph.addWidget(self.splitter)
        QTimer.singleShot(0, lambda: self.splitter.setSizes([self.height() // 2, self.height() // 2]))

        def _constrain_splitter_sizes():
            sizes = self.splitter.sizes()
            if len(sizes) < 2:
                return
            total = sum(sizes)
            top, bot = sizes[0], sizes[1]

            min_top  = int(self._min_top_h or 0)
            min_bot  = int(self._min_bottom_h or 0)

            max_top_by_bot_min = max(0, total - min_bot)
            hard_max_top = self._max_top_h if self._max_top_h is not None else max_top_by_bot_min
            hard_max_top = min(hard_max_top, max_top_by_bot_min)

            top = max(min_top, min(top, hard_max_top))
            bot = total - top
            if self._max_bottom_h is not None:
                bot = min(bot, self._max_bottom_h)
                top = total - bot

            if bot < min_bot:
                bot = min_bot
                top = total - bot
                top = max(min_top, top)

            if [top, bot] != sizes[:2]:
                self.splitter.blockSignals(True)
                self.splitter.setSizes([top, bot])
                self.splitter.blockSignals(False)

        def _on_splitter_moved(*_):
            _constrain_splitter_sizes()
            if self._hscroll_ready:
                _apply_fixed_range()
            if self._hscroll_ready_main:
                _apply_main_fixed_range()
            QTimer.singleShot(10, self._refresh_table_scrollbars)

        self.splitter.splitterMoved.connect(_on_splitter_moved)

    def _setup_table_scrollbar_sync(self):
        """Setup synchronization between custom table scrollbar and table's internal scrollbar"""
        table_inner_hbar = self.ui.tableWidgetDefect.horizontalScrollBar()
        VIRTUAL_MAX = 2000

        def _eff_table_bounds():
            imin, imax = table_inner_hbar.minimum(), table_inner_hbar.maximum()
            eff_max = max(imin, imax - 50)  # Small right margin
            return imin, eff_max

        def _map_table_top_to_inner(v_top: int) -> int:
            imin, eff_max = _eff_table_bounds()
            rng = max(1, eff_max - imin)
            return int(round(imin + (v_top / VIRTUAL_MAX) * rng))

        def _map_table_inner_to_top(v_inner: int) -> int:
            imin, eff_max = _eff_table_bounds()
            rng = max(1, eff_max - imin)
            return int(round(((v_inner - imin) / rng) * VIRTUAL_MAX))

        def _apply_table_fixed_range():
            self.table_scrollbar.blockSignals(True)
            self.table_scrollbar.setRange(0, VIRTUAL_MAX)
            self.table_scrollbar.setPageStep(100)
            self.table_scrollbar.setSingleStep(10)
            self.table_scrollbar.setValue(_map_table_inner_to_top(table_inner_hbar.value()))
            self.table_scrollbar.blockSignals(False)

        def _on_table_top_changed(v):
            if not self._hscroll_ready_table:
                return
            table_inner_hbar.setValue(_map_table_top_to_inner(v))

        def _on_table_inner_changed(v):
            if not self._hscroll_ready_table:
                return
            self.table_scrollbar.blockSignals(True)
            self.table_scrollbar.setValue(_map_table_inner_to_top(v))
            self.table_scrollbar.blockSignals(False)

        # Connect the signals
        self.table_scrollbar.valueChanged.connect(_on_table_top_changed)
        table_inner_hbar.valueChanged.connect(_on_table_inner_changed)

        def _on_table_inner_range_changed(_min, _max):
            if _max > _min:
                self._hscroll_ready_table = True
                _apply_table_fixed_range()

        table_inner_hbar.rangeChanged.connect(_on_table_inner_range_changed)

        # Initial setup nudge
        QTimer.singleShot(100, lambda: table_inner_hbar.setValue(table_inner_hbar.value()))

    def _refresh_table_scrollbars(self):
        """Comprehensive table scrollbar refresh after container resize"""
        try:
            # For tableWidgetDefect (QTableWidget)
            if hasattr(self.ui, 'tableWidgetDefect'):
                tw = self.ui.tableWidgetDefect
                # Force scroll mode and policy
                tw.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
                tw.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
                
                # Set scroll speed
                tw.verticalScrollBar().setSingleStep(2)
                
                # Force geometry updates
                tw.viewport().update()
                tw.updateGeometry()
                tw.resizeRowsToContents()
                
                # Force scrollbar range recalculation
                vsb = tw.verticalScrollBar()
                vsb.update()
                # Trigger a fake scroll to force range update
                current_val = vsb.value()
                vsb.setValue(min(current_val + 1, vsb.maximum()))
                vsb.setValue(current_val)

            # For tableView (QTableView with model)
            if hasattr(self.ui, 'tableView'):
                tv = self.ui.tableView
                tv.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
                tv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
                
                # Set scroll speed
                tv.verticalScrollBar().setSingleStep(2)
                
                tv.viewport().update()
                tv.updateGeometry()
                
                vsb = tv.verticalScrollBar()
                vsb.update()
                current_val = vsb.value()
                vsb.setValue(min(current_val + 1, vsb.maximum()))
                vsb.setValue(current_val)
                
        except Exception as e:
            print(f"Error refreshing table scrollbars: {e}")

    def _show_watermark(self):
        try:
            html_path = Path(resource_path("ui/icons/VDT_watermark.html"))
            base_url = QUrl.fromLocalFile(str(html_path.parent) + "/")
            with open(html_path, "r", encoding="utf-8") as f:
                self.web_view.setHtml(f.read(), base_url)
        except Exception:
            self.web_view.setUrl(QUrl())
        self.bottom_stack.setCurrentIndex(0)
        self.web_view2.setUrl(QUrl())

    def _tick(self):
        if self._t0:
            dt = time.time() - self._t0
            self.right_status_label.setText(f"{dt:.1f}s    ")

    def set_loading(self, msg="Loading"):
        self.current_message = msg
        self.statusBar().showMessage(f'           Status:      {self.current_message}')
        self._t0 = time.time()
        self.timer.start(100)

    def set_idle(self):
        self.current_message = 'App running'
        self.statusBar().showMessage(f'           Status:      {self.current_message}')
        self.timer.stop()
        self._t0 = None
        self.right_status_label.setText("0.0s    ")

    def setup_actions(self):
        a = self.ui
        a.action_Create_Proj.triggered.connect(self.open_project)
        a.action_Close_Proj.triggered.connect(self.close_project)
        a.action_Quit.triggered.connect(self.quit_app)
        a.action_About.triggered.connect(self.open_About)
        a.actionAdmin_Panel.triggered.connect(self.open_Admin)
        a.action_ERF.triggered.connect(self.open_ERF)
        a.action_XYZ.triggered.connect(self.open_XYZ)
        self.ui.action_Export_Table.triggered.connect(self.gen_data)
        a.action_Final_Report.triggered.connect(self.open_Report)
        a.action_graphs.triggered.connect(self.open_graphs)
        a.action_Assessment.triggered.connect(self.open_Assessment)
        a.action_Cluster.triggered.connect(self.open_Cluster)
        a.action_Pipe_High.triggered.connect(self.open_PipeHigh)
        a.action_Pipe_Sch.triggered.connect(self.open_PipeScheme)
        a.actionMetal_Loss_Distribution_MLD.triggered.connect(self.open_CMLD)
        a.actionDepth_Based_Anomalies_Distribution_DBAD.triggered.connect(self.open_DBAD)
        a.actionERF_Based_Anomalies_Distribution_E_AD.triggered.connect(self.open_EAD)
        a.action_Custom.triggered.connect(self.add_plot_custom)
        a.action_Telemetry.triggered.connect(self.add_plot_tele)
        a.actionAnomalies_Distribution.triggered.connect(self.add_plot_ad)
        a.action_DefectDetect.triggered.connect(self.draw_boxes_v2)
        if hasattr(a, "pushButtonNext"): a.pushButtonNext.clicked.connect(lambda: None)
        if hasattr(a, "pushButtonPrev"): a.pushButtonPrev.clicked.connect(lambda: None)
        a.Final_Report.triggered.connect(self.open_Final_Report)
        a.action_Preliminary_Report.triggered.connect(self.open_Preliminary_Report)
        a.action__pipetally.triggered.connect(self.open_pipe_tally)
        a.action_Manual.triggered.connect(self.open_manual)
        a.actionStandard.triggered.connect(self.open_digs)  # original (by defect no.)

    def open_project(self):
        try:
            dlg = QFileDialog(self)
            dlg.setFileMode(QFileDialog.FileMode.Directory)
            dlg.setOption(QFileDialog.Option.ShowDirsOnly)
            dlg.setWindowTitle("Select Project Folder (PKLs + pipe_* folders)")
            if dlg.exec() != QFileDialog.DialogCode.Accepted:
                self.project_is_open = False
                self._toggle_plot_ui(False)
                self._show_watermark()
                self._update_project_actions()
                return

            root = dlg.selectedFiles()[0]
            self.project_root = root

            self.pipe_tally = None
            loaded_tally = self._auto_load_pipe_tally(root)
            if not loaded_tally:
                print("[pipe_tally] No tally file found in this project; graphs/reports will warn if needed.")

            self.pkl_files = [
                os.path.join(root, f)
                for f in os.listdir(root)
                if f.lower().endswith(".pkl")
            ]

            def nkey(path):
                filename = os.path.basename(path)
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", filename)]
            self.pkl_files.sort(key=nkey)

            cb = self.ui.comboBoxPipe
            cb.blockSignals(True)
            cb.clear()
            names = [os.path.splitext(os.path.basename(f))[0] for f in self.pkl_files]
            if names:
                cb.addItems(names)
                cb.setCurrentIndex(0)
            cb.lineEdit().setPlaceholderText("Type pipe number...")
            cb.completer().setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
            cb.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            cb.blockSignals(False)

            try:
                cb.lineEdit().returnPressed.disconnect()
            except Exception:
                pass
            cb.lineEdit().returnPressed.connect(self.jump_to_number)

            if self.pkl_files:
                self.project_is_open = True
                self._hide_create_project_message()
                self._toggle_plot_ui(True)
                self.load_selected_by_index(0)
            else:
                self.project_is_open = False
                self._toggle_plot_ui(False)
                self._show_watermark()
                QMessageBox.warning(self, "No PKLs", "No .pkl files found in the selected folder.")

            self._update_project_actions()
        except Exception as e:
            self.project_is_open = False
            self._toggle_plot_ui(False)
            self._show_watermark()
            self._update_project_actions()
            self.open_Error(e)

    def _apply_scrollbar_theme(self, _accent_ignored="#b8b8b8"):
        handle_radius = 10
        btn_wh = 22         # arrow circle size
        bar_h  = 14         # unified height for all top bars
        bar_w  = 16

        # SVG paths
        left  = resource_path("ui/icons/arrow_left.svg").replace("\\", "/")
        right = resource_path("ui/icons/arrow_right.svg").replace("\\", "/")
        up    = resource_path("ui/icons/arrow_up.svg").replace("\\", "/")
        down  = resource_path("ui/icons/arrow_down.svg").replace("\\", "/")

        # ---- HORIZONTAL: all three custom top bars ----
        h_style = f"""
        QScrollBar#proxTopBar:horizontal,
        QScrollBar#mainTopBar:horizontal,
        QScrollBar#tableTopBar:horizontal {{
            height:{bar_h}px;
            background: transparent;
            margin: 0 {btn_wh + 3}px 0 {btn_wh + 3}px;             /* kill outer margin */
            padding: 0;             /* kill inner padding */
            border: 0;
        }}

        /* handle (thumb) */
        QScrollBar#proxTopBar::handle:horizontal,
        QScrollBar#mainTopBar::handle:horizontal,
        QScrollBar#tableTopBar::handle:horizontal {{
            min-width: 40px;
            border-radius:{handle_radius}px;
            border:1px solid rgba(0,0,0,0.18);
            background:#d9d9d9;
        }}
        QScrollBar#proxTopBar::handle:horizontal:hover,
        QScrollBar#mainTopBar::handle:horizontal:hover,
        QScrollBar#tableTopBar::handle:horizontal:hover {{
            background:#bfbfbf; border-color:rgba(0,0,0,0.28);
        }}
        QScrollBar#proxTopBar::handle:horizontal:pressed,
        QScrollBar#mainTopBar::handle:horizontal:pressed,
        QScrollBar#tableTopBar::handle:horizontal:pressed {{
            background:#9a9a9a; border-color:rgba(0,0,0,0.38);
        }}

        /* arrows */
        QScrollBar#proxTopBar::sub-line:horizontal,
        QScrollBar#mainTopBar::sub-line:horizontal,
        QScrollBar#tableTopBar::sub-line:horizontal {{
            width:{btn_wh}px; height:{btn_wh}px;
            subcontrol-origin: margin;
            subcontrol-position: left;
            border: none;
            border-radius:{btn_wh//2}px;
            background:#e9e9e9;
            image: url("{left}");
        }}
        QScrollBar#proxTopBar::add-line:horizontal,
        QScrollBar#mainTopBar::add-line:horizontal,
        QScrollBar#tableTopBar::add-line:horizontal {{
            width:{btn_wh}px; height:{btn_wh}px;
            subcontrol-origin: margin;
            subcontrol-position: right;
            border: none;
            border-radius:{btn_wh//2}px;
            background:#e9e9e9;
            image: url("{right}");
        }}

        /* hover states */
        QScrollBar#proxTopBar::sub-line:horizontal:hover,
        QScrollBar#mainTopBar::sub-line:horizontal:hover,
        QScrollBar#tableTopBar::sub-line:horizontal:hover,
        QScrollBar#proxTopBar::add-line:horizontal:hover,
        QScrollBar#mainTopBar::add-line:horizontal:hover,
        QScrollBar#tableTopBar::add-line:horizontal:hover {{
            background:#d6d6d6;
        }}
        QScrollBar#proxTopBar::sub-line:horizontal:pressed,
        QScrollBar#mainTopBar::sub-line:horizontal:pressed,
        QScrollBar#tableTopBar::sub-line:horizontal:pressed,
        QScrollBar#proxTopBar::add-line:horizontal:pressed,
        QScrollBar#mainTopBar::add-line:horizontal:pressed,
        QScrollBar#tableTopBar::add-line:horizontal:pressed {{
            background:#c2c2c2;
        }}

        /* pages transparent */
        QScrollBar#proxTopBar::add-page:horizontal,
        QScrollBar#proxTopBar::sub-page:horizontal,
        QScrollBar#mainTopBar::add-page:horizontal,
        QScrollBar#mainTopBar::sub-page:horizontal,
        QScrollBar#tableTopBar::add-page:horizontal,
        QScrollBar#tableTopBar::sub-page:horizontal {{
            background: transparent;
        }}
        """

        # ---- VERTICAL: style the scrollareas' vertical bars (optional) ----
        v_style = f"""
        QScrollBar:vertical {{
            width:{bar_w}px;
            margin:{btn_wh + 8}px 0;
            background: transparent;
        }}
        QScrollBar::handle:vertical {{
            min-height:40px;
            border-radius:{handle_radius}px;
            border:1px solid rgba(0,0,0,0.18);
            background:#d9d9d9;
        }}
        QScrollBar::handle:vertical:hover  {{ background:#bfbfbf; border-color:rgba(0,0,0,0.28); }}
        QScrollBar::handle:vertical:pressed{{ background:#9a9a9a; border-color:rgba(0,0,0,0.38); }}

        QScrollBar::sub-line:vertical {{
            height:{btn_wh}px; width:{btn_wh}px;
            subcontrol-origin: margin;
            subcontrol-position: top;
            border:none; border-radius:{btn_wh//2}px;
            background:#e9e9e9;
            image: url("{up}");
        }}
        QScrollBar::add-line:vertical {{
            height:{btn_wh}px; width:{btn_wh}px;
            subcontrol-origin: margin;
            subcontrol-position: bottom;
            border:none; border-radius:{btn_wh//2}px;
            background:#e9e9e9;
            image: url("{down}");
        }}
        QScrollBar::sub-line:vertical:hover,
        QScrollBar::add-line:vertical:hover {{ background:#d6d6d6; }}
        QScrollBar::sub-line:vertical:pressed,
        QScrollBar::add-line:vertical:pressed {{ background:#c2c2c2; }}

        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{ background: transparent; }}
        """

        # apply
        self.top_scrollbar.setStyleSheet(h_style)
        self.main_top_scrollbar.setStyleSheet(h_style)
        self.table_scrollbar.setStyleSheet(h_style)
        self.web_scroll_area.verticalScrollBar().setStyleSheet(v_style)
        self.main_web_scroll_area.verticalScrollBar().setStyleSheet(v_style)

    def gen_data(self):
        try:
            if 'genData' not in self.child_windows or not self.child_windows['genData'].isVisible():
                self.script_runner_window = ScriptRunnerApp()
                self.script_runner_window.show()
                self.child_windows['genData'] = self.script_runner_window
            else:
                self.child_windows['genData'].raise_()
                self.child_windows['genData'].activateWindow()
        except Exception as e:
            self.open_Error(e)

    def _toggle_plot_ui(self, enabled: bool):
        tab_names = {"Heatmap", "LineChart", "Line Chart", "Line Plot", "3D Graph", "3D"}
        tw = self.ui.tabWidgetM
        for i in range(tw.count()):
            if tw.tabText(i) in tab_names:
                tw.setTabEnabled(i, enabled)
        try:
            self.update_digsheet_button_state()
        except Exception:
            pass

    def on_combo_index_changed(self, combo_idx: int):
        if not self.project_is_open or combo_idx < 0:
            return
        self.load_selected_by_index(combo_idx)

    def load_selected_by_index(self, idx: int):
        try:
            if idx < 0 or idx >= len(self.pkl_files):
                return
            
            pkl_path = self.pkl_files[idx]
            name = os.path.splitext(os.path.basename(pkl_path))[0]
            pipe_idx = self._extract_index(name)
            
            # Show loading dialog
            self.loading_dialog = ModernLoadingDialog(self)
            self.loading_dialog.show()
            
            # Create and start worker thread
            self.loader_worker = PipeLoaderWorker(pkl_path, self.project_root, pipe_idx)
            
            # Connect signals
            self.loader_worker.progress_updated.connect(self.loading_dialog.update_progress)
            self.loader_worker.time_estimate.connect(self.loading_dialog.update_time_estimate)
            self.loader_worker.data_loaded.connect(self.on_data_loaded)
            self.loader_worker.assets_loaded.connect(self.on_assets_loaded)
            self.loader_worker.table_data_ready.connect(self.on_table_data_ready)
            self.loader_worker.error_occurred.connect(self.on_loading_error)
            self.loader_worker.finished.connect(self.on_loading_finished)
            
            # Start the worker
            self.loader_worker.start()
            
        except Exception as e:
            self.open_Error(f"load_selected_by_index error: {e}")

    # def on_data_loaded(self, df):
    #     """Handle loaded DataFrame - runs on main thread"""
    #     self.curr_data = df
    #     self.header_list = list(df.columns)
        
    #     # Update table model
    #     self.model.clear()
    #     self.model.setHorizontalHeaderLabels([str(c) for c in df.columns])
    #     for _, row in df.iterrows():
    #         row_items = []
    #         for v in row.values:
    #             item = QStandardItem("" if pd.isna(v) else str(v))
    #             item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    #             row_items.append(item)
    #         self.model.appendRow(row_items)
        
    #     self.ui.tableView.setModel(self.model)
    #     self.ui.tableView.setSortingEnabled(True)
    def on_data_loaded(self, df):
        """Handle loaded DataFrame - runs on main thread"""
        self.curr_data = df
        self.header_list = list(df.columns)

        # Use lightweight model instead of building QStandardItem rows
        self.df_model = PandasModel(df)
        self.proxy_model.setSourceModel(self.df_model)
        self.ui.tableView.setModel(self.proxy_model)
        self.ui.tableView.setSortingEnabled(True)

    def on_assets_loaded(self, assets):
        """Handle loaded assets"""
        self.hmap = assets.get('hmap')
        self.hmap_r = assets.get('hmap_r')
        self.heatmap_box = assets.get('heatmap_box')
        self.lplot = assets.get('lplot')
        self.lplot_r = assets.get('lplot_r')
        self.pipe3d = assets.get('pipe3d')
        self.prox_linechart = assets.get('prox_linechart')

    def on_table_data_ready(self, df):
        """Handle processed table data"""
        if df is not None:
            # Check if this is a PipeTally format or defects.csv format
            if "Feature Type" in df.columns:
                self._populate_defect_table_from_tally(df)
            else:
                self._populate_defect_table_from_csv(df)
        else:
            self._show_no_defects_message()

    def on_loading_error(self, error_msg):
        """Handle loading errors"""
        if self.loading_dialog:
            self.loading_dialog.close()
        self.open_Error(f"Loading error: {error_msg}")

    def on_loading_finished(self):
        """Clean up when loading is complete"""
        # If the batched table fill is still running, delay closing the dialog
        if getattr(self, "_is_filling_table", False):
            self._pending_close_loader = True
        else:
            if self.loading_dialog:
                try:
                    self.loading_dialog.close()
                except Exception:
                    pass
                self.loading_dialog = None

        if self.loader_worker:
            self.loader_worker.deleteLater()
            self.loader_worker = None

        # Refresh the current view and topbars
        self._refresh_current_view()
        QTimer.singleShot(0, self._arm_topbar)
        QTimer.singleShot(0, self._arm_main_topbar)
        self.update_digsheet_button_state()


    @staticmethod
    def _extract_index(text: str) -> str:
        m = re.search(r'\d+', text)
        return m.group(0) if m else text

    # ✅ Updated _populate_defect_table_from_tally with "No Defects Found" logic
    def _populate_defect_table_from_tally(self, df: pd.DataFrame):
        """
        Show PipeTally CSV in the bottom defect table.
        - Keeps only Feature Type = Metal Loss
        - Normalizes columns
        - Fills table incrementally to avoid UI freeze
        """
        if df is None or df.empty:
            self._show_no_defects_message()
            return

        original_count = len(df)
        if "Feature Type" in df.columns:
            df = df[df["Feature Type"].astype(str).str.strip().str.lower() == "metal loss"]

        if df.empty:
            print(f"⚠️ No Metal Loss defects found (filtered from {original_count} rows)")
            self._show_no_defects_message()
            return

        # normalize column variants
        variants = {
            "s_no": "Defect_id",
            "Dimensions  Classification": "Dimensions Classification",
            "Depth % ": "Depth %",
            "Psafe (ASME B31G) bar": "Psafe (ASME B31G) Barg",
            "Pipe Length": "Pipe Length (mm)",
            "Length": "Length (mm)",
            "Width": "Width (mm)",
            "WT": "WT (mm)",
        }
        for src, dst in variants.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        # ensure Defect_id exists
        if "Defect_id" not in df.columns:
            df = df.reset_index(drop=True)
            df["Defect_id"] = np.arange(1, len(df) + 1)

        desired_cols = [
            "Defect_id","Abs. Distance (m)","Distance to U/S GW(m)","Pipe Number","Pipe Length (mm)",
            "Feature Identification","Dimensions Classification","Orientation o' clock","Length (mm)",
            "Width (mm)","WT (mm)","Depth %","Depth (mm)","Type","ERF (ASME B31G)","Psafe (ASME B31G) Barg",
            "Latitude","Longitude","Comment",
        ]
        for col in desired_cols:
            if col not in df.columns:
                df[col] = ""

        view = df[desired_cols].copy()

        tw = self.ui.tableWidgetDefect
        tw.clear()
        tw.setRowCount(len(view))
        tw.setColumnCount(len(view.columns))
        tw.setHorizontalHeaderLabels([str(c) for c in view.columns])
        tw.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set column widths
        column_widths = {
            'Defect_id': 150,
            'Abs. Distance (m)': 150,
            'Distance to U/S GW(m)': 150,
            'Pipe Number': 150,
            'Pipe Length (mm)': 150,
            'Feature Identification': 150,
            'Dimensions Classification': 150,
            'Orientation o\' clock': 150,
            'Length (mm)': 150,
            'Width (mm)': 150,
            'WT (mm)': 150,
            'Depth %': 150,
            'Depth (mm)': 150,
            'Type': 150,
            'ERF (ASME B31G)': 150,
            'Psafe (ASME B31G) Barg': 150,
            'Latitude': 150,
            'Longitude': 150,
            'Comment': 570
        }
        
        for c, col_name in enumerate(view.columns):
            if col_name in column_widths:
                tw.setColumnWidth(c, column_widths[col_name])
            else:
                tw.setColumnWidth(c, 100)

        self._show_defects_table()
        self._start_fill_qtablewidget_batched(view, chunk_size=300)

    
    def _start_fill_qtablewidget_batched(self, df: pd.DataFrame, *, chunk_size: int = 200):
        """Fill self.ui.tableWidgetDefect incrementally to keep UI responsive."""
        tw = self.ui.tableWidgetDefect
        columns = list(df.columns)

        tw.clear()
        tw.setColumnCount(len(columns))
        tw.setHorizontalHeaderLabels([str(c) for c in columns])
        tw.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        tw.setRowCount(len(df))            # preallocate
        tw.setUpdatesEnabled(False)        # defer UI updates

        # batching state
        self._table_fill_row = 0
        self._table_fill_df  = df
        self._table_fill_chunk = max(50, int(chunk_size))
        self._is_filling_table = True
        self._pending_close_loader = False

        # Start first batch
        QTimer.singleShot(0, self._fill_tablewidget_chunk)

    
    def _fill_tablewidget_chunk(self):
        """Append a batch of rows to QTableWidget without freezing UI."""
        tw = self.ui.tableWidgetDefect
        df = self._table_fill_df
        start = self._table_fill_row
        end   = min(start + self._table_fill_chunk, len(df))

        # Fill rows for this batch
        for r in range(start, end):
            row_vals = df.iloc[r].to_list()
            for c, v in enumerate(row_vals):
                if isinstance(v, float):
                    text = f"{v:.6g}"
                elif pd.isna(v):
                    text = ""
                else:
                    text = str(v)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Make items non-editable
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                
                tw.setItem(r, c, item)

        self._table_fill_row = end

        # update loader/progress
        if self.loading_dialog:
            done = end
            total = len(df)
            pct = int(100 * done / max(1, total))
            self.loading_dialog.update_progress(pct, f"Preparing table ({done}/{total})...")
            QtWidgets.QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)

        if end >= len(df):
            # finished
            tw.setUpdatesEnabled(True)
            tw.viewport().update()
            header = tw.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(False)
            self._is_filling_table = False

            # Apply styling after table is filled
            self._setup_table_styling()

            if self.loading_dialog and self._pending_close_loader:
                try:
                    self.loading_dialog.close()
                except Exception:
                    pass
                self.loading_dialog = None

            self.update_digsheet_button_state()
            QTimer.singleShot(0, self._refresh_table_scrollbars)
        else:
            # schedule next chunk (async → UI stays alive)
            QTimer.singleShot(0, self._fill_tablewidget_chunk)




    # ✅ Updated _populate_defect_table_from_csv with "No Defects Found" logic
    def _populate_defect_table_from_csv(self, df: pd.DataFrame):
        if df is None or df.empty:
            self._show_no_defects_message()
            return
        
        # Show table since we have data
        self._show_defects_table()
        
        header_indices = {
            'Defect_id': 0,
            'Absolute_Distance': 1,
            'Upstream_Distance': 2,
            'Feature_Type': 3,
            'Dimension_Class': 4,
            'Orientation': 5,
            'WT': 6,
            'Length': 7,
            'Width': 8,
            'Depth_Peak': 9
        }
        colmap_candidates = {
            'Box Number': 'Defect_id',
            'Defect_id': 'Defect_id',
            'Absolute Distance': 'Absolute_Distance',
            'Abs. Distance (m)': 'Absolute_Distance',
            'Upstream': 'Upstream_Distance',
            'Distance to U/S GW(m)': 'Upstream_Distance',
            'Type': 'Feature_Type',
            'Dimensions  Classification': 'Dimension_Class',
            "Orientation o' clock": 'Orientation',
            'Ori Val': 'Orientation',
            'WT (mm)': 'WT',
            'WT': 'WT',
            'Width': 'Width',
            'Breadth': 'Width',
            'Peak Value': 'Depth_Peak',
            'Depth % ': 'Depth_Peak',
            'Depth %': 'Depth_Peak',
            'Length': 'Length'
        }
        column_mapping = {}
        for src, dst in colmap_candidates.items():
            if src in df.columns: column_mapping[src] = dst

        tw = self.ui.tableWidgetDefect
        num_rows = len(df); num_cols = len(header_indices)
        tw.setRowCount(num_rows); tw.setColumnCount(num_cols)
        tw.setHorizontalHeaderLabels(list(header_indices.keys()))
        
        for r, (_, row) in enumerate(df.iterrows()):
            for src, dst in column_mapping.items():
                if dst in header_indices:
                    c = header_indices[dst]
                    v = row[src]
                    if isinstance(v, float): v = f"{v:.2f}"
                    item = QTableWidgetItem(str(v))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    # Make items non-editable
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    
                    tw.setItem(r, c, item)

        # Apply styling
        self._setup_table_styling()
        self.update_digsheet_button_state()


    # Guarded tab change handler (prevents switching when no project and shows popup)
    def _on_middle_tab_changed(self, index: int):
        if self._reverting_tab:
            return

        if not self.project_is_open:
            if self._ui_ready:
                self._project_required_popup()
            self._reverting_tab = True
            try:
                self.ui.tabWidgetM.setCurrentIndex(self._last_allowed_tab_index)
            finally:
                self._reverting_tab = False
            return

        self._last_allowed_tab_index = index
        self.tab_switcher2()
        self.update_digsheet_button_state()

    def tab_switcher2(self, *_):
        if not self.project_is_open:
            self._show_watermark()
            return
        try:
            tab = self.ui.tabWidgetM.tabText(self.ui.tabWidgetM.currentIndex())
            if tab == "Heatmap":
                if self.hmap:
                    self._load_scrollable_chart(self.web_view, self.hmap, min_w=2200, min_h=1400)
                else:
                    self.web_view.setUrl(QUrl())
                self.bottom_stack.setCurrentIndex(0)
                self.web_view2.setUrl(QUrl())
                # Setup scrollbar for heatmap
                QTimer.singleShot(100, self._arm_main_topbar)

            elif tab in ("LineChart", "Line Chart", "Line Plot"):
                if self.lplot:
                    self._load_scrollable_chart(self.web_view, self.lplot, min_w=2200, min_h=1400)
                else:
                    self.web_view.setUrl(QUrl())
                if self.prox_linechart and os.path.exists(self.prox_linechart):
                    self.bottom_stack.setCurrentIndex(2)
                    self._load_scrollable_chart(self.web_view2, self.prox_linechart, min_w=2000, min_h=900)
                    QTimer.singleShot(0, self._arm_topbar)
                    QTimer.singleShot(120, self._arm_topbar)  # small safety nudge
                    QTimer.singleShot(500, lambda: self._setup_web_view_scrollbars(self.web_view2))
                else:
                    self.bottom_stack.setCurrentIndex(0)
                    self.web_view2.setUrl(QUrl())
                # Setup scrollbar for line chart main view
                QTimer.singleShot(100, self._arm_main_topbar)

            elif tab in ("3D Graph", "3D"):
                if self.pipe3d:
                    try:
                        self._load_scrollable_chart(self.web_view, self.pipe3d, min_w=2200, min_h=1400)
                    except AttributeError:
                        self.web_view.setUrl(QUrl.fromLocalFile(self.pipe3d))
                else:
                    self.web_view.setUrl(QUrl())
                self.bottom_stack.setCurrentIndex(0)
                self.web_view2.setUrl(QUrl())
                # Setup scrollbar for 3D graph
                QTimer.singleShot(100, self._arm_main_topbar)

            self.update_digsheet_button_state()
        except Exception as e:
            self.open_Error(e)

    def _refresh_current_view(self):
        """Force the current tab to re-render with latest asset paths."""
        try:
            # Clear both views to avoid showing stale content
            self.web_view.setUrl(QUrl())
            self.web_view2.setUrl(QUrl())
        except Exception:
            pass
        # Let the event loop breathe, then render the right thing for the active tab
        QTimer.singleShot(0, self.tab_switcher2)

    def _load_scrollable_chart(self, view: QWebEngineView, html_path: str, min_w: int = 2200, min_h: int = 1400):
        if not html_path or not os.path.exists(html_path):
            view.setUrl(QUrl())
            return
        effective_min_w = max(0, min_w - self._right_margin_px)
        
        safe = html_path.replace('\\', '/')
        wrapper = f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
    * {{
        scrollbar-width: auto !important;
        -webkit-appearance: auto !important;
    }}
    html, body {{ 
        height: 100%; 
        margin: 0; 
        overflow: hidden;
    }}
    .wrap {{ 
        height: 100vh; 
        width: 100vw; 
        overflow: scroll !important;
        overflow-x: scroll !important;
        overflow-y: scroll !important;
        scrollbar-width: auto !important;
        -ms-overflow-style: scrollbar !important;
    }}
    .wrap::-webkit-scrollbar {{
        width: 18px !important;
        height: 18px !important;
        background: #f5f5f5 !important;
        display: block !important;
    }}
    .wrap::-webkit-scrollbar-track {{
        background: #e0e0e0 !important;
        border: 1px solid #ccc !important;
    }}
    .wrap::-webkit-scrollbar-thumb {{
        background: #666 !important;
        border: 2px solid #999 !important;
        border-radius: 2px !important;
    }}
    .wrap::-webkit-scrollbar-thumb:hover {{
        background: #333 !important;
    }}
    .wrap::-webkit-scrollbar-corner {{
        background: #e0e0e0 !important;
    }}
    iframe {{ 
        border: 0; 
        width: {effective_min_w}px !important; 
        height: {min_h}px !important;
        min-width: {effective_min_w}px !important;
        min-height: {min_h}px !important;
        display: block;
    }}
    </style>
    </head>
    <body>
    <div class="wrap" id="scrollContainer">
    <iframe sandbox="allow-scripts allow-same-origin allow-forms" src="file:///{safe}"></iframe>
    </div>
    <script>
    // Force scrollbars to be visible
    document.addEventListener('DOMContentLoaded', function() {{
    const container = document.getElementById('scrollContainer');
    
    // Force a reflow to ensure scrollbars appear
    container.style.overflow = 'hidden';
    setTimeout(() => {{
        container.style.overflow = 'scroll';
        container.style.overflowX = 'scroll';
        container.style.overflowY = 'scroll';
    }}, 10);
    
    // Trigger scroll to force scrollbar appearance
    container.scrollLeft = 1;
    container.scrollTop = 1;
    setTimeout(() => {{
        container.scrollLeft = 0;
        container.scrollTop = 0;
    }}, 100);
    }});
    </script>
    </body>
    </html>"""
        base = QUrl.fromLocalFile(os.path.dirname(html_path) + os.sep)
        view.setHtml(wrapper, base)

    def draw_boxes_v2(self):
        if not self.project_is_open:
            return
        try:
            if self.heatmap_box and os.path.exists(self.heatmap_box):
                self.web_view.setUrl(QUrl.fromLocalFile(self.heatmap_box))
            else:
                self.open_Error("Boxed heatmap not found for the selected pipe.")
        except Exception as e:
            self.open_Error(e)

    def minimize_tabs(self):
        self.ui.tabWidgetM.hide()

    def maximize_tabs(self):
        self.ui.tabWidgetM.show()

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

            import importlib.util
            spec = importlib.util.spec_from_file_location("graphs_ui", ui_file_path)
            graphs_ui = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(graphs_ui)

            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(12, 12, 12, 12)
            v.setSpacing(10)

            header = QHBoxLayout()
            back_btn = QPushButton("◀ Back")
            back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            back_btn.clicked.connect(self._close_graphs_view)
            title = QLabel("Graphs")
            title.setStyleSheet("font-weight: 600; font-size: 14pt;")
            header.addWidget(back_btn); header.addSpacing(12); header.addWidget(title); header.addStretch(1)
            v.addLayout(header)

            graphs_widget = graphs_ui.GraphApp(dataframe=self.pipe_tally)
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

    def _close_graphs_view(self):
        try:
            if self.centralWidget() is self._central_original:
                return
            graphs_central = self.takeCentralWidget()
            if graphs_central is not None:
                graphs_central.deleteLater()
            if self._central_original is not None:
                if self._central_original.parent() is not self:
                    self._central_original.setParent(self)
                self.setCentralWidget(self._central_original)
            self._graphs_widget = None
            self._central_graphs = None
        except Exception as e:
            print("⚠️ _close_graphs_view:", e)
    
    def _setup_web_view_scrollbars(self, web_view):
        """Force scrollbars to be visible on QWebEngineView"""
        try:
            # Enable scrollbars at the widget level
            web_view.page().settings().setAttribute(
                web_view.page().settings().WebAttribute.ShowScrollBars, True
            )
            
            # Inject CSS to force scrollbar visibility
            css = """
            ::-webkit-scrollbar { 
                width: 16px !important; 
                height: 16px !important; 
                display: block !important; 
            }
            ::-webkit-scrollbar-track { 
                background: #f0f0f0 !important; 
            }
            ::-webkit-scrollbar-thumb { 
                background: #888 !important; 
                border-radius: 4px !important; 
            }
            html, body { 
                overflow: scroll !important; 
            }
            """
            
            web_view.page().runJavaScript(f"""
            var style = document.createElement('style');
            style.textContent = `{css}`;
            document.head.appendChild(style);
            """)
        except Exception as e:
            print(f"Error setting up scrollbars: {e}")

    def _auto_load_pipe_tally(self, root: str) -> bool:
        candidates = [
            os.path.join(root, "pipetally", "pipe_tally.xlsx"),
            os.path.join(root, "pipetally", "pipe_tally.csv"),
            os.path.join(root, "pipe_tally.xlsx"),
            os.path.join(root, "pipe_tally.csv"),
        ]
        for d in (root, os.path.join(root, "pipetally")):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    name = f.lower()
                    if name.endswith((".xlsx", ".xls", ".csv")):
                        candidates.append(os.path.join(d, f))
        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if not os.path.exists(path): continue
            try:
                if path.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(path)
                df.columns = [str(c).strip() for c in df.columns]

                # ✅ Round numeric columns to 3 decimal places
                numeric_columns = [
                    'Depth %', 'Depth (mm)', 'ERF (ASME B31G)', 'Psafe (ASME B31G) Barg',
                    'Abs. Distance (m)', 'Distance to U/S GW(m)', 'Length (mm)', 
                    'Width (mm)', 'WT (mm)', 'Pipe Length (mm)'
                ]
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').round(3)

                missing = [c for c in self.REQUIRED_TALLY_COLS if c not in df.columns]
                if missing:
                    print(f"[pipe_tally] Loaded {os.path.basename(path)} (missing cols: {missing})")
                else:
                    print(f"[pipe_tally] Loaded {os.path.basename(path)}")
                self.pipe_tally = df
                return True
            except Exception as e:
                print(f"[pipe_tally] Failed to load {path}: {e}")
        self.pipe_tally = None
        return False
    
    def open_XYZ(self):
        try:
            if sys.platform == "win32":
                path = r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe"
            elif sys.platform == "darwin":
                path = "/Applications/Google Earth Pro.app/Contents/MacOS/Google Earth Pro"
            else:
                path = "/usr/bin/google-earth-pro"
            if os.path.exists(path):
                subprocess.Popen([path])
        except Exception:
            pass

    def open_Cluster(self):
        Cluster_Dialog().exec()
        
    def open_Ptal(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Pipe Tally File", "", "CSV/Excel Files (*.csv *.xlsx *.xls);;All Files (*)"
            )
            if not file_path: return
            self.pipe_tally = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
            QMessageBox.information(self, "Pipe Tally", "Pipe tally loaded successfully.")
            self._toggle_plot_ui(self.project_is_open)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Pipe tally load failed: {e}")

    def jump_to_number(self):
        if not self.project_is_open:
            return
        text = self.ui.comboBoxPipe.currentText().strip()
        if not text: return
        try:
            base_names = [os.path.splitext(os.path.basename(f))[0] for f in self.pkl_files]
            if text in base_names:
                idx = base_names.index(text)
            else:
                idx = next((i for i, n in enumerate(base_names) if re.search(rf'\b{text}\b', n)), None)
                if idx is None: return
            self.ui.comboBoxPipe.setCurrentIndex(idx)
        except Exception as e:
            self.open_Error(f"Jump error: {e}")

    def open_About(self):
        About_Dialog().exec()

    def open_Admin(self):
        self.ap = Admin_Panel(); self.ap.show()

    def open_Assessment(self):
        Assess_Dialog().exec()

    def open_PipeHigh(self):
        try:
            from pages.Pipe_Highlights import run_app
            run_app()
        except Exception as e:
            self.open_Error(f"Error running Pipe Highlight:\n{e}")

    def open_PipeScheme(self):
        try:
            from pipeline_schema.pipeline_schema import run_app
            run_app()
        except Exception as e:
            self.open_Error(f"Error running Pipeline Schema:\n{e}")

    def open_Report(self):
        cols = [r"Abs. Distance (m)", r"Depth %", r"Type", r"ERF (ASME B31G)", r"Orientation o' clock"]
        if not isinstance(self.pipe_tally, pd.DataFrame):
            QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded."); return
        for c in cols:
            if c not in self.pipe_tally.columns:
                QMessageBox.critical(self, "Error", f"Missing column: {c}"); return
        fil = self.pipe_tally[cols].copy()
        fil = fil.dropna(subset=["Abs. Distance (m)"])
        fil["Abs. Distance (m)"] = fil["Abs. Distance (m)"].astype(int)
        fil["Depth %"] = pd.to_numeric(fil["Depth %"], errors='coerce')
        fil["Type"] = fil["Type"].astype(str)
        fil["ERF (ASME B31G)"] = pd.to_numeric(fil["ERF (ASME B31G)"], errors='coerce')
        fil[r"Orientation o' clock"] = fil[r"Orientation o' clock"].astype(str)
        fil["Surface Location"] = fil["Type"].apply(
                        lambda x: "Internal" if "Internal" in x else ("External" if "External" in x else "Unknown")
        )
        self.fr = Report(fil); self.fr.show()

    def open_ERF(self):
        self.erf = ERF()
        def update_result():
            OD = self.erf.doubleSpinBox.value()
            WT = self.erf.doubleSpinBox_3.value()
            SMYS = self.erf.doubleSpinBox_2.value()
            MAOP = self.erf.doubleSpinBox_4.value()
            SF = self.erf.doubleSpinBox_5.value()
            Axial_L = self.erf.doubleSpinBox_8.value()
            Depth_P = self.erf.doubleSpinBox_9.value()
            if OD == 0 or WT == 0 or SF == 0:
                self.erf.lineEdit_2.setText("-"); self.erf.lineEdit_3.setText("-"); return
            flow_stress = 1.1 * SMYS
            z_factor = (Axial_L ** 2) / (OD * WT)
            M = (1 + 0.8 * z_factor) ** 0.5
            y = 1 - 2/3 * Depth_P / WT
            z = 1 - 2/3 * Depth_P / WT / M
            k = y / z
            S = (flow_stress * k) if z_factor <= 20 else (flow_stress * (1 - Depth_P / WT))
            EFP = (2 * S * WT) / OD
            PSafe = EFP / SF if SF else 0
            if PSafe == 0:
                self.erf.lineEdit_2.setText("-"); self.erf.lineEdit_3.setText("-"); return
            ERFv = MAOP / PSafe
            self.erf.lineEdit_2.setText(f"{ERFv:.2f}")
            self.erf.lineEdit_3.setText(f"{PSafe:.2f}")
            import numpy as np
            def calc_B(d_over_t):
                if d_over_t >= 0.175:
                    B = np.sqrt(((d_over_t / (1.1 * d_over_t - 0.15)) ** 2) - 1)
                    return B if B <= 4 else 4
                return 4
            xs = np.linspace(0, 1, 100)
            ys = [calc_B(x) for x in xs]
            Xc = Axial_L / 300; Yc = Depth_P / 20
            color = 'green' if Yc < calc_B(Xc) else 'red'
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='ASME B31G'))
            fig.add_trace(go.Scatter(x=[Xc], y=[Yc], mode='markers',
                                     marker=dict(color=color, size=10),
                                     name='Defect'))
            fig.update_layout(xaxis_title='Axial Length (mm)', yaxis_title='Peak Depth', height=450, width=1000)
            fp = resource_path('backend/files/ASME.html'); fig.write_html(fp)
            self.erf.web_viewERF.setUrl(QUrl.fromLocalFile(fp))
        for w in (self.erf.doubleSpinBox, self.erf.doubleSpinBox_3, self.erf.doubleSpinBox_2,
                  self.erf.doubleSpinBox_4, self.erf.doubleSpinBox_5,
                  self.erf.doubleSpinBox_8, self.erf.doubleSpinBox_9):
            w.valueChanged.connect(update_result)
        update_result()
        self.erf.show()
       
    def open_Final_Report(self):
        p = resource_path(os.path.join("final_report", "Final_Report.pdf"))
        if os.path.exists(p): os.startfile(p)
        else: self.open_Error("Final report PDF not found.")

    def open_Preliminary_Report(self):
        p = resource_path(os.path.join("preliminary_report", "Preliminary_Report.pdf"))
        if os.path.exists(p): os.startfile(p)
        else: self.open_Error("Prelimary report is not found.")

    def open_pipe_tally(self):
        p = resource_path(os.path.join("pipetally", "pipe_tally.xlsx"))
        if os.path.exists(p): os.startfile(p)
        else: self.open_Error("Pipetally not found.")

    def open_manual(self):
        p = resource_path(os.path.join("manual", "user_manual.pdf"))
        if os.path.exists(p): os.startfile(p)
        else: self.open_Error("User manual is not found.")

    def add_plot_custom(self):
        try:
            self.cplot_widget = customPlot(self.header_list)
            self.ui.graphLayout.addWidget(self.cplot_widget)
            self.cplot_widget.closeCustom.clicked.connect(self.cplot_widget.close_window)
            self.cplot_widget.comboBox.currentIndexChanged.connect(self.plot_c)
        except Exception as e:
            self.open_Error(e)

    def plot_c(self):
        try:
            y_label = self.cplot_widget.comboBox.currentText()
            x_label = self.cplot_widget.comboBox_2.currentText()
            if x_label not in self.curr_data or y_label not in self.curr_data:
                raise ValueError("Selected labels are not in the current data.")
            x_data = self.curr_data[x_label]; y_data = self.curr_data[y_label]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=y_label))
            fig.update_layout(title=f'{y_label} vs {x_label}', xaxis_title=x_label, yaxis_title=y_label, height=450)
            fp = resource_path('backend/files/customplot.html'); fig.write_html(fp)
            self.cplot_widget.webviewCustom.setUrl(QUrl.fromLocalFile(fp))
            self.web_view.setUrl(QUrl.fromLocalFile(fp))
        except Exception as e:
            self.open_Error(e)

    def add_plot_tele(self):
        try:
            if self.curr_data is None or self.curr_data.empty:
                QMessageBox.critical(self, "Error", "Please load a project first."); return
            import re as _re
            tlist = [c for c in self.header_list if _re.match(r'^F\d+', c)]
            if not tlist:
                QMessageBox.warning(self, "No Telemetry Data", "No telemetry (F...) columns found."); return
            self.tplot_widget = telePlot(tlist)
            self.ui.graphLayout.addWidget(self.tplot_widget)
            self.tplot_widget.closeTele.clicked.connect(self.tplot_widget.close_window)
            self.tplot_widget.checkBox.stateChanged.connect(self.magnetisation)
            self.tplot_widget.checkBox_2.stateChanged.connect(self.velocity)
            self.tplot_widget.comboBox.currentIndexChanged.connect(self.plot_telemetry)
            if len(tlist) > 0:
                self.tplot_widget.comboBox.setCurrentIndex(1)
                self.plot_telemetry()
        except Exception as e:
            self.open_Error(e)

    def magnetisation(self):
        try:
            if not self.tplot_widget.checkBox.isChecked():
                fp = resource_path('backend/files/telemetryplot.html')
                go.Figure().write_html(fp)
            else:
                filtered = [c for c in self.curr_data.columns if c.startswith('F')]
                tele = self.curr_data[filtered]
                mag = tele.mean(axis=1) * 0.0004854
                x = self.curr_data['ODDO1']; y = mag
                fig = go.Figure(); fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Mag'))
                fig.update_layout(title='Magnetisation View', xaxis_title='Oddometer (mm)', yaxis_title='Magnetisation', height=450)
                fp = resource_path('backend/files/magnetisation.html')
                fig.write_html(fp)
            self.tplot_widget.webviewTele.setUrl(QUrl.fromLocalFile(fp))
            self.web_view.setUrl(QUrl.fromLocalFile(fp))
        except Exception as e:
            self.open_Error(e)

    def velocity(self):
        try:
            if not self.tplot_widget.checkBox_2.isChecked():
                fp = resource_path('backend/files/telemetryplot.html')
                go.Figure().write_html(fp)
            else:
                oddo = self.curr_data['ODDO1'].to_numpy()
                vel = [(oddo[i+1]-oddo[i]) / 0.000666667 for i in range(len(oddo)-1)]
                if vel: vel.append(vel[-1])
                fig = go.Figure(); fig.add_trace(go.Scatter(x=oddo, y=vel, mode='lines', name='Velocity'))
                fig.update_layout(title='Velocity View', xaxis_title='Oddometer(mm)', yaxis_title='Velocity', height=450)
                fp = resource_path('backend/files/velocity.html'); fig.write_html(fp)
            self.tplot_widget.webviewTele.setUrl(QUrl.fromLocalFile(fp))
            self.web_view.setUrl(QUrl.fromLocalFile(fp))
        except Exception as e:
            self.open_Error(e)

    def plot_telemetry(self):
        try:
            param = self.tplot_widget.comboBox.currentText()
            if param == "-Select-" or param not in self.curr_data.columns: return
            filtered = [c for c in self.curr_data.columns if c.startswith('F')]
            tele = self.curr_data[filtered]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tele.index, y=tele[param], mode='lines', name=param))
            fig.update_layout(title=f'Telemetry Plot for {param}', xaxis_title='Counter', yaxis_title=param, height=450)
            fp = resource_path("telemetryplot.html"); fig.write_html(fp)
            self.tplot_widget.webviewTele.setUrl(QUrl.fromLocalFile(fp))
            self.web_view.setUrl(QUrl.fromLocalFile(fp))
        except Exception as e:
            self.open_Error(e)

    def add_plot_ad(self):
        try:
            self.adplot_widget = adPlot(self.curr_data if isinstance(self.curr_data, list) else self.curr_data)
            self.ui.graphLayout.addWidget(self.adplot_widget)
            self.adplot_widget.closeAnamoly.clicked.connect(self.adplot_widget.close_window)
        except Exception as e:
            self.open_Error(e)

    def on_row_selection_changed(self, *_):
        idxs = self.ui.tableWidgetDefect.selectionModel().selectedRows()
        if not idxs:
            self.update_digsheet_button_state()
            return
        row = idxs[0].row()
        item = self.ui.tableWidgetDefect.item(row, 0)
        if item:
            defect_id = item.text()
            try:
                self.web_view.page().runJavaScript(f"highlightBox({defect_id});")
            except Exception:
                pass
        self.update_digsheet_button_state()

    def _get_selected_abs_distance_from_defect_table(self) -> Optional[str]:
        tw = self.ui.tableWidgetDefect
        if tw.rowCount() == 0 or tw.columnCount() == 0:
            QMessageBox.warning(self, "No data", "Defect table is empty.")
            return None

        abs_col = self._abs_col_index_silent()
        if abs_col is None:
            QMessageBox.warning(self, "Missing column", "Could not find the Absolute Distance column.")
            return None

        sel_model = tw.selectionModel()
        rows = [idx.row() for idx in sel_model.selectedRows()] or [i.row() for i in tw.selectedIndexes()]
        rows = list(dict.fromkeys(rows))
        if len(rows) != 1:
            QMessageBox.information(self, "Select one row", "Please select exactly one row in the defect table.")
            return None

        item = tw.item(rows[0], abs_col)
        if item is None or not item.text().strip():
            QMessageBox.warning(self, "No Absolute Distance", "Selected row has empty Absolute Distance.")
            return None

        return item.text().strip()

    # ---------------------------
    # Helpers + global event filter popups
    # ---------------------------
    def _show_disabled_digsheet_hint(self):
        QMessageBox.information(
            self,
            "Digsheet",
            "Please choose <b>Absolute Distance</b> from the defect table below to generate the digsheet."
        )

    def _project_required_popup(self):
        QMessageBox.information(
            self,
            "Project Required",
            "Please create project before proceeding further."
        )

    def _project_gate_targets(self):
        names = [
            "btnHeatmap", "btnLinechart", "btn3D",
            "toolButtonHeatmap", "toolButtonLine", "toolButton3D",
        ]
        widgets = [self.btnDigsheetAbs]
        for n in names:
            w = getattr(self.ui, n, None)
            if w is not None:
                widgets.append(w)
        return [w for w in widgets if hasattr(w, "mapFromGlobal")]

    def eventFilter(self, obj, ev):
        try:
            # Intercept mid tab bar clicks when no project (so repeated clicks also show popup)
            if obj is self.mid_tabbar and ev.type() == QEvent.Type.MouseButtonPress:
                if self._ui_ready and not self.project_is_open:
                    self._project_required_popup()
                    return True  # consume

            if ev.type() == QEvent.Type.MouseButtonPress:
                # PROJECT GATE for widget buttons
                if self._ui_ready and not self.project_is_open:
                    if hasattr(ev, "globalPosition"):
                        gp = ev.globalPosition().toPoint()
                    else:
                        gp = ev.globalPos()
                    for w in self._project_gate_targets():
                        if w and w.isVisible():
                            local = w.mapFromGlobal(gp)
                            if w.rect().contains(local):
                                self._project_required_popup()
                                return True  # consume

                # DISABLED DIGSHEET HINT
                btn = getattr(self, "btnDigsheetAbs", None)
                if btn is not None and btn.isVisible() and not btn.isEnabled():
                    if hasattr(ev, "globalPosition"):
                        gp = ev.globalPosition().toPoint()
                    else:
                        gp = ev.globalPos()
                    local = btn.mapFromGlobal(gp)
                    if btn.rect().contains(local):
                        self._show_disabled_digsheet_hint()
                        return True  # consume
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # ---------------------------
    # Digsheet enable logic + cursor/tooltip polish
    # ---------------------------
    def _abs_col_candidates(self):
        return ("Absolute_Distance", "Abs. Distance (m)", "Absolute Distance")

    def _abs_col_index_silent(self) -> Optional[int]:
        tw = self.ui.tableWidgetDefect
        if tw.columnCount() == 0:
            return None
        for c in range(tw.columnCount()):
            hdr = tw.horizontalHeaderItem(c)
            name = hdr.text().strip() if hdr else ""
            if name in self._abs_col_candidates():
                return c
        return 1 if tw.columnCount() > 1 else (0 if tw.columnCount() == 1 else None)

    def _has_valid_abs_selection(self) -> bool:
        tw = self.ui.tableWidgetDefect
        if tw.rowCount() == 0 or tw.columnCount() == 0:
            return False

        abs_col = self._abs_col_index_silent()
        if abs_col is None:
            return False

        sel_model = tw.selectionModel()
        if sel_model is None:
            return False

        # Prefer row-based selection (what we configured). Fallback to generic indexes.
        rows = [idx.row() for idx in sel_model.selectedRows()] or [i.row() for i in tw.selectedIndexes()]
        rows = list(dict.fromkeys(rows))  # unique, order preserved

        if len(rows) != 1:
            return False

        row = rows[0]
        item = tw.item(row, abs_col)
        return bool(item and item.text().strip())

    def _is_graph_tab_ok(self) -> bool:
        tab = self.ui.tabWidgetM.tabText(self.ui.tabWidgetM.currentIndex())
        return tab in ("Heatmap", "3D Graph", "3D")

    def update_digsheet_button_state(self):
        can_show = (
            self.project_is_open
            and isinstance(self.pipe_tally, pd.DataFrame)
            and self._is_graph_tab_ok()
            and self._has_valid_abs_selection()
        )
        self.btnDigsheetAbs.setEnabled(bool(can_show))

        if can_show:
            self.btnDigsheetAbs.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btnDigsheetAbs.setToolTip("Click to generate Digsheet for the selected Absolute Distance.")
        else:
            self.btnDigsheetAbs.setCursor(Qt.CursorShape.ForbiddenCursor)
            self.btnDigsheetAbs.setToolTip("Select an Absolute Distance cell in the table below to enable.")

    def open_digsheet_by_abs_from_selection(self):
        try:
            if not self.project_is_open or not isinstance(self.pipe_tally, pd.DataFrame):
                QMessageBox.warning(self, "No Pipe Tally", "Load a project/tally first."); return
            abs_text = self._get_selected_abs_distance_from_defect_table()
            if not abs_text: return

            tally_pkl = _dump_tally_to_temp(self.pipe_tally)
            dig_py_abs = resource_path(os.path.join("dig", "digsheet_abs.py"))
            if not os.path.exists(dig_py_abs):
                QMessageBox.critical(self, "Script not found", f"Missing: {dig_py_abs}"); return

            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable, "--run-digsheet-abs", tally_pkl, str(abs_text)])
            else:
                subprocess.Popen([sys.executable, dig_py_abs, tally_pkl, str(abs_text)])
        except Exception as e:
            self.open_Error(f"Error opening ABS-distance digsheet:\n{e}")

    def close_project(self):
        try:
            # Stop graphs view and guard tab-change side effects
            self._close_graphs_view()

            tw = getattr(self.ui, "tabWidgetM", None)
            if tw is not None:
                tw.blockSignals(True)          # <<< prevent _on_middle_tab_changed from firing

            # Flip the state and clear everything
            self.project_is_open = False
            self.project_root = None
            self.pkl_files = []
            self.hmap = self.hmap_r = self.lplot = self.lplot_r = self.pipe3d = self.heatmap_box = None
            self.curr_data = None
            self.header_list = []

            cb = self.ui.comboBoxPipe
            cb.blockSignals(True); cb.clear(); cb.addItem("-Pipe-"); cb.blockSignals(False)

            self.model.clear()
            self.ui.tableWidgetDefect.clear()

            # ✅ Hide the "No Defects Found" message when closing project
            if hasattr(self, '_no_defects_container') and self._no_defects_container:
                self._no_defects_container.hide()

            self._show_create_project_message()
            self.web_view.setUrl(QUrl())
            self.web_view2.setUrl(QUrl())
            self.bottom_stack.setCurrentIndex(0)
            self._show_watermark()
            self._toggle_plot_ui(False)

            # Reset the tab index quietly
            if tw is not None:
                try:
                    tw.setCurrentIndex(0)       # no signal -> no popup
                except Exception:
                    pass

            self.btnDigsheetAbs.setEnabled(False)
            self._update_project_actions()

            QMessageBox.information(self, "Project Closed", "The project has been successfully closed.")
        except Exception as e:
            self.open_Error(e)
        finally:
            # Re-enable signals
            if tw is not None:
                tw.blockSignals(False)

    def open_CMLD(self):
        selected_columns = [r"Abs. Distance (m)", r"Type", r"Orientation o' clock"]
        if not isinstance(self.pipe_tally, pd.DataFrame):
            QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.")
            return
        for col in selected_columns:
            if col not in self.pipe_tally.columns:
                QMessageBox.critical(self, "Error", f"Missing column: {col}")
                return
        fil_tally = self.pipe_tally[selected_columns].copy()
        try:
            fil_tally["Abs. Distance (m)"] = fil_tally["Abs. Distance (m)"].astype(int)
            fil_tally["Type"] = fil_tally["Type"].astype(str)
            fil_tally[r"Orientation o' clock"] = fil_tally[r"Orientation o' clock"].astype(str)

            self.m3 = Main03Tab(fil_tally)
            self.m3.setWindowTitle("Circumferential Metal Loss Distribution")
            self.m3.resize(1285, 913)
            self.m3.show()
        except Exception as e:
            self.open_Error(e)

    def open_DBAD(self):
        selected_columns = [r"Abs. Distance (m)", r"Depth %", r"Type"]
        if not isinstance(self.pipe_tally, pd.DataFrame):
            QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.")
            return
        for col in selected_columns:
            if col not in self.pipe_tally.columns:
                QMessageBox.critical(self, "Error", f"Missing column: {col}")
                return
        fil_tally = self.pipe_tally[selected_columns].copy()
        try:
            fil_tally["Abs. Distance (m)"] = fil_tally["Abs. Distance (m)"].astype(int)
            fil_tally["Depth %"] = pd.to_numeric(fil_tally["Depth %"], errors='coerce')
            fil_tally["Type"] = fil_tally["Type"].astype(str)

            self.m2 = Main02Tab(fil_tally)
            self.m2.setWindowTitle("Depth Based Anomalies Distribution")
            self.m2.resize(1285, 913)
            self.m2.show()
        except Exception as e:
            self.open_Error(e)

    def open_EAD(self):
        selected_columns = [r"Abs. Distance (m)", r"Type", r"ERF (ASME B31G)"]
        if not isinstance(self.pipe_tally, pd.DataFrame):
            QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.")
            return
        for col in selected_columns:
            if col not in self.pipe_tally.columns:
                QMessageBox.critical(self, "Error", f"Missing column: {col}")
                return
        fil_tally = self.pipe_tally[selected_columns].copy()
        try:
            fil_tally["Abs. Distance (m)"] = fil_tally["Abs. Distance (m)"].astype(int)
            fil_tally["Type"] = fil_tally["Type"].astype(str)
            fil_tally["ERF (ASME B31G)"] = pd.to_numeric(fil_tally["ERF (ASME B31G)"], errors='coerce')

            self.m1 = Main01Tab(fil_tally)
            self.m1.setWindowTitle("ERF Based Anomalies Distribution")
            self.m1.resize(1285, 913)
            self.m1.show()
        except Exception as e:
            self.open_Error(e)

    def open_digs(self):
        try:
            if not isinstance(self.pipe_tally, pd.DataFrame):
                QMessageBox.warning(self, "No Pipe Tally", "Load a pipe tally first."); return
            tally_pkl = _dump_tally_to_temp(self.pipe_tally)
            dig_py = resource_path(os.path.join("dig", "dig_sheet.py"))
            if not os.path.exists(dig_py):
                QMessageBox.critical(self, "Script not found", f"Missing: {dig_py}"); return

            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable, "--run-digsheet", tally_pkl])
            else:
                subprocess.Popen([sys.executable, dig_py, tally_pkl])
        except Exception as e:
            self.open_Error(f"An error occurred: {e}")

    def _arm_topbar(self, virtual_max: int = 2000):
        """Re-sync the top scrollbar with the inner QScrollArea hbar and enable mapping."""
        try:
            inner = self.web_scroll_area.horizontalScrollBar()
            imin, imax = inner.minimum(), inner.maximum()
            rng = max(1, imax - imin)
            # map inner -> top
            top_val = int(round(((inner.value() - imin) / rng) * virtual_max))
            self._hscroll_ready = True
            self.top_scrollbar.blockSignals(True)
            self.top_scrollbar.setRange(0, virtual_max)
            self.top_scrollbar.setPageStep(100)
            self.top_scrollbar.setSingleStep(10)
            self.top_scrollbar.setValue(top_val)
            self.top_scrollbar.blockSignals(False)
        except Exception:
            # don't crash UI if something is missing during early init
            self._hscroll_ready = True

    def _setup_create_project_label(self):
        """Setup 'Create the Project in File' message box (shown at startup)."""
        self._create_proj_container = QWidget()
        self._create_proj_container.setMaximumSize(500, 200)
        self._create_proj_container.setMinimumSize(400, 150)
        self._create_proj_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self._create_proj_container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._create_proj_label = QLabel("Create the Project in File Menu")
        self._create_proj_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._create_proj_label.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                color: #666666;
                font-weight: bold;
                background-color: #f8f8f8;
                border: 2px dashed #cccccc;
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
            }
        """)
        layout.addWidget(self._create_proj_label)

        self._create_proj_container.hide()

        table_parent = self.ui.tableWidgetDefect.parentWidget()
        if table_parent:
            self._create_proj_container.setParent(table_parent)
            self._create_proj_container.move(500, 50)  # adjust as needed
    
    def _show_create_project_message(self):
        """Show 'Create the Project in File' message, hide table + scrollbars."""
        try:
            if hasattr(self, '_create_proj_container') and self._create_proj_container:
                self._create_proj_container.show()

            if hasattr(self.ui, 'tableWidgetDefect'):
                self.ui.tableWidgetDefect.hide()

            if hasattr(self, '_no_defects_container') and self._no_defects_container:
                self._no_defects_container.hide()

            if hasattr(self, 'table_scrollbar') and self.table_scrollbar:
                self.table_scrollbar.hide()   # 👈 also hide table top bar

            print("📋 Displaying 'Create the Project in File' message")
        except Exception as e:
            print(f"Error showing create project message: {e}")

    def _hide_create_project_message(self):
        if hasattr(self, '_create_proj_container'):
            self._create_proj_container.hide()

    def _arm_main_topbar(self, virtual_max: int = 2000):
        """Re-sync the main top scrollbar with the inner QScrollArea hbar and enable mapping."""
        try:
            inner = self.main_web_scroll_area.horizontalScrollBar()
            imin, imax = inner.minimum(), inner.maximum()
            rng = max(1, imax - imin)
            # map inner -> top
            top_val = int(round(((inner.value() - imin) / rng) * virtual_max))
            self._hscroll_ready_main = True
            self.main_top_scrollbar.blockSignals(True)
            self.main_top_scrollbar.setRange(0, virtual_max)
            self.main_top_scrollbar.setPageStep(100)
            self.main_top_scrollbar.setSingleStep(10)
            self.main_top_scrollbar.setValue(top_val)
            self.main_top_scrollbar.blockSignals(False)
        except Exception:
            # don't crash UI if something is missing during early init
            self._hscroll_ready_main = True

    def open_Error(self, e):
        try:
            dlg = QDialog(self); dlg.setWindowTitle("Error"); dlg.resize(700, 400)
            lay = QVBoxLayout(dlg)
            t = QTextEdit(); t.setReadOnly(True); t.setText(str(e))
            t.setStyleSheet("font-size: 10pt; font-family: Consolas; color: #aa0000;")
            lay.addWidget(t)
            b = QPushButton("Close"); b.clicked.connect(dlg.accept); lay.addWidget(b)
            dlg.exec()
        except Exception as err:
            print("Error dialog failed:", err)

    def quit_app(self):
        QApplication.quit()


if __name__ == "__main__":
    # Handle special modes in the frozen EXE so it doesn't relaunch the main UI
    if "--run-digsheet-abs" in sys.argv:
        i = sys.argv.index("--run-digsheet-abs")
        tally_pkl = sys.argv[i+1]; abs_val = sys.argv[i+2]
        dig_py_abs = resource_path(os.path.join("dig", "digsheet_abs.py"))
        sys.argv = [dig_py_abs, tally_pkl, abs_val]
        runpy.run_path(dig_py_abs, run_name="__main__")
        sys.exit(0)

    if "--run-digsheet" in sys.argv:
        i = sys.argv.index("--run-digsheet")
        tally_pkl = sys.argv[i+1]
        dig_py = resource_path(os.path.join("dig", "dig_sheet.py"))
        sys.argv = [dig_py, tally_pkl]
        runpy.run_path(dig_py, run_name="__main__")
        sys.exit(0)
    
    app = MainApp(sys.argv)
    app.start()
    sys.exit(app.exec())
