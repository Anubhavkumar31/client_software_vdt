
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
    QTimer, QUrl, QEvent
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
    QTabBar, QFrame, QHBoxLayout as _QHBoxLayout, QSplitterHandle, QComboBox
)
# PyQt6 WebEngine
from PyQt6.QtWebEngineWidgets import QWebEngineView

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


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


base_dir = os.path.dirname(__file__)
ui_path = os.path.join(base_dir, "ui", "landing.ui")
SplashScreen, SplashWindow = uic.loadUiType(ui_path)
ui_path_main = os.path.join(base_dir, "ui", "main_window.ui")
Form, Window = uic.loadUiType(ui_path_main)


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

        # guard state
        self._reverting_tab = False
        self._last_allowed_tab_index = 0
        self._ui_ready = False  # set true after first layout/show

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
        from PyQt6.QtWidgets import QAbstractItemView

        # --- hook table signals so the button can update when user selects a row ---
        tw = self.ui.tableWidgetDefect
        tw.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

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

    # ---------- action enable/disable toggler ----------
    def _update_project_actions(self):
        a = self.ui
        act_create = getattr(a, "action_Create_Proj", None)
        act_close  = getattr(a, "action_Close_Proj", None)
        if isinstance(act_create, QAction):
            act_create.setEnabled(not self.project_is_open)
        if isinstance(act_close, QAction):
            act_close.setEnabled(self.project_is_open)
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
    # ---------------------------------------------------

    def _build_splitter(self):
        self.web_view = QWebEngineView()
        self.bottom_stack = QStackedWidget()

        self.defect_table_page = QWidget()
        dl = QVBoxLayout(self.defect_table_page); dl.setContentsMargins(0, 0, 0, 0)
        old_parent_def = self.ui.tableWidgetDefect.parentWidget()
        if old_parent_def and old_parent_def.layout():
            try: old_parent_def.layout().removeWidget(self.ui.tableWidgetDefect)
            except Exception: pass
        self.ui.tableWidgetDefect.setParent(self.defect_table_page)
        dl.addWidget(self.ui.tableWidgetDefect)

        self.data_table_page = QWidget()
        tl = QVBoxLayout(self.data_table_page); tl.setContentsMargins(0, 0, 0, 0)
        old_parent_data = self.ui.tableView.parentWidget()
        if old_parent_data and old_parent_data.layout():
            try: old_parent_data.layout().removeWidget(self.ui.tableView)
            except Exception: pass
        self.ui.tableView.setParent(self.data_table_page)
        tl.addWidget(self.ui.tableView)

        self.web_page = QWidget()
        wl = QVBoxLayout(self.web_page); wl.setContentsMargins(0, 0, 0, 0)
        self.web_view2 = QWebEngineView()
        self.web_view2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        wl.addWidget(self.web_view2)

        self.bottom_stack.addWidget(self.defect_table_page)
        self.bottom_stack.addWidget(self.data_table_page)
        self.bottom_stack.addWidget(self.web_page)

        self.splitter = MidBarSplitter(self, tabbar=self.mid_tabbar)
        self.splitter.addWidget(self.web_view)
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
        # a.action_Pipe_Tally.triggered.connect(self.open_Ptal)

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
            df = pd.read_pickle(pkl_path)
            self.curr_data = df
            self.header_list = list(df.columns)

            self.model.clear()
            self.model.setHorizontalHeaderLabels([str(c) for c in df.columns])
            for _, row in df.iterrows():
                items = [QStandardItem(str(v)) for v in row.values]
                self.model.appendRow(items)
            self.ui.tableView.setModel(self.model)
            self.ui.tableView.setSortingEnabled(True)

            name = os.path.splitext(os.path.basename(pkl_path))[0]
            pipe_idx = self._extract_index(name)
            self.load_assets_for_index(pipe_idx)
        except Exception as e:
            self.open_Error(f"load_selected_by_index error: {e}")

    @staticmethod
    def _extract_index(text: str) -> str:
        m = re.search(r'\d+', text)
        return m.group(0) if m else text

    def load_assets_for_index(self, pipe_idx: str):
        try:
            if not (self.project_root and self.project_is_open):
                return

            candidates = [
                os.path.join(self.project_root, f"pipe_{pipe_idx}"),
                os.path.join(self.project_root, f"pipe-{pipe_idx}"),
                os.path.join(self.project_root, f"Pipe_{pipe_idx}"),
            ]
            pipe_dir = next((d for d in candidates if os.path.isdir(d)), None)
            if not pipe_dir:
                return self.open_Error(f"Folder not found for pipe index {pipe_idx} (looked for pipe_{pipe_idx}).")

            def pick_one(patterns, exclude=None):
                exclude = exclude or []
                hits = []
                for pat in patterns:
                    hits.extend(glob(os.path.join(pipe_dir, pat)))
                hits = [h for h in hits if not any(ex in os.path.basename(h).lower() for ex in (exclude or []))]
                exact = [h for h in hits if re.search(rf'{re.escape(str(pipe_idx))}\b', os.path.basename(h))]
                return exact[0] if exact else (hits[0] if hits else None)

            self.hmap       = pick_one(["*heatmap*.html"], exclude=["raw", "box"])
            self.hmap_r     = pick_one(["*heatmap*raw*.html", "*raw*heatmap*.html"])
            self.heatmap_box= pick_one(["*heatmap*box*.html", "*box*heatmap*.html"])
            self.lplot      = pick_one(["*lineplot*.html", "*line*.html"], exclude=["raw"])
            self.lplot_r    = pick_one(["*lineplot*raw*.html", "*line*raw*.html"])
            self.pipe3d     = pick_one(["*pipe3d*.html", "pipe3d*.html"])
            self.prox_linechart = pick_one(["proximity_linechart*.html", "*proximity_linechart*.html"])

            ds_csv = pick_one(["*defectS*.csv", "*defects*.csv"])
            if ds_csv:
                try:
                    ds = pd.read_csv(ds_csv)
                    self._populate_defect_table_from_csv(ds)
                except Exception as e:
                    print("⚠️ Failed to load defect CSV:", e)

            self.tab_switcher2()
        except Exception as e:
            self.open_Error(f"load_assets_for_index error: {e}")

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

            elif tab in ("LineChart", "Line Chart", "Line Plot"):
                if self.lplot:
                    self._load_scrollable_chart(self.web_view, self.lplot, min_w=2200, min_h=1400)
                else:
                    self.web_view.setUrl(QUrl())
                if self.prox_linechart and os.path.exists(self.prox_linechart):
                    self.bottom_stack.setCurrentIndex(2)
                    self._load_scrollable_chart(self.web_view2, self.prox_linechart, min_w=1800, min_h=900)
                else:
                    self.bottom_stack.setCurrentIndex(0)
                    self.web_view2.setUrl(QUrl())

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

            self.update_digsheet_button_state()
        except Exception as e:
            self.open_Error(e)

    def _load_scrollable_chart(self, view: QWebEngineView, html_path: str, min_w: int = 2200, min_h: int = 1400):
        if not html_path or not os.path.exists(html_path):
            view.setUrl(QUrl())
            return
        safe = html_path.replace('\\', '/')
        wrapper = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ height:100%; margin:0; }}
  .wrap {{ height:100vh; width:100%; overflow:auto; }}
  iframe {{ border:0; width:{min_w}px; height:{min_h}px; }}
</style>
</head>
<body>
<div class="wrap">
  <iframe sandbox="allow-scripts allow-same-origin" src="file:///{safe}"></iframe>
</div>
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

    def _populate_defect_table_from_csv(self, df: pd.DataFrame):
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
            'Width': 'Length',
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
                    tw.setItem(r, c, QTableWidgetItem(str(v)))

        self.update_digsheet_button_state()

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
    # ---------------------------

    def open_digsheet_by_abs_from_selection(self):
        try:
            if not self.project_is_open:
                QMessageBox.information(self, "Open a project", "Please open a project first.")
                return
            if not isinstance(self.pipe_tally, pd.DataFrame):
                QMessageBox.warning(self, "No Pipe Tally",
                                    "Pipe tally data is missing. Load a pipe tally and try again.")
                return

            abs_text = self._get_selected_abs_distance_from_defect_table()
            if not abs_text:
                return

            import pickle
            base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
            pipe_tally_file = os.path.join(base_dir, "pipe_tally.pkl")
            with open(pipe_tally_file, "wb") as f:
                pickle.dump(self.pipe_tally, f)

            dig_py_abs = resource_path(os.path.join("dig", "digsheet_abs.py"))
            if not os.path.exists(dig_py_abs):
                QMessageBox.critical(self, "Script not found",
                                     f"ABS-distance digsheet script not found:\n{dig_py_abs}\n\n"
                                     f"Save your ABS-distance Tk script there, or update the path in code.")
                return

            subprocess.Popen([sys.executable, dig_py_abs, pipe_tally_file, str(abs_text)])
        except Exception as e:
            self.open_Error(f"Error opening ABS-distance digsheet:\n{e}")

    # def close_project(self):
    #     try:
    #         self._close_graphs_view()
    #         self.project_is_open = False
    #         self.project_root = None
    #         self.pkl_files = []
    #         self.hmap = self.hmap_r = self.lplot = self.lplot_r = self.pipe3d = self.heatmap_box = None
    #         self.curr_data = None
    #         self.header_list = []

    #         cb = self.ui.comboBoxPipe
    #         cb.blockSignals(True); cb.clear(); cb.addItem("-Pipe-"); cb.blockSignals(False)

    #         self.model.clear()
    #         self.ui.tableWidgetDefect.clear()

    #         self.web_view.setUrl(QUrl()); self.web_view2.setUrl(QUrl())
    #         self.bottom_stack.setCurrentIndex(0)
    #         self._show_watermark()
    #         self._toggle_plot_ui(False)

    #         try: self.ui.tabWidgetM.setCurrentIndex(0)
    #         except Exception: pass

    #         self.btnDigsheetAbs.setEnabled(False)
    #         self._update_project_actions()

    #         QMessageBox.information(self, "Project Closed", "The project has been successfully closed.")
    #     except Exception as e:
    #         self.open_Error(e)

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
            import pickle
            base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
            dig_py = resource_path(os.path.join("dig", "dig_sheet.py"))
            pipe_tally_file = os.path.join(base_dir, "pipe_tally.pkl")
            with open(pipe_tally_file, "wb") as f:
                pickle.dump(self.pipe_tally, f)
            subprocess.Popen([sys.executable, dig_py, pipe_tally_file])
        except Exception as e:
            self.open_Error(f"An error occurred: {e}")

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
    app = MainApp(sys.argv)
    app.start()
    sys.exit(app.exec())

