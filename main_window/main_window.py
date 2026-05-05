import os
import sys
import traceback
from pathlib import Path

from main_window.components.main_section_view.build_main_section import _build_main_section
from main_window.components.main_section_view.workers.table_data_worker import _setup_table_models_and_behavior
from main_window.components.menubar.view_menu.open_cluster import ClusterSummaryDialog

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
except ImportError:
    # Some builds moved QWebEnginePage to a separate submodule
    from PyQt6.QtWebEngineCore import QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView

import matplotlib
# matplotlib.use("QtAgg")

from PyQt6 import uic
from PyQt6.QtCore import QUrl, QTimer

# PyQt6 Widgets
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QDialog, QTextEdit, QPushButton, QWidget, QFrame, QApplication
)

from reportlab.pdfgen import canvas  # noqa
#helper functions imports

from main_window.components.create_buttons.buttons.comboBoxpipe import comboBoxPipe_setup
from main_window.components.create_buttons.setup_buttons import setup_buttons
from main_window.components.helper_func import create_instances
from main_window.components.setup_canvas_and_statusbar import setup_canvas_and_statusbar
from main_window.components.setup_initial_ui_state import setup_initial_ui_state
from main_window.components.setup_menu_actions import setup_menu_actions
from main_window.components.setup_table_system import setup_table_system
from main_window.components.setup_tabsystem import setup_tab_system
from main_window.components.setup_ui import setup_ui
from pages.XYZ import XYZ  # noqa
from pages.metrics import Metric_Dialog  # noqa
from pages.errorBox import Error_Dialog  # noqa
from backend.heatmap import HeatmapWindow as hm, pre_process, pre_process2  # noqa
from backend.clustering import run_clustering
from backend.clustering import build_cluster_rows


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)



base_dir = os.path.dirname(__file__)
ui_path = os.path.join(base_dir, "ui", "landing.ui")
SplashScreen, SplashWindow = uic.loadUiType(ui_path)
ui_path_main = os.path.join(base_dir, "ui", "main_window.ui")
Form, Window = uic.loadUiType(ui_path_main)


class MyMainWindow(QMainWindow):
    REQUIRED_TALLY_COLS = [
        r"Abs. Distance (m)", r"Depth %", r"Type",
        r"ERF (ASME B31G)", r"Orientation o' clock"
    ]

    def __init__(self):
        super().__init__()
        setup_ui(self)
        create_instances(self)
        comboBoxPipe_setup(self)
        _setup_table_models_and_behavior(self)
        setup_buttons(self)
        setup_tab_system(self)
        _build_main_section(self)
        setup_table_system(self)
        setup_canvas_and_statusbar(self)
        setup_menu_actions(self)
        setup_initial_ui_state(self)
        # debug_all(self.centralWidget())
        # QTimer.singleShot(2000, lambda: self.debug_reverse_hide_1(0, 163, False))
        # from PyQt6.QtWidgets import QScrollBar, QWidget
        #
        # for w in self.findChildren(QWidget):
        #     w.setStyleSheet("border: 1px solid red;")


    def debug_reverse_hide_1(self, start=0, end=50, process_all=False):
        from PyQt6.QtWidgets import QApplication, QWidget
        import time

        print("\n=== REVERSE DEBUG START ===")

        widgets = self.findChildren(QWidget)
        print("Total widgets:", len(widgets))

        # Reverse order
        widgets = list(reversed(widgets))

        # Clamp range safely
        start = max(0, start)
        end = min(len(widgets), end)
        if process_all:
            start = 0
            end = len(widgets)

        print(f"Processing range: {start} → {end}")

        for i in range(start, end):
            w = widgets[i]

            try:
                g = w.geometry()

                print(f"\n[{i}] HIDING:")
                print("  Class:", type(w))
                print("  Name:", w.objectName())
                print("  Geometry:", g)

                # Highlight BEFORE hiding
                w.setStyleSheet("background: yellow;")
                QApplication.processEvents()

                time.sleep(0.5)

                # Hide it
                w.hide()
                QApplication.processEvents()

            except Exception as e:
                print("Error:", e)



    # def _show_watermark(self):
    #     try:
    #         html_path = Path(resource_path("ui/icons/VDT_watermark.html"))
    #         base_url = QUrl.fromLocalFile(str(html_path.parent) + "/")
    #         with open(html_path, "r", encoding="utf-8") as f:
    #             self.web_view.setHtml(f.read(), base_url)
    #     except Exception:
    #         traceback.print_exc()
    #         self.web_view.setUrl(QUrl())
    #     self.bottom_stack.setCurrentIndex(0)
    #     self.web_view2.setUrl(QUrl())

    from pathlib import Path
    from PyQt6.QtCore import QUrl
    import traceback

    def _show_watermark(self):
        try:
            html_path = Path(resource_path("ui/icons/VDT_watermark.html"))

            # 🔍 Debug prints
            print("[WATERMARK] HTML PATH:", html_path)
            print("[WATERMARK] EXISTS:", html_path.exists())
            print("[WATERMARK] PARENT:", html_path.parent)

            base_url = QUrl.fromLocalFile(str(html_path.parent) + "/")

            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            print("[WATERMARK] HTML LOADED (length):", len(html_content))

            self.web_view.setHtml(html_content, base_url)

        except Exception:
            print("[WATERMARK] ERROR:")
            traceback.print_exc()
            self.web_view.setUrl(QUrl())

        self.bottom_stack.setCurrentIndex(0)
        self.web_view2.setUrl(QUrl())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_select_pipe_container") and self._select_pipe_container.isVisible():
            central = self.centralWidget().rect()
            header_height = self.ui.comboBoxPipe.height() + 20
            self._select_pipe_container.setGeometry(
                0,
                header_height,
                central.width(),
                central.height() - header_height
            )
        if hasattr(self, "_create_proj_container") and self._create_proj_container.isVisible():
            central = self.centralWidget().rect()
            self._create_proj_container.setGeometry(central)

    def update_load_button_state(self, idx: int):
        if not hasattr(self, "btnLoadPipe"):
            return

        text = self.ui.comboBoxPipe.currentText().strip()
        items = [self.ui.comboBoxPipe.itemText(i) for i in range(self.ui.comboBoxPipe.count())]

        # ✅ Enable Load if: a valid index OR a valid typed text
        if self.project_is_open and (idx >= 0 or text in items):
            self.btnLoadPipe.setEnabled(True)
            # ❌ Do NOT hide overlay here anymore
        else:
            self.btnLoadPipe.setEnabled(False)

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

    def on_view_clusters_clicked(self):
        from PyQt6.QtWidgets import QMessageBox
        import pandas as pd

        # Safety check
        if not hasattr(self, "pipe_tally") or self.pipe_tally is None or self.pipe_tally.empty:
            QMessageBox.warning(self, "Clusters", "No pipe data loaded")
            return

        # Run clustering
        clusters = run_clustering(self.pipe_tally)

        if not clusters:
            QMessageBox.information(self, "Clusters", "No clusters formed")
            return

        cluster_rows = build_cluster_rows(clusters)
        cluster_df = pd.DataFrame(cluster_rows)

        # Show dialog
        dlg = ClusterSummaryDialog(cluster_df, self)
        dlg.exec()

def debug_all(widget, indent=0):
    pad = "  " * indent
    print(f"{pad}{type(widget).__name__} | {widget.objectName()} | {widget.geometry()}")

    for child in widget.children():
        if hasattr(child, "geometry"):
            debug_all(child, indent + 1)

