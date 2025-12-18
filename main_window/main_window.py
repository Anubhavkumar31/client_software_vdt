import os
import sys
from pathlib import Path
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
except ImportError:
    # Some builds moved QWebEnginePage to a separate submodule
    from PyQt6.QtWebEngineCore import QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView

import matplotlib
matplotlib.use("Qt5Agg")

from PyQt6 import uic
from PyQt6.QtCore import QUrl

# PyQt6 Widgets
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QDialog, QTextEdit, QPushButton
)

from reportlab.pdfgen import canvas  # noqa



#helper functions imports
from main_section_view.build_main_section import _build_main_section
from main_section_view.table_data_worker import _setup_table_models_and_behavior
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