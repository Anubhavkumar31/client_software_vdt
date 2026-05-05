import os

import pandas as pd
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction

from main_window.components.main_section_view.utils import _toggle_plot_ui
from main_window.components.helper_func import _update_project_actions


def setup_initial_ui_state(self):
    """
    -------------------------------------------------------------
    INITIAL UI STATE CONFIGURATION
    -------------------------------------------------------------
    Applies all startup UI settings:
      • Disables heatmap/graph UI until a project is loaded
      • Updates menu action availability
      • Applies main stylesheet
      • Maximizes window
      • Marks UI as "ready" (prevents early popups)
      • Displays watermark

    This is the last step of initialization and ensures the
    app opens in a clean, stable, predictable state.
    -------------------------------------------------------------
    """
    _toggle_plot_ui(self, False)
    _update_project_actions(self)

    self.setStyleSheet("QMainWindow { background-color: #FFFFFF; color: #000000; }")
    self.showMaximized()

    QTimer.singleShot(0, lambda: setattr(self, "_ui_ready", True))

    self._show_watermark()


