from PyQt6.QtCore import QTimer


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
    self._toggle_plot_ui(False)
    self._update_project_actions()

    self.setStyleSheet("QMainWindow { background-color: #FFFFFF; color: #000000; }")
    self.showMaximized()

    QTimer.singleShot(0, lambda: setattr(self, "_ui_ready", True))

    self._show_watermark()
