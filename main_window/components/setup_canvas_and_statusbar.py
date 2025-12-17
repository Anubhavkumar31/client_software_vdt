import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QStatusBar, QWidget, QHBoxLayout, QLabel

from backend.line_plot import PlotWindow
def setup_canvas_and_statusbar(self):
    """
    -------------------------------------------------------------
    GRAPH CANVAS + STATUS BAR + TIMER
    -------------------------------------------------------------
    Sets up:
      • Matplotlib canvas for plots
      • Status bar with left message and right live timer
      • QTimer used for benchmarking / live updates

    Centralized place for all plotting-related runtime UI elements.
    -------------------------------------------------------------
    """
    self.canvas = PlotWindow(self, width=5, height=4, dpi=100)

    self.setStatusBar(QStatusBar(self))
    self.current_message = 'App running'
    self.statusBar().showMessage('           Status:      ' + self.current_message)

    right_container = QWidget()
    rl = QHBoxLayout(right_container)
    rl.setContentsMargins(0, 0, 0, 0)
    self.right_status_label = QLabel('0.0s    ')
    rl.addWidget(self.right_status_label)
    self.statusBar().addPermanentWidget(right_container)

    self.timer = QTimer()
    self.timer.timeout.connect(lambda : _tick(self))
    self._t0 = None

def _tick(self):
    if self._t0:
        dt = time.time() - self._t0
        self.right_status_label.setText(f"{dt:.1f}s    ")
