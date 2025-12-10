import os
import sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QPushButton


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def create_filter_column_btn(self):
    """
     This block creates your “Filter Columns” button, styles it, adds an icon, connects its click signal,
     and inserts it into your UI layout next to other pipe-related buttons.
    """
    # create the button (you already have this)
    self.btnOpenFilterDlg = QPushButton("Filter Columns", self)
    self.btnOpenFilterDlg.setEnabled(False)

    # attach icon
    filter_icon_path = resource_path("ui/icons/filter.svg")  # or .png
    self.btnOpenFilterDlg.setIcon(QIcon(filter_icon_path))
    self.btnOpenFilterDlg.setIconSize(QSize(16, 16))  # 16–18px works well for a 28px-high button
    self.btnOpenFilterDlg.setCursor(Qt.CursorShape.PointingHandCursor)

    # optional: keep your outlined styling unchanged
    self.btnOpenFilterDlg.setStyleSheet("""
        QPushButton {
            background-color:#FFFFFF;
            color: #000000;
            border: 1.5px solid #000000;
            border-radius: 6px;
            padding: 4px 12px;  /* enough padding so icon+text breathe */
            font-weight: 500;
        }
        QPushButton:hover { background-color: #d6d3ce; }
        QPushButton:pressed { background-color: #111111; }
        QPushButton:disabled {
            background-color: #a6a6a6;
            color: #f0f0f0;
            border: 2px solid #6e6e6e; 
        }
    """)
    _parent = self.ui.comboBoxPipe.parentWidget()
    if _parent and _parent.layout():
        pos = _parent.layout().indexOf(self.btnLoadPipe)
        _parent.layout().insertWidget(pos + 2, self.btnOpenFilterDlg)
    else:
        self.btnOpenFilterDlg.setParent(_parent)

    return self.btnOpenFilterDlg