
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
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPen, QPolygonF
from PyQt6.QtCore import QPointF


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




# from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QPushButton
# from PyQt6.QtGui import QPen, QPolygonF, QPainter
# from PyQt6.QtCore import Qt, QPointF, pyqtSignal
# import pandas as pd
# import math
#
#
# class PipeLocatorWidget(QGraphicsView):
#     """
#     Pipe Locator (Production / Heatmap-like)
#     ----------------------------------------
#     • WHITE rows   → WELD
#     • ORANGE rows  → FEATURES
#     • Weld  → very small vertical line ABOVE pipe + distance text
#     • Feature → blue arrow BELOW pipe + feature name
#     • Long pipe with horizontal scrollbar
#     • Mouse drag pan
#     • Cursor-centric zoom (Ctrl + Mouse Wheel)
#     • Back button
#     """
#
#     backRequested = pyqtSignal()   # 🔙 back button signal
#
#     def __init__(self, pipe_tally: pd.DataFrame, parent=None):
#         super().__init__(parent)
#
#         self.df = pipe_tally.copy()
#
#         # ---------------- GRAPHICS SETUP ----------------
#         self.scene = QGraphicsScene(self)
#         self.setScene(self.scene)
#         self.setRenderHint(QPainter.RenderHint.Antialiasing)
#
#         # Scroll / pan behaviour
#         self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#         self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#         self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
#         self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
#
#         # 🔥 HEATMAP-LIKE ZOOM (MOST IMPORTANT)
#         self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#         self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#
#         self.setMinimumHeight(270)
#
#         # Zoom control
#         self._zoom = 1.0
#         self._zoom_step = 1.25
#         self._zoom_min = 0.4
#         self._zoom_max = 6.0
#
#         # Back button
#         self._add_back_button()
#
#         # Prepare + draw
#         self._prepare_data()
#         self._draw_pipe()
#
#     # -------------------------------------------------
#     # BACK BUTTON
#     # -------------------------------------------------
#     def _add_back_button(self):
#         self.back_btn = QPushButton("← Back", self)
#         self.back_btn.setFixedSize(70, 26)
#         self.back_btn.move(10, 10)
#         self.back_btn.raise_()
#         self.back_btn.clicked.connect(self.backRequested.emit)
#
#     # -------------------------------------------------
#     # DATA PREPARATION (Pipe_Tally_8inch logic)
#     # -------------------------------------------------
#     def _prepare_data(self):
#         df = self.df
#
#         # WHITE row = weld (Feature Type empty / NaN)
#         if "Feature Type" in df.columns:
#             df["__is_weld__"] = (
#                 df["Feature Type"].isna() |
#                 (df["Feature Type"].astype(str).str.strip() == "")
#             )
#         else:
#             df["__is_weld__"] = False
#
#         # Label (NaN-safe)
#         def label(row):
#             dist = row.get("Abs. Distance (m)")
#
#             if row["__is_weld__"]:
#                 if dist is None or (isinstance(dist, float) and math.isnan(dist)):
#                     return "Weld"
#                 return f"{int(round(dist))} m"
#
#             return str(
#                 row.get("Feature Type")
#                 or row.get("Type")
#                 or "Feature"
#             )
#
#         df["__label__"] = df.apply(label, axis=1)
#
#         # Remove rows that cannot be drawn
#         df = df[pd.notna(df["Abs. Distance (m)"])]
#
#         self.df = df
#
#     # -------------------------------------------------
#     # DRAW PIPE LOCATOR
#     # -------------------------------------------------
#     def _draw_pipe(self):
#         self.scene.clear()
#
#         if self.df.empty:
#             return
#
#         max_dist = int(self.df["Abs. Distance (m)"].max())
#
#         scale = 10.0           # pixels per meter
#         pipe_y = 140
#         pipe_len_px = max_dist * scale
#
#         # ---------------- PIPE LINE ----------------
#         self.scene.addLine(
#             0,
#             pipe_y,
#             pipe_len_px,
#             pipe_y,
#             QPen(Qt.GlobalColor.black, 3)
#         )
#
#         # ---------------- WELDS ----------------
#         welds = self.df[self.df["__is_weld__"]]
#         for _, r in welds.iterrows():
#             x = float(r["Abs. Distance (m)"]) * scale
#
#             # very small vertical line ABOVE pipe
#             self.scene.addLine(
#                 x,
#                 pipe_y - 8,
#                 x,
#                 pipe_y - 1,
#                 QPen(Qt.GlobalColor.black, 2)
#             )
#
#             # distance text above
#             t = self.scene.addText(r["__label__"])
#             t.setDefaultTextColor(Qt.GlobalColor.black)
#             t.setPos(x - 18, pipe_y - 34)
#
#         # ---------------- FEATURES ----------------
#         feats = self.df[~self.df["__is_weld__"]]
#         for _, r in feats.iterrows():
#             x = float(r["Abs. Distance (m)"]) * scale
#
#             # blue arrow BELOW pipe
#             arrow = QPolygonF([
#                 QPointF(x, pipe_y + 18),
#                 QPointF(x - 7, pipe_y + 34),
#                 QPointF(x + 7, pipe_y + 34),
#             ])
#             self.scene.addPolygon(
#                 arrow,
#                 QPen(Qt.GlobalColor.blue),
#                 Qt.GlobalColor.blue
#             )
#
#             # feature name below arrow
#             t = self.scene.addText(r["__label__"])
#             t.setDefaultTextColor(Qt.GlobalColor.blue)
#             t.setPos(x - 35, pipe_y + 38)
#
#         # Big scene = smooth scroll
#         self.setSceneRect(0, 0, pipe_len_px + 400, 320)
#
#     # -------------------------------------------------
#     # HEATMAP-LIKE ZOOM + SCROLL
#     # -------------------------------------------------
#     def wheelEvent(self, event):
#         # 🔍 Ctrl + wheel → cursor-centric zoom
#         if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
#             zoom_factor = self._zoom_step if event.angleDelta().y() > 0 else (1 / self._zoom_step)
#             new_zoom = self._zoom * zoom_factor
#
#             if self._zoom_min <= new_zoom <= self._zoom_max:
#                 self.scale(zoom_factor, zoom_factor)
#                 self._zoom = new_zoom
#
#             event.accept()
#             return
#
#         # 🖱 Normal wheel → horizontal scroll
#         bar = self.horizontalScrollBar()
#         bar.setValue(bar.value() - event.angleDelta().y())
#         event.accept()



# from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QPushButton, QGraphicsPolygonItem
# from PyQt6.QtGui import QPen, QPolygonF, QPainter
# from PyQt6.QtCore import Qt, QPointF, pyqtSignal
# import pandas as pd
# import math
#
#
# class PipeLocatorWidget(QGraphicsView):
#     """
#     Pipe Locator (Final – Click Based Labels)
#     ----------------------------------------
#     • WHITE rows   → WELD
#     • ORANGE rows  → FEATURES
#     • Weld  → small vertical line ABOVE pipe + distance text
#     • Feature → blue arrow BELOW pipe
#     • Feature name:
#         - NOT shown by default
#         - shown ONLY when that arrow is clicked
#     • Ctrl + Mouse wheel → cursor-centric zoom
#     • Mouse wheel → horizontal scroll
#     • Mouse drag → pan
#     • ← Back / ✕ / ESC → go to previous view
#     """
#
#     backRequested = pyqtSignal()
#
#     def __init__(self, pipe_tally: pd.DataFrame, parent=None):
#         super().__init__(parent)
#
#         self.df = pipe_tally.copy()
#         self.scene = QGraphicsScene(self)
#         self.setScene(self.scene)
#         self.setRenderHint(QPainter.RenderHint.Antialiasing)
#
#         # Scroll / pan
#         self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#         self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#         self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
#         self.setInteractive(True)
#         self.setMouseTracking(True)
#
#         # 🔥 TRUE heatmap-like zoom
#         self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#         self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#
#         # Zoom control
#         self._zoom = 1.0
#         self._zoom_step = 1.25
#         self._zoom_min = 0.4
#         self._zoom_max = 6.0
#
#         self.setMinimumHeight(280)
#
#         # Track selected label
#         self._active_label = None
#
#         self._add_buttons()
#         self._prepare_data()
#         self._draw_pipe()
#
#     # -------------------------------------------------
#     # BACK / CLOSE / ESC
#     # -------------------------------------------------
#     def _add_buttons(self):
#         self.back_btn = QPushButton("← Back", self)
#         self.back_btn.setFixedSize(70, 26)
#         self.back_btn.move(10, 10)
#         self.back_btn.clicked.connect(self.backRequested.emit)
#         self.back_btn.raise_()
#
#         self.close_btn = QPushButton("✕", self)
#         self.close_btn.setFixedSize(26, 26)
#         self.close_btn.clicked.connect(self.backRequested.emit)
#         self.close_btn.raise_()
#
#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         self.close_btn.move(self.width() - 36, 10)
#
#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Escape:
#             self.backRequested.emit()
#             event.accept()
#             return
#         super().keyPressEvent(event)
#
#     # -------------------------------------------------
#     # DATA PREP
#     # -------------------------------------------------
#     def _prepare_data(self):
#         df = self.df
#
#         if "Feature Type" in df.columns:
#             df["__is_weld__"] = (
#                 df["Feature Type"].isna() |
#                 (df["Feature Type"].astype(str).str.strip() == "")
#             )
#         else:
#             df["__is_weld__"] = False
#
#         def label(row):
#             dist = row.get("Abs. Distance (m)")
#             if row["__is_weld__"]:
#                 if dist is None or (isinstance(dist, float) and math.isnan(dist)):
#                     return "Weld"
#                 return f"{int(round(dist))} m"
#             return str(row.get("Feature Type") or "Feature")
#
#         df["__label__"] = df.apply(label, axis=1)
#         df = df[pd.notna(df["Abs. Distance (m)"])]
#         self.df = df
#
#     # -------------------------------------------------
#     # DRAW PIPE
#     # -------------------------------------------------
#     def _draw_pipe(self):
#         self.scene.clear()
#         self._active_label = None
#
#         max_dist = int(self.df["Abs. Distance (m)"].max())
#         scale = 10.0
#         pipe_y = 150
#
#         # Pipe
#         self.scene.addLine(
#             0, pipe_y,
#             max_dist * scale, pipe_y,
#             QPen(Qt.GlobalColor.black, 3)
#         )
#
#         # Welds
#         for _, r in self.df[self.df["__is_weld__"]].iterrows():
#             x = r["Abs. Distance (m)"] * scale
#             self.scene.addLine(
#                 x, pipe_y - 8, x, pipe_y - 1,
#                 QPen(Qt.GlobalColor.black, 2)
#             )
#             t = self.scene.addText(r["__label__"])
#             t.setDefaultTextColor(Qt.GlobalColor.black)
#             t.setPos(x - 18, pipe_y - 36)
#
#         # Features (CLICK-ONLY LABEL)
#         for _, r in self.df[~self.df["__is_weld__"]].iterrows():
#             x = r["Abs. Distance (m)"] * scale
#
#             arrow = QGraphicsPolygonItem(QPolygonF([
#                 QPointF(x, pipe_y + 18),
#                 QPointF(x - 7, pipe_y + 34),
#                 QPointF(x + 7, pipe_y + 34),
#             ]))
#             arrow.setBrush(Qt.GlobalColor.blue)
#             arrow.setPen(QPen(Qt.GlobalColor.blue))
#             arrow.setData(0, r["__label__"])   # store label
#             arrow.setData(1, x)               # store x
#             arrow.setData(2, pipe_y)
#             arrow.setFlag(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable)
#
#             self.scene.addItem(arrow)
#
#         self.setSceneRect(0, 0, max_dist * scale + 400, 330)
#
#     # -------------------------------------------------
#     # CLICK HANDLER (SHOW ONLY ONE LABEL)
#     # -------------------------------------------------
#     def mousePressEvent(self, event):
#         super().mousePressEvent(event)
#
#         item = self.itemAt(event.position().toPoint())
#         if isinstance(item, QGraphicsPolygonItem):
#             # remove old label
#             if self._active_label:
#                 self.scene.removeItem(self._active_label)
#                 self._active_label = None
#
#             label = item.data(0)
#             x = item.data(1)
#             pipe_y = item.data(2)
#
#             t = self.scene.addText(label)
#             t.setDefaultTextColor(Qt.GlobalColor.blue)
#             t.setPos(x - 35, pipe_y + 52)
#             self._active_label = t
#
#     # -------------------------------------------------
#     # TRUE HEATMAP-LIKE ZOOM + SCROLL
#     # -------------------------------------------------
#     def wheelEvent(self, event):
#         delta = event.angleDelta().y()
#
#         # Ctrl + wheel → zoom at cursor (NO redraw)
#         if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
#             factor = self._zoom_step if delta > 0 else (1 / self._zoom_step)
#             new_zoom = self._zoom * factor
#
#             if self._zoom_min <= new_zoom <= self._zoom_max:
#                 self.scale(factor, factor)
#                 self._zoom = new_zoom
#
#             event.accept()
#             return
#
#         # Normal wheel → horizontal scroll
#         hbar = self.horizontalScrollBar()
#         hbar.setValue(hbar.value() - delta)
#         event.accept()


# from PyQt6.QtWidgets import (
#     QGraphicsView, QGraphicsScene, QPushButton,
#     QGraphicsPolygonItem, QLineEdit, QLabel
# )
# from PyQt6.QtGui import QPen, QPolygonF, QPainter
# from PyQt6.QtCore import Qt, QPointF, pyqtSignal
# import pandas as pd
# import math
#
#
# class PipeLocatorWidget(QGraphicsView):
#     """
#     Pipe Locator – FINAL (Range Filter + Click Label)
#     ------------------------------------------------
#     • Start–End distance filter
#     • WHITE rows   → WELD
#     • ORANGE rows  → FEATURES
#     • Weld  → small vertical line ABOVE pipe + distance text
#     • Feature → blue arrow BELOW pipe
#     • Feature name → shown ONLY when arrow clicked
#     • Ctrl + Mouse wheel → cursor-centric zoom
#     • Mouse wheel → horizontal scroll
#     • Mouse drag → pan
#     • ← Back / ✕ / ESC → go to previous view
#     """
#
#     backRequested = pyqtSignal()
#
#     def __init__(self, pipe_tally: pd.DataFrame, parent=None):
#         super().__init__(parent)
#
#         # ---------- DATA ----------
#         self._df_full = pipe_tally.copy()   # full data (never modified)
#         self.df = pipe_tally.copy()         # filtered data
#         self._range = None                  # (start, end)
#
#         # ---------- GRAPHICS ----------
#         self.scene = QGraphicsScene(self)
#         self.setScene(self.scene)
#         self.setRenderHint(QPainter.RenderHint.Antialiasing)
#
#         self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#         self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#         self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
#         self.setInteractive(True)
#         self.setMouseTracking(True)
#
#         # 🔥 true heatmap-like zoom
#         self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#         self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#
#         self._zoom = 1.0
#         self._zoom_step = 1.25
#         self._zoom_min = 0.4
#         self._zoom_max = 6.0
#
#         self._active_label = None
#         self.setMinimumHeight(300)
#
#         # UI
#         self._add_buttons()
#         self._add_range_filter()
#
#         # prepare + draw
#         self._prepare_data()
#         self._draw_pipe()
#
#     # -------------------------------------------------
#     # BACK / CLOSE / ESC  (ALL SAME = BACK)
#     # -------------------------------------------------
#     def _add_buttons(self):
#         self.back_btn = QPushButton("← Back", self)
#         self.back_btn.setFixedSize(70, 26)
#         self.back_btn.move(10, 10)
#         self.back_btn.clicked.connect(self.backRequested.emit)
#         self.back_btn.raise_()
#
#         self.close_btn = QPushButton("✕", self)
#         self.close_btn.setFixedSize(26, 26)
#         self.close_btn.clicked.connect(self.backRequested.emit)
#         self.close_btn.raise_()
#
#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         self.close_btn.move(self.width() - 36, 10)
#
#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Escape:
#             self.backRequested.emit()
#             event.accept()
#             return
#         super().keyPressEvent(event)
#
#     # -------------------------------------------------
#     # RANGE FILTER UI
#     # -------------------------------------------------
#     def _add_range_filter(self):
#         self.lbl_start = QLabel("Start (m):", self)
#         self.lbl_start.move(100, 12)
#
#         self.start_edit = QLineEdit(self)
#         self.start_edit.setFixedWidth(70)
#         self.start_edit.move(165, 10)
#
#         self.lbl_end = QLabel("End (m):", self)
#         self.lbl_end.move(245, 12)
#
#         self.end_edit = QLineEdit(self)
#         self.end_edit.setFixedWidth(70)
#         self.end_edit.move(300, 10)
#
#         self.apply_btn = QPushButton("Apply", self)
#         self.apply_btn.move(380, 9)
#         self.apply_btn.clicked.connect(self._apply_range_filter)
#
#         self.reset_btn = QPushButton("Reset", self)
#         self.reset_btn.move(450, 9)
#         self.reset_btn.clicked.connect(self._reset_range_filter)
#
#         for w in (
#             self.lbl_start, self.start_edit,
#             self.lbl_end, self.end_edit,
#             self.apply_btn, self.reset_btn
#         ):
#             w.raise_()
#
#     # -------------------------------------------------
#     # APPLY / RESET RANGE  (🔥 FIXED)
#     # -------------------------------------------------
#     def _apply_range_filter(self):
#         try:
#             start = float(self.start_edit.text())
#             end = float(self.end_edit.text())
#             if start >= end:
#                 return
#         except ValueError:
#             return
#
#         self._range = (start, end)
#
#         df = self._df_full
#         df = df[
#             (df["Abs. Distance (m)"] >= start) &
#             (df["Abs. Distance (m)"] <= end)
#         ]
#
#         self.df = df.copy()
#
#         # 🔥 IMPORTANT: rebuild metadata columns
#         self._prepare_data()
#
#         self._draw_pipe()
#
#     def _reset_range_filter(self):
#         self._range = None
#         self.df = self._df_full.copy()
#
#         # 🔥 IMPORTANT
#         self._prepare_data()
#
#         self._draw_pipe()
#
#     # -------------------------------------------------
#     # DATA PREPARATION (ALWAYS CREATES __is_weld__)
#     # -------------------------------------------------
#     def _prepare_data(self):
#         df = self.df
#
#         if "Feature Type" in df.columns:
#             df["__is_weld__"] = (
#                 df["Feature Type"].isna() |
#                 (df["Feature Type"].astype(str).str.strip() == "")
#             )
#         else:
#             df["__is_weld__"] = False
#
#         def make_label(row):
#             dist = row.get("Abs. Distance (m)")
#             if row["__is_weld__"]:
#                 if dist is None or (isinstance(dist, float) and math.isnan(dist)):
#                     return "Weld"
#                 return f"{int(round(dist))} m"
#             return str(row.get("Feature Type") or "Feature")
#
#         df["__label__"] = df.apply(make_label, axis=1)
#
#         df.dropna(subset=["Abs. Distance (m)"], inplace=True)
#
#         self.df = df
#
#     # -------------------------------------------------
#     # DRAW PIPE (RANGE-AWARE)
#     # -------------------------------------------------
#     def _draw_pipe(self):
#         self.scene.clear()
#         self._active_label = None
#
#         if self.df.empty:
#             return
#
#         if self._range:
#             start, end = self._range
#         else:
#             start = 0.0
#             end = self.df["Abs. Distance (m)"].max()
#
#         scale = 10.0
#         pipe_y = 160
#         pipe_len_px = (end - start) * scale
#
#         # Pipe
#         self.scene.addLine(
#             0, pipe_y,
#             pipe_len_px, pipe_y,
#             QPen(Qt.GlobalColor.black, 3)
#         )
#
#         # Welds
#         for _, r in self.df[self.df["__is_weld__"]].iterrows():
#             x = (r["Abs. Distance (m)"] - start) * scale
#             self.scene.addLine(
#                 x, pipe_y - 8,
#                 x, pipe_y - 1,
#                 QPen(Qt.GlobalColor.black, 2)
#             )
#             t = self.scene.addText(r["__label__"])
#             t.setDefaultTextColor(Qt.GlobalColor.black)
#             t.setPos(x - 18, pipe_y - 36)
#
#         # Features (CLICK ONLY)
#         for _, r in self.df[~self.df["__is_weld__"]].iterrows():
#             x = (r["Abs. Distance (m)"] - start) * scale
#
#             arrow = QGraphicsPolygonItem(QPolygonF([
#                 QPointF(x, pipe_y + 18),
#                 QPointF(x - 7, pipe_y + 34),
#                 QPointF(x + 7, pipe_y + 34),
#             ]))
#             arrow.setBrush(Qt.GlobalColor.blue)
#             arrow.setPen(QPen(Qt.GlobalColor.blue))
#             arrow.setData(0, r["__label__"])
#             arrow.setData(1, x)
#             arrow.setData(2, pipe_y)
#             arrow.setFlag(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable)
#             self.scene.addItem(arrow)
#
#         self.setSceneRect(0, 0, pipe_len_px + 400, 360)
#
#     # -------------------------------------------------
#     # CLICK → SHOW ONLY ONE FEATURE NAME
#     # -------------------------------------------------
#     def mousePressEvent(self, event):
#         super().mousePressEvent(event)
#
#         item = self.itemAt(event.position().toPoint())
#         if isinstance(item, QGraphicsPolygonItem):
#             if self._active_label:
#                 self.scene.removeItem(self._active_label)
#                 self._active_label = None
#
#             label = item.data(0)
#             x = item.data(1)
#             pipe_y = item.data(2)
#
#             t = self.scene.addText(label)
#             t.setDefaultTextColor(Qt.GlobalColor.blue)
#             t.setPos(x - 35, pipe_y + 52)
#             self._active_label = t
#
#     # -------------------------------------------------
#     # TRUE HEATMAP-LIKE ZOOM + SCROLL
#     # -------------------------------------------------
#     def wheelEvent(self, event):
#         delta = event.angleDelta().y()
#
#         # Ctrl + wheel → zoom at cursor
#         if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
#             factor = self._zoom_step if delta > 0 else (1 / self._zoom_step)
#             new_zoom = self._zoom * factor
#
#             if self._zoom_min <= new_zoom <= self._zoom_max:
#                 self.scale(factor, factor)
#                 self._zoom = new_zoom
#
#             event.accept()
#             return
#
#         # Normal wheel → horizontal scroll
#         hbar = self.horizontalScrollBar()
#         hbar.setValue(hbar.value() - delta)
#         event.accept()


# from PyQt6.QtWidgets import (
#     QGraphicsView, QGraphicsScene, QPushButton,
#     QGraphicsPolygonItem, QLineEdit, QLabel
# )
# from PyQt6.QtGui import QPen, QPolygonF, QPainter
# from PyQt6.QtCore import Qt, QPointF, pyqtSignal
# import pandas as pd
# import math
#
#
# class PipeLocatorWidget(QGraphicsView):
#     """
#     Pipe Locator – FINAL (Dynamic Zoom Scale)
#     ----------------------------------------
#     • Range filter (Start–End)
#     • Weld distance spreads on zoom
#     • Weld labels staggered (no overlap)
#     • Feature label shown only on click
#     • Ctrl + Mouse wheel → cursor-centric zoom
#     • Mouse wheel → horizontal scroll
#     • Mouse drag → pan
#     • Back / ✕ / ESC → previous view
#     """
#
#     backRequested = pyqtSignal()
#
#     def __init__(self, pipe_tally: pd.DataFrame, parent=None):
#         super().__init__(parent)
#
#         # -------- DATA --------
#         self._df_full = pipe_tally.copy()
#         self.df = pipe_tally.copy()
#         self._range = None
#
#         # -------- GRAPHICS --------
#         self.scene = QGraphicsScene(self)
#         self.setScene(self.scene)
#         self.setRenderHint(QPainter.RenderHint.Antialiasing)
#
#         self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#         self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#         self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
#         self.setInteractive(True)
#         self.setMouseTracking(True)
#
#         # cursor-centric zoom
#         self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#         self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#
#         # zoom control
#         self._zoom = 1.0
#         self._zoom_step = 1.25
#         self._zoom_min = 0.4
#         self._zoom_max = 8.0
#
#         self._active_label = None
#         self.setMinimumHeight(320)
#
#         # UI
#         self._add_buttons()
#         self._add_range_filter()
#
#         self._prepare_data()
#         self._draw_pipe()
#
#     # -------------------------------------------------
#     # BACK / CLOSE / ESC
#     # -------------------------------------------------
#     def _add_buttons(self):
#         self.back_btn = QPushButton("← Back", self)
#         self.back_btn.setFixedSize(70, 26)
#         self.back_btn.move(10, 10)
#         self.back_btn.clicked.connect(self.backRequested.emit)
#         self.back_btn.raise_()
#
#         self.close_btn = QPushButton("✕", self)
#         self.close_btn.setFixedSize(26, 26)
#         self.close_btn.clicked.connect(self.backRequested.emit)
#         self.close_btn.raise_()
#
#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         self.close_btn.move(self.width() - 36, 10)
#
#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Escape:
#             self.backRequested.emit()
#             event.accept()
#             return
#         super().keyPressEvent(event)
#
#     # -------------------------------------------------
#     # RANGE FILTER UI
#     # -------------------------------------------------
#     def _add_range_filter(self):
#         self.lbl_start = QLabel("Start (m):", self)
#         self.lbl_start.move(100, 12)
#
#         self.start_edit = QLineEdit(self)
#         self.start_edit.setFixedWidth(70)
#         self.start_edit.move(165, 10)
#
#         self.lbl_end = QLabel("End (m):", self)
#         self.lbl_end.move(245, 12)
#
#         self.end_edit = QLineEdit(self)
#         self.end_edit.setFixedWidth(70)
#         self.end_edit.move(300, 10)
#
#         self.apply_btn = QPushButton("Apply", self)
#         self.apply_btn.move(380, 9)
#         self.apply_btn.clicked.connect(self._apply_range_filter)
#
#         self.reset_btn = QPushButton("Reset", self)
#         self.reset_btn.move(450, 9)
#         self.reset_btn.clicked.connect(self._reset_range_filter)
#
#         for w in (
#             self.lbl_start, self.start_edit,
#             self.lbl_end, self.end_edit,
#             self.apply_btn, self.reset_btn
#         ):
#             w.raise_()
#
#     def _apply_range_filter(self):
#         try:
#             start = float(self.start_edit.text())
#             end = float(self.end_edit.text())
#             if start >= end:
#                 return
#         except ValueError:
#             return
#
#         self._range = (start, end)
#         self.df = self._df_full[
#             (self._df_full["Abs. Distance (m)"] >= start) &
#             (self._df_full["Abs. Distance (m)"] <= end)
#         ].copy()
#
#         self._prepare_data()
#         self._draw_pipe()
#
#     def _reset_range_filter(self):
#         self._range = None
#         self.df = self._df_full.copy()
#         self._prepare_data()
#         self._draw_pipe()
#
#     # -------------------------------------------------
#     # DATA PREPARATION
#     # -------------------------------------------------
#     def _prepare_data(self):
#         df = self.df
#
#         if "Feature Type" in df.columns:
#             df["__is_weld__"] = (
#                 df["Feature Type"].isna() |
#                 (df["Feature Type"].astype(str).str.strip() == "")
#             )
#         else:
#             df["__is_weld__"] = False
#
#         def make_label(row):
#             dist = row.get("Abs. Distance (m)")
#             if row["__is_weld__"]:
#                 if pd.isna(dist):
#                     return "Weld"
#                 return f"{int(round(dist))} m"
#             return str(row.get("Feature Type") or "Feature")
#
#         df["__label__"] = df.apply(make_label, axis=1)
#         df.dropna(subset=["Abs. Distance (m)"], inplace=True)
#         self.df = df
#
#     # -------------------------------------------------
#     # DRAW PIPE (DYNAMIC SCALE)
#     # -------------------------------------------------
#     def _draw_pipe(self):
#         self.scene.clear()
#         self._active_label = None
#
#         if self.df.empty:
#             return
#
#         if self._range:
#             start, end = self._range
#         else:
#             start = 0.0
#             end = self.df["Abs. Distance (m)"].max()
#
#         base_scale = 10.0
#         scale = base_scale * self._zoom   # 🔥 THIS IS THE KEY FIX
#         pipe_y = 170
#         pipe_len_px = (end - start) * scale
#
#         # Pipe
#         self.scene.addLine(
#             0, pipe_y,
#             pipe_len_px, pipe_y,
#             QPen(Qt.GlobalColor.black, 3)
#         )
#
#         # Welds (staggered)
#         welds = self.df[self.df["__is_weld__"]].sort_values("Abs. Distance (m)")
#         for i, (_, r) in enumerate(welds.iterrows()):
#             x = (r["Abs. Distance (m)"] - start) * scale
#             self.scene.addLine(
#                 x, pipe_y - 8,
#                 x, pipe_y - 1,
#                 QPen(Qt.GlobalColor.black, 2)
#             )
#             y_offset = (i % 3) * 12
#             t = self.scene.addText(r["__label__"])
#             t.setDefaultTextColor(Qt.GlobalColor.black)
#             t.setPos(x - 18, pipe_y - 36 - y_offset)
#
#         # Features (click only)
#         for _, r in self.df[~self.df["__is_weld__"]].iterrows():
#             x = (r["Abs. Distance (m)"] - start) * scale
#
#             arrow = QGraphicsPolygonItem(QPolygonF([
#                 QPointF(x, pipe_y + 18),
#                 QPointF(x - 7, pipe_y + 34),
#                 QPointF(x + 7, pipe_y + 34),
#             ]))
#             arrow.setBrush(Qt.GlobalColor.blue)
#             arrow.setPen(QPen(Qt.GlobalColor.blue))
#             arrow.setData(0, r["__label__"])
#             arrow.setData(1, x)
#             arrow.setData(2, pipe_y)
#             arrow.setFlag(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable)
#             self.scene.addItem(arrow)
#
#         self.setSceneRect(0, 0, pipe_len_px + 400, 380)
#
#     # -------------------------------------------------
#     # CLICK → SHOW FEATURE NAME
#     # -------------------------------------------------
#     def mousePressEvent(self, event):
#         super().mousePressEvent(event)
#         item = self.itemAt(event.position().toPoint())
#
#         if isinstance(item, QGraphicsPolygonItem):
#             if self._active_label:
#                 self.scene.removeItem(self._active_label)
#
#             label = item.data(0)
#             x = item.data(1)
#             pipe_y = item.data(2)
#
#             t = self.scene.addText(label)
#             t.setDefaultTextColor(Qt.GlobalColor.blue)
#             t.setPos(x - 35, pipe_y + 54)
#             self._active_label = t
#
#     # -------------------------------------------------
#     # ZOOM + SCROLL (REDRAW ON ZOOM)
#     # -------------------------------------------------
#     def wheelEvent(self, event):
#         delta = event.angleDelta().y()
#
#         # Ctrl + wheel → zoom
#         if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
#             factor = self._zoom_step if delta > 0 else (1 / self._zoom_step)
#             new_zoom = self._zoom * factor
#
#             if self._zoom_min <= new_zoom <= self._zoom_max:
#                 self._zoom = new_zoom
#                 self._draw_pipe()   # 🔥 redraw with new scale
#
#             event.accept()
#             return
#
#         # normal wheel → horizontal scroll
#         hbar = self.horizontalScrollBar()
#         hbar.setValue(hbar.value() - delta)
#         event.accept()



# from PyQt6.QtWidgets import (
#     QGraphicsView, QGraphicsScene, QPushButton,
#     QGraphicsPolygonItem, QLineEdit, QLabel
# )
# from PyQt6.QtGui import QPen, QPolygonF, QPainter
# from PyQt6.QtCore import Qt, QPointF, pyqtSignal
# import pandas as pd
# import math
#
#
# class PipeLocatorWidget(QGraphicsView):
#
#
#     backRequested = pyqtSignal()
#
#     def __init__(self, pipe_tally: pd.DataFrame, parent=None):
#         super().__init__(parent)
#
#         # -------- DATA --------
#         self._df_full = pipe_tally.copy()
#         self.df = pipe_tally.copy()
#         self._range = None
#
#         # -------- GRAPHICS --------
#         self.scene = QGraphicsScene(self)
#         self.setScene(self.scene)
#         self.setRenderHint(QPainter.RenderHint.Antialiasing)
#
#         self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#         self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#         self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
#         self.setInteractive(True)
#         self.setMouseTracking(True)
#
#         # cursor-centric zoom
#         self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#         self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
#
#         # zoom control
#         self._zoom = 1.0
#         self._zoom_step = 1.25
#         self._zoom_min = 0.4
#         self._zoom_max = 8.0
#
#         self._active_label = None
#         self.setMinimumHeight(320)
#
#         # UI
#         self._add_buttons()
#         self._add_range_filter()
#
#         self._prepare_data()
#         self._draw_pipe()
#
#     # -------------------------------------------------
#     # BACK / CLOSE / ESC
#     # -------------------------------------------------
#     def _add_buttons(self):
#         self.back_btn = QPushButton("← Back", self)
#         self.back_btn.setFixedSize(70, 26)
#         self.back_btn.move(10, 10)
#         self.back_btn.clicked.connect(self.backRequested.emit)
#         self.back_btn.raise_()
#
#         self.close_btn = QPushButton("✕", self)
#         self.close_btn.setFixedSize(26, 26)
#         self.close_btn.clicked.connect(self.backRequested.emit)
#         self.close_btn.raise_()
#
#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         self.close_btn.move(self.width() - 36, 10)
#
#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Escape:
#             self.backRequested.emit()
#             event.accept()
#             return
#         super().keyPressEvent(event)
#
#     # -------------------------------------------------
#     # RANGE FILTER UI
#     # -------------------------------------------------
#     def _add_range_filter(self):
#         self.lbl_start = QLabel("Start (m):", self)
#         self.lbl_start.move(100, 12)
#
#         self.start_edit = QLineEdit(self)
#         self.start_edit.setFixedWidth(70)
#         self.start_edit.move(165, 10)
#
#         self.lbl_end = QLabel("End (m):", self)
#         self.lbl_end.move(245, 12)
#
#         self.end_edit = QLineEdit(self)
#         self.end_edit.setFixedWidth(70)
#         self.end_edit.move(300, 10)
#
#         self.apply_btn = QPushButton("Apply", self)
#         self.apply_btn.move(380, 9)
#         self.apply_btn.clicked.connect(self._apply_range_filter)
#
#         self.reset_btn = QPushButton("Reset", self)
#         self.reset_btn.move(450, 9)
#         self.reset_btn.clicked.connect(self._reset_range_filter)
#
#         for w in (
#             self.lbl_start, self.start_edit,
#             self.lbl_end, self.end_edit,
#             self.apply_btn, self.reset_btn
#         ):
#             w.raise_()
#
#     def _apply_range_filter(self):
#         try:
#             start = float(self.start_edit.text())
#             end = float(self.end_edit.text())
#             if start >= end:
#                 return
#         except ValueError:
#             return
#
#         self._range = (start, end)
#         self.df = self._df_full[
#             (self._df_full["Abs. Distance (m)"] >= start) &
#             (self._df_full["Abs. Distance (m)"] <= end)
#         ].copy()
#
#         self._prepare_data()
#         self._draw_pipe()
#
#     # def _apply_range_filter(self):
#     #     try:
#     #         start = float(self.start_edit.text())
#     #         end = float(self.end_edit.text())
#     #         if start >= end:
#     #             return
#     #     except ValueError:
#     #         return
#     #
#     #     self._range = (start, end)
#     #     self.df = self._df_full[
#     #         (self._df_full["Abs. Distance (m)"] >= start) &
#     #         (self._df_full["Abs. Distance (m)"] <= end)
#     #         ].copy()
#     #
#     #     self._prepare_data()
#     #     self._draw_pipe()
#     #
#     #     # 🔒 STOP event propagation (VERY IMPORTANT)
#     #     self.setFocus()
#
#     def _reset_range_filter(self):
#         self._range = None
#         self.df = self._df_full.copy()
#         self._prepare_data()
#         self._draw_pipe()
#
#     # -------------------------------------------------
#     # DATA PREPARATION
#     # -------------------------------------------------
#     def _prepare_data(self):
#         df = self.df
#
#         if "Feature Type" in df.columns:
#             df["__is_weld__"] = (
#                 df["Feature Type"].isna() |
#                 (df["Feature Type"].astype(str).str.strip() == "")
#             )
#         else:
#             df["__is_weld__"] = False
#
#         def make_label(row):
#             dist = row.get("Abs. Distance (m)")
#             if row["__is_weld__"]:
#                 if pd.isna(dist):
#                     return "Weld"
#                 return f"{int(round(dist))} m"
#             return str(row.get("Feature Type") or "Feature")
#
#         df["__label__"] = df.apply(make_label, axis=1)
#         df.dropna(subset=["Abs. Distance (m)"], inplace=True)
#         self.df = df
#
#     # -------------------------------------------------
#     # DRAW PIPE (DYNAMIC SCALE)
#     # -------------------------------------------------
#     def _draw_pipe(self):
#         self.scene.clear()
#         self._active_label = None
#
#         if self.df.empty:
#             return
#
#         if self._range:
#             start, end = self._range
#         else:
#             start = 0.0
#             end = self.df["Abs. Distance (m)"].max()
#
#         base_scale = 10.0
#         scale = base_scale * self._zoom
#         pipe_y = 170
#         pipe_len_px = (end - start) * scale
#
#         # Pipe
#         self.scene.addLine(
#             0, pipe_y,
#             pipe_len_px, pipe_y,
#             QPen(Qt.GlobalColor.black, 3)
#         )
#
#         # Welds (staggered)
#         welds = self.df[self.df["__is_weld__"]].sort_values("Abs. Distance (m)")
#         for i, (_, r) in enumerate(welds.iterrows()):
#             x = (r["Abs. Distance (m)"] - start) * scale
#             self.scene.addLine(
#                 x, pipe_y - 8,
#                 x, pipe_y - 1,
#                 QPen(Qt.GlobalColor.black, 2)
#             )
#             y_offset = (i % 3) * 12
#             t = self.scene.addText(r["__label__"])
#             t.setDefaultTextColor(Qt.GlobalColor.black)
#             t.setPos(x - 18, pipe_y - 36 - y_offset)
#
#         # -------- FEATURES (ABOVE PIPE, TOUCHING PIPE) --------
#         for _, r in self.df[~self.df["__is_weld__"]].iterrows():
#             x = (r["Abs. Distance (m)"] - start) * scale
#
#             arrow = QGraphicsPolygonItem(QPolygonF([
#                 QPointF(x, pipe_y),          # 👈 touch pipe
#                 QPointF(x - 7, pipe_y - 16),
#                 QPointF(x + 7, pipe_y - 16),
#             ]))
#             arrow.setBrush(Qt.GlobalColor.blue)
#             arrow.setPen(QPen(Qt.GlobalColor.blue))
#             arrow.setData(0, r["__label__"])
#             arrow.setData(1, x)
#             arrow.setData(2, pipe_y)
#             arrow.setFlag(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable)
#             self.scene.addItem(arrow)
#
#         self.setSceneRect(0, 0, pipe_len_px + 400, 380)
#
#     # -------------------------------------------------
#     # CLICK → SHOW FEATURE NAME (ABOVE PIPE)
#     # -------------------------------------------------
#     def mousePressEvent(self, event):
#         super().mousePressEvent(event)
#         item = self.itemAt(event.position().toPoint())
#
#         if isinstance(item, QGraphicsPolygonItem):
#             if self._active_label:
#                 self.scene.removeItem(self._active_label)
#
#             label = item.data(0)
#             x = item.data(1)
#             pipe_y = item.data(2)
#
#             t = self.scene.addText(label)
#             t.setDefaultTextColor(Qt.GlobalColor.blue)
#             t.setPos(x - 35, pipe_y - 40)
#             self._active_label = t
#
#     # -------------------------------------------------
#     # ZOOM + SCROLL (REDRAW ON ZOOM)
#     # -------------------------------------------------
#     def wheelEvent(self, event):
#         delta = event.angleDelta().y()
#
#         if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
#             factor = self._zoom_step if delta > 0 else (1 / self._zoom_step)
#             new_zoom = self._zoom * factor
#
#             if self._zoom_min <= new_zoom <= self._zoom_max:
#                 self._zoom = new_zoom
#                 self._draw_pipe()
#
#             event.accept()
#             return
#
#         hbar = self.horizontalScrollBar()
#         hbar.setValue(hbar.value() - delta)
#         event.accept()


# ================== IMPORTS ==================
from PyQt6.QtWidgets import (
    QDialog, QGraphicsView, QGraphicsScene,
    QVBoxLayout, QPushButton, QLabel, QLineEdit,
    QGraphicsPolygonItem
)
from PyQt6.QtGui import QPen, QPolygonF, QPainter
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
import pandas as pd


# ================== PIPE LOCATOR VIEW ==================
class PipeLocatorWidget(QGraphicsView):
    backRequested = pyqtSignal()

    def __init__(self, pipe_tally: pd.DataFrame, parent=None):
        super().__init__(parent)

        self.df = pipe_tally.copy() if isinstance(pipe_tally, pd.DataFrame) else pd.DataFrame()
        self._range = None
        self._zoom = 1.0
        self._active_feature_label = None

        # Graphics
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setMinimumHeight(320)

        self._add_controls()
        self._prepare_data()
        self._draw_pipe()

    # ---------- TOP CONTROLS ----------
    def _add_controls(self):
        self.back_btn = QPushButton("← Back", self)
        self.back_btn.move(10, 10)
        self.back_btn.clicked.connect(self.backRequested.emit)
        self.back_btn.raise_()

        QLabel("Start (m):", self).move(100, 12)
        self.start_edit = QLineEdit(self)
        self.start_edit.setFixedWidth(70)
        self.start_edit.move(165, 10)

        QLabel("End (m):", self).move(245, 12)
        self.end_edit = QLineEdit(self)
        self.end_edit.setFixedWidth(70)
        self.end_edit.move(300, 10)

        QPushButton("Apply", self, clicked=self._apply_filter).move(380, 9)
        QPushButton("Reset", self, clicked=self._reset_filter).move(450, 9)

    # ---------- FILTER ----------
    def _apply_filter(self):
        try:
            s = float(self.start_edit.text())
            e = float(self.end_edit.text())
            if s >= e:
                return
            self._range = (s, e)
        except Exception:
            return
        self._draw_pipe()

    def _reset_filter(self):
        self._range = None
        self.start_edit.clear()
        self.end_edit.clear()
        self._draw_pipe()

    # ---------- DATA ----------
    def _prepare_data(self):
        if self.df.empty:
            return

        if "Feature Type" in self.df.columns:
            self.df["__is_weld__"] = (
                self.df["Feature Type"].isna() |
                (self.df["Feature Type"].astype(str).str.strip() == "")
            )
        else:
            self.df["__is_weld__"] = False

        def label(row):
            d = row.get("Abs. Distance (m)")
            if pd.isna(d):
                return ""
            # integer distance only
            return f"{int(round(float(d)))} m" if row["__is_weld__"] else str(
                row.get("Feature Type", "Feature")
            )

        self.df["__label__"] = self.df.apply(label, axis=1)
        self.df.dropna(subset=["Abs. Distance (m)"], inplace=True)

    # ---------- DRAW ----------
    def _draw_pipe(self):
        self.scene.clear()
        self._active_feature_label = None

        if self.df.empty:
            return

        if self._range:
            start, end = self._range
            data = self.df[
                (self.df["Abs. Distance (m)"] >= start) &
                (self.df["Abs. Distance (m)"] <= end)
            ]
        else:
            start = 0
            end = self.df["Abs. Distance (m)"].max()
            data = self.df

        scale = 10 * self._zoom
        pipe_y = 180
        pipe_len = (end - start) * scale

        # Pipe
        self.scene.addLine(0, pipe_y, pipe_len, pipe_y, QPen(Qt.GlobalColor.black, 3))

        weld_index = 0  # 🔥 for zig-zag labels

        for _, r in data.iterrows():
            x = (r["Abs. Distance (m)"] - start) * scale

            # -------- WELD (zig-zag labels) --------
            if r["__is_weld__"]:
                # weld line
                self.scene.addLine(
                    x, pipe_y - 8,
                    x, pipe_y,
                    QPen(Qt.GlobalColor.black, 2)
                )

                # 🔥 zig-zag Y offset
                # even → higher, odd → slightly lower
                y_offset = -34 if (weld_index % 2 == 0) else -50

                t = self.scene.addText(r["__label__"])
                t.setDefaultTextColor(Qt.GlobalColor.black)
                t.setPos(x - 18, pipe_y + y_offset)

                weld_index += 1

            # -------- FEATURE (ARROW ABOVE PIPE) --------
            else:
                # arrow ABOVE pipe (point touching pipe)
                arrow = QGraphicsPolygonItem(QPolygonF([
                    QPointF(x, pipe_y),          # tip touching pipe
                    QPointF(x - 7, pipe_y - 16),
                    QPointF(x + 7, pipe_y - 16),
                ]))

                arrow.setBrush(Qt.GlobalColor.blue)
                arrow.setPen(QPen(Qt.GlobalColor.blue))

                # store data for click
                arrow.setData(0, r["__label__"])
                arrow.setData(1, x)
                arrow.setData(2, pipe_y)

                arrow.setFlag(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable)
                self.scene.addItem(arrow)

        self.setSceneRect(0, 0, pipe_len + 300, 360)

    # ---------- FEATURE CLICK (show name) ----------
    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        item = self.itemAt(event.position().toPoint())
        if isinstance(item, QGraphicsPolygonItem):
            # remove old label
            if self._active_feature_label:
                self.scene.removeItem(self._active_feature_label)
                self._active_feature_label = None

            label = item.data(0)
            x = item.data(1)
            pipe_y = item.data(2)

            if label:
                t = self.scene.addText(label)
                t.setDefaultTextColor(Qt.GlobalColor.blue)
                # show name just above arrow
                t.setPos(x - 35, pipe_y - 32)
                self._active_feature_label = t

    # ---------- ZOOM ----------
    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._zoom *= 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self._zoom = max(0.4, min(self._zoom, 6))
            self._draw_pipe()
            event.accept()
            return

        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - event.angleDelta().y())
        event.accept()


# ================== PIPE LOCATOR DIALOG ==================
class PipeLocatorDialog(QDialog):
    def __init__(self, pipe_tally, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pipe Locator")
        self.resize(1100, 500)

        screen = self.screen().availableGeometry()
        self.move(
            screen.center().x() - self.width() // 2,
            screen.center().y() - self.height() // 2
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = PipeLocatorWidget(pipe_tally, self)
        self.view.backRequested.connect(self.close)
        layout.addWidget(self.view)


# ================== MAIN WINDOW FUNCTION ==================
def open_pipe_locator(self):
    if getattr(self, "_pipe_locator_dialog", None):
        self._pipe_locator_dialog.raise_()
        self._pipe_locator_dialog.activateWindow()
        return

    pipe_tally = self.pipe_tally if isinstance(self.pipe_tally, pd.DataFrame) else pd.DataFrame()

    dlg = PipeLocatorDialog(pipe_tally, self)
    self._pipe_locator_dialog = dlg

    if hasattr(self.ui, "widgetControls"):
        self.ui.widgetControls.hide()

    def _cleanup():
        self._pipe_locator_dialog = None
        if hasattr(self.ui, "widgetControls"):
            self.ui.widgetControls.show()

    dlg.finished.connect(_cleanup)
    dlg.show()








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
    table.verticalScrollBar().setSingleStep(15)


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
        # Look for pipe directories inside pipes_data subfolder
        pipes_data_dir = os.path.join(self.project_root, "pipes_data")
        if not os.path.isdir(pipes_data_dir):
            print(f"[Warning] pipes_data directory not found in {self.project_root}")
            return None
        
        candidates = [
            os.path.join(pipes_data_dir, f"pipe_{self.pipe_idx}"),
            os.path.join(pipes_data_dir, f"pipe-{self.pipe_idx}"),
            os.path.join(pipes_data_dir, f"Pipe_{self.pipe_idx}"),
        ]
        return next((d for d in candidates if os.path.isdir(d)), None)


    def _load_html_assets(self, pipedir):
        if not pipedir:
            return {}
        
        def pickone(*patterns, exclude=None):
            exclude = exclude or []
            hits = []
            for pat in patterns:
                hits.extend(glob(os.path.join(pipedir, pat)))
            hits = [h for h in hits if not any(ex in os.path.basename(h).lower() for ex in (exclude or []))]
            exact = [h for h in hits if re.search(rf"{re.escape(str(self.pipe_idx))}", os.path.basename(h))]
            return exact[0] if exact else (hits[0] if hits else None)
        
        return {
            "hmap": pickone("heatmap*.html", exclude=["raw", "box"]),
            "hmap_r": pickone("heatmap_raw*.html", "raw_heatmap*.html"),
            "heatmap_box": pickone("heatmap_box*.html", "box_heatmap*.html"),
            "lplot": pickone("lineplot*.html", "line*.html", exclude=["raw"]),
            "lplot_r": pickone("lineplot_raw*.html", "line_raw*.html"),
            "pipe3d": pickone("pipe_3d*.html", "pipe3d*.html"),
            "prox_linechart": pickone("proximity_linechart*.html", "proximitylinechart*.html"),
            "hallsensor_heatmap": pickone("hallsensor_heatmap*.html"),
            "proximity_heatmap": pickone("proximity_heatmap*.html")
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

    # def _process_table_data(self, df):
    #     if df is None or df.empty:
    #         return None

    #     # Check if this is a PipeTally file (has Feature Type column) or defects.csv
    #     if "Feature Type" in df.columns:
    #         # Filter Metal Loss defects
    #         original_count = len(df)
    #         df = df[df["Feature Type"].astype(str).str.strip().str.lower() == "metal loss"]

    #         if df.empty:
    #             return None

    #         # Round numeric columns
    #         numeric_columns = [
    #             'Depth %', 'Depth (mm)', 'ERF (ASME B31G)', 'Psafe (ASME B31G) Barg',
    #             'Abs. Distance (m)', 'Distance to U/S GW(m)', 'Length (mm)',
    #             'Width (mm)', 'WT (mm)', 'Pipe Length (mm)'
    #         ]
    #         for col in numeric_columns:
    #             if col in df.columns:
    #                 df[col] = pd.to_numeric(df[col], errors='coerce').round(3)

    #     return df
    def _process_table_data(self, df):
        """Return full PipeTally data without filtering or skipping."""
        if df is None or df.empty:
            return None

        # Just round numeric columns, no filtering
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

class ColumnFilterDialog(QDialog):
    def __init__(self, *, headers: list[str], checked: set[str], locked: set[str], parent=None):
        super().__init__(parent)
        self.unit_boxes = {}

        self.setWindowTitle("Select Columns")
        self.setModal(True)
        self.resize(420, 520)

        self._locked = set(locked)
        # only show headers that are NOT locked
        visible_headers = [h for h in headers if h not in self._locked]

        # widgets
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLineEdit, QListView, QPushButton, QLabel
        from PyQt6.QtGui import QStandardItemModel, QStandardItem
        from PyQt6.QtCore import Qt, QSortFilterProxyModel

        lay = QVBoxLayout(self)

        # search
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search columns…")
        lay.addWidget(self.search)

        # self.model = QStandardItemModel(self)
        # self.proxy = QSortFilterProxyModel(self)
        # self.proxy.setSourceModel(self.model)
        #
        # self.view = QListView(self)
        # self.view.setModel(self.proxy)
        # lay.addWidget(self.view)

        #list (checkable)

        self.model = QStandardItemModel(self)
        for name in visible_headers:
            it = QStandardItem(name)
            it.setCheckable(True)
            it.setCheckState(Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked)
            it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.model.appendRow(it)

            base = it.text().split(" (")[0]

            if base in parent._unit_columns:
                cb = QComboBox()
                cb.addItems(["m", "cm", "mm", "km", "feet"])
                cb.setCurrentText(parent._unit_columns[base])
                self.unit_boxes[base] = cb

        # self.unit_map = {}  # store dropdown for these columns
        #
        # UNIT_COLUMNS = [
        #     "Abs. Distance (m)",
        #     "Pipe Length (mm)",
        #     "WT (mm)",
        #     "Length (mm)",
        #     "Width (mm)",
        #     "Depth (mm)",
        # ]
        #
        # for name in visible_headers:
        #     row_widget = QWidget()
        #     row_layout = QHBoxLayout(row_widget)
        #     row_layout.setContentsMargins(0, 0, 0, 0)
        #
        #     # checkbox
        #     item = QStandardItem(name)
        #     item.setCheckable(True)
        #     item.setCheckState(Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked)
        #     self.model.appendRow(item)
        #
        #     # dropdown for select columns
        #     if name in UNIT_COLUMNS:
        #         cb = QComboBox()
        #         cb.addItems(["m", "cm", "mm", "feet", "km"])
        #         cb.setFixedWidth(80)
        #         self.unit_map[name] = cb
        #
        #         row_layout.addWidget(cb)
        #     else:
        #         spacer = QWidget()
        #         spacer.setFixedWidth(80)
        #         row_layout.addWidget(spacer)
        #
        #     # add the composite widget in place of plain text
        #     index = self.model.index(self.model.rowCount() - 1, 0)
        #     self.view.setIndexWidget(self.proxy.mapFromSource(index), row_widget)

        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(0)
        self.proxy.setSourceModel(self.model)

        self.view = QListView(self)
        self.view.setModel(self.proxy)
        self.view.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.view, 1)

        # quick actions
        row = QHBoxLayout()
        self.btnAll = QPushButton("Select All")
        self.btnNone = QPushButton("Select None")
        row.addWidget(self.btnAll)
        row.addWidget(self.btnNone)
        row.addStretch(1)
        lay.addLayout(row)

        for col, cb in self.unit_boxes.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(col))
            row.addWidget(cb)
            lay.addLayout(row)

        # footer
        foot = QHBoxLayout()
        self.info = QLabel("")  # shows e.g. "12 selected"
        foot.addWidget(self.info)
        foot.addStretch(1)
        self.btnCancel = QPushButton("Cancel")
        self.btnApply = QPushButton("Apply")
        foot.addWidget(self.btnCancel)
        foot.addWidget(self.btnApply)
        lay.addLayout(foot)

        # wire up
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.btnAll.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        self.btnNone.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        self.btnCancel.clicked.connect(self.reject)
        self.btnApply.clicked.connect(self.accept)

        self._update_info()
        self.model.itemChanged.connect(lambda *_: self._update_info())

    # def selected_units(self):
    #     return {col: self.unit_map[col].currentText() for col in self.unit_map}

    def _set_all(self, state: Qt.CheckState):
        for r in range(self.model.rowCount()):
            self.model.item(r).setCheckState(state)
        self._update_info()

    def _update_info(self):
        total = self.model.rowCount()
        sel = sum(1 for r in range(total) if self.model.item(r).checkState() == Qt.CheckState.Checked)
        self.info.setText(f"{sel} / {total} visible columns selected")

    def selected_names(self) -> set[str]:
        """Return the names selected in the dialog (locked not included, they’re enforced by caller)."""
        out = set()
        for r in range(self.model.rowCount()):
            it = self.model.item(r)
            if it.checkState() == Qt.CheckState.Checked:
                out.add(it.text())
        return out

    def selected_units(self):
        return {k: cb.currentText() for k, cb in self.unit_boxes.items()}


class ConsoleRelayPage(QWebEnginePage):
    """Catches JS console messages to ferry Plotly relayout/hover to Python."""
    relayout_json = pyqtSignal(dict)    # emits on plotly_relayout
    hover_json    = pyqtSignal(dict)    # (optional) emits on plotly_hover

    def javaScriptConsoleMessage(self, level, msg, line, source):
        if msg.startswith("RANGE:"):
            import json
            try:
                payload = json.loads(msg[6:])
                self.relayout_json.emit(payload)
            except Exception:
                pass
        elif msg.startswith("HOVER:"):
            import json
            try:
                payload = json.loads(msg[6:])
                self.hover_json.emit(payload)
            except Exception:
                pass
        # still let base handle logging
        return super().javaScriptConsoleMessage(level, msg, line, source)
    
class SyncPlotlyView(QWebEngineView):
    """
    A webview that, after the Plotly HTML loads, injects small JS hooks that:
      - listen for plotly_relayout and emit to Python
      - expose a JS function to apply ranges from Python
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._page = ConsoleRelayPage(self)
        self.setPage(self._page)
        self._installed = False
        self._busy = False
        self.loadFinished.connect(self._install_hooks_if_needed)

    @property
    def relay(self) -> ConsoleRelayPage:
        return self._page

    def _install_hooks_if_needed(self, ok: bool):
        if not ok or self._installed:
            return

        js = r"""
        (function(){
          if (window.__pie_hooks_installed) return;
          window.__pie_hooks_installed = true;

          function getGraph(){
            let g = document.querySelector('.js-plotly-plot');
            if (!g) g = document.querySelector('div[data-plotly]');
            if (!g) {
              const cand = Array.from(document.querySelectorAll('div'));
              g = cand.find(d => d && d._fullLayout);
            }
            return g;
          }

          function emitRange(){
            const g = getGraph();
            if (!g || !window.Plotly) return;
            const x = g.layout?.xaxis?.range;
            const y = g.layout?.yaxis?.range;
            if (x && y) {
              try {
                console.log('RANGE:' + JSON.stringify({'xaxis.range':x, 'yaxis.range':y}));
              } catch(e){}
            }
          }

          function install(){
            const g = getGraph();
            if (!g || !window.Plotly) { setTimeout(install, 200); return; }

            // Catch all interactions that change zoom/pan
            g.on('plotly_relayout', emitRange);
            g.on('plotly_doubleclick', emitRange);
            g.on('plotly_afterplot', emitRange);
            g.on('plotly_redraw', emitRange);
            g.on('plotly_autosize', emitRange);
            g.on('plotly_restyle', emitRange);

            //  Support mouse wheel zoom
            g.addEventListener('wheel', () => setTimeout(emitRange, 200));

            // 🔹 Support laptop touchpad pinch / scroll gestures
            g.addEventListener('gesturechange', () => setTimeout(emitRange, 200));
            g.addEventListener('touchmove', () => setTimeout(emitRange, 200));

            // 🔹 Function called from Python to apply the other heatmap's range
            window.__pie_applyRelayout = function(payload){
              try {
                const g2 = getGraph();
                if (g2 && window.Plotly) Plotly.relayout(g2, payload);
              } catch(err){}
            };
          }

          install();
        })();
        """
        self.page().runJavaScript(js)
        self._installed = True


    def apply_relayout(self, payload: dict):
        """Apply ranges from the other view (with a feedback guard)."""
        if self._busy:
            return
        self._busy = True
        self.page().runJavaScript(
            f"window.__pie_applyRelayout({payload!r});",
            lambda _=None: self._clear_busy()
        )

    def _clear_busy(self):
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, lambda: setattr(self, "_busy", False))

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPen, QPolygonF
from PyQt6.QtCore import QPointF


class MyMainWindow(QMainWindow):
    REQUIRED_TALLY_COLS = [
        r"Abs. Distance (m)", r"Depth %", r"Type",
        r"ERF (ASME B31G)", r"Orientation o' clock"
    ]

    def __init__(self):
        super().__init__()
        self.ui = Form()

        self.ui.setupUi(self)
        # self._pipe_locator_dialog = None
        # self._pipe_locator_lock = False
        self._pipe_locator_dialog = None

        # Hide unwanted menu actions
        # if hasattr(self.ui, "action_Pipe_Locator"):
        #     self.ui.action_Pipe_Locator.setVisible(False)

        if hasattr(self.ui, "action_Pipe_Locator"):
            self.ui.action_Pipe_Locator.setVisible(True)
            self.ui.action_Pipe_Sch.setEnabled(True)
            self.ui.action_Pipe_Locator.triggered.connect(self.open_pipe_locator)

        if hasattr(self.ui, "action_ERF"):
            self.ui.action_ERF.setVisible(True)

        if hasattr(self.ui, "action_Pipe_Sch"):
            self.ui.action_Pipe_Sch.setVisible(True)
            self.ui.action_Pipe_Sch.setEnabled(False)

        for tb in self.findChildren(QtWidgets.QToolBar):
            if self.toolBarArea(tb) == Qt.ToolBarArea.LeftToolBarArea:
                self.removeToolBar(tb)
                tb.setParent(None)
                tb.deleteLater()
        self.menuBar().setStyleSheet("""
    QMenuBar {
        background-color: #000000;
        color: white;
    }
    QMenuBar::item {
        background: transparent;
        padding: 4px 12px;
    }
    QMenuBar::item:selected {
        background: #333333;
        color: white;
    }

    /* Dropdown menus stay white */
    QMenu {
        background-color: #ffffff;
        color: black;
        border: 1px solid #cccccc;
    }
    QMenu::item:selected {
        background: #c0c0c0;
        color: #000000;
    }
""")
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
        self._selected_columns: set[str] = set()
        # --- unit config ---
        self._unit_columns = {
            "Abs. Distance": "m",
            "Distance to U/S GW": "m",
            "Pipe Length": "mm",
            "WT": "mm",
            "Width": "mm",
            "Length": "mm",
            "Depth": "mm",
        }

        self._unit_factor = {
            "m": 1.0,
            "cm": 100.0,
            "mm": 1000.0,
            "km": 0.001,
            "feet": 3.28084,
        }

        self.hhmap = None  # hallsensor_heatmap*.html
        self.phmap = None  # proximity_heatmap*.html
        self._hm_layout_mode = "vertical"  # "horizontal" = side-by-side, "vertical" = stacked
        self.hm_left_ratio = 0.40  # 50-50 split in side-by-side mode

        # ✅ Initialize "No Defects Found" label
        self._no_defects_label = None

        # Threading setup
        self.loader_worker = None
        self.loading_dialog = None

        self.ui.comboBoxPipe.setEditable(True)
        import os

        arrow_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "arrow_down.svg").replace("\\", "/")

        self.ui.comboBoxPipe.setStyleSheet(f"""
            QComboBox {{
                padding: 4px 8px;
                border: 2px solid #000000;
                border-radius: 6px;
                background: white;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border-left: 2px solid #000000;
            }}
            QComboBox::down-arrow {{
                image: url({arrow_path});
                width: 12px;
                height: 12px;
            }}
            QComboBox QAbstractItemView {{
                border: 2px solid #000000; 
                selection-background-color: #3498db;
                selection-color: white;
            }}
        """)
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
        # ✅ Prevent the tables from auto-resizing to content (so scrollbars appear)
        self.ui.tableWidgetDefect.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )

        self.ui.tableView.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )

        # Digsheet button (ABS-based)
        self.btnDigsheetAbs = QPushButton("Digsheet")
        self.btnDigsheetAbs.setToolTip("Select an Absolute Distance cell in the defect table (on Heatmap/3D) to enable.")
        self.btnDigsheetAbs.setEnabled(False)
        self.btnDigsheetAbs.setStyleSheet("""
            QPushButton {
                background: white;
                border: 1px solid #3498db;
                color: #3498db;
                border-radius: 6px;
                padding: 4px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #ecf6fd;
            }
            QPushButton:pressed {
                background: #d0e9fa;
            }
            QPushButton:disabled {
                color: #a0a0a0;
                background: #f5f5f5;
                border: 2px solid #6e6e6e;
            }
        """)
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

        # Add Load button next to comboBoxPipe
        self.btnLoadPipe = QPushButton("Load")
        self.btnLoadPipe.setEnabled(False)
        self.btnLoadPipe.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: 1px solid #2980b9;
                border-radius: 6px;
                padding: 4px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f5f8a;
            }
            QPushButton:disabled {
            background-color: #a6a6a6;   
            color: #f0f0f0;              
            border: 2px solid #6e6e6e;   
        }
        """)
        _parent = self.ui.comboBoxPipe.parentWidget()
        _lay = _parent.layout()
        if _lay is not None:
            pos = _lay.indexOf(self.ui.comboBoxPipe)
            if pos != -1:
                _lay.insertWidget(pos + 1, self.btnLoadPipe)
            else:
                _lay.addWidget(self.btnLoadPipe)
        else:
            self.btnLoadPipe.setParent(_parent)

        # connect the load button
        self.btnLoadPipe.clicked.connect(self.load_selected_pipe)

        #         # --- Column Filter UI ---
        # self.columnFilter = QComboBox(self)
        # self.columnFilter.setEditable(True)  # allow search
        # self.columnFilter.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        # self.columnFilter.setMaxVisibleItems(15)
        # # cf_model = QStandardItemModel()          # <- use the top-level import
        # # self.columnFilter.setModel(cf_model)

        # self._cf_model = QStandardItemModel(self.columnFilter)
        # self.columnFilter.setModel(self._cf_model)


        # self.btnApplyFilter = QPushButton("Apply Columns", self)
        # self.btnApplyFilter.setEnabled(True)
        # self.btnApplyFilter.clicked.connect(self.apply_column_filter)
        # self.btnOpenFilterDlg = QPushButton("Filter Columns…", self)
        # self.btnOpenFilterDlg.clicked.connect(self.open_column_filter_dialog)

        # _parent = self.ui.comboBoxPipe.parentWidget()
        # if _parent and _parent.layout():
        #     pos = _parent.layout().indexOf(self.btnApplyFilter)
        #     _parent.layout().insertWidget(pos + 1, self.btnOpenFilterDlg)
        # else:
        #     self.btnOpenFilterDlg.setParent(_parent)


        # self.columnFilter.view().pressed.connect(self._on_column_item_pressed)

        # # Make the line edit a read-only summary like "5 selected"
        # self.columnFilter.setEditable(True)
        # self.columnFilter.lineEdit().setReadOnly(True)
        # self.columnFilter.lineEdit().setPlaceholderText("Columns…")

        # # insert into same row as comboBoxPipe + btnLoadPipe
        # _parent = self.ui.comboBoxPipe.parentWidget()
        # if _parent and _parent.layout():
        #     pos = _parent.layout().indexOf(self.btnLoadPipe)
        #     _parent.layout().insertWidget(pos + 1, self.columnFilter)
        #     _parent.layout().insertWidget(pos + 2, self.btnApplyFilter)

        # self.btnOpenFilterDlg = QPushButton("Filter Columns…")

        # # Style: white background, blue border, rounded corners
        # self.btnOpenFilterDlg.setStyleSheet("""
        #     QPushButton {
        #         background-color: #ffffff;
        #         color: #3498db;
        #         border: 1.5px solid #3498db;
        #         border-radius: 6px;
        #         padding: 4px 12px;
        #         font-weight: 500;
        #     }
        #     QPushButton:hover {
        #         background-color: #ecf6fd;   /* very light blue */
        #     }
        #     QPushButton:pressed {
        #         background-color: #d6ebfa;   /* slightly darker when pressed */
        #     }
        #     QPushButton:disabled {
        #         background-color: #f2f2f2;
        #         color: #a6a6a6;
        #         border: 1.5px solid #cccccc;
        #     }
        # """)

        # self.btnOpenFilterDlg.setCursor(Qt.CursorShape.PointingHandCursor)
        # self.btnOpenFilterDlg.setToolTip("Choose which columns to show in the table below")




        from PyQt6.QtGui import QIcon
        from PyQt6.QtCore import QSize

        # create the button (you already have this)
        self.btnOpenFilterDlg = QPushButton("Filter Columns", self)
        self.btnOpenFilterDlg.setEnabled(False)

        # attach icon
        filter_icon_path = resource_path("ui/icons/filter.svg")   # or .png
        self.btnOpenFilterDlg.setIcon(QIcon(filter_icon_path))
        self.btnOpenFilterDlg.setIconSize(QSize(16, 16))          # 16–18px works well for a 28px-high button
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


        self.btnOpenFilterDlg.clicked.connect(self.open_column_filter_dialog)

        _parent = self.ui.comboBoxPipe.parentWidget()
        if _parent and _parent.layout():
            pos = _parent.layout().indexOf(self.btnLoadPipe)
            _parent.layout().insertWidget(pos + 2, self.btnOpenFilterDlg)
        else:
            self.btnOpenFilterDlg.setParent(_parent)


                # Create the dropdown tab switcher
        self.tabSwitcherDropdown = QComboBox(self)
        self.tabSwitcherDropdown.setToolTip("Switch between chart tabs")
        self.tabSwitcherDropdown.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tabSwitcherDropdown.setMinimumWidth(120)
        self.tabSwitcherDropdown.setMaximumWidth(150)

        # Style the dropdown to match your other buttons
        arrow_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "arrow_down.svg").replace("\\", "/")

        self.tabSwitcherDropdown.setStyleSheet(f"""
            QComboBox {{
                background-color: #FFFFFF;
                color: #000000;
                border: 1.5px solid #000000;
                border-radius: 6px;
                padding: 4px 12px;
                font-weight: 500;
            }}
            QComboBox:hover {{
                background-color: #d6d3ce;
            }}
            QComboBox:pressed {{
                background-color: #111111;
                color: white;
            }}
            QComboBox:disabled {{
                background-color: #a6a6a6;     /* same as Load button */
                color: #f0f0f0;                /* same as Load button */
                border: 2px solid #6e6e6e;     /* same as Load button */
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1.5px solid #000000;
            }}
            QComboBox::down-arrow {{
                image: url({arrow_path});
                width: 12px;
                height: 12px;
            }}
        """)
        self.tabSwitcherDropdown.setEnabled(False)

        # Populate dropdown with tab names from tabWidgetM
        for i in range(self.ui.tabWidgetM.count()):
            tab_text = self.ui.tabWidgetM.tabText(i)
            self.tabSwitcherDropdown.addItem(tab_text)

        # Set current index to match tabWidgetM
        # self.tabSwitcherDropdown.setCurrentIndex(self.ui.tabWidgetM.currentIndex())
        # Set current index to Heatmap (index 0) by default
        self.tabSwitcherDropdown.setCurrentIndex(0)

        # Add the dropdown right after the filter button (pos + 3)
        _parent = self.ui.comboBoxPipe.parentWidget()
        if _parent and _parent.layout():
            pos = _parent.layout().indexOf(self.btnOpenFilterDlg)
            _parent.layout().insertWidget(pos + 1, self.tabSwitcherDropdown)
        else:
            self.tabSwitcherDropdown.setParent(_parent)

        # Connect the dropdown
        self.tabSwitcherDropdown.currentIndexChanged.connect(self.ondropdowntabchanged)

        print("✅ Tab switcher dropdown created and positioned after Filter button")

        # Create the Hide/Show Table toggle button
        self.btnToggleTable = QPushButton("Hide Table", self)
        # ✅ Correct single connection
        try:
            self.btnToggleTable.clicked.disconnect()
        except Exception:
            pass

        self._table_hidden = True
        self.btnToggleTable.setText("Show Table")
        # self.btnToggleTable.clicked.connect(self._toggle_table_visibility)

        self.btnToggleTable.setToolTip("Toggle table visibility (Heatmap only)")
        self.btnToggleTable.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnToggleTable.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #000000;
                border: 1.5px solid #000000;
                border-radius: 6px;
                padding: 4px 12px;
                font-weight: 500;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d6d3ce;
            }
            QPushButton:pressed {
                background-color: #111111;
                color: white;
            }
            QPushButton:disabled {
                background-color: #a6a6a6;     /* same as Load button */
                color: #f0f0f0;                /* same as Load button */
                border: 2px solid #6e6e6e;     /* same as Load button */
            }
        """)
        self.btnToggleTable.setEnabled(False)

        # Create the Stack/Side-by-side layout toggle button
        self.btnToggleHmLayout = QPushButton("Stack", self)
    #     self.btnToggleHmLayout.setStyleSheet("""
    #     QPushButton {
    #         background-color: #2b2b2b;
    #         color: white;
    #         border: 1px solid #3a3a3a;
    #         padding: 4px 10px;
    #         border-radius: 4px;
    #     }
    #     QPushButton:hover:!disabled {
    #         background-color: #444;
    #     }
    #     QPushButton:disabled {
    #         background-color: #1e1e1e;
    #         color: #666;
    #         border: 1px solid #2a2a2a;
    #     }
    # """)

        self.btnToggleHmLayout.setToolTip("Toggle dual-heatmap layout (side-by-side / stacked)")
        self.btnToggleHmLayout.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnToggleHmLayout.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #000000;
                border: 1.5px solid #000000;
                border-radius: 6px;
                padding: 4px 12px;
                font-weight: 500;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d6d3ce;
            }
            QPushButton:pressed {
                background-color: #111111;
                color: white;
            }
            QPushButton:disabled {
                background-color: #a6a6a6;     /* same as Load button */
                color: #f0f0f0;                /* same as Load button */
                border: 2px solid #6e6e6e;     /* same as Load button */
            }
        """)
        self.btnToggleHmLayout.setEnabled(False)

        # Add button right after the Toggle Table button
        # ✅ Put both buttons after the dropdown, in the order you want
        parent = self.ui.comboBoxPipe.parentWidget()
        if parent and parent.layout():
            row = parent.layout()
            # make sure they’re not already sitting somewhere in the row
            for w in (self.btnToggleTable, self.btnToggleHmLayout):
                try:
                    row.removeWidget(w)
                except Exception:
                    pass
                w.setParent(parent)

            pos = row.indexOf(self.tabSwitcherDropdown)
            row.insertWidget(pos + 1, self.btnToggleTable)     # Hide/Show
            row.insertWidget(pos + 2, self.btnToggleHmLayout)  # Stack (to the right)
        else:
            self.btnToggleTable.setParent(parent)
            self.btnToggleHmLayout.setParent(parent)


        # Connect the button
        self.btnToggleHmLayout.clicked.connect(lambda: self._apply_heatmap_layout(
            "vertical" if self._hm_layout_mode == "horizontal" else "horizontal"
        ))

        print("✅ Toggle Heatmap Layout button created")


        # Add button right after the dropdown
        parent = self.ui.comboBoxPipe.parentWidget()
        if parent and parent.layout():
            pos = parent.layout().indexOf(self.tabSwitcherDropdown)
            parent.layout().insertWidget(pos + 1, self.btnToggleTable)
        else:
            self.btnToggleTable.setParent(parent)

        # Connect the button
        # self.btnToggleTable.clicked.connect(self.toggletablevisibility)

        # Initialize flag - default is shown (False = not hidden)
        self._table_hidden = True
        self.btnToggleTable.setText("Show Table")
        self.btnToggleTable.clicked.connect(self._toggle_table_visibility)
        # self.bottom_stack.hide()

        print("✅ Toggle Table button created")

        self.ui.comboBoxPipe.currentIndexChanged.connect(self.update_load_button_state)

        # Global event filter for disabled-button popups + tabbar clicks
        QtWidgets.QApplication.instance().installEventFilter(self)

        # Resizable splitter with tabbar-handle
        self.mid_tabbar = QTabBar()
        for i in range(self.ui.tabWidgetM.count()):
            self.mid_tabbar.addTab(self.ui.tabWidgetM.tabText(i))
        self.mid_tabbar.setExpanding(False)
        # self.mid_tabbar.currentChanged.connect(lambda i: self.ui.tabWidgetM.setCurrentIndex(i))
        # self.ui.tabWidgetM.currentChanged.connect(lambda i: self.mid_tabbar.setCurrentIndex(i))
        self.mid_tabbar.currentChanged.connect(lambda i: [
            self.ui.tabWidgetM.setCurrentIndex(i),
            self._sync_dropdown_with_tabs(i)
        ][0])
        self.ui.tabWidgetM.currentChanged.connect(lambda i: [
            self.mid_tabbar.setCurrentIndex(i),
            self._sync_dropdown_with_tabs(i)
        ][0])

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
        self._setup_select_pipe_label()

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

        #self.ui.comboBoxPipe.currentIndexChanged.connect(self.on_combo_index_changed)

        # replace direct tab switcher with guarded handler
        try:
            self.ui.tabWidgetM.currentChanged.disconnect()
        except Exception:
            pass
        self.ui.tabWidgetM.currentChanged.connect(self._on_middle_tab_changed)
        self.ui.tabWidgetM.currentChanged.connect(self.syncdropdownwithtabs)


        # initial UI state
        self._toggle_plot_ui(False)
        self._update_project_actions()  # Create enabled, Close disabled

        self.setStyleSheet("QMainWindow { background-color: #FFFFFF; color: #000000; }")
        self.showMaximized()

        # mark UI ready on next tick (prevents popup at startup)
        QTimer.singleShot(0, lambda: setattr(self, "_ui_ready", True))

        # try:
        #     excel_path = resource_path("14inch Petrofac pipetally.xlsx")
        #     if os.path.exists(excel_path) and self.pipe_tally is None:
        #         self.pipe_tally = pd.read_excel(excel_path)
        # except Exception:
        #     pass

        self._show_watermark()

    def _reset_ui_to_start_state(self):
        # mark app state
        self.project_is_open = False

        # clear data/paths
        for attr in [
            "curr_data", "pipe_tally", "hmap", "hmap_r", "heatmap_box",
            "lplot", "lplot_r", "pipe3d", "prox_linechart", "hhmap", "phmap"
        ]:
            setattr(self, attr, None)
        self.pkl_files = []
        self.project_root = None

        # combo + load
        cb = self.ui.comboBoxPipe
        cb.blockSignals(True)
        cb.clear(); cb.addItem("-Pipe-"); cb.setCurrentIndex(0)
        cb.blockSignals(False)
        self.btnLoadPipe.setEnabled(False)

        # tables
        try:
            self.ui.tableWidgetDefect.clear()
            self.ui.tableWidgetDefect.setRowCount(0)
            self.ui.tableWidgetDefect.setColumnCount(0)
            self.ui.tableWidgetDefect.hide()
        except Exception:
            pass

        # bottom area
        self._table_hidden = True
        if hasattr(self, "btnToggleTable"):
            self.btnToggleTable.setText("Show Table")
            self.btnToggleTable.setEnabled(False)
        if hasattr(self, "bottom_stack"):
            self.bottom_stack.hide()

        # top area → back to startup (single page + watermark)
        try:
            if hasattr(self, "top_stack"):
                self.top_stack.setCurrentIndex(0)   # single_chart_page
            # blank any old heatmaps / prox views
            for wname in ("web_view_left", "web_view_right", "web_view2"):
                if hasattr(self, wname):
                    getattr(self, wname).setUrl(QUrl())
            # show startup watermark in main web view
            self._show_watermark()
        except Exception:
            pass

        # disable heatmap layout toggle & dropdown until a project opens
        if hasattr(self, "btnToggleHmLayout"):
            self.btnToggleHmLayout.setEnabled(False)
        if hasattr(self, "tabSwitcherDropdown"):
            self.tabSwitcherDropdown.setCurrentIndex(0)
            self.tabSwitcherDropdown.setEnabled(False)

        # disable graph tabs and update menu actions
        self._toggle_plot_ui(False)
        self._update_project_actions()

        # show the “Create Project” overlay again
        if hasattr(self, "_show_create_project_message"):
            self._show_create_project_message()

        # reset scroll sync guards
        self._hscroll_ready = False
        self._hscroll_ready_main = False
        self._hscroll_ready_table = False

    # def _apply_unit_conversion_to_table(self):
    #     if not hasattr(self, "column_units"):
    #         return
    #
    #     conversion = {
    #         "m": 1,
    #         "cm": 100,
    #         "mm": 1000,
    #         "feet": 3.28084,
    #         "km": 0.001,
    #     }
    #
    #     table = self.ui.tableWidgetDefect
    #     cols = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    #
    #     for col_name, unit in self.column_units.items():
    #         if col_name not in cols:
    #             continue
    #
    #         multiplier = conversion.get(unit, 1)
    #         col_idx = cols.index(col_name)
    #
    #         for r in range(table.rowCount()):
    #             item = table.item(r, col_idx)
    #             if not item:
    #                 continue
    #
    #             try:
    #                 base_value = float(item.text())  # assuming original is in meters/mm
    #                 new_value = base_value * multiplier
    #                 item.setText(f"{new_value:.3f}")
    #             except:
    #                 pass

    def _apply_unit_conversion(self, selected_units: dict):
        table = self.ui.tableWidgetDefect
        if table.columnCount() == 0:
            return

        for c in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(c)
            if not header_item:
                continue

            header = header_item.text()
            base = header.split(" (")[0]

            if base not in selected_units:
                continue

            new_unit = selected_units[base]
            old_unit = self._unit_columns.get(base, new_unit)

            if old_unit == new_unit:
                continue

            factor = self._unit_factor[new_unit] / self._unit_factor[old_unit]

            for r in range(table.rowCount()):
                item = table.item(r, c)
                if not item:
                    continue
                try:
                    val = float(item.text())
                    item.setText(f"{val * factor:.3f}")
                except:
                    pass

            header_item.setText(f"{base} ({new_unit})")
            self._unit_columns[base] = new_unit

            table.viewport().update()
            table.horizontalHeader().repaint()


    # def _on_column_item_pressed(self, index):
    #     """Toggle the check state for a pressed item and keep the popup open."""
    #     m = self.columnFilter.model()
    #     item = m.itemFromIndex(index)
    #     if not item:
    #         return
    #     item.setCheckState(
    #         Qt.CheckState.Unchecked
    #         if item.checkState() == Qt.CheckState.Checked
    #         else Qt.CheckState.Checked
    #     )
    #     # Re-show popup so it doesn't close on each click
    #     QTimer.singleShot(0, self.columnFilter.showPopup)
    #     # Refresh summary text
    #     self._column_summary_text()


    # def _toggle_table_visibility(self):
    #     self._table_hidden = not self._table_hidden
    #     if self._table_hidden:
    #         self.bottom_stack.hide()
    #         self.btnToggleTable.setText("Show Table")
    #     else:
    #         self.bottom_stack.show()
    #         self.btnToggleTable.setText("Hide Table")


    def _reset_splitter_ratio(self, top_ratio: float = 0.6):
        """Force consistent top/bottom height ratio for the stack layout."""
        if not hasattr(self, "splitter"):
            return

        def apply_ratio():
            sizes = self.splitter.sizes()
            total = sum(sizes) if sizes else self.splitter.height()
            if total > 0:
                top = int(total * top_ratio)
                bottom = total - top
                self.splitter.setSizes([top, bottom])
                # optional debug
                print(f"[DEBUG] Splitter resized: top={top}, bottom={bottom}, total={total}")

        # 🔹 Delay the resize slightly so the layout stabilizes first
        QTimer.singleShot(120, apply_ratio)



    def _toggle_table_visibility(self):
        """Show/hide bottom defect table."""
        self._table_hidden = not self._table_hidden

        if self._table_hidden:
            self.bottom_stack.hide()
            self.btnToggleTable.setText("Show Table")
            print("Table visibility toggled: Hidden")
        else:
            # Ensure the correct bottom page is visible (in case it's a QStackedWidget)
            if hasattr(self, "defect_table_page") and self.bottom_stack.indexOf(self.defect_table_page) != -1:
                self.bottom_stack.setCurrentWidget(self.defect_table_page)

            self.bottom_stack.show()
            self.btnToggleTable.setText("Hide Table")

            # 🔹 Ensure bottom area has height when showing
            if hasattr(self, "splitter"):
                sizes = self.splitter.sizes()
                if len(sizes) >= 2 and sizes[1] < 40:
                    total = max(sum(sizes), self.height())
                    bot = max(250, total // 3)
                    self.splitter.setSizes([total - bot, bot])

            print("Table visibility toggled: Shown")
            QTimer.singleShot(100, self._refresh_table_scrollbars)
            QTimer.singleShot(300, self._reset_table_state)

    def _sync_heatmap_range(self, target_view, payload):
        """Synchronize zoom/pan between both heatmaps."""
        if not isinstance(target_view, SyncPlotlyView):
            return

        clean_payload = {}
        if "xaxis.range" in payload:
            clean_payload["xaxis.range"] = payload["xaxis.range"]
        if "yaxis.range" in payload:
            clean_payload["yaxis.range"] = payload["yaxis.range"]

        # Apply to the other view
        target_view.apply_relayout(clean_payload)

    def _refresh_table_scrollbars(self):
        """Force scrollbar redraw for all tables."""
        for tw in (getattr(self.ui, "tableWidgetDefect", None),
                getattr(self.ui, "tableView", None)):
            if tw:
                hsb, vsb = tw.horizontalScrollBar(), tw.verticalScrollBar()
                if hsb: hsb.update()
                if vsb: vsb.update()


    def _current_headers_for_filter(self) -> list[str]:
        """Mirror the same header source used by _refresh_column_filter_options()."""
        headers = []
        if hasattr(self.ui, "tableWidgetDefect") and self.ui.tableWidgetDefect.columnCount() > 0:
            headers = [
                (self.ui.tableWidgetDefect.horizontalHeaderItem(c).text()
                if self.ui.tableWidgetDefect.horizontalHeaderItem(c) else f"Col {c}")
                for c in range(self.ui.tableWidgetDefect.columnCount())
            ]
        elif hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
            model = self.ui.tableView.model()
            headers = [str(model.headerData(c, Qt.Orientation.Horizontal)) for c in range(model.columnCount())]
        return headers

    def _currently_checked_in_dropdown(self) -> set[str]:
        """Read the check state from the existing dropdown (_cf_model)."""
        out = set()
        for r in range(self._cf_model.rowCount()):
            it = self._cf_model.item(r)
            if it.checkState() == Qt.CheckState.Checked:
                out.add(it.text())
        return out


    def open_column_filter_dialog(self):
        """Open column selector dialog and apply the result."""
        headers = self._current_headers_for_filter()
        locked = set(getattr(self, "BACKEND_LOCKED_COLS", set()))

        # default: first time, select everything that's not locked
        if not self._selected_columns:
            checked = set(h for h in headers if h not in locked)
        else:
            checked = set(h for h in self._selected_columns if h in headers and h not in locked)

        dlg = ColumnFilterDialog(headers=headers, checked=checked, locked=locked, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # self.column_units = dlg.selected_units()
        # self._apply_unit_conversion_to_table()

        # persist + apply (locked are always enforced)
        self._selected_columns = set(dlg.selected_names()) | locked
        #self._selected_columns = set(dlg.selected_names())
        self._apply_unit_conversion(dlg.selected_units())
        self.apply_column_filter()


    # def apply_column_filter(self):
    #     """Hide/show columns based on self._selected_columns + locked columns."""
    #     locked = set(getattr(self, "BACKEND_LOCKED_COLS", set()))
    #
    #     # If we have no selection yet, treat as 'show all'
    #     if not self._selected_columns:
    #         self._selected_columns = set(self._current_headers_for_filter()) | locked
    #
    #     names_to_keep = set(self._selected_columns) | locked
    #
    #     # Prefer bottom QTableWidgetDefect if it has columns
    #     if hasattr(self.ui, "tableWidgetDefect") and self.ui.tableWidgetDefect.columnCount() > 0:
    #         header_map = {
    #             c: (self.ui.tableWidgetDefect.horizontalHeaderItem(c).text()
    #                 if self.ui.tableWidgetDefect.horizontalHeaderItem(c) else f"Col {c}")
    #             for c in range(self.ui.tableWidgetDefect.columnCount())
    #         }
    #         for c, name in header_map.items():
    #             hide = (name not in names_to_keep) and (name not in locked)
    #             self.ui.tableWidgetDefect.setColumnHidden(c, hide)
    #         QTimer.singleShot(0, self._refresh_table_scrollbars)
    #         return
    #
    #     # Fallback to the top QTableView
    #     if hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
    #         model = self.ui.tableView.model()
    #         header_names = [str(model.headerData(c, Qt.Orientation.Horizontal)) for c in range(model.columnCount())]
    #         for c, name in enumerate(header_names):
    #             hide = (name not in names_to_keep) and (name not in locked)
    #             self.ui.tableView.setColumnHidden(c, hide)

    def apply_column_filter(self):
        table = self.ui.tableWidgetDefect
        if table.columnCount() == 0:
            return

        # 🔹 selected column base names
        selected = set()
        for name in self._selected_columns:
            base = name
            for u in ["(m)", "(mm)", "(cm)", "(km)", "(feet)"]:
                base = base.replace(u, "")
            selected.add(base.strip())

        # 🔥 FIXED RULE: selected = show, unselected = hide
        for c in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(c)
            if not header_item:
                continue

            header = header_item.text()
            base = header
            for u in ["(m)", "(mm)", "(cm)", "(km)", "(feet)"]:
                base = base.replace(u, "")
            base = base.strip()

            if base in selected:
                table.setColumnHidden(c, False)  # SHOW
            else:
                table.setColumnHidden(c, True)  # HIDE

    def _on_column_item_pressed(self, index):
        """Toggle the check state; keep popup open and update summary."""
        item = self._cf_model.itemFromIndex(index)
        if not item:
            return
        item.setCheckState(
            Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
        )
        self._update_column_summary()
        # keep popup open for multi-select
        QTimer.singleShot(0, self.columnFilter.showPopup)


    def _column_summary_text(self):
        """Show 'N selected' in the combobox line edit."""
        m = self.columnFilter.model()
        checked = sum(1 for i in range(m.rowCount()) if m.item(i).checkState() == Qt.CheckState.Checked)
        if self.columnFilter.isEditable() and self.columnFilter.lineEdit():
            self.columnFilter.lineEdit().setText(f"{checked} selected" if checked else "None")


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
    def populate_column_filter(self, df: pd.DataFrame):
        """Fill dropdown with all DataFrame columns (checkable)."""
        model = self.columnFilter.model()
        model.clear()

        for col in df.columns:
            it = QStandardItem(str(col))
            # Make it user-checkable and enabled
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            it.setData(Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
            model.appendRow(it)

        # Update summary (e.g., "12 selected")
        self._column_summary_text()




    def _restore_all_columns(self):
        """Show all columns again (useful when closing a project)."""
        if hasattr(self.ui, "tableWidgetDefect"):
            for c in range(self.ui.tableWidgetDefect.columnCount()):
                self.ui.tableWidgetDefect.setColumnHidden(c, False)
        if hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
            model = self.ui.tableView.model()
            for c in range(model.columnCount()):
                self.ui.tableView.setColumnHidden(c, False)


    def _refresh_column_filter_options(self):
        headers = []
        if hasattr(self.ui, "tableWidgetDefect") and self.ui.tableWidgetDefect.columnCount() > 0:
            headers = [
                (self.ui.tableWidgetDefect.horizontalHeaderItem(c).text()
                if self.ui.tableWidgetDefect.horizontalHeaderItem(c) else f"Col {c}")
                for c in range(self.ui.tableWidgetDefect.columnCount())
            ]
        elif hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
            model = self.ui.tableView.model()
            headers = [str(model.headerData(c, Qt.Orientation.Horizontal)) for c in range(model.columnCount())]

        self._cf_model.clear()
        for name in headers:
            if name in self.BACKEND_LOCKED_COLS:
                continue  # ← don't show in dropdown, but still exists in table
            it = QStandardItem(name)
            it.setCheckable(True)
            it.setCheckState(Qt.CheckState.Checked)
            it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._cf_model.appendRow(it)

        self._update_column_summary()





    def _update_column_summary(self):
        """Show 'All' / 'None' / 'N selected' in the combo line edit."""
        total = self._cf_model.rowCount()
        selected = sum(1 for r in range(total) if self._cf_model.item(r).checkState() == Qt.CheckState.Checked)
        if not self.columnFilter.isEditable() or not self.columnFilter.lineEdit():
            return
        if selected == 0:
            self.columnFilter.lineEdit().setText("None")
        elif selected == total:
            self.columnFilter.lineEdit().setText("All")
        else:
            self.columnFilter.lineEdit().setText(f"{selected} selected")




    def _setup_select_pipe_label(self):
        """Create a polished overlay asking user to select a pipe"""
        central = self.centralWidget()
        self._select_pipe_container = QWidget(central)
        self._select_pipe_container.setGeometry(central.rect())
        self._select_pipe_container.setStyleSheet("""
            background-color: rgba(255, 255, 255, 180);  /* frosted background */
        """)

        layout = QVBoxLayout(self._select_pipe_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- Inner card widget ---
        card = QFrame()
        card.setFixedWidth(500)
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 16px;
                border: 1px solid #d0d0d0;
                padding: 30px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Icon
        icon_label = QLabel("📂")
        icon_label.setStyleSheet("font-size: 42px;")
        card_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Title
        title = QLabel("No Pipe Selected")
        title.setStyleSheet("""
            font-size: 22pt;
            font-weight: 600;
            color: #2c3e50;
        """)
        card_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        # Subtitle
        subtitle = QLabel("Please choose a pipe number from the list above to continue.")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 12pt;
            color: #555;
            margin-top: 10px;
        """)
        card_layout.addWidget(subtitle)

        # Hint / efficiency tip
        hint = QLabel("💡 You can also type a pipe number directly in the box.")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("""
            font-size: 10pt;
            color: #888;
            margin-top: 15px;
        """)
        card_layout.addWidget(hint)

        layout.addWidget(card)
        self._select_pipe_container.hide()

    # ✅ Helper methods for showing/hiding message vs table
    def _show_no_defects_message(self):
        try:
            if hasattr(self, '_no_defects_container'):
                self._no_defects_container.show()
            if hasattr(self.ui, 'tableWidgetDefect'):
                self.ui.tableWidgetDefect.clearSelection()
                self.ui.tableWidgetDefect.hide()
            if hasattr(self, 'table_scrollbar'):
                self.table_scrollbar.hide()

            if hasattr(self, 'left_vscrollbar'):
                self.left_vscrollbar.hide()

        except Exception as e:
            print(f"Error showing no defects message: {e}")

    def _force_table_scroll_update(self):
        """Force table to refresh layout and scroll range after re-showing."""
        try:
            tw = getattr(self.ui, "tableWidgetDefect", None)
            if not tw:
                return

            tw.viewport().update()
            tw.updateGeometry()
            tw.resizeRowsToContents()

            tw.horizontalScrollBar().setValue(0)
            tw.verticalScrollBar().update()
            tw.horizontalScrollBar().update()
            print("[DEBUG] Table scroll recalculated.")
        except Exception as e:
            print(f"[ERROR] Scroll recalculation failed: {e}")

    def _reset_table_state(self):
        """Force reset of table state when re-entering a pipe."""
        try:
            tw = self.ui.tableWidgetDefect
            if not tw:
                return
            # Reset batching state variables
            self._is_filling_table = False
            self._pending_close_loader = False
            self._table_fill_df = None
            self._table_fill_row = 0

            # Force Qt to rebuild scroll region
            tw.clearSelection()
            tw.viewport().update()
            tw.updateGeometry()
            tw.verticalScrollBar().setValue(0)
            tw.horizontalScrollBar().setValue(0)
            tw.verticalScrollBar().update()
            tw.horizontalScrollBar().update()
            QTimer.singleShot(200, self._refresh_table_scrollbars)
            print("[DEBUG] Table state reset and scrollbars refreshed.")
        except Exception as e:
            print(f"[ERROR] Table reset failed: {e}")



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

            if hasattr(self, 'left_vscrollbar'):
                self.left_vscrollbar.show()

            QTimer.singleShot(150, self._refresh_table_scrollbars)
            QTimer.singleShot(200, self._force_table_scroll_update)
            QTimer.singleShot(250, self._reset_table_state)



            print("📊 Displaying defects table")
        except Exception as e:
            print(f"Error showing defects table: {e}")

    def _show_select_pipe_message(self):
        if hasattr(self, "_select_pipe_container"):
            central = self.centralWidget().rect()

            # Leave space for the pipe selection row (comboBox + Load button)
            header_height = self.ui.comboBoxPipe.height() + 20

            self._select_pipe_container.setGeometry(
                0,
                header_height,
                central.width(),
                central.height() - header_height
            )
            self._select_pipe_container.show()

        # Hide other views
        if hasattr(self.ui, "tableWidgetDefect"):
            self.ui.tableWidgetDefect.hide()
        if hasattr(self.ui, "tableView"):
            self.ui.tableView.hide()

        self.btnLoadPipe.setEnabled(False)


    def _update_project_actions(self):
        a = self.ui
        act_create = getattr(a, "action_Create_Proj", None)
        act_close = getattr(a, "action_Close_Proj", None)
        act_graphs = getattr(a, "action_graphs", None)
        act_xyz = getattr(a, "action_XYZ", None)
        act_pipehigh = getattr(a, "action_Pipe_High", None)
        if isinstance(act_create, QAction):
            act_create.setEnabled(not self.project_is_open)
        if isinstance(act_close, QAction):
            act_close.setEnabled(self.project_is_open)
        if isinstance(act_graphs, QAction):
            act_graphs.setEnabled(self.project_is_open)
        if isinstance(act_xyz, QAction):  # ← Add this block
            act_xyz.setEnabled(self.project_is_open)
        if isinstance(act_pipehigh, QAction):  # ← ADD THIS BLOCK
            act_pipehigh.setEnabled(self.project_is_open)
        self._update_generate_actions()


    def open_graphs_window(self):
        if self.pipe_tally is None:
            QMessageBox.warning(self, "No Pipe Tally", "Please create or load a project first.")
            return

        if self._central_graphs is None:
            self._central_graphs = GraphApp(self.pipe_tally,self.project_root)
        self.setCentralWidget(self._central_graphs)

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
        if hasattr(self.ui, "action_graphs"):
            self.ui.action_graphs.triggered.connect(self.open_graphs_window)

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

    def _install_left_vbar(self, tw: QtWidgets.QTableWidget):
        """
        Place a custom vertical scrollbar inside the table's left margin and
        sync it to the table's internal vertical scrollbar.
        """
        LEFT_GUTTER = 16  # width for the left vbar inside the table
        # Reserve space on the left *inside* the table for our bar
        tw.setViewportMargins(LEFT_GUTTER, 0, 0, 0)

        # Create the bar as a child of the table so it sits in the viewport area
        self.left_vbar = QScrollBar(Qt.Orientation.Vertical, tw)
        self.left_vbar.setObjectName("leftTableVBar")
        self.left_vbar.setStyleSheet(SCROLLBAR_STYLE)
        self.left_vbar.setFixedWidth(LEFT_GUTTER)

        # Hide the table's built-in right vbar; we will drive it via the left one
        tw.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner_vbar = tw.verticalScrollBar()  # still exists, just hidden

        # keep ranges/values in sync
        def _apply_range():
            self.left_vbar.blockSignals(True)
            self.left_vbar.setRange(inner_vbar.minimum(), inner_vbar.maximum())
            self.left_vbar.setPageStep(inner_vbar.pageStep())
            self.left_vbar.setSingleStep(inner_vbar.singleStep())
            self.left_vbar.setValue(inner_vbar.value())
            self.left_vbar.blockSignals(False)

        def _on_left_changed(v):
            inner_vbar.setValue(v)

        def _on_inner_changed(v):
            self.left_vbar.blockSignals(True)
            self.left_vbar.setValue(v)
            self.left_vbar.blockSignals(False)

        def _on_inner_range_changed(_min, _max):
            _apply_range()

        self.left_vbar.valueChanged.connect(_on_left_changed)
        inner_vbar.valueChanged.connect(_on_inner_changed)
        inner_vbar.rangeChanged.connect(_on_inner_range_changed)

        # position the left bar so it starts below the header and fills the viewport height
        self._update_left_vbar_geometry(tw)
        tw.installEventFilter(self)  # so we can reposition it on resize/show

        # first-time sync after layout settles
        QTimer.singleShot(0, _apply_range)
        self._style_left_vertical_bar()

    def _update_left_vbar_geometry(self, tw: QtWidgets.QTableWidget):
        """Keep the left scrollbar aligned with the table’s viewport (below header)."""
        try:
            header_h = tw.horizontalHeader().height() if tw.horizontalHeader() else 0
            x = 0
            y = header_h
            w = self.left_vbar.width()
            h = tw.viewport().height()
            self.left_vbar.setGeometry(x, y, w, h)
            self.left_vbar.raise_()
        except Exception:
            pass

    def _build_splitter(self):
        # ---------- tiny local helpers so this method is self-contained ----------
        if not hasattr(self, "_hm_layout_mode"):
            self._hm_layout_mode = "vertical"  # persisted layout mode


        # ---------- TOP: build a stack (single view + dual heatmaps) ----------
        self.main_web_page = QWidget()
        main_web_layout = QVBoxLayout(self.main_web_page)
        main_web_layout.setContentsMargins(0, 0, 0, 0)
        main_web_layout.setSpacing(0)

        # page 0: original single chart page (used by Line/3D)
        self.single_chart_page = QWidget()
        single_lay = QVBoxLayout(self.single_chart_page)
        single_lay.setContentsMargins(0, 0, 0, 0)
        single_lay.setSpacing(0)

        self.main_web_scroll_area = QScrollArea()
        self.main_web_scroll_area.setWidgetResizable(False)
        self.main_web_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.main_web_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self.web_view = QWebEngineView()
        self.web_view.setFixedSize(2500, 650)
        self.main_web_scroll_area.setWidget(self.web_view)
        single_lay.addWidget(self.main_web_scroll_area)

        # page 1: dual heatmaps page (NEW)
        self.dual_heatmaps_page = QWidget()
        dual_lay = QVBoxLayout(self.dual_heatmaps_page)
        dual_lay.setContentsMargins(0, 0, 0, 0)
        dual_lay.setSpacing(6)



                # --- tiny toolbar with the toggle button + show/hide table ---
        top_toolbar = QHBoxLayout()
        top_toolbar.setContentsMargins(8, 6, 8, 4)
        top_toolbar.setSpacing(8)


        top_toolbar.addStretch(1)
        dual_lay.addLayout(top_toolbar)


        # --- the dual-heatmap splitter ---
        self.top_hsplit = QSplitter(Qt.Orientation.Horizontal if self._hm_layout_mode == "horizontal"
                                    else Qt.Orientation.Vertical)
        self.top_hsplit.setChildrenCollapsible(False)
        self.top_hsplit.setStretchFactor(0, 1)
        self.top_hsplit.setStretchFactor(1, 1)
        self.top_hsplit.setObjectName("TopHSplit")
        self.top_hsplit.setStyleSheet("""
            QSplitter#TopHSplit::handle {
                background-color: #3a3a3a;     /* darker, more visible */
                border: 1px solid #2a2a2a;     /* subtle edge so it stands out */
            }
            QSplitter#TopHSplit::handle:hover {
                background-color: #4a4a4a;     /* a touch brighter on hover */
            }
        """)
        # left heatmap (Hall-sensor)
        self.web_view_left  = SyncPlotlyView(self)
        self.left_scroll  = QScrollArea(); self.left_scroll.setWidgetResizable(False)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.web_view_left.setFixedSize(2500, 650)
        self.left_scroll.setWidget(self.web_view_left)

        # right heatmap (Proximity)
        self.web_view_right = SyncPlotlyView(self)
        self.right_scroll = QScrollArea(); self.right_scroll.setWidgetResizable(False)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.web_view_right.setFixedSize(2500, 650)
        self.right_scroll.setWidget(self.web_view_right)

        self.top_hsplit.addWidget(self.left_scroll)
        self.top_hsplit.addWidget(self.right_scroll)
        dual_lay.addWidget(self.top_hsplit)

        # ensure sizes apply after first layout
        # QTimer.singleShot(0, _apply_heatmap_layout)
        QTimer.singleShot(0, self._apply_heatmap_layout)

        # stack
        self.top_stack = QStackedWidget()
        self.top_stack.addWidget(self.single_chart_page)   # index 0
        self.top_stack.addWidget(self.dual_heatmaps_page)  # index 1

        # add stack to layout
        main_web_layout.addWidget(self.top_stack)

        # top bar under the stack (used for single chart pages only)
        self.main_top_scrollbar = self._make_topbar_row(
            "mainTopBar", main_web_layout, bar_h=10, left_px=1300, right_px=570
        )

        # ---------- Keep your original main top scrollbar sync for the single page ----------
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
            if not getattr(self, "_hscroll_ready_main", False):
                return
            main_inner_hbar.setValue(_map_main_top_to_inner(v))

        def _on_main_inner_changed(v):
            if not getattr(self, "_hscroll_ready_main", False):
                return
            self.main_top_scrollbar.blockSignals(True)
            self.main_top_scrollbar.setValue(_map_main_inner_to_top(v))
            self.main_top_scrollbar.blockSignals(False)

        self._hscroll_ready_main = False
        self.main_top_scrollbar.valueChanged.connect(_on_main_top_changed)
        main_inner_hbar.valueChanged.connect(_on_main_inner_changed)

        def _on_main_inner_range_changed(_min, _max):
            if _max > _min:
                self._hscroll_ready_main = True
                _apply_main_fixed_range()

        main_inner_hbar.rangeChanged.connect(_on_main_inner_range_changed)

        # ---------- BOTTOM STACK ----------
        self.bottom_stack = QStackedWidget()
        self.bottom_stack.hide()
        self.bottom_stack.setContentsMargins(0, 0, 0, 0)
        self.bottom_stack.currentChanged.connect(lambda idx: self._arm_topbar() if idx == 2 else None)
        self._HM_FOOTER_H = 44  # tweak if you want more/less
        self.footer_page = QWidget(objectName="heatmapFooterPage")
        self.footer_page.setMinimumHeight(self._HM_FOOTER_H)
        self.footer_page.setMaximumHeight(self._HM_FOOTER_H)
        # --------------------------- Defect table page (bottom) ---------------------------
        self.defect_table_page = QWidget()
        defect_layout = QVBoxLayout(self.defect_table_page)
        defect_layout.setContentsMargins(0, 0, 0, 0)
        defect_layout.setSpacing(0)

        # Re-parent tableWidgetDefect into this page
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

        # Hide built-in vertical header and install left-side custom vbar
        vh = self.ui.tableWidgetDefect.verticalHeader()
        vh.setVisible(False)
        self.ui.tableWidgetDefect.setCornerButtonEnabled(False)
        self._install_left_vbar(self.ui.tableWidgetDefect)

        # Sync custom table bar with internal table hbar
        self._setup_table_scrollbar_sync()

        # --------------------------- Data table page (model view) ---------------------------
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

        # --------------------------- Proximity line chart page (bottom) ---------------------------
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
            if not getattr(self, "_hscroll_ready", False):
                return
            inner_hbar.setValue(_map_top_to_inner(v))

        def _on_inner_changed(v):
            if not getattr(self, "_hscroll_ready", False):
                return
            self.top_scrollbar.blockSignals(True)
            self.top_scrollbar.setValue(_map_inner_to_top(v))
            self.top_scrollbar.blockSignals(False)

        self._hscroll_ready = False
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
        self.bottom_stack.addWidget(self.footer_page)

        # ---------- Splitter with mid tabbar ----------
        self.splitter = MidBarSplitter(self, tabbar=self.mid_tabbar)
        self.splitter.setStretchFactor(0, 3)  # top area (heatmaps)
        self.splitter.setStretchFactor(1, 1)  # bottom area (table)
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

        # initial splitter sizes
        INIT_SPLIT_BOTTOM_RATIO = 0.45   # 45% bottom, 55% top
        QTimer.singleShot(0, lambda: self.splitter.setSizes([
            int(self.height() * (1 - INIT_SPLIT_BOTTOM_RATIO)),   # top
            int(self.height() * INIT_SPLIT_BOTTOM_RATIO)          # bottom
        ]))

        # ---------- Constrain splitter sizes + refresh top bars on move ----------
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
            if getattr(self, "_hscroll_ready", False):
                _apply_fixed_range()
            if getattr(self, "_hscroll_ready_main", False):
                _apply_main_fixed_range()
            QTimer.singleShot(10, self._refresh_table_scrollbars)

        self.splitter.splitterMoved.connect(_on_splitter_moved)




    def _setup_left_vertical_scrollbar_sync(self):
        """Sync the custom left vertical scrollbar with tableWidgetDefect's internal vbar."""
        tw = self.ui.tableWidgetDefect
        inner_vbar = tw.verticalScrollBar()  # still exists even if hidden
        left_vbar = self.left_vscrollbar

        # Mirror range/page/single step from the table's scrollbar
        def _apply_range():
            left_vbar.blockSignals(True)
            left_vbar.setRange(inner_vbar.minimum(), inner_vbar.maximum())
            left_vbar.setPageStep(inner_vbar.pageStep())
            left_vbar.setSingleStep(inner_vbar.singleStep())
            left_vbar.setValue(inner_vbar.value())
            left_vbar.blockSignals(False)

        # When user drags the left bar -> scroll table
        def _on_left_changed(v):
            inner_vbar.setValue(v)

        # When table scrolls (keyboard, wheel, selection, data fill, etc.) -> move left bar
        def _on_inner_changed(v):
            left_vbar.blockSignals(True)
            left_vbar.setValue(v)
            left_vbar.blockSignals(False)

        def _on_inner_range_changed(_min, _max):
            _apply_range()

        # Connect both ways
        left_vbar.valueChanged.connect(_on_left_changed)
        inner_vbar.valueChanged.connect(_on_inner_changed)
        inner_vbar.rangeChanged.connect(_on_inner_range_changed)

        # Initial apply on next tick (table might not have full range yet)
        QTimer.singleShot(0, _apply_range)


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
                tw.verticalScrollBar().setSingleStep(15)

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
                tv.verticalScrollBar().setSingleStep(15)

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
        self.right_status_label.setText("0.0s")

    def setup_actions(self):
        a = self.ui
        # a.action_Pipe_Locator.triggered.connect(self.open_pipe_locator())
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
        a.actionMetal_Loss_Distribution_MLD.triggered.connect(self.open_CMLD)
        a.actionDepth_Based_Anomalies_Distribution_DBAD.triggered.connect(self.open_DBAD)
        a.actionERF_Based_Anomalies_Distribution_E_AD.triggered.connect(self.open_EAD)
        a.action_Custom.triggered.connect(self.add_plot_custom)
        a.action_Telemetry.triggered.connect(self.add_plot_tele)
        a.actionAnomalies_Distribution.triggered.connect(self.add_plot_ad)
        a.action_DefectDetect.triggered.connect(self.draw_boxes_v2)
        if hasattr(a, "pushButtonNext"): a.pushButtonNext.clicked.connect(self.load_next_pipe)
        if hasattr(a, "pushButtonPrev"): a.pushButtonPrev.clicked.connect(self.load_prev_pipe)
        a.Final_Report.triggered.connect(self.open_Final_Report)
        a.action_Preliminary_Report.triggered.connect(self.open_Preliminary_Report)
        a.action__pipetally.triggered.connect(self.open_pipe_tally)
        a.action_Manual.triggered.connect(self.open_manual)
        a.actionStandard.triggered.connect(self.open_digs)  # original (by defect no.)
        a.action_Pipe_Locator.triggered.connect(self.open_pipe_locator)


    def load_next_pipe(self):
        """Go to next pipe and load automatically"""
        cb = self.ui.comboBoxPipe
        idx = cb.currentIndex()
        if idx < cb.count() - 1:  # not last
            cb.setCurrentIndex(idx + 1)
            self.load_selected_pipe()

    def load_prev_pipe(self):
        """Go to previous pipe and load automatically"""
        cb = self.ui.comboBoxPipe
        idx = cb.currentIndex()
        if idx > 0:  # not first
            cb.setCurrentIndex(idx - 1)
            self.load_selected_pipe()


    def _force_full_start_state(self):
        """Hard reset the UI to startup layout (Heatmap, table hidden, buttons off)."""
        # reset flags
        self._table_hidden = True
        self._hm_layout_mode = "vertical"
        self._last_allowed_tab_index = 0
        self._reverting_tab = False

        # top area → dual heatmap page
        if hasattr(self, "top_stack"):
            try:
                self.top_stack.setCurrentIndex(1)  # heatmap dual page
            except Exception:
                pass

        # hide bottom table area
        if hasattr(self, "bottom_stack"):
            self.bottom_stack.hide()
            self.bottom_stack.setCurrentIndex(0)

        # disable buttons
        if hasattr(self, "btnToggleTable"):
            self.btnToggleTable.setEnabled(False)
            self.btnToggleTable.setText("Show Table")
        if hasattr(self, "btnToggleHmLayout"):
            self.btnToggleHmLayout.setEnabled(False)
            self.btnToggleHmLayout.setText("Side-by-side")

        # reset middle tab and dropdown to Heatmap
        tw = getattr(self.ui, "tabWidgetM", None)
        if tw is not None:
            tw.blockSignals(True)
            tw.setCurrentIndex(0)
            tw.blockSignals(False)
        if hasattr(self, "tabSwitcherDropdown"):
            self.tabSwitcherDropdown.blockSignals(True)
            self.tabSwitcherDropdown.setCurrentIndex(0)
            self.tabSwitcherDropdown.blockSignals(False)

        # clear/blank out main web views
        for w in ("web_view", "web_view2", "web_view_left", "web_view_right"):
            if hasattr(self, w):
                getattr(self, w).setUrl(QUrl())
        self._show_watermark()


    def open_project(self):
        try:
            # hide overlay immediately when trying to open
            if hasattr(self, "_create_proj_container") and self._create_proj_container:
                self._create_proj_container.hide()

            dlg = QFileDialog(self)
            dlg.setFileMode(QFileDialog.FileMode.Directory)
            dlg.setOption(QFileDialog.Option.ShowDirsOnly)
            dlg.setWindowTitle("Select Project Folder (PKLs + pipe_* folders)")
            if dlg.exec() != QFileDialog.DialogCode.Accepted:
                self.project_is_open = False
                self._toggle_plot_ui(False)
                self._show_watermark()
                self._update_project_actions()

                # show overlay back if user cancelled
                if hasattr(self, "_create_proj_container") and self._create_proj_container:
                    self._create_proj_container.show()
                return

            root = dlg.selectedFiles()[0]
            self.project_root = root
            self._force_full_start_state()

            self.pipe_tally = None
            loaded_tally = self._auto_load_pipe_tally(root)
            if not loaded_tally:
                print("[pipe_tally] No tally file found in this project; graphs/reports will warn if needed.")

            pickle_data_dir = os.path.join(root, "pickle_data")
            if os.path.isdir(pickle_data_dir):
                self.pkl_files = [
                    os.path.join(pickle_data_dir, f)
                    for f in os.listdir(pickle_data_dir)
                    if f.lower().endswith(".pkl")
                ]
            else:
                self.pkl_files = []
                print(f"[Warning] pickle_data directory not found in {root}")

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
                cb.setCurrentIndex(-1)
            else:
                cb.addItem("-Pipe-")  # 👈 nothing selected

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
                self._force_heatmap_start()
                    # 🔹 Force-enable Heatmap control buttons since Heatmap is the first visible tab
                if hasattr(self, "btnToggleTable"):
                    self.btnToggleTable.setEnabled(True)
                    self.btnToggleTable.setText("Show Table")
                if hasattr(self, "btnToggleHmLayout"):
                    self.btnToggleHmLayout.setEnabled(True)
                    self.btnToggleHmLayout.setText("Side-by-side")



                # Show overlay instead of auto-loading
                self._show_select_pipe_message()

                # 👇 Force check so Load button activates if default pipe is already selected
                self.update_load_button_state(self.ui.comboBoxPipe.currentIndex())
            else:
                self.project_is_open = False
                self._toggle_plot_ui(False)
                self._show_watermark()
                QMessageBox.warning(self, "No PKLs", "No .pkl files found in the selected folder.")

                # show overlay back if no valid files
                if hasattr(self, "_create_proj_container") and self._create_proj_container:
                    self._create_proj_container.show()

            self._update_project_actions()
        except Exception as e:
            self.project_is_open = False
            self._toggle_plot_ui(False)
            self._show_watermark()
            self._update_project_actions()

            # show overlay back on error
            if hasattr(self, "_create_proj_container") and self._create_proj_container:
                self._create_proj_container.show()

            self.open_Error(e)
        self.ui.action_Pipe_Sch.setEnabled(True)

    def _style_left_vertical_bar(self):
        # icon paths
        up    = resource_path("ui/icons/arrow_up.svg").replace("\\", "/")
        down  = resource_path("ui/icons/arrow_down.svg").replace("\\", "/")

        # dimensions
        btn = 18       # arrow button size
        w   = 16       # bar width
        r   = 8        # thumb radius

        style = f"""
        /* entire bar */
        QScrollBar#leftTableVBar:vertical {{
            width:{w}px;
            margin:{btn + 2}px 0;           /* room for arrow buttons */
            background: transparent;
            border: none;
        }}

        /* the thumb */
        QScrollBar#leftTableVBar::handle:vertical {{
            min-height: 36px;
            border-radius:{r}px;
            background: #6b6b6b;
            border: 1px solid rgba(0,0,0,0.25);
        }}
        QScrollBar#leftTableVBar::handle:vertical:hover {{
            background: #7f7f7f;
        }}
        QScrollBar#leftTableVBar::handle:vertical:pressed {{
            background: #4f4f4f;
        }}

        /* top arrow */
        QScrollBar#leftTableVBar::sub-line:vertical {{
            height:{btn}px; width:{btn}px;
            subcontrol-origin: margin;
            subcontrol-position: top;
            border: none;
            border-radius:{btn//2}px;
            background: #e7e7e7;
            image: url("{up}");
        }}
        /* bottom arrow */
        QScrollBar#leftTableVBar::add-line:vertical {{
            height:{btn}px; width:{btn}px;
            subcontrol-origin: margin;
            subcontrol-position: bottom;
            border: none;
            border-radius:{btn//2}px;
            background: #e7e7e7;
            image: url("{down}");
        }}
        QScrollBar#leftTableVBar::sub-line:vertical:hover,
        QScrollBar#leftTableVBar::add-line:vertical:hover {{
            background: #d7d7d7;
        }}
        QScrollBar#leftTableVBar::sub-line:vertical:pressed,
        QScrollBar#leftTableVBar::add-line:vertical:pressed {{
            background: #c7c7c7;
        }}

        /* the “pages” above/below the thumb */
        QScrollBar#leftTableVBar::sub-page:vertical,
        QScrollBar#leftTableVBar::add-page:vertical {{
            background: #f2f2f2;
            border: none;
        }}
        """
        self.left_vbar.setStyleSheet(style)


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

    def load_selected_by_index(self, idx: int):
        try:
            if idx < 0 or idx >= len(self.pkl_files):
                return
            if hasattr(self, 'btnDigsheetAbs'):
                self.btnDigsheetAbs.setEnabled(False)
            if hasattr(self.ui, 'tableWidgetDefect'):
                self.ui.tableWidgetDefect.clearSelection()

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



    def load_selected_pipe(self):
        if not self.project_is_open:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return

        idx = self.ui.comboBoxPipe.currentIndex()
        text = self.ui.comboBoxPipe.currentText().strip()

        # ✅ If typed text matches an item, resolve index
        if idx < 0 and text:
            try:
                idx = [self.ui.comboBoxPipe.itemText(i) for i in range(self.ui.comboBoxPipe.count())].index(text)
            except ValueError:
                QMessageBox.warning(self, "Invalid Selection", f"No pipe named '{text}' found.")
                return

        if idx < 0 or idx >= len(self.pkl_files):
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid pipe.")
            return

        if hasattr(self, "_select_pipe_container"):
            self._select_pipe_container.hide()

        self.btnLoadPipe.setEnabled(False)
        self.load_selected_by_index(idx)

        #self.btnLoadPipe.clicked.connect(self.load_selected_pipe)


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



    def on_data_loaded(self, df):
        """Handle loaded DataFrame - runs on main thread"""
        self.curr_data = df
        self.header_list = list(df.columns)

        # Use lightweight model instead of building QStandardItem rows
        self.df_model = PandasModel(df)
        self.proxy_model.setSourceModel(self.df_model)

        self.ui.tableView.setSortingEnabled(True)

    def on_assets_loaded(self, assets):
        """Handle loaded assets"""
        self.hmap = assets.get("hmap")
        self.hmap_r = assets.get("hmap_r")
        self.heatmap_box = assets.get("heatmap_box")
        self.lplot = assets.get("lplot")
        self.lplot_r = assets.get("lplot_r")
        self.pipe3d = assets.get("pipe3d")
        self.prox_linechart = assets.get("prox_linechart")
        self.hhmap = assets.get("hallsensor_heatmap")
        self.phmap = assets.get("proximity_heatmap")

    def on_table_data_ready(self, df):
        """Handle processed table data"""
        self.curr_data = df  # 👈 make sure we keep a reference for filtering later

        if df is not None:
            # 👇 populate the column filter dropdown with available columns

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
        QTimer.singleShot(100, self.update_digsheet_button_state)
        # 👇 keep Load button disabled after file load
        self.btnLoadPipe.setEnabled(False)
        # Reset dropdown to Heatmap when pipe loads
        if hasattr(self, 'tabSwitcherDropdown'):
            self.tabSwitcherDropdown.blockSignals(True)
            self.tabSwitcherDropdown.setCurrentIndex(0)
            self.tabSwitcherDropdown.blockSignals(False)

        if hasattr(self, "btnOpenFilterDlg"):
            self.btnOpenFilterDlg.setEnabled(True)

        if hasattr(self, "tabSwitcherDropdown"):
            self.tabSwitcherDropdown.setEnabled(True)

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
        tw = self.ui.tableWidgetDefect
        tw.clearSelection()

        if df is None or df.empty:
            self._show_no_defects_message()
            return

        # original_count = len(df)
        # if "Feature Type" in df.columns:
        #     df = df[df["Feature Type"].astype(str).str.strip().str.lower() == "metal loss"]

        # if df.empty:
        #     print(f"⚠️ No Metal Loss defects found (filtered from {original_count} rows)")
        #     self._show_no_defects_message()
        #     return

        # original_count = len(df)
        # if "Feature Type" in df.columns:
        #     df = df[df["Feature Type"].astype(str).str.strip().str.lower() == "metal loss"]

        # if df.empty:
        #     print(f"⚠️ No Metal Loss defects found (filtered from {original_count} rows)")
        #     self._show_no_defects_message()
        #     return

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
            "Defect_id","Abs. Distance (m)","Distance to U/S GW(m)","Pipe Number","Pipe Length (mm)","Feature Type",
            "Feature Identification","Dimensions Classification","Orientation o' clock","WT (mm)","Length (mm)",
            "Width (mm)","Depth %","Depth (mm)","Location","ERF (ASME B31G)","Psafe (ASME B31G) Barg",
            "Latitude","Longitude" ,"Altitude","Comment","Empty"
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
            'Feature Type': 150,
            'Feature Identification': 150,
            'Dimensions Classification': 150,
            'Orientation o\' clock': 150,
            'WT (mm)': 150,
            'Length (mm)': 150,
            'Width (mm)': 150,
            'Depth %': 150,
            'Depth (mm)': 150,
            'Location': 150,
            'ERF (ASME B31G)': 150,
            'Psafe (ASME B31G) Barg': 150,
            'Latitude': 150,
            'Longitude': 150,
            'Altitude': 150,
            'Comment': 150,
            'Empty': 530
        }

        for c, col_name in enumerate(view.columns):
            if col_name in column_widths:
                tw.setColumnWidth(c, column_widths[col_name])
            else:
                tw.setColumnWidth(c, 100)

        self._show_defects_table()
        self._start_fill_qtablewidget_batched(view, chunk_size=300)

        setup_table_scroll(self.ui.tableWidgetDefect)
        QTimer.singleShot(150, self._refresh_table_scrollbars)


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


    # def _fill_tablewidget_chunk(self):
    #     """Append a batch of rows to QTableWidget without freezing UI."""
    #     tw = self.ui.tableWidgetDefect
    #     df = self._table_fill_df
    #     start = self._table_fill_row
    #     end   = min(start + self._table_fill_chunk, len(df))

    #     # Fill rows for this batch
    #     for r in range(start, end):
    #         row_vals = df.iloc[r].to_list()
    #         for c, v in enumerate(row_vals):
    #             if isinstance(v, float):
    #                 text = f"{v:.6g}"
    #             elif pd.isna(v):
    #                 text = ""
    #             else:
    #                 text = str(v)
    #             item = QTableWidgetItem(text)
    #             item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    #             # Make items non-editable
    #             item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

    #             tw.setItem(r, c, item)

    #     self._table_fill_row = end

    #     # update loader/progress
    #     if self.loading_dialog:
    #         done = end
    #         total = len(df)
    #         pct = int(100 * done / max(1, total))
    #         self.loading_dialog.update_progress(pct, f"Preparing table ({done}/{total})...")
    #         QtWidgets.QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)

    #     if end >= len(df):
    #         # finished
    #         tw.setUpdatesEnabled(True)
    #         tw.viewport().update()
    #         header = tw.horizontalHeader()
    #         header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    #         header.setStretchLastSection(False)
    #         self._is_filling_table = False

    #         # Apply styling after table is filled
    #         self._setup_table_styling()

    #         if self.loading_dialog and self._pending_close_loader:
    #             try:
    #                 self.loading_dialog.close()
    #             except Exception:
    #                 pass
    #             self.loading_dialog = None

    #         self.update_digsheet_button_state()
    #         QTimer.singleShot(0, self._refresh_table_scrollbars)
    #     else:
    #         # schedule next chunk (async → UI stays alive)
    #         QTimer.singleShot(0, self._fill_tablewidget_chunk)

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

            # ✅ make the dropdown mirror the final table headers (== desired cols)
            if not self._selected_columns:
                self._selected_columns = set(self._current_headers_for_filter()) | set(self.BACKEND_LOCKED_COLS)
            self.apply_column_filter()


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



    def _populate_defect_table_from_csv(self, df: pd.DataFrame):
        tw = self.ui.tableWidgetDefect
        tw.clearSelection()

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
            if src in df.columns:
                column_mapping[src] = dst

        num_rows = len(df)
        num_cols = len(header_indices)
        tw.setRowCount(num_rows)
        tw.setColumnCount(num_cols)
        tw.setHorizontalHeaderLabels(list(header_indices.keys()))

        for r, (_, row) in enumerate(df.iterrows()):
            for src, dst in column_mapping.items():
                if dst in header_indices:
                    c = header_indices[dst]
                    v = row[src]
                    if isinstance(v, float):
                        v = f"{v:.2f}"
                    item = QTableWidgetItem(str(v))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                    # Make items non-editable
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

                    tw.setItem(r, c, item)

        # Apply styling
        self._setup_table_styling()
        self.update_digsheet_button_state()

        # ✅ keep the dropdown in sync with the visible table headers
        if not self._selected_columns:
            self._selected_columns = set(self._current_headers_for_filter()) | set(self.BACKEND_LOCKED_COLS)
        self.apply_column_filter()








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

        # Get current tab name
        tab_text = self.ui.tabWidgetM.tabText(index).strip()
                # Fix: Switch the upper frame content correctly
        if hasattr(self, "top_stack"):
            if tab_text.lower() == "heatmap":
                # show the dual-heatmaps page
                self.top_stack.setCurrentWidget(self.dual_heatmaps_page)
            else:
                # show the single-chart page (for LineChart, 3D Graph, etc.)
                self.top_stack.setCurrentWidget(self.single_chart_page)

        # Always show table for LineChart and 3D Graph tabs
        if tab_text in {"LineChart", "Line Chart", "Line Plot", "3D Graph", "3D"}:
            self.bottom_stack.show()
            # Disable the toggle button for non-Heatmap tabs
            if hasattr(self, 'btnToggleTable'):
                self.btnToggleTable.setEnabled(False)
            if hasattr(self, "btnToggleHmLayout"):
                self.btnToggleHmLayout.setEnabled(False)
        # For Heatmap, respect the toggle flag
        elif tab_text == "Heatmap":
            if getattr(self, '_table_hidden', False):
                self.bottom_stack.hide()
            else:
                self.bottom_stack.show()
            # Enable the toggle button for Heatmap tab
            if hasattr(self, 'btnToggleTable'):
                self.btnToggleTable.setEnabled(True)
            if hasattr(self, "btnToggleHmLayout"):
                self.btnToggleHmLayout.setEnabled(True)
            QTimer.singleShot(100, lambda: self._reset_splitter_ratio(0.45))

        self.tab_switcher2()
        self.update_digsheet_button_state()

    def _set_top_mode(self, mode: str):
        """mode: 'dual' for heatmaps, 'single' for line/3D"""
        mode = mode.lower()
        if mode == "dual":
            # show the dual heatmaps page on top
            self.top_stack.setCurrentWidget(self.dual_heatmaps_page)
            self.main_top_scrollbar.hide()
        else:
            # show the single chart page on top
            self.top_stack.setCurrentWidget(self.single_chart_page)
            self.main_top_scrollbar.show()

        # optional: blank out views that aren't visible so you never see stale content
        if self.top_stack.currentWidget() is self.single_chart_page:
            # blank dual views
            try:
                self.web_view_left.setHtml("<html></html>")
                self.web_view_right.setHtml("<html></html>")
            except Exception:
                pass
        else:
            # blank single view
            try:
                self.web_view.setHtml("<html></html>")
            except Exception:
                pass

    def tab_switcher2(self, *_):
        if not self.project_is_open:
            self._show_watermark()
            return
        try:
            tab = self.ui.tabWidgetM.tabText(self.ui.tabWidgetM.currentIndex())
            # if tab == "Heatmap":
            #     if self.hmap:
            #         self._load_scrollable_chart(self.web_view, self.hmap, min_w=2200, min_h=1400)
            #     else:
            #         self.web_view.setUrl(QUrl())
            #     self.bottom_stack.setCurrentIndex(0)
            #     self.web_view2.setUrl(QUrl())
            #     # Setup scrollbar for heatmap
            #     QTimer.singleShot(100, self._arm_main_topbar)
            # if tab == "Heatmap":
            #     # Set dual mode layout
            #     self._set_top_mode("dual")

            #     # Load both heatmaps into the splitter
            #     if self.hhmap:
            #         self._load_scrollable_chart(self.web_view_left, self.hhmap, min_w=2200, min_h=1400)
            #     else:
            #         self.web_view_left.setUrl(QUrl())

            #     if self.phmap:
            #         self._load_scrollable_chart(self.web_view_right, self.phmap, min_w=2200, min_h=1400)
            #     else:
            #         self.web_view_right.setUrl(QUrl())

            #     # Apply the current layout mode (horizontal or vertical)
            #     self.apply_heatmap_layout(self.hm_layout_mode)

            #     self.bottom_stack.setCurrentIndex(0)
            #     QTimer.singleShot(100, self._arm_main_topbar)
            if tab == "Heatmap":
                # Only proceed if UI is fully initialized
                if not hasattr(self, 'top_stack'):
                    print("Warning: top_stack not yet initialized, skipping heatmap view")
                    return

                # Set dual mode layout
                self._set_top_mode("dual")

                # Load both heatmaps into the splitter
                if self.hhmap and hasattr(self, 'web_view_left'):
                    self._load_scrollable_chart(self.web_view_left, self.hhmap, min_w=2200, min_h=1400)
                else:
                    if hasattr(self, 'web_view_left'):
                        self.web_view_left.setUrl(QUrl())

                if self.phmap and hasattr(self, 'web_view_right'):
                    self._load_scrollable_chart(self.web_view_right, self.phmap, min_w=2200, min_h=1400)
                else:
                    if hasattr(self, 'web_view_right'):
                        self.web_view_right.setUrl(QUrl())

                # Apply the current layout mode
                self._apply_heatmap_layout(self._hm_layout_mode)
                # --- 🔄 Synchronize zoom/pan between both heatmaps ---
                try:
                    if hasattr(self, "web_view_left") and hasattr(self, "web_view_right"):
                        self.web_view_left.relay.relayout_json.connect(
                            lambda payload: self._sync_heatmap_range(self.web_view_right, payload)
                        )
                        self.web_view_right.relay.relayout_json.connect(
                            lambda payload: self._sync_heatmap_range(self.web_view_left, payload)
                        )
                        print("✅ Heatmap synchronization connections established")
                except Exception as sync_err:
                    print(f"⚠️ Heatmap sync setup failed: {sync_err}")

                left_pixel_offset = 120     # your desired vertical pixel scroll offset for left heatmap
                right_pixel_offset = 120     # desired offset for right heatmap

                QTimer.singleShot(100, lambda: self.left_scroll.verticalScrollBar().setValue(left_pixel_offset))
                QTimer.singleShot(100, lambda: self.right_scroll.verticalScrollBar().setValue(right_pixel_offset))


                self.bottom_stack.setCurrentIndex(0)
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


    BACKEND_LOCKED_COLS = {"Empty"}  # for styling purpose this is takin extra ,DONT REMOVE IT FROM THE SET


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
            back_btn = QPushButton("Back")
            back_btn.setIcon(QIcon("ui/icons/arrow_left.svg"))  # replace with your arrow icon path
            back_btn.setIconSize(QSize(16, 16))
            back_btn.setCursor(Qt.CursorShape.PointingHandCursor)

            back_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1.5px solid #000000;
                    border-radius: 8px;
                    padding: 5px 14px;
                    font-size: 13px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #f2f2f2;
                }
                QPushButton:pressed {
                    background-color: #e0e0e0;
                }
                QPushButton:disabled {
                    background-color: #f9f9f9;
                    color: #aaaaaa;
                    border: 1.5px solid #cccccc;
                }
            """)
            back_btn.clicked.connect(self._close_graphs_view)
            title = QLabel("Graphs")
            title.setStyleSheet("font-weight: 600; font-size: 14pt;")
            header.addWidget(back_btn); header.addSpacing(12); header.addWidget(title); header.addStretch(1)
            v.addLayout(header)

            graphs_widget = graphs_ui.GraphApp(dataframe=self.pipe_tally, project_root=self.project_root)
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
            # Look for pipe tally files inside pipetally_main subfolder
        pipetally_dir = os.path.join(root, "pipetally_main")
        if not os.path.isdir(pipetally_dir):
            print(f"[Warning] pipetally_main directory not found in {root}")
            self.pipe_tally = None
            return False

        candidates = [
            os.path.join(pipetally_dir, "pipe_tally.xlsx"),
            os.path.join(pipetally_dir, "pipe_tally.csv"),
        ]

        # Also scan for any tally-related files in the pipetally_main directory
        for f in os.listdir(pipetally_dir):
            name = f.lower()
            if name.endswith((".xlsx", ".xls", ".csv")):
                candidates.append(os.path.join(pipetally_dir, f))
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
        if not self.project_is_open:
            if self._ui_ready:
                self._project_required_popup()
            return
        try:
            # First check if a project is open
            if not self.project_is_open or not self.project_root:
                QMessageBox.warning(
                    self,
                    "No Project Open",
                    "Please open a project first to load KML files from the project folder."
                )
                return

            # Search for KML files in the project folder
            kml_files = []
            project_path = Path(self.project_root)

            # Search for KML files in project root and subdirectories
            kml_patterns = ["*.kml", "*.KML"]
            for pattern in kml_patterns:
                kml_files.extend(project_path.glob(pattern))
                kml_files.extend(project_path.glob(f"**/{pattern}"))  # Search subdirectories too

            # Remove duplicates and convert to strings
            kml_files = list(set(str(f) for f in kml_files))

            if not kml_files:
                QMessageBox.information(
                    self,
                    "No KML Files Found",
                    f"No KML files were found in the project folder:\n{self.project_root}\n\n"
                    "Please ensure your KML files are placed in the project directory."
                )
                return

            # If multiple KML files found, let user choose
            kml_path = None
            if len(kml_files) == 1:
                kml_path = kml_files[0]
            else:
                # Show selection dialog for multiple KML files
                file_names = [os.path.basename(f) for f in kml_files]
                selected_file, ok = QInputDialog.getItem(
                    self,
                    "Select KML File",
                    f"Found {len(kml_files)} KML files. Please select one to open:",
                    file_names,
                    0,
                    False
                )
                if ok and selected_file:
                    # Find the full path for the selected file
                    kml_path = next((f for f in kml_files if os.path.basename(f) == selected_file), None)

            if not kml_path:
                return

            # Determine Google Earth Pro path based on platform
            if sys.platform == "win32":
                earth_path = r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe"
            elif sys.platform == "darwin":
                earth_path = "/Applications/Google Earth Pro.app/Contents/MacOS/Google Earth Pro"
            else:
                earth_path = "/usr/bin/google-earth-pro"

            # Check if Google Earth Pro is installed
            if not os.path.exists(earth_path):
                # Show installation message
                reply = QMessageBox.question(
                    self,
                    "Google Earth Pro Not Found",
                    "Google Earth Pro is not installed on your system.\n\n"
                    "Would you like to download and install it?\n\n"
                    "Click 'Yes' to open the download page, or 'No' to cancel.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Open download page in default browser
                    import webbrowser
                    webbrowser.open("https://www.google.com/earth/versions/#earth-pro")
                return

            # Launch Google Earth Pro with the selected KML file
            try:
                subprocess.Popen([earth_path, kml_path])
                # QMessageBox.information(
                #     self,
                #     "Success",
                #     f"Google Earth Pro has been launched with:\n{os.path.basename(kml_path)}"
                # )
            except Exception as launch_error:
                QMessageBox.critical(
                    self,
                    "Launch Error",
                    f"Failed to launch Google Earth Pro with the KML file:\n{str(launch_error)}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An unexpected error occurred while searching for KML files:\n{str(e)}"
            )





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
        """Open Pipeline Highlights embedded in the main window"""
        try:
            # Check if pipe_tally is loaded
            if not hasattr(self, 'pipe_tally') or not isinstance(self.pipe_tally,
                                                                 pd.DataFrame) or self.pipe_tally.empty:
                QMessageBox.warning(
                    self,
                    "No Pipe Tally Data",
                    "Please load a project with pipe tally data first.\n\n"
                    "Steps to load data:\n"
                    "1. Go to File → Create Project\n"
                    "2. Select a folder containing pipe tally files\n"
                    "3. Wait for the data to load\n"
                    "4. Try opening Pipe Highlights again"
                )
                return

            # Check if Pipeline Highlights is already open
            if hasattr(self, '_central_pipeline') and self.centralWidget() is self._central_pipeline:
                return  # Already showing Pipeline Highlights

            # Save the original central widget
            if not hasattr(self, '_central_original') or self._central_original is None:
                self._central_original = self.centralWidget()

            print(f"🔍 Opening Pipeline Highlights with {len(self.pipe_tally)} rows of data")
            print(f"📊 Available columns: {list(self.pipe_tally.columns)}")

            # Import the embedded version
            from pages.Pipe_Highlights_Embedded import PipeHighlightEmbedded

            # Create container widget
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            # Header with back button
            header_layout = QHBoxLayout()
            back_btn = QPushButton("Back")
            back_btn.setIcon(QIcon("ui/icons/arrow_left.svg"))  # replace with your arrow icon path
            back_btn.setIconSize(QSize(16, 16))
            back_btn.setCursor(Qt.CursorShape.PointingHandCursor)

            back_btn.setStyleSheet("""
                   QPushButton {
                       background-color: #ffffff;
                       color: #000000;
                       border: 1.5px solid #000000;
                       border-radius: 8px;
                       padding: 5px 14px;
                       font-size: 13px;
                       font-weight: 500;
                   }
                   QPushButton:hover {
                       background-color: #f2f2f2;
                   }
                   QPushButton:pressed {
                       background-color: #e0e0e0;
                   }
                   QPushButton:disabled {
                       background-color: #f9f9f9;
                       color: #aaaaaa;
                       border: 1.5px solid #cccccc;
                   }
               """)
            back_btn.clicked.connect(self._close_pipeline_view)

            title_label = QLabel("")
            title_label.setStyleSheet("font-weight: 600; font-size: 16pt; color: #2c3e50;")

            header_layout.addWidget(back_btn)
            header_layout.addSpacing(20)
            header_layout.addWidget(title_label)
            header_layout.addStretch(1)

            layout.addLayout(header_layout)

            # Create and add the Pipeline Highlights widget
            self._pipeline_widget = PipeHighlightEmbedded(parent=container, pipe_tally_df=self.pipe_tally,
                                                          project_root=self.project_root)
            layout.addWidget(self._pipeline_widget, stretch=1)

            # Store reference and switch central widget
            self._central_pipeline = container

            # Switch to Pipeline Highlights view
            if self._central_original is not None and self._central_original.parent() is self:
                self.takeCentralWidget()
            self.setCentralWidget(container)

            print("✅ Pipeline Highlights opened successfully in embedded mode")

        except ImportError as e:
            self.open_Error(
                f"Could not import Pipeline Highlights module:\n{e}\n\nPlease check if the Pipe_Highlights_Embedded.py file exists in the pages folder.")
        except Exception as e:
            self.open_Error(f"Error running Pipeline Highlights:\n{e}")
            # Restore original view on error
            try:
                if hasattr(self, '_central_original') and self._central_original is not None:
                    if self.centralWidget() is not self._central_original:
                        self.setCentralWidget(self._central_original)
            except Exception:
                pass

    def _close_pipeline_view(self):
        """Close Pipeline Highlights and return to main view"""
        try:
            if self.centralWidget() is getattr(self, '_central_original', None):
                return  # Already showing original view

            # Take current widget and delete it
            pipeline_central = self.takeCentralWidget()
            if pipeline_central is not None:
                pipeline_central.deleteLater()

            # Restore original central widget
            if hasattr(self, '_central_original') and self._central_original is not None:
                if self._central_original.parent() is not self:
                    self._central_original.setParent(self)
                self.setCentralWidget(self._central_original)

            # Clean up references
            if hasattr(self, '_pipeline_widget'):
                self._pipeline_widget = None
            if hasattr(self, '_central_pipeline'):
                self._central_pipeline = None

            print("✅ Returned to main view from Pipeline Highlights")

        except Exception as e:
            print(f"⚠️ Error closing Pipeline Highlights view: {e}")

    def open_PipeScheme(self):
        try:
            import subprocess, sys, os
            pipeline_path = os.path.join("pipeline_schema", "pipeline_schema.py")
            subprocess.Popen([sys.executable, pipeline_path, self.project_root])
        except Exception as e:
            self.open_Error(f"Error running Pipeline Schema:\n{e}")

    # def open_PipeScheme(self):
    #     try:
    #         import subprocess, sys, os
    #         pipeline_path = os.path.join("pipeline_schema", "pipeline_schema.py")
    #         subprocess.Popen([sys.executable, pipeline_path])
    #     except Exception as e:
    #         self.open_Error(f"Error running Pipeline Schema:\n{e}")

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

    # def open_ERF(self):
    #     self.erf = ERF()
    #     def update_result():
    #         OD = self.erf.doubleSpinBox.value()
    #         WT = self.erf.doubleSpinBox_3.value()
    #         SMYS = self.erf.doubleSpinBox_2.value()
    #         MAOP = self.erf.doubleSpinBox_4.value()
    #         SF = self.erf.doubleSpinBox_5.value()
    #         Axial_L = self.erf.doubleSpinBox_8.value()
    #         Depth_P = self.erf.doubleSpinBox_9.value()
    #         if OD == 0 or WT == 0 or SF == 0:
    #             self.erf.lineEdit_2.setText("-"); self.erf.lineEdit_3.setText("-"); return
    #         flow_stress = 1.1 * SMYS
    #         z_factor = (Axial_L ** 2) / (OD * WT)
    #         M = (1 + 0.8 * z_factor) ** 0.5
    #         y = 1 - 2/3 * Depth_P / WT
    #         z = 1 - 2/3 * Depth_P / WT / M
    #         k = y / z
    #         S = (flow_stress * k) if z_factor <= 20 else (flow_stress * (1 - Depth_P / WT))
    #         EFP = (2 * S * WT) / OD
    #         PSafe = EFP / SF if SF else 0
    #         if PSafe == 0:
    #             self.erf.lineEdit_2.setText("-"); self.erf.lineEdit_3.setText("-"); return
    #         ERFv = MAOP / PSafe
    #         self.erf.lineEdit_2.setText(f"{ERFv:.2f}")
    #         self.erf.lineEdit_3.setText(f"{PSafe:.2f}")
    #         import numpy as np
    #         def calc_B(d_over_t):
    #             if d_over_t >= 0.175:
    #                 B = np.sqrt(((d_over_t / (1.1 * d_over_t - 0.15)) ** 2) - 1)
    #                 return B if B <= 4 else 4
    #             return 4
    #         xs = np.linspace(0, 1, 100)
    #         ys = [calc_B(x) for x in xs]
    #         Xc = Axial_L / 300; Yc = Depth_P / 20
    #         color = 'green' if Yc < calc_B(Xc) else 'red'
    #         fig = go.Figure()
    #         fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='ASME B31G'))
    #         fig.add_trace(go.Scatter(x=[Xc], y=[Yc], mode='markers',
    #                                  marker=dict(color=color, size=10),
    #                                  name='Defect'))
    #         fig.update_layout(xaxis_title='Axial Length (mm)', yaxis_title='Peak Depth', height=450, width=1000)
    #         fp = resource_path('backend/files/ASME.html'); fig.write_html(fp)
    #         self.erf.web_viewERF.setUrl(QUrl.fromLocalFile(fp))
    #     for w in (self.erf.doubleSpinBox, self.erf.doubleSpinBox_3, self.erf.doubleSpinBox_2,
    #               self.erf.doubleSpinBox_4, self.erf.doubleSpinBox_5,
    #               self.erf.doubleSpinBox_8, self.erf.doubleSpinBox_9):
    #         w.valueChanged.connect(update_result)
    #     update_result()
    #     self.erf.show()

    def open_ERF(self):
        import threading

        # Inner function - no self parameter
        def run_erf():
            erf_app = ERF(self.project_root)
            erf_app.run()

        # Start ERF calculator in a background thread
        threading.Thread(target=run_erf, daemon=True).start()

    def _back_from_pipe_locator(self):
        # Go back only if previous widget exists and is valid
        if hasattr(self, "_pipe_locator_prev_widget") and self._pipe_locator_prev_widget:
            self.top_stack.setCurrentWidget(self._pipe_locator_prev_widget)
        else:
            # fallback: go to first available widget
            if self.top_stack.count() > 0:
                self.top_stack.setCurrentIndex(0)

    def _close_pipe_locator(self):
        if hasattr(self, "_pipe_locator_view"):
            self.top_stack.removeWidget(self._pipe_locator_view)
            self._pipe_locator_view.deleteLater()
            del self._pipe_locator_view

            # Go back to previous view safely
            if hasattr(self, "_pipe_locator_prev_widget") and self._pipe_locator_prev_widget:
                self.top_stack.setCurrentWidget(self._pipe_locator_prev_widget)



    # def _close_pipe_locator(self):
    #
    #     if hasattr(self, "_pipe_locator_view") and self._pipe_locator_view:
    #         self._pipe_locator_view.setParent(None)
    #         self._pipe_locator_view.deleteLater()
    #         self._pipe_locator_view = None
    #
    #     # 🔥 RE-ENABLE DIGSHEET
    #     if hasattr(self, "actionDigsheet"):
    #         self.actionDigsheet.setEnabled(True)

    # def open_pipe_locator(self):
    #     if self.pipe_tally is None or self.pipe_tally.empty:
    #         QMessageBox.warning(self, "Pipe Locator", "Pipe tally not loaded")
    #         return
    #
    #     # create once
    #     if not hasattr(self, "_pipe_locator_view"):
    #         self._pipe_locator_view = PipeLocatorWidget(self.pipe_tally, self)
    #
    #         # 🔥 ADD TO TOP STACK
    #         self.top_stack.addWidget(self._pipe_locator_view)
    #
    #     # 🔥 SHOW PIPE LOCATOR
    #     self.top_stack.setCurrentWidget(self._pipe_locator_view)

    # def open_pipe_locator(self):
    #     if self.pipe_tally is None or self.pipe_tally.empty:
    #         QMessageBox.warning(self, "Pipe Locator", "Pipe tally not loaded")
    #         return
    #
    #     # ✅ Save previous view ONLY if we are not already in PipeLocator
    #     current = self.top_stack.currentWidget()
    #
    #     if not isinstance(current, PipeLocatorWidget):
    #         self._pipe_locator_prev_widget = current
    #
    #     # Create Pipe Locator only once
    #     if not hasattr(self, "_pipe_locator_view"):
    #         self._pipe_locator_view = PipeLocatorWidget(self.pipe_tally, self)
    #         self.top_stack.addWidget(self._pipe_locator_view)
    #
    #         # 🔙 Back / ✕ / ESC → go back properly
    #         self._pipe_locator_view.backRequested.connect(
    #             self._back_from_pipe_locator
    #         )
    #
    #     # Show Pipe Locator
    #     self.top_stack.setCurrentWidget(self._pipe_locator_view)

    def open_pipe_locator(self):

        if getattr(self, "_pipe_locator_dialog", None):
            dlg = self._pipe_locator_dialog
            dlg.raise_()
            dlg.activateWindow()
            return

        pipe_tally = self.pipe_tally if isinstance(self.pipe_tally, pd.DataFrame) else pd.DataFrame()

        dlg = PipeLocatorDialog(pipe_tally, self)
        self._pipe_locator_dialog = dlg

        # hide top controls
        if hasattr(self.ui, "widgetControls"):
            self.ui.widgetControls.hide()

        def cleanup():
            self._pipe_locator_dialog = None
            if hasattr(self.ui, "widgetControls"):
                self.ui.widgetControls.show()

        dlg.finished.connect(cleanup)

        # 🔥 SHOW ORDER MATTERS
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _clear_pipe_locator_ref(self):
        self._pipe_locator_dialog = None

    def open_Final_Report(self):
        # Check if a project is open
        if not self.project_is_open or not self.project_root:
            QMessageBox.warning(
                self,
                "No Project Open",
                "Please create/open a project first to access the Final Report."
            )
            return

        # Look for Final_Report.pdf in the report folder within project root
        report_dir = os.path.join(self.project_root, "report")
        final_report_path = os.path.join(report_dir, "FR.pdf")

        if not os.path.exists(final_report_path):
            QMessageBox.warning(
                self,
                "Final Report Not Found",
                f"Could not find 'Final_Report.pdf' in the report directory:\n{report_dir}"
            )
            return

        try:
            os.startfile(final_report_path)
        except Exception as e:
            self.open_Error(f"Failed to open Final Report:\n{e}")


    def open_Preliminary_Report(self):
        # Check if a project is open
        if not self.project_is_open or not self.project_root:
            QMessageBox.warning(
                self,
                "No Project Open",
                "Please create/open a project first to access the Preliminary Report.\n\n"
                "Steps:\n"
                "1. Go to File → Create Project\n"
                "2. Select a project folder\n"
                "3. Then try accessing Preliminary Report again"
            )
            return

        # Look for PR.pdf in the report folder within project root
        report_dir = os.path.join(self.project_root, "report")
        prelim_report_path = os.path.join(report_dir, "PR.pdf")

        if not os.path.exists(prelim_report_path):
            QMessageBox.warning(
                self,
                "Preliminary Report Not Found",
                f"Could not find 'PR.pdf' in the report directory:\n{report_dir}\n\n"
                "Please ensure the report folder exists in your project and contains PR.pdf"
            )
            return

        # Open the preliminary report
        try:
            os.startfile(prelim_report_path)
        except Exception as e:
            self.open_Error(f"Failed to open Preliminary Report:\n{e}")


    def open_pipe_tally(self):
        # Check if a project is open
        if not self.project_is_open or not self.project_root:
            QMessageBox.warning(
                self,
                "No Project Open",
                "Please create/open a project first to access the pipe tally file.\n\n"
                "Steps:\n"
                "1. Go to File → Create Project\n"
                "2. Select a project folder\n"
                "3. Then try accessing Pipe Tally again"
            )
            return

        if not hasattr(self, 'pipe_tally') or self.pipe_tally is None:
            QMessageBox.warning(
                self,
                "No Pipe Tally Loaded",
                "No pipe tally data is currently loaded from this project."
            )
            return

        # Search for pipe tally files ONLY in the project root directory (not subdirectories)
        pipe_tally_files = []
        project_path = Path(self.project_root)

        # Define pattern to match pipe tally related files (case-insensitive)
        # Matches: pipetally, pipe_tally, tally_pipe, pipe-tally, etc.
        import re
        tally_pattern = re.compile(r'.*(pipe.*tally|tally.*pipe|pipetally|pipe_tally|pipe-tally).*\.(xlsx?|csv)$', re.IGNORECASE)

        # Search ONLY in project root (not subdirectories)
       # Search ONLY in pipetally_main subfolder
        pipetally_main_path = project_path / "pipetally_main"
        if not pipetally_main_path.is_dir():
            QMessageBox.warning(
                self,
                "Pipetally Directory Not Found",
                f"Could not find 'pipetally_main' folder in the project directory:\n{self.project_root}\n\n"
                "Please ensure the pipetally_main folder exists in your project."
            )
            return

        try:
            for file_path in pipetally_main_path.iterdir():  # Only direct children of pipetally_main
                if file_path.is_file() and tally_pattern.match(file_path.name):
                    pipe_tally_files.append(str(file_path))

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error searching for pipe tally files:\n{e}"
            )
            return

        if not pipe_tally_files:
            QMessageBox.warning(
                self,
                "Pipe Tally File Not Found",
                f"Could not find any pipe tally files in the project root directory:\n{self.project_root}\n\n"
                "Looking for files containing: 'pipetally', 'pipe_tally', 'tally_pipe', etc.\n"
                "Note: Only searching in the root folder, not inside pipe subdirectories.\n\n"
                "The pipe tally data is loaded in memory, but the source file could not be located."
            )
            return

        # If multiple files found, let user choose
        pipe_tally_file = None
        if len(pipe_tally_files) == 1:
            pipe_tally_file = pipe_tally_files[0]
        else:
            # Show selection dialog for multiple pipe tally files
            file_names = [os.path.basename(f) for f in pipe_tally_files]
            selected_file, ok = QInputDialog.getItem(
                self,
                "Select Pipe Tally File",
                f"Found {len(pipe_tally_files)} pipe tally files in the root directory. Please select one to open:",
                file_names,
                0,
                False
            )
            if ok and selected_file:
                # Find the full path for the selected file
                pipe_tally_file = next((f for f in pipe_tally_files if os.path.basename(f) == selected_file), None)

        # Open the selected file
        if pipe_tally_file:
            try:
                os.startfile(pipe_tally_file)
            except Exception as e:
                self.open_Error(f"Failed to open pipe tally file:\n{e}")
        else:
            QMessageBox.information(self, "No Selection", "No file was selected.")


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
            "toolButtonHeatmap", "toolButtonLine", "toolButton3D", "toolButtonXYZ",
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
        # if tw.rowCount() == 0 or tw.columnCount() == 0:
        #     return False
        if not tw.isVisible() or tw.rowCount() == 0 or tw.columnCount() == 0:
            return False

        # ✅ Check if "no defects" message is showing
        if hasattr(self, '_no_defects_container') and self._no_defects_container and self._no_defects_container.isVisible():
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
        if not self.project_is_open:
            self.btnDigsheetAbs.setEnabled(False)
            self.btnDigsheetAbs.setCursor(Qt.CursorShape.ForbiddenCursor)
            self.btnDigsheetAbs.setToolTip("Create a project first to enable Digsheet generation.")
            return
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

            # if getattr(sys, "frozen", False):
            #     subprocess.Popen([sys.executable, "--run-digsheet-abs", tally_pkl, str(abs_text)])
            # else:
            #     subprocess.Popen([sys.executable, dig_py_abs, tally_pkl, str(abs_text)])
            if getattr(sys, "frozen", False):
                subprocess.Popen([
                    sys.executable,
                    "--run-digsheet-abs",
                    tally_pkl,
                    str(abs_text),
                    self.project_root  # ✅ Pass project root
                ])
            else:
                subprocess.Popen([
                    sys.executable,
                    dig_py_abs,
                    tally_pkl,
                    str(abs_text),
                    self.project_root  # ✅ Pass project root
                ])

        except Exception as e:
            self.open_Error(f"Error opening ABS-distance digsheet:\n{e}")

    def _update_generate_actions(self):
        """Update Generate menu buttons based on project and data status"""
        # Check if pipe tally data is available
        has_pipe_tally = isinstance(self.pipe_tally, pd.DataFrame) and not self.pipe_tally.empty

        # Check if preliminary report exists
        has_prelim_report = False
        if self.project_is_open and self.project_root:
            report_dir = os.path.join(self.project_root, "report")
            prelim_report_path = os.path.join(report_dir, "PR.pdf")
            has_prelim_report = os.path.exists(prelim_report_path)

        # Check if final report exists
        has_final_report = False
        if self.project_is_open and self.project_root:
            report_dir = os.path.join(self.project_root, "report")
            final_report_path = os.path.join(report_dir, "FR.pdf")
            has_final_report = os.path.exists(final_report_path)

        # Update BOTH Final Report actions ✅
        if hasattr(self.ui, 'action_Final_Report'):
            self.ui.action_Final_Report.setEnabled(self.project_is_open and has_final_report)

        if hasattr(self.ui, 'Final_Report'):  # ← Add this block
            self.ui.Final_Report.setEnabled(self.project_is_open and has_final_report)

        # Update Pipe Tally button/action
        if hasattr(self.ui, 'action__pipetally'):
            self.ui.action__pipetally.setEnabled(self.project_is_open and has_pipe_tally)

        # Update Preliminary Report action
        if hasattr(self.ui, 'action_Preliminary_Report'):
            self.ui.action_Preliminary_Report.setEnabled(self.project_is_open and has_prelim_report)

        # Update Digsheet actions (both standard and ABS-based)
        if hasattr(self.ui, 'actionStandard'):  # Standard digsheet
            self.ui.actionStandard.setEnabled(self.project_is_open and has_pipe_tally)




    def _force_heatmap_start(self):
        """Ensure middle view opens on Heatmap before the next load."""
        self._last_allowed_tab_index = 0
        self._reverting_tab = False
        tw = getattr(self.ui, "tabWidgetM", None)
        if tw is not None:
            tw.blockSignals(True)
            tw.setCurrentIndex(0)
            tw.blockSignals(False)
        if hasattr(self, "tabSwitcherDropdown"):
            self.tabSwitcherDropdown.blockSignals(True)
            self.tabSwitcherDropdown.setCurrentIndex(0)
            self.tabSwitcherDropdown.blockSignals(False)


    def close_project(self):
        try:
            # 1) Stop any secondary views / background loaders
            self._close_graphs_view()
            try:
                if getattr(self, "loader_worker", None) and self.loader_worker.isRunning():
                    self.loader_worker.requestInterruption()
                    self.loader_worker.quit()
                    self.loader_worker.wait(1500)
            except Exception:
                pass
            self.loader_worker = None

            # 2) Block tab/dropdown signals while we reset widgets
            tw = getattr(self.ui, "tabWidgetM", None)
            if tw is not None:
                tw.blockSignals(True)
            if hasattr(self, "tabSwitcherDropdown"):
                self.tabSwitcherDropdown.blockSignals(True)

            # 3) Mark project closed + clear file lists/state
            self.project_is_open = False
            self.project_root = None
            self.pkl_files = []
            self.pipe_tally = None
            self.curr_data = None
            self.header_list = []
            self.hmap = self.hmap_r = self.lplot = self.lplot_r = self.pipe3d = self.heatmap_box = None
            self.hhmap = self.phmap = None
            self._selected_columns = set()

            # 4) Reset the "allowed tab" & guard flags so next project starts on Heatmap
            self._reverting_tab = False
            self._last_allowed_tab_index = 0  # 0 == Heatmap tab (middle stack)

            # 5) Reset combo + top controls
            cb = self.ui.comboBoxPipe
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("-Pipe-")
            cb.blockSignals(False)

            if hasattr(self, "btnLoadPipe"):    self.btnLoadPipe.setEnabled(False)
            if hasattr(self, "btnDigsheetAbs"): self.btnDigsheetAbs.setEnabled(False)

            # 6) Heatmap-specific UI back to defaults
            self._hm_layout_mode = "vertical"            # default = stacked
            try:
                self._apply_heatmap_layout("vertical")
            except Exception:
                pass

            self._table_hidden = True                    # default label = "Show Table"
            if hasattr(self, "btnToggleTable"):
                self.btnToggleTable.setEnabled(False)
                self.btnToggleTable.setText("Show Table")
            if hasattr(self, "btnToggleHmLayout"):
                self.btnToggleHmLayout.setEnabled(False)
                # optional: set text according to your toggling semantics
                # self.btnToggleHmLayout.setText("Side-by-side")

            # 7) Tables and models
            try:
                self.model.clear()
            except Exception:
                pass
            if hasattr(self.ui, "tableWidgetDefect"):
                self.ui.tableWidgetDefect.clear()
                self.ui.tableWidgetDefect.hide()
            if hasattr(self, "table_scrollbar") and self.table_scrollbar:
                self.table_scrollbar.hide()

            # 8) Middle tab + dropdown back to Heatmap
            if tw is not None:
                try:
                    tw.setCurrentIndex(0)  # Heatmap
                except Exception:
                    pass
            if hasattr(self, "tabSwitcherDropdown"):
                self.tabSwitcherDropdown.setCurrentIndex(0)

            # 9) Web views / stacks / overlays
            try:
                self.web_view.setUrl(QUrl("about:blank"))
                self.web_view2.setUrl(QUrl("about:blank"))
            except Exception:
                pass
            if hasattr(self, "bottom_stack"):
                self.bottom_stack.setCurrentIndex(0)  # hide table pane
            self._show_watermark()
            self._toggle_plot_ui(False)

            if hasattr(self, "_select_pipe_container") and self._select_pipe_container:
                self._select_pipe_container.hide()
            if hasattr(self, "_no_defects_container") and self._no_defects_container:
                self._no_defects_container.hide()
            if hasattr(self, "_create_proj_container") and self._create_proj_container:
                self._create_proj_container.show()
            if hasattr(self, "btnOpenFilterDlg"):
                self.btnOpenFilterDlg.setEnabled(False)
            if hasattr(self, "tabSwitcherDropdown"):
                self.tabSwitcherDropdown.setEnabled(False)

            self._force_full_start_state()

            self._update_project_actions()
            QMessageBox.information(self, "Project Closed", "The project has been successfully closed.")

        except Exception as e:
            self.open_Error(e)
        finally:
            # Re-enable signals
            if tw is not None:
                tw.blockSignals(False)
            if hasattr(self, "tabSwitcherDropdown"):
                self.tabSwitcherDropdown.blockSignals(False)
        self.ui.action_Pipe_Sch.setEnabled(False)

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
            if not self.project_is_open:
                QMessageBox.warning(
                    self,
                    "No Project Open",
                    "Please create/open a project first to generate digsheets."
                )
                return
            if not isinstance(self.pipe_tally, pd.DataFrame):
                QMessageBox.warning(self, "No Pipe Tally", "Load a pipe tally first.")
                return

            tally_pkl = _dump_tally_to_temp(self.pipe_tally)
            dig_py = resource_path(os.path.join("dig", "dig_sheet.py"))
            if not os.path.exists(dig_py):
                QMessageBox.critical(self, "Script not found", f"Missing: {dig_py}")
                return

            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable, "--run-digsheet", tally_pkl, self.project_root])
            else:
                subprocess.Popen([sys.executable, dig_py, tally_pkl, self.project_root])
        except Exception as e:
            self.open_Error(f"An error occurred: {e}")

    def _on_dropdown_tab_changed(self, index: int):
        """Handle tab change from dropdown"""
        try:
            # Block signals to prevent infinite loop
            self.ui.tabWidgetM.blockSignals(True)
            self.mid_tabbar.blockSignals(True)

            # Set the tab index
            self.ui.tabWidgetM.setCurrentIndex(index)
            self.mid_tabbar.setCurrentIndex(index)

            # Unblock signals
            self.ui.tabWidgetM.blockSignals(False)
            self.mid_tabbar.blockSignals(False)

            # Trigger the actual tab change logic
            self.onmiddletabchanged(index)

        except Exception as e:
            print(f"Error in dropdown tab change: {e}")

    def _sync_dropdown_with_tabs(self, index: int):
        """Sync dropdown when tab changes from other sources"""
        try:
            # Block signals to prevent infinite loop
            self.tabSwitcherDropdown.blockSignals(True)

            # Update dropdown selection
            self.tabSwitcherDropdown.setCurrentIndex(index)

            # Unblock signals
            self.tabSwitcherDropdown.blockSignals(False)

        except Exception as e:
            print(f"Error syncing dropdown: {e}")



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

    from PyQt6.QtGui import QPixmap

    def _setup_create_project_label(self):
        """Create a centered overlay for 'Create Project' message"""
        central = self.centralWidget()
        self._create_proj_container = QWidget(central)
        self._create_proj_container.setGeometry(central.rect())
        self._create_proj_container.setStyleSheet("""
            background-color: rgba(245, 247, 250, 200);
        """)

        layout = QVBoxLayout(self._create_proj_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Main card
        card = QFrame()
        card.setFixedWidth(420)
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 14px;
                border: 1px solid #e0e0e0;
                padding: 30px 20px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(20)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Proper icon (no cropping)
        icon_label = QLabel()
        pixmap = QPixmap("icons/folder.png")  # ✅ use your own folder.png here
        if not pixmap.isNull():
            pixmap = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            icon_label.setPixmap(pixmap)
        else:
            icon_label.setText("📁")  # fallback emoji
            icon_label.setStyleSheet("font-size: 48px;")

        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(icon_label)

        # Title
        title = QLabel("Create the Project")
        title.setStyleSheet("""
            font-size: 20pt;
            font-weight: 600;
            color: #2c3e50;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #e0e0e0; margin: 8px 0;")
        card_layout.addWidget(divider)

        # Subtitle (fixed clipping issue)
        subtitle = QLabel("Go to <b>File → Create Project</b> in the menu bar")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        subtitle.setStyleSheet("""
            font-size: 12pt;
            color: #555;
        """)
        card_layout.addWidget(subtitle)

        layout.addWidget(card)
        self._create_proj_container.hide()

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

    def ondropdowntabchanged(self, index: int):
        """Handle tab changes from dropdown switcher"""
        if index >= 0:
            self.ui.tabWidgetM.blockSignals(True)
            self.mid_tabbar.blockSignals(True)

            self.ui.tabWidgetM.setCurrentIndex(index)
            self.mid_tabbar.setCurrentIndex(index)
            self.tabSwitcherDropdown.setCurrentIndex(index)

            self.ui.tabWidgetM.blockSignals(False)
            self.mid_tabbar.blockSignals(False)

            self._on_middle_tab_changed(index)

    def syncdropdownwithtabs(self, index: int):
        """Sync dropdown when tab changes from other sources"""
        try:
            if hasattr(self, 'tabSwitcherDropdown'):
                self.tabSwitcherDropdown.blockSignals(True)
                self.tabSwitcherDropdown.setCurrentIndex(index)
                self.tabSwitcherDropdown.blockSignals(False)
        except Exception as e:
            print(f"Error syncing dropdown: {e}")

    def toggletablevisibility(self):
        """Toggle table visibility in heatmap view only"""
        # Only work in Heatmap tab
        current_tab = self.ui.tabWidgetM.tabText(self.ui.tabWidgetM.currentIndex()).strip()
        if current_tab != "Heatmap":
            QMessageBox.information(self, "Heatmap Only",
                                "Table toggle only works in Heatmap view.")
            return

        # Toggle the flag
        self._table_hidden = not self._table_hidden

        # Hide or show the bottom section (table area)
        if self._table_hidden:
            self.bottom_stack.hide()
            self.btnToggleTable.setText("Show Table")
        else:
            self.bottom_stack.show()
            # self.ui.tableWidgetDefect.setMinimumHeight(250)
            # self.bottom_stack.setMinimumHeight(250)

            self.btnToggleTable.setText("Hide Table")

        print(f"Table visibility toggled: {'Hidden' if self._table_hidden else 'Shown'}")

    # def _apply_heatmap_layout(self, mode: str = None):
    #     """Apply horizontal (side-by-side) or vertical (stacked) layout for dual heatmaps"""
    #     if not hasattr(self, 'top_h_split'):
    #         return

    #     self.hm_layout_mode = mode

    #     # Change splitter orientation
    #     if mode == "horizontal":
    #         self.top_h_split.setOrientation(Qt.Orientation.Horizontal)
    #         self.btnToggleHmLayout.setText("Stack")
    #         # Apply 50-50 split
    #         total = self.top_h_split.width()
    #         left = int(total * self.hm_left_ratio)
    #         right = total - left
    #         self.top_h_split.setSizes([left, right])
    #     else:  # vertical
    #         self.top_h_split.setOrientation(Qt.Orientation.Vertical)
    #         self.btnToggleHmLayout.setText("Side-by-side")
    #         # Apply 50-50 split
    #         total = self.top_h_split.height()
    #         top = total // 2
    #         bottom = total - top
    #         self.top_h_split.setSizes([top, bottom])

    #     print(f"Heatmap layout changed to: {mode}")

    def _apply_heatmap_layout(self, mode: str = None):
        """Apply horizontal (side-by-side) or vertical (stacked) layout for dual heatmaps"""
        # Use provided mode or fall back to current mode
        if mode is None:
            mode = getattr(self, '_hm_layout_mode', 'horizontal')

        # Safety checks
        if not hasattr(self, 'top_hsplit'):
            print("Warning: top_hsplit not found, skipping layout change")
            return

        self._hm_layout_mode = mode

        # Change splitter orientation
        if mode == "horizontal":
            self.top_hsplit.setOrientation(Qt.Orientation.Horizontal)
            if hasattr(self, 'btnToggleHmLayout'):
                self.btnToggleHmLayout.setText("stack" if mode == "horizontal" else "side-by-side")
            # Apply 50-50 split
            total = self.top_hsplit.width()
            left = int(total * 0.38)
            right = total - left
            self.top_hsplit.setSizes([left, right])
        else:  # vertical
            self.top_hsplit.setOrientation(Qt.Orientation.Vertical)
            if hasattr(self, 'btnToggleHmLayout'):
                self.btnToggleHmLayout.setText("Side-by-side")
            # Apply 50-50 split
            total = self.top_hsplit.height()
            top = (total // 2) - 95
            bottom = total - top
            self.top_hsplit.setSizes([top, bottom])

        print(f"Heatmap layout changed to: {mode}")




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