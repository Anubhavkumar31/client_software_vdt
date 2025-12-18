import os
import sys
import tempfile
import uuid

from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QCursor
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget, QStackedWidget, QHBoxLayout, QSplitter, QTabBar, \
    QSplitterHandle, QFrame, QScrollBar
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QHeaderView, QInputDialog,
    QSpacerItem, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem,
    QStatusBar, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox,
    QDialog, QTextEdit, QPushButton, QSplitter, QStackedWidget,
    QTabBar, QFrame, QHBoxLayout as _QHBoxLayout, QSplitterHandle, QComboBox,
    QAbstractItemView, QAbstractScrollArea, QProgressBar
)
from PyQt6 import QtWidgets
from main_section_view.helpers_temp import _arm_topbar
from main_section_view.utils import _apply_heatmap_layout

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
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def _dump_tally_to_temp(df):
    import pickle
    p = os.path.join(tempfile.gettempdir(), f"pipe_tally_{uuid.uuid4().hex}.pkl")
    with open(p, "wb") as f: pickle.dump(df, f)
    return p

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

class MidBarSplitter(QSplitter):
    def __init__(self, parent=None, tabbar: Optional[QTabBar] = None):
        super().__init__(Qt.Orientation.Vertical, parent)
        self._tabbar = tabbar

    def createHandle(self):
        return MidBarHandle(self.orientation(), self, self._tabbar)


#ORIGINAL (FORMALLY named as "_build_splitter")
# def _build_main_section(self):
#     # ---------- tiny local helpers so this method is self-contained ----------
#     if not hasattr(self, "_hm_layout_mode"):
#         self._hm_layout_mode = "vertical"  # persisted layout mode
#
#     # ---------- TOP: build a stack (single view + dual heatmaps) ----------
#     self.main_web_page = QWidget()
#     main_web_layout = QVBoxLayout(self.main_web_page)
#     main_web_layout.setContentsMargins(0, 0, 0, 0)
#     main_web_layout.setSpacing(0)
#
#     # page 0: original single chart page (used by Line/3D)
#     self.single_chart_page = QWidget()
#     single_lay = QVBoxLayout(self.single_chart_page)
#     single_lay.setContentsMargins(0, 0, 0, 0)
#     single_lay.setSpacing(0)
#
#     self.main_web_scroll_area = QScrollArea()
#     self.main_web_scroll_area.setWidgetResizable(False)
#     self.main_web_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#     self.main_web_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#
#     self.web_view = QWebEngineView()
#     self.web_view.setFixedSize(2500, 650)
#     self.main_web_scroll_area.setWidget(self.web_view)
#     single_lay.addWidget(self.main_web_scroll_area)
#
#     # page 1: dual heatmaps page (NEW)
#     self.dual_heatmaps_page = QWidget()
#     dual_lay = QVBoxLayout(self.dual_heatmaps_page)
#     dual_lay.setContentsMargins(0, 0, 0, 0)
#     dual_lay.setSpacing(6)
#
#     # --- tiny toolbar with the toggle button + show/hide table ---
#     top_toolbar = QHBoxLayout()
#     top_toolbar.setContentsMargins(8, 6, 8, 4)
#     top_toolbar.setSpacing(8)
#
#     top_toolbar.addStretch(1)
#     dual_lay.addLayout(top_toolbar)
#
#     # --- the dual-heatmap splitter ---
#     self.top_hsplit = QSplitter(Qt.Orientation.Horizontal if self._hm_layout_mode == "horizontal"
#                                 else Qt.Orientation.Vertical)
#     self.top_hsplit.setChildrenCollapsible(False)
#     self.top_hsplit.setStretchFactor(0, 1)
#     self.top_hsplit.setStretchFactor(1, 1)
#     self.top_hsplit.setObjectName("TopHSplit")
#     self.top_hsplit.setStyleSheet("""
#         QSplitter#TopHSplit::handle {
#             background-color: #3a3a3a;     /* darker, more visible */
#             border: 1px solid #2a2a2a;     /* subtle edge so it stands out */
#         }
#         QSplitter#TopHSplit::handle:hover {
#             background-color: #4a4a4a;     /* a touch brighter on hover */
#         }
#     """)
#     # left heatmap (Hall-sensor)
#     self.web_view_left = SyncPlotlyView(self)
#     self.left_scroll = QScrollArea();
#     self.left_scroll.setWidgetResizable(False)
#     self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#     self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#     self.web_view_left.setFixedSize(2500, 650)
#     self.left_scroll.setWidget(self.web_view_left)
#
#     # right heatmap (Proximity)
#     self.web_view_right = SyncPlotlyView(self)
#     self.right_scroll = QScrollArea();
#     self.right_scroll.setWidgetResizable(False)
#     self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#     self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#     self.web_view_right.setFixedSize(2500, 650)
#     self.right_scroll.setWidget(self.web_view_right)
#
#     self.top_hsplit.addWidget(self.left_scroll)
#     self.top_hsplit.addWidget(self.right_scroll)
#     dual_lay.addWidget(self.top_hsplit)
#
#     # ensure sizes apply after first layout
#     # QTimer.singleShot(0, _apply_heatmap_layout)
#     QTimer.singleShot(0, self._apply_heatmap_layout)
#
#     # stack
#     self.top_stack = QStackedWidget()
#     self.top_stack.addWidget(self.single_chart_page)  # index 0
#     self.top_stack.addWidget(self.dual_heatmaps_page)  # index 1
#
#     # add stack to layout
#     main_web_layout.addWidget(self.top_stack)
#
#     # top bar under the stack (used for single chart pages only)
#     self.main_top_scrollbar = self._make_topbar_row(
#         "mainTopBar", main_web_layout, bar_h=10, left_px=1300, right_px=570
#     )
#
#     # ---------- Keep your original main top scrollbar sync for the single page ----------
#     main_inner_hbar = self.main_web_scroll_area.horizontalScrollBar()
#     VIRTUAL_MAX = 2000
#
#     def _eff_main_bounds():
#         imin, imax = main_inner_hbar.minimum(), main_inner_hbar.maximum()
#         eff_max = max(imin, imax - self._right_margin_px)
#         return imin, eff_max
#
#     def _map_main_top_to_inner(v_top: int) -> int:
#         imin, eff_max = _eff_main_bounds()
#         rng = max(1, eff_max - imin)
#         return int(round(imin + (v_top / VIRTUAL_MAX) * rng))
#
#     def _map_main_inner_to_top(v_inner: int) -> int:
#         imin, eff_max = _eff_main_bounds()
#         rng = max(1, eff_max - imin)
#         return int(round(((v_inner - imin) / rng) * VIRTUAL_MAX))
#
#     def _apply_main_fixed_range():
#         self.main_top_scrollbar.blockSignals(True)
#         self.main_top_scrollbar.setRange(0, VIRTUAL_MAX)
#         self.main_top_scrollbar.setPageStep(100)
#         self.main_top_scrollbar.setSingleStep(10)
#         self.main_top_scrollbar.setValue(_map_main_inner_to_top(main_inner_hbar.value()))
#         self.main_top_scrollbar.blockSignals(False)
#
#     def _on_main_top_changed(v):
#         if not getattr(self, "_hscroll_ready_main", False):
#             return
#         main_inner_hbar.setValue(_map_main_top_to_inner(v))
#
#     def _on_main_inner_changed(v):
#         if not getattr(self, "_hscroll_ready_main", False):
#             return
#         self.main_top_scrollbar.blockSignals(True)
#         self.main_top_scrollbar.setValue(_map_main_inner_to_top(v))
#         self.main_top_scrollbar.blockSignals(False)
#
#     self._hscroll_ready_main = False
#     self.main_top_scrollbar.valueChanged.connect(_on_main_top_changed)
#     main_inner_hbar.valueChanged.connect(_on_main_inner_changed)
#
#     def _on_main_inner_range_changed(_min, _max):
#         if _max > _min:
#             self._hscroll_ready_main = True
#             _apply_main_fixed_range()
#
#     main_inner_hbar.rangeChanged.connect(_on_main_inner_range_changed)
#
#     # ---------- BOTTOM STACK ----------
#     self.bottom_stack = QStackedWidget()
#     self.bottom_stack.hide()
#     self.bottom_stack.setContentsMargins(0, 0, 0, 0)
#     self.bottom_stack.currentChanged.connect(lambda idx: self._arm_topbar() if idx == 2 else None)
#     self._HM_FOOTER_H = 44  # tweak if you want more/less
#     self.footer_page = QWidget(objectName="heatmapFooterPage")
#     self.footer_page.setMinimumHeight(self._HM_FOOTER_H)
#     self.footer_page.setMaximumHeight(self._HM_FOOTER_H)
#     # --------------------------- Defect table page (bottom) ---------------------------
#     self.defect_table_page = QWidget()
#     defect_layout = QVBoxLayout(self.defect_table_page)
#     defect_layout.setContentsMargins(0, 0, 0, 0)
#     defect_layout.setSpacing(0)
#
#     # Re-parent tableWidgetDefect into this page
#     old_parent_def = self.ui.tableWidgetDefect.parentWidget()
#     if old_parent_def and old_parent_def.layout():
#         try:
#             old_parent_def.layout().removeWidget(self.ui.tableWidgetDefect)
#         except Exception:
#             pass
#     self.ui.tableWidgetDefect.setParent(self.defect_table_page)
#     self.ui.tableWidgetDefect.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#
#     # Tight top bar (TABLE)
#     self.table_scrollbar = self._make_topbar_row("tableTopBar", defect_layout, bar_h=10, left_px=1300, right_px=570)
#     self.table_scrollbar.hide()
#
#     # Table directly under the bar
#     defect_layout.addWidget(self.ui.tableWidgetDefect)
#
#     # Hide built-in vertical header and install left-side custom vbar
#     vh = self.ui.tableWidgetDefect.verticalHeader()
#     vh.setVisible(False)
#     self.ui.tableWidgetDefect.setCornerButtonEnabled(False)
#     self._install_left_vbar(self.ui.tableWidgetDefect)
#
#     # Sync custom table bar with internal table hbar
#     self._setup_table_scrollbar_sync()
#
#     # --------------------------- Data table page (model view) ---------------------------
#     self.data_table_page = QWidget()
#     tl = QVBoxLayout(self.data_table_page)
#     tl.setContentsMargins(0, 0, 0, 0)
#     tl.setSpacing(0)
#     old_parent_data = self.ui.tableView.parentWidget()
#     if old_parent_data and old_parent_data.layout():
#         try:
#             old_parent_data.layout().removeWidget(self.ui.tableView)
#         except Exception:
#             pass
#     self.ui.tableView.setParent(None)
#     self.ui.tableView.setVisible(True)
#     tl.addWidget(self.ui.tableView)
#
#     # --------------------------- Proximity line chart page (bottom) ---------------------------
#     self.web_page = QWidget()
#     web_layout = QVBoxLayout(self.web_page)
#     web_layout.setContentsMargins(0, 0, 0, 0)
#     web_layout.setSpacing(0)
#
#     # Tight top bar (PROX)
#     self.top_scrollbar = self._make_topbar_row("proxTopBar", web_layout, bar_h=10, left_px=1300, right_px=570)
#
#     # Scroll area without bottom horizontal bar
#     self.web_scroll_area = QScrollArea()
#     self.web_scroll_area.setWidgetResizable(False)
#     self.web_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#     self.web_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
#
#     self.web_view2 = QWebEngineView()
#     self.web_view2.setFixedSize(2500, 600)
#     self.web_scroll_area.setWidget(self.web_view2)
#
#     web_layout.addWidget(self.web_scroll_area)
#
#     # Apply scrollbar theme to bars/areas
#     self._apply_scrollbar_theme("#6AA2FF")
#
#     # Sync top scrollbar with scroll area for proximity line chart
#     inner_hbar = self.web_scroll_area.horizontalScrollBar()
#
#     def _eff_prox_bounds():
#         imin, imax = inner_hbar.minimum(), inner_hbar.maximum()
#         eff_max = max(imin, imax - self._right_margin_px)
#         return imin, eff_max
#
#     def _map_top_to_inner(v_top: int) -> int:
#         imin, eff_max = _eff_prox_bounds()
#         rng = max(1, eff_max - imin)
#         return int(round(imin + (v_top / VIRTUAL_MAX) * rng))
#
#     def _map_inner_to_top(v_inner: int) -> int:
#         imin, eff_max = _eff_prox_bounds()
#         rng = max(1, eff_max - imin)
#         return int(round(((v_inner - imin) / rng) * VIRTUAL_MAX))
#
#     def _apply_fixed_range():
#         self.top_scrollbar.blockSignals(True)
#         self.top_scrollbar.setRange(0, VIRTUAL_MAX)
#         self.top_scrollbar.setPageStep(100)
#         self.top_scrollbar.setSingleStep(10)
#         self.top_scrollbar.setValue(_map_inner_to_top(inner_hbar.value()))
#         self.top_scrollbar.blockSignals(False)
#
#     def _on_top_changed(v):
#         if not getattr(self, "_hscroll_ready", False):
#             return
#         inner_hbar.setValue(_map_top_to_inner(v))
#
#     def _on_inner_changed(v):
#         if not getattr(self, "_hscroll_ready", False):
#             return
#         self.top_scrollbar.blockSignals(True)
#         self.top_scrollbar.setValue(_map_inner_to_top(v))
#         self.top_scrollbar.blockSignals(False)
#
#     self._hscroll_ready = False
#     self.top_scrollbar.valueChanged.connect(_on_top_changed)
#     inner_hbar.valueChanged.connect(_on_inner_changed)
#
#     def _on_inner_range_changed(_min, _max):
#         if _max > _min:
#             self._hscroll_ready = True
#             _apply_fixed_range()
#
#     inner_hbar.rangeChanged.connect(_on_inner_range_changed)
#
#     # nudge once to ensure a rangeChanged after layout
#     QTimer.singleShot(0, lambda: inner_hbar.setValue(inner_hbar.value()))
#     QTimer.singleShot(0, lambda: main_inner_hbar.setValue(main_inner_hbar.value()))
#
#     # Assemble bottom pages
#     self.bottom_stack.addWidget(self.defect_table_page)
#     self.bottom_stack.addWidget(self.data_table_page)
#     self.bottom_stack.addWidget(self.web_page)
#     self.bottom_stack.addWidget(self.footer_page)
#
#     # ---------- Splitter with mid tabbar ----------
#     self.splitter = MidBarSplitter(self, tabbar=self.mid_tabbar)
#     self.splitter.setStretchFactor(0, 3)  # top area (heatmaps)
#     self.splitter.setStretchFactor(1, 1)  # bottom area (table)
#     self.splitter.addWidget(self.main_web_page)
#     self.splitter.addWidget(self.bottom_stack)
#     self.splitter.setChildrenCollapsible(False)
#     self.splitter.setHandleWidth(40)
#     self.splitter.setStretchFactor(0, 1)
#     self.splitter.setStretchFactor(1, 1)
#     self.splitter.setStyleSheet("""
#         QSplitter::handle#MidBarHandle { background: #16181c; }
#         #MidBarFrame { background: #16181c; }
#         QTabBar::tab { color: #d8d8d8; padding: 6px 14px; margin: 0px; border: 0; background: transparent; }
#         QTabBar::tab:selected { color: white; font-weight: 600; }
#     """)
#     self.ui.verticalLayoutGraph.addWidget(self.splitter)
#
#     # initial splitter sizes
#     INIT_SPLIT_BOTTOM_RATIO = 0.45  # 45% bottom, 55% top
#     QTimer.singleShot(0, lambda: self.splitter.setSizes([
#         int(self.height() * (1 - INIT_SPLIT_BOTTOM_RATIO)),  # top
#         int(self.height() * INIT_SPLIT_BOTTOM_RATIO)  # bottom
#     ]))
#
#     # ---------- Constrain splitter sizes + refresh top bars on move ----------
#     def _constrain_splitter_sizes():
#         sizes = self.splitter.sizes()
#         if len(sizes) < 2:
#             return
#         total = sum(sizes)
#         top, bot = sizes[0], sizes[1]
#
#         min_top = int(self._min_top_h or 0)
#         min_bot = int(self._min_bottom_h or 0)
#
#         max_top_by_bot_min = max(0, total - min_bot)
#         hard_max_top = self._max_top_h if self._max_top_h is not None else max_top_by_bot_min
#         hard_max_top = min(hard_max_top, max_top_by_bot_min)
#
#         top = max(min_top, min(top, hard_max_top))
#         bot = total - top
#         if self._max_bottom_h is not None:
#             bot = min(bot, self._max_bottom_h)
#             top = total - bot
#
#         if bot < min_bot:
#             bot = min_bot
#             top = total - bot
#             top = max(min_top, top)
#
#         if [top, bot] != sizes[:2]:
#             self.splitter.blockSignals(True)
#             self.splitter.setSizes([top, bot])
#             self.splitter.blockSignals(False)
#
#     def _on_splitter_moved(*_):
#         _constrain_splitter_sizes()
#         if getattr(self, "_hscroll_ready", False):
#             _apply_fixed_range()
#         if getattr(self, "_hscroll_ready_main", False):
#             _apply_main_fixed_range()
#         QTimer.singleShot(10, self._refresh_table_scrollbars)
#
#     self.splitter.splitterMoved.connect(_on_splitter_moved)




def _build_main_section(self):
    """
    Builds the main visualization area:
    - Top stack (single chart + dual heatmaps)
    - Bottom stack (tables + proximity chart)
    - Mid splitter with constraints and scrollbar sync
    """

    # ======================================================================
    # STATE INITIALIZATION
    # ======================================================================
    if not hasattr(self, "_hm_layout_mode"):
        self._hm_layout_mode = "vertical"  # persisted layout mode

    # ======================================================================
    # TOP SECTION: MAIN WEB PAGE CONTAINER
    # ======================================================================
    self.main_web_page = QWidget()
    main_web_layout = QVBoxLayout(self.main_web_page)
    main_web_layout.setContentsMargins(0, 0, 0, 0)
    main_web_layout.setSpacing(0)

    # ----------------------------------------------------------------------
    # PAGE 0: SINGLE CHART PAGE (Line / 3D)
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # PAGE 1: DUAL HEATMAPS PAGE
    # ----------------------------------------------------------------------
    self.dual_heatmaps_page = QWidget()
    dual_lay = QVBoxLayout(self.dual_heatmaps_page)
    dual_lay.setContentsMargins(0, 0, 0, 0)
    dual_lay.setSpacing(6)

    # Toolbar placeholder (toggle / controls)
    top_toolbar = QHBoxLayout()
    top_toolbar.setContentsMargins(8, 6, 8, 4)
    top_toolbar.setSpacing(8)
    top_toolbar.addStretch(1)
    dual_lay.addLayout(top_toolbar)

    # Dual heatmap splitter
    self.top_hsplit = QSplitter(
        Qt.Orientation.Horizontal if self._hm_layout_mode == "horizontal"
        else Qt.Orientation.Vertical
    )
    self.top_hsplit.setChildrenCollapsible(False)
    self.top_hsplit.setStretchFactor(0, 1)
    self.top_hsplit.setStretchFactor(1, 1)
    self.top_hsplit.setObjectName("TopHSplit")
    self.top_hsplit.setStyleSheet("""
        QSplitter#TopHSplit::handle {
            background-color: #3a3a3a;
            border: 1px solid #2a2a2a;
        }
        QSplitter#TopHSplit::handle:hover {
            background-color: #4a4a4a;
        }
    """)

    # Left heatmap (Hall sensor)
    self.web_view_left = SyncPlotlyView(self)
    self.left_scroll = QScrollArea()
    self.left_scroll.setWidgetResizable(False)
    self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    self.web_view_left.setFixedSize(2500, 650)
    self.left_scroll.setWidget(self.web_view_left)

    # Right heatmap (Proximity)
    self.web_view_right = SyncPlotlyView(self)
    self.right_scroll = QScrollArea()
    self.right_scroll.setWidgetResizable(False)
    self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    self.web_view_right.setFixedSize(2500, 650)
    self.right_scroll.setWidget(self.web_view_right)

    self.top_hsplit.addWidget(self.left_scroll)
    self.top_hsplit.addWidget(self.right_scroll)
    dual_lay.addWidget(self.top_hsplit)

    QTimer.singleShot(0, lambda : _apply_heatmap_layout(self))

    # ----------------------------------------------------------------------
    # TOP STACK (Single Chart / Dual Heatmaps)
    # ----------------------------------------------------------------------
    self.top_stack = QStackedWidget()
    self.top_stack.addWidget(self.single_chart_page)     # index 0
    self.top_stack.addWidget(self.dual_heatmaps_page)    # index 1

    main_web_layout.addWidget(self.top_stack)

    # Top scrollbar (used only for single chart)
    # self.main_top_scrollbar = self._make_topbar_row(
    #     "mainTopBar",
    #     main_web_layout,
    #     bar_h=10,
    #     left_px=1300,
    #     right_px=570
    # )
    self.main_top_scrollbar = _make_topbar_row(
        self,
        "mainTopBar",
        main_web_layout,
        bar_h=10,
        left_px=1300,
        right_px=570
    )

    # ======================================================================
    # SINGLE CHART SCROLLBAR SYNC LOGIC
    # ======================================================================
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
        self.main_top_scrollbar.setValue(
            _map_main_inner_to_top(main_inner_hbar.value())
        )
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

    # ======================================================================
    # BOTTOM STACK CONTAINER
    # ======================================================================
    self.bottom_stack = QStackedWidget()
    self.bottom_stack.hide()
    self.bottom_stack.setContentsMargins(0, 0, 0, 0)
    self.bottom_stack.currentChanged.connect(
        lambda idx: lambda : _arm_topbar(self, idx) if idx == 2 else None
    )

    self._HM_FOOTER_H = 44
    self.footer_page = QWidget(objectName="heatmapFooterPage")
    self.footer_page.setMinimumHeight(self._HM_FOOTER_H)
    self.footer_page.setMaximumHeight(self._HM_FOOTER_H)

    # ----------------------------------------------------------------------
    # DEFECT TABLE PAGE
    # ----------------------------------------------------------------------
    self.defect_table_page = QWidget()
    defect_layout = QVBoxLayout(self.defect_table_page)
    defect_layout.setContentsMargins(0, 0, 0, 0)
    defect_layout.setSpacing(0)

    old_parent_def = self.ui.tableWidgetDefect.parentWidget()
    if old_parent_def and old_parent_def.layout():
        try:
            old_parent_def.layout().removeWidget(self.ui.tableWidgetDefect)
        except Exception:
            pass

    self.ui.tableWidgetDefect.setParent(self.defect_table_page)
    self.ui.tableWidgetDefect.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )

    # self.table_scrollbar = self._make_topbar_row(
    #     "tableTopBar",
    #     defect_layout,
    #     bar_h=10,
    #     left_px=1300,
    #     right_px=570
    # )
    self.table_scrollbar = _make_topbar_row(
        self,
        "tableTopBar",
        defect_layout,
        bar_h=10,
        left_px=1300,
        right_px=570
    )
    self.table_scrollbar.hide()

    defect_layout.addWidget(self.ui.tableWidgetDefect)

    vh = self.ui.tableWidgetDefect.verticalHeader()
    vh.setVisible(False)
    self.ui.tableWidgetDefect.setCornerButtonEnabled(False)

    _install_left_vbar(self, self.ui.tableWidgetDefect)
    _setup_table_scrollbar_sync(self)

    # ----------------------------------------------------------------------
    # DATA TABLE PAGE
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # PROXIMITY LINE CHART PAGE
    # ----------------------------------------------------------------------
    self.web_page = QWidget()
    web_layout = QVBoxLayout(self.web_page)
    web_layout.setContentsMargins(0, 0, 0, 0)
    web_layout.setSpacing(0)

    # self.top_scrollbar = self._make_topbar_row(
    #     "proxTopBar",
    #     web_layout,
    #     bar_h=10,
    #     left_px=1300,
    #     right_px=570
    # )
    self.top_scrollbar = _make_topbar_row(
        self,
        "proxTopBar",
        web_layout,
        bar_h=10,
        left_px=1300,
        right_px=570
    )

    self.web_scroll_area = QScrollArea()
    self.web_scroll_area.setWidgetResizable(False)
    self.web_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.web_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

    self.web_view2 = QWebEngineView()
    self.web_view2.setFixedSize(2500, 600)

    self.web_scroll_area.setWidget(self.web_view2)
    web_layout.addWidget(self.web_scroll_area)
    # self._apply_scrollbar_theme("#6AA2FF")
    _apply_scrollbar_theme(self, "#6AA2FF")

    # ======================================================================
    # PROXIMITY SCROLLBAR SYNC LOGIC
    # ======================================================================
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
        self.top_scrollbar.setValue(
            _map_inner_to_top(inner_hbar.value())
        )
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

    QTimer.singleShot(0, lambda: inner_hbar.setValue(inner_hbar.value()))
    QTimer.singleShot(0, lambda: main_inner_hbar.setValue(main_inner_hbar.value()))

    # ======================================================================
    # ASSEMBLE BOTTOM STACK
    # ======================================================================
    self.bottom_stack.addWidget(self.defect_table_page)
    self.bottom_stack.addWidget(self.data_table_page)
    self.bottom_stack.addWidget(self.web_page)
    self.bottom_stack.addWidget(self.footer_page)

    # ======================================================================
    # MID BAR SPLITTER (UNCHANGED)
    # ======================================================================
    self.splitter = MidBarSplitter(self, tabbar=self.mid_tabbar)
    self.splitter.setStretchFactor(0, 3)
    self.splitter.setStretchFactor(1, 1)
    self.splitter.addWidget(self.main_web_page)
    self.splitter.addWidget(self.bottom_stack)
    self.splitter.setChildrenCollapsible(False)
    self.splitter.setHandleWidth(40)
    self.splitter.setStretchFactor(0, 1)
    self.splitter.setStretchFactor(1, 1)
    self.splitter.setStyleSheet("""
        QSplitter::handle#MidBarHandle { background: #16181c; }
        #MidBarFrame { background: #16181c; }
        QTabBar::tab {
            color: #d8d8d8;
            padding: 6px 14px;
            margin: 0px;
            border: 0;
            background: transparent;
        }
        QTabBar::tab:selected {
            color: white;
            font-weight: 600;
        }
    """)

    self.ui.verticalLayoutGraph.addWidget(self.splitter)

    INIT_SPLIT_BOTTOM_RATIO = 0.45
    QTimer.singleShot(0, lambda: self.splitter.setSizes([
        int(self.height() * (1 - INIT_SPLIT_BOTTOM_RATIO)),
        int(self.height() * INIT_SPLIT_BOTTOM_RATIO),
    ]))

    # ======================================================================
    # SPLITTER CONSTRAINTS + REFRESH
    # ======================================================================
    def _constrain_splitter_sizes():
        sizes = self.splitter.sizes()
        if len(sizes) < 2:
            return

        total = sum(sizes)
        top, bot = sizes[0], sizes[1]

        min_top = int(self._min_top_h or 0)
        min_bot = int(self._min_bottom_h or 0)

        max_top_by_bot_min = max(0, total - min_bot)
        hard_max_top = (
            self._max_top_h if self._max_top_h is not None
            else max_top_by_bot_min
        )
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
    _update_left_vbar_geometry(self, tw)
    tw.installEventFilter(self)  # so we can reposition it on resize/show

    # first-time sync after layout settles
    QTimer.singleShot(0, _apply_range)
    _style_left_vertical_bar(self)


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