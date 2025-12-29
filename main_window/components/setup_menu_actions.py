import pandas as pd
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QPen, QPolygonF, QPainter
from PyQt6.QtWidgets import QMessageBox, QGraphicsPolygonItem, QGraphicsView, QGraphicsScene, QPushButton, QLineEdit, \
    QLabel

from main_section_view.helpers_temp import _on_middle_tab_changed, syncdropdownwithtabs, _connect_guarded_graph_controls
from main_section_view.workers.load_button_working import load_prev_pipe, load_next_pipe
from main_section_view.utils import update_load_button_state
from menubar.File_menu.close_project import close_project
from menubar.File_menu.open_project import open_project
from menubar.File_menu.quit_app import quit_app
from menubar.help_menu.open_about import open_About
from menubar.help_menu.open_manual import open_manual
from menubar.report_menu.generate.digsheet import open_digs
from menubar.report_menu.generate.final_report import open_Final_Report
from menubar.report_menu.generate.pipetally import open_pipe_tally
from menubar.report_menu.generate.preliminary_report import open_Preliminary_Report
from menubar.report_menu.open_PipeScheme import open_PipeScheme
from menubar.report_menu.open_pipehigh import open_PipeHigh
from menubar.view_menu.open_ERF import open_ERF
from menubar.view_menu.open_XYZ import open_XYZ
from menubar.view_menu.open_graphs import open_graphs


def setup_menu_actions(self):
    """
    -------------------------------------------------------------
    MENU ACTIONS + GRAPH CONTROL SIGNALS
    -------------------------------------------------------------
    Connects all QAction menu items to their handlers.
    Also initializes graph-related UI controls and guards
    them so they activate only when a project/pipe is loaded.

    Replaces default tab change handler with guarded switching.
    -------------------------------------------------------------
    """
    #normal button connections
    setup_actions(self)

    # load button state on depend on whats in pipe number selection dropdown
    self.ui.comboBoxPipe.currentIndexChanged.connect(lambda idx: update_load_button_state(self, idx))

    #heatmap/linechart/3dgraph guarded connections
    _connect_guarded_graph_controls(self)

    try:
        self.ui.tabWidgetM.currentChanged.disconnect()
    except:
        pass

    self.ui.tabWidgetM.currentChanged.connect(lambda index: _on_middle_tab_changed(self, index))
    self.ui.tabWidgetM.currentChanged.connect(lambda index: syncdropdownwithtabs(self, index))


from test_main import ExcelDualAxisZoomChart

class PipeLocatorWidget(QGraphicsView):


    backRequested = pyqtSignal()

    def __init__(self, pipe_tally: pd.DataFrame, parent=None):
        super().__init__(parent)

        # -------- DATA --------
        self._df_full = pipe_tally.copy()
        self.df = pipe_tally.copy()
        self._range = None

        # -------- GRAPHICS --------
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setInteractive(True)
        self.setMouseTracking(True)

        # cursor-centric zoom
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # zoom control
        self._zoom = 1.0
        self._zoom_step = 1.25
        self._zoom_min = 0.4
        self._zoom_max = 8.0

        self._active_label = None
        self.setMinimumHeight(320)

        # UI
        self._add_buttons()
        self._add_range_filter()

        self._prepare_data()
        self._draw_pipe()

    # -------------------------------------------------
    # BACK / CLOSE / ESC
    # -------------------------------------------------

    def _add_buttons(self):
        self.back_btn = QPushButton("← Back", self)
        self.back_btn.setFixedSize(70, 26)
        self.back_btn.move(10, 10)
        self.back_btn.clicked.connect(self.backRequested.emit)
        self.back_btn.raise_()

        self.close_btn = QPushButton("✕", self)
        self.close_btn.setFixedSize(26, 26)
        self.close_btn.clicked.connect(self.backRequested.emit)
        self.close_btn.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.close_btn.move(self.width() - 36, 10)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.backRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    # -------------------------------------------------
    # RANGE FILTER UI
    # -------------------------------------------------
    def _add_range_filter(self):
        self.lbl_start = QLabel("Start (m):", self)
        self.lbl_start.move(100, 12)

        self.start_edit = QLineEdit(self)
        self.start_edit.setFixedWidth(70)
        self.start_edit.move(165, 10)

        self.lbl_end = QLabel("End (m):", self)
        self.lbl_end.move(245, 12)

        self.end_edit = QLineEdit(self)
        self.end_edit.setFixedWidth(70)
        self.end_edit.move(300, 10)

        self.apply_btn = QPushButton("Apply", self)
        self.apply_btn.move(380, 9)
        self.apply_btn.clicked.connect(self._apply_range_filter)

        self.reset_btn = QPushButton("Reset", self)
        self.reset_btn.move(450, 9)
        self.reset_btn.clicked.connect(self._reset_range_filter)

        for w in (
            self.lbl_start, self.start_edit,
            self.lbl_end, self.end_edit,
            self.apply_btn, self.reset_btn
        ):
            w.raise_()

    def _apply_range_filter(self):
        try:
            start = float(self.start_edit.text())
            end = float(self.end_edit.text())
            if start >= end:
                return
        except ValueError:
            return

        self._range = (start, end)
        self.df = self._df_full[
            (self._df_full["Abs. Distance (m)"] >= start) &
            (self._df_full["Abs. Distance (m)"] <= end)
        ].copy()

        self._prepare_data()
        self._draw_pipe()

    def _reset_range_filter(self):
        self._range = None
        self.df = self._df_full.copy()
        self._prepare_data()
        self._draw_pipe()

    # -------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------
    def _prepare_data(self):
        df = self.df

        if "Feature Type" in df.columns:
            df["__is_weld__"] = (
                df["Feature Type"].isna() |
                (df["Feature Type"].astype(str).str.strip() == "")
            )
        else:
            df["__is_weld__"] = False

        def make_label(row):
            dist = row.get("Abs. Distance (m)")
            if row["__is_weld__"]:
                if pd.isna(dist):
                    return "Weld"
                return f"{int(round(dist))} m"
            return str(row.get("Feature Type") or "Feature")

        df["__label__"] = df.apply(make_label, axis=1)
        df.dropna(subset=["Abs. Distance (m)"], inplace=True)
        self.df = df

    # -------------------------------------------------
    # DRAW PIPE (DYNAMIC SCALE)
    # -------------------------------------------------
    def _draw_pipe(self):
        self.scene.clear()
        self._active_label = None

        if self.df.empty:
            return

        if self._range:
            start, end = self._range
        else:
            start = 0.0
            end = self.df["Abs. Distance (m)"].max()

        base_scale = 10.0
        scale = base_scale * self._zoom
        pipe_y = 170
        pipe_len_px = (end - start) * scale

        # Pipe
        self.scene.addLine(
            0, pipe_y,
            pipe_len_px, pipe_y,
            QPen(Qt.GlobalColor.black, 3)
        )

        # Welds (staggered)
        welds = self.df[self.df["__is_weld__"]].sort_values("Abs. Distance (m)")
        for i, (_, r) in enumerate(welds.iterrows()):
            x = (r["Abs. Distance (m)"] - start) * scale
            self.scene.addLine(
                x, pipe_y - 8,
                x, pipe_y - 1,
                QPen(Qt.GlobalColor.black, 2)
            )
            y_offset = (i % 3) * 12
            t = self.scene.addText(r["__label__"])
            t.setDefaultTextColor(Qt.GlobalColor.black)
            t.setPos(x - 18, pipe_y - 36 - y_offset)

        # -------- FEATURES (ABOVE PIPE, TOUCHING PIPE) --------
        for _, r in self.df[~self.df["__is_weld__"]].iterrows():
            x = (r["Abs. Distance (m)"] - start) * scale

            arrow = QGraphicsPolygonItem(QPolygonF([
                QPointF(x, pipe_y),          # 👈 touch pipe
                QPointF(x - 7, pipe_y - 16),
                QPointF(x + 7, pipe_y - 16),
            ]))
            arrow.setBrush(Qt.GlobalColor.blue)
            arrow.setPen(QPen(Qt.GlobalColor.blue))
            arrow.setData(0, r["__label__"])
            arrow.setData(1, x)
            arrow.setData(2, pipe_y)
            arrow.setFlag(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable)
            self.scene.addItem(arrow)

        self.setSceneRect(0, 0, pipe_len_px + 400, 380)

    # -------------------------------------------------
    # CLICK → SHOW FEATURE NAME (ABOVE PIPE)
    # -------------------------------------------------
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        item = self.itemAt(event.position().toPoint())

        if isinstance(item, QGraphicsPolygonItem):
            if self._active_label:
                self.scene.removeItem(self._active_label)

            label = item.data(0)
            x = item.data(1)
            pipe_y = item.data(2)

            t = self.scene.addText(label)
            t.setDefaultTextColor(Qt.GlobalColor.blue)
            t.setPos(x - 35, pipe_y - 40)
            self._active_label = t

    # -------------------------------------------------
    # ZOOM + SCROLL (REDRAW ON ZOOM)
    # -------------------------------------------------
    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = self._zoom_step if delta > 0 else (1 / self._zoom_step)
            new_zoom = self._zoom * factor

            if self._zoom_min <= new_zoom <= self._zoom_max:
                self._zoom = new_zoom
                self._draw_pipe()

            event.accept()
            return

        hbar = self.horizontalScrollBar()
        hbar.setValue(hbar.value() - delta)
        event.accept()


def _back_from_pipe_locator(self):
    # Go back only if previous widget exists and is valid
    if hasattr(self, "_pipe_locator_prev_widget") and self._pipe_locator_prev_widget:
        self.top_stack.setCurrentWidget(self._pipe_locator_prev_widget)
    else:
        # fallback: go to first available widget
        if self.top_stack.count() > 0:
            self.top_stack.setCurrentIndex(0)
def open_pipe_locator(self):
    if self.pipe_tally is None or self.pipe_tally.empty:
        QMessageBox.warning(self, "Pipe Locator", "Pipe tally not loaded")
        return

    # ✅ Save previous view ONLY if we are not already in PipeLocator
    current = self.top_stack.currentWidget()

    if not isinstance(current, PipeLocatorWidget):
        self._pipe_locator_prev_widget = current

    # Create Pipe Locator only once
    if not hasattr(self, "_pipe_locator_view"):
        self._pipe_locator_view = PipeLocatorWidget(self.pipe_tally, self)
        self.top_stack.addWidget(self._pipe_locator_view)

        # 🔙 Back / ✕ / ESC → go back properly
        self._pipe_locator_view.backRequested.connect(
            lambda :_back_from_pipe_locator(self)
        )

    # Show Pipe Locator
    self.top_stack.setCurrentWidget(self._pipe_locator_view)

def open_customplot(self):
    if not self.project_is_open:
        QMessageBox.information(
            self,
            "Project Required",
            "Please create or open a project first."
        )
        return
    if not hasattr(self, "_custom_plot_window"):
        self._custom_plot_window = ExcelDualAxisZoomChart(self)

    self._custom_plot_window.show()
    self._custom_plot_window.raise_()
    self._custom_plot_window.activateWindow()



def setup_actions(self):
    a = self.ui
    #File menu section
    a.action_Create_Proj.triggered.connect(lambda: open_project(self))
    a.action_Close_Proj.triggered.connect(lambda: close_project(self))
    a.action_Quit.triggered.connect(lambda: quit_app(self))
    a.action_Pipe_Locator.triggered.connect(lambda :open_pipe_locator(self))


    #View section
    a.action_ERF.triggered.connect(lambda: open_ERF(self))
    a.action_XYZ.triggered.connect(lambda: open_XYZ(self))
    a.action_customplot.triggered.connect(lambda : open_customplot(self))
    a.action_graphs.triggered.connect(lambda: open_graphs(self))
    # self.ui.action_Export_Table.triggered.connect(self.gen_data)

    #report section
    a.action_Pipe_High.triggered.connect(lambda: open_PipeHigh(self))
    a.action_Pipe_Sch.triggered.connect(lambda: open_PipeScheme(self))
    a.Final_Report.triggered.connect(lambda: open_Final_Report(self))
    a.action_Preliminary_Report.triggered.connect(lambda: open_Preliminary_Report(self))
    a.actionStandard.triggered.connect(lambda: open_digs(self))  # original (by defect no.)
    a.action__pipetally.triggered.connect(lambda: open_pipe_tally(self))

    #help section
    a.action_Manual.triggered.connect(lambda: open_manual(self))
    a.action_About.triggered.connect(lambda: open_About(self))

    if hasattr(a, "pushButtonNext"): a.pushButtonNext.clicked.connect(lambda : load_next_pipe(self))
    if hasattr(a, "pushButtonPrev"): a.pushButtonPrev.clicked.connect(lambda : load_prev_pipe(self))


    #extra
    # a.action_Final_Report.triggered.connect(self.open_Report)
    # a.action_Assessment.triggered.connect(self.open_Assessment)
    # a.action_Cluster.triggered.connect(self.open_Cluster)
    # a.actionMetal_Loss_Distribution_MLD.triggered.connect(self.open_CMLD)
    # a.actionDepth_Based_Anomalies_Distribution_DBAD.triggered.connect(self.open_DBAD)
    # a.actionERF_Based_Anomalies_Distribution_E_AD.triggered.connect(self.open_EAD)
    # a.action_Custom.triggered.connect(self.add_plot_custom)
    # a.action_Telemetry.triggered.connect(self.add_plot_tele)
    # a.actionAnomalies_Distribution.triggered.connect(self.add_plot_ad)
    # a.action_DefectDetect.triggered.connect(self.draw_boxes_v2)
    # a.actionAdmin_Panel.triggered.connect(self.open_Admin)



