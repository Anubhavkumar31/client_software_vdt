import pandas as pd
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QPen, QPolygonF, QPainter
from PyQt6.QtWidgets import QMessageBox, QGraphicsPolygonItem, QGraphicsView, QGraphicsScene, QPushButton, QLineEdit, \
    QLabel



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


def _back_from_pipe_locator(self):
    # Go back only if previous widget exists and is valid
    if hasattr(self, "_pipe_locator_prev_widget") and self._pipe_locator_prev_widget:
        self.top_stack.setCurrentWidget(self._pipe_locator_prev_widget)
    else:
        # fallback: go to first available widget
        if self.top_stack.count() > 0:
            self.top_stack.setCurrentIndex(0)

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