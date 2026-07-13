import pandas as pd
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QPen, QPolygonF, QPainter, QFont
from PyQt6.QtWidgets import QMessageBox, QGraphicsPolygonItem, QGraphicsView, QGraphicsScene, QPushButton, QLineEdit, \
    QLabel, QGraphicsRectItem, QGraphicsTextItem

# ================== IMPORTS ==================
from PyQt6.QtWidgets import (
    QDialog, QGraphicsView, QGraphicsScene,
    QVBoxLayout, QPushButton, QLabel, QLineEdit,
    QGraphicsPolygonItem
)
from PyQt6.QtGui import QPen, QPolygonF, QPainter, QFont, QColor, QBrush
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
        self._detail_box_widget = None  # Store the overlay widget

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
        # ... (keep existing controls code) ...
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
        # ... (keep existing filter code) ...
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
        # ... (keep existing reset code) ...
        self._range = None
        self.start_edit.clear()
        self.end_edit.clear()
        self._draw_pipe()

    # ---------- DATA ----------
    def _prepare_data(self):
        # ... (keep existing prepare data code) ...
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
            return f"{int(round(float(d)))} m" if row["__is_weld__"] else str(
                row.get("Feature Type", "Feature")
            )

        self.df["__label__"] = self.df.apply(label, axis=1)
        self.df.dropna(subset=["Abs. Distance (m)"], inplace=True)

    # ---------- DRAW ----------
    def _draw_pipe(self):
        # ... (keep existing draw code) ...
        self.scene.clear()
        self._active_feature_label = None
        self._clear_detail_box()  # Clear the overlay widget

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

        weld_index = 0

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
                    QPointF(x, pipe_y),  # tip touching pipe
                    QPointF(x - 7, pipe_y - 16),
                    QPointF(x + 7, pipe_y - 16),
                ]))

                arrow.setBrush(Qt.GlobalColor.blue)
                arrow.setPen(QPen(Qt.GlobalColor.blue))

                # store ALL row data for click handler
                arrow.setData(0, r["__label__"])
                arrow.setData(1, x)
                arrow.setData(2, pipe_y)
                # Store the full row data as a dict for detail box
                arrow.setData(3, r.to_dict())

                arrow.setFlag(QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable)
                self.scene.addItem(arrow)

        self.setSceneRect(0, 0, pipe_len + 300, 360)

    # ---------- CLEAR DETAIL BOX ----------
    def _clear_detail_box(self):
        """Remove the overlay detail box widget."""
        if self._detail_box_widget:
            self._detail_box_widget.deleteLater()
            self._detail_box_widget = None

    # ---------- BUILD DETAIL BOX (as overlay widget) ----------
    def _show_detail_box(self, row_data):
        """Display a detail box as an overlay widget in the upper right corner."""
        # Remove existing detail box
        self._clear_detail_box()

        # Create a container widget
        container = QDialog(self)  # Use QDialog as a lightweight overlay
        container.setWindowFlags(Qt.WindowType.Widget)  # Make it a child widget
        container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        container.setStyleSheet("background: transparent;")

        # Box dimensions
        box_x = self.width() - 220 - 10  # 10px padding from right edge
        box_y = 50  # Below the control buttons
        box_width = 200
        line_height = 18

        # Extract fields
        feature_type = str(row_data.get("Feature Type", "Unknown"))
        distance = row_data.get("Abs. Distance (m)", "N/A")
        length = row_data.get("Length (mm)", row_data.get("Length", "N/A"))
        width = row_data.get("Width (mm)", row_data.get("Width", "N/A"))
        depth = row_data.get("Depth %", row_data.get("Depth", "N/A"))
        orientation = row_data.get("Orientation o' clock", row_data.get("Orientation", "N/A"))

        # Format distance
        if pd.notna(distance):
            distance = f"{float(distance):.2f} m"
        else:
            distance = "N/A"

        # Build detail lines
        details = [
            f"Feature: {feature_type}",
            f"Distance: {distance}",
            f"Length: {length if pd.notna(length) else 'N/A'}",
            f"Width: {width if pd.notna(width) else 'N/A'}",
            f"Depth: {depth if pd.notna(depth) else 'N/A'}",
            f"Orientation: {orientation if pd.notna(orientation) else 'N/A'}",
        ]

        box_height = len(details) * line_height + 34

        # Create a frame widget with a background
        frame = QLabel(container)
        frame.setGeometry(0, 0, box_width, box_height)
        frame.setStyleSheet("""
            background-color: #F0F8FF;
            border: 2px solid #4682B4;
            border-radius: 4px;
        """)

        # Title bar
        title_bar = QLabel(frame)
        title_bar.setGeometry(0, 0, box_width, 22)
        title_bar.setStyleSheet("""
            background-color: #4682B4;
            border-radius: 2px 2px 0 0;
        """)

        # Title text
        title = QLabel("Defect Details", frame)
        title.setGeometry(6, 2, box_width - 12, 20)
        title.setStyleSheet("color: white; font-weight: bold; font-size: 9pt; background: transparent;")

        # Detail lines
        for i, line in enumerate(details):
            label = QLabel(line, frame)
            label.setGeometry(8, 26 + i * line_height, box_width - 16, line_height)
            label.setStyleSheet("color: black; font-size: 8pt; background: transparent;")

        # Position the container
        container.setGeometry(box_x, box_y, box_width, box_height)
        container.show()

        # Store reference
        self._detail_box_widget = container

    # ---------- FEATURE CLICK (show details in box) ----------
    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        item = self.itemAt(event.position().toPoint())
        if isinstance(item, QGraphicsPolygonItem):
            # Get the stored row data
            row_data = item.data(3)
            if row_data:
                self._show_detail_box(row_data)
            else:
                # Fallback to just label if no row data
                self._clear_detail_box()
                label = item.data(0)
                if label:
                    t = self.scene.addText(label)
                    t.setDefaultTextColor(Qt.GlobalColor.blue)
                    t.setPos(item.data(1) - 35, item.data(2) - 32)
                    self._active_feature_label = t

    # ---------- RESIZE EVENT ----------
    def resizeEvent(self, event):
        """Reposition the detail box when the view is resized."""
        super().resizeEvent(event)
        if self._detail_box_widget:
            # Reposition to top right with padding
            self._detail_box_widget.move(self.width() - self._detail_box_widget.width() - 10, 50)

    # ---------- ZOOM ----------
    def wheelEvent(self, event):
        # ... (keep existing wheel code) ...
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