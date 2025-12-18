import os
import re
import time

import pandas as pd
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QAbstractTableModel, QVariant, QUrl
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout, QMessageBox
from glob import glob

from main_section_view.helpers_temp import _arm_topbar, _arm_main_topbar, tab_switcher2
from main_section_view.table_data_worker import on_table_data_ready_con
from main_section_view.utils import update_digsheet_button_state


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
    # self.load_selected_by_index(idx)
    load_selected_by_index(self, idx)
    #self.btnLoadPipe.clicked.connect(self.load_selected_pipe)


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
        pipe_idx = _extract_index(self, name)

        # Show loading dialog
        self.loading_dialog = ModernLoadingDialog(self)
        self.loading_dialog.show()

        # Create and start worker thread
        self.loader_worker = PipeLoaderWorker(pkl_path, self.project_root, pipe_idx)

        # Connect signals
        self.loader_worker.progress_updated.connect(self.loading_dialog.update_progress)
        self.loader_worker.time_estimate.connect(self.loading_dialog.update_time_estimate)
        self.loader_worker.data_loaded.connect(lambda df: on_data_loaded(self, df))
        self.loader_worker.assets_loaded.connect(lambda assets: on_assets_loaded(self, assets))
        # self.loader_worker.table_data_ready.connect(lambda df: on_table_data_ready(self, df))
        self.loader_worker.table_data_ready.connect(lambda df: on_table_data_ready(self, df))
        self.loader_worker.error_occurred.connect(lambda error_msg: on_loading_error(self, error_msg))
        self.loader_worker.finished.connect(lambda : on_loading_finished(self))

        # Start the worker
        self.loader_worker.start()

    except Exception as e:
        self.open_Error(f"load_selected_by_index error: {e}")



def on_data_loaded(self, df):
    """Handle loaded DataFrame - runs on main thread"""
    self.curr_data = df
    self.header_list = list(df.columns)

    # Use lightweight model instead of building QStandardItem rows
    self.df_model = PandasModel(df)
    self.proxy_model.setSourceModel(self.df_model)
    self.ui.tableView.setModel(self.proxy_model)
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
    on_table_data_ready_con(self, df)



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
    _refresh_current_view(self)
    QTimer.singleShot(0, lambda : _arm_topbar(self))
    QTimer.singleShot(0, lambda : _arm_main_topbar(self))
    update_digsheet_button_state(self)
    QTimer.singleShot(100, lambda : update_digsheet_button_state(self))
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



def load_next_pipe(self):
    """Go to next pipe and load automatically"""
    cb = self.ui.comboBoxPipe
    idx = cb.currentIndex()
    if idx < cb.count() - 1:  # not last
        cb.setCurrentIndex(idx + 1)
        load_selected_pipe(self)

def load_prev_pipe(self):
    """Go to previous pipe and load automatically"""
    cb = self.ui.comboBoxPipe
    idx = cb.currentIndex()
    if idx > 0:  # not first
        cb.setCurrentIndex(idx - 1)
        load_selected_pipe(self)

@staticmethod
def _extract_index(self, text: str) -> str:
    m = re.search(r'\d+', text)
    return m.group(0) if m else text

def _refresh_current_view(self):
    """Force the current tab to re-render with latest asset paths."""
    try:
        # Clear both views to avoid showing stale content
        self.web_view.setUrl(QUrl())
        self.web_view2.setUrl(QUrl())
    except Exception:
        pass
    # Let the event loop breathe, then render the right thing for the active tab
    QTimer.singleShot(0, lambda: tab_switcher2(self))
