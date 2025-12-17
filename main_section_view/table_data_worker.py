import numpy as np
import pandas as pd
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QTimer, QEventLoop
from PyQt6.QtWidgets import QTableWidgetItem, QHeaderView

# from main_window.main_window import setup_table_scroll

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
def on_table_data_ready_con(self, df):
    """Handle processed table data"""
    self.curr_data = df  # 👈 make sure we keep a reference for filtering later

    if df is not None:
        # 👇 populate the column filter dropdown with available columns

        # Check if this is a PipeTally format or defects.csv format
        if "Feature Type" in df.columns:
            _populate_defect_table_from_tally(self, df)
        else:
            _populate_defect_table_from_csv(self, df)
    else:
        self._show_no_defects_message()


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

    _show_defects_table(self)
    _start_fill_qtablewidget_batched(self, view, chunk_size=300)

    setup_table_scroll(self.ui.tableWidgetDefect)
    QTimer.singleShot(150, self._refresh_table_scrollbars)


def _populate_defect_table_from_csv(self, df: pd.DataFrame):
    tw = self.ui.tableWidgetDefect
    tw.clearSelection()

    if df is None or df.empty:
        self._show_no_defects_message()
        return

    # Show table since we have data
    _show_defects_table(self)

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
    _setup_table_styling(self)
    self.update_digsheet_button_state()

    # ✅ keep the dropdown in sync with the visible table headers
    if not self._selected_columns:
        self._selected_columns = set(self._current_headers_for_filter()) | set(self.BACKEND_LOCKED_COLS)
    self.apply_column_filter()



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
    QTimer.singleShot(0, lambda : _fill_tablewidget_chunk(self))


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
        _setup_table_styling(self)

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
        QTimer.singleShot(0, lambda : _fill_tablewidget_chunk(self))


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