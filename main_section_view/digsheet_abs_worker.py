import os
import subprocess
import sys
import tempfile
import uuid

from PyQt6.QtWidgets import QMessageBox

from typing import Optional



def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def _dump_tally_to_temp(df):
    import pickle
    p = os.path.join(tempfile.gettempdir(), f"pipe_tally_{uuid.uuid4().hex}.pkl")
    with open(p, "wb") as f: pickle.dump(df, f)
    return p

def open_digsheet_by_abs_from_selection_con(self):
    try:
        if not self.project_is_open or not isinstance(self.pipe_tally, pd.DataFrame):
            QMessageBox.warning(self, "No Pipe Tally", "Load a project/tally first.");
            return
        abs_text = _get_selected_abs_distance_from_defect_table(self)
        if not abs_text: return

        tally_pkl = _dump_tally_to_temp(self.pipe_tally)
        dig_py_abs = resource_path(os.path.join("dig", "digsheet_abs.py"))
        if not os.path.exists(dig_py_abs):
            QMessageBox.critical(self, "Script not found", f"Missing: {dig_py_abs}");
            return

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

def _get_selected_abs_distance_from_defect_table(self) -> Optional[str]:
    tw = self.ui.tableWidgetDefect
    if tw.rowCount() == 0 or tw.columnCount() == 0:
        QMessageBox.warning(self, "No data", "Defect table is empty.")
        return None

    abs_col = _abs_col_index_silent(self)
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
# Digsheet enable logic + cursor/tooltip polish
# ---------------------------
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt

def update_digsheet_button_state(self):
    if not self.project_is_open:
        self.btnDigsheetAbs.setEnabled(False)
        self.btnDigsheetAbs.setCursor(Qt.CursorShape.ForbiddenCursor)
        self.btnDigsheetAbs.setToolTip("Create a project first to enable Digsheet generation.")
        return
    can_show = (
            self.project_is_open
            and isinstance(self.pipe_tally, pd.DataFrame)
            and _is_graph_tab_ok(self)
            and _has_valid_abs_selection(self)
    )
    self.btnDigsheetAbs.setEnabled(bool(can_show))

    if can_show:
        self.btnDigsheetAbs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnDigsheetAbs.setToolTip("Click to generate Digsheet for the selected Absolute Distance.")
    else:
        self.btnDigsheetAbs.setCursor(Qt.CursorShape.ForbiddenCursor)
        self.btnDigsheetAbs.setToolTip("Select an Absolute Distance cell in the table below to enable.")

def _abs_col_candidates(self):
    return ("Absolute_Distance", "Abs. Distance (m)", "Absolute Distance")


def _abs_col_index_silent(self) -> Optional[int]:
    tw = self.ui.tableWidgetDefect
    if tw.columnCount() == 0:
        return None
    for c in range(tw.columnCount()):
        hdr = tw.horizontalHeaderItem(c)
        name = hdr.text().strip() if hdr else ""
        if name in _abs_col_candidates(self):
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

    abs_col = _abs_col_index_silent(self)
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


