
import os
import subprocess
import sys
import tempfile
import uuid
import pandas as pd
from typing import Optional
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMessageBox, QDialog

from main_section_view.column_filter_worker import open_column_filter_dialog_con
from main_section_view.helpers_temp import _on_middle_tab_changed, apply_column_filter
from main_section_view.load_button_working import load_selected_by_index


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def _dump_tally_to_temp(df):
    import pickle
    p = os.path.join(tempfile.gettempdir(), f"pipe_tally_{uuid.uuid4().hex}.pkl")
    with open(p, "wb") as f: pickle.dump(df, f)
    return p



#digsheet button connection to open with abs. distance
def open_digsheet_by_abs_from_selection(self):
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

#load selected pipe button connection
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

#filter button connection
# def open_column_filter_dialog(self):
#     """Open column selector dialog and apply the result."""
#     headers = self._current_headers_for_filter()
#     locked = set(getattr(self, "BACKEND_LOCKED_COLS", set()))
#
#     # default: first time, select everything that's not locked
#     if not self._selected_columns:
#         checked = set(h for h in headers if h not in locked)
#     else:
#         checked = set(h for h in self._selected_columns if h in headers and h not in locked)
#
#     dlg = ColumnFilterDialog(headers=headers, checked=checked, locked=locked, parent=self)
#     if dlg.exec() != QDialog.DialogCode.Accepted:
#         return
#
#     # persist + apply (locked are always enforced)
#     self._selected_columns = set(dlg.selected_names()) | locked
#     apply_column_filter(self)
def open_column_filter_dialog(self):
    open_column_filter_dialog_con(self)

#tab switcher connection
def ondropdowntabchanged(self, index: int):
    """Handle tab changes from dropdown switcher"""
    # print("inside ondropdowntabchanged")
    if index >= 0:
        self.ui.tabWidgetM.blockSignals(True)
        self.mid_tabbar.blockSignals(True)

        self.ui.tabWidgetM.setCurrentIndex(index)
        self.mid_tabbar.setCurrentIndex(index)
        self.tabSwitcherDropdown.setCurrentIndex(index)

        self.ui.tabWidgetM.blockSignals(False)
        self.mid_tabbar.blockSignals(False)

        _on_middle_tab_changed(self, index)

#hide/show table button connection
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

#stack/horizontol button connection
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