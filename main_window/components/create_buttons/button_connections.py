
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
from main_section_view.digsheet_abs_worker import open_digsheet_by_abs_from_selection_con
from main_section_view.helpers_temp import _on_middle_tab_changed, apply_column_filter
from main_section_view.load_button_working import load_selected_by_index
from main_section_view.table_data_worker import _toggle_table_visibility_con


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
    open_digsheet_by_abs_from_selection_con(self)


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
    _toggle_table_visibility_con(self)


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