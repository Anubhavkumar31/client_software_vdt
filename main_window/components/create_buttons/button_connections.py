from main_window.components.main_section_view.workers.column_filter_worker import open_column_filter_dialog_con
from main_window.components.main_section_view.workers.digsheet_abs_worker import open_digsheet_by_abs_from_selection_con
from main_window.components.main_section_view.workers.load_button_working import load_selected_pipe_con
from main_window.components.main_section_view.workers.stack_hori_views import _apply_heatmap_layout_con
from main_window.components.main_section_view.workers.tab_switcher_dropdown import ondropdowntabchanged_con
from main_window.components.main_section_view.workers.table_data_worker import _toggle_table_visibility_con




#digsheet button connection to open with abs. distance
def open_digsheet_by_abs_from_selection(self):
    open_digsheet_by_abs_from_selection_con(self)


#load selected pipe button connection
def load_selected_pipe(self):
    load_selected_pipe_con(self)
    if hasattr(self, "btnToggleTable"):
        self.btnToggleTable.setEnabled(True)
        self.btnToggleTable.setText("Show Table")
    if hasattr(self, "btnToggleHmLayout"):
        self.btnToggleHmLayout.setEnabled(True)
        self.btnToggleHmLayout.setText("Side-by-side")


#filter button connection
def open_column_filter_dialog(self):
    open_column_filter_dialog_con(self)

#tab switcher connection
def ondropdowntabchanged(self, index: int):
    ondropdowntabchanged_con(self, index)

#hide/show table button connection
def _toggle_table_visibility(self):
    _toggle_table_visibility_con(self)


#stack/horizontol button connection
def _apply_heatmap_layout(self, mode: str = None):
    _apply_heatmap_layout_con(self, mode)

