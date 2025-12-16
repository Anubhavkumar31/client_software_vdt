
from main_window.components.create_buttons.button_connections import open_digsheet_by_abs_from_selection, \
    load_selected_pipe, open_column_filter_dialog, ondropdowntabchanged, _toggle_table_visibility, _apply_heatmap_layout
from main_window.components.create_buttons.buttons.Load_btn import create_Load_btn
from main_window.components.create_buttons.buttons.digsheet_btn_main_ui import create_digsheet_btn
from main_window.components.create_buttons.buttons.filter_column_btn import create_filter_column_btn
from main_window.components.create_buttons.buttons.hide_show_table_btn import create_hide_show_table
from main_window.components.create_buttons.buttons.stack_horizontal_view_btn import create_stack_H_btn
from main_window.components.create_buttons.buttons.tab_switcher_dropdown import create_tabSwitcher_dropdown




def setup_buttons(self):
    """
    -------------------------------------------------------------
    BUTTONS + TOP CONTROL BAR
    -------------------------------------------------------------
    Creates all primary interactive controls:
      • Load Pipe button
      • Digsheet button
      • Filter Columns button
      • Tab-switch dropdown
      • Hide/Show table toggle
      • Horizontal/Vertical heatmap layout toggle

    Also connects each control to its proper callback function.
    All layout placement of these widgets happens inside the
    respective create_* helper functions.
    -------------------------------------------------------------
    """
    #digsheet
    self.btnDigsheetAbs = create_digsheet_btn(self)
    self.btnDigsheetAbs.clicked.connect(lambda : open_digsheet_by_abs_from_selection(self))

    #load button
    self.btnLoadPipe = create_Load_btn(self)
    self.btnLoadPipe.clicked.connect(lambda : load_selected_pipe(self))

    #filter column
    self.btnOpenFilterDlg = create_filter_column_btn(self)
    self.btnOpenFilterDlg.clicked.connect(lambda : open_column_filter_dialog(self))

    #tab switcher dropdown
    self.tabSwitcherDropdown = create_tabSwitcher_dropdown(self)
    self.tabSwitcherDropdown.currentIndexChanged.connect(lambda index: ondropdowntabchanged(self, index))

    #hide/show table
    self.btnToggleTable = create_hide_show_table(self)
    self.btnToggleTable.clicked.connect(lambda : _toggle_table_visibility(self))

    #stack/horizontal view
    self.btnToggleHmLayout = create_stack_H_btn(self)
    self.btnToggleHmLayout.clicked.connect(
        lambda : _apply_heatmap_layout(
            self,
            "vertical" if self._hm_layout_mode == "horizontal" else "horizontal"
        )
    )



