from PyQt6.QtCore import QTimer

from main_section_view.helpers_temp import _reset_splitter_ratio, tab_switcher2
from main_section_view.utils import update_digsheet_button_state


def ondropdowntabchanged_con(self, index: int):
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

def _on_middle_tab_changed(self, index: int):
    # print("inside middole tab change ")
    if self._reverting_tab:
        return

    if not self.project_is_open:
        if self._ui_ready:
            self._project_required_popup()
        self._reverting_tab = True
        try:
            self.ui.tabWidgetM.setCurrentIndex(self._last_allowed_tab_index)
        finally:
            self._reverting_tab = False
        return

    self._last_allowed_tab_index = index

    # Get current tab name
    tab_text = self.ui.tabWidgetM.tabText(index).strip()
    # Fix: Switch the upper frame content correctly
    if hasattr(self, "top_stack"):
        if tab_text.lower() == "heatmap":
            # show the dual-heatmaps page
            self.top_stack.setCurrentWidget(self.dual_heatmaps_page)
        else:
            # show the single-chart page (for LineChart, 3D Graph, etc.)
            self.top_stack.setCurrentWidget(self.single_chart_page)

    # Always show table for LineChart and 3D Graph tabs
    if tab_text in {"LineChart", "Line Chart", "Line Plot", "3D Graph", "3D"}:
        self.bottom_stack.show()
        # Disable the toggle button for non-Heatmap tabs
        if hasattr(self, 'btnToggleTable'):
            self.btnToggleTable.setEnabled(False)
        if hasattr(self, "btnToggleHmLayout"):
            self.btnToggleHmLayout.setEnabled(False)
    # For Heatmap, respect the toggle flag
    elif tab_text == "Heatmap":
        if getattr(self, '_table_hidden', False):
            self.bottom_stack.hide()
        else:
            self.bottom_stack.show()
        # Enable the toggle button for Heatmap tab
        if hasattr(self, 'btnToggleTable'):
            self.btnToggleTable.setEnabled(True)
        if hasattr(self, "btnToggleHmLayout"):
            self.btnToggleHmLayout.setEnabled(True)
        QTimer.singleShot(100, lambda : _reset_splitter_ratio(self, 0.45))

    tab_switcher2(self)
    update_digsheet_button_state(self)

def syncdropdownwithtabs(self, index: int):
    """Sync dropdown when tab changes from other sources"""
    try:
        if hasattr(self, 'tabSwitcherDropdown'):
            self.tabSwitcherDropdown.blockSignals(True)
            self.tabSwitcherDropdown.setCurrentIndex(index)
            self.tabSwitcherDropdown.blockSignals(False)
    except Exception as e:
        print(f"Error syncing dropdown: {e}")