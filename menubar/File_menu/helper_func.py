from PyQt6.QtCore import QUrl


def _force_full_start_state(self):
    """Hard reset the UI to startup layout (Heatmap, table hidden, buttons off)."""
    # reset flags
    self._table_hidden = True
    self._hm_layout_mode = "vertical"
    self._last_allowed_tab_index = 0
    self._reverting_tab = False

    # top area → dual heatmap page
    if hasattr(self, "top_stack"):
        try:
            self.top_stack.setCurrentIndex(1)  # heatmap dual page
        except Exception:
            pass

    # hide bottom table area
    if hasattr(self, "bottom_stack"):
        self.bottom_stack.hide()
        self.bottom_stack.setCurrentIndex(0)

    # disable buttons
    if hasattr(self, "btnToggleTable"):
        self.btnToggleTable.setEnabled(False)
        self.btnToggleTable.setText("Show Table")
    if hasattr(self, "btnToggleHmLayout"):
        self.btnToggleHmLayout.setEnabled(False)
        self.btnToggleHmLayout.setText("Side-by-side")

    # reset middle tab and dropdown to Heatmap
    tw = getattr(self.ui, "tabWidgetM", None)
    if tw is not None:
        tw.blockSignals(True)
        tw.setCurrentIndex(0)
        tw.blockSignals(False)
    if hasattr(self, "tabSwitcherDropdown"):
        self.tabSwitcherDropdown.blockSignals(True)
        self.tabSwitcherDropdown.setCurrentIndex(0)
        self.tabSwitcherDropdown.blockSignals(False)

    # clear/blank out main web views
    for w in ("web_view", "web_view2", "web_view_left", "web_view_right"):
        if hasattr(self, w):
            getattr(self, w).setUrl(QUrl())
    self._show_watermark()