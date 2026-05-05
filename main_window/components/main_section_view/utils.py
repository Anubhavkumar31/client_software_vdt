import pandas as pd
from PyQt6.QtCore import Qt, QEvent

#used by column_filter_worker and table_data_worker
from PyQt6.QtGui import QStandardItem
from PyQt6.QtWidgets import QAbstractItemView, QMessageBox

from main_window.components.main_section_view.workers.digsheet_abs_worker import _is_graph_tab_ok, _has_valid_abs_selection



def _current_headers_for_filter(self) -> list[str]:
    """Mirror the same header source used by _refresh_column_filter_options()."""
    headers = []
    if hasattr(self.ui, "tableWidgetDefect") and self.ui.tableWidgetDefect.columnCount() > 0:
        headers = [
            (self.ui.tableWidgetDefect.horizontalHeaderItem(c).text()
             if self.ui.tableWidgetDefect.horizontalHeaderItem(c) else f"Col {c}")
            for c in range(self.ui.tableWidgetDefect.columnCount())
        ]
    elif hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
        model = self.ui.tableView.model()
        headers = [str(model.headerData(c, Qt.Orientation.Horizontal)) for c in range(model.columnCount())]
    return headers


def _refresh_table_scrollbars(self):
    """Comprehensive table scrollbar refresh after container resize"""
    try:
        # For tableWidgetDefect (QTableWidget)
        if hasattr(self.ui, 'tableWidgetDefect'):
            tw = self.ui.tableWidgetDefect
            # Force scroll mode and policy
            tw.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
            tw.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

            # Set scroll speed
            tw.verticalScrollBar().setSingleStep(15)

            # Force geometry updates
            tw.viewport().update()
            tw.updateGeometry()
            tw.resizeRowsToContents()

            # Force scrollbar range recalculation
            vsb = tw.verticalScrollBar()
            vsb.update()
            # Trigger a fake scroll to force range update
            current_val = vsb.value()
            vsb.setValue(min(current_val + 1, vsb.maximum()))
            vsb.setValue(current_val)

        # For tableView (QTableView with model)
        if hasattr(self.ui, 'tableView'):
            tv = self.ui.tableView
            tv.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
            tv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

            # Set scroll speed
            tv.verticalScrollBar().setSingleStep(15)

            tv.viewport().update()
            tv.updateGeometry()

            vsb = tv.verticalScrollBar()
            vsb.update()
            current_val = vsb.value()
            vsb.setValue(min(current_val + 1, vsb.maximum()))
            vsb.setValue(current_val)

    except Exception as e:
        print(f"Error refreshing table scrollbars: {e}")

def _refresh_column_filter_options(self):
    headers = []
    if hasattr(self.ui, "tableWidgetDefect") and self.ui.tableWidgetDefect.columnCount() > 0:
        headers = [
            (self.ui.tableWidgetDefect.horizontalHeaderItem(c).text()
            if self.ui.tableWidgetDefect.horizontalHeaderItem(c) else f"Col {c}")
            for c in range(self.ui.tableWidgetDefect.columnCount())
        ]
    elif hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
        model = self.ui.tableView.model()
        headers = [str(model.headerData(c, Qt.Orientation.Horizontal)) for c in range(model.columnCount())]

    self._cf_model.clear()
    for name in headers:
        if name in self.BACKEND_LOCKED_COLS:
            continue  # ← don't show in dropdown, but still exists in table
        it = QStandardItem(name)
        it.setCheckable(True)
        it.setCheckState(Qt.CheckState.Checked)
        it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._cf_model.appendRow(it)

    _update_column_summary(self)

def _update_column_summary(self):
    """Show 'All' / 'None' / 'N selected' in the combo line edit."""
    total = self._cf_model.rowCount()
    selected = sum(1 for r in range(total) if self._cf_model.item(r).checkState() == Qt.CheckState.Checked)
    if not self.columnFilter.isEditable() or not self.columnFilter.lineEdit():
        return
    if selected == 0:
        self.columnFilter.lineEdit().setText("None")
    elif selected == total:
        self.columnFilter.lineEdit().setText("All")
    else:
        self.columnFilter.lineEdit().setText(f"{selected} selected")



# ✅ Helper methods for showing/hiding message vs table
def _show_no_defects_message(self):
    try:
        if hasattr(self, '_no_defects_container'):
            self._no_defects_container.show()
        if hasattr(self.ui, 'tableWidgetDefect'):
            self.ui.tableWidgetDefect.clearSelection()
            self.ui.tableWidgetDefect.hide()
        if hasattr(self, 'table_scrollbar'):
            self.table_scrollbar.hide()

        if hasattr(self, 'left_vscrollbar'):
            self.left_vscrollbar.hide()

    except Exception as e:
        print(f"Error showing no defects message: {e}")


def _toggle_plot_ui(self, enabled: bool):
    tab_names = {"Heatmap", "LineChart", "Line Chart", "Line Plot", "3D Graph", "3D"}
    tw = self.ui.tabWidgetM
    for i in range(tw.count()):
        if tw.tabText(i) in tab_names:
            tw.setTabEnabled(i, enabled)
    try:
        self.update_digsheet_button_state()
    except Exception:
        pass


BACKEND_LOCKED_COLS = {"Empty"}  # for styling purpose this is takin extra ,DONT REMOVE IT FROM THE SET


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



def _show_disabled_digsheet_hint(self):
    QMessageBox.information(
        self,
        "Digsheet",
        "Please choose <b>Absolute Distance</b> from the defect table below to generate the digsheet."
    )

def _project_required_popup(self):
    QMessageBox.information(
        self,
        "Project Required",
        "Please create project before proceeding further."
    )

def _project_gate_targets(self):
    names = [
        "btnHeatmap", "btnLinechart", "btn3D",
        "toolButtonHeatmap", "toolButtonLine", "toolButton3D", "toolButtonXYZ",
    ]
    widgets = [self.btnDigsheetAbs]
    for n in names:
        w = getattr(self.ui, n, None)
        if w is not None:
            widgets.append(w)
    return [w for w in widgets if hasattr(w, "mapFromGlobal")]

def eventFilter(self, obj, ev):
    try:
        # Intercept mid tab bar clicks when no project (so repeated clicks also show popup)
        if obj is self.mid_tabbar and ev.type() == QEvent.Type.MouseButtonPress:
            if self._ui_ready and not self.project_is_open:
                self._project_required_popup()
                return True  # consume

        if ev.type() == QEvent.Type.MouseButtonPress:
            # PROJECT GATE for widget buttons
            if self._ui_ready and not self.project_is_open:
                if hasattr(ev, "globalPosition"):
                    gp = ev.globalPosition().toPoint()
                else:
                    gp = ev.globalPos()
                for w in self._project_gate_targets():
                    if w and w.isVisible():
                        local = w.mapFromGlobal(gp)
                        if w.rect().contains(local):
                            self._project_required_popup()
                            return True  # consume

            # DISABLED DIGSHEET HINT
            btn = getattr(self, "btnDigsheetAbs", None)
            if btn is not None and btn.isVisible() and not btn.isEnabled():
                if hasattr(ev, "globalPosition"):
                    gp = ev.globalPosition().toPoint()
                else:
                    gp = ev.globalPos()
                local = btn.mapFromGlobal(gp)
                if btn.rect().contains(local):
                    self._show_disabled_digsheet_hint()
                    return True  # consume
    except Exception:
        pass
    return super().eventFilter(obj, ev)


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


def update_load_button_state(self, idx: int):
    if not hasattr(self, "btnLoadPipe"):
        return

    text = self.ui.comboBoxPipe.currentText().strip()
    items = [self.ui.comboBoxPipe.itemText(i) for i in range(self.ui.comboBoxPipe.count())]

    # ✅ Enable Load if: a valid index OR a valid typed text
    if self.project_is_open and (idx >= 0 or text in items):
        self.btnLoadPipe.setEnabled(True)
        # ❌ Do NOT hide overlay here anymore
    else:
        self.btnLoadPipe.setEnabled(False)