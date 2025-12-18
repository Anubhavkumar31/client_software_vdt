from PyQt6.QtCore import Qt, QTimer

#used by column_filter_worker and table_data_worker
from PyQt6.QtGui import QStandardItem
from PyQt6.QtWidgets import QAbstractItemView


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



