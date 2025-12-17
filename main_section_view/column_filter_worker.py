from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QDialog


class ColumnFilterDialog(QDialog):
    def __init__(self, *, headers: list[str], checked: set[str], locked: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Columns")
        self.setModal(True)
        self.resize(420, 520)

        self._locked = set(locked)
        # only show headers that are NOT locked
        visible_headers = [h for h in headers if h not in self._locked]

        # widgets
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLineEdit, QListView, QPushButton, QLabel
        from PyQt6.QtGui import QStandardItemModel, QStandardItem
        from PyQt6.QtCore import Qt, QSortFilterProxyModel

        lay = QVBoxLayout(self)

        # search
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search columns…")
        lay.addWidget(self.search)

        # list (checkable)
        self.model = QStandardItemModel(self)
        for name in visible_headers:
            it = QStandardItem(name)
            it.setCheckable(True)
            it.setCheckState(Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked)
            it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.model.appendRow(it)

        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(0)
        self.proxy.setSourceModel(self.model)

        self.view = QListView(self)
        self.view.setModel(self.proxy)
        self.view.setEditTriggers(QListView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.view, 1)

        # quick actions
        row = QHBoxLayout()
        self.btnAll = QPushButton("Select All")
        self.btnNone = QPushButton("Select None")
        row.addWidget(self.btnAll)
        row.addWidget(self.btnNone)
        row.addStretch(1)
        lay.addLayout(row)

        # footer
        foot = QHBoxLayout()
        self.info = QLabel("")  # shows e.g. "12 selected"
        foot.addWidget(self.info)
        foot.addStretch(1)
        self.btnCancel = QPushButton("Cancel")
        self.btnApply = QPushButton("Apply")
        foot.addWidget(self.btnCancel)
        foot.addWidget(self.btnApply)
        lay.addLayout(foot)

        # wire up
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.btnAll.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        self.btnNone.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        self.btnCancel.clicked.connect(self.reject)
        self.btnApply.clicked.connect(self.accept)

        self._update_info()
        self.model.itemChanged.connect(lambda *_: self._update_info())

    def _set_all(self, state: Qt.CheckState):
        for r in range(self.model.rowCount()):
            self.model.item(r).setCheckState(state)
        self._update_info()

    def _update_info(self):
        total = self.model.rowCount()
        sel = sum(1 for r in range(total) if self.model.item(r).checkState() == Qt.CheckState.Checked)
        self.info.setText(f"{sel} / {total} visible columns selected")

    def selected_names(self) -> set[str]:
        """Return the names selected in the dialog (locked not included, they’re enforced by caller)."""
        out = set()
        for r in range(self.model.rowCount()):
            it = self.model.item(r)
            if it.checkState() == Qt.CheckState.Checked:
                out.add(it.text())
        return out

def open_column_filter_dialog_con(self):
    """Open column selector dialog and apply the result."""
    headers = self._current_headers_for_filter()
    locked = set(getattr(self, "BACKEND_LOCKED_COLS", set()))

    # default: first time, select everything that's not locked
    if not self._selected_columns:
        checked = set(h for h in headers if h not in locked)
    else:
        checked = set(h for h in self._selected_columns if h in headers and h not in locked)

    dlg = ColumnFilterDialog(headers=headers, checked=checked, locked=locked, parent=self)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    # persist + apply (locked are always enforced)
    self._selected_columns = set(dlg.selected_names()) | locked
    apply_column_filter(self)


def apply_column_filter(self):
    """Hide/show columns based on self._selected_columns + locked columns."""
    locked = set(getattr(self, "BACKEND_LOCKED_COLS", set()))

    # If we have no selection yet, treat as 'show all'
    if not self._selected_columns:
        self._selected_columns = set(self._current_headers_for_filter()) | locked

    names_to_keep = set(self._selected_columns) | locked

    # Prefer bottom QTableWidgetDefect if it has columns
    if hasattr(self.ui, "tableWidgetDefect") and self.ui.tableWidgetDefect.columnCount() > 0:
        header_map = {
            c: (self.ui.tableWidgetDefect.horizontalHeaderItem(c).text()
                if self.ui.tableWidgetDefect.horizontalHeaderItem(c) else f"Col {c}")
            for c in range(self.ui.tableWidgetDefect.columnCount())
        }
        for c, name in header_map.items():
            hide = (name not in names_to_keep) and (name not in locked)
            self.ui.tableWidgetDefect.setColumnHidden(c, hide)
        QTimer.singleShot(0, self._refresh_table_scrollbars)
        return

    # Fallback to the top QTableView
    if hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
        model = self.ui.tableView.model()
        header_names = [str(model.headerData(c, Qt.Orientation.Horizontal)) for c in range(model.columnCount())]
        for c, name in enumerate(header_names):
            hide = (name not in names_to_keep) and (name not in locked)
            self.ui.tableView.setColumnHidden(c, hide)