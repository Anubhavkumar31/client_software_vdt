from PyQt6.QtCore import QTimer, Qt
from ui.graphs_ui import GraphApp
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMessageBox, QDialog


#used in setup_menu_actions.py inside main_window.components
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
        QTimer.singleShot(100, lambda: self._reset_splitter_ratio(0.45))

    self.tab_switcher2()
    self.update_digsheet_button_state()

def syncdropdownwithtabs(self, index: int):
    """Sync dropdown when tab changes from other sources"""
    try:
        if hasattr(self, 'tabSwitcherDropdown'):
            self.tabSwitcherDropdown.blockSignals(True)
            self.tabSwitcherDropdown.setCurrentIndex(index)
            self.tabSwitcherDropdown.blockSignals(False)
    except Exception as e:
        print(f"Error syncing dropdown: {e}")



#helper functions for _on_middle_tab_changed
def _reset_splitter_ratio(self, top_ratio: float = 0.6):
    """Force consistent top/bottom height ratio for the stack layout."""
    if not hasattr(self, "splitter"):
        return

    def apply_ratio():
        sizes = self.splitter.sizes()
        total = sum(sizes) if sizes else self.splitter.height()
        if total > 0:
            top = int(total * top_ratio)
            bottom = total - top
            self.splitter.setSizes([top, bottom])
            # optional debug
            print(f"[DEBUG] Splitter resized: top={top}, bottom={bottom}, total={total}")

    # 🔹 Delay the resize slightly so the layout stabilizes first
    QTimer.singleShot(120, apply_ratio)



#tab switcher button helpers in setup_buttons.py
# def ondropdowntabchanged(self, index: int):
#     """Handle tab changes from dropdown switcher"""
#     # print("inside ondropdowntabchanged")
#     if index >= 0:
#         self.ui.tabWidgetM.blockSignals(True)
#         self.mid_tabbar.blockSignals(True)
#
#         self.ui.tabWidgetM.setCurrentIndex(index)
#         self.mid_tabbar.setCurrentIndex(index)
#         self.tabSwitcherDropdown.setCurrentIndex(index)
#
#         self.ui.tabWidgetM.blockSignals(False)
#         self.mid_tabbar.blockSignals(False)
#
#         _on_middle_tab_changed(self, index)



# ---------- guarded connections for heatmap/line/3D / action_graphs----------
def _connect_guarded_graph_controls(self):
    a = self.ui
    # QActions from menu/toolbar
    action_map = [
        ("actionHeatmap", "Heatmap"),
        ("action_LineChart", "LineChart"),
        ("action_3D_Graph", "3D"),
    ]
    if hasattr(self.ui, "action_graphs"):
        self.ui.action_graphs.triggered.connect(lambda : open_graphs_window(self))

    for aname, tab in action_map:
        act = getattr(a, aname, None)
        if isinstance(act, QAction):
            try: act.triggered.disconnect()
            except Exception: pass
            act.triggered.connect(lambda _=False, t=tab: lambda t: _guarded_open_tab(self, t))

    # Buttons / toolbuttons
    widget_map = [
        ("btnHeatmap", "Heatmap"),
        ("toolButtonHeatmap", "Heatmap"),
        ("btnLinechart", "LineChart"),
        ("toolButtonLine", "LineChart"),
        ("btn3D", "3D"),
        ("toolButton3D", "3D"),
    ]
    for wname, tab in widget_map:
        w = getattr(a, wname, None)
        if w is not None and hasattr(w, "clicked"):
            try: w.clicked.disconnect()
            except Exception: pass
            w.clicked.connect(lambda _=False, t=tab: lambda t: _guarded_open_tab(self, t))


def _guarded_open_tab(self, tab_name: str):
    if not self.project_is_open:
        if self._ui_ready:
            self._project_required_popup()
        return
    wanted = {
        "Heatmap": {"Heatmap"},
        "LineChart": {"LineChart", "Line Chart", "Line Plot"},
        "3D": {"3D Graph", "3D"},
    }.get(tab_name, {tab_name})

    tw = self.ui.tabWidgetM
    for i in range(tw.count()):
        if tw.tabText(i) in wanted:
            tw.setCurrentIndex(i)
            self.tab_switcher2()
            return
    QMessageBox.information(self, "Tab not found", f"Could not locate tab: {tab_name}")


def open_graphs_window(self):
    if self.pipe_tally is None:
        QMessageBox.warning(self, "No Pipe Tally", "Please create or load a project first.")
        return

    if self._central_graphs is None:
        self._central_graphs = GraphApp(self.pipe_tally,self.project_root)
    self.setCentralWidget(self._central_graphs)



#filter column button helper functions
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
