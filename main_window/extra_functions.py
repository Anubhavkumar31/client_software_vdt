# extras
from PyQt6.QtWidgets import QMessageBox

from pages.Report import Report
from pages.adminPanel import Admin_Panel


def open_Admin(self):
    self.ap = Admin_Panel();
    self.ap.show()


def gen_data(self):
    try:
        if 'genData' not in self.child_windows or not self.child_windows['genData'].isVisible():
            self.script_runner_window = ScriptRunnerApp()
            self.script_runner_window.show()
            self.child_windows['genData'] = self.script_runner_window
        else:
            self.child_windows['genData'].raise_()
            self.child_windows['genData'].activateWindow()
    except Exception as e:
        self.open_Error(e)


def open_Report(self):
    cols = [r"Abs. Distance (m)", r"Depth %", r"Type", r"ERF (ASME B31G)", r"Orientation o' clock"]
    if not isinstance(self.pipe_tally, pd.DataFrame):
        QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.");
        return
    for c in cols:
        if c not in self.pipe_tally.columns:
            QMessageBox.critical(self, "Error", f"Missing column: {c}");
            return
    fil = self.pipe_tally[cols].copy()
    fil = fil.dropna(subset=["Abs. Distance (m)"])
    fil["Abs. Distance (m)"] = fil["Abs. Distance (m)"].astype(int)
    fil["Depth %"] = pd.to_numeric(fil["Depth %"], errors='coerce')
    fil["Type"] = fil["Type"].astype(str)
    fil["ERF (ASME B31G)"] = pd.to_numeric(fil["ERF (ASME B31G)"], errors='coerce')
    fil[r"Orientation o' clock"] = fil[r"Orientation o' clock"].astype(str)
    fil["Surface Location"] = fil["Type"].apply(
        lambda x: "Internal" if "Internal" in x else ("External" if "External" in x else "Unknown")
    )
    self.fr = Report(fil);
    self.fr.show()


def open_Assessment(self):
    Assess_Dialog().exec()


def open_Cluster(self):
    Cluster_Dialog().exec()


def open_CMLD(self):
    selected_columns = [r"Abs. Distance (m)", r"Type", r"Orientation o' clock"]
    if not isinstance(self.pipe_tally, pd.DataFrame):
        QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.")
        return
    for col in selected_columns:
        if col not in self.pipe_tally.columns:
            QMessageBox.critical(self, "Error", f"Missing column: {col}")
            return
    fil_tally = self.pipe_tally[selected_columns].copy()
    try:
        fil_tally["Abs. Distance (m)"] = fil_tally["Abs. Distance (m)"].astype(int)
        fil_tally["Type"] = fil_tally["Type"].astype(str)
        fil_tally[r"Orientation o' clock"] = fil_tally[r"Orientation o' clock"].astype(str)

        self.m3 = Main03Tab(fil_tally)
        self.m3.setWindowTitle("Circumferential Metal Loss Distribution")
        self.m3.resize(1285, 913)
        self.m3.show()
    except Exception as e:
        self.open_Error(e)


def open_DBAD(self):
    selected_columns = [r"Abs. Distance (m)", r"Depth %", r"Type"]
    if not isinstance(self.pipe_tally, pd.DataFrame):
        QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.")
        return
    for col in selected_columns:
        if col not in self.pipe_tally.columns:
            QMessageBox.critical(self, "Error", f"Missing column: {col}")
            return
    fil_tally = self.pipe_tally[selected_columns].copy()
    try:
        fil_tally["Abs. Distance (m)"] = fil_tally["Abs. Distance (m)"].astype(int)
        fil_tally["Depth %"] = pd.to_numeric(fil_tally["Depth %"], errors='coerce')
        fil_tally["Type"] = fil_tally["Type"].astype(str)

        self.m2 = Main02Tab(fil_tally)
        self.m2.setWindowTitle("Depth Based Anomalies Distribution")
        self.m2.resize(1285, 913)
        self.m2.show()
    except Exception as e:
        self.open_Error(e)


def open_EAD(self):
    selected_columns = [r"Abs. Distance (m)", r"Type", r"ERF (ASME B31G)"]
    if not isinstance(self.pipe_tally, pd.DataFrame):
        QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.")
        return
    for col in selected_columns:
        if col not in self.pipe_tally.columns:
            QMessageBox.critical(self, "Error", f"Missing column: {col}")
            return
    fil_tally = self.pipe_tally[selected_columns].copy()
    try:
        fil_tally["Abs. Distance (m)"] = fil_tally["Abs. Distance (m)"].astype(int)
        fil_tally["Type"] = fil_tally["Type"].astype(str)
        fil_tally["ERF (ASME B31G)"] = pd.to_numeric(fil_tally["ERF (ASME B31G)"], errors='coerce')

        self.m1 = Main01Tab(fil_tally)
        self.m1.setWindowTitle("ERF Based Anomalies Distribution")
        self.m1.resize(1285, 913)
        self.m1.show()
    except Exception as e:
        self.open_Error(e)


def add_plot_custom(self):
    try:
        self.cplot_widget = customPlot(self.header_list)
        self.ui.graphLayout.addWidget(self.cplot_widget)
        self.cplot_widget.closeCustom.clicked.connect(self.cplot_widget.close_window)
        self.cplot_widget.comboBox.currentIndexChanged.connect(self.plot_c)
    except Exception as e:
        self.open_Error(e)


def add_plot_tele(self):
    try:
        if self.curr_data is None or self.curr_data.empty:
            QMessageBox.critical(self, "Error", "Please load a project first.");
            return
        import re as _re
        tlist = [c for c in self.header_list if _re.match(r'^F\d+', c)]
        if not tlist:
            QMessageBox.warning(self, "No Telemetry Data", "No telemetry (F...) columns found.");
            return
        self.tplot_widget = telePlot(tlist)
        self.ui.graphLayout.addWidget(self.tplot_widget)
        self.tplot_widget.closeTele.clicked.connect(self.tplot_widget.close_window)
        self.tplot_widget.checkBox.stateChanged.connect(self.magnetisation)
        self.tplot_widget.checkBox_2.stateChanged.connect(self.velocity)
        self.tplot_widget.comboBox.currentIndexChanged.connect(self.plot_telemetry)
        if len(tlist) > 0:
            self.tplot_widget.comboBox.setCurrentIndex(1)
            self.plot_telemetry()
    except Exception as e:
        self.open_Error(e)


def add_plot_ad(self):
    try:
        self.adplot_widget = adPlot(self.curr_data if isinstance(self.curr_data, list) else self.curr_data)
        self.ui.graphLayout.addWidget(self.adplot_widget)
        self.adplot_widget.closeAnamoly.clicked.connect(self.adplot_widget.close_window)
    except Exception as e:
        self.open_Error(e)


def draw_boxes_v2(self):
    if not self.project_is_open:
        return
    try:
        if self.heatmap_box and os.path.exists(self.heatmap_box):
            self.web_view.setUrl(QUrl.fromLocalFile(self.heatmap_box))
        else:
            self.open_Error("Boxed heatmap not found for the selected pipe.")
    except Exception as e:
        self.open_Error(e)


def plot_c(self):
    try:
        y_label = self.cplot_widget.comboBox.currentText()
        x_label = self.cplot_widget.comboBox_2.currentText()
        if x_label not in self.curr_data or y_label not in self.curr_data:
            raise ValueError("Selected labels are not in the current data.")
        x_data = self.curr_data[x_label];
        y_data = self.curr_data[y_label]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=y_label))
        fig.update_layout(title=f'{y_label} vs {x_label}', xaxis_title=x_label, yaxis_title=y_label, height=450)
        fp = resource_path('backend/files/customplot.html');
        fig.write_html(fp)
        self.cplot_widget.webviewCustom.setUrl(QUrl.fromLocalFile(fp))
        self.web_view.setUrl(QUrl.fromLocalFile(fp))
    except Exception as e:
        self.open_Error(e)


def plot_telemetry(self):
    try:
        param = self.tplot_widget.comboBox.currentText()
        if param == "-Select-" or param not in self.curr_data.columns: return
        filtered = [c for c in self.curr_data.columns if c.startswith('F')]
        tele = self.curr_data[filtered]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tele.index, y=tele[param], mode='lines', name=param))
        fig.update_layout(title=f'Telemetry Plot for {param}', xaxis_title='Counter', yaxis_title=param, height=450)
        fp = resource_path("telemetryplot.html");
        fig.write_html(fp)
        self.tplot_widget.webviewTele.setUrl(QUrl.fromLocalFile(fp))
        self.web_view.setUrl(QUrl.fromLocalFile(fp))
    except Exception as e:
        self.open_Error(e)


def magnetisation(self):
    try:
        if not self.tplot_widget.checkBox.isChecked():
            fp = resource_path('backend/files/telemetryplot.html')
            go.Figure().write_html(fp)
        else:
            filtered = [c for c in self.curr_data.columns if c.startswith('F')]
            tele = self.curr_data[filtered]
            mag = tele.mean(axis=1) * 0.0004854
            x = self.curr_data['ODDO1'];
            y = mag
            fig = go.Figure();
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Mag'))
            fig.update_layout(title='Magnetisation View', xaxis_title='Oddometer (mm)', yaxis_title='Magnetisation',
                              height=450)
            fp = resource_path('backend/files/magnetisation.html')
            fig.write_html(fp)
        self.tplot_widget.webviewTele.setUrl(QUrl.fromLocalFile(fp))
        self.web_view.setUrl(QUrl.fromLocalFile(fp))
    except Exception as e:
        self.open_Error(e)


def velocity(self):
    try:
        if not self.tplot_widget.checkBox_2.isChecked():
            fp = resource_path('backend/files/telemetryplot.html')
            go.Figure().write_html(fp)
        else:
            oddo = self.curr_data['ODDO1'].to_numpy()
            vel = [(oddo[i + 1] - oddo[i]) / 0.000666667 for i in range(len(oddo) - 1)]
            if vel: vel.append(vel[-1])
            fig = go.Figure();
            fig.add_trace(go.Scatter(x=oddo, y=vel, mode='lines', name='Velocity'))
            fig.update_layout(title='Velocity View', xaxis_title='Oddometer(mm)', yaxis_title='Velocity', height=450)
            fp = resource_path('backend/files/velocity.html');
            fig.write_html(fp)
        self.tplot_widget.webviewTele.setUrl(QUrl.fromLocalFile(fp))
        self.web_view.setUrl(QUrl.fromLocalFile(fp))
    except Exception as e:
        self.open_Error(e)


# might be extras
def _setup_left_vertical_scrollbar_sync(self):
    """Sync the custom left vertical scrollbar with tableWidgetDefect's internal vbar."""
    tw = self.ui.tableWidgetDefect
    inner_vbar = tw.verticalScrollBar()  # still exists even if hidden
    left_vbar = self.left_vscrollbar

    # Mirror range/page/single step from the table's scrollbar
    def _apply_range():
        left_vbar.blockSignals(True)
        left_vbar.setRange(inner_vbar.minimum(), inner_vbar.maximum())
        left_vbar.setPageStep(inner_vbar.pageStep())
        left_vbar.setSingleStep(inner_vbar.singleStep())
        left_vbar.setValue(inner_vbar.value())
        left_vbar.blockSignals(False)

    # When user drags the left bar -> scroll table
    def _on_left_changed(v):
        inner_vbar.setValue(v)

    # When table scrolls (keyboard, wheel, selection, data fill, etc.) -> move left bar
    def _on_inner_changed(v):
        left_vbar.blockSignals(True)
        left_vbar.setValue(v)
        left_vbar.blockSignals(False)

    def _on_inner_range_changed(_min, _max):
        _apply_range()

    # Connect both ways
    left_vbar.valueChanged.connect(_on_left_changed)
    inner_vbar.valueChanged.connect(_on_inner_changed)
    inner_vbar.rangeChanged.connect(_on_inner_range_changed)

    # Initial apply on next tick (table might not have full range yet)
    QTimer.singleShot(0, _apply_range)


def open_Ptal(self):
    try:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Pipe Tally File", "", "CSV/Excel Files (*.csv *.xlsx *.xls);;All Files (*)"
        )
        if not file_path: return
        self.pipe_tally = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
        QMessageBox.information(self, "Pipe Tally", "Pipe tally loaded successfully.")
        self._toggle_plot_ui(self.project_is_open)
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Pipe tally load failed: {e}")


def minimize_tabs(self):
    self.ui.tabWidgetM.hide()


def maximize_tabs(self):
    self.ui.tabWidgetM.show()


def toggletablevisibility(self):
    """Toggle table visibility in heatmap view only"""
    # Only work in Heatmap tab
    current_tab = self.ui.tabWidgetM.tabText(self.ui.tabWidgetM.currentIndex()).strip()
    if current_tab != "Heatmap":
        QMessageBox.information(self, "Heatmap Only",
                                "Table toggle only works in Heatmap view.")
        return

    # Toggle the flag
    self._table_hidden = not self._table_hidden

    # Hide or show the bottom section (table area)
    if self._table_hidden:
        self.bottom_stack.hide()
        self.btnToggleTable.setText("Show Table")
    else:
        self.bottom_stack.show()
        # self.ui.tableWidgetDefect.setMinimumHeight(250)
        # self.bottom_stack.setMinimumHeight(250)

        self.btnToggleTable.setText("Hide Table")

    print(f"Table visibility toggled: {'Hidden' if self._table_hidden else 'Shown'}")


def on_combo_index_changed(self, combo_idx: int):
    if not self.project_is_open or combo_idx < 0:
        return
    self.load_selected_by_index(combo_idx)


# def populate_column_filter(self, df: pd.DataFrame):
#     """Fill dropdown with all DataFrame columns (checkable)."""
#     model = self.columnFilter.model()
#     model.clear()
#
#     for col in df.columns:
#         it = QStandardItem(str(col))
#         # Make it user-checkable and enabled
#         it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
#         it.setData(Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
#         model.appendRow(it)
#
#     # Update summary (e.g., "12 selected")
#     self._column_summary_text()
#
# def _column_summary_text(self):
#     """Show 'N selected' in the combobox line edit."""
#     m = self.columnFilter.model()
#     checked = sum(1 for i in range(m.rowCount()) if m.item(i).checkState() == Qt.CheckState.Checked)
#     if self.columnFilter.isEditable() and self.columnFilter.lineEdit():
#         self.columnFilter.lineEdit().setText(f"{checked} selected" if checked else "None")

def set_loading(self, msg="Loading"):
    self.current_message = msg
    self.statusBar().showMessage(f'           Status:      {self.current_message}')
    self._t0 = time.time()
    self.timer.start(100)


def set_idle(self):
    self.current_message = 'App running'
    self.statusBar().showMessage(f'           Status:      {self.current_message}')
    self.timer.stop()
    self._t0 = None
    self.right_status_label.setText("0.0s")


def _restore_all_columns(self):
    """Show all columns again (useful when closing a project)."""
    if hasattr(self.ui, "tableWidgetDefect"):
        for c in range(self.ui.tableWidgetDefect.columnCount()):
            self.ui.tableWidgetDefect.setColumnHidden(c, False)
    if hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
        model = self.ui.tableView.model()
        for c in range(model.columnCount()):
            self.ui.tableView.setColumnHidden(c, False)

# def _reset_ui_to_start_state(self):
#     # mark app state
#     self.project_is_open = False
#
#     # clear data/paths
#     for attr in [
#         "curr_data", "pipe_tally", "hmap", "hmap_r", "heatmap_box",
#         "lplot", "lplot_r", "pipe3d", "prox_linechart", "hhmap", "phmap"
#     ]:
#         setattr(self, attr, None)
#     self.pkl_files = []
#     self.project_root = None
#
#     # combo + load
#     cb = self.ui.comboBoxPipe
#     cb.blockSignals(True)
#     cb.clear(); cb.addItem("-Pipe-"); cb.setCurrentIndex(0)
#     cb.blockSignals(False)
#     self.btnLoadPipe.setEnabled(False)
#
#     # tables
#     try:
#         self.ui.tableWidgetDefect.clear()
#         self.ui.tableWidgetDefect.setRowCount(0)
#         self.ui.tableWidgetDefect.setColumnCount(0)
#         self.ui.tableWidgetDefect.hide()
#     except Exception:
#         pass
#
#     # bottom area
#     self._table_hidden = True
#     if hasattr(self, "btnToggleTable"):
#         self.btnToggleTable.setText("Show Table")
#         self.btnToggleTable.setEnabled(False)
#     if hasattr(self, "bottom_stack"):
#         self.bottom_stack.hide()
#
#     # top area → back to startup (single page + watermark)
#     try:
#         if hasattr(self, "top_stack"):
#             self.top_stack.setCurrentIndex(0)   # single_chart_page
#         # blank any old heatmaps / prox views
#         for wname in ("web_view_left", "web_view_right", "web_view2"):
#             if hasattr(self, wname):
#                 getattr(self, wname).setUrl(QUrl())
#         # show startup watermark in main web view
#         self._show_watermark()
#     except Exception:
#         pass
#
#     # disable heatmap layout toggle & dropdown until a project opens
#     if hasattr(self, "btnToggleHmLayout"):
#         self.btnToggleHmLayout.setEnabled(False)
#     if hasattr(self, "tabSwitcherDropdown"):
#         self.tabSwitcherDropdown.setCurrentIndex(0)
#         self.tabSwitcherDropdown.setEnabled(False)
#
#     # disable graph tabs and update menu actions
#     self._toggle_plot_ui(False)
#     self._update_project_actions()
#
#     # show the “Create Project” overlay again
#     if hasattr(self, "_show_create_project_message"):
#         self._show_create_project_message()
#
#     # reset scroll sync guards
#     self._hscroll_ready = False
#     self._hscroll_ready_main = False
#     self._hscroll_ready_table = False


# def on_row_selection_changed(self, *_):
#     idxs = self.ui.tableWidgetDefect.selectionModel().selectedRows()
#     if not idxs:
#         self.update_digsheet_button_state()
#         return
#     row = idxs[0].row()
#     item = self.ui.tableWidgetDefect.item(row, 0)
#     if item:
#         defect_id = item.text()
#         try:
#             self.web_view.page().runJavaScript(f"highlightBox({defect_id});")
#         except Exception:
#             pass
#     self.update_digsheet_button_state()


# def _currently_checked_in_dropdown(self) -> set[str]:
#     """Read the check state from the existing dropdown (_cf_model)."""
#     out = set()
#     for r in range(self._cf_model.rowCount()):
#         it = self._cf_model.item(r)
#         if it.checkState() == Qt.CheckState.Checked:
#             out.add(it.text())
#     return out