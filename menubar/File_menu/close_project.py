from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QMessageBox

from main_window.components.helper_func import _update_project_actions


def close_project(self):
    try:
        # 1) Stop any secondary views / background loaders
        self._close_graphs_view()
        try:
            if getattr(self, "loader_worker", None) and self.loader_worker.isRunning():
                self.loader_worker.requestInterruption()
                self.loader_worker.quit()
                self.loader_worker.wait(1500)
        except Exception:
            pass
        self.loader_worker = None

        # 2) Block tab/dropdown signals while we reset widgets
        tw = getattr(self.ui, "tabWidgetM", None)
        if tw is not None:
            tw.blockSignals(True)
        if hasattr(self, "tabSwitcherDropdown"):
            self.tabSwitcherDropdown.blockSignals(True)

        # 3) Mark project closed + clear file lists/state
        self.project_is_open = False
        self.project_root = None
        self.pkl_files = []
        self.pipe_tally = None
        self.curr_data = None
        self.header_list = []
        self.hmap = self.hmap_r = self.lplot = self.lplot_r = self.pipe3d = self.heatmap_box = None
        self.hhmap = self.phmap = None
        self._selected_columns = set()

        # 4) Reset the "allowed tab" & guard flags so next project starts on Heatmap
        self._reverting_tab = False
        self._last_allowed_tab_index = 0  # 0 == Heatmap tab (middle stack)

        # 5) Reset combo + top controls
        cb = self.ui.comboBoxPipe
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("-Pipe-")
        cb.blockSignals(False)

        if hasattr(self, "btnLoadPipe"):    self.btnLoadPipe.setEnabled(False)
        if hasattr(self, "btnDigsheetAbs"): self.btnDigsheetAbs.setEnabled(False)

        # 6) Heatmap-specific UI back to defaults
        self._hm_layout_mode = "vertical"  # default = stacked
        try:
            self._apply_heatmap_layout("vertical")
        except Exception:
            pass

        self._table_hidden = True  # default label = "Show Table"
        if hasattr(self, "btnToggleTable"):
            self.btnToggleTable.setEnabled(False)
            self.btnToggleTable.setText("Show Table")
        if hasattr(self, "btnToggleHmLayout"):
            self.btnToggleHmLayout.setEnabled(False)
            # optional: set text according to your toggling semantics
            # self.btnToggleHmLayout.setText("Side-by-side")

        # 7) Tables and models
        try:
            self.model.clear()
        except Exception:
            pass
        if hasattr(self.ui, "tableWidgetDefect"):
            self.ui.tableWidgetDefect.clear()
            self.ui.tableWidgetDefect.hide()
        if hasattr(self, "table_scrollbar") and self.table_scrollbar:
            self.table_scrollbar.hide()

        # 8) Middle tab + dropdown back to Heatmap
        if tw is not None:
            try:
                tw.setCurrentIndex(0)  # Heatmap
            except Exception:
                pass
        if hasattr(self, "tabSwitcherDropdown"):
            self.tabSwitcherDropdown.setCurrentIndex(0)

        # 9) Web views / stacks / overlays
        try:
            self.web_view.setUrl(QUrl("about:blank"))
            self.web_view2.setUrl(QUrl("about:blank"))
        except Exception:
            pass
        if hasattr(self, "bottom_stack"):
            self.bottom_stack.setCurrentIndex(0)  # hide table pane
        self._show_watermark()
        self._toggle_plot_ui(False)

        if hasattr(self, "_select_pipe_container") and self._select_pipe_container:
            self._select_pipe_container.hide()
        if hasattr(self, "_no_defects_container") and self._no_defects_container:
            self._no_defects_container.hide()
        if hasattr(self, "_create_proj_container") and self._create_proj_container:
            self._create_proj_container.show()
        if hasattr(self, "btnOpenFilterDlg"):
            self.btnOpenFilterDlg.setEnabled(False)
        if hasattr(self, "tabSwitcherDropdown"):
            self.tabSwitcherDropdown.setEnabled(False)

        self._force_full_start_state()

        _update_project_actions(self)
        QMessageBox.information(self, "Project Closed", "The project has been successfully closed.")

    except Exception as e:
        self.open_Error(e)
    finally:
        # Re-enable signals
        if tw is not None:
            tw.blockSignals(False)
        if hasattr(self, "tabSwitcherDropdown"):
            self.tabSwitcherDropdown.blockSignals(False)
    self.ui.action_Pipe_Sch.setEnabled(False)