import os
import re

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMessageBox, QFileDialog


def open_project(self):
    try:
        # hide overlay immediately when trying to open
        if hasattr(self, "_create_proj_container") and self._create_proj_container:
            self._create_proj_container.hide()

        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly)
        dlg.setWindowTitle("Select Project Folder (PKLs + pipe_* folders)")
        if dlg.exec() != QFileDialog.DialogCode.Accepted:
            self.project_is_open = False
            self._toggle_plot_ui(False)
            self._show_watermark()
            self._update_project_actions()

            # show overlay back if user cancelled
            if hasattr(self, "_create_proj_container") and self._create_proj_container:
                self._create_proj_container.show()
            return

        root = dlg.selectedFiles()[0]
        self.project_root = root
        self._force_full_start_state()

        self.pipe_tally = None
        loaded_tally = self._auto_load_pipe_tally(root)
        if not loaded_tally:
            print("[pipe_tally] No tally file found in this project; graphs/reports will warn if needed.")

        pickle_data_dir = os.path.join(root, "pickle_data")
        if os.path.isdir(pickle_data_dir):
            self.pkl_files = [
                os.path.join(pickle_data_dir, f)
                for f in os.listdir(pickle_data_dir)
                if f.lower().endswith(".pkl")
            ]
        else:
            self.pkl_files = []
            print(f"[Warning] pickle_data directory not found in {root}")

        def nkey(path):
            filename = os.path.basename(path)
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", filename)]

        self.pkl_files.sort(key=nkey)

        cb = self.ui.comboBoxPipe
        cb.blockSignals(True)
        cb.clear()
        names = [os.path.splitext(os.path.basename(f))[0] for f in self.pkl_files]
        if names:
            cb.addItems(names)
            cb.setCurrentIndex(-1)
        else:
            cb.addItem("-Pipe-")  # 👈 nothing selected

        cb.lineEdit().setPlaceholderText("Type pipe number...")
        cb.completer().setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        cb.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        cb.blockSignals(False)

        try:
            cb.lineEdit().returnPressed.disconnect()
        except Exception:
            pass
        cb.lineEdit().returnPressed.connect(lambda : jump_to_number(self))

        if self.pkl_files:
            self.project_is_open = True
            _hide_create_project_message(self)
            self._toggle_plot_ui(True)
            _force_heatmap_start(self)
            # 🔹 Force-enable Heatmap control buttons since Heatmap is the first visible tab
            if hasattr(self, "btnToggleTable"):
                self.btnToggleTable.setEnabled(True)
                self.btnToggleTable.setText("Show Table")
            if hasattr(self, "btnToggleHmLayout"):
                self.btnToggleHmLayout.setEnabled(True)
                self.btnToggleHmLayout.setText("Side-by-side")

            # Show overlay instead of auto-loading
            self._show_select_pipe_message()

            # 👇 Force check so Load button activates if default pipe is already selected
            self.update_load_button_state(self.ui.comboBoxPipe.currentIndex())
        else:
            self.project_is_open = False
            self._toggle_plot_ui(False)
            self._show_watermark()
            QMessageBox.warning(self, "No PKLs", "No .pkl files found in the selected folder.")

            # show overlay back if no valid files
            if hasattr(self, "_create_proj_container") and self._create_proj_container:
                self._create_proj_container.show()

        self._update_project_actions()
    except Exception as e:
        self.project_is_open = False
        self._toggle_plot_ui(False)
        self._show_watermark()
        self._update_project_actions()

        # show overlay back on error
        if hasattr(self, "_create_proj_container") and self._create_proj_container:
            self._create_proj_container.show()

        self.open_Error(e)
    self.ui.action_Pipe_Sch.setEnabled(True)


def jump_to_number(self):
    if not self.project_is_open:
        return
    text = self.ui.comboBoxPipe.currentText().strip()
    if not text: return
    try:
        base_names = [os.path.splitext(os.path.basename(f))[0] for f in self.pkl_files]
        if text in base_names:
            idx = base_names.index(text)
        else:
            idx = next((i for i, n in enumerate(base_names) if re.search(rf'\b{text}\b', n)), None)
            if idx is None: return
        self.ui.comboBoxPipe.setCurrentIndex(idx)
    except Exception as e:
        self.open_Error(f"Jump error: {e}")


def _force_heatmap_start(self):
    """Ensure middle view opens on Heatmap before the next load."""
    self._last_allowed_tab_index = 0
    self._reverting_tab = False
    tw = getattr(self.ui, "tabWidgetM", None)
    if tw is not None:
        tw.blockSignals(True)
        tw.setCurrentIndex(0)
        tw.blockSignals(False)
    if hasattr(self, "tabSwitcherDropdown"):
        self.tabSwitcherDropdown.blockSignals(True)
        self.tabSwitcherDropdown.setCurrentIndex(0)
        self.tabSwitcherDropdown.blockSignals(False)


def _hide_create_project_message(self):
    if hasattr(self, '_create_proj_container'):
        self._create_proj_container.hide()