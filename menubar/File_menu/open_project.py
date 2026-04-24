import os
import re

import pandas as pd
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMessageBox, QFileDialog

from main_section_view.utils import _toggle_plot_ui
from main_window.components.helper_func import _update_project_actions
from menubar.File_menu.helper_func import _force_full_start_state


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
            _toggle_plot_ui(self, False)
            self._show_watermark()
            _update_project_actions(self)

            # show overlay back if user cancelled
            if hasattr(self, "_create_proj_container") and self._create_proj_container:
                self._create_proj_container.show()
            return

        root = dlg.selectedFiles()[0]
        self.project_root = root
        _force_full_start_state(self)

        self.pipe_tally = None
        loaded_tally, self.pipetally_dir = _auto_load_pipe_tally(self, root)
        print(f"pipetally path : {self.pipetally_dir}")
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
            _toggle_plot_ui(self, True)
            _force_heatmap_start(self)
            # 🔹 Force-enable Heatmap control buttons since Heatmap is the first visible tab
            if hasattr(self, "btnToggleTable"):
                self.btnToggleTable.setEnabled(True)
                self.btnToggleTable.setText("Show Table")
            if hasattr(self, "btnToggleHmLayout"):
                self.btnToggleHmLayout.setEnabled(True)
                self.btnToggleHmLayout.setText("Side-by-side")

            # Show overlay instead of auto-loading
            _show_select_pipe_message(self)

            # 👇 Force check so Load button activates if default pipe is already selected
            self.update_load_button_state(self.ui.comboBoxPipe.currentIndex())
        else:
            self.project_is_open = False
            _toggle_plot_ui(self, False)
            self._show_watermark()
            QMessageBox.warning(self, "No PKLs", "No .pkl files found in the selected folder.")

            # show overlay back if no valid files
            if hasattr(self, "_create_proj_container") and self._create_proj_container:
                self._create_proj_container.show()

        _update_project_actions(self)
    except Exception as e:
        self.project_is_open = False
        _toggle_plot_ui(self, False)
        self._show_watermark()
        _update_project_actions(self)

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



def _auto_load_pipe_tally(self, root: str):
    import os
    import pandas as pd

    pipetally_dir = os.path.join(root, "pipetally_main")

    if not os.path.isdir(pipetally_dir):
        print(f"[Error] pipetally_main directory not found in {root}")
        self.pipe_tally = None
        return False, None

    # ✅ Step 1: find only files starting with pipe_tally
    candidates = []
    for f in os.listdir(pipetally_dir):
        name = f.lower()

        if name.startswith("pipetally_main") and name.endswith((".xlsx", ".xls", ".csv")):
            candidates.append(os.path.join(pipetally_dir, f))

    print("PIPE TALLY CANDIDATES:", candidates)

    # ❌ Step 2: no file found
    if len(candidates) == 0:
        self.pipe_tally = None
        msg = "No pipe_tally file found in pipetally_main folder."
        print("[ERROR]", msg)
        self.open_Error(msg)
        return False, None

    # ❌ Step 3: more than one found
    if len(candidates) > 1:
        self.pipe_tally = None
        msg = "Multiple pipe_tally files found. Only one is allowed in pipetally_main."
        print("[ERROR]", msg)
        self.open_Error(msg)
        return False, None

    # ✅ Step 4: exactly one file
    path = candidates[0]

    try:
        print(f"[LOADING] {os.path.basename(path)}")

        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        df.columns = [str(c).strip() for c in df.columns]

        # ✅ optional: numeric cleanup
        numeric_columns = [
            'Depth %', 'Depth (mm)', 'ERF (ASME B31G)', 'Psafe (ASME B31G) Barg',
            'Abs. Distance (m)', 'Distance to U/S GW(m)', 'Length (mm)',
            'Width (mm)', 'WT (mm)', 'Pipe Length (mm)'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(3)

        self.pipe_tally = df

        print(f"[SUCCESS] Loaded {os.path.basename(path)}")
        return True, path

    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        self.open_Error(f"Failed to load pipe_tally file:\n{e}")
        self.pipe_tally = None
        return False, None



def _show_select_pipe_message(self):
    if hasattr(self, "_select_pipe_container"):
        central = self.centralWidget().rect()

        # Leave space for the pipe selection row (comboBox + Load button)
        header_height = self.ui.comboBoxPipe.height() + 20

        self._select_pipe_container.setGeometry(
            0,
            header_height,
            central.width(),
            central.height() - header_height
        )
        self._select_pipe_container.show()

    # Hide other views
    if hasattr(self.ui, "tableWidgetDefect"):
        self.ui.tableWidgetDefect.hide()
    if hasattr(self.ui, "tableView"):
        self.ui.tableView.hide()

    self.btnLoadPipe.setEnabled(False)