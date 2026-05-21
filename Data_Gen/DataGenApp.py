# #
# #
# # import sys
# # import re
# # import os
# # from pathlib import Path
# # from PyQt6.QtCore import QThread, pyqtSignal
# # from PyQt6.QtWidgets import (
# #     QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
# #     QPushButton, QLabel, QFileDialog, QTextEdit, QScrollArea, QSizePolicy
# # )
# # import pandas as pd
# #
# # from defectS_creator import create_defectSheet_and_heatmap_box
# # from pipeTally_filter import create_pipe_tally
# # from html_filter import create_html_and_csv_from_pkl, WORKERS
# #
# #
# #
# #
# # class ScriptRunnerApp(QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.worker_displays = {}
# #         self.initUI()
# #         self.ptal_path = ''
# #         self.pkl_folder = ''
# #         self.output_folder = ''
# #
# #     def initUI(self):
# #         layout = QVBoxLayout()
# #
# #         # ── Path selectors ────────────────────────────────────────────────────
# #         self.ptal_label = QLabel('Select Pipe Tally file:')
# #         layout.addWidget(self.ptal_label)
# #
# #         self.ptal_button = QPushButton('Browse Pipe Tally')
# #         self.ptal_button.clicked.connect(self.browse_ptal)
# #         layout.addWidget(self.ptal_button)
# #
# #         self.pkl_label = QLabel('Select Pipes folder:')
# #         layout.addWidget(self.pkl_label)
# #
# #         self.pkl_button = QPushButton('Browse Pipes Folder')
# #         self.pkl_button.clicked.connect(self.browse_pkl_folder)
# #         layout.addWidget(self.pkl_button)
# #
# #         self.output_label = QLabel('Select Output folder:')
# #         layout.addWidget(self.output_label)
# #
# #         self.output_button = QPushButton('Browse Output Folder')
# #         self.output_button.clicked.connect(self.browse_output_folder)
# #         layout.addWidget(self.output_button)
# #
# #         # ── Run button ────────────────────────────────────────────────────────
# #         self.run_button = QPushButton('Run Scripts')
# #         self.run_button.clicked.connect(self.run_scripts)
# #         layout.addWidget(self.run_button)
# #
# #         # ── Main log (small, always visible at top) ───────────────────────────
# #         layout.addWidget(QLabel('Main Log:'))
# #         self.output_display = QTextEdit()
# #         self.output_display.setReadOnly(True)
# #         self.output_display.setMaximumHeight(100)
# #         layout.addWidget(self.output_display)
# #
# #         # ── Per-pipe panels inside a scroll area ──────────────────────────────
# #         layout.addWidget(QLabel('Per-Pipe Output:'))
# #
# #         self.worker_grid = QGridLayout()
# #         self.worker_grid.setSpacing(6)
# #
# #         grid_container = QWidget()
# #         grid_container.setLayout(self.worker_grid)
# #
# #         scroll = QScrollArea()
# #         scroll.setWidgetResizable(True)
# #         scroll.setWidget(grid_container)
# #         scroll.setMinimumHeight(300)
# #         layout.addWidget(scroll)
# #
# #         self.setLayout(layout)
# #         self.setWindowTitle('Script Runner')
# #         self.resize(1200, 800)
# #
# #     # ── Path browsing ─────────────────────────────────────────────────────────
# #     def browse_ptal(self):
# #         self.ptal_path, _ = QFileDialog.getOpenFileName(
# #             self, 'Select Pipe Tally file', '', 'Excel Files (*.xlsx)'
# #         )
# #         if self.ptal_path:
# #             self.ptal_label.setText(f'Selected Pipe Tally: {self.ptal_path}')
# #
# #     def browse_pkl_folder(self):
# #         self.pkl_folder = QFileDialog.getExistingDirectory(self, 'Select Pipes folder')
# #         if self.pkl_folder:
# #             self.pkl_label.setText(f'Selected Pipes Folder: {self.pkl_folder}')
# #
# #     def browse_output_folder(self):
# #         self.output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
# #         if self.output_folder:
# #             self.output_label.setText(f'Selected Output Folder: {self.output_folder}')
# #
# #     # ── Worker panels ─────────────────────────────────────────────────────────
# #     def setup_worker_panels(self, pkl_paths):
# #         # Clear previous panels
# #         for i in reversed(range(self.worker_grid.count())):
# #             item = self.worker_grid.itemAt(i)
# #             if item and item.widget():
# #                 item.widget().setParent(None)
# #         self.worker_displays.clear()
# #
# #         cols = min(len(pkl_paths), 4)   # max 4 columns
# #         for idx, pkl_path in enumerate(pkl_paths):
# #             pipe_name = Path(pkl_path).stem
# #
# #             panel = QTextEdit()
# #             panel.setReadOnly(True)
# #             panel.setMinimumHeight(220)
# #             panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
# #
# #             lbl = QLabel(f'🟡 Pipe {pipe_name}')
# #             lbl.setStyleSheet('font-weight: bold; font-size: 12px;')
# #
# #             container = QWidget()
# #             container.setStyleSheet('border: 1px solid #ccc; border-radius: 4px; padding: 2px;')
# #             v = QVBoxLayout(container)
# #             v.setContentsMargins(4, 4, 4, 4)
# #             v.addWidget(lbl)
# #             v.addWidget(panel)
# #
# #             row, col = divmod(idx, cols)
# #             self.worker_grid.addWidget(container, row, col)
# #             self.worker_displays[pipe_name] = (panel, lbl)
# #
# #     def _mark_pipe_done(self, pipe_id):
# #         if pipe_id in self.worker_displays:
# #             _, lbl = self.worker_displays[pipe_id]
# #             lbl.setText(f'✅ Pipe {pipe_id}')
# #             lbl.setStyleSheet('font-weight: bold; font-size: 12px; color: green;')
# #
# #     def _mark_pipe_error(self, pipe_id):
# #         if pipe_id in self.worker_displays:
# #             _, lbl = self.worker_displays[pipe_id]
# #             lbl.setText(f'❌ Pipe {pipe_id}')
# #             lbl.setStyleSheet('font-weight: bold; font-size: 12px; color: red;')
# #
# #     # ── Run ───────────────────────────────────────────────────────────────────
# #     def run_scripts(self):
# #         if not self.ptal_path or not self.pkl_folder or not self.output_folder:
# #             self.output_display.append(
# #                 "Please select all required paths before running the scripts.\n"
# #             )
# #             return
# #
# #         # Collect pkl paths and build panels BEFORE starting the thread
# #         pkl_paths = sorted([
# #             str(Path(self.pkl_folder) / f)
# #             for f in os.listdir(self.pkl_folder)
# #             if f.lower().endswith('.pkl')
# #         ])
# #         self.setup_worker_panels(pkl_paths)
# #
# #         self.worker = ScriptWorker(self.ptal_path, self.pkl_folder, self.output_folder)
# #         self.worker.output_signal.connect(self.update_output)
# #         self.worker.finished_signal.connect(self.on_scripts_finished)
# #
# #         self.output_display.append("Running scripts...\n")
# #         self.worker.start()
# #
# #     # ── Output routing ────────────────────────────────────────────────────────
# #     def update_output(self, message):
# #         # Always append to main log
# #         self.output_display.append(message)
# #         self.output_display.ensureCursorVisible()
# #
# #         # Route to the matching pipe panel
# #         match = re.search(r'pipe[_\s]*(\d+)', message, re.IGNORECASE)
# #         if match:
# #             pipe_id = match.group(1)
# #             if pipe_id in self.worker_displays:
# #                 panel, _ = self.worker_displays[pipe_id]
# #                 panel.append(message)
# #                 panel.ensureCursorVisible()
# #
# #                 # Update label colour based on keywords
# #                 lower = message.lower()
# #                 if any(k in lower for k in ('complete', 'finished', 'end', '✅')):
# #                     self._mark_pipe_done(pipe_id)
# #                 elif any(k in lower for k in ('error', 'crash', '❌')):
# #                     self._mark_pipe_error(pipe_id)
# #
# #     def on_scripts_finished(self):
# #         self.output_display.append("All scripts finished!\n")
# #
# #     def closeEvent(self, event):
# #         if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
# #             self.worker.terminate()
# #             self.worker.wait()
# #         event.accept()
# #
# #
# # import sys
# #
# # class StdoutRedirector:
# #     def __init__(self, signal_func):
# #         self.signal_func = signal_func
# #         self._original = sys.stdout
# #
# #     def write(self, text):
# #         if text.strip():
# #             self.signal_func(text.strip())
# #         self._original.write(text)
# #
# #     def flush(self):
# #         self._original.flush()
# #
# #
# # # ── Worker thread ─────────────────────────────────────────────────────────────
# # class ScriptWorker(QThread):
# #     output_signal = pyqtSignal(str)
# #     finished_signal = pyqtSignal()
# #
# #     def __init__(self, ptal_path, pkl_folder, output_folder):
# #         super().__init__()
# #         self.ptal_path = ptal_path
# #         self.pkl_folder = pkl_folder
# #         self.output_folder = output_folder
# #
# #     # def run(self):
# #     #     try:
# #     #         self.output_signal.emit("Running pipeTally_filter.py...\n")
# #     #         pdf = pd.read_excel(self.ptal_path)
# #     #         create_pipe_tally(
# #     #             pdf,
# #     #             output_folder=self.output_folder,
# #     #             output_callback=self.emit_output
# #     #         )
# #     #         self.output_signal.emit("pipeTally_filter.py finished.\n")
# #     #
# #     #         # self.output_signal.emit("Running defects_creator.py...\n")
# #     #         # create_defectSheet_and_heatmap_box(
# #     #         #     pkl_folder=self.pkl_folder,
# #     #         #     output_folder=self.output_folder,
# #     #         #     output_callback=self.emit_output
# #     #         # )
# #     #         # self.output_signal.emit("defectS_creator.py finished.\n")
# #     #
# #     #         self.output_signal.emit("Running html_filter.py...\n")
# #     #         create_html_and_csv_from_pkl(
# #     #             pkl_folder=self.pkl_folder,
# #     #             output_folder=self.output_folder,
# #     #             output_callback=self.emit_output
# #     #         )
# #     #         self.output_signal.emit("html_filter.py finished.\n")
# #     #
# #     #     except Exception as e:
# #     #         self.output_signal.emit(f"Error: {str(e)}\n")
# #     #     finally:
# #     #         self.finished_signal.emit()
# #     def run(self):
# #         redirector = StdoutRedirector(self.output_signal.emit)
# #         sys.stdout = redirector
# #         try:
# #             self.output_signal.emit("Running pipeTally_filter.py...\n")
# #             pdf = pd.read_excel(self.ptal_path)
# #             create_pipe_tally(
# #                 pdf,
# #                 output_folder=self.output_folder,
# #                 pkl_folder=self.pkl_folder,
# #                 output_callback=self.emit_output
# #             )
# #             self.output_signal.emit("pipeTally_filter.py finished.\n")
# #
# #             # self.output_signal.emit("Running defects_creator.py...\n")
# #             # create_defectSheet_and_heatmap_box(
# #             #     pkl_folder=self.pkl_folder,
# #             #     output_folder=self.output_folder,
# #             #     output_callback=self.emit_output
# #             # )
# #             # self.output_signal.emit("defectS_creator.py finished.\n")
# #
# #             self.output_signal.emit("Running html_filter.py...\n")
# #             create_html_and_csv_from_pkl(
# #                 pipetally_path = self.ptal_path,
# #                 pkl_folder=self.pkl_folder,
# #                 output_folder=self.output_folder,
# #                 output_callback=self.emit_output
# #             )
# #             self.output_signal.emit("html_filter.py finished.\n")
# #         finally:
# #             sys.stdout = redirector._original  # restore
# #             self.finished_signal.emit()
# #
# #     def emit_output(self, message):
# #         self.output_signal.emit(message)
# #
# #
# # if __name__ == '__main__':
# #     app = QApplication(sys.argv)
# #     runner = ScriptRunnerApp()
# #     runner.show()
# #     sys.exit(app.exec())
#
#
#
#
#
#
#
#
#
#
# import sys
# import re
# import os
# from pathlib import Path
# from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve
# from PyQt6.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QHBoxLayout,
#     QPushButton, QLabel, QFileDialog, QTextEdit, QFrame,
#     QSizePolicy, QGraphicsOpacityEffect
# )
# from PyQt6.QtGui import QFont, QColor
# import pandas as pd
#
# from defectS_creator import create_defectSheet_and_heatmap_box
# from pipeTally_filter import create_pipe_tally
# from html_filter import create_html_and_csv_from_pkl, WORKERS
#
#
# STYLE = """
# * { box-sizing: border-box; }
#
# QWidget {
#     background-color: #111318;
#     color: #c8cdd8;
#     font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
#     font-size: 12px;
# }
#
# /* ── Section labels ── */
# QLabel#section_label {
#     color: #3d4255;
#     font-size: 10px;
#     letter-spacing: 2px;
#     font-weight: bold;
#     text-transform: uppercase;
# }
#
# /* ── Path display ── */
# QLabel#path_label {
#     color: #7090b8;
#     font-size: 12px;
#     padding: 7px 12px;
#     background-color: #181b22;
#     border: 1px solid #1f2330;
#     border-radius: 3px;
# }
# QLabel#path_label[empty="true"] {
#     color: #2e3348;
# }
#
# /* ── Buttons ── */
# QPushButton {
#     background-color: #181b22;
#     color: #8896b0;
#     border: 1px solid #20242e;
#     border-radius: 3px;
#     padding: 7px 14px;
#     font-size: 11px;
#     letter-spacing: 0.5px;
# }
# QPushButton:hover {
#     background-color: #1e2230;
#     border-color: #3060a0;
#     color: #c0d4f0;
# }
# QPushButton:pressed {
#     background-color: #151820;
# }
#
# QPushButton#run_button {
#     background-color: #0d1e38;
#     color: #5090d0;
#     border: 1px solid #1a4070;
#     font-size: 12px;
#     font-weight: bold;
#     padding: 11px 28px;
#     letter-spacing: 2px;
#     border-radius: 3px;
#     min-width: 160px;
# }
# QPushButton#run_button:hover {
#     background-color: #102540;
#     border-color: #3070b0;
#     color: #80b8f0;
# }
# QPushButton#run_button:pressed {
#     background-color: #0a1828;
# }
# QPushButton#run_button:disabled {
#     background-color: #111318;
#     color: #1e2535;
#     border-color: #171b24;
# }
#
# /* ── Log ── */
# QTextEdit {
#     background-color: #0c0e13;
#     color: #8090a8;
#     border: 1px solid #181b22;
#     border-radius: 3px;
#     padding: 10px 12px;
#     font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
#     font-size: 11px;
#     selection-background-color: #1e3a5a;
#     line-height: 1.5;
# }
#
# /* ── Divider ── */
# QFrame#divider {
#     background-color: #181b22;
#     max-height: 1px;
# }
#
# /* ── Scrollbars ── */
# QScrollBar:vertical {
#     background: #0c0e13;
#     width: 6px;
#     border-radius: 3px;
# }
# QScrollBar::handle:vertical {
#     background: #252a38;
#     border-radius: 3px;
#     min-height: 24px;
# }
# QScrollBar::handle:vertical:hover { background: #303548; }
# QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
#
# /* ── Status dot ── */
# QLabel#status_dot {
#     font-size: 9px;
# }
# """
#
#
# # ── Animated status indicator ─────────────────────────────────────────────────
# class StatusIndicator(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         layout = QHBoxLayout(self)
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.setSpacing(8)
#
#         self._dot = QLabel('●')
#         self._dot.setObjectName('status_dot')
#         self._dot.setStyleSheet('color: #1e2535; font-size: 10px;')
#
#         self._text = QLabel('')
#         self._text.setStyleSheet('color: #3d4255; font-size: 11px; letter-spacing: 0.5px;')
#
#         layout.addWidget(self._dot)
#         layout.addWidget(self._text)
#         layout.addStretch()
#
#         self._opacity = QGraphicsOpacityEffect()
#         self._dot.setGraphicsEffect(self._opacity)
#         self._anim = QPropertyAnimation(self._opacity, b"opacity")
#         self._anim.setDuration(900)
#         self._anim.setStartValue(1.0)
#         self._anim.setEndValue(0.2)
#         self._anim.setEasingCurve(QEasingCurve.Type.SineCurve)
#         self._anim.setLoopCount(-1)
#
#     def set_idle(self):
#         self._anim.stop()
#         self._opacity.setOpacity(1.0)
#         self._dot.setStyleSheet('color: #1e2535; font-size: 10px;')
#         self._text.setStyleSheet('color: #3d4255; font-size: 11px;')
#         self._text.setText('')
#
#     def set_running(self, text='Running…'):
#         self._dot.setStyleSheet('color: #3a80d0; font-size: 10px;')
#         self._text.setStyleSheet('color: #4a7aaa; font-size: 11px; letter-spacing: 0.5px;')
#         self._text.setText(text)
#         self._anim.start()
#
#     def set_done(self):
#         self._anim.stop()
#         self._opacity.setOpacity(1.0)
#         self._dot.setStyleSheet('color: #3a9060; font-size: 10px;')
#         self._text.setStyleSheet('color: #4a8870; font-size: 11px; letter-spacing: 0.5px;')
#         self._text.setText('Completed')
#
#     def set_error(self):
#         self._anim.stop()
#         self._opacity.setOpacity(1.0)
#         self._dot.setStyleSheet('color: #c04050; font-size: 10px;')
#         self._text.setStyleSheet('color: #a04050; font-size: 11px; letter-spacing: 0.5px;')
#         self._text.setText('Error')
#
#
# # ── Main window ───────────────────────────────────────────────────────────────
# class ScriptRunnerApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.ptal_path = ''
#         self.pkl_folder = ''
#         self.output_folder = ''
#         self.initUI()
#
#     def initUI(self):
#         self.setStyleSheet(STYLE)
#         self.setWindowTitle('Pipeline Script Runner')
#         self.resize(860, 680)
#
#         root = QVBoxLayout(self)
#         root.setContentsMargins(32, 28, 32, 28)
#         root.setSpacing(0)
#
#         # ── Header ──────────────────────────────────────────────────────────
#         hdr_row = QHBoxLayout()
#         title = QLabel('PIPELINE RUNNER')
#         title.setStyleSheet(
#             'font-size: 15px; font-weight: bold; letter-spacing: 4px; color: #d0d8e8;'
#         )
#         subtitle = QLabel('tally  /  defects  /  html')
#         subtitle.setStyleSheet(
#             'font-size: 10px; color: #2a3045; letter-spacing: 1.5px; padding-left: 2px;'
#         )
#         hdr_col = QVBoxLayout()
#         hdr_col.setSpacing(2)
#         hdr_col.addWidget(title)
#         hdr_col.addWidget(subtitle)
#         hdr_row.addLayout(hdr_col)
#         hdr_row.addStretch()
#         root.addLayout(hdr_row)
#         root.addSpacing(24)
#         self._divider(root)
#         root.addSpacing(22)
#
#         # ── Path selectors ───────────────────────────────────────────────────
#         self.ptal_label, ptal_row   = self._path_row('PIPE TALLY FILE',  'Browse .xlsx')
#         self.pkl_label,  pkl_row    = self._path_row('PIPES FOLDER',      'Browse Folder')
#         self.out_label,  out_row    = self._path_row('OUTPUT FOLDER',     'Browse Folder')
#
#         # Wire up browse buttons (last widget in each row layout)
#         self._browse_btn(ptal_row).clicked.connect(self.browse_ptal)
#         self._browse_btn(pkl_row).clicked.connect(self.browse_pkl_folder)
#         self._browse_btn(out_row).clicked.connect(self.browse_output_folder)
#
#         paths = QVBoxLayout()
#         paths.setSpacing(14)
#         for row in (ptal_row, pkl_row, out_row):
#             w = QWidget()
#             w.setLayout(row)
#             paths.addWidget(w)
#         root.addLayout(paths)
#         root.addSpacing(26)
#         self._divider(root)
#         root.addSpacing(20)
#
#         # ── Run row ──────────────────────────────────────────────────────────
#         run_row = QHBoxLayout()
#         run_row.setSpacing(18)
#
#         self.run_button = QPushButton('▶  RUN')
#         self.run_button.setObjectName('run_button')
#         self.run_button.setFixedHeight(40)
#         self.run_button.clicked.connect(self.run_scripts)
#
#         self.status_ind = StatusIndicator()
#
#         run_row.addWidget(self.run_button)
#         run_row.addWidget(self.status_ind)
#         run_row.addStretch()
#         root.addLayout(run_row)
#         root.addSpacing(22)
#         self._divider(root)
#         root.addSpacing(14)
#
#         # ── Log ──────────────────────────────────────────────────────────────
#         log_lbl = QLabel('OUTPUT LOG')
#         log_lbl.setObjectName('section_label')
#         root.addWidget(log_lbl)
#         root.addSpacing(8)
#
#         self.output_display = QTextEdit()
#         self.output_display.setReadOnly(True)
#         self.output_display.setMinimumHeight(240)
#         root.addWidget(self.output_display)
#
#         self.setLayout(root)
#
#     # ── UI helpers ────────────────────────────────────────────────────────────
#     def _divider(self, layout):
#         f = QFrame()
#         f.setObjectName('divider')
#         f.setFrameShape(QFrame.Shape.HLine)
#         layout.addWidget(f)
#
#     def _browse_btn(self, row_layout):
#         # last item in the HBoxLayout is the button
#         return row_layout.itemAt(row_layout.count() - 1).widget()
#
#     def _path_row(self, label_text, btn_text):
#         row = QHBoxLayout()
#         row.setSpacing(12)
#
#         col = QVBoxLayout()
#         col.setSpacing(5)
#
#         lbl_sec = QLabel(label_text)
#         lbl_sec.setObjectName('section_label')
#
#         path_lbl = QLabel('not selected')
#         path_lbl.setObjectName('path_label')
#         path_lbl.setProperty('empty', 'true')
#         path_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
#
#         col.addWidget(lbl_sec)
#         col.addWidget(path_lbl)
#
#         btn = QPushButton(btn_text)
#         btn.setFixedWidth(110)
#         btn.setFixedHeight(32)
#
#         row.addLayout(col)
#         row.addWidget(btn, alignment=Qt.AlignmentFlag.AlignBottom)
#         return path_lbl, row
#
#     # ── Browsing ──────────────────────────────────────────────────────────────
#     def _set_path_label(self, lbl, text):
#         lbl.setText(text)
#         lbl.setProperty('empty', 'false')
#         lbl.setStyleSheet('')   # let stylesheet re-evaluate
#
#     def browse_ptal(self):
#         path, _ = QFileDialog.getOpenFileName(self, 'Select Pipe Tally file', '', 'Excel Files (*.xlsx)')
#         if path:
#             self.ptal_path = path
#             self._set_path_label(self.ptal_label, Path(path).name)
#
#     def browse_pkl_folder(self):
#         path = QFileDialog.getExistingDirectory(self, 'Select Pipes folder')
#         if path:
#             self.pkl_folder = path
#             self._set_path_label(self.pkl_label, path)
#
#     def browse_output_folder(self):
#         path = QFileDialog.getExistingDirectory(self, 'Select Output folder')
#         if path:
#             self.output_folder = path
#             self._set_path_label(self.out_label, path)
#
#     # ── Run ───────────────────────────────────────────────────────────────────
#     def run_scripts(self):
#         if not self.ptal_path or not self.pkl_folder or not self.output_folder:
#             self._log('⚠  Please select all required paths before running.', '#c06060')
#             return
#
#         self.run_button.setEnabled(False)
#         self.output_display.clear()
#         self.status_ind.set_running()
#
#         self.worker = ScriptWorker(self.ptal_path, self.pkl_folder, self.output_folder)
#         self.worker.output_signal.connect(self._on_output)
#         self.worker.finished_signal.connect(self._on_finished)
#         self.worker.start()
#
#     # ── Output ────────────────────────────────────────────────────────────────
#     def _log(self, message, color='#8090a8'):
#         self.output_display.append(f'<span style="color:{color};">{message.strip()}</span>')
#         self.output_display.ensureCursorVisible()
#
#     def _on_output(self, message):
#         lower = message.lower()
#         if any(k in lower for k in ('error', 'crash', '❌')):
#             color = '#c05060'
#         elif any(k in lower for k in ('finished', 'complete', 'done', '✅')):
#             color = '#5a9a70'
#         elif any(k in lower for k in ('running', '🟢', '🚀', '🔍')):
#             color = '#4a80c0'
#         elif '🔴' in message:
#             color = '#a06070'
#         else:
#             color = '#6070a0'
#         self._log(message, color)
#
#     def _on_finished(self):
#         self._log('', '')
#         self._log('━━  All scripts finished  ━━', '#3a7050')
#         self.run_button.setEnabled(True)
#         self.status_ind.set_done()
#
#     def closeEvent(self, event):
#         if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
#             self.worker.terminate()
#             self.worker.wait()
#         event.accept()
#
#
# # ── Stdout redirector ─────────────────────────────────────────────────────────
# class StdoutRedirector:
#     def __init__(self, signal_func):
#         self.signal_func = signal_func
#         self._original = sys.stdout
#
#     def write(self, text):
#         if text.strip():
#             self.signal_func(text.strip())
#         self._original.write(text)
#
#     def flush(self):
#         self._original.flush()
#
#
# # ── Worker thread ─────────────────────────────────────────────────────────────
# class ScriptWorker(QThread):
#     output_signal   = pyqtSignal(str)
#     finished_signal = pyqtSignal()
#
#     def __init__(self, ptal_path, pkl_folder, output_folder):
#         super().__init__()
#         self.ptal_path     = ptal_path
#         self.pkl_folder    = pkl_folder
#         self.output_folder = output_folder
#
#     def run(self):
#         redirector = StdoutRedirector(self.output_signal.emit)
#         sys.stdout = redirector
#         try:
#             self.output_signal.emit("Running pipeTally_filter.py…")
#             pdf = pd.read_excel(self.ptal_path)
#             create_pipe_tally(
#                 pdf,
#                 output_folder=self.output_folder,
#                 pkl_folder=self.pkl_folder,
#                 output_callback=self.output_signal.emit,
#             )
#             self.output_signal.emit("pipeTally_filter.py finished.")
#
#             # self.output_signal.emit("Running defects_creator.py…")
#             # create_defectSheet_and_heatmap_box(
#             #     pkl_folder=self.pkl_folder,
#             #     output_folder=self.output_folder,
#             #     output_callback=self.output_signal.emit,
#             # )
#             # self.output_signal.emit("defectS_creator.py finished.")
#
#             self.output_signal.emit("Running html_filter.py…")
#             create_html_and_csv_from_pkl(
#                 pipetally_path=self.ptal_path,
#                 pkl_folder=self.pkl_folder,
#                 output_folder=self.output_folder,
#                 output_callback=self.output_signal.emit,
#             )
#             self.output_signal.emit("html_filter.py finished.")
#         finally:
#             sys.stdout = redirector._original
#             self.finished_signal.emit()
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     runner = ScriptRunnerApp()
#     runner.show()
#     sys.exit(app.exec())



import sys
import re
import os
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QFrame,
    QSizePolicy, QGraphicsOpacityEffect
)

import pandas as pd

from defectS_creator import create_defectSheet_and_heatmap_box
from pipeTally_filter import create_pipe_tally
from html_filter import create_html_and_csv_from_pkl, WORKERS


STYLE = """
* { box-sizing: border-box; }

QWidget {
    background-color: #111318;
    color: #c8cdd8;
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}

QLabel#section_label {
    color: #3d4255;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: bold;
}

QLabel#path_label {
    color: #7090b8;
    font-size: 12px;
    padding: 7px 12px;
    background-color: #181b22;
    border: 1px solid #1f2330;
    border-radius: 3px;
}
QLabel#path_label[empty="true"] { color: #2e3348; }

QPushButton {
    background-color: #181b22;
    color: #8896b0;
    border: 1px solid #20242e;
    border-radius: 3px;
    padding: 7px 14px;
    font-size: 11px;
    letter-spacing: 0.5px;
}
QPushButton:hover {
    background-color: #1e2230;
    border-color: #3060a0;
    color: #c0d4f0;
}
QPushButton:pressed { background-color: #151820; }

QPushButton#run_button {
    background-color: #0d1e38;
    color: #5090d0;
    border: 1px solid #1a4070;
    font-size: 12px;
    font-weight: bold;
    padding: 11px 28px;
    letter-spacing: 2px;
    border-radius: 3px;
    min-width: 160px;
}
QPushButton#run_button:hover {
    background-color: #102540;
    border-color: #3070b0;
    color: #80b8f0;
}
QPushButton#run_button:pressed { background-color: #0a1828; }
QPushButton#run_button:disabled {
    background-color: #111318;
    color: #1e2535;
    border-color: #171b24;
}

QTextEdit {
    background-color: #0c0e13;
    color: #8090a8;
    border: 1px solid #181b22;
    border-radius: 3px;
    padding: 10px 12px;
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    selection-background-color: #1e3a5a;
}

QFrame#divider {
    background-color: #181b22;
    max-height: 1px;
}

QScrollBar:vertical {
    background: #0c0e13;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #252a38;
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #303548; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }

"""


# ── Animated status indicator ─────────────────────────────────────────────────
class StatusIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._dot = QLabel('●')
        self._dot.setStyleSheet('color: #1e2535; font-size: 10px;')
        self._text = QLabel('')
        self._text.setStyleSheet('color: #3d4255; font-size: 11px; letter-spacing: 0.5px;')

        layout.addWidget(self._dot)
        layout.addWidget(self._text)
        layout.addStretch()

        self._opacity = QGraphicsOpacityEffect()
        self._dot.setGraphicsEffect(self._opacity)
        self._anim = QPropertyAnimation(self._opacity, b"opacity")
        self._anim.setDuration(900)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.2)
        self._anim.setEasingCurve(QEasingCurve.Type.SineCurve)
        self._anim.setLoopCount(-1)

    def set_idle(self):
        self._anim.stop()
        self._opacity.setOpacity(1.0)
        self._dot.setStyleSheet('color: #1e2535; font-size: 10px;')
        self._text.setStyleSheet('color: #3d4255; font-size: 11px;')
        self._text.setText('')

    def set_running(self, text='Running…'):
        self._dot.setStyleSheet('color: #3a80d0; font-size: 10px;')
        self._text.setStyleSheet('color: #4a7aaa; font-size: 11px; letter-spacing: 0.5px;')
        self._text.setText(text)
        self._anim.start()

    def set_done(self):
        self._anim.stop()
        self._opacity.setOpacity(1.0)
        self._dot.setStyleSheet('color: #3a9060; font-size: 10px;')
        self._text.setStyleSheet('color: #4a8870; font-size: 11px; letter-spacing: 0.5px;')
        self._text.setText('Completed')

    def set_error(self):
        self._anim.stop()
        self._opacity.setOpacity(1.0)
        self._dot.setStyleSheet('color: #c04050; font-size: 10px;')
        self._text.setStyleSheet('color: #a04050; font-size: 11px; letter-spacing: 0.5px;')
        self._text.setText('Error')


# ── Main window ───────────────────────────────────────────────────────────────
class ScriptRunnerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ptal_path = ''
        self.pkl_folder = ''
        self.output_folder = ''
        self.initUI()

    def initUI(self):
        self.setStyleSheet(STYLE)
        self.setWindowTitle('Pipeline Script Runner')
        self.resize(860, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────────
        title = QLabel('PIPELINE RUNNER')
        title.setStyleSheet('font-size: 15px; font-weight: bold; letter-spacing: 4px; color: #d0d8e8;')
        subtitle = QLabel('tally  /  defects  /  html')
        subtitle.setStyleSheet('font-size: 10px; color: #2a3045; letter-spacing: 1.5px; padding-left: 2px;')
        hdr_col = QVBoxLayout()
        hdr_col.setSpacing(2)
        hdr_col.addWidget(title)
        hdr_col.addWidget(subtitle)
        hdr_row = QHBoxLayout()
        hdr_row.addLayout(hdr_col)
        hdr_row.addStretch()
        root.addLayout(hdr_row)
        root.addSpacing(24)
        self._divider(root)
        root.addSpacing(22)

        # ── Path selectors ───────────────────────────────────────────────────
        self.ptal_label, ptal_row = self._path_row('PIPE TALLY FILE', 'Browse .xlsx')
        self.pkl_label,  pkl_row  = self._path_row('PIPES FOLDER',    'Browse Folder')
        self.out_label,  out_row  = self._path_row('OUTPUT FOLDER',   'Browse Folder')

        self._browse_btn(ptal_row).clicked.connect(self.browse_ptal)
        self._browse_btn(pkl_row).clicked.connect(self.browse_pkl_folder)
        self._browse_btn(out_row).clicked.connect(self.browse_output_folder)

        paths = QVBoxLayout()
        paths.setSpacing(14)
        for row in (ptal_row, pkl_row, out_row):
            w = QWidget()
            w.setLayout(row)
            paths.addWidget(w)
        root.addLayout(paths)
        root.addSpacing(26)
        self._divider(root)
        root.addSpacing(20)

        # ── Run row ──────────────────────────────────────────────────────────
        run_row = QHBoxLayout()
        run_row.setSpacing(18)
        self.run_button = QPushButton('▶  RUN')
        self.run_button.setObjectName('run_button')
        self.run_button.setFixedHeight(40)
        self.run_button.clicked.connect(self.run_scripts)
        self.status_ind = StatusIndicator()
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.status_ind)
        run_row.addStretch()
        root.addLayout(run_row)
        root.addSpacing(22)
        self._divider(root)
        root.addSpacing(14)

        # ── Log ──────────────────────────────────────────────────────────────
        log_lbl = QLabel('OUTPUT LOG')
        log_lbl.setObjectName('section_label')
        root.addWidget(log_lbl)
        root.addSpacing(8)
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setMinimumHeight(220)
        root.addWidget(self.output_display)

        self.setLayout(root)

    # ── UI helpers ────────────────────────────────────────────────────────────
    def _divider(self, layout):
        f = QFrame()
        f.setObjectName('divider')
        f.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(f)

    def _browse_btn(self, row_layout):
        return row_layout.itemAt(row_layout.count() - 1).widget()

    def _path_row(self, label_text, btn_text):
        row = QHBoxLayout()
        row.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(5)
        lbl_sec = QLabel(label_text)
        lbl_sec.setObjectName('section_label')
        path_lbl = QLabel('not selected')
        path_lbl.setObjectName('path_label')
        path_lbl.setProperty('empty', 'true')
        path_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        col.addWidget(lbl_sec)
        col.addWidget(path_lbl)
        btn = QPushButton(btn_text)
        btn.setFixedWidth(110)
        btn.setFixedHeight(32)
        row.addLayout(col)
        row.addWidget(btn, alignment=Qt.AlignmentFlag.AlignBottom)
        return path_lbl, row

    def _set_path_label(self, lbl, text):
        lbl.setText(text)
        lbl.setProperty('empty', 'false')
        lbl.setStyleSheet('')

    # ── Browsing ──────────────────────────────────────────────────────────────
    def browse_ptal(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Pipe Tally file', '', 'Excel Files (*.xlsx)')
        if path:
            self.ptal_path = path
            self._set_path_label(self.ptal_label, Path(path).name)

    def browse_pkl_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Pipes folder')
        if path:
            self.pkl_folder = path
            self._set_path_label(self.pkl_label, path)

    def browse_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Output folder')
        if path:
            self.output_folder = path
            self._set_path_label(self.out_label, path)

    # ── Run ───────────────────────────────────────────────────────────────────
    def run_scripts(self):
        if not self.ptal_path or not self.pkl_folder or not self.output_folder:
            self._log('⚠  Please select all required paths before running.', '#c06060')
            return

        self.run_button.setEnabled(False)
        self.output_display.clear()
        self.status_ind.set_running()

        self.worker = ScriptWorker(self.ptal_path, self.pkl_folder, self.output_folder)
        self.worker.output_signal.connect(self._on_output)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    # ── Output ────────────────────────────────────────────────────────────────
    def _log(self, message, color='#8090a8'):
        self.output_display.append(f'<span style="color:{color};">{message.strip()}</span>')
        self.output_display.ensureCursorVisible()

    def _on_output(self, message):
        lower = message.lower()

        # ── colour the log line ───────────────────────────────────────────────
        if any(k in lower for k in ('error', 'crash', '❌')):
            color = '#c05060'
        elif any(k in lower for k in ('finished', 'complete', 'done', '✅')):
            color = '#5a9a70'
        elif any(k in lower for k in ('running', '🟢', '🚀', '🔍')):
            color = '#4a80c0'
        elif '🔴' in message:
            color = '#7a6080'
        else:
            color = '#6070a0'

        self._log(message, color)

    def _on_finished(self):
        self._log('', '')
        self._log('━━  All scripts finished  ━━', '#3a7050')
        self.run_button.setEnabled(True)
        self.status_ind.set_done()

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()


# ── Stdout redirector ─────────────────────────────────────────────────────────
class StdoutRedirector:
    def __init__(self, signal_func):
        self.signal_func = signal_func
        self._original = sys.stdout

    def write(self, text):
        if text.strip():
            self.signal_func(text.strip())
        self._original.write(text)

    def flush(self):
        self._original.flush()


# ── Worker thread ─────────────────────────────────────────────────────────────
class ScriptWorker(QThread):
    output_signal   = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, ptal_path, pkl_folder, output_folder):
        super().__init__()
        self.ptal_path     = ptal_path
        self.pkl_folder    = pkl_folder
        self.output_folder = output_folder

    def run(self):
        redirector = StdoutRedirector(self.output_signal.emit)
        sys.stdout = redirector
        try:
            self.output_signal.emit("Running pipeTally_filter.py…")
            pdf = pd.read_excel(self.ptal_path)
            create_pipe_tally(
                pdf,
                output_folder=self.output_folder,
                pkl_folder=self.pkl_folder,
                output_callback=self.output_signal.emit,
            )
            self.output_signal.emit("pipeTally_filter.py finished.")

            # self.output_signal.emit("Running defects_creator.py…")
            # create_defectSheet_and_heatmap_box(
            #     pkl_folder=self.pkl_folder,
            #     output_folder=self.output_folder,
            #     output_callback=self.output_signal.emit,
            # )
            # self.output_signal.emit("defectS_creator.py finished.")

            self.output_signal.emit("Running html_filter.py…")
            create_html_and_csv_from_pkl(
                pipetally_path=self.ptal_path,
                pkl_folder=self.pkl_folder,
                output_folder=self.output_folder,
                output_callback=self.output_signal.emit,
            )
            self.output_signal.emit("html_filter.py finished.")
        finally:
            sys.stdout = redirector._original
            self.finished_signal.emit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    runner = ScriptRunnerApp()
    runner.show()
    sys.exit(app.exec())
