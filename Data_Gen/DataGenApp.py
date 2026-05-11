# import sys
# from PyQt6.QtCore import QThread, pyqtSignal
# from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
# import pandas as pd
#
# from defectS_creator import create_defectSheet_and_heatmap_box
# # from Data_Gen.defectS_compare import create_defectSheet
# from pipeTally_filter import create_pipe_tally
# from html_filter import create_html_and_csv_from_pkl
#
#
# class ScriptRunnerApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#         self.ptal_path = ''
#         self.pkl_folder = ''
#         self.output_folder = ''
#
#     def initUI(self):
#         layout = QVBoxLayout()
#
#         # Label for ptal.xlsx
#         self.ptal_label = QLabel('Select Pipe Tally file:')
#         layout.addWidget(self.ptal_label)
#
#         # Button to browse ptal.xlsx
#         self.ptal_button = QPushButton('Browse Pipe Tally')
#         self.ptal_button.clicked.connect(self.browse_ptal)
#         layout.addWidget(self.ptal_button)
#
#         # Label for pipes3 folder
#         self.pkl_label = QLabel('Select Pipes folder:')
#         layout.addWidget(self.pkl_label)
#
#         # Button to browse pipes3 folder
#         self.pkl_button = QPushButton('Browse Pipes Folder')
#         self.pkl_button.clicked.connect(self.browse_pkl_folder)
#         layout.addWidget(self.pkl_button)
#
#         # Label for output folder
#         self.output_label = QLabel('Select Output folder:')
#         layout.addWidget(self.output_label)
#
#         # Button to browse output folder
#         self.output_button = QPushButton('Browse Output Folder')
#         self.output_button.clicked.connect(self.browse_output_folder)
#         layout.addWidget(self.output_button)
#
#         # Button to run scripts
#         self.run_button = QPushButton('Run Scripts')
#         self.run_button.clicked.connect(self.run_scripts)
#         layout.addWidget(self.run_button)
#
#         # Terminal output display
#         self.output_display = QTextEdit()
#         self.output_display.setReadOnly(True)
#         layout.addWidget(self.output_display)
#
#         # Set the layout for the main window
#         self.setLayout(layout)
#         self.setWindowTitle('Script Runner')
#
#     def browse_ptal(self):
#         self.ptal_path, _ = QFileDialog.getOpenFileName(self, 'Select Pipe Tally file', '', 'Excel Files (*.xlsx)')
#         if self.ptal_path:
#             self.ptal_label.setText(f'Selected Pipe Tally: {self.ptal_path}')
#
#     def browse_pkl_folder(self):
#         self.pkl_folder = QFileDialog.getExistingDirectory(self, 'Select Pipes folder')
#         if self.pkl_folder:
#             self.pkl_label.setText(f'Selected Pipes Folder: {self.pkl_folder}')
#
#     def browse_output_folder(self):
#         self.output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
#         if self.output_folder:
#             self.output_label.setText(f'Selected Output Folder: {self.output_folder}')
#
#     def run_scripts(self):
#         if not self.ptal_path or not self.pkl_folder or not self.output_folder:
#             self.output_display.append("Please select all required paths before running the scripts.\n")
#             return
#
#         # Create a worker thread to run the scripts
#         self.worker = ScriptWorker(self.ptal_path, self.pkl_folder, self.output_folder)
#         self.worker.output_signal.connect(self.update_output)  # Connect the signal to the output display
#         self.worker.finished_signal.connect(self.on_scripts_finished)
#
#         # Start the worker thread
#         self.output_display.append("Running scripts...\n")
#         self.worker.start()
#
#     def update_output(self, message):
#         self.output_display.append(message)
#         self.output_display.ensureCursorVisible()  # Auto-scroll to the latest message
#
#     def on_scripts_finished(self):
#         self.output_display.append("All scripts finished!\n")
#
#     def closeEvent(self, event):
#         if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
#             self.worker.terminate()
#             self.worker.wait()
#         event.accept()
#
#
#
# class ScriptWorker(QThread):
#     output_signal = pyqtSignal(str)
#     finished_signal = pyqtSignal()
#
#     def __init__(self, ptal_path, pkl_folder, output_folder):
#         super().__init__()
#         self.ptal_path = ptal_path
#         self.pkl_folder = pkl_folder
#         self.output_folder = output_folder
#
#     def run(self):
#         try:
#             # Run the first script function (pipeTally_filter)
#             self.output_signal.emit("Running pipeTally_filter.py...\n")
#             pdf = pd.read_excel(self.ptal_path)
#             create_pipe_tally(pdf, output_folder=self.output_folder, output_callback=self.emit_output)
#             self.output_signal.emit("pipeTally_filter.py finished.\n")
#
#             # # Run the second script function (defectS_creator)
#             # self.output_signal.emit("Running defects_creator.py...\n")
#             # create_defectSheet_and_heatmap_box(pkl_folder=self.pkl_folder, output_folder=self.output_folder, output_callback=self.emit_output)
#             # # create_defectSheet(pkl_folder=self.pkl_folder, output_folder=self.output_folder, output_callback=self.emit_output)
#             # self.output_signal.emit("defectS_creator.py finished.\n")
#
#             # Run the third script function (html_filter)
#             self.output_signal.emit("Running html_filter.py...\n")
#             create_html_and_csv_from_pkl(pkl_folder=self.pkl_folder, output_folder=self.output_folder, output_callback=self.emit_output)
#             self.output_signal.emit("html_filter.py finished.\n")
#
#         except Exception as e:
#             self.output_signal.emit(f"Error: {str(e)}\n")
#         finally:
#             self.finished_signal.emit()
#
#     def emit_output(self, message):
#         self.output_signal.emit(message)
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
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QScrollArea, QSizePolicy
)
import pandas as pd

from defectS_creator import create_defectSheet_and_heatmap_box
from pipeTally_filter import create_pipe_tally
from html_filter import create_html_and_csv_from_pkl, WORKERS




class ScriptRunnerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.worker_displays = {}
        self.initUI()
        self.ptal_path = ''
        self.pkl_folder = ''
        self.output_folder = ''

    def initUI(self):
        layout = QVBoxLayout()

        # ── Path selectors ────────────────────────────────────────────────────
        self.ptal_label = QLabel('Select Pipe Tally file:')
        layout.addWidget(self.ptal_label)

        self.ptal_button = QPushButton('Browse Pipe Tally')
        self.ptal_button.clicked.connect(self.browse_ptal)
        layout.addWidget(self.ptal_button)

        self.pkl_label = QLabel('Select Pipes folder:')
        layout.addWidget(self.pkl_label)

        self.pkl_button = QPushButton('Browse Pipes Folder')
        self.pkl_button.clicked.connect(self.browse_pkl_folder)
        layout.addWidget(self.pkl_button)

        self.output_label = QLabel('Select Output folder:')
        layout.addWidget(self.output_label)

        self.output_button = QPushButton('Browse Output Folder')
        self.output_button.clicked.connect(self.browse_output_folder)
        layout.addWidget(self.output_button)

        # ── Run button ────────────────────────────────────────────────────────
        self.run_button = QPushButton('Run Scripts')
        self.run_button.clicked.connect(self.run_scripts)
        layout.addWidget(self.run_button)

        # ── Main log (small, always visible at top) ───────────────────────────
        layout.addWidget(QLabel('Main Log:'))
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setMaximumHeight(100)
        layout.addWidget(self.output_display)

        # ── Per-pipe panels inside a scroll area ──────────────────────────────
        layout.addWidget(QLabel('Per-Pipe Output:'))

        self.worker_grid = QGridLayout()
        self.worker_grid.setSpacing(6)

        grid_container = QWidget()
        grid_container.setLayout(self.worker_grid)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_container)
        scroll.setMinimumHeight(300)
        layout.addWidget(scroll)

        self.setLayout(layout)
        self.setWindowTitle('Script Runner')
        self.resize(1200, 800)

    # ── Path browsing ─────────────────────────────────────────────────────────
    def browse_ptal(self):
        self.ptal_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Pipe Tally file', '', 'Excel Files (*.xlsx)'
        )
        if self.ptal_path:
            self.ptal_label.setText(f'Selected Pipe Tally: {self.ptal_path}')

    def browse_pkl_folder(self):
        self.pkl_folder = QFileDialog.getExistingDirectory(self, 'Select Pipes folder')
        if self.pkl_folder:
            self.pkl_label.setText(f'Selected Pipes Folder: {self.pkl_folder}')

    def browse_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if self.output_folder:
            self.output_label.setText(f'Selected Output Folder: {self.output_folder}')

    # ── Worker panels ─────────────────────────────────────────────────────────
    def setup_worker_panels(self, pkl_paths):
        # Clear previous panels
        for i in reversed(range(self.worker_grid.count())):
            item = self.worker_grid.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self.worker_displays.clear()

        cols = min(len(pkl_paths), 4)   # max 4 columns
        for idx, pkl_path in enumerate(pkl_paths):
            pipe_name = Path(pkl_path).stem

            panel = QTextEdit()
            panel.setReadOnly(True)
            panel.setMinimumHeight(220)
            panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            lbl = QLabel(f'🟡 Pipe {pipe_name}')
            lbl.setStyleSheet('font-weight: bold; font-size: 12px;')

            container = QWidget()
            container.setStyleSheet('border: 1px solid #ccc; border-radius: 4px; padding: 2px;')
            v = QVBoxLayout(container)
            v.setContentsMargins(4, 4, 4, 4)
            v.addWidget(lbl)
            v.addWidget(panel)

            row, col = divmod(idx, cols)
            self.worker_grid.addWidget(container, row, col)
            self.worker_displays[pipe_name] = (panel, lbl)

    def _mark_pipe_done(self, pipe_id):
        if pipe_id in self.worker_displays:
            _, lbl = self.worker_displays[pipe_id]
            lbl.setText(f'✅ Pipe {pipe_id}')
            lbl.setStyleSheet('font-weight: bold; font-size: 12px; color: green;')

    def _mark_pipe_error(self, pipe_id):
        if pipe_id in self.worker_displays:
            _, lbl = self.worker_displays[pipe_id]
            lbl.setText(f'❌ Pipe {pipe_id}')
            lbl.setStyleSheet('font-weight: bold; font-size: 12px; color: red;')

    # ── Run ───────────────────────────────────────────────────────────────────
    def run_scripts(self):
        if not self.ptal_path or not self.pkl_folder or not self.output_folder:
            self.output_display.append(
                "Please select all required paths before running the scripts.\n"
            )
            return

        # Collect pkl paths and build panels BEFORE starting the thread
        pkl_paths = sorted([
            str(Path(self.pkl_folder) / f)
            for f in os.listdir(self.pkl_folder)
            if f.lower().endswith('.pkl')
        ])
        self.setup_worker_panels(pkl_paths)

        self.worker = ScriptWorker(self.ptal_path, self.pkl_folder, self.output_folder)
        self.worker.output_signal.connect(self.update_output)
        self.worker.finished_signal.connect(self.on_scripts_finished)

        self.output_display.append("Running scripts...\n")
        self.worker.start()

    # ── Output routing ────────────────────────────────────────────────────────
    def update_output(self, message):
        # Always append to main log
        self.output_display.append(message)
        self.output_display.ensureCursorVisible()

        # Route to the matching pipe panel
        match = re.search(r'[Pp]ipe[_\s]*(\d+)', message)
        if match:
            pipe_id = match.group(1)
            if pipe_id in self.worker_displays:
                panel, _ = self.worker_displays[pipe_id]
                panel.append(message)
                panel.ensureCursorVisible()

                # Update label colour based on keywords
                lower = message.lower()
                if any(k in lower for k in ('complete', 'finished', 'end', '✅')):
                    self._mark_pipe_done(pipe_id)
                elif any(k in lower for k in ('error', 'crash', '❌')):
                    self._mark_pipe_error(pipe_id)

    def on_scripts_finished(self):
        self.output_display.append("All scripts finished!\n")

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()


import sys

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
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, ptal_path, pkl_folder, output_folder):
        super().__init__()
        self.ptal_path = ptal_path
        self.pkl_folder = pkl_folder
        self.output_folder = output_folder

    # def run(self):
    #     try:
    #         self.output_signal.emit("Running pipeTally_filter.py...\n")
    #         pdf = pd.read_excel(self.ptal_path)
    #         create_pipe_tally(
    #             pdf,
    #             output_folder=self.output_folder,
    #             output_callback=self.emit_output
    #         )
    #         self.output_signal.emit("pipeTally_filter.py finished.\n")
    #
    #         # self.output_signal.emit("Running defects_creator.py...\n")
    #         # create_defectSheet_and_heatmap_box(
    #         #     pkl_folder=self.pkl_folder,
    #         #     output_folder=self.output_folder,
    #         #     output_callback=self.emit_output
    #         # )
    #         # self.output_signal.emit("defectS_creator.py finished.\n")
    #
    #         self.output_signal.emit("Running html_filter.py...\n")
    #         create_html_and_csv_from_pkl(
    #             pkl_folder=self.pkl_folder,
    #             output_folder=self.output_folder,
    #             output_callback=self.emit_output
    #         )
    #         self.output_signal.emit("html_filter.py finished.\n")
    #
    #     except Exception as e:
    #         self.output_signal.emit(f"Error: {str(e)}\n")
    #     finally:
    #         self.finished_signal.emit()
    def run(self):
        redirector = StdoutRedirector(self.output_signal.emit)
        sys.stdout = redirector
        try:
            self.output_signal.emit("Running pipeTally_filter.py...\n")
            pdf = pd.read_excel(self.ptal_path)
            create_pipe_tally(
                pdf,
                output_folder=self.output_folder,
                output_callback=self.emit_output
            )
            self.output_signal.emit("pipeTally_filter.py finished.\n")

            # self.output_signal.emit("Running defects_creator.py...\n")
            # create_defectSheet_and_heatmap_box(
            #     pkl_folder=self.pkl_folder,
            #     output_folder=self.output_folder,
            #     output_callback=self.emit_output
            # )
            # self.output_signal.emit("defectS_creator.py finished.\n")

            self.output_signal.emit("Running html_filter.py...\n")
            create_html_and_csv_from_pkl(
                pkl_folder=self.pkl_folder,
                output_folder=self.output_folder,
                output_callback=self.emit_output
            )
            self.output_signal.emit("html_filter.py finished.\n")
        finally:
            sys.stdout = redirector._original  # restore
            self.finished_signal.emit()

    def emit_output(self, message):
        self.output_signal.emit(message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    runner = ScriptRunnerApp()
    runner.show()
    sys.exit(app.exec())