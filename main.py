import sys
import os
import time
import threading
import subprocess
import webbrowser
import re

import numpy as np
import pandas as pd
import chardet
import matplotlib
matplotlib.use("Qt5Agg")  # Use Qt5 backend for matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import importlib.util
from PyQt6.QtCore import QUrl
from pathlib import Path
# PyQt6 Core
from PyQt6.QtCore import (
    Qt, QSortFilterProxyModel, QFile, QIODevice, QThread, pyqtSignal,
    QTimer, QDateTime, QUrl
)

# PyQt6 GUI
from PyQt6.QtGui import (
    QStandardItemModel, QStandardItem, QAction, QIcon, QMovie, QPixmap , QImage
)

# PyQt6 Widgets
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QHeaderView, QMenu,
    QInputDialog, QSpacerItem, QLabel, QSizePolicy, QTableWidget,
    QTableWidgetItem, QStatusBar, QVBoxLayout, QWidget, QHBoxLayout,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QDialog, QTextEdit, QPushButton
)

# PyQt6 WebEngine
from PyQt6.QtWebEngineWidgets import QWebEngineView
from utils import resource_path

# UI Designer
from PyQt6 import uic, QtWidgets

# Excel
from openpyxl import load_workbook

# Matplotlib with Qt backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- Project-specific imports ---
from reportlab.pdfgen import canvas
from pages.customPlot import CPlot_Frame as customPlot 
from pages.telemetryPlot import TPlot_Frame as telePlot 
from pages.anamolyPlot import ADPlot_Frame as adPlot 
from pages.about import About_Dialog 
from pages.adminPanel import Admin_Panel
from pages.ERF import ERF
from pages.XYZ import XYZ
from pages.metrics import Metric_Dialog
from pages.cluster import Cluster_Dialog
from pages.assessMethod import Assess_Dialog
from pages.errorBox import Error_Dialog
from pages.report1 import Report01, Main01Tab, Main02Tab, Main03Tab
from pages.Report import Report, Main1Tab, Main2Tab, Main3Tab
from backend.line_plot import PlotWindow
from backend.heatmap import HeatmapWindow as hm, pre_process, pre_process2
# from pages.Pipe_Highlights import run_app 
from ui.graphs_ui import GraphApp
from Data_Gen.DataGenApp import ScriptRunnerApp 


base_dir = os.path.dirname(__file__)
ui_path = os.path.join(base_dir, "ui", "landing.ui")
SplashScreen, SplashWindow = uic.loadUiType(ui_path)

ui_path_main = os.path.join(base_dir, "ui", "main_window.ui")
Form, Window = uic.loadUiType(ui_path_main)

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

excel_path = resource_path("14inch Petrofac pipetally.xlsx")

class Worker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, data, mode_index):
        super().__init__()
        self.data = data
        self.mode_index = mode_index

    def run(self):
        try:
            ds = pre_process2(self.data)
            self.finished.emit((ds, self.mode_index))
        except Exception as e:
            self.error.emit(str(e))

class SplashScreenWidget(QtWidgets.QWidget, SplashScreen):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class MainApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.splash = None
        self.main_window = None
        

    def show_splash_screen(self):
        """Create and show the splash screen with animated GIF."""
        self.splash = SplashScreenWidget()
        self.splash.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        self.label = self.splash.findChild(QLabel, 'label')
        if self.label:
            gif_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "VDT_ani.gif")
            if not os.path.exists(gif_path):
                print(f"GIF not found at: {gif_path}")
            self.movie = QMovie(gif_path)
            self.label.setMovie(self.movie)
            self.movie.start()
        else:
            print("QLabel 'label' not found in splash UI.")

        self.splash.show()

    def close_splash_screen(self):
        if self.splash:
            self.splash.close()

    def show_main_window(self):
        self.main_window = MyMainWindow()
        self.main_window.show()

    def start(self):
        self.show_splash_screen()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.initialize_app)
        self.timer.start(4000)  # Show splash for 4 seconds

    def initialize_app(self):
        self.close_splash_screen()
        self.show_main_window()

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.child_windows = {}
        self.ui = Form()
        self.ui.setupUi(self)
        # act = self.findChild(QAction, "action_Pipe_High")
        # if act:
        #     act.triggered.connect(self.open_PipeHigh)
        # else:
        #     print("⚠ action_Pipe_Highlights not found; skipping hookup")

        try:
            excel_path = resource_path("14inch Petrofac pipetally.xlsx")
            self.pipe_tally = pd.read_excel(excel_path)
        except Exception as e:
            self.pipe_tally = None
            self.open_Error(f"Error loading pipe tally file:\n{e}")

        self.model = QStandardItemModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)
        self.ui.tableView.setModel(self.proxy_model)

        self.header_list = []
        self.heatmap_box = None
        self.hmap = None
        self.hmap_r = None
        self.lplot = None
        self.lplot_r = None
        self.pipe3d = None

        self.canvas = PlotWindow(self, width=5, height=4, dpi=100)
        self.scene = QtWidgets.QGraphicsScene()

        # self.heatmaplabel = QLabel(self)
        # self.ui.verticalLayoutGraph.addWidget(self.heatmaplabel, stretch=1)

        self.verticalGraphSpacer = QSpacerItem(20, 550, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.ui.verticalLayoutUpper.addSpacerItem(self.verticalGraphSpacer)
        

        self.web_view = QWebEngineView()
        self.web_view2 = QWebEngineView()

        # watermark_html = resource_path("ui/icons/VDT_watermark.html")
        # self.web_view.setUrl(QUrl.fromLocalFile(watermark_html))
        html_path = Path(resource_path("ui/icons/VDT_watermark.html"))
        base_url = QUrl.fromLocalFile(str(html_path.parent) + "/")

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        self.web_view.setHtml(html_content, base_url)

        self.ui.verticalLayoutGraph.addWidget(self.web_view, stretch=3)
        self.ui.verticalLayoutGraph.addWidget(self.web_view2, stretch=1)
        
        # self.ui.tableWidgetDefect.setVisible(False)
        self.web_view2.setVisible(False)
        # self.setCentralWidget(self.web_view)

        self.curr_file_path = ''
        self.erf_file_path = 'backend/files/ASME.html'
        self.curr_data = None
        self.tele_data = None

        self.defect_sheet = None
        # self.pipe_tally = None  # <-- Set to None, not a string
        self.pipe_tally_path = '/14inch Petrofac pipetally/'
        # self.pipe_tally_path = None
        self.setup_table_view()

        self.setup_actions()

        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
                color: #000000; 
            }
        """)

        self.showMaximized()
        # self.setup_tab_switcher()
        self.setup_tab_switcher2()
        self.tabplots = {
            'custom': None,
            'telemetry': None,
            'anomaly': None
        }

        self.ui.tableWidgetDefect.selectionModel().selectionChanged.connect(self.on_row_selection_changed)
        # self.ui.comboBoxPipe.currentIndexChanged.connect(self.load_selected_file)
        self.ui.comboBoxPipe.currentIndexChanged.connect(self.load_selected_folder)

        self.setup_status_bar()
        self.child_windows = {}
        if not hasattr(self, 'child_windows'):
            self.child_windows = {}
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time_display)
        self.start_time = None
        self.elapsed_time = 0

    def setup_status_bar(self):
        if not self.statusBar():
            self.setStatusBar(QStatusBar(self))

        self.current_message = 'App running'
        self.statusBar().showMessage(f'           Status:      {self.current_message}')

        right_container = QWidget()
        right_layout = QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.right_status_label = QLabel('0.0s    ')
        right_layout.addWidget(self.right_status_label)

        self.statusBar().addPermanentWidget(right_container)      

    # def setup_tab_switcher(self):
    #     self.ui.tabWidgetM.currentChanged.connect(self.tab_switcher)

    def setup_tab_switcher2(self):
        self.ui.tabWidgetM.currentChanged.connect(self.tab_switcher2)


    def set_loading_status(self):
        self.current_message = 'Loading'
        self.statusBar().showMessage(f'           Status:      {self.current_message}') 
        self.start_time = time.time()
        self.timer.start(100) 

    def set_idle_status(self):
        self.current_message = 'App running'
        self.statusBar().showMessage(f'           Status:      {self.current_message}')
        self.timer.stop()  
        self.elapsed_time = time.time() - self.start_time
        self.update_elapsed_time_display()
    
    def update_elapsed_time_display(self):
        if self.start_time:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            self.right_status_label.setText(f'{elapsed_time:.1f}s    ')

    def setup_table_view(self):
        self.ui.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ui.tableView.verticalHeader().setVisible(False)
        self.ui.tableView.setSortingEnabled(True)
        self.ui.tableView.setSortingEnabled(False)  # Initially disable sorting

    def setup_actions(self):
        print("⚙️ setup_actions() running")

        try:
            self.ui.action_Create_Proj.triggered.connect(self.open_project)
            # self.ui.menu_File.addAction(self.ui.action_Create_Proj)
            self.ui.action_Close_Proj.triggered.connect(self.close_project)
            # self.ui.menu_File.addAction(self.ui.action_Close_Proj)
            self.ui.action_Quit.triggered.connect(self.quit_app)
            self.ui.action_About.triggered.connect(self.open_About)
            self.ui.actionAdmin_Panel.triggered.connect(self.open_Admin)
            # self.action_Edit_Table.triggered.connect(self.edit_table)
            self.ui.action_ERF.triggered.connect(self.open_ERF)
            self.ui.action_XYZ.triggered.connect(self.open_XYZ)
            self.ui.action_Final_Report.triggered.connect(self.open_Report)
            self.ui.action_graphs.triggered.connect(self.open_graphs)
            self.ui.action_Assessment.triggered.connect(self.open_Assessment)
            self.ui.action_Cluster.triggered.connect(self.open_Cluster)
            self.ui.action_Pipe_High.triggered.connect(self.open_PipeHigh)
            self.ui.action_Pipe_Sch.triggered.connect(self.open_PipeScheme)
            self.ui.actionMetal_Loss_Distribution_MLD.triggered.connect(self.open_CMLD)
            self.ui.actionDepth_Based_Anomalies_Distribution_DBAD.triggered.connect(self.open_DBAD)
            self.ui.actionERF_Based_Anomalies_Distribution_E_AD.triggered.connect(self.open_EAD)
            self.ui.action_Custom.triggered.connect(self.add_plot_custom)
            self.ui.action_Telemetry.triggered.connect(self.add_plot_tele)
            self.ui.actionAnomalies_Distribution.triggered.connect(self.add_plot_ad)
            self.ui.actionHeatmap.triggered.connect(self.plot_heatmap)
            self.ui.action_Export_Table.triggered.connect(self.gen_data)
            # self.ui.action_DefectDetect.triggered.connect(self.draw_boxes)
            self.ui.action_DefectDetect.triggered.connect(self.draw_boxes2)

            self.ui.minTabWidg.clicked.connect(self.minimize_tabs)
            self.ui.maxTabWidg.clicked.connect(self.maximize_tabs)
            # self.ui.pushButtonNext.clicked.connect(self.load_next_file)
            # self.ui.pushButtonPrev.clicked.connect(self.load_prev_file)
            self.ui.pushButtonNext.clicked.connect(self.load_next_folder)
            self.ui.pushButtonPrev.clicked.connect(self.load_prev_folder)
            self.ui.Final_Report.triggered.connect(self.open_Final_Report)
            self.ui.action_Preliminary_Report.triggered.connect(self.open_Preliminary_Report)
            self.ui.action__pipetally.triggered.connect(self.open_pipe_tally)
            self.ui.action_Manual.triggered.connect(self.open_manual)

            self.ui.actionStandard.triggered.connect(self.open_digs)
            self.ui.action_Pipe_Tally.triggered.connect(self.open_Ptal)
            # self.ui.action_Pipe_Highlights.triggered.connect(run_app)
            # self.ui.action_Pipe_Highlights.triggered.connect(self.open_PipeHigh)

            
        except Exception as e:
            self.open_Error(e)

    def minimize_tabs(self):
        self.ui.tabWidgetM.hide()
        # self.ui.tabWidgetM.setVisible(False)
        self.ui.minTabWidg.setVisible(False)
        self.ui.maxTabWidg.raise_()
        self.ui.maxTabWidg.setVisible(True)
        self.ui.verticalLayoutUpper.removeItem(self.verticalGraphSpacer)
        self.web_view2.setVisible(True)

        current_index = self.ui.tabWidgetM.currentIndex()
        current_tab_name = self.ui.tabWidgetM.tabText(current_index)
        if current_tab_name == 'Plot':
            self.web_view2.setVisible(False)


    def open_graphs(self):
        try:
            print("📊 Importing GraphApp...")
            import traceback
            import sys

            def try_import_graphapp():
                try:
                    # Correct path to the graphs_ui.py file in the bundled executable
                    ui_file_path = resource_path(os.path.join("ui", "graphs_ui.py"))

                    spec = importlib.util.spec_from_file_location("graphs_ui", ui_file_path)
                    graphs_ui = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(graphs_ui)
                    print("✅ graphs_ui imported")
                    return graphs_ui.GraphApp
                except Exception as e:
                    print("❌ Import crashed:", e)
                    print(traceback.format_exc())
                    sys.exit(1)

            GraphApp = try_import_graphapp()
            # self.graph_window = GraphApp()
            if self.pipe_tally is None:
                self.open_Error("Pipe tally not loaded yet.")
                return

            self.graph_window = GraphApp(dataframe=self.pipe_tally)
            self.graph_window.show()
        except Exception as e:
            print("❌ open_graphs failed:", e)

    def gen_data(self):
        '''Open Data Generator UI'''
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

    def maximize_tabs(self):
        self.ui.tabWidgetM.show()
        # self.ui.tabWidgetM.setVisible(True)
        self.ui.minTabWidg.setVisible(True)
        self.ui.maxTabWidg.hide()
        self.ui.maxTabWidg.setVisible(False)
        self.ui.verticalLayoutUpper.addSpacerItem(self.verticalGraphSpacer)
        self.web_view2.setVisible(False)

    def open_digs(self):
        import subprocess
        import sys
        import os
        import pickle  # To serialize pipe_tally for passing it through subprocess

        try:
            # Resource path for dig_sheet.py script when running from PyInstaller EXE
            base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
            dig_py = resource_path(os.path.join("dig", "dig_sheet.py"))  # Path to dig_sheet.py within the bundle

            # Serialize pipe_tally to pass it to subprocess
            pipe_tally_file = os.path.join(base_dir, "pipe_tally.pkl")
            with open(pipe_tally_file, "wb") as f:
                pickle.dump(self.pipe_tally, f)

            # Call dig_sheet.py with the pipe_tally file
            subprocess.Popen([sys.executable, dig_py, pipe_tally_file])  # Launch dig_sheet.py from the bundled exe

        except Exception as e:
            self.open_Error(f"An error occurred: {e}")



            
    def open_PipeHigh(self):
        try:
            from pages.Pipe_Highlights import run_app
            run_app()
        except Exception as e:
            self.open_Error(f"Error running Pipe Highlight:\n{e}")

    def open_PipeScheme(self):
        try:
            from pipeline_schema.pipeline_schema import run_app
            run_app()
        except Exception as e:
            self.open_Error(f"Error running Pipeline Schema:\n{e}")


    # def open_Ptal(self):
        # import subprocess
        # subprocess.Popen(["python", "pages/pipe_tally.py"])
    
    def open_Ptal(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Pipe Tally File",
                "",
                "CSV/Excel Files (*.csv *.xlsx *.xls);;All Files (*)"
            )

            if file_path:
                if file_path.endswith('.csv'):
                    self.pipe_tally = pd.read_csv(file_path)
                else:
                    self.pipe_tally = pd.read_excel(file_path)
                QMessageBox.information(self, "Pipe Tally", "Pipe tally loaded successfully.")

                # on_load_click(self.pipe_tally)  # Call the function to handle the loaded pipe tally
            else:
                QMessageBox.warning(self, "Pipe Tally", "No file selected.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Pipe tally load failed: {e}")


## Folder cases

    def open_project(self):
        print("Opening project...")
        try:
            self.set_loading_status()
            # Open a folder dialog to select the directory
            folder_dialog = QFileDialog(self)
            folder_dialog.setFileMode(QFileDialog.FileMode.Directory)
            folder_dialog.setOption(QFileDialog.Option.ShowDirsOnly)
            folder_dialog.setWindowTitle("Select Folder")

            if folder_dialog.exec() == QFileDialog.DialogCode.Accepted:
                folder_path = folder_dialog.selectedFiles()[0]
                if folder_path:
                    # Collect all folders from the selected folder
                    self.folders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
                                    if os.path.isdir(os.path.join(folder_path, d))]
                    if self.folders:
                        self.current_index = 0
                        self.populate_combobox()
                        self.load_folder(self.folders[self.current_index])
                    else:
                        self.open_Error("No folders found in the selected directory.")
            self.set_idle_status()

        except Exception as e:
            self.open_Error(e)

    def populate_combobox(self):
        self.ui.comboBoxPipe.clear()
        self.ui.comboBoxPipe.addItems([os.path.basename(f) for f in self.folders])

    def load_selected_folder(self):
        index = self.ui.comboBoxPipe.currentIndex()
        if 0 <= index < len(self.folders):
            self.load_folder(self.folders[index])

    def load_folder(self, folder_path):
        try:
            self.set_loading_status()
            
            folder_contents = os.listdir(folder_path)
            self.curr_folder_path = folder_path
            
            folder_name = os.path.basename(folder_path)
        
            pipe_name = [f for f in os.listdir(folder_path)
                            if os.path.isfile(os.path.join(folder_path, f)) and
                            os.path.splitext(f)[0] == folder_name]
            pipeD = folder_path + f'/{pipe_name[0]}'
            
            folder_number = folder_name.split('_')[-1]
            hmap = hmap_r = hmap_b = lplot = lplot_r = pipe3d = defS = pTal = None

            for file in folder_contents:
                file_path = os.path.join(folder_path, file)
                
                if file.endswith(f'{folder_number}.html'):
                    if 'heatmap' in file:
                        if 'box' in file:
                            hmap_b = file_path
                        elif 'raw' in file:
                            hmap_r = file_path
                        else:
                            hmap = file_path
                    elif 'lineplot' in file:
                        if 'raw' in file:
                            lplot_r = file_path
                        else:
                            lplot = file_path
                    elif 'pipe3d' in file:
                        pipe3d = file_path
                elif file.endswith(f'{folder_number}.xlsx'):
                    if 'pipe' in file:
                        pipe3d = file_path

                elif file.endswith(f'{folder_number}.csv'):
                    if 'defectS' in file:
                        defS = file_path
                    else:
                        pTal = file_path

            self.curr_data = pd.read_excel(pipeD)
            dsheet = pd.read_csv(defS)
            pTall = pd.read_csv(pTal)

            self.heatmap_box = hmap_b
            self.hmap = hmap
            self.hmap_r = hmap_r
            self.lplot = lplot
            self.lplot_r = lplot_r
            self.pipe3d = pipe3d
            self.pipe_tally = pTall

            self.web_view.setUrl(QUrl.fromLocalFile(self.hmap))
            self.web_view2.setUrl(QUrl.fromLocalFile(self.hmap_r))
             #defect sheet
            column_mapping = {
                'Box Number': 'Defect_id',
                'Type': 'Feature_Type',
                'Width': 'Length',
                'Absolute Distance': 'Absolute_Distance',
                # 'Peak Value': 'Depth_Peak',
                'Depth % ':'Depth_Peak',
                'Breadth':'Width',
                # 'Ori Val':'Orientation',
                "Orientation o' clock":'Orientation',
                'WT (mm)':'WT',
                'Dimensions  Classification':'Dimension_Class',
                'Distance to U/S GW(m)':'Upstream_Distance'
            }
            header_indices = {
                'Defect_id': 0,
                'Absolute_Distance': 1,
                'Upstream_Distance': 2,
                'Feature_Type': 3,
                'Dimension_Class': 4,
                'Orientation': 5,
                'WT': 6,
                'Length': 7,
                'Width': 8,
                'Depth_Peak': 9
            }
            def populate_table_widget(table_widget: QTableWidget, df: pd.DataFrame, column_mapping: dict):
                num_rows = len(df)
                num_cols = len(header_indices)
                table_widget.setRowCount(num_rows)
                table_widget.setColumnCount(num_cols)
                headers = list(header_indices.keys())
                table_widget.setHorizontalHeaderLabels(headers)

                for row_idx, (_, row) in enumerate(df.iterrows()):
                    for src_col, dest_col in column_mapping.items():
                        if dest_col in header_indices:
                            col_idx = header_indices[dest_col]
                            value = row[src_col]

                            if isinstance(value, float):
                                value = f"{value:.2f}"

                            table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

                for row_idx in range(num_rows):
                    for col_name, col_idx in header_indices.items():
                        if col_name not in column_mapping.values():
                            table_widget.setItem(row_idx, col_idx, QTableWidgetItem(''))

                for row_idx in range(num_rows):
                    table_widget.setItem(row_idx, 6, QTableWidgetItem('7.5'))

            populate_table_widget(self.ui.tableWidgetDefect, dsheet, column_mapping)
            self.load_excel(pipeD)
            
            self.set_idle_status()
            
        except Exception as e:
            self.open_Error(e)
            
    def draw_boxes2(self):
        self.web_view.setUrl(QUrl.fromLocalFile(self.heatmap_box))

    def load_next_folder(self):
        if self.folders:
            self.current_index = (self.current_index + 1) % len(self.folders)
            self.load_folder(self.folders[self.current_index])

    def load_prev_folder(self):
        if self.folders:
            self.current_index = (self.current_index - 1) % len(self.folders)
            self.load_folder(self.folders[self.current_index])

    def load_excel(self, file_path):
        try:
            df = pd.read_excel(file_path)

            self.model.clear()

            if not df.empty:
                self.model.setHorizontalHeaderLabels(df.columns.tolist())
                self.header_list = df.columns.tolist()
                self.style_header()

                for _, row in df.iterrows():
                    row_items = [QStandardItem(str(field)) for field in row]
                    self.model.appendRow(row_items)

            self.ui.tableView.setModel(self.model)
            self.ui.tableView.setSortingEnabled(True)

        except Exception as e:
            self.open_Error(e)

    def style_header(self):
        header = self.ui.tableView.horizontalHeader()
        # header.minimumSectionSize(250) 
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: lightgray;
                color: black;
                font-weight: bold;
                border: 1px solid black;
                padding: 4px;
                width: 100px;            
            }
            QHeaderView {
                background-color: white;
            }
        """)
    def tab_switcher2(self, flag=None):
        try:
            current_index = self.ui.tabWidgetM.currentIndex()
            current_tab_name = self.ui.tabWidgetM.tabText(current_index)

            if not self.folders:  # If no project is loaded
                self.web_view.setUrl(QUrl())  # Clear web view
                self.web_view2.setUrl(QUrl())  # Clear second web view
                return 

            if current_tab_name == 'Defect Sheet':
                self.web_view.setUrl(QUrl.fromLocalFile(self.hmap))
                self.web_view2.setUrl(QUrl.fromLocalFile(self.hmap_r))
            elif current_tab_name == 'Plot':
                self.web_view.setUrl(QUrl.fromLocalFile(self.pipe3d))
            else:
                self.web_view.setUrl(QUrl.fromLocalFile(self.lplot))
                self.web_view2.setUrl(QUrl.fromLocalFile(self.lplot_r))
        except Exception as e:
            self.open_Error(e)

    def tab_switcher(self, flag=None):
        try:
            current_index = self.ui.tabWidgetM.currentIndex()
            current_tab_name = self.ui.tabWidgetM.tabText(current_index)
            if current_tab_name == 'Defect Sheet':
                file_path = 'backend/files/heatmap.html'
                script_dir = os.path.dirname(__file__)
                full_path = os.path.join(script_dir, file_path)
                local_url = QUrl.fromLocalFile(full_path)
                self.web_view.setUrl(local_url)

                file_path = 'backend/files/heatmap_raw.html'
                script_dir = os.path.dirname(__file__)
                full_path = os.path.join(script_dir, file_path)
                local_url = QUrl.fromLocalFile(full_path)
                self.web_view2.setUrl(local_url)
            elif current_tab_name == 'Plot':
                file_path = 'backend/files/pipe3d.html'
                script_dir = os.path.dirname(__file__)
                full_path = os.path.join(script_dir, file_path)
                local_url = QUrl.fromLocalFile(full_path)
                self.web_view.setUrl(local_url)
            else:
                file_path = 'backend/files/lineplot.html'
                script_dir = os.path.dirname(__file__)
                full_path = os.path.join(script_dir, file_path)
                local_url = QUrl.fromLocalFile(full_path)
                self.web_view.setUrl(local_url)

                file_path = 'backend/files/lineplot_raw.html'
                script_dir = os.path.dirname(__file__)
                full_path = os.path.join(script_dir, file_path)
                local_url = QUrl.fromLocalFile(full_path)
                self.web_view2.setUrl(local_url)
        except Exception as e:
            self.open_Error(e)

    def sort_column(self, order):
        header = self.ui.tableView.horizontalHeader()
        logical_index = header.logicalIndex(header.currentSection())
        self.proxy_model.sort(logical_index, order)

    def clear_filter(self):
        self.proxy_model.setFilterRegExp(Qt.RegExp(""))  # Clearing filter

    def apply_filter(self):
        filter_text, ok = QInputDialog.getText(self, "Apply Filter", "Enter filter text:")
        if ok:
            self.proxy_model.setFilterRegExp(Qt.RegExp(filter_text))

    def close_project(self):
        # self.model.clear()
        try:
            # Reset the project-related attributes
            self.folders = []
            self.current_index = -1
            self.curr_folder_path = None
            self.curr_data = None
            
            # Clear the UI components
            self.ui.comboBoxPipe.clear()
            self.ui.tableWidgetDefect.clear()
            self.model.clear()  # Assuming `self.model` is for the table view

            # Reset web views or any other visual components
            self.web_view.setUrl(QUrl())  # Clear the web view
            self.web_view2.setUrl(QUrl())  # Clear the second web view

            # Reset table view
            self.ui.tableView.setModel(QStandardItemModel())  # Reset the model to empty

            # Optionally, you can display a message
            QMessageBox.information(self, "Project Closed", "The project has been successfully closed.")

        except Exception as e:
            self.open_Error(e)


    def add_plot_custom(self):
        try:
            self.cplot_widget = customPlot(self.header_list)
            self.ui.graphLayout.addWidget(self.cplot_widget)
            #Buttons
            self.cplot_widget.closeCustom.clicked.connect(self.cplot_widget.close_window)
            self.cplot_widget.comboBox.currentIndexChanged.connect(self.plot_c)
            self.tabplots['custom'] = self.cplot_widget
        except Exception as e:
            self.open_Error(e)
        
    def plot_c(self):
        try:
            y_label = self.cplot_widget.comboBox.currentText()
            x_label = self.cplot_widget.comboBox_2.currentText()
            
            if x_label not in self.curr_data or y_label not in self.curr_data:
                raise ValueError("Selected labels are not in the current data.")

            x_data = self.curr_data[x_label]
            y_data = self.curr_data[y_label]

            figure = go.Figure()
            figure.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=y_label))
            figure.update_layout(title=f'{y_label} vs {x_label}', xaxis_title=x_label, yaxis_title=y_label,height=450)

            file_path = 'backend/files/customplot.html'
            figure.write_html(file_path)
            script_dir = os.path.dirname(__file__)
            full_path = os.path.join(script_dir, file_path)
            local_url = QUrl.fromLocalFile(full_path)
            self.cplot_widget.webviewCustom.setUrl(local_url)
            if self.tabplots['custom']:
                self.web_view.setUrl(local_url)

        except Exception as e:
            self.open_Error(e)

    def add_plot_tele(self):
        try:
            if self.curr_data is None or self.curr_data.empty:
                QMessageBox.critical(self, "Error", "Please load a folder/project before using Telemetry Plot.")
                return

            import re
            tlist = [col for col in self.header_list if re.match(r'^F\d+', col)]
            print("✅ Detected telemetry columns:", tlist)

            if not tlist:
                QMessageBox.warning(self, "No Telemetry Data", "No telemetry (F...) columns found in the current file.")
                return

            self.tplot_widget = telePlot(tlist)

            self.ui.graphLayout.addWidget(self.tplot_widget)

            self.tplot_widget.closeTele.clicked.connect(self.tplot_widget.close_window)

            self.tplot_widget.checkBox.stateChanged.connect(self.magnetisation)

            self.tplot_widget.checkBox_2.stateChanged.connect(self.velocity)

            self.tplot_widget.comboBox.currentIndexChanged.connect(self.plot_telemetry)

            self.tabplots['telemetry'] = self.tplot_widget

            # ✅ Auto-select first valid telemetry parameter and plot
            if len(tlist) > 0:
                self.tplot_widget.comboBox.setCurrentIndex(1)
                self.plot_telemetry()

        except Exception as e:
            self.open_Error(e)
            print("❌ EXCEPTION:", str(e))



    def magnetisation(self):
        try:
            is_checked = self.tplot_widget.checkBox.isChecked()
            parameter = self.tplot_widget.comboBox.currentText()

            if is_checked:
                filtered_columns = [col for col in self.curr_data.columns if col.startswith('F')]
                self.tele_data = self.curr_data[filtered_columns]

                if parameter not in self.tele_data.columns:
                    raise ValueError("Selected parameter is not in the current data.")

                magnetisation_by_rows = self.tele_data.mean(axis=1)
                factored_magnetisation = magnetisation_by_rows * 0.0004854
                x_data = self.curr_data['ODDO1']
                y_data = factored_magnetisation

                figure = go.Figure()
                figure.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=parameter))
                figure.update_layout(title='Magnetisation View', xaxis_title='Oddometer (mm)', yaxis_title='Magnetisation', height=450)

                file_path = 'backend/files/magnetisation.html'
                full_path = resource_path(file_path)
                figure.write_html(full_path)

            else:
                # Still define file_path and full_path even in else
                file_path = 'backend/files/telemetryplot.html'
                full_path = resource_path(file_path)
                figure = go.Figure()
                figure.write_html(full_path)

            # ✅ After both blocks
            print("📄 Writing HTML to:", full_path)
            print("📂 Exists?", os.path.exists(full_path))
            print("📏 Size:", os.path.getsize(full_path) if os.path.exists(full_path) else "❌")

            local_url = QUrl.fromLocalFile(full_path)
            self.tplot_widget.webviewTele.setUrl(local_url)

            if self.tabplots['telemetry']:
                self.web_view.setUrl(local_url)

        except Exception as e:
            self.open_Error(e)
            print("❌ magnetisation EXCEPTION:", e)

        
    def velocity(self):
        try:
            is_checked = self.tplot_widget.checkBox_2.isChecked()
            parameter = self.tplot_widget.comboBox.currentText()

            if is_checked:
                filtered_columns = [col for col in self.curr_data.columns if col.startswith('F')]
                self.tele_data = self.curr_data[filtered_columns]
                if parameter not in self.tele_data.columns:
                    raise ValueError("Selected parameter is not in the current data.")

                k = []
                oddoC = self.curr_data['ODDO1']
                for i in range(len(oddoC)-1):
                    t = oddoC[i+1] - oddoC[i]
                    sp = t / 0.000666667
                    k.append(sp)
                k.append(k[-1])  # repeat last velocity to match length

                x_data = oddoC
                y_data = k

                figure = go.Figure()
                figure.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=parameter))
                figure.update_layout(title='Velocity View', xaxis_title='Oddometer(mm)', yaxis_title='Velocity', height=450)

                file_path = 'backend/files/velocity.html'
            else:
                file_path = 'backend/files/telemetryplot.html'
                figure = go.Figure()

            full_path = resource_path(file_path)
            figure.write_html(full_path)

            print("✅ Velocity HTML written to:", full_path)
            print("📂 Exists?", os.path.exists(full_path))
            print("📏 Size:", os.path.getsize(full_path) if os.path.exists(full_path) else "❌")

            local_url = QUrl.fromLocalFile(full_path)
            self.tplot_widget.webviewTele.setUrl(local_url)
            if self.tabplots['telemetry']:
                self.web_view.setUrl(local_url)

        except Exception as e:
            self.open_Error(e)
            print("❌ velocity() EXCEPTION:", e)


    def plot_telemetry(self):
        try:
            parameter = self.tplot_widget.comboBox.currentText()

            # ✅ Skip invalid selections
            if parameter == "-Select-" or parameter not in self.curr_data.columns:
                print("⚠️ Invalid parameter selected, skipping plot.")
                return

            # ✅ Filter telemetry data
            filtered_columns = [col for col in self.curr_data.columns if col.startswith('F')]
            self.tele_data = self.curr_data[filtered_columns]

            x_data = self.tele_data.index
            y_data = self.tele_data[parameter]

            figure = go.Figure()
            figure.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=parameter))
            figure.update_layout(
                title=f'Telemetry Plot for {parameter}',
                xaxis_title='Counter',
                yaxis_title=parameter,
                height=450
            )

            # ✅ Save Plotly HTML in a writable location
            output_html = os.path.join(os.path.abspath("."), "telemetryplot.html")
            figure.write_html(output_html)

            # ✅ Load in WebView
            local_url = QUrl.fromLocalFile(output_html)
            self.tplot_widget.webviewTele.setUrl(local_url)

            if self.tabplots.get('telemetry'):
                self.web_view.setUrl(local_url)

        except Exception as e:
            self.open_Error(e)
            print("❌ plot_telemetry ERROR:", str(e))


    def add_plot_ad(self):
        try:
            self.adplot_widget = adPlot(self.defect_sheet[0])
            self.ui.graphLayout.addWidget(self.adplot_widget)
            #Buttons
            self.adplot_widget.closeAnamoly.clicked.connect(self.adplot_widget.close_window)
            self.plot_pipe3d()
            self.tabplots['anomaly'] = self.adplot_widget
        except Exception as e:
            self.open_Error(e)

    def plot_pipe3d(self):
        try:
            # figure = go.Figure()
            # figure.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=parameter))
            # figure.update_layout(title=f'Telemetry Plot for {parameter}', xaxis_title='Time', yaxis_title=parameter,height=450)

            file_path = 'backend/files/pipe3d.html'
            # figure.write_html(file_path)
            script_dir = os.path.dirname(__file__)
            full_path = os.path.join(script_dir, file_path)
            local_url = QUrl.fromLocalFile(full_path)
            self.adplot_widget.webviewAna.setUrl(local_url)
            # if self.tabplots['anomaly']:
            self.web_view.setUrl(local_url)
        except Exception as e:
            self.open_Error(e)

    def open_Report(self):
        selected_columns = [r"Abs. Distance (m)", r"Depth %", r"Type", r"ERF (ASME B31G)", r"Orientation o' clock"]

        if not isinstance(self.pipe_tally, pd.DataFrame):
            QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.")
            return
        try:
            fil_tally = self.pipe_tally[selected_columns].copy()
        except KeyError as e:
            print("❌ Column(s) missing:", e)
            QMessageBox.critical(self, "Error", f"Missing column(s): {e}")
            return

        # Type conversion
        fil_tally = fil_tally.dropna(subset=["Abs. Distance (m)"])
        fil_tally["Abs. Distance (m)"] = fil_tally["Abs. Distance (m)"].astype(int)
        fil_tally["Depth %"] = pd.to_numeric(fil_tally["Depth %"], errors='coerce')
        fil_tally["Type"] = fil_tally["Type"].astype(str)
        fil_tally["ERF (ASME B31G)"] = pd.to_numeric(fil_tally["ERF (ASME B31G)"], errors='coerce')
        fil_tally[r"Orientation o' clock"] = fil_tally[r"Orientation o' clock"].astype(str)

        # ✅ Add missing column
        fil_tally["Surface Location"] = fil_tally["Type"].apply(
            lambda x: "Internal" if "Internal" in x else ("External" if "External" in x else "Unknown")
        )

        self.fr = Report(fil_tally)
        self.fr.show()


        
    def open_Final_Report(self):
        try:
            # Path to the Final Report PDF
            final_report_path = resource_path(os.path.join("final_report", "Final_Report.pdf"))

            # Check if the file exists
            if not os.path.exists(final_report_path):
                self.open_Error("Final report PDF not found.")
                return
            
            os.startfile(final_report_path)  # Open the PDF file using the default viewer

        except Exception as e:
            self.open_Error(f"Error opening Final Report: {str(e)}")

    def open_Preliminary_Report(self):
        try:
            # Path to the Final Report PDF
            final_report_path = resource_path(os.path.join("preliminary_report", "Preliminary_Report.pdf"))

            # Check if the file exists
            if not os.path.exists(final_report_path):
                self.open_Error("Prelimary report is not found.")
                return
            
            os.startfile(final_report_path)  # Open the PDF file using the default viewer


        except Exception as e:
            self.open_Error(f"Error opening Preliminary report: {str(e)}")

    def open_pipe_tally(self):
        try:
            # Path to the Final Report PDF
            final_report_path = resource_path(os.path.join("pipetally", "pipe_tally.xlsx"))

            # Check if the file exists
            if not os.path.exists(final_report_path):
                self.open_Error("Pipetally not found.")
                return
            
            os.startfile(final_report_path)  # Open the PDF file using the default viewer


        except Exception as e:
            self.open_Error(f"Error opening Pipe_tally: {str(e)}")

    def open_manual(self):
        try:
            # Path to the Final Report PDF
            final_report_path = resource_path(os.path.join("manual" , "user_manual.pdf"))

            # Check if the file exists
            if not os.path.exists(final_report_path):
                self.open_Error("User manual is not found.")
                return
            
            os.startfile(final_report_path)  # Open the PDF file using the default viewer


        except Exception as e:
            self.open_Error(f"Error opening User manual: {str(e)}")

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

    def open_About(self):
        ad = About_Dialog()
        ad.exec()

    def open_Admin(self):
        self.ap = Admin_Panel()
        self.ap.show()

    def open_ERF(self):
        self.erf = ERF()
        
        def update_result():
            OD = self.erf.doubleSpinBox.value()
            WT = self.erf.doubleSpinBox_3.value()
            SMYS = self.erf.doubleSpinBox_2.value()
            MAOP = self.erf.doubleSpinBox_4.value()
            SF = self.erf.doubleSpinBox_5.value()
            Axial_L = self.erf.doubleSpinBox_8.value()
            Depth_P = self.erf.doubleSpinBox_9.value()
            
            if OD == 0 or WT == 0 or SF == 0:
                self.erf.lineEdit_2.setText("-")
                self.erf.lineEdit_3.setText("-")
                return

            flow_stress = 1.1 * SMYS

            z_factor = (Axial_L ** 2) / (OD * WT)
            x = 1 + 0.8 * z_factor
            Building_stress_magmification_factor_M = pow(x, 1 / 2)
            y = 1 - 2 / 3 * Depth_P / WT
            z = 1 - 2 / 3 * Depth_P / WT / Building_stress_magmification_factor_M
            k = y / z

            if z_factor <= 20:
                Estimated_failure_stress_level_SF = flow_stress * k
            else:
                Estimated_failure_stress_level_SF = flow_stress * (1 - Depth_P / WT)

            EFP = (2 * Estimated_failure_stress_level_SF * WT) / OD             #Estimated Failure Pressure
            PSafe = EFP / SF

            if PSafe == 0:
                self.erf.lineEdit_2.setText("-")
                self.erf.lineEdit_3.setText("-")
                return
            ERF = MAOP / PSafe

            self.erf.lineEdit_2.setText(f"{ERF:.2f}")
            self.erf.lineEdit_3.setText(f"{PSafe:.2f}")


            #ASME Plot
            def calculate_B(d_over_t):
                if d_over_t >= 0.175:
                    B = np.sqrt(((d_over_t / (1.1 * d_over_t - 0.15)) ** 2) - 1)
                    return B if B <= 4 else 4
                else:
                    return 4
            d_over_t_values = np.linspace(0, 1, 100)
            B_values = [calculate_B(d_over_t) for d_over_t in d_over_t_values]

            X_coord = Axial_L/300                 
            Y_coord = Depth_P/20  
            curve_B_value = calculate_B(X_coord)
            color = 'green' if Y_coord < curve_B_value else 'red'

            figERF = go.Figure()

            figERF.add_trace(go.Scatter(x=d_over_t_values, y=B_values, mode='lines', name='ASME B31G'))
            v = 'Unacceptable' if color == 'red' else 'Acceptable'
            figERF.add_trace(go.Scatter(
                x=[X_coord],
                y=[Y_coord],
                mode='markers',
                marker=dict(color=color, size=10),
                name=f'Defect :{v}'
            ))

            figERF.update_layout(
                xaxis_title='Axial Length (mm)',
                yaxis_title='Peak Depth',
                title='',
                xaxis=dict(
                    range=[0, 1],  
                    tickvals=np.linspace(0, 1, 7),  
                    ticktext=[f'{int(val*300)}' for val in np.linspace(0, 1, 7)]  
                ),
                yaxis=dict(
                    range=[0, 5],  
                    tickvals=np.linspace(0, 5, 6), 
                    ticktext=[f'{int(val*20)}' for val in np.linspace(0, 5, 6)] 
                ),
                height = 450,
                width = 1000
            )
            file_path = 'backend/files/ASME.html'
            figERF.write_html(file_path)
            self.erf.web_viewERF.setUrl(QUrl.fromLocalFile(os.path.join(os.path.dirname(__file__), file_path)))


        self.erf.doubleSpinBox.valueChanged.connect(update_result)
        self.erf.doubleSpinBox_3.valueChanged.connect(update_result)
        self.erf.doubleSpinBox_2.valueChanged.connect(update_result)
        self.erf.doubleSpinBox_4.valueChanged.connect(update_result)
        self.erf.doubleSpinBox_5.valueChanged.connect(update_result)
        self.erf.doubleSpinBox_8.valueChanged.connect(update_result)
        self.erf.doubleSpinBox_9.valueChanged.connect(update_result)

        update_result()
 
        file_path = 'backend/files/ASME.html'
        self.erf.web_viewERF.setUrl(QUrl.fromLocalFile(os.path.join(os.path.dirname(__file__), file_path)))

        self.erf.show()

    def open_XYZ(self):
        try:
            # Path to the Google Earth executable
            if sys.platform == "win32":
                google_earth_path = r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe"  # Adjust the path if necessary
            elif sys.platform == "darwin":  # macOS
                google_earth_path = "/Applications/Google Earth Pro.app/Contents/MacOS/Google Earth Pro"
            else:
                google_earth_path = "/usr/bin/google-earth-pro"  # Example for Linux, adjust as needed

            # Check if the path exists
            if os.path.exists(google_earth_path):
                # Open Google Earth application
                subprocess.Popen([google_earth_path])
            else:
                print(f"Google Earth not found at: {google_earth_path}")
        
        except Exception as e:
            print(f"An error occurred: {e}")

    def open_pipe_highlights(self):
        self.pipe_window = PipeHighlightApp()
        self.pipe_window.show()


    def open_Cluster(self):
        cl = Cluster_Dialog()
        cl.exec()

    # def open_PipeScheme(self):
    #     subprocess.Popen(["python", "pipeline_schema/pipeline_schema.py"])
    def open_pipeline_schema(self):
        script_path = os.path.join(os.path.dirname(__file__), "pipeline_schema", "pipeline_schema.py")
        subprocess.Popen([sys.executable, script_path])

    def open_Assessment(self):
        assess = Assess_Dialog()
        assess.exec()

    # def open_Error(self, exception):
    #     self.error = Error_Dialog(str(exception))
    #     self.error.exec()

    def open_Error(self, e):
        '''Display a custom error dialog with large window and small text.'''
        try:
            if 'error' not in self.child_windows or not self.child_windows['error'].isVisible():
                dialog = QDialog(self)
                dialog.setWindowTitle("Error")
                dialog.resize(700, 400)

                layout = QVBoxLayout(dialog)

                error_text = QTextEdit()
                error_text.setReadOnly(True)
                error_text.setText(str(e))
                error_text.setStyleSheet("font-size: 10pt; font-family: Consolas; color: #aa0000;")
                layout.addWidget(error_text)

                close_btn = QPushButton("Close")
                close_btn.clicked.connect(dialog.accept)
                layout.addWidget(close_btn)

                self.child_windows['error'] = dialog
                dialog.exec()

            else:
                self.child_windows['error'].raise_()
                self.child_windows['error'].activateWindow()

        except Exception as err:
            print(f"Failed to open error dialog: {err}")


    def plot_heatmap(self, flag=0):
        try:
            self.set_loading_status()

            # Create and start the worker thread
            self.worker = Worker(self.curr_data, self.ui.comboBoxMode.currentIndex())
            self.worker.finished.connect(self.handle_worker_finished)
            self.worker.error.connect(self.open_Error)
            self.worker.start()

        except Exception as e:
            self.open_Error(str(e))

    def handle_worker_finished(self, result):
        ds, mode_index = result
        self.defect_sheet = ds
        script_dir = os.path.dirname(__file__)

        if mode_index == 1:
            file_path = 'backend/files/lineplot.html'
            full_path = os.path.join(script_dir, file_path)
            local_url = QUrl.fromLocalFile(full_path)
            self.web_view.setUrl(local_url)

            file_path = 'backend/files/lineplot_raw.html'
            full_path = os.path.join(script_dir, file_path)
            local_url = QUrl.fromLocalFile(full_path)
            self.web_view2.setUrl(local_url)

        else:
            file_path = 'backend/files/heatmap.html'
            full_path = os.path.join(script_dir, file_path)
            local_url = QUrl.fromLocalFile(full_path)
            self.web_view.setUrl(local_url)

            file_path = 'backend/files/heatmap_raw.html'
            full_path = os.path.join(script_dir, file_path)
            local_url = QUrl.fromLocalFile(full_path)
            self.web_view2.setUrl(local_url)

        self.set_idle_status()

    
    def draw_boxes(self):
        try:
            self.set_loading_status()
            if self.ui.comboBoxMode.currentIndex() != 1:
                file_path = 'backend/files/heatmap_box.html'
                script_dir = os.path.dirname(__file__)
                full_path = os.path.join(script_dir, file_path)
                local_url = QUrl.fromLocalFile(full_path)
                self.web_view.setUrl(local_url)

                column_mapping = {
                    'Box Number': 'Defect_id',
                    'Type': 'Feature_Type',
                    'Width': 'Length',
                    'Absolute Distance': 'Absolute_Distance',
                    'Peak Value': 'Depth_Peak',
                    'Breadth':'Width',
                    'Ori Val':'Orientation'
                }
                header_indices = {
                    'Defect_id': 0,
                    'Absolute_Distance': 1,
                    'Upstream_Distance': 2,
                    'Feature_Type': 3,
                    'Dimension_Class': 4,
                    'Orientation': 5,
                    'WT': 6,
                    'Length': 7,
                    'Width': 8,
                    'Depth_Peak': 9
                }
                def populate_table_widget(table_widget: QTableWidget, df: pd.DataFrame, column_mapping: dict):
                    num_rows = len(df)
                    num_cols = len(header_indices)
                    table_widget.setRowCount(num_rows)
                    table_widget.setColumnCount(num_cols)
                    headers = list(header_indices.keys())
                    table_widget.setHorizontalHeaderLabels(headers)

                    for row_idx, (_, row) in enumerate(df.iterrows()):
                        for src_col, dest_col in column_mapping.items():
                            if dest_col in header_indices:
                                col_idx = header_indices[dest_col]
                                value = row[src_col]
                                
                                # Format the value to 2 decimal places if it's a float
                                if isinstance(value, float):
                                    value = f"{value:.2f}"

                                table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

                populate_table_widget(self.ui.tableWidgetDefect, self.defect_sheet, column_mapping)

            else:
                raise Exception("Heatmap mode is not selected")
            self.set_idle_status()
        except Exception as e:
            self.open_Error(e)

    def on_row_selection_changed(self, selected, deselected):
        print("Selection changed")
        indexes = self.ui.tableWidgetDefect.selectionModel().selectedRows()
        if indexes:
            row = indexes[0].row()
            defect_id_item = self.ui.tableWidgetDefect.item(row, 0)  # Assuming column 0 has the Defect_id
            if defect_id_item:
                defect_id = defect_id_item.text()
                print(f"Defect id no: {defect_id}")
                self.highlight_box_in_webview(defect_id)
            else:
                print("Defect id item not found")
        else:
            print("No rows selected")

    def highlight_box_in_webview(self, defect_id):
        js_code = f"highlightBox({defect_id});"
        self.ui.web_view.page().runJavaScript(js_code)

    def close_window(win):
        win.close()

    def quit_app(self):
        QApplication.quit()

    def get_checked_tally(self, columns):
        if not isinstance(self.pipe_tally, pd.DataFrame):
            QMessageBox.critical(self, "Error", "Pipe tally data is missing or not loaded.")
            return None
        for col in columns:
            if col not in self.pipe_tally.columns:
                QMessageBox.critical(self, "Error", f"Missing column: {col}")
                return None
        return self.pipe_tally[columns].copy()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())