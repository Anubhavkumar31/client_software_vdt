import pandas as pd
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QPen, QPolygonF, QPainter
from PyQt6.QtWidgets import QMessageBox, QGraphicsPolygonItem, QGraphicsView, QGraphicsScene, QPushButton, QLineEdit, \
    QLabel

from main_window.components.main_section_view.helpers_temp import _on_middle_tab_changed, syncdropdownwithtabs, _connect_guarded_graph_controls
from main_window.components.main_section_view.workers.load_button_working import load_prev_pipe, load_next_pipe
from main_window.components.main_section_view.utils import update_load_button_state
from main_window.components.menubar.File_menu.close_project import close_project
from main_window.components.menubar.File_menu.open_project import open_project
from main_window.components.menubar.File_menu.quit_app import quit_app
from main_window.components.menubar.help_menu.open_about import open_About
from main_window.components.menubar.help_menu.open_manual import open_manual
from main_window.components.menubar.report_menu.generate.digsheet import digsheet_runner
from main_window.components.menubar.report_menu.generate.final_report import open_Final_Report
from main_window.components.menubar.report_menu.generate.pipetally import open_pipe_tally
from main_window.components.menubar.report_menu.generate.preliminary_report import open_Preliminary_Report
from main_window.components.menubar.report_menu.open_PipeScheme import open_PipeScheme
from main_window.components.menubar.report_menu.open_pipehigh import open_PipeHigh
from main_window.components.menubar.view_menu.open_ERF import open_ERF
from main_window.components.menubar.view_menu.open_XYZ import open_XYZ
from main_window.components.menubar.view_menu.open_customplot import open_customplot
from main_window.components.menubar.view_menu.open_graphs import open_graphs
from main_window.components.menubar.view_menu.open_pipe_locator import open_pipe_locator
from main_window.components.menubar.view_menu.open_pipetally import open_pipetally


def setup_menu_actions(self):
    """
    -------------------------------------------------------------
    MENU ACTIONS + GRAPH CONTROL SIGNALS
    -------------------------------------------------------------
    Connects all QAction menu items to their handlers.
    Also initializes graph-related UI controls and guards
    them so they activate only when a project/pipe is loaded.

    Replaces default tab change handler with guarded switching.
    -------------------------------------------------------------
    """
    #normal button connections
    setup_actions(self)

    # load button state on depend on whats in pipe number selection dropdown
    self.ui.comboBoxPipe.currentIndexChanged.connect(lambda idx: update_load_button_state(self, idx))

    #heatmap/linechart/3dgraph guarded connections
    _connect_guarded_graph_controls(self)

    try:
        self.ui.tabWidgetM.currentChanged.disconnect()
    except:
        pass

    self.ui.tabWidgetM.currentChanged.connect(lambda index: _on_middle_tab_changed(self, index))
    self.ui.tabWidgetM.currentChanged.connect(lambda index: syncdropdownwithtabs(self, index))













def setup_actions(self):
    a = self.ui
    #File menu section
    a.action_Create_Proj.triggered.connect(lambda: open_project(self))
    a.action_Close_Proj.triggered.connect(lambda: close_project(self))
    a.action_Quit.triggered.connect(lambda: quit_app(self))



    #View section
    a.action_Pipe_Locator.triggered.connect(lambda: open_pipe_locator(self))
    self.ui.action_View_Clusters.triggered.connect(self.on_view_clusters_clicked)
    a.action_ERF.triggered.connect(lambda: open_ERF(self))
    a.action_XYZ.triggered.connect(lambda: open_XYZ(self))
    a.action_customplot.triggered.connect(lambda : open_customplot(self))
    a.action_pipetally.triggered.connect(lambda: open_pipetally(self, self.pipetally_dir))
    a.action_graphs.triggered.connect(lambda: open_graphs(self))
    # self.ui.action_Export_Table.triggered.connect(self.gen_data)

    #report section
    a.action_Pipe_High.triggered.connect(lambda: open_PipeHigh(self))
    a.action_Pipe_Sch.triggered.connect(lambda: open_PipeScheme(self))
    a.Final_Report.triggered.connect(lambda: open_Final_Report(self))
    a.action_Preliminary_Report.triggered.connect(lambda: open_Preliminary_Report(self))
    # a.actionStandard.triggered.connect(lambda: open_digs(self))  # original (by defect no.)
    a.actionStandard.triggered.connect(lambda: digsheet_runner(self))
    a.action__pipetally.triggered.connect(lambda: open_pipe_tally(self))

    #help section
    a.action_Manual.triggered.connect(lambda: open_manual(self))
    a.action_About.triggered.connect(lambda: open_About(self))

    if hasattr(a, "pushButtonNext"): a.pushButtonNext.clicked.connect(lambda : load_next_pipe(self))
    if hasattr(a, "pushButtonPrev"): a.pushButtonPrev.clicked.connect(lambda : load_prev_pipe(self))


    #extra
    # a.action_Final_Report.triggered.connect(self.open_Report)
    # a.action_Assessment.triggered.connect(self.open_Assessment)
    # a.action_Cluster.triggered.connect(self.open_Cluster)
    # a.actionMetal_Loss_Distribution_MLD.triggered.connect(self.open_CMLD)
    # a.actionDepth_Based_Anomalies_Distribution_DBAD.triggered.connect(self.open_DBAD)
    # a.actionERF_Based_Anomalies_Distribution_E_AD.triggered.connect(self.open_EAD)
    # a.action_Custom.triggered.connect(self.add_plot_custom)
    # a.action_Telemetry.triggered.connect(self.add_plot_tele)
    # a.actionAnomalies_Distribution.triggered.connect(self.add_plot_ad)
    # a.action_DefectDetect.triggered.connect(self.draw_boxes_v2)
    # a.actionAdmin_Panel.triggered.connect(self.open_Admin)



