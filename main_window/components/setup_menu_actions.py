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
    setup_actions(self)
    self._connect_guarded_graph_controls()

    try:
        self.ui.tabWidgetM.currentChanged.disconnect()
    except:
        pass

    self.ui.tabWidgetM.currentChanged.connect(self._on_middle_tab_changed)
    self.ui.tabWidgetM.currentChanged.connect(self.syncdropdownwithtabs)


def setup_actions(self):
    a = self.ui
    a.action_Create_Proj.triggered.connect(self.open_project)
    a.action_Close_Proj.triggered.connect(self.close_project)
    a.action_Quit.triggered.connect(self.quit_app)
    a.action_About.triggered.connect(self.open_About)
    a.actionAdmin_Panel.triggered.connect(self.open_Admin)
    a.action_ERF.triggered.connect(self.open_ERF)
    a.action_XYZ.triggered.connect(self.open_XYZ)
    # self.ui.action_Export_Table.triggered.connect(self.gen_data)
    a.action_Final_Report.triggered.connect(self.open_Report)
    a.action_graphs.triggered.connect(self.open_graphs)
    a.action_Assessment.triggered.connect(self.open_Assessment)
    a.action_Cluster.triggered.connect(self.open_Cluster)
    a.action_Pipe_High.triggered.connect(self.open_PipeHigh)
    a.action_Pipe_Sch.triggered.connect(self.open_PipeScheme)
    a.actionMetal_Loss_Distribution_MLD.triggered.connect(self.open_CMLD)
    a.actionDepth_Based_Anomalies_Distribution_DBAD.triggered.connect(self.open_DBAD)
    a.actionERF_Based_Anomalies_Distribution_E_AD.triggered.connect(self.open_EAD)
    a.action_Custom.triggered.connect(self.add_plot_custom)
    a.action_Telemetry.triggered.connect(self.add_plot_tele)
    a.actionAnomalies_Distribution.triggered.connect(self.add_plot_ad)
    a.action_DefectDetect.triggered.connect(self.draw_boxes_v2)
    if hasattr(a, "pushButtonNext"): a.pushButtonNext.clicked.connect(self.load_next_pipe)
    if hasattr(a, "pushButtonPrev"): a.pushButtonPrev.clicked.connect(self.load_prev_pipe)
    a.Final_Report.triggered.connect(self.open_Final_Report)
    a.action_Preliminary_Report.triggered.connect(self.open_Preliminary_Report)
    a.action__pipetally.triggered.connect(self.open_pipe_tally)
    a.action_Manual.triggered.connect(self.open_manual)
    a.actionStandard.triggered.connect(self.open_digs)  # original (by defect no.)