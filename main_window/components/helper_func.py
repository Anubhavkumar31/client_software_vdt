import os

import pandas as pd
from PyQt6.QtGui import QAction


def _update_project_actions(self):
    a = self.ui
    act_create = getattr(a, "action_Create_Proj", None)
    act_close = getattr(a, "action_Close_Proj", None)
    act_graphs = getattr(a, "action_graphs", None)
    act_xyz = getattr(a, "action_XYZ", None)
    act_pipehigh = getattr(a, "action_Pipe_High", None)
    if isinstance(act_create, QAction):
        act_create.setEnabled(not self.project_is_open)
    if isinstance(act_close, QAction):
        act_close.setEnabled(self.project_is_open)
    if isinstance(act_graphs, QAction):
        act_graphs.setEnabled(self.project_is_open)
    if isinstance(act_xyz, QAction):  # ← Add this block
        act_xyz.setEnabled(self.project_is_open)
    if isinstance(act_pipehigh, QAction):  # ← ADD THIS BLOCK
        act_pipehigh.setEnabled(self.project_is_open)
    _update_generate_actions(self)


def _update_generate_actions(self):
    """Update Generate menu buttons based on project and data status"""
    # Check if pipe tally data is available
    has_pipe_tally = isinstance(self.pipe_tally, pd.DataFrame) and not self.pipe_tally.empty

    # Check if preliminary report exists
    has_prelim_report = False
    if self.project_is_open and self.project_root:
        report_dir = os.path.join(self.project_root, "report")
        prelim_report_path = os.path.join(report_dir, "PR.pdf")
        has_prelim_report = os.path.exists(prelim_report_path)

    # Check if final report exists
    has_final_report = False
    if self.project_is_open and self.project_root:
        report_dir = os.path.join(self.project_root, "report")
        final_report_path = os.path.join(report_dir, "FR.pdf")
        has_final_report = os.path.exists(final_report_path)

    # Update BOTH Final Report actions ✅
    if hasattr(self.ui, 'action_Final_Report'):
        self.ui.action_Final_Report.setEnabled(self.project_is_open and has_final_report)

    if hasattr(self.ui, 'Final_Report'):  # ← Add this block
        self.ui.Final_Report.setEnabled(self.project_is_open and has_final_report)

    # Update Pipe Tally button/action
    if hasattr(self.ui, 'action__pipetally'):
        self.ui.action__pipetally.setEnabled(self.project_is_open and has_pipe_tally)

    # Update Preliminary Report action
    if hasattr(self.ui, 'action_Preliminary_Report'):
        self.ui.action_Preliminary_Report.setEnabled(self.project_is_open and has_prelim_report)

    # Update Digsheet actions (both standard and ABS-based)
    if hasattr(self.ui, 'actionStandard'):  # Standard digsheet
        self.ui.actionStandard.setEnabled(self.project_is_open and has_pipe_tally)


def _close_graphs_view(self):
    try:
        if self.centralWidget() is self._central_original:
            return
        graphs_central = self.takeCentralWidget()
        if graphs_central is not None:
            graphs_central.deleteLater()
        if self._central_original is not None:
            if self._central_original.parent() is not self:
                self._central_original.setParent(self)
            self.setCentralWidget(self._central_original)
        self._graphs_widget = None
        self._central_graphs = None
    except Exception as e:
        print("⚠️ _close_graphs_view:", e)


def create_instances(self):
    self.child_windows = {}

    self._central_original = self.centralWidget()
    self._central_graphs = None
    self._graphs_widget = None

    self.project_is_open = False
    self.project_root = None
    self.pkl_files = []
    self.curr_data = None
    self.header_list = []
    self.pipe_tally = None
    self.prox_linechart = None

    self.hmap = None
    self.hmap_r = None
    self.lplot = None
    self.lplot_r = None
    self.pipe3d = None
    self.heatmap_box = None
    self._hscroll_ready = False  # gate to avoid big first jump
    self._hscroll_ready_main = False  # gate for main web view scrollbar
    # --- Splitter limits (pixels) ---
    self._min_top_h = 220  # top pane (charts) must be at least this tall
    self._min_bottom_h = 250  # bottom pane (tables/proximity) must be at least this tall
    self._max_top_h = None  # or set e.g. 900
    self._max_bottom_h = None  # or set e.g. 900
    self._right_margin_px = 300
    self._hscroll_ready_table = False  # gate for table scrollbar... # guard state
    self._reverting_tab = False
    self._last_allowed_tab_index = 0
    self._ui_ready = False  # set true after first layout/show
    self._selected_columns: set[str] = set()
    self.hhmap = None  # hallsensor_heatmap*.html
    self.phmap = None  # proximity_heatmap*.html
    self._hm_layout_mode = "vertical"  # "horizontal" = side-by-side, "vertical" = stacked
    self.hm_left_ratio = 0.40  # 50-50 split in side-by-side mode

    # ✅ Initialize "No Defects Found" label
    self._no_defects_label = None

    # Threading setup
    self.loader_worker = None
    self.loading_dialog = None

    self._unit_columns = {
        "Abs. Distance": "m",
        "Distance to U/S GW": "m",
        "Pipe Length": "mm",
        "WT": "mm",
        "Width": "mm",
        "Length": "mm",
        "Depth": "mm",
    }
    self._unit_factor = {
        "m": 1.0,
        "cm": 100.0,
        "mm": 1000.0,
        "km": 0.001,
        "feet": 3.28084,
    }
