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