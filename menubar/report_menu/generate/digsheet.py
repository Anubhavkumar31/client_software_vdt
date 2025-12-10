import os
import sys
import tempfile
import uuid

import pandas as pd
from PyQt6.QtWidgets import QMessageBox
import subprocess


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def _dump_tally_to_temp(df):
    import pickle
    p = os.path.join(tempfile.gettempdir(), f"pipe_tally_{uuid.uuid4().hex}.pkl")
    with open(p, "wb") as f: pickle.dump(df, f)
    return p

def open_digs(self):
    try:
        if not self.project_is_open:
            QMessageBox.warning(
                self,
                "No Project Open",
                "Please create/open a project first to generate digsheets."
            )
            return
        if not isinstance(self.pipe_tally, pd.DataFrame):
            QMessageBox.warning(self, "No Pipe Tally", "Load a pipe tally first.")
            return

        tally_pkl = _dump_tally_to_temp(self.pipe_tally)
        dig_py = resource_path(os.path.join("dig", "dig_sheet.py"))
        if not os.path.exists(dig_py):
            QMessageBox.critical(self, "Script not found", f"Missing: {dig_py}")
            return

        if getattr(sys, "frozen", False):
            subprocess.Popen([sys.executable, "--run-digsheet", tally_pkl, self.project_root])
        else:
            subprocess.Popen([sys.executable, dig_py, tally_pkl, self.project_root])
    except Exception as e:
        self.open_Error(f"An error occurred: {e}")