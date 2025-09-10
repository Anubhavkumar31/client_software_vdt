# config/paths.py
import os
import sys
import tempfile
import uuid
from PyQt6 import uic

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and PyInstaller bundles.
    """
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, relative_path)  # noqa: PYI056
    return os.path.join(os.path.abspath("."), relative_path)

def dump_tally_to_temp(df) -> str:
    """
    Dump a pipe tally DataFrame to a temporary pickle file.
    Returns the file path.
    """
    import pickle
    p = os.path.join(tempfile.gettempdir(), f"pipe_tally_{uuid.uuid4().hex}.pkl")
    with open(p, "wb") as f:
        pickle.dump(df, f)
    return p

def _dump_tally_to_temp(df):
    import pickle
    p = os.path.join(tempfile.gettempdir(), f"pipe_tally_{uuid.uuid4().hex}.pkl")
    with open(p, "wb") as f: pickle.dump(df, f)
    return p

# --- UI file paths and compiled Qt classes ---
# Treat project root as parent of this file's directory
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ui_path_splash = resource_path(os.path.join(_base_dir, "ui", "landing.ui"))
ui_path_main   = resource_path(os.path.join(_base_dir, "ui", "main_window.ui"))

# Load .ui at import time (matches your current approach)
SplashScreen, SplashWindow = uic.loadUiType(ui_path_splash)
Form, Window               = uic.loadUiType(ui_path_main)
ICON_ARROW_DOWN = resource_path(os.path.join("ui", "icons", "arrow_down.svg"))
