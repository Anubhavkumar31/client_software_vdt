# utils/assets.py
import glob
import os
import pandas as pd
from config.constants import PIPE_TALLY_FILE_PATTERNS, HTML_ASSET_PATTERNS
from utils.data_processing import process_table_data

def pick_one(patterns, base_dir=".") -> str | None:
    """
    Return the first file matching given patterns inside base_dir.
    """
    for pat in patterns:
        matches = glob.glob(os.path.join(base_dir, pat))
        if matches:
            return matches[0]
    return None

def load_html_assets(base_dir: str) -> dict[str, list[str]]:
    """
    Load available HTML assets (heatmap, line plot, 3D graph, proximity linechart).
    Returns dict mapping type -> list of file paths.
    """
    assets: dict[str, list[str]] = {}
    for key, (patterns, _) in HTML_ASSET_PATTERNS.items():
        found = []
        for pat in patterns:
            found.extend(glob.glob(os.path.join(base_dir, pat)))
        if found:
            assets[key] = found
    return assets

def load_pipe_tally_data(path: str) -> pd.DataFrame:
    """
    Load a pipe tally file (CSV/XLSX) and preprocess it for display.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported tally file format: {ext}")
    return process_table_data(df)
