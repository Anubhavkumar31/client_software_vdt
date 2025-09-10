# config/constants.py
"""
Application constants and data structures
Used across UI, pipe tally processing, reporting, and graph handling.
"""

import numpy as np

# ================================
# PIPE TALLY REQUIREMENTS
# ================================
REQUIRED_TALLY_COLS = [
    r"Abs. Distance (m)",
    r"Depth %",
    r"Type",
    r"ERF (ASME B31G)",
    r"Orientation o' clock",
]

NUMERIC_COLUMNS = [
    "Depth %", "Depth (mm)", "ERF (ASME B31G)", "Psafe (ASME B31G) Barg",
    "Abs. Distance (m)", "Distance to U/S GW(m)", "Length (mm)",
    "Width (mm)", "WT (mm)", "Pipe Length (mm)",
]

# ================================
# TABLE COLUMN CONFIGURATIONS
# ================================
COLUMN_WIDTHS = {
    "Defect_id": 150,
    "Abs. Distance (m)": 150,
    "Distance to U/S GW(m)": 150,
    "Pipe Number": 150,
    "Pipe Length (mm)": 150,
    "Feature Identification": 150,
    "Dimensions Classification": 150,
    "Orientation o' clock": 150,
    "Length (mm)": 150,
    "Width (mm)": 150,
    "WT (mm)": 150,
    "Depth %": 150,
    "Depth (mm)": 150,
    "Type": 150,
    "ERF (ASME B31G)": 150,
    "Psafe (ASME B31G) Barg": 150,
    "Latitude": 150,
    "Longitude": 150,
    "Comment": 570,
}

DESIRED_COLUMNS = [
    "Defect_id", "Abs. Distance (m)", "Distance to U/S GW(m)", "Pipe Number",
    "Pipe Length (mm)", "Feature Identification", "Dimensions Classification",
    "Orientation o' clock", "Length (mm)", "Width (mm)", "WT (mm)", "Depth %",
    "Depth (mm)", "Type", "ERF (ASME B31G)", "Psafe (ASME B31G) Barg",
    "Latitude", "Longitude", "Comment",
]

COLUMN_VARIANTS = {
    "s_no": "Defect_id",
    "Dimensions  Classification": "Dimensions Classification",
    "Depth % ": "Depth %",
    "Psafe (ASME B31G) bar": "Psafe (ASME B31G) Barg",
    "Pipe Length": "Pipe Length (mm)",
    "Length": "Length (mm)",
    "Width": "Width (mm)",
    "WT": "WT (mm)",
}

HEADER_INDICES = {
    "Defect_id": 0,
    "Absolute_Distance": 1,
    "Upstream_Distance": 2,
    "Feature_Type": 3,
    "Dimension_Class": 4,
    "Orientation": 5,
    "WT": 6,
    "Length": 7,
    "Width": 8,
    "Depth_Peak": 9,
}

COLUMN_MAPPING_CANDIDATES = {
    "Box Number": "Defect_id",
    "Defect_id": "Defect_id",
    "Absolute Distance": "Absolute_Distance",
    "Abs. Distance (m)": "Absolute_Distance",
    "Upstream": "Upstream_Distance",
    "Distance to U/S GW(m)": "Upstream_Distance",
    "Type": "Feature_Type",
    "Dimensions  Classification": "Dimension_Class",
    "Orientation o' clock": "Orientation",
    "Ori Val": "Orientation",
    "WT (mm)": "WT",
    "WT": "WT",
    "Width": "Width",
    "Breadth": "Width",
    "Peak Value": "Depth_Peak",
    "Depth % ": "Depth_Peak",
    "Depth %": "Depth_Peak",
    "Length": "Length",
}

ABS_DISTANCE_CANDIDATES = (
    "Absolute_Distance",
    "Abs. Distance (m)",
    "Absolute Distance",
)

# ================================
# TAB NAMES FOR GRAPH SWITCHING
# ================================
GRAPH_TAB_NAMES = {
    "Heatmap": {"Heatmap"},
    "LineChart": {"LineChart", "Line Chart", "Line Plot"},
    "3D": {"3D Graph", "3D"},
}
GRAPH_TAB_NAMES_SET = {"Heatmap", "LineChart", "Line Chart", "Line Plot", "3D Graph", "3D"}
VALID_DIGSHEET_TABS = ("Heatmap", "3D Graph", "3D")

# ================================
# FILE PATTERNS
# ================================
HTML_ASSET_PATTERNS = {
    "hmap": (["*heatmap*.html"], ["raw", "box"]),
    "hmap_r": (["*heatmap*raw*.html", "*raw*heatmap*.html"], []),
    "heatmap_box": (["*heatmap*box*.html", "*box*heatmap*.html"], []),
    "lplot": (["*lineplot*.html", "*line*.html"], ["raw"]),
    "lplot_r": (["*lineplot*raw*.html", "*line*raw*.html"], []),
    "pipe3d": (["*pipe3d*.html", "pipe3d*.html"], []),
    "prox_linechart": (["proximity_linechart*.html", "*proximity_linechart*.html"], []),
}
PIPE_TALLY_PATTERNS = ["*PipeTally*.csv", "*PipeTally*.xlsx", "*defectS*.csv", "*defects*.csv"]
KML_PATTERNS = ["*.kml", "*.KML"]
PIPE_TALLY_FILE_PATTERNS = ["pipe_tally.xlsx", "pipe_tally.csv"]
TALLY_FILE_REGEX = r".*(pipe.*tally|tally.*pipe|pipetally|pipe_tally|pipe-tally).*\.(xlsx?|csv)$"

# ================================
# REPORT COLUMNS
# ================================
REPORT_REQUIRED_COLS = [
    r"Abs. Distance (m)", r"Depth %", r"Type",
    r"ERF (ASME B31G)", r"Orientation o' clock",
]
MLD_COLUMNS = [r"Abs. Distance (m)", r"Type", r"Orientation o' clock"]
DBAD_COLUMNS = [r"Abs. Distance (m)", r"Depth %", r"Type"]
EAD_COLUMNS = [r"Abs. Distance (m)", r"Type", r"ERF (ASME B31G)"]

# ================================
# PROCESSING CONSTANTS
# ================================
DEFAULT_DEFECT_ID_START = 1
FLOAT_PRECISION = 6
ROUNDING_PRECISION = 3
DEFAULT_ROW_HEIGHT = 40
DEFAULT_COLUMN_WIDTH = 100
VIRTUAL_SCROLLBAR_MAX = 2000
MAGNETISATION_FACTOR = 0.0004854
VELOCITY_FACTOR = 0.000666667

# ================================
# UI / LAYOUT CONSTANTS
# ================================
HANDLE_RADIUS = 10
SPLITTER_MIN = 100
SPLITTER_MAX = 8000
SCROLLBAR_BTN_WH = 22
SCROLLBAR_BAR_H = 14
SCROLLBAR_BAR_W = 16
DEFAULT_RIGHT_MARGIN = 200
