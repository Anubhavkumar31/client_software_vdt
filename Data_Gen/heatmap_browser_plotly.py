#!/usr/bin/env python3
"""
Render the same HEATMAP (with optional red square overlays)
using Plotly instead of Seaborn with both interactive and browser modes.

Hardcode your inputs in the CONFIG section below.

Expected input: a raw .pkl loaded as a pandas DataFrame that includes:
  - ROLL (degrees)
  - ODDO1 (odometer, mm)  [optional; falls back to index if missing]
  - Sensor columns like F1H1..F24H4 (96 total) — any missing ones are ignored

Usage:
  1) pip install pandas numpy plotly scipy
  2) python heatmap_browser_plotly.py
  → Opens interactive window or browser based on INTERACTIVE_MODE setting
"""

import os
# Limit intra-process math threads to avoid over-subscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import re
from pathlib import Path
from glob import glob
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from scipy.signal import savgol_filter
from typing import Optional, Union, List
import tempfile
import webbrowser

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- CONFIG (HARD-CODE THESE) --------------------
PKL_PATH = r"F:\work_new\client_software\test_data_cs\35.pkl"   # <-- CHANGE THIS to your .pkl
ROLL_COL = "ROLL"
ODDO_COL = "ODDO1"
INITIAL_READ = 0.0
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3
PIPETALLY_PATH = r"F:\work_new\client_software\test_data_cs\picle_2\Pipe_35\PipeTally35.csv"



# ==================== Overlay / PipeTally helpers ====================

def _find_pipe_tally_file(pipe_number: Union[str, int]) -> Optional[str]:
    """Recursively search CWD for a per-pipe PipeTally CSV/XLSX."""
    pn = str(pipe_number)
    patterns = [
        f"**/*PipeTally*{pn}*.xlsx", f"**/*Pipe_Tally*{pn}*.xlsx",
        f"**/*PipeTally*{pn}*.csv",  f"**/*Pipe_Tally*{pn}*.csv",
        f"**/{pn}*PipeTally*.xlsx",  f"**/{pn}*PipeTally*.csv",
    ]
    for pat in patterns:
        hits = glob(pat, recursive=True)
        if hits:
            hits.sort(key=len)  # prefer closest
            return hits[0]
    return None

def _pick_col(df: pd.DataFrame, preferred: List[str], tokens: List[str]) -> Optional[str]:
    """Pick a column by exact name (case-insensitive) or by 'contains all tokens'."""
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}

    for name in preferred:
        nlow = name.lower()
        if nlow in low:
            return low[nlow]

    for c in cols:
        cl = c.lower()
        if all(t in cl for t in tokens):
            return c
    return None

def _parse_ori_to_seconds(v) -> Optional[int]:
    """Convert '8', '8.5', '8:30', '08:30:00' → seconds on a 12h dial."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None

    if isinstance(v, (int, float)):
        h = int(v) % 12
        m = int(round((float(v) - int(v)) * 60))
        return h * 3600 + m * 60

    s = str(v).strip().lower()
    s = re.sub(r"[^0-9:.]", "", s)
    if not s:
        return None

    if ":" in s:
        parts = s.split(":")
        try:
            h = int(parts[0]) % 12
            m = int(parts[1]) if len(parts) > 1 else 0
            sec = int(parts[2]) if len(parts) > 2 else 0
            return h * 3600 + m * 60 + sec
        except Exception:
            return None

    try:
        f = float(s)
        h = int(f) % 12
        m = int(round((f - int(f)) * 60))
        return h * 3600 + m * 60
    except Exception:
        pass

    try:
        h = int(s) % 12
        return h * 3600
    except Exception:
        return None

def _hhmmss_to_seconds(t: str) -> int:
    h, m, s = [int(x) for x in str(t).split(":")]
    return (h % 12) * 3600 + m * 60 + s

def _nearest_band_label(seconds: int, band_labels: List[str]) -> str:
    band_labels_str = [str(x) for x in band_labels]
    band_secs = np.array([_hhmmss_to_seconds(lbl) for lbl in band_labels_str], dtype=int)
    idx = int(np.argmin(np.abs(band_secs - seconds)))
    return band_labels_str[idx]

def _load_overlay_points_for_pipe(pipe_number, y_band_labels, pipetally_path=None, *, debug_prefix="OVERLAY DEBUG"):
    """Return (xs, ys, labels) for overlay markers if a PipeTally file is found."""
    path = pipetally_path if pipetally_path else _find_pipe_tally_file(pipe_number)
    if not path:
        print(f"{debug_prefix}: pipe {pipe_number}: no PipeTally file found.")
        return None

    try:
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    except Exception as e:
        print(f"{debug_prefix}: pipe {pipe_number}: failed reading '{path}': {e}")
        return None

    x_col    = _pick_col(df, ["Abs. Distance (m)", "Absolute Distance"], ["abs", "distance"]) \
            or _pick_col(df, [], ["distance"])
    ori_col  = _pick_col(df, ["Orientation o' clock", "Orientation", "Ori"], ["ori"]) \
            or _pick_col(df, [], ["orient"])
    feat_col = _pick_col(df, ["Feature Type", "Feature"], ["feature", "type"])
    pipe_col = _pick_col(df, ["Pipe Number", "Pipe"], ["pipe", "number"])
    sno_col  = _pick_col(df, ["s_no", "S_No", "Serial Number", "SNo"], ["s_no", "sno", "serial"])

    print(f"{debug_prefix}: pipe {pipe_number}: file='{path}' cols: x='{x_col}', ori='{ori_col}', feature='{feat_col}', pipe='{pipe_col}', s_no='{sno_col}'")

    if x_col is None or ori_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: missing required columns; no overlay.")
        return None

    if pipe_col is not None:
        mask = df[pipe_col].astype(str).str.contains(str(pipe_number), na=False)
        if mask.any():
            df = df[mask]

    total_rows = len(df)

    if feat_col is not None:
        ml_mask = df[feat_col].astype(str).str.contains("metal", case=False, na=False)
        if ml_mask.any():
            df = df[ml_mask]

    after_filter = len(df)

    xs, ys, labels = [], [], []
    skipped_no_x, skipped_no_ori = 0, 0

    for _, row in df.iterrows():
        x = pd.to_numeric(row.get(x_col), errors="coerce")
        if pd.isna(x):
            skipped_no_x += 1
            continue

        ori_sec = _parse_ori_to_seconds(row.get(ori_col))
        if ori_sec is None:
            skipped_no_ori += 1
            continue

        y = _nearest_band_label(ori_sec, list(y_band_labels))
        
        if sno_col is not None:
            lbl = row.get(sno_col)
            if not pd.isna(lbl) and lbl is not None and str(lbl).strip() != "":
                lbl = str(lbl)
            else:
                lbl = row.get("Defect_id")
                if pd.isna(lbl) or lbl is None or str(lbl).strip() == "":
                    lbl = str(len(labels) + 1)
                else:
                    lbl = str(lbl)
        else:
            lbl = row.get("Defect_id")
            if pd.isna(lbl) or lbl is None or str(lbl).strip() == "":
                lbl = str(len(labels) + 1)
            else:
                lbl = str(lbl)

        xs.append(float(x))
        ys.append(str(y))
        labels.append(lbl)

    print(f"{debug_prefix}: pipe {pipe_number}: total_rows={total_rows}, after_filter={after_filter}, plotted={len(xs)}, skipped_no_x={skipped_no_x}, skipped_no_ori={skipped_no_ori}")

    if not xs:
        return None
    return xs, ys, labels

# ==================== Heatmap preparation ====================

def _create_time_dict():
    """Create 12-hour dial split into 7.5-minute bands."""
    return {
        key: [] for key in [
            '00:00:00','00:07:30','00:15:00','00:22:30','00:30:00','00:37:30','00:45:00','00:52:30',
            '01:00:00','01:07:30','01:15:00','01:22:30','01:30:00','01:37:30','01:45:00','01:52:30',
            '02:00:00','02:07:30','02:15:00','02:22:30','02:30:00','02:37:30','02:45:00','02:52:30',
            '03:00:00','03:07:30','03:15:00','03:22:30','03:30:00','03:37:30','03:45:00','03:52:30',
            '04:00:00','04:07:30','04:15:00','04:22:30','04:30:00','04:37:30','04:45:00','04:52:30',
            '05:00:00','05:07:30','05:15:00','05:22:30','05:30:00','05:37:30','05:45:00','05:52:30',
            '06:00:00','06:07:30','06:15:00','06:22:30','06:30:00','06:37:30','06:45:00','06:52:30',
            '07:00:00','07:07:30','07:15:00','07:22:30','07:30:00','07:37:30','07:45:00','07:52:30',
            '08:00:00','08:07:30','08:15:00','08:22:30','08:30:00','08:37:30','08:45:00','08:52:30',
            '09:00:00','09:07:30','09:15:00','09:22:30','09:30:00','09:37:30','09:45:00','09:52:30',
            '10:00:00','10:07:30','10:15:00','10:22:30','10:30:00','10:37:30','10:45:00','10:52:30',
            '11:00:00','11:07:30','11:15:00','11:22:30','11:30:00','11:37:30','11:45:00','11:52:30'
        ]
    }

def _degrees_to_hhmmss(degrees: float) -> str:
    """Map roll degrees to a 12h clock string HH:MM:SS."""
    if degrees < 0:
        degrees = degrees % 360
    elif degrees >= 360:
        degrees %= 360
    degrees_per_second = 360 / (12 * 60 * 60)
    total_seconds = degrees / degrees_per_second
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _check_time_range(time_str: str) -> bool:
    start_time = '00:00:00'
    end_time   = '00:07:29'
    t = datetime.strptime(time_str, '%H:%M:%S')
    return datetime.strptime(start_time, '%H:%M:%S') <= t <= datetime.strptime(end_time, '%H:%M:%S')

def pre_process_for_heatmap(df_in: pd.DataFrame, pipe_number: Union[str, int] = None):
    """Prepare test_val (z), map_ori_sens (hover text), x-axis values, and y-band labels."""
    # ---- sensor matrix (use only present columns to avoid KeyErrors) ----
    expected = [f"F{i}H{j}" for i in range(1, 25) for j in range(1, 5)]
    sens_cols = [c for c in expected if c in df_in.columns]
    if not sens_cols:
        raise ValueError("No F*H* sensor columns found in the input DataFrame.")

    df_sens = pd.DataFrame(df_in, columns=sens_cols).copy()
    df_sens_raw = df_sens.copy(deep=True)  # Keep raw copy for CSV export
    df_mean_cols = df_sens_raw[[f"F{i}H{j}" for i in range(1, 25) for j in range(1, 5)]]
    Mean1 = df_mean_cols.mean()
    df_raw_plot = ((df_mean_cols - Mean1)/Mean1)*100

    # ---- Simplified data processing (similar to working example) ----
    # Convert all sensor data to numeric
    for col in sens_cols:
        df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    
    # Fill NaN values with forward fill, then zeros
    df_sens = df_sens.fillna(method='ffill').fillna(0.0)
    
    # Calculate mean for percentage normalization
    Mean1 = df_sens.mean()
    
    # Normalize as percentage deviation from mean
    df_sens_normalized = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100  # Add small epsilon to avoid division by zero
    
    # Zero out values above mean threshold
    for col in sens_cols:
        df_sens_normalized.loc[df_sens_normalized[col] > Mean1[col], col] = 0
    
    # Apply additional filtering if needed
    if UPPER_SENS_MUL > 0 and LOWER_SENS_MUL > 0:
        print("Applying additional sigma-based filtering...")
        sens_std = df_sens.std(axis=0, skipna=True)
        upper = Mean1 + UPPER_SENS_MUL * sens_std
        lower = Mean1 - LOWER_SENS_MUL * sens_std
        
        for col in sens_cols:
            mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
            df_sens_normalized.loc[mask, col] = 0

    # Use the normalized data for plotting
    df_sens_plot = df_sens_normalized.copy()

    # ---- roll → clock bands ----
    if ROLL_COL not in df_in.columns:
        raise KeyError(f"'{ROLL_COL}' column not found in input DataFrame.")

    roll = pd.to_numeric(df_in[ROLL_COL], errors='coerce').fillna(0.0) - INITIAL_READ

    d = [{"Roll_Sensor_0": pos} for pos in roll]

    # add keys Roll_Sensor_1..95, each +3.75 degrees
    upd_d = []
    for e in d:
        nd = {**e}
        for i in range(1, 96):
            nd[f"Roll_Sensor_{i}"] = e['Roll_Sensor_0'] + (3.75 * i)
        upd_d.append(nd)

    ori_df = pd.DataFrame.from_dict(upd_d)
    clock_df = ori_df.applymap(_degrees_to_hhmmss)

    # normalize time strings
    tmp = clock_df.apply(pd.to_datetime, format='%H:%M:%S')
    tmp = tmp.applymap(lambda x: x.strftime('%H:%M:%S'))
    tmp = tmp.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
    tmp = tmp.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
    tmp = tmp.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

    # rename time-matrix columns to sensor column names, keeping shape consistent
    tmp = tmp.rename(columns=dict(zip(tmp.columns, df_sens_plot.columns)))

    # build time_dict mapping bands → per-row tuples (idx, sensor_name, hh:mm:ss)
    time_dict = _create_time_dict()
    bands = list(time_dict.keys())

    for _, row in tmp.iterrows():
        row_list = list(row)
        row_dict = dict(row)
        keys = list(row_dict.keys())

        # find starting index where clock lies in the 00:00:00..00:07:29 band
        ind = 0
        for _, dval in row_dict.items():
            if _check_time_range(dval):
                ind = row_list.index(dval)
                break

        c = 0
        curr = ind
        while True:
            ck = keys[curr]
            time_dict[bands[c]].append((curr, ck, row_dict[ck]))
            c += 1
            curr = (curr + 1) % len(keys)
            if curr == ind:
                break

    map_ori_sens = pd.DataFrame(time_dict)

    # extract sensor names from tuples → a frame of sensor IDs per band
    val_ori_sens = map_ori_sens.applymap(lambda cell: cell[1])

    # fill test_val with the actual sensor values picked by val_ori_sens mapping
    test_val = val_ori_sens.copy()
    for r, e in val_ori_sens.iterrows():
        c = 0
        for _, sensor_col in e.items():
            test_val.iloc[r, c] = df_sens_plot.at[r, sensor_col]
            c += 1

    # x-axis values (meters) and y-band labels
    if ODDO_COL in df_in.columns:
        x_vals = (pd.to_numeric(df_in[ODDO_COL], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m)'
    else:
        x_vals = pd.Series(np.arange(len(df_in)))
        x_label = 'Index'

    y_bands = [str(c) for c in test_val.columns]
    
    # Align df_sens_plot columns with test_val for consistency
    df_sens_final = df_sens_plot.copy()
    df_sens_final.columns = test_val.columns
    
    return test_val, map_ori_sens, x_vals, x_label, y_bands, df_sens_final, df_raw_plot

# ==================== Plotly Rendering ====================

def render_heatmap_in_browser(df_sens_final: pd.DataFrame,
                              map_ori_sens: pd.DataFrame,
                              x_vals: pd.Series,
                              x_label: str,
                              y_bands: List[str],
                              pipe_number: Union[str, int],
                              df_raw_plot: pd.DataFrame):
    
    # Create heatmap data - transpose so y_bands are on y-axis
    heatmap_data = df_raw_plot.T
    
    # Ensure all data is numeric
    print(f"DEBUG: Original data types: {heatmap_data.dtypes.unique()}")
    print(f"DEBUG: Data shape: {heatmap_data.shape}")
    
    # Convert all columns to numeric, replacing errors with NaN
    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    
    # Fill NaN values with 0
    heatmap_data = heatmap_data.fillna(0.0)
    
    # Ensure data is float64
    heatmap_data = heatmap_data.astype(np.float64)
    
    print(f"DEBUG: Processed data types: {heatmap_data.dtypes.unique()}")
    print(f"DEBUG: Data range: {heatmap_data.min().min():.3f} to {heatmap_data.max().max():.3f}")
    
    # Check for any remaining issues
    if not np.isfinite(heatmap_data.values).all():
        print("WARNING: Non-finite values detected, replacing with 0")
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    
    # Create the interactive heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_vals.round(2),
        y=y_bands,
        colorscale='jet',
        zmin=-3,
        zmax=8,
        colorbar=dict(title="Sensor Value (%)"),
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>' +
                     '<b>%{y}</b><br>' +
                     '<b>Value: %{z:.2f}%</b>' +
                     '<extra></extra>'
    ))
    
    # Add overlay points with FIXED SIZE SQUARES
    overlay_added = False
    
    # PipeTally file logic
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands, PIPETALLY_PATH)
    if pts is not None:
        xs, ys, labels = pts
        
        print(f"DEBUG: Heatmap x-axis range: {x_vals.min():.2f} to {x_vals.max():.2f}")
        print(f"DEBUG: Overlay x-points: {[round(x, 2) for x in xs]}")
        
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                
                # Fixed size square - no dynamic calculation bullshit
                fig.add_shape(
                    type="rect",
                    x0=x - 0.05, y0=y_idx - 0.35,  # Fixed 0.1m width, 0.7 band height
                    x1=x + 0.05, y1=y_idx + 0.35,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(255,0,0,0.6)"
                )
                
                # Add label annotation
                fig.add_annotation(
                    x=x, y=y_idx,
                    text=label,
                    showarrow=False,
                    font=dict(color="white", size=8, family="Arial Black"),
                    bgcolor="red",
                    bordercolor="black",
                    borderwidth=1
                )
                overlay_added = True
    
    # Update layout
    fig.update_layout(
        title=f"Interactive Plotly Heatmap — Pipe {pipe_number}",
        xaxis_title=x_label,
        yaxis_title="Orientation (12h bands)",
        width=1400,
        height=700,
        font=dict(size=12),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    print("SAVE MODE: Writing interactive HTML file...")

    html_file = Path(f"heatmap{pipe_number}.html")

    fig.write_html(
        str(html_file),
        full_html=True,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'heatmap_pipe_{pipe_number}',
                'height': 700,
                'width': 1400,
                'scale': 2
            }
        },
        include_plotlyjs=True
    )

    print(f"Saved interactive heatmap: {html_file.resolve()}")
    print(f"Interactive heatmap opened in browser: {html_file}")
    print(f"Features available:")
    print(f"- Hover tooltips with exact values")
    print(f"- Zoom and pan functionality")
    print(f"- Download as PNG option")
    print(f"- Overlays: {'Yes' if overlay_added else 'None found'}")

# ==================== Main ====================
if __name__ == "__main__":
    # Derive pipe number from filename by default
    PIPE_NUMBER = Path(PKL_PATH).stem

    print(f"Loading data from: {PKL_PATH}")
    
    try:
        # Load the raw data
        df = pd.read_pickle(PKL_PATH)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")

        # Process data for heatmap
        test_val, map_ori_sens, x_vals, x_label, y_bands, df_sens_final, df_raw_plot = pre_process_for_heatmap(df, PIPE_NUMBER)
        
        print(f"Processing complete. Heatmap shape: {df_sens_final.shape}")
        print(f"X-axis range: {x_vals.min():.2f} to {x_vals.max():.2f}")
        print(f"Y-bands: {len(y_bands)} bands")
        
        # Render the heatmap
        render_heatmap_in_browser(df_sens_final, map_ori_sens, x_vals, x_label, y_bands, PIPE_NUMBER, df_raw_plot)
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {PKL_PATH}")
        print("Please check the PKL_PATH in the CONFIG section")
    except KeyError as e:
        print(f"ERROR: Missing required column: {e}")
        print("Please check your data contains the required columns (ROLL, sensor columns)")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()