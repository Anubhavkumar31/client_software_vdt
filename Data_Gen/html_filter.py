
import os
# Limit intra-process math threads to avoid over-subscription
from Data_Gen.heatmap_hall_sensor import save_interactive_heatmap_v2

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
from pathlib import Path
import numpy as np
import re
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import datetime
from joblib import Parallel, delayed
import warnings
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
from typing import Optional, Union, List
from glob import glob
from datetime import datetime, timedelta
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------ Plotly export options ------------
PLOTLY_JS_MODE   = "directory"
PLOTLY_COMPRESS  = True
HTML_DEFAULT_W   = "100%"
HTML_DEFAULT_H   = 500

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "displayModeBar": True,
    "scrollZoom": False,
}

def write_plotly_html(fig, out_path: str):
    common = dict(
        include_plotlyjs=PLOTLY_JS_MODE,
        full_html=True,
        config=PLOTLY_CONFIG,
        auto_open=False,
        default_width=HTML_DEFAULT_W,
        default_height=HTML_DEFAULT_H,
    )
    try:
        fig.write_html(out_path, compress_data=PLOTLY_COMPRESS, **common)
    except TypeError:
        fig.write_html(out_path, **common)


# -------------------- CONFIG --------------------
INITIAL_READ = 0.0
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3
WORKERS = 4
# ------------------------------------------------


# ============================================================
#  OPTIMIZED CORE HELPERS
# ============================================================

def _degrees_to_hhmm_vectorized(degrees_arr: np.ndarray) -> np.ndarray:
    """
    Vectorised replacement for the per-cell degrees_to_hours_minutes2 loop.
    Returns an object array of 'HH:MM' strings.
    """
    d = degrees_arr % 360                          # [0, 360)
    total_seconds = d * (12 * 3600 / 360)         # map to 12h dial
    hours   = (total_seconds // 3600).astype(int) % 12
    minutes = ((total_seconds % 3600) // 60).astype(int)

    # vectorised string formatting
    hh = np.char.zfill(hours.astype(str),   2)
    mm = np.char.zfill(minutes.astype(str), 2)
    return np.char.add(np.char.add(hh, ':'), mm)


def _fix_hour_prefix_vectorized(arr: np.ndarray) -> np.ndarray:
    """Replace 23:→11:, 22:→10:, 12:→00: in a string array (vectorised)."""
    replacements = [('23:', '11:'), ('22:', '10:'), ('12:', '00:')]
    out = arr.copy()
    for old, new in replacements:
        mask = np.char.startswith(out, old)
        if mask.any():
            out[mask] = np.char.replace(out[mask], old, new)
    return out


def _build_clock_data_fast(oriData: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised replacement for oriData.applymap(degrees_to_hours_minutes2)
    + the four applymap string-replace calls + pd.to_datetime conversion.

    Returns a DataFrame of 'HH:MM' strings — same shape as oriData.
    """
    arr = oriData.to_numpy(dtype=float)           # (n_rows, n_cols)
    flat = arr.ravel()

    hhmm_flat = _degrees_to_hhmm_vectorized(flat)
    hhmm_2d   = hhmm_flat.reshape(arr.shape)

    # fix hour prefixes row-by-row via vectorised string ops
    for col_i in range(hhmm_2d.shape[1]):
        hhmm_2d[:, col_i] = _fix_hour_prefix_vectorized(hhmm_2d[:, col_i])

    return pd.DataFrame(hhmm_2d, columns=oriData.columns, index=oriData.index)


def _remap_sensor_timeline_fast(
    test_clockData: pd.DataFrame,
    rang: list,
    total_sensors: int,
) -> tuple:
    """
    Vectorised replacement for the row-by-row Python loop that builds
    time_dict_1 (stage 22).

    Returns (map_ori_sens, val_ori_sens_labels) where:
      map_ori_sens      — DataFrame of tuples (col_idx, col_name, hhmm_str)
      val_ori_sens_labels — DataFrame of sensor-column-name strings (for stage 26 lookup)
    """
    n_rows, n_cols = test_clockData.shape
    col_names = list(test_clockData.columns)

    # Convert HH:MM → integer seconds once for every cell
    def hhmm_to_sec(hhmm_2d: np.ndarray) -> np.ndarray:
        h = hhmm_2d[:, :, :2].view(np.uint8)   # won't work on string arrays
        # simpler: parse via vectorised split
        flat = hhmm_2d.ravel()
        secs = np.array([int(s[:2]) * 60 + int(s[3:]) for s in flat], dtype=np.int32)
        return secs.reshape(hhmm_2d.shape)

    # Build a (n_rows, n_cols) int array of seconds
    raw = test_clockData.to_numpy()             # dtype=object, strings 'HH:MM'
    flat_raw = raw.ravel()
    flat_secs = np.array(
        [int(s[:2]) * 60 + int(s[3:]) for s in flat_raw],
        dtype=np.int32
    ).reshape(n_rows, n_cols)

    # Target seconds for rang[0] (first band)
    start_sec = int(rang[0][:2]) * 60 + int(rang[0][3:])

    # For each row, find the column whose value == rang[0] (or nearest)
    diff = np.abs(flat_secs - start_sec)          # (n_rows, n_cols)
    ind_arr = np.argmin(diff, axis=1)             # (n_rows,) — start column per row

    # Build the remapped label matrix using advanced indexing
    # For row r, the output column c gets col_names[(ind_arr[r] + c) % n_cols]
    col_idx_grid = (ind_arr[:, None] + np.arange(n_cols)[None, :]) % n_cols  # (n_rows, n_cols)

    # label matrix: sensor column names, shape (n_rows, n_cols)
    col_names_arr = np.array(col_names)
    label_matrix = col_names_arr[col_idx_grid]    # (n_rows, n_cols)

    # hhmm values at those positions
    row_idx = np.arange(n_rows)[:, None]
    hhmm_matrix = raw[row_idx, col_idx_grid]      # (n_rows, n_cols)

    # Build map_ori_sens as a DataFrame of tuples
    # Each cell: (col_idx, col_name, hhmm_str)
    # Store as three separate arrays for speed, merge into tuples only when needed
    map_ori_sens_data = np.empty((n_rows, n_cols), dtype=object)
    for c in range(n_cols):
        col_indices = col_idx_grid[:, c]
        for r in range(n_rows):
            map_ori_sens_data[r, c] = (
                int(col_indices[r]),
                label_matrix[r, c],
                hhmm_matrix[r, c]
            )

    map_ori_sens = pd.DataFrame(
        map_ori_sens_data,
        columns=rang[:n_cols],
        index=test_clockData.index
    )

    val_ori_sens_labels = pd.DataFrame(
        label_matrix,
        columns=rang[:n_cols],
        index=test_clockData.index
    )

    return map_ori_sens, val_ori_sens_labels


def _map_sensor_values_fast(
    val_ori_sens_labels: pd.DataFrame,
    df_new_tab9: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vectorised replacement for stage 26 (the 4.3M-iteration nested loop).

    val_ori_sens_labels[r, c] = column name in df_new_tab9 to look up for row r.
    We want test_val[r, c] = df_new_tab9.at[r, col_name].

    Strategy: for each unique column name that appears in val_ori_sens_labels,
    scatter its entire column vector into the right positions of the output matrix.
    This turns O(n_rows * n_cols) individual .at lookups into O(unique_cols) bulk
    numpy assignments.
    """
    n_rows, n_cols = val_ori_sens_labels.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)

    label_arr = val_ori_sens_labels.to_numpy()   # (n_rows, n_cols) strings

    # Convert df_new_tab9 to numpy for O(1) column access
    col_to_idx = {col: i for i, col in enumerate(df_new_tab9.columns)}
    data_mat   = df_new_tab9.to_numpy(dtype=np.float64)  # (n_rows, n_sensors)

    for c in range(n_cols):
        col_labels = label_arr[:, c]             # (n_rows,) — one sensor name per row

        # group rows by which sensor column they point to
        unique_sensors = np.unique(col_labels)
        for sensor in unique_sensors:
            if sensor not in col_to_idx:
                continue
            row_mask     = col_labels == sensor
            sensor_col_i = col_to_idx[sensor]
            out[row_mask, c] = data_mat[row_mask, sensor_col_i]

    return pd.DataFrame(
        out,
        columns=val_ori_sens_labels.columns,
        index=val_ori_sens_labels.index
    )


# ============================================================
#  PRE-PROCESS DATA  (drop-in replacement)
# ============================================================

def pre_process_data(
    pkl_path,
    datafile,
    pipe_number,
    output_folder,
    total_sensors,
    column_names,
    minute_sensors,
    degree_sensors,
    sensor_type,
    debug=False
):
    import time

    dbg_start = time.time()

    def dbg(msg):
        if debug:
            elapsed = round(time.time() - dbg_start, 2)
            print(f"[PIPE {pipe_number} | {sensor_type}] {msg} | {elapsed}s", flush=True)

    dbg("START pre_process_data")

    # ── initial copies ────────────────────────────────────────────────────────
    datafile_original = datafile.copy(deep=True)
    dbg("1: datafile_original copy complete")

    df_new_tab9 = pd.DataFrame(datafile, columns=column_names)
    dbg(f"2: df_new_tab9 created | shape={df_new_tab9.shape}")

    df_new_tab10 = df_new_tab9.copy()
    dbg("3: df_new_tab10 copy complete")

    sensor_columns = df_new_tab9.columns.tolist()
    dbg(f"4: sensor_columns loaded | count={len(sensor_columns)}")

    # ── Savitzky-Golay denoise ────────────────────────────────────────────────
    # OPTIMISATION: batch detrend + filter using matrix ops instead of per-column loop
    dbg("5: starting denoise (vectorised)")

    data_mat   = df_new_tab9.to_numpy(dtype=np.float64)          # (n, n_cols)
    time_index = np.arange(data_mat.shape[0], dtype=np.float64)

    # Fit degree-2 polynomial for every column at once via lstsq
    A      = np.column_stack([time_index**2, time_index, np.ones_like(time_index)])
    coeffs = np.linalg.lstsq(A, data_mat, rcond=None)[0]         # (3, n_cols)
    trends = A @ coeffs                                           # (n, n_cols)

    detrended = data_mat - trends
    smoothed  = np.apply_along_axis(
        lambda col: savgol_filter(col, 15, 2), axis=0, arr=detrended
    )
    df_new_tab9 = pd.DataFrame(smoothed, columns=sensor_columns, index=df_new_tab9.index)

    dbg("6: denoise complete (vectorised)")

    df_raw_straight = df_new_tab9.copy()
    dbg("7: df_raw_straight copy complete")

    # ── noise filtering ───────────────────────────────────────────────────────
    dbg("8: calculating means/std")
    sens_mean          = df_new_tab9.abs().mean()
    standard_deviation = df_new_tab9.std(axis=0, skipna=True)
    mean_plus_sigma    = sens_mean + UPPER_SENS_MUL * standard_deviation
    mean_negative_sigma= sens_mean - LOWER_SENS_MUL * standard_deviation

    dbg("9: applying noise filtering (vectorised)")
    mat   = df_new_tab9.to_numpy(dtype=np.float64)
    lower = mean_negative_sigma.to_numpy()
    upper = mean_plus_sigma.to_numpy()
    mask  = (mat >= lower) & (mat <= upper)
    mat[mask] = 0.0
    df_new_tab9 = pd.DataFrame(mat, columns=sensor_columns, index=df_new_tab9.index)
    dbg("10: noise filtering complete")

    # ── roll prep ─────────────────────────────────────────────────────────────
    initial_read = INITIAL_READ
    roll = (datafile['ROLL'] - initial_read).to_numpy(dtype=np.float64)
    dbg("11: roll normalization complete")

    # ── build oriData — vectorised ────────────────────────────────────────────
    dbg("12-15: building oriData (vectorised)")

    # Each row: [roll[r] + degree_sensors*i  for i in 0..total_sensors-1]
    offsets  = np.arange(total_sensors, dtype=np.float64) * degree_sensors  # (n_cols,)
    ori_mat  = roll[:, None] + offsets[None, :]                              # (n_rows, n_cols)

    col_labels_orig = [f'Roll_Sensor_{i}' for i in range(total_sensors)]
    oriData = pd.DataFrame(ori_mat, columns=col_labels_orig)
    dbg(f"15: oriData created | shape={oriData.shape}")

    # ── degrees → HH:MM  (vectorised, replaces applymap + 4 string-replace applymap) ──
    dbg("16-20: clockData (vectorised)")
    test_clockData = _build_clock_data_fast(oriData)
    test_clockData.columns = df_new_tab9.columns          # rename to sensor column names
    dbg("20: clockData renamed")

    # ── time dict ─────────────────────────────────────────────────────────────
    dbg("21: creating time dictionary")
    time_list     = [timedelta(minutes=i * minute_sensors) for i in range(total_sensors)]
    time_ranges_2 = [(datetime.min + t).strftime('%H:%M') for t in time_list]
    rang          = time_ranges_2                         # list of band labels

    # ── remap sensor timeline  (vectorised — replaces 29884-row Python loop) ──
    dbg("22: remapping sensor timeline (vectorised)")
    map_ori_sens, val_ori_sens_labels = _remap_sensor_timeline_fast(
        test_clockData, rang, total_sensors
    )
    dbg("23: timeline remap complete")
    dbg(f"24: map_ori_sens created | shape={map_ori_sens.shape}")

    # ── map sensor values  (vectorised — replaces 4.3M-iteration loop) ───────
    dbg("25-26: mapping sensor values (vectorised)")
    test_val = _map_sensor_values_fast(val_ori_sens_labels, df_new_tab9)
    dbg(f"26: test_val built | shape={test_val.shape}")

    # ── map_val_sens  (stage 27 — now trivially fast) ────────────────────────
    dbg("27: building map_val_sens")
    # Reconstruct map_val_sens as before (tuple of (col_idx, col_name, hhmm, value))
    # This is only used downstream if callers inspect it; build it efficiently.
    map_ori_arr  = map_ori_sens.to_numpy()               # object array of 3-tuples
    test_val_arr = test_val.to_numpy(dtype=np.float64)

    map_val_mat  = np.empty(map_ori_arr.shape, dtype=object)
    for c in range(map_ori_arr.shape[1]):
        for r in range(map_ori_arr.shape[0]):
            t = map_ori_arr[r, c]
            map_val_mat[r, c] = (*t, test_val_arr[r, c])

    map_val_sens = pd.DataFrame(
        map_val_mat, columns=map_ori_sens.columns, index=map_ori_sens.index
    )
    dbg("27: map_val_sens complete")

    # ── plotting ──────────────────────────────────────────────────────────────
    dbg("28: plotting stage")
    if sensor_type == "Hall":
        create_plots_hall(
            pkl_path, df_new_tab9, df_raw_straight, datafile,
            test_val, map_ori_sens, pipe_number, output_folder,
            df_new_tab10, datafile_original
        )
    else:
        create_plots_proximity(
            pkl_path, df_new_tab9, df_raw_straight, datafile,
            test_val, map_ori_sens, pipe_number, output_folder,
            df_new_tab10, datafile_original
        )

    dbg("29: COMPLETE pre_process_data")
    return (datafile, df_new_tab9, datafile_original, test_val, map_ori_sens, df_new_tab10)


# ============================================================
#  EVERYTHING BELOW IS UNCHANGED FROM ORIGINAL
# ============================================================

def _find_pipe_tally_file(pipe_number: Union[str, int], folder_path: str) -> Optional[str]:
    pn = str(pipe_number)
    tally_path = f"{folder_path}/PipeTally{pn}.csv"
    if os.path.exists(tally_path):
        return tally_path
    patterns = [
        f"{folder_path}/*PipeTally*{pn}*.csv",
        f"{folder_path}/*PipeTally*.csv",
        f"{folder_path}/*Pipe_Tally*.csv"
    ]
    for pat in patterns:
        hits = glob(pat)
        if hits:
            return hits[0]
    return None

def _pick_col(df: pd.DataFrame, preferred: List[str], tokens: List[str]) -> Optional[str]:
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
    parts = str(t).split(":")
    if len(parts) == 3:
        h, m, s = [int(x) for x in parts]
    elif len(parts) == 2:
        h, m = [int(x) for x in parts]
        s = 0
    else:
        raise ValueError(f"Invalid time format: {t}")
    return (h % 12) * 3600 + m * 60 + s

def _nearest_band_label(seconds: int, band_labels: List[str]) -> str:
    band_labels_str = [str(x) for x in band_labels]
    band_secs = np.array([_hhmmss_to_seconds(lbl) for lbl in band_labels_str], dtype=int)
    idx = int(np.argmin(np.abs(band_secs - seconds)))
    return band_labels_str[idx]

def _load_overlay_points_for_pipe(pipe_number, y_band_labels, folder_path, *, debug_prefix="OVERLAY DEBUG"):
    path = _find_pipe_tally_file(pipe_number, folder_path)
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

    if x_col is None or ori_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: missing required columns; no overlay.")
        return None
    if pipe_col is not None:
        mask = df[pipe_col].astype(str).str.contains(str(pipe_number), na=False)
        if mask.any():
            df = df[mask]
    total_rows = len(df)
    if feat_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: no feature column; overlays disabled.")
        return None
    feat_series = df[feat_col].astype(str).str.strip().str.lower()
    metal_loss_mask = feat_series.str.fullmatch(r"metal\s*loss", case=False, na=False)
    df = df[metal_loss_mask]
    after_filter = len(df)

    xs, ys, labels = [], [], []
    skipped_no_x, skipped_no_ori, skipped_other = 0, 0, 0

    for _, row in df.iterrows():
        x = pd.to_numeric(row.get(x_col), errors="coerce")
        if pd.isna(x):
            skipped_no_x += 1
            continue
        ori_sec = _parse_ori_to_seconds(row.get(ori_col))
        if ori_sec is None:
            skipped_no_ori += 1
            continue
        y = _nearest_band_label(int(ori_sec), list(y_band_labels))
        if y not in y_band_labels:
            skipped_other += 1
            continue
        if sno_col is not None:
            lbl = row.get(sno_col)
            lbl = str(lbl).strip() if (lbl is not None and str(lbl).strip() != "" and not pd.isna(lbl)) else None
        else:
            lbl = None
        if lbl is None:
            lbl2 = row.get("Defect_id")
            lbl = str(lbl2).strip() if (lbl2 is not None and str(lbl2).strip() != "" and not pd.isna(lbl2)) else str(len(labels) + 1)
        xs.append(float(x))
        ys.append(str(y))
        labels.append(lbl)

    print(
        f"\n{debug_prefix}: pipe {pipe_number}: total_rows={total_rows}, "
        f"after_feature='Metal Loss'={after_filter}, plotted={len(xs)}, "
        f"skipped_no_x={skipped_no_x}, skipped_no_ori={skipped_no_ori}, skipped_other={skipped_other}"
    )
    if not xs:
        return None
    return xs, ys, labels


def pre_process_for_interactive_heatmap(df_in: pd.DataFrame, datafile: pd.DataFrame, test_val: pd.DataFrame, map_ori_sens: pd.DataFrame):
    sens_cols = df_in.columns.tolist()
    print(f"Sensor columns in datafile for preprocess_heatmap_hall_sensor: {sens_cols}")
    if not sens_cols:
        raise ValueError("No sensor columns found in the input DataFrame.")

    df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
    df_sens_raw = df_sens.copy(deep=True)
    df_mean_cols = df_sens_raw
    Mean1 = df_mean_cols.mean()
    df_raw_plot = ((df_mean_cols - Mean1) / Mean1) * 100

    for col in sens_cols:
        df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    df_sens = df_sens.fillna(method='ffill').fillna(0.0)
    Mean1 = df_sens.mean()
    df_sens_normalized = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100

    for col in sens_cols:
        df_sens_normalized.loc[df_sens_normalized[col] > Mean1[col], col] = 0

    if UPPER_SENS_MUL > 0 and LOWER_SENS_MUL > 0:
        sens_std = df_sens.std(axis=0, skipna=True)
        upper = Mean1 + UPPER_SENS_MUL * sens_std
        lower = Mean1 - LOWER_SENS_MUL * sens_std
        for col in sens_cols:
            mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
            df_sens_normalized.loc[mask, col] = 0

    df_plot_rearranged = test_val.copy()
    for r in range(len(test_val)):
        for c, band in enumerate(test_val.columns):
            sensor_info = map_ori_sens.iloc[r, c]
            if isinstance(sensor_info, tuple) and len(sensor_info) >= 2:
                sensor_col = sensor_info[1]
                if sensor_col in df_sens_normalized.columns:
                    df_plot_rearranged.iloc[r, c] = df_sens_normalized.loc[r, sensor_col]

    return df_plot_rearranged, df_raw_plot


def save_interactive_heatmap(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10):
    df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)

    if 'ODDO1' in datafile.columns:
        x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m) --- Hall Sensor Heatmap'
    else:
        x_vals = pd.Series(np.arange(len(datafile)))
        x_label = 'Index'

    y_bands = [str(c) for c in test_val.columns]
    heatmap_data = df_raw_plot.T

    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    heatmap_data = heatmap_data.astype("float32").round(3)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=x_vals.round(2), y=y_bands,
        colorscale='jet', zmin=-3, zmax=8, showscale=False, hoverongaps=False,
        hovertemplate='<b>%{x}</b><br><b>%{y}</b><br><b>Value: %{z:.2f}%</b><extra></extra>'
    ))

    overlay_added = False
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
    if pts is not None:
        xs, ys, labels = pts
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                fig.add_shape(type="rect",
                    x0=x-0.05, y0=y_idx-0.35, x1=x+0.05, y1=y_idx+0.35,
                    line=dict(color="black", width=2), fillcolor="rgba(255,0,0,0.6)")
                fig.add_annotation(x=x, y=y_idx, text=label, showarrow=False,
                    font=dict(color="white", size=8, family="Arial Black"),
                    bgcolor="red", bordercolor="black", borderwidth=1)
                overlay_added = True

    fig.update_layout(
        title=dict(text=f"Hall-Sensor Heatmap — Joint Number {pipe_number}",
            x=0.5, xanchor="center", font=dict(size=18, family="Arial Black")),
        xaxis_title=x_label, yaxis_title=" ", width=1500, height=500,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    fig.update_yaxes(autorange="reversed")
    write_plotly_html(fig, f'{folder_path}/hallsensor_heatmap{pipe_number}.html')
    print(f"Saved hallsensor heatmap: {folder_path}/hallsensor_heatmap{pipe_number}.html")
    print(f"Overlays: {'Yes' if overlay_added else 'None found'}")


def save_interactive_heatmap_proximity(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10):
    print("RUNNING PROXIMITY HEATMAP GENERATION \n")
    pattern = re.compile(r'^F\d+P\d+$')
    datafile_original = datafile.copy(deep=True)
    matching_columns = [col for col in datafile_original.columns if pattern.match(col)]
    df_new_tab9 = pd.DataFrame(datafile, columns=matching_columns)
    df_new_tab10 = df_new_tab9.copy()

    df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)

    if 'ODDO1' in datafile.columns:
        x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m) --- Proximity Sensor Heatmap'
    else:
        x_vals = pd.Series(np.arange(len(datafile)))
        x_label = 'Index'

    y_bands = [str(c) for c in test_val.columns]
    heatmap_data = df_raw_plot.T

    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    heatmap_data = heatmap_data.astype("float32").round(3)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=x_vals.round(2), y=y_bands,
        colorscale='jet', zmin=-3, zmax=7, showscale=False, hoverongaps=False,
        hovertemplate='<b>%{x}</b><br><b>%{y}</b><br><b>Value: %{z:.2f}%</b><extra></extra>'
    ))

    overlay_added = False
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
    if pts is not None:
        xs, ys, labels = pts
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                fig.add_shape(type="rect",
                    x0=x-0.05, y0=y_idx-0.35, x1=x+0.05, y1=y_idx+0.35,
                    line=dict(color="black", width=2), fillcolor="rgba(255,0,0,0.6)")
                fig.add_annotation(x=x, y=y_idx, text=label, showarrow=False,
                    font=dict(color="white", size=8, family="Arial Black"),
                    bgcolor="red", bordercolor="black", borderwidth=1)
                overlay_added = True

    fig.update_layout(
        title=dict(text=f"Proximity-Sensor Heatmap — Joint Number {pipe_number}",
            x=0.5, xanchor="center", font=dict(size=18, family="Arial Black")),
        xaxis_title=x_label, yaxis_title=" ", width=1500, height=500,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    fig.update_yaxes(autorange="reversed")
    write_plotly_html(fig, f'{folder_path}/proximity_heatmap{pipe_number}.html')
    print(f"Saved proximity heatmap: {folder_path}/proximity_heatmap{pipe_number}.html")
    print(f"Overlays: {'Yes' if overlay_added else 'None found'}")


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacements for save_interactive_heatmap / save_interactive_heatmap_proximity
# Uses bounding-box rectangles + numbered labels from defects_clock_hm.csv
#
# New function names (swap in when ready):
#   save_interactive_heatmap_v2(...)
#   save_interactive_heatmap_proximity_v2(...)
#
# Signature is IDENTICAL to the originals — no call-site changes needed.
#
# KEY COORDINATE FIX:
#   x-axis → row['upstream']        (metres from upstream GW, local to pipe)
#             NOT start_oddo1/end_oddo1 (all zeros) and NOT start_index/end_index (global)
#   y-axis → sensor_to_yidx lookup  (start_sensor/end_sensor → local y_bands index)
#             NOT raw sensor integers (they don't match local heatmap y-axis directly)
# ─────────────────────────────────────────────────────────────────────────────
#
# import os
# import re
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
#
# # ── Defect CSV loader (cached per session) ────────────────────────────────────
#
# _DEFECTS_DF_CACHE = None
#
#
# def _load_defects_df(folder_path: str) -> pd.DataFrame:
#     """
#     Lazy-load defects_clock_hm.csv.
#     Looks two levels up from folder_path:
#       output_folder/Pipe_N/  →  output_folder/../../defects_clock_hm.csv
#     Adjust the path below if your layout differs.
#     """
#     global _DEFECTS_DF_CACHE
#     if _DEFECTS_DF_CACHE is not None:
#         return _DEFECTS_DF_CACHE
#
#     candidate = r"D:\Anubhav\runid_data\12inch\12_inch_runid_27\defects_clock_hm.csv"
#     if os.path.exists(candidate):
#         _DEFECTS_DF_CACHE = pd.read_csv(candidate)
#         print(f"[bbox overlay] Loaded defects CSV: {candidate}  ({len(_DEFECTS_DF_CACHE)} rows)")
#     else:
#         _DEFECTS_DF_CACHE = pd.DataFrame()
#         print(
#             f"[bbox overlay] WARNING: defects_clock_hm.csv not found at:\n  {candidate}\n  No overlays will be drawn.")
#
#     return _DEFECTS_DF_CACHE
#
#
# # ── Colour by depth % ─────────────────────────────────────────────────────────
#
# def _defect_color(depth: float):
#     """Returns (fill_rgba, line_color) based on depth %."""
#     if depth >= 80:
#         return 'rgba(200, 0,   0,   0.20)', 'red'
#     elif depth >= 50:
#         return 'rgba(255, 140, 0,   0.20)', 'darkorange'
#     else:
#         return 'rgba(255, 200, 0,   0.20)', 'goldenrod'
#
#
# # ── Core overlay drawing ──────────────────────────────────────────────────────
#
# def _draw_bbox_overlays(fig, pipe_number: int, folder_path: str, y_bands: list) -> bool:
#     """
#     Draw one bounding-box rect + numbered annotation per defect.
#
#     X-axis uses ABSOLUTE DISTANCE so it aligns with ODDO1 heatmap axis.
#     Y-axis maps sensor numbers into local heatmap sensor indices.
#     """
#
#     defects_df = _load_defects_df(folder_path)
#
#     if defects_df.empty:
#         return False
#
#     # safer pipe filtering
#     pipe_defects = defects_df[
#         defects_df['pipe_id'].astype(str) == str(pipe_number)
#     ].reset_index(drop=True)
#
#     if pipe_defects.empty:
#         print(f"[bbox overlay] No defects found for pipe {pipe_number}")
#         return False
#
#     # ─────────────────────────────────────────────────────────────
#     # Build sensor → local heatmap y-index lookup
#     # ─────────────────────────────────────────────────────────────
#
#     sensor_to_yidx = {}
#
#     for local_idx, band_str in enumerate(y_bands):
#         try:
#             sensor_to_yidx[int(float(band_str))] = local_idx
#         except Exception:
#             pass
#
#     print("\n=== BBOX DEBUG ===")
#     print(f"Pipe Number: {pipe_number}")
#     print("First heatmap sensors:", list(sensor_to_yidx.keys())[:20])
#
#     overlay_added = False
#     drawn_count = 0
#
#     for defect_counter, row in pipe_defects.iterrows():
#
#         label_num = defect_counter + 1
#
#         # ─────────────────────────────────────────────────────────
#         # X POSITION
#         # IMPORTANT:
#         # use ABSOLUTE DISTANCE because heatmap x-axis uses ODDO1
#         # ─────────────────────────────────────────────────────────
#
#         x_center = float(row.get('absolute_distance', 0) or 0)
#
#         length_m = float(row.get('length', 0) or 0) / 1000.0
#
#         if length_m <= 0:
#             length_m = 0.05
#
#         x0 = x_center - length_m / 2.0
#         x1 = x_center + length_m / 2.0
#         x_mid = x_center
#
#         # ─────────────────────────────────────────────────────────
#         # Y POSITION
#         # ─────────────────────────────────────────────────────────
#
#         s_sensor = int(float(row.get('start_sensor', 0) or 0))
#         e_sensor = int(float(row.get('end_sensor', 0) or 0))
#
#         if s_sensor > e_sensor:
#             s_sensor, e_sensor = e_sensor, s_sensor
#
#         local_ys = [
#             sensor_to_yidx[s]
#             for s in range(s_sensor, e_sensor + 1)
#             if s in sensor_to_yidx
#         ]
#
#         print(
#             f"[Defect #{label_num}] "
#             f"abs_dist={x_center:.3f}  "
#             f"sensors={s_sensor}-{e_sensor}  "
#             f"matched={len(local_ys)}"
#         )
#
#         if not local_ys:
#             print(f"--> skipped (sensor mismatch)")
#             continue
#
#         y0 = min(local_ys) - 0.5
#         y1 = max(local_ys) + 0.5
#
#         y_mid = (min(local_ys) + max(local_ys)) / 2.0
#         y_label = min(local_ys) - 1.0
#
#         # ─────────────────────────────────────────────────────────
#         # COLOUR
#         # ─────────────────────────────────────────────────────────
#
#         depth = float(row.get('depth_new', 0) or 0)
#
#         fill_color, line_color = _defect_color(depth)
#
#         # ─────────────────────────────────────────────────────────
#         # DRAW RECTANGLE
#         # ─────────────────────────────────────────────────────────
#
#         fig.add_shape(
#             type='rect',
#             x0=x0,
#             x1=x1,
#             y0=y0,
#             y1=y1,
#             line=dict(
#                 color=line_color,
#                 width=3
#             ),
#             fillcolor=fill_color,
#             layer='above'
#         )
#
#         # ─────────────────────────────────────────────────────────
#         # LABEL
#         # ─────────────────────────────────────────────────────────
#
#         fig.add_annotation(
#             x=x_mid,
#             y=y_label,
#             text=str(label_num),
#             showarrow=False,
#             font=dict(
#                 color=line_color,
#                 size=10,
#                 family='Arial Black'
#             ),
#             bgcolor='white',
#             bordercolor='black',
#             borderwidth=1
#         )
#
#         # ─────────────────────────────────────────────────────────
#         # HOVER TOOLTIP
#         # ─────────────────────────────────────────────────────────
#
#         orient = row.get('orientation', 'N/A')
#         dim_cls = row.get('dimension_classification', 'N/A')
#         defect_type = row.get('defect_type', 'N/A')
#
#         length_mm = row.get('length', 'N/A')
#
#         width_mm = row.get(
#             'width_final',
#             row.get('Width', 'N/A')
#         )
#
#         abs_dist = row.get('absolute_distance', 'N/A')
#
#         hover_text = (
#             f"<b>Defect #{label_num}</b><br>"
#             f"Pipe: {pipe_number}<br>"
#             f"Depth: {depth:.0f}%<br>"
#             f"Orientation: {orient}<br>"
#             f"Classification: {dim_cls}<br>"
#             f"Type: {defect_type}<br>"
#             f"Length: {length_mm} mm<br>"
#             f"Width: {width_mm} mm<br>"
#             f"Abs. Distance: {abs_dist} m"
#         )
#
#         fig.add_trace(
#             go.Scatter(
#                 x=[x_mid],
#                 y=[y_mid],
#                 mode='markers',
#                 marker=dict(
#                     size=12,
#                     color=line_color,
#                     opacity=0.0
#                 ),
#                 hovertemplate=hover_text + '<extra></extra>',
#                 showlegend=False,
#             )
#         )
#
#         overlay_added = True
#         drawn_count += 1
#
#     print(f"[bbox overlay] Pipe {pipe_number}: drew {drawn_count} defect box(es)")
#
#     return overlay_added
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # NEW FUNCTION 1 — Hall-Sensor Heatmap
# # ─────────────────────────────────────────────────────────────────────────────
#
# def save_interactive_heatmap_v2(
#         df_new_tab9, datafile, test_val, map_ori_sens,
#         folder_path, pipe_number, df_new_tab10
# ):
#     """
#     Hall-sensor heatmap with bounding-box defect overlays (v2).
#     Drop-in replacement for save_interactive_heatmap().
#     """
#     df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(
#         df_new_tab10, datafile, test_val, map_ori_sens
#     )
#
#     if 'ODDO1' in datafile.columns:
#         x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
#         x_label = 'Absolute Distance (m) — Hall Sensor Heatmap'
#     else:
#         x_vals = pd.Series(np.arange(len(datafile)))
#         x_label = 'Index'
#
#     y_bands = [str(c) for c in df_raw_plot.columns]
#     heatmap_data = df_raw_plot.T
#
#     for col in heatmap_data.columns:
#         heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
#     heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
#     if not np.isfinite(heatmap_data.values).all():
#         heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
#     heatmap_data = heatmap_data.astype('float32').round(3)
#
#     fig = go.Figure(data=go.Heatmap(
#         z=heatmap_data.values,
#         x=x_vals.round(2),
#         y=y_bands,
#         colorscale='jet',
#         zmin=-3, zmax=8,
#         showscale=False,
#         hoverongaps=False,
#         hovertemplate='<b>%{x}</b><br><b>%{y}</b><br><b>Value: %{z:.2f}%</b><extra></extra>'
#     ))
#     print("\n=== HEATMAP DEBUG ===")
#     print("test_val shape:", test_val.shape)
#     print("test_val columns:")
#     print(test_val.columns.tolist()[:50])
#
#     print("df_raw_plot shape:", df_raw_plot.shape)
#     print("df_raw_plot columns:")
#     print(df_raw_plot.columns.tolist()[:20])
#
#     overlay_added = _draw_bbox_overlays(fig, pipe_number, folder_path, y_bands)
#
#     fig.update_layout(
#         title=dict(
#             text=f'Hall-Sensor Heatmap — Joint Number {pipe_number}',
#             x=0.5, xanchor='center',
#             font=dict(size=18, family='Arial Black')
#         ),
#         xaxis_title=x_label,
#         yaxis_title=' ',
#         width=1500, height=500,
#         font=dict(size=12),
#         xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#         yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#     )
#     fig.update_yaxes(autorange='reversed')
#
#     out_path = f'{folder_path}/hallsensor_heatmap_v2_{pipe_number}.html'
#     write_plotly_html(fig, out_path)
#     print(f'Saved hall heatmap v2: {out_path}')
#     print(f'Overlays: {"Yes" if overlay_added else "None found"}')

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Defect CSV loader (cached per session) ────────────────────────────────────

_DEFECTS_DF_CACHE = None

def _load_defects_df(csv_path: str) -> pd.DataFrame:
    global _DEFECTS_DF_CACHE
    if _DEFECTS_DF_CACHE is not None:
        return _DEFECTS_DF_CACHE
    if os.path.exists(csv_path):
        _DEFECTS_DF_CACHE = pd.read_csv(csv_path)
        print(f"[bbox] Loaded: {csv_path}  ({len(_DEFECTS_DF_CACHE)} rows)")
    else:
        _DEFECTS_DF_CACHE = pd.DataFrame()
        print(f"[bbox] WARNING: not found: {csv_path}")
    return _DEFECTS_DF_CACHE








# ─────────────────────────────────────────────────────────────────────────────
# Hall-Sensor Heatmap v2  —  SAME SIGNATURE as original
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Proximity-Sensor Heatmap v2  —  SAME SIGNATURE as original
# ───────────────────────────────────────────────────────────────────────


def create_plots_hall(pkl_path, df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder, df_new_tab10, datafile_original):
    folder_path = f'{output_folder}/Pipe_{pipe_number}'
    os.makedirs(folder_path, exist_ok=True)
    # save_lineplot(pkl_path, folder_path, test_val, datafile, pipe_number)
    # save_pipe3d(test_val, test_val, folder_path, pipe_number, pkl_path)
    # save_interactive_heatmap(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10)
    save_interactive_heatmap_v2(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10)


def create_plots_proximity(pkl_path, df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder, df_new_tab10, datafile_original):
    folder_path = f'{output_folder}/Pipe_{pipe_number}'
    os.makedirs(folder_path, exist_ok=True)
    # save_proximity_linechart(folder_path, datafile, pipe_number)
    # save_interactive_heatmap_proximity(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10)


def save_heatmap(test_val, datafile, map_ori_sens, folder_path, pipe_number):
    fighm = go.Figure(data=go.Heatmap(
        z=test_val.T, y=test_val.columns,
        x=(datafile['ODDO1'] / 1000).round(2),
        colorscale='jet',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],
    ))
    fighm.update_layout(xaxis_title='Absolute Distance (m)', height=500, width=1500,
                        margin=dict(l=20, r=20, t=50, b=20))
    write_plotly_html(fighm, f'{folder_path}/heatmap{pipe_number}.html')


FP_PATTERN = re.compile(r'^F\d+P\d+$', re.IGNORECASE)


# def save_proximity_linechart(
#     folder_path: str,
#     datafile: pd.DataFrame,
#     pipe_number,
#     *,
#     offset_step: float = 0.10,
#     dtick: int = 1000,
#     x_pref: str = "auto"
# ):
#     from scipy.signal import lfilter
#     from collections import defaultdict
#
#     df = datafile.copy()
#     candidates = [c for c in df.columns if isinstance(c, str) and FP_PATTERN.match(c.strip())]
#     if not candidates:
#         print(f"No F#P# columns found for pipe {pipe_number}. Skipping.")
#         return
#
#     res_cols = []
#     for c in candidates:
#         if not is_numeric_dtype(df[c]):
#             coerced = pd.to_numeric(df[c], errors='coerce')
#             if coerced.notna().any():
#                 df[c] = coerced
#         if is_numeric_dtype(df[c]):
#             res_cols.append(c)
#
#     if not res_cols:
#         print(f"No numeric F#P# columns for pipe {pipe_number}. Skipping.")
#         return
#
#     df1 = df.fillna(method='ffill')
#
#     if x_pref.lower() == "oddo1" or (x_pref == "auto" and "ODDO1" in df1.columns):
#         x_vals = (pd.to_numeric(df1["ODDO1"], errors="coerce") / 1000.0).round(3)
#         x_label = "Abs. Distance (m) — ODDO1"
#     else:
#         x_vals = df1.index
#         x_label = "Index"
#
#     scaler = MinMaxScaler()
#     scaled_values = scaler.fit_transform(df1[res_cols])
#     for i, col in enumerate(res_cols):
#         df1[col] = scaled_values[:, i]
#
#     n = 30
#     b = [1.0 / n] * n
#     a = 1
#     offsets = [round(i * offset_step, 3) for i in range(len(res_cols))]
#
#     fig = go.Figure()
#     for i, col in enumerate(res_cols):
#         yy = lfilter(b, a, df1[col])
#         fig.add_trace(go.Scatter(
#             x=x_vals, y=yy + offsets[i], mode='lines',
#             line=dict(width=1), hoverinfo='x+y+name',
#             showlegend=False, name=col
#         ))
#
#     fig.update_layout(
#         title=dict(text=f"Proximity-Sensor Lineplot — Joint Number {pipe_number}",
#             x=0.5, xanchor="center", font=dict(size=18, family="Arial Black")),
#         width=1500, height=500, margin=dict(l=20, r=20, t=50, b=20),
#         template='plotly_white', showlegend=False, xaxis_title=x_label,
#     )
#
#     num_ticks = 12
#     tick_positions = np.linspace(0, len(x_vals) - 1, num_ticks).astype(int)
#     if hasattr(x_vals, "iloc"):
#         tickvals = [x_vals.iloc[i] for i in tick_positions]
#     else:
#         tickvals = [x_vals[i] for i in tick_positions]
#     ticktext = [f"{v:.3f}" for v in tickvals]
#
#     fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickfont=dict(size=10),
#                      tickangle=0, showgrid=True, gridcolor="rgba(0,0,0,0.10)")
#     fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
#     # fig.update_yaxes(autorange="reversed")
#
#     try:
#         x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
#     except Exception:
#         x_min, x_max = (float(df1.index.min()), float(df1.index.max()))
#
#     y_bands_dummy = ["00:00", "06:00"]
#     pts = _load_overlay_points_for_pipe(pipe_number, y_bands_dummy, folder_path)
#     if pts is not None:
#         xs, _, labels = pts
#         xs_labels = [(float(x), str(lbl)) for x, lbl in zip(xs, labels) if x_min <= float(x) <= x_max]
#         at_x = defaultdict(list)
#         for x, lbl in xs_labels:
#             at_x[x].append(lbl)
#         for x, lbls in at_x.items():
#             fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1,
#                           xref="x", yref="paper",
#                           line=dict(color="black", width=1, dash="dot"))
#             fig.add_annotation(x=x, y=1.02, xref="x", yref="paper",
#                                 text=", ".join(lbls), showarrow=False,
#                                 bgcolor="red", bordercolor="black", borderwidth=1,
#                                 font=dict(color="white", size=10, family="Arial Black"),
#                                 align="center")
#
#     write_plotly_html(fig, f'{folder_path}/proximity_linechart{pipe_number}.html')
#     print(f"Saved {folder_path}/proximity_linechart{pipe_number}.html")

def save_proximity_linechart(
    folder_path: str,
    datafile: pd.DataFrame,
    pipe_number,
    *,
    offset_step: float = 0.3,
    x_pref: str = "auto"
):
    from scipy.signal import lfilter
    from collections import defaultdict

    df = datafile.copy()

    # Proximity sensor columns
    res = [c for c in df.columns if isinstance(c, str) and re.match(r'^F\d+P\d+$', c.strip())]
    if not res:
        print(f"No F#P# columns found for pipe {pipe_number}. Skipping.")
        return

    df1 = df[res].apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)

    # MinMaxScale (same as plot_linechart_sensor prox section)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df1[res])
    for i, col in enumerate(res):
        df1[col] = scaled_values[:, i]

    # Stack with offset
    offsets = [round(i * offset_step, 3) for i in range(len(res))]

    # Moving average filter
    n = 15
    b = [1.0 / n] * n
    a = 1

    # X axis
    if x_pref.lower() == "oddo1" or (x_pref == "auto" and "ODDO1" in df.columns):
        x_vals = (pd.to_numeric(df["ODDO1"], errors="coerce") / 1000.0).round(3)
        x_label = "Abs. Distance (m) — ODDO1"
    else:
        x_vals = df.index.to_series()
        x_label = "Index"

    fig = go.Figure()
    for i, col in enumerate(res):
        yy = lfilter(b, a, df1[col])
        fig.add_trace(go.Scatter(
            x=x_vals, y=yy + offsets[i], mode='lines',
            line=dict(width=1), name=col,
            hoverinfo='x+y+name', showlegend=False
        ))

    fig.update_layout(
        title=dict(text=f"Proximity-Sensor Lineplot — Joint Number {pipe_number}",
            x=0.5, xanchor="center", font=dict(size=18, family="Arial Black")),
        width=1500, height=500, margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_white', xaxis_title=x_label,
    )

    num_ticks = 12
    tick_positions = np.linspace(0, len(x_vals) - 1, num_ticks).astype(int)
    tickvals = [x_vals.iloc[i] if hasattr(x_vals, "iloc") else x_vals[i] for i in tick_positions]
    ticktext = [f"{v:.3f}" for v in tickvals]

    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickfont=dict(size=10),
                     tickangle=0, showgrid=True, gridcolor="rgba(0,0,0,0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")

    # Overlays
    try:
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    except Exception:
        x_min, x_max = float(df.index.min()), float(df.index.max())

    y_bands_dummy = ["00:00", "06:00"]
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands_dummy, folder_path)
    if pts is not None:
        xs, _, labels = pts
        at_x = defaultdict(list)
        for x, lbl in [(float(x), str(l)) for x, l in zip(xs, labels) if x_min <= float(x) <= x_max]:
            at_x[x].append(lbl)
        for x, lbls in at_x.items():
            fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1,
                          xref="x", yref="paper",
                          line=dict(color="black", width=1, dash="dot"))
            fig.add_annotation(x=x, y=1.02, xref="x", yref="paper",
                                text=", ".join(lbls), showarrow=False,
                                bgcolor="red", bordercolor="black", borderwidth=1,
                                font=dict(color="white", size=10, family="Arial Black"))

    write_plotly_html(fig, f'{folder_path}/proximity_linechart{pipe_number}.html')
    print(f"Saved {folder_path}/proximity_linechart{pipe_number}.html")


def save_lineplot(pkl_path, folder_path, test_val, datafile, pipe_number):
    from scipy.signal import savgol_filter, lfilter

    figmlp = go.Figure()
    offset_step = 1400

    print(f"pkl path received for pipe number: {pipe_number} is {pkl_path}")
    df_pipe = pd.read_pickle(pkl_path)

    F_columns = 36
    res = [f'F{i}H{j}' for i in range(1, F_columns + 1) for j in range(1, 5)]

    df1 = df_pipe[res].apply(pd.to_numeric, errors='coerce')
    x_vals = df_pipe['index']
    abs_dist_vals = (df_pipe['ODDO1'] / 1000).values
    x_vals_arr = x_vals.values

    def dist_to_index(dist_m):
        idx = np.argmin(np.abs(abs_dist_vals - dist_m))
        return float(x_vals_arr[idx])

    window_length = 15
    polyorder = 2
    n = 15
    b = [1.0 / n] * n
    a = 1

    for i, col in enumerate(res):
        data = df1[col].values
        time_index = np.arange(len(df1))
        coeffs = np.polyfit(time_index, data, polyorder)
        trend = np.polyval(coeffs, time_index)
        detrended = data - trend
        smoothed = savgol_filter(detrended, window_length, polyorder)
        offset_data = smoothed + i * offset_step
        filtered_data = lfilter(b, a, offset_data)
        figmlp.add_trace(go.Scatter(
            x=x_vals, y=filtered_data, mode='lines', line=dict(width=1),
            showlegend=False, name=col, customdata=abs_dist_vals,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Index: %{x}<br>"
                "Abs Distance: %{customdata:.2f} m<br>Amplitude: %{y:.1f}<extra></extra>"
            )
        ))

    valid_mask = ~np.isnan(abs_dist_vals)
    if valid_mask.any():
        all_x = x_vals.values[valid_mask]
        all_d = abs_dist_vals[valid_mask]
        n_ticks = 20
        idx = np.round(np.linspace(0, len(all_x) - 1, n_ticks)).astype(int)
        tick_x = all_x[idx]
        tick_d = all_d[idx]
        figmlp.update_xaxes(
            tickmode='array', tickvals=tick_x.tolist(),
            ticktext=[f"{d:.1f}m" for d in tick_d],
            tickangle=45, tickfont=dict(size=8), title_text="Abs Distance (m)"
        )

    try:
        y_bands = list(test_val.columns)
        pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
        print(f"DEBUG: pts = {pts}")
        if pts is not None:
            xs, ys, labels = pts
            for x_dist, y_band, label in zip(xs, ys, labels):
                if y_band not in y_bands:
                    continue
                x = dist_to_index(x_dist)
                if not (np.nanmin(x_vals) <= x <= np.nanmax(x_vals)):
                    continue
                y_idx = y_bands.index(y_band)
                y_pos = y_idx * offset_step
                figmlp.add_trace(go.Scatter(
                    x=[x], y=[y_pos], mode="markers",
                    marker=dict(size=8, color="red", line=dict(width=1, color="black")),
                    showlegend=False, name=f"{label} @ {x_dist:.2f}m",
                    hovertemplate=f"<b>{label}</b><br>Abs Dist: {x_dist:.2f} m<br>Band: {y_band}<extra></extra>"
                ))
                figmlp.add_annotation(
                    x=x, y=y_pos, text=str(label), showarrow=True, arrowhead=2,
                    arrowsize=1, arrowwidth=1, ax=0, ay=-20,
                    bgcolor="red", bordercolor="black",
                    font=dict(color="white", size=10, family="Arial Black")
                )
    except Exception as e:
        print(f"Overlay labels on lineplot failed: {e}")
        traceback.print_exc()

    figmlp.update_layout(
        template='plotly_white', height=500, width=1500,
        margin=dict(l=20, r=20, t=50, b=20), showlegend=False,
        title=dict(text=f"Hall-Sensor Lineplot — Joint Number {pipe_number}",
            x=0.5, xanchor="center", font=dict(size=18, family="Arial Black"))
    )

    step = 6
    total = len(res)
    tick_indices = list(range(0, total, step))
    figmlp.update_yaxes(
        tickmode='array',
        tickvals=[i * offset_step for i in tick_indices],
        ticktext=[res[i] for i in tick_indices],
        tickfont=dict(size=9), autorange="reversed"
    )

    write_plotly_html(figmlp, f'{folder_path}/lineplot{pipe_number}.html')


def save_lineplot_raw(folder_path, test_val, pipe_number):
    figmlpraw = go.Figure()
    for _, col in enumerate(test_val.columns):
        figmlpraw.add_trace(go.Scatter(
            x=test_val.index, y=test_val[col], mode='lines', name=col,
            line=dict(width=1), hoverinfo='x+y+name', showlegend=False
        ))
    figmlpraw.update_layout(
        xaxis_title='Counter', template='plotly_white',
        height=300, width=1500, margin=dict(l=20, r=20, t=50, b=20)
    )
    write_plotly_html(figmlpraw, f'{folder_path}/lineplot_raw{pipe_number}.html')


def save_pipe3d(data, data_cp, folder_path, pipe_number, pkl_path):
    df_pipe = pd.read_pickle(pkl_path)
    oddo_vals = (df_pipe['ODDO1'] / 1000).values

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.shape[0] > 1500:
        data = data[::2, :]
        oddo_vals = oddo_vals[::2]
    if data.shape[1] > 128:
        data = data[:, ::2]

    num_rows, num_cols = data.shape
    if len(oddo_vals) != num_rows:
        oddo_vals = np.interp(
            np.linspace(0, 1, num_rows),
            np.linspace(0, 1, len(oddo_vals)),
            oddo_vals
        )

    theta = np.linspace(0, 2 * np.pi, num_cols)
    theta_grid, _ = np.meshgrid(theta, np.zeros(num_rows))

    radius = 109.5
    odometer_start = float(np.nanmin(oddo_vals))
    odometer_end   = float(np.nanmax(oddo_vals))
    dist_range     = odometer_end - odometer_start

    x  = np.outer(oddo_vals, np.ones(num_cols))
    y  = radius * np.cos(theta_grid)
    zc = radius * np.sin(theta_grid)

    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=zc, surfacecolor=data,
        colorscale='jet', customdata=data_cp, showscale=False,
        hovertemplate='Dist: %{x:.2f} m<br>Value: %{surfacecolor:.2f}<extra></extra>'
    )])

    clock_labels = [
        dict(y=0,       z=radius,  text="12", name="12 o'clock"),
        dict(y=radius,  z=0,       text="3",  name="3 o'clock"),
        dict(y=0,       z=-radius, text="6",  name="6 o'clock"),
        dict(y=-radius, z=0,       text="9",  name="9 o'clock"),
    ]
    for cl in clock_labels:
        fig.add_trace(go.Scatter3d(
            x=[odometer_start, odometer_end],
            y=[cl['y'], cl['y']], z=[cl['z'], cl['z']],
            text=[cl['text'], cl['text']], mode='text',
            textposition="middle center", marker=dict(size=0),
            name=cl['name'], textfont=dict(size=16, color="#61090c"), showlegend=False
        ))

    x_ratio = max(4, min((dist_range / (2 * radius)) * 6, 14))
    cam_dist = 1.5 + x_ratio * 0.3
    camera = dict(
        eye=dict(x=0.1, y=-3.5, z=0.3),
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='Abs Distance (m)', yaxis_title='', zaxis_title='',
            aspectmode='manual', aspectratio=dict(x=x_ratio, y=1, z=1),
            camera=camera,
            xaxis=dict(showgrid=True, range=[odometer_start, odometer_end], autorange=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False),
        ),
        scene_dragmode='orbit', height=600, width=1500,
        title=dict(text=f'Pipe 3D Visualization — Joint {pipe_number}',
            x=0.5, xanchor='center', font=dict(size=18, family='Arial Black')),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    write_plotly_html(fig, f'{folder_path}/pipe3d{pipe_number}.html')


def _resolve_workers(workers):
    if workers in (None, 0, -1, "auto"):
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1)
    if isinstance(workers, int):
        cpu = os.cpu_count() or 1
        return max(1, min(workers, cpu))
    return 1


import time as _time_module

def _process_one_pkl(pkl_path, output_folder):
    start_time = _time_module.time()
    start_clock = datetime.now().strftime("%H:%M:%S")
    sensor_type = ["Hall", "Proximity"]

    try:
        pipe_number = int(Path(pkl_path).stem)
        print(f"\n🟢 PIPE {pipe_number} START: {start_clock}", flush=True)
        pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
        pipe_folder.mkdir(exist_ok=True)

        data = pd.read_pickle(pkl_path)
        hall, prox = count_pattern_minute_degree(pkl_path)

        total_sensors_count_hall = hall["count"]
        column_names_hall        = hall["columns"]
        minute_sensors_hall      = hall["minute"]
        degree_sensors_hall      = hall["degree"]

        total_sensors_count_prox = prox["count"]
        column_names_prox        = prox["columns"]
        minute_sensors_prox      = prox["minute"]
        degree_sensors_prox      = prox["degree"]

        for current_sensor_type in sensor_type:
            try:
                print(f"\n{'='*50}\n🚀 START SENSOR TYPE: {current_sensor_type}\nPIPE: {pipe_number}\n{'='*50}", flush=True)

                if current_sensor_type == "Hall":
                    print(f"📌 HALL: count={total_sensors_count_hall}, first_10={column_names_hall[:10]}", flush=True)
                    dfile = pre_process_data(
                        pkl_path, data, pipe_number, output_folder,
                        total_sensors_count_hall, column_names_hall,
                        minute_sensors_hall, degree_sensors_hall,
                        current_sensor_type, debug=True
                    )
                else:
                    print(f"📌 PROX: count={total_sensors_count_prox}, first_10={column_names_prox[:10]}", flush=True)
                    (dfile, df_new_tab9, datafile_original, test_val, map_ori_sens, df_new_tab10) = pre_process_data(
                        pkl_path, data, pipe_number, output_folder,
                        total_sensors_count_prox, column_names_prox,
                        minute_sensors_prox, degree_sensors_prox,
                        current_sensor_type, debug=True
                    )

                print(f"🏁 COMPLETED SENSOR TYPE: {current_sensor_type}\n", flush=True)

            except Exception as e:
                print(f"\n❌ CRASH\nPIPE: {pipe_number}\nSENSOR: {current_sensor_type}\n{traceback.format_exc()}", flush=True)
                raise

        xlsx_path = pipe_folder / f"Pipe_{pipe_number}.xlsx"
        dfile.to_excel(xlsx_path, index=False)

        total_time = round(_time_module.time() - start_time, 2)
        print(f"🔴 PIPE {pipe_number} END | TOTAL: {total_time}s ({round(total_time/60,2)} min)\n", flush=True)
        return f"Processed {os.path.basename(pkl_path)} and saved to {pipe_folder}"

    except Exception as e:
        print(f"Error loading {os.path.basename(pkl_path)}: {e}")
        traceback.print_exc()
        return f"Error loading {os.path.basename(pkl_path)}: {e}"


def count_pattern_minute_degree(datafile_path):
    df = pd.read_pickle(datafile_path)
    pattern_hall      = re.compile(r'^F\d+H\d+$', re.IGNORECASE)
    pattern_proximity = re.compile(r'^F\d+P\d+$', re.IGNORECASE)

    matching_columns_hall      = [col for col in df.columns if pattern_hall.match(col)]
    matching_columns_proximity = [col for col in df.columns if pattern_proximity.match(col)]

    count_hall = len(matching_columns_hall)
    minute_sensors_hall  = 720 / count_hall      if count_hall > 0 else None
    degree_sensors_hall  = minute_sensors_hall/2 if count_hall > 0 else None

    count_proximity = len(matching_columns_proximity)
    minute_sensors_proximity  = 720 / count_proximity      if count_proximity > 0 else None
    degree_sensors_proximity  = minute_sensors_proximity/2 if count_proximity > 0 else None

    hall = {"columns": matching_columns_hall, "count": count_hall,
            "minute": minute_sensors_hall, "degree": degree_sensors_hall}
    proximity = {"columns": matching_columns_proximity, "count": count_proximity,
                 "minute": minute_sensors_proximity, "degree": degree_sensors_proximity}

    print(f"hall sensor details: {hall} \n proximity sensor details: {proximity}")
    return hall, proximity


def create_html_and_csv_from_pkl(
    pkl_folder='pipes3',
    output_folder='Client_Pipes',
    output_callback=None,
    workers=WORKERS
):
    try:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        msg = f"❌ Failed creating output folder '{output_folder}': {e}"
        (output_callback or print)(msg)
        return

    try:
        pkl_paths = [str(Path(pkl_folder)/f) for f in os.listdir(pkl_folder) if f.lower().endswith('.pkl')]
    except Exception as e:
        msg = f"❌ Failed reading PKL folder '{pkl_folder}': {e}"
        (output_callback or print)(msg)
        return

    if not pkl_paths:
        msg = f"⚠ No .pkl files found in {pkl_folder}"
        (output_callback or print)(msg)
        return

    n_jobs = _resolve_workers(workers)

    def safe_process(pkl_path):
        import time, traceback, pandas as pd
        fname = os.path.basename(pkl_path)
        start = time.time()
        try:
            print(f"\n🔍 STARTING: {fname}")
            test_obj = pd.read_pickle(pkl_path)
            print(f"✅ PKL LOAD OK: {fname} | shape={getattr(test_obj,'shape','?')}")
            result = _process_one_pkl(pkl_path, output_folder)
            print(f"🏁 FINISHED: {fname} in {round(time.time()-start,2)}s")
            return result
        except Exception as e:
            return (f"\n❌ ERROR: {fname}\n⏱ {round(time.time()-start,2)}s\n{str(e)}\n{traceback.format_exc()}")

    try:
        results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(safe_process)(p) for p in pkl_paths
        )
    except Exception as e:
        msg = f"❌ Parallel execution crashed: {e}\n{traceback.format_exc()}"
        (output_callback or print)(msg)
        return

    for msg in results:
        (output_callback or print)(msg)










# -------------------- MAIN --------------------
if __name__ == "__main__":
    import time
    st = time.time()
    create_html_and_csv_from_pkl(workers=WORKERS)
    print(f'Total time: {time.time()-st} seconds')