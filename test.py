import os
# Limit intra-process math threads to avoid over-subscription
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


# ------------ Plotly export options (keep interactive + smaller HTML) ------------
PLOTLY_JS_MODE   = "directory"   # "directory" (offline, small HTML) or "cdn" (requires internet)
PLOTLY_COMPRESS  = True          # compress embedded data
HTML_DEFAULT_W   = "100%"        # let the container control size
HTML_DEFAULT_H   = 500           # or "100%" if you want full-height wrappers

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d"
    ],
    # , "zoomIn2d", "zoomOut2d",
    #     "pan2d", "toImage", "hoverCompareCartesian", "hoverClosestCartesian"
    "displayModeBar": True, 
    "scrollZoom": False,
}

# -------------------- CONFIG --------------------
INITIAL_READ = 0.0      # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3
# -1 / 0 / None / "auto" => auto (CPU-1, at least 1). Or set an int, e.g. 4
WORKERS = 4
# ------------------------------------------------


def pre_process_data(datafile, pipe_number, output_folder, total_sensors, column_names, minute_sensors, degree_sensors):

    datafile_original = datafile.copy(deep=True)
    df_new_tab9 = pd.DataFrame(
        datafile,
        columns=column_names
    )

    df_new_tab10 = df_new_tab9.copy()
    sensor_columns = df_new_tab9.columns.tolist()

    # Denoising using Savitzky-Golay filter
    window_length = 15
    polyorder = 2
    for col in sensor_columns:
        data = df_new_tab9[col].values
        time_index = np.arange(len(df_new_tab9))
        trend = np.polyval(np.polyfit(time_index, data, 2), time_index)
        data_denoised = savgol_filter(data - trend, window_length, polyorder)
        df_new_tab9[col] = data_denoised

    df_raw_straight = df_new_tab9.copy()
    
    # Setting bounds and applying conditions
    sens_mean = df_new_tab9.abs().mean()
    standard_deviation = df_new_tab9.std(axis=0, skipna=True)

    mean_plus_sigma = sens_mean + UPPER_SENS_MUL * standard_deviation
    mean_negative_sigma = sens_mean - LOWER_SENS_MUL * standard_deviation

    # Apply noise filtering to zero-out in-bound values
    for col in df_new_tab9.columns:
        if col in mean_plus_sigma.index and col in mean_negative_sigma.index:
            df_new_tab9[col] = np.where(
                (df_new_tab9[col] >= mean_negative_sigma[col]) &
                (df_new_tab9[col] <= mean_plus_sigma[col]),
                0,
                df_new_tab9[col]
            )

    initial_read = INITIAL_READ
    roll = datafile['ROLL'] - initial_read

    def degrees_to_hours_minutes2(degrees):
        if (degrees < 0):
            degrees = degrees % 360
        elif degrees >= 360:
            degrees %= 360
        degrees_per_second = 360 / (12 * 60 * 60)
        total_seconds = degrees / degrees_per_second
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}"

    def add_sensor_keys(d):
        for e in d:
            new_dict = {**e}
            for i in range(1, total_sensors):
                new_dict[f'Roll_Sensor_{i}'] = e['Roll_Sensor_0'] + (degree_sensors * i)
            yield new_dict

    def check_time_range(time_str):
        start_time = list(time_dict_1.keys())[0]
        end_time_dt = datetime.strptime(list(time_dict_1.keys())[1], '%H:%M') - timedelta(seconds=1)
        end_time = list(time_dict_1.keys())[1]
        time_to_check = datetime.strptime(time_str, '%H:%M')
        start_time_dt = datetime.strptime(start_time, '%H:%M')
        return start_time_dt <= time_to_check <= end_time_dt

    d = []
    for pos in roll:
        d.append({f"Roll_Sensor_0": pos})

    upd_d = list(add_sensor_keys(d))
    oriData = pd.DataFrame.from_dict(data=upd_d)
    clockData = oriData.applymap(degrees_to_hours_minutes2)

    test_clockData = clockData.copy()

    # Parse flexibly with mixed formats (works for both HH:MM and HH:MM:SS)
    test_clockData = test_clockData.apply(pd.to_datetime, format='mixed')

    # Now format the datetime objects to strings 'HH:MM' dropping seconds
    test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M'))
    test_clockData = test_clockData.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
    test_clockData = test_clockData.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
    test_clockData = test_clockData.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

    test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_new_tab9.columns)))

    def create_time_dict():
        time_list = [timedelta(minutes=i * minute_sensors) for i in range(total_sensors)]
        time_ranges_2 = [(datetime.min + t).strftime('%H:%M') for t in time_list]
        return {key: [] for key in time_ranges_2}

    time_dict_1 = create_time_dict()
    rang = list(time_dict_1.keys())

    for _, row in test_clockData.iterrows():
        xl = list(row)
        xd = dict(row)
        xkeys = list(xd.keys())
        c = 0
        for _, dval in xd.items():
            if check_time_range(dval):
                ind = xl.index(dval)
                _ = xl[ind:] + xl[:ind]  # not used later but kept for clarity
                break

        curr = ind
        while True:
            ck = xkeys[curr]
            time_dict_1[rang[c]].append((curr, ck, xd[ck]))
            c += 1
            curr = (curr + 1) % len(xkeys)
            if curr == ind:
                break

    map_ori_sens = pd.DataFrame(time_dict_1)

    val_ori_sens = map_ori_sens.copy()

    def extract_string(cell):
        return cell[1]

    val_ori_sens = val_ori_sens.applymap(extract_string)

    test_val = val_ori_sens.copy()

    for r, e in val_ori_sens.iterrows():
        c = 0
        for _, tup_value in e.items():
            cell_v = df_new_tab9.at[r, tup_value]
            test_val.iloc[r, c] = cell_v
            c += 1

    map_val_sens = pd.DataFrame(index=test_val.index, columns=test_val.columns)
    for column in test_val.columns:
        for i in range(test_val.shape[0]):
            map_value = map_ori_sens.loc[i, column]
            test_value = test_val.loc[i, column]
            map_val_sens.loc[i, column] = (*map_value, test_value)
    
    create_plots(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original)
    return datafile



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



def pre_process_for_interactive_heatmap(df_in: pd.DataFrame, datafile: pd.DataFrame, test_val: pd.DataFrame, map_ori_sens: pd.DataFrame):
    """Process data specifically for interactive heatmap - percentage-based."""
    
    # Get sensor columns (use raw data, not the processed df_new_tab9)
    sens_cols = df_in.columns.tolist()
    print(f"Sensor columns in datafile for preprocess_heatmap: {sens_cols}")
    if not sens_cols:
        raise ValueError("No F*H* sensor columns found in the input DataFrame.")

    df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
    df_sens_raw = df_sens.copy(deep=True)  # Keep raw copy for CSV export
    df_mean_cols = df_sens_raw
    print(list(df_mean_cols.columns))
    Mean1 = df_mean_cols.mean()
    df_raw_plot = ((df_mean_cols - Mean1)/Mean1)*100
    
    # Convert all sensor data to numeric
    for col in sens_cols:
        df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    
    # Fill NaN values with forward fill, then zeros
    df_sens = df_sens.fillna(method='ffill').fillna(0.0)
    
    # Calculate mean for percentage normalization
    Mean1 = df_sens.mean()
    
    # Normalize as percentage deviation from mean
    df_sens_normalized = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100
    
    # Zero out values above mean threshold (different from main processing)
    # for col in sens_cols:
    #     df_sens_normalized.loc[df_sens_normalized[col] > Mean1[col], col] = 0
    
    # Apply additional filtering if needed
    # if UPPER_SENS_MUL > 0 and LOWER_SENS_MUL > 0:
    #     sens_std = df_sens.std(axis=0, skipna=True)
    #     upper = Mean1 + UPPER_SENS_MUL * sens_std
    #     lower = Mean1 - LOWER_SENS_MUL * sens_std
        
    #     for col in sens_cols:
    #         mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
    #         df_sens_normalized.loc[mask, col] = 0

    # Rearrange data to match test_val structure using map_ori_sens
    df_plot_rearranged = test_val.copy()
    for r in range(len(test_val)):
        for c, band in enumerate(test_val.columns):
            # Get the sensor mapping from map_ori_sens
            sensor_info = map_ori_sens.iloc[r, c]
            if isinstance(sensor_info, tuple) and len(sensor_info) >= 2:
                sensor_col = sensor_info[1]  # sensor column name
                if sensor_col in df_sens_normalized.columns:
                    df_plot_rearranged.iloc[r, c] = df_sens_normalized.loc[r, sensor_col]
    
    return df_plot_rearranged, df_raw_plot

def save_interactive_heatmap_proximity(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10):
    """Create and save interactive heatmap with overlays."""
    # # case-insensitive F#P# like F1P1, F12P3
    pattern = re.compile(r'^F\d+P\d+$', re.IGNORECASE)
    datafile_original = datafile.copy(deep=True)
    
    pattern = re.compile(r'^F\d+P\d+$')
    matching_columns = [col for col in datafile_original.columns if pattern.match(col)]
    df_new_tab9 = pd.DataFrame(
        datafile,
        columns=matching_columns
    )
    print("doing processing with datacolms: ", df_new_tab9.columns)
    df_new_tab10 = df_new_tab9.copy()

    # Use the specialized preprocessing for interactive heatmap
    df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)
    
    # Get x-axis values
    if 'ODDO1' in datafile.columns:
        x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m)'
    else:
        x_vals = pd.Series(np.arange(len(datafile)))
        x_label = 'Index'
    
    # Get y-band labels
    y_bands = [str(c) for c in test_val.columns]
    
    # Create heatmap data - transpose so y_bands are on y-axis
    heatmap_data = df_raw_plot.T
    
    # Ensure all data is numeric
    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    
    # Replace infinite values
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    
    heatmap_data = heatmap_data.astype("float32").round(3)
    
    # Create the interactive heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_vals.round(2),
        y=y_bands,
        colorscale='jet',
        # zmin=-4,
        # zmax=8,
        colorbar=dict(title="Sensor Value (%)"),
        showscale=True,
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>' +
                     '<b>%{y}</b><br>' +
                     '<b>Value: %{z:.2f}%</b>' +
                     '<extra></extra>'
    ))
    
    # Add overlay points with FIXED SIZE SQUARES
    overlay_added = False
    # pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
    pts = None
    if pts is not None:
        xs, ys, labels = pts
        
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                
                # Fixed size square
                fig.add_shape(
                    type="rect",
                    x0=x - 0.05, y0=y_idx - 0.35,
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
        title=dict(
        text=f"Proximity-Sensor Heatmap — Joint Number {pipe_number}",  # 👈 chart title
        x=0.5,        # center
        xanchor="center",
        font=dict(size=18, family="Arial Black")),  # customize font,
        xaxis_title=x_label,
        yaxis_title="Orientation (12h bands)",
        width=1500,
        height=500,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    # Save the interactive heatmap
    write_plotly_html(fig, f'{folder_path}/proximity_heatmap{pipe_number}.html')

    
    print(f"Saved interactive heatmap: {folder_path}/interactive_heatmap{pipe_number}.html")
    print(f"Overlays: {'Yes' if overlay_added else 'None found'}")



def save_interactive_heatmap(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10):
    """Create and save interactive heatmap with overlays."""
    

    # Use the specialized preprocessing for interactive heatmap
    df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)
    
    # Get x-axis values
    if 'ODDO1' in datafile.columns:
        x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m)'
    else:
        x_vals = pd.Series(np.arange(len(datafile)))
        x_label = 'Index'
    
    # Get y-band labels
    y_bands = [str(c) for c in test_val.columns]
    
    # Create heatmap data - transpose so y_bands are on y-axis
    heatmap_data = df_raw_plot.T
    
    # Ensure all data is numeric
    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    
    # Replace infinite values
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    
    heatmap_data = heatmap_data.astype("float32").round(3)
    
    # Create the interactive heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_vals.round(2),
        y=y_bands,
        colorscale='jet',
        zmin=-3,
        zmax=8,
        # colorbar=dict(title="Sensor Value (%)"),
        showscale=False,
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>' +
                     '<b>%{y}</b><br>' +
                     '<b>Value: %{z:.2f}%</b>' +
                     '<extra></extra>'
    ))
    
    # Add overlay points with FIXED SIZE SQUARES
    overlay_added = False
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
    if pts is not None:
        xs, ys, labels = pts
        
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                
                # Fixed size square
                fig.add_shape(
                    type="rect",
                    x0=x - 0.05, y0=y_idx - 0.35,
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
        title=dict(
        text=f"Hall-Sensor Heatmap — Joint Number {pipe_number}",  # 👈 chart title
        x=0.5,        # center
        xanchor="center",
        font=dict(size=18, family="Arial Black")),  # customize font,
        xaxis_title=x_label,
        yaxis_title="Orientation (12h bands)",
        width=1500,
        height=500,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    # Save the interactive heatmap
    write_plotly_html(fig, f'{folder_path}/hallsensor_heatmap{pipe_number}.html')

    
    print(f"Saved hallsensor heatmap: {folder_path}/hallsensor_heatmap{pipe_number}.html")
    print(f"Overlays: {'Yes' if overlay_added else 'None found'}")

def _load_overlay_points_for_pipe(pipe_number, y_band_labels, folder_path, *, debug_prefix="OVERLAY DEBUG"):
    """Return (xs, ys, labels) for overlay markers if a PipeTally file is found.
       Only plot rows where Feature Type == 'Metal Loss' (case-insensitive).
       Skip any row missing usable x or orientation.
    """
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

    print(f"{debug_prefix}: pipe {pipe_number}: file='{path}' cols: x='{x_col}', ori='{ori_col}', feature='{feat_col}', pipe='{pipe_col}', s_no='{sno_col}'")

    # Must have x and orientation columns to plot
    if x_col is None or ori_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: missing required columns; no overlay.")
        return None

    # If there is a 'Pipe Number' column, restrict to this pipe (best-effort)
    if pipe_col is not None:
        mask = df[pipe_col].astype(str).str.contains(str(pipe_number), na=False)
        if mask.any():
            df = df[mask]

    total_rows = len(df)

    # Feature filtering: ONLY "Metal Loss" (case-insensitive, tolerant to spacing)
    # If feat_col doesn't exist, we bail (since we must only plot Metal Loss)
    if feat_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: no feature column; overlays disabled (Metal Loss only).")
        return None

    feat_series = df[feat_col].astype(str).str.strip().str.lower()
    metal_loss_mask = feat_series.str.fullmatch(r"metal\s*loss", case=False, na=False)
    df = df[metal_loss_mask]

    after_filter = len(df)

    xs, ys, labels = [], [], []
    skipped_no_x, skipped_no_ori, skipped_other = 0, 0, 0

    for _, row in df.iterrows():
        # x must be a valid number
        x = pd.to_numeric(row.get(x_col), errors="coerce")
        if pd.isna(x):
            skipped_no_x += 1
            continue

        # orientation must parse
        ori_sec = _parse_ori_to_seconds(row.get(ori_col))
        if ori_sec is None:
            skipped_no_ori += 1
            continue

        # find nearest y-band
        y = _nearest_band_label(int(ori_sec), list(y_band_labels))
        if y not in y_band_labels:
            skipped_other += 1
            continue

        # label: prefer s_no; fallback to Defect_id; fallback to running index
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
        f"{debug_prefix}: pipe {pipe_number}: total_rows={total_rows}, "
        f"after_feature='Metal Loss'={after_filter}, plotted={len(xs)}, "
        f"skipped_no_x={skipped_no_x}, skipped_no_ori={skipped_no_ori}, skipped_other={skipped_other}"
    )

    if not xs:
        return None
    return xs, ys, labels


def create_plots(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens,
                 pipe_number, output_folder, df_new_tab10, datafile_original):
    folder_path = f'{output_folder}/Pipe_{pipe_number}'
    os.makedirs(folder_path, exist_ok=True)

    # save_lineplot(folder_path, test_val, datafile, pipe_number)
    # save_pipe3d(test_val, test_val, folder_path, pipe_number)
    # save_proximity_linechart(folder_path, datafile, pipe_number)

    # # Both heatmaps
    save_interactive_heatmap(df_new_tab9, datafile_original, test_val,
                             map_ori_sens, folder_path, pipe_number, df_new_tab10)

    save_interactive_heatmap_proximity(df_new_tab9, datafile_original, test_val,
                                       map_ori_sens, folder_path, pipe_number, df_new_tab10)




# def _process_one_pkl(pkl_path, output_folder):
#     try:
#         pipe_number = Path(pkl_path).stem
#         pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
#         pipe_folder.mkdir(exist_ok=True)

#         # data = pd.read_pickle(pkl_path)
#         data = pd.read_csv(pkl_path)
#         total_sensors_count, column_names, minute_sensors, degree_sensors = count_pattern_minute_degree(pkl_path)
#         print(f" total_sensors_count: {total_sensors_count}, column_names: {column_names}, minute_sensors: {minute_sensors}, degree_sensors: {degree_sensors}")
#         dfile = pre_process_data(data, pipe_number, output_folder, total_sensors_count, column_names, minute_sensors, degree_sensors)

#         # Save the Excel
#         xlsx_path = pipe_folder / f"Pipe_{pipe_number}.xlsx"
#         dfile.to_excel(xlsx_path, index=False)

#         return f"Processed {os.path.basename(pkl_path)} and saved to {pipe_folder}"
#     except Exception as e:
#         print(f"Error loading {os.path.basename(pkl_path)}: {e}")
#         traceback.print_exc()
#         return f"Error loading {os.path.basename(pkl_path)}: {e}"


def _process_one_pkl(pkl_path, output_folder):
    try:
        pipe_number = Path(pkl_path).stem
        pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
        pipe_folder.mkdir(exist_ok=True)

        data = pd.read_pickle(pkl_path)
        sensors = count_pattern_minute_degree(pkl_path)

        # Process Hall sensors (if present)
        if sensors["hall"]["count"] > 0:
            print(f"[PIPE {pipe_number}] Processing HALL sensors ({sensors['hall']['count']})...")
            dfile_hall = pre_process_data(
                data, pipe_number, output_folder,
                sensors["hall"]["count"],
                sensors["hall"]["cols"],
                sensors["hall"]["minute"],
                sensors["hall"]["degree"]
            )

        # Process Proximity sensors (if present)
        if sensors["prox"]["count"] > 0:
            print(f"[PIPE {pipe_number}] Processing PROXIMITY sensors ({sensors['prox']['count']})...")
            dfile_prox = pre_process_data(
                data, pipe_number, output_folder,
                sensors["prox"]["count"],
                sensors["prox"]["cols"],
                sensors["prox"]["minute"],
                sensors["prox"]["degree"]
            )

        # Save combined Excel
        xlsx_path = pipe_folder / f"Pipe_{pipe_number}.xlsx"
        data.to_excel(xlsx_path, index=False)
        return f"Processed {os.path.basename(pkl_path)} with Hall+Proximity → {pipe_folder}"

    except Exception as e:
        print(f"Error processing {os.path.basename(pkl_path)}: {e}")
        traceback.print_exc()
        return f"Error processing {os.path.basename(pkl_path)}: {e}"


# def count_pattern_minute_degree(datafile_path):
#     # df = pd.read_pickle(datafile_path)
#     df = pd.read_csv(datafile_path)

#     pattern = re.compile(r'^F\d+P\d+$')
#     matching_columns = [col for col in df.columns if pattern.match(col)]
#     count = len(matching_columns)
#     minute_sensors = 720 / count 
#     degree_sensors = minute_sensors / 2
#     return count, matching_columns, minute_sensors, degree_sensors


def count_pattern_minute_degree(datafile_path):
    df = pd.read_pickle(datafile_path)
    pattern_hall = re.compile(r'^F\d+H\d+$', re.IGNORECASE)
    pattern_prox = re.compile(r'^F\d+P\d+$', re.IGNORECASE)

    hall_cols = [c for c in df.columns if pattern_hall.match(c)]
    prox_cols = [c for c in df.columns if pattern_prox.match(c)]

    hall_count = len(hall_cols)
    prox_count = len(prox_cols)

    hall_minute = 720 / hall_count if hall_count else None
    hall_degree = hall_minute / 2 if hall_count else None
    prox_minute = 720 / prox_count if prox_count else None
    prox_degree = prox_minute / 2 if prox_count else None

    return {
        "hall": dict(cols=hall_cols, count=hall_count, minute=hall_minute, degree=hall_degree),
        "prox": dict(cols=prox_cols, count=prox_count, minute=prox_minute, degree=prox_degree)
    }


_process_one_pkl(pkl_path=r"F:\work_new\client_software\test_data_cs\pickle_test\36.pkl", output_folder=r"F:\work_new\client_software\test_data_cs\pickle_test")
