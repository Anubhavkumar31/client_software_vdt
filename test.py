import re
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from Data_Gen.html_filter import _map_sensor_values_fast, _remap_sensor_timeline_fast, \
    _build_clock_data_fast, count_pattern_minute_degree, pre_process_for_interactive_heatmap, \
    _load_overlay_points_for_pipe, write_plotly_html


def generate_heatmap_from_pkl(
    pkl_path,
    folder_path,
    pipe_number
):
    """
    Fully automatic hall/proximity heatmap generation pipeline.

    Inputs:
    --------
    pkl_path     : path to raw PKL sensor file
    folder_path  : output directory
    pipe_number  : pipe/joint identifier
    """

    # =========================================================
    # LOAD PKL
    # =========================================================
    datafile = pd.read_pickle(pkl_path)

    print(f"Loaded PKL: {pkl_path}")
    print(f"Data shape: {datafile.shape}")

    # =========================================================
    # DETECT SENSOR DETAILS
    # =========================================================
    hall, prox = count_pattern_minute_degree(
        pkl_path
    )

    column_names_prox = prox["columns"]

    total_sensors = prox["count"]

    minute_sensors = prox["minute"]

    degree_sensors = prox["degree"]

    print(f"Detected sensors: {total_sensors}")
    print(f"Minute spacing : {minute_sensors}")
    print(f"Degree spacing : {degree_sensors}")

    # =========================================================
    # CREATE SENSOR DATAFRAME
    # =========================================================
    df_new_tab9 = pd.DataFrame(
        datafile,
        columns=column_names_prox
    )

    print(
        f"df_new_tab9 created | "
        f"shape={df_new_tab9.shape}"
    )

    df_new_tab10 = df_new_tab9.copy()

    # =========================================================
    # BUILD CLOCK RANGE LABELS
    # =========================================================
    time_list = [
        timedelta(minutes=i * minute_sensors)
        for i in range(total_sensors)
    ]

    time_ranges_2 = [
        (datetime.min + t).strftime('%H:%M')
        for t in time_list
    ]

    rang = time_ranges_2

    print(f"Generated rang labels: {rang[:5]} ...")

    # =========================================================
    # BUILD ROLL ARRAY
    # =========================================================
    initial_read = 0

    roll = (
        datafile['ROLL'] - initial_read
    ).to_numpy(dtype=np.float64)

    print(f"Roll shape: {roll.shape}")

    # =========================================================
    # BUILD ORIENTATION MATRIX
    # =========================================================
    offsets = (
        np.arange(
            total_sensors,
            dtype=np.float64
        ) * degree_sensors
    )

    ori_mat = (
        roll[:, None] +
        offsets[None, :]
    )

    print(f"ori_mat shape: {ori_mat.shape}")

    # =========================================================
    # CREATE ORIENTATION DATAFRAME
    # =========================================================
    col_labels_orig = [
        f'Roll_Sensor_{i}'
        for i in range(total_sensors)
    ]

    oriData = pd.DataFrame(
        ori_mat,
        columns=col_labels_orig
    )

    print(f"oriData shape: {oriData.shape}")

    # =========================================================
    # BUILD CLOCK DATA
    # =========================================================
    test_clockData = _build_clock_data_fast(
        oriData
    )

    test_clockData.columns = df_new_tab9.columns

    print(
        f"test_clockData shape: "
        f"{test_clockData.shape}"
    )

    # =========================================================
    # REMAP SENSOR TIMELINE
    # =========================================================
    map_ori_sens, val_ori_sens_labels = (
        _remap_sensor_timeline_fast(
            test_clockData,
            rang,
            total_sensors
        )
    )

    print("Sensor remapping complete.")

    # =========================================================
    # GENERATE HEATMAP LAYOUT
    # =========================================================
    test_val = _map_sensor_values_fast(
        val_ori_sens_labels,
        df_new_tab9
    )

    print(f"test_val shape: {test_val.shape}")

    # =========================================================
    # GENERATE + SAVE HEATMAP
    # =========================================================
    save_interactive_heatmap(
        df_new_tab9,
        datafile,
        test_val,
        map_ori_sens,
        folder_path,
        pipe_number,
        df_new_tab10
    )

    print(
        f"Heatmap generation completed "
        f"for pipe {pipe_number}"
    )

import plotly.graph_objects as go
# def save_interactive_heatmap(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10):
#     df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)
#
#     if 'ODDO1' in datafile.columns:
#         x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
#         x_label = 'Absolute Distance (m) --- Hall Sensor Heatmap'
#     else:
#         x_vals = pd.Series(np.arange(len(datafile)))
#         x_label = 'Index'
#
#     y_bands = [str(c) for c in test_val.columns]
#     heatmap_data = df_raw_plot.T
#
#     for col in heatmap_data.columns:
#         heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
#     heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
#     if not np.isfinite(heatmap_data.values).all():
#         heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
#     heatmap_data = heatmap_data.astype("float32").round(3)
#
#     fig = go.Figure(data=go.Heatmap(
#         z=heatmap_data.values, x=x_vals.round(2), y=y_bands,
#         colorscale='jet', zmin=-2, zmax=6, showscale=False, hoverongaps=False,
#         hovertemplate='<b>%{x}</b><br><b>%{y}</b><br><b>Value: %{z:.2f}%</b><extra></extra>'
#     ))
#
#     overlay_added = False
#     pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
#     if pts is not None:
#         xs, ys, labels = pts
#         for x, y_band, label in zip(xs, ys, labels):
#             if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
#                 y_idx = y_bands.index(y_band)
#                 fig.add_shape(type="rect",
#                     x0=x-0.05, y0=y_idx-0.35, x1=x+0.05, y1=y_idx+0.35,
#                     line=dict(color="black", width=2), fillcolor="rgba(255,0,0,0.6)")
#                 fig.add_annotation(x=x, y=y_idx, text=label, showarrow=False,
#                     font=dict(color="white", size=8, family="Arial Black"),
#                     bgcolor="red", bordercolor="black", borderwidth=1)
#                 overlay_added = True
#
#     fig.update_layout(
#         title=dict(text=f"Hall-Sensor Heatmap — Joint Number {pipe_number}",
#             x=0.5, xanchor="center", font=dict(size=18, family="Arial Black")),
#         xaxis_title=x_label, yaxis_title=" ", width=1500, height=500,
#         font=dict(size=12),
#         xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#         yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
#     )
#     fig.update_yaxes(autorange="reversed")
#     write_plotly_html(fig, f'{folder_path}/hallsensor_heatmap{pipe_number}.html')
#     print(f"Saved hallsensor heatmap: {folder_path}/hallsensor_heatmap{pipe_number}.html")
#     print(f"Overlays: {'Yes' if overlay_added else 'None found'}")

INITIAL_READ = 0.0
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3
# def pre_process_for_interactive_heatmap(df_in: pd.DataFrame, datafile: pd.DataFrame, test_val: pd.DataFrame, map_ori_sens: pd.DataFrame):
#     sens_cols = df_in.columns.tolist()
#     print(f"Sensor columns in datafile for preprocess_heatmap_hall_sensor: {sens_cols}")
#     if not sens_cols:
#         raise ValueError("No sensor columns found in the input DataFrame.")
#
#     df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
#     df_sens_raw = df_sens.copy(deep=True)
#     df_mean_cols = df_sens_raw
#     Mean1 = df_mean_cols.mean()
#     df_raw_plot = ((df_mean_cols - Mean1) / Mean1) * 100
#
#     for col in sens_cols:
#         df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
#     df_sens = df_sens.fillna(method='ffill').fillna(0.0)
#     Mean1 = df_sens.mean()
#     df_sens_normalized = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100
#
#     for col in sens_cols:
#         df_sens_normalized.loc[df_sens_normalized[col] > Mean1[col], col] = 0
#
#     if UPPER_SENS_MUL > 0 and LOWER_SENS_MUL > 0:
#         sens_std = df_sens.std(axis=0, skipna=True)
#         upper = Mean1 + UPPER_SENS_MUL * sens_std
#         lower = Mean1 - LOWER_SENS_MUL * sens_std
#         for col in sens_cols:
#             mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
#             df_sens_normalized.loc[mask, col] = 0
#
#     df_plot_rearranged = test_val.copy()
#     for r in range(len(test_val)):
#         for c, band in enumerate(test_val.columns):
#             sensor_info = map_ori_sens.iloc[r, c]
#             if isinstance(sensor_info, tuple) and len(sensor_info) >= 2:
#                 sensor_col = sensor_info[1]
#                 if sensor_col in df_sens_normalized.columns:
#                     df_plot_rearranged.iloc[r, c] = df_sens_normalized.loc[r, sensor_col]
#
#     return df_plot_rearranged, df_raw_plot



def pre_process_for_interactive_heatmap(df_in: pd.DataFrame, datafile: pd.DataFrame, test_val: pd.DataFrame, map_ori_sens: pd.DataFrame):
    sens_cols = df_in.columns.tolist()
    print(f"Sensor columns for heatmap preprocessing: {sens_cols[:10]}... ({len(sens_cols)} total)")
    if not sens_cols:
        raise ValueError("No sensor columns found in the input DataFrame.")

    # 1. Pull and clean sensor data
    df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
    for col in sens_cols:
        df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    df_sens = df_sens.ffill().fillna(0.0)

    # 2. Normalise: % deviation from column mean
    Mean1 = df_sens.mean()
    denom = Mean1.abs() + 1e-8
    df_pct = ((df_sens - Mean1) / denom) * 100
    window = 2000
    df_baseline = df_pct.rolling(window=window, center=True, min_periods=1).mean()
    df_pct = df_pct - df_baseline
    # Smooth spatially to reduce noise before plotting
    from scipy.ndimage import uniform_filter1d
    df_pct = pd.DataFrame(
        uniform_filter1d(df_pct.values, size=100, axis=0),
        columns=df_pct.columns,
        index=df_pct.index
    )

    # 3. NO sigma filter — it was zeroing out all real signal.
    #    The heatmap needs to show the full % deviation picture.

    # 4. Clip extreme outliers only (e.g. ±500%) so colorscale isn't wrecked
    df_pct = df_pct.clip(-500, 500)

    # 5. Rearrange into clock-position bands via map_ori_sens
    n_rows  = len(test_val)
    n_bands = len(test_val.columns)
    out_mat = np.zeros((n_rows, n_bands), dtype=np.float64)

    map_arr  = map_ori_sens.to_numpy()
    col_idx  = {col: i for i, col in enumerate(df_pct.columns)}
    pct_data = df_pct.to_numpy(dtype=np.float64)

    for c in range(n_bands):
        col_tuples = map_arr[:, c]
        sensor_names = np.array([
            t[1] if isinstance(t, tuple) and len(t) >= 2 else None
            for t in col_tuples
        ])
        for sensor in np.unique(sensor_names):
            if sensor is None or sensor not in col_idx:
                continue
            rows = sensor_names == sensor
            out_mat[rows, c] = pct_data[rows, col_idx[sensor]]

    df_plot_rearranged = pd.DataFrame(
        out_mat, columns=test_val.columns, index=test_val.index
    )
    df_raw_plot = df_pct.copy()

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

    # Use the clock-remapped data, NOT df_raw_plot
    heatmap_data = df_plot_rearranged.T.copy()
    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    heatmap_data = heatmap_data.astype("float32").round(3)

    nonzero = np.count_nonzero(heatmap_data.values)
    print(f"DEBUG heatmap_data: shape={heatmap_data.shape}, nonzero={nonzero}, "
          f"min={heatmap_data.values.min():.2f}, max={heatmap_data.values.max():.2f}, "
          f"mean={heatmap_data.values.mean():.4f}")

    # Auto color scale centered at zero
    # Robust color scale — ignores outlier bands
    # Colorscale based on median row, ignoring the dominant outlier band
    row_abs_mean = np.abs(heatmap_data.values).mean(axis=1)
    median_band_level = np.median(row_abs_mean)
    # Only use rows that are close to median for scale calculation
    normal_rows = row_abs_mean < (median_band_level * 5)
    flat = heatmap_data.values[normal_rows, :].ravel()
    flat = flat[np.isfinite(flat)]
    flat = flat[flat != 0]
    if len(flat) > 0:
        zmin = float(np.percentile(flat, 40))
        zmax = float(np.percentile(flat, 95))
    else:
        zmin, zmax = -2.0, 2.0


    print(f"values of zmin: {zmin} and zmax: {zmax}")

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=x_vals.round(2), y=y_bands,
        colorscale='jet', zmin=zmin, zmax=zmax,
        showscale=True, hoverongaps=False,
        colorbar=dict(title="% deviation"),
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



generate_heatmap_from_pkl(
    pkl_path=r"D:\Anubhav\softwares\client software\Data\project_oil_sample\pickle_data\7.pkl",
    folder_path=r"D:\Anubhav\softwares\client software\Data\project_oil_sample\pickle_data",
    pipe_number=7
)