import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from datetime import datetime, timedelta

from Data_Gen.html_filter import (
    _map_sensor_values_fast,
    _remap_sensor_timeline_fast,
    _build_clock_data_fast,
    count_pattern_minute_degree,
    _load_overlay_points_for_pipe,
    write_plotly_html,
)


# ─────────────────────────────────────────────────────────────
#  PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────

def generate_heatmap_from_pkl(pkl_path, folder_path, pipe_number):
    """
    Fully automatic hall/proximity heatmap generation pipeline.

    Parameters
    ----------
    pkl_path     : path to raw PKL sensor file
    folder_path  : output directory
    pipe_number  : pipe/joint identifier
    """

    # ── Load ──────────────────────────────────────────────────
    datafile = pd.read_pickle(pkl_path)
    print(f"Loaded PKL : {pkl_path}")
    print(f"Data shape : {datafile.shape}")

    # ── Detect sensor layout ──────────────────────────────────
    hall, prox = count_pattern_minute_degree(pkl_path)

    column_names_prox = prox["columns"]
    total_sensors     = prox["count"]
    minute_sensors    = prox["minute"]
    degree_sensors    = prox["degree"]

    print(f"Detected sensors : {total_sensors}")
    print(f"Minute spacing   : {minute_sensors}")
    print(f"Degree spacing   : {degree_sensors}")

    # ── Sensor dataframe ──────────────────────────────────────
    df_new_tab9  = pd.DataFrame(datafile, columns=column_names_prox)
    df_new_tab10 = df_new_tab9.copy()
    print(f"df_new_tab9 shape : {df_new_tab9.shape}")

    # ── Clock range labels ────────────────────────────────────
    time_list = [timedelta(minutes=i * minute_sensors) for i in range(total_sensors)]
    rang = [(datetime.min + t).strftime('%H:%M') for t in time_list]
    print(f"Clock labels (first 5): {rang[:5]} …")

    # ── Roll array ────────────────────────────────────────────
    roll = (datafile['ROLL'] - 0).to_numpy(dtype=np.float64)
    print(f"Roll shape : {roll.shape}")

    # ── Orientation matrix ────────────────────────────────────
    offsets = np.arange(total_sensors, dtype=np.float64) * degree_sensors
    ori_mat = roll[:, None] + offsets[None, :]
    print(f"ori_mat shape : {ori_mat.shape}")

    oriData = pd.DataFrame(
        ori_mat,
        columns=[f'Roll_Sensor_{i}' for i in range(total_sensors)]
    )

    # ── Clock data ────────────────────────────────────────────
    test_clockData = _build_clock_data_fast(oriData)
    test_clockData.columns = df_new_tab9.columns
    print(f"test_clockData shape : {test_clockData.shape}")

    # ── Remap sensor timeline ─────────────────────────────────
    map_ori_sens, val_ori_sens_labels = _remap_sensor_timeline_fast(
        test_clockData, rang, total_sensors
    )
    print("Sensor remapping complete.")

    # ── Heatmap layout ────────────────────────────────────────
    test_val = _map_sensor_values_fast(val_ori_sens_labels, df_new_tab9)
    print(f"test_val shape : {test_val.shape}")

    # ── Generate & save ───────────────────────────────────────
    save_interactive_heatmap(
        df_new_tab9, datafile, test_val,
        map_ori_sens, folder_path, pipe_number, df_new_tab10
    )
    print(f"Heatmap generation completed for pipe {pipe_number}")


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING  (median / IQR normalisation)
# ─────────────────────────────────────────────────────────────

def pre_process_for_interactive_heatmap(df_in, datafile, test_val, map_ori_sens):
    from scipy.ndimage import uniform_filter1d

    sens_cols = df_in.columns.tolist()
    if not sens_cols:
        raise ValueError("No sensor columns found.")

    df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
    for col in sens_cols:
        df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    df_sens = df_sens.ffill().fillna(0.0)

    # ── Median / IQR normalisation ────────────────────────────
    sensor_median = df_sens.median()
    sensor_iqr    = (df_sens.quantile(0.75) - df_sens.quantile(0.25)) + 1e-6
    df_norm = (df_sens - sensor_median) / sensor_iqr

    # ── KEY FIX: zero out bad rows per sensor BEFORE smoothing ─
    # F36H2 and ~19 others have dropout spikes (raw value far below
    # their own median). After IQR norm these become -40, -20 etc.
    # Zeroing them (= "sensor reading was normal here") is correct
    # because a dropout is not a pipe anomaly.
    # Threshold: any single sample beyond ±4 IQR is a sensor fault.
    FAULT_THRESHOLD = 4.0
    df_norm = df_norm.where(df_norm.abs() <= FAULT_THRESHOLD, other=0.0)

    # ── Spatial smoothing ─────────────────────────────────────
    df_norm = pd.DataFrame(
        uniform_filter1d(df_norm.values, size=100, axis=0),
        columns=df_norm.columns,
        index=df_norm.index,
    )

    # ── Final clip ────────────────────────────────────────────
    df_norm = df_norm.clip(-2.0, 4.0)

    # ── Rearrange into clock bands ────────────────────────────
    n_rows   = len(test_val)
    n_bands  = len(test_val.columns)
    out_mat  = np.zeros((n_rows, n_bands), dtype=np.float64)
    map_arr  = map_ori_sens.to_numpy()
    col_idx  = {col: i for i, col in enumerate(df_norm.columns)}
    norm_data = df_norm.to_numpy(dtype=np.float64)

    for c in range(n_bands):
        col_tuples   = map_arr[:, c]
        sensor_names = np.array([
            t[1] if isinstance(t, tuple) and len(t) >= 2 else None
            for t in col_tuples
        ])
        for sensor in np.unique(sensor_names):
            if sensor is None or sensor not in col_idx:
                continue
            rows = sensor_names == sensor
            out_mat[rows, c] = norm_data[rows, col_idx[sensor]]

    df_plot_rearranged = pd.DataFrame(
        out_mat, columns=test_val.columns, index=test_val.index
    )
    return df_plot_rearranged, df_norm


# ─────────────────────────────────────────────────────────────
#  SAVE HEATMAP
# ─────────────────────────────────────────────────────────────

def save_interactive_heatmap(
    df_new_tab9, datafile, test_val,
    map_ori_sens, folder_path, pipe_number, df_new_tab10
):
    df_plot_rearranged, _ = pre_process_for_interactive_heatmap(
        df_new_tab10, datafile, test_val, map_ori_sens
    )

    # ── X axis ────────────────────────────────────────────────
    if 'ODDO1' in datafile.columns:
        x_vals  = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m) — Hall Sensor Heatmap'
    else:
        x_vals  = pd.Series(np.arange(len(datafile)))
        x_label = 'Index'

    y_bands = [str(c) for c in test_val.columns]

    # ── Heatmap matrix ────────────────────────────────────────
    heatmap_data = df_plot_rearranged.T.copy()
    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    heatmap_data = heatmap_data.astype("float32").round(3)

    print(f"DEBUG heatmap_data : shape={heatmap_data.shape}  "
          f"nonzero={np.count_nonzero(heatmap_data.values)}  "
          f"min={heatmap_data.values.min():.2f}  "
          f"max={heatmap_data.values.max():.2f}  "
          f"mean={heatmap_data.values.mean():.4f}")

    # ── Colour scale ──────────────────────────────────────────
    #    Data is IQR-normalised.  Normal baseline ≈ 0 (white).
    #    Anomaly is a POSITIVE spike (+2 … +7).
    #    zmin=-2  keeps mild negative deviations visible in blue.
    #    zmax=+4  saturates the big anomaly at full red — still
    #             clearly distinguishable from moderate deviations.
    ZMIN = -2.0
    ZMAX =  4.0

    print(f"Colour scale : zmin={ZMIN}  zmax={ZMAX}  (IQR units)")

    # ── Figure ────────────────────────────────────────────────
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_vals.round(2),
        y=y_bands,
        colorscale='RdBu_r',   # blue=below baseline, red=above
        zmin=ZMIN,
        zmax=ZMAX,
        zmid=0,
        showscale=True,
        hoverongaps=False,
        colorbar=dict(title="IQR units"),
        hovertemplate=(
            '<b>Distance: %{x} m</b><br>'
            '<b>Clock:     %{y}</b><br>'
            '<b>Value:     %{z:.2f} IQR</b>'
            '<extra></extra>'
        ),
    ))

    # ── Overlays ──────────────────────────────────────────────
    overlay_added = False
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
    if pts is not None:
        xs, ys, labels = pts
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                fig.add_shape(
                    type="rect",
                    x0=x - 0.05, y0=y_idx - 0.35,
                    x1=x + 0.05, y1=y_idx + 0.35,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(255,0,0,0.6)",
                )
                fig.add_annotation(
                    x=x, y=y_idx, text=label, showarrow=False,
                    font=dict(color="white", size=8, family="Arial Black"),
                    bgcolor="red", bordercolor="black", borderwidth=1,
                )
                overlay_added = True

    # ── Layout ────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Hall-Sensor Heatmap — Joint {pipe_number}",
            x=0.5, xanchor="center",
            font=dict(size=18, family="Arial Black"),
        ),
        xaxis_title=x_label,
        yaxis_title="Clock position",
        width=1500, height=500,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    )
    fig.update_yaxes(autorange="reversed")

    out_path = f'{folder_path}/hallsensor_heatmap{pipe_number}.html'
    write_plotly_html(fig, out_path)
    print(f"Saved : {out_path}")
    print(f"Overlays : {'Yes' if overlay_added else 'None found'}")


# ─────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────

generate_heatmap_from_pkl(
    pkl_path=r"D:\Anubhav\softwares\client software\Data\project_oil_sample\pickle_data\150.pkl",
    folder_path=r"D:\Anubhav\softwares\client software\Data\project_oil_sample\pickle_data",
    pipe_number=150,
)