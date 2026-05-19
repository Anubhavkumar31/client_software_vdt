import os
import warnings

import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go



DEFECTS_CSV = r"D:\Anubhav\runid_data\12inch\12_inch_runid_27\defects_clock_hm.csv"

INITIAL_READ = 0.0
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3





_DEFECTS_DF_CACHE = None


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


# ── Colour by depth % ─────────────────────────────────────────────────────────

def _defect_color(depth: float):
    if depth >= 80:
        return 'rgba(200, 0, 0, 0.2)', 'red'
    elif depth >= 50:
        return 'rgba(255, 140, 0, 0.2)', 'darkorange'
    else:
        return 'rgba(255, 200, 0, 0.2)', 'goldenrod'

# ── Core overlay drawing ──────────────────────────────────────────────────────

def _draw_bbox_overlays(fig, pipe_number: int, defects_csv_path: str,
                        y_bands: list, x_vals: pd.Series) -> bool:
    """
    x_vals  — the heatmap x series (datafile['index'])
    y_bands — clock labels ['00:00','00:05',...] from test_val.columns
    start_index/end_index in CSV match datafile['index'] values directly.
    start_sensor/end_sensor are 0-based positions into y_bands.
    """
    defects_df = _load_defects_df(defects_csv_path)
    if defects_df.empty:
        return False

    pipe_defects = defects_df[
        defects_df['pipe_id'].astype(str) == str(pipe_number)
    ].reset_index(drop=True)

    if pipe_defects.empty:
        print(f"[bbox] No defects for pipe {pipe_number}")
        return False

    n_bands       = len(y_bands)
    idx_min       = float(x_vals.min())
    idx_max       = float(x_vals.max())
    overlay_added = False
    drawn_count   = 0

    for defect_counter, row in pipe_defects.iterrows():
        label_num = defect_counter + 1

        start_reading = float(row.get('start_index', 0) or 0)
        end_reading   = float(row.get('end_index',   0) or 0)
        if start_reading > end_reading:
            start_reading, end_reading = end_reading, start_reading

        if end_reading < idx_min or start_reading > idx_max:
            print(f"  [Defect #{label_num}] index {start_reading:.0f}-{end_reading:.0f} "
                  f"outside pipe range {idx_min:.0f}-{idx_max:.0f} — skipped")
            continue

        start_sensor = int(float(row.get('start_sensor', 0) or 0))
        end_sensor   = int(float(row.get('end_sensor',   0) or 0))
        if start_sensor > end_sensor:
            start_sensor, end_sensor = end_sensor, start_sensor
        start_sensor = max(0, min(start_sensor, n_bands - 1))
        end_sensor   = max(0, min(end_sensor,   n_bands - 1))

        depth = float(row.get('depth_new', 0) or 0)
        fill_color, line_color = _defect_color(depth)

        # exactly like figx112
        fig.add_shape(
            type='rect',
            # x0=start_reading - 0.5,
            # x1=end_reading   + 0.5,
            x0=x_vals[int(start_reading)],
            x1=x_vals[int(end_reading)],
            y0=start_sensor  - 0.5,
            y1=end_sensor    + 0.5,
            line=dict(color='black', width=2),
            fillcolor=fill_color,
            layer='above'
        )

        fig.add_annotation(
            x=(start_reading + end_reading) / 2,
            y=start_sensor - 1,
            text=str(label_num),
            showarrow=False,
            font=dict(color=line_color, size=10),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )

        orient      = row.get('orientation',              'N/A')
        dim_cls     = row.get('dimension_classification', 'N/A')
        defect_type = row.get('defect_type',              'N/A')
        length_mm   = row.get('length',                   'N/A')
        width_mm    = row.get('width_final', row.get('Width', 'N/A'))
        abs_dist    = row.get('absolute_distance',        'N/A')

        # fig.add_trace(go.Scatter(
        #     # x=[(start_reading + end_reading) / 2],
        #     x=[x_vals[int((start_reading + end_reading) / 2)]],
        #     y=[(start_sensor  + end_sensor)  / 2],
        #     mode='markers',
        #     marker=dict(size=12, color=line_color, opacity=0.0),
        #     hovertemplate=(
        #         f"<b>Defect #{label_num}</b><br>"
        #         f"Pipe: {pipe_number}<br>"
        #         f"Depth: {depth:.0f}%<br>"
        #         f"Orientation: {orient}<br>"
        #         f"Classification: {dim_cls}<br>"
        #         f"Type: {defect_type}<br>"
        #         f"Length: {length_mm} mm<br>"
        #         f"Width: {width_mm} mm<br>"
        #         f"Abs. Distance: {abs_dist} m<extra></extra>"
        #     ),
        #     showlegend=False,
        # ))

        overlay_added = True
        drawn_count  += 1
        clock_s = y_bands[start_sensor] if start_sensor < n_bands else '?'
        clock_e = y_bands[end_sensor]   if end_sensor   < n_bands else '?'
        print(f"  [Defect #{label_num}] x={start_reading:.0f}-{end_reading:.0f}  "
              f"sensor={start_sensor}-{end_sensor} ({clock_s}→{clock_e})")

    print(f"[bbox] Pipe {pipe_number}: drew {drawn_count} box(es)")
    return overlay_added

def save_interactive_heatmap_v2(
    df_new_tab9, datafile, test_val, map_ori_sens,
    folder_path, pipe_number, df_new_tab10
):
    df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(
        df_new_tab10, datafile, test_val, map_ori_sens
    )

    # x-axis: datafile['index'] — same values start_index/end_index reference
    # x_vals   = datafile['index']
    x_vals = pd.Series(np.arange(len(datafile)))
    abs_dist = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000).values

    print("df_raw_plot columns:", df_raw_plot.columns.tolist()[:10])
    print("df_raw_plot columns last 5:", df_raw_plot.columns.tolist()[-5:])
    print("test_val columns last 5:", test_val.columns.tolist()[-5:])
    print("df_raw_plot shape:", df_raw_plot.shape)
    print("df_plot_rearranged shape:", df_plot_rearranged.shape)

    # y_bands: clock labels, sensor 0..143 maps directly as y position
    y_bands      = [str(c) for c in test_val.columns]
    heatmap_data = df_raw_plot.T

    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    heatmap_data = heatmap_data.astype('float32').round(3)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_vals,
        y=y_bands,
        colorscale='jet',
        zmin=-3, zmax=8,
        showscale=False,
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br><b>%{y}</b><br><b>Value: %{z:.2f}%</b><extra></extra>'
    ))
    print("=== BOX DEBUG ===")
    print("x_vals range:", datafile['index'].min(), "-", datafile['index'].max())
    df_check = _load_defects_df(DEFECTS_CSV)
    pipe_check = df_check[df_check['pipe_id'].astype(str) == str(pipe_number)]
    print(f"Defects found for pipe {pipe_number}:", len(pipe_check))
    print(pipe_check[['start_index', 'end_index', 'start_sensor', 'end_sensor']].to_string())

    _draw_bbox_overlays(fig, pipe_number, DEFECTS_CSV, y_bands, x_vals)

    # tick labels show metres instead of raw index
    n_ticks     = 20
    tick_pos    = np.round(np.linspace(0, len(x_vals) - 1, n_ticks)).astype(int)
    tick_vals   = x_vals.values[tick_pos]
    tick_labels = [f"{abs_dist[i]:.1f}m" for i in tick_pos]

    fig.update_layout(
        title=dict(
            text=f'Hall-Sensor Heatmap — Joint Number {pipe_number}',
            x=0.5, xanchor='center',
            font=dict(size=18, family='Arial Black')
        ),
        xaxis=dict(
            title='Absolute Distance (m) — Hall Sensor Heatmap',
            tickmode='array', tickvals=tick_vals.tolist(), ticktext=tick_labels,
            showgrid=True, gridwidth=1, gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            automargin=False
        ),
        yaxis_title=None, width=1500, height=500, font=dict(size=12),
    )
    fig.update_yaxes(
        autorange='reversed',
        tickmode='array',
        tickvals=y_bands[::12],
        ticktext=y_bands[::12],
        type='category',

        range=[
            y_bands[-1],
            y_bands[0]
        ]
    )

    out_path = f'{folder_path}/hallsensor_heatmap_v2_{pipe_number}.html'
    write_plotly_html(fig, out_path)
    print(f'Saved hall heatmap v2: {out_path}')
