#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings, re
from glob import glob
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)


def run_heatmap_plotly(
    *,
    # --- inputs ---
    pkl_path: Optional[Union[str, Path]] = None,        # path to .pkl OR pass df directly
    df_in: Optional[pd.DataFrame] = None,
    pipe_number: Optional[Union[str, int]] = None,      # if None and pkl_path given -> stem
    # --- columns/parameters ---
    roll_col: str = "ROLL",
    oddo_col: str = "ODDO1",
    initial_read: float = 0.0,
    upper_sens_mul: float = 1.0,
    lower_sens_mul: float = 3.0,
    pipetally_path: Optional[Union[str, Path]] = None,  # optional direct path
    # --- rendering/output ---
    output_html: Optional[Union[str, Path]] = None,     # default: heatmap{pipe}.html in CWD
    colorscale: str = "jet",
    zmin: float = -3.0,
    zmax: float = 8.0,
    include_overlay: bool = True,
    # --- behavior ---
    plot_mode: str = "raw",   # "raw" -> raw % of mean; "processed" -> sigma zeroed %
) -> Dict[str, Union[str, Path, pd.DataFrame, go.Figure]]:
    """
    Run the standalone heatmap pipeline in one call.

    Returns dict with:
      {
        'pipe_number': str,
        'html_path': Path,
        'fig': plotly.graph_objects.Figure,
        'df_raw_percent': DataFrame (% deviation raw),
        'df_processed_percent': DataFrame (% deviation processed),
        'x_vals': Series, 'y_bands': list[str], 'map_ori_sens': DataFrame
      }
    """

    # ----------------- helpers (kept local to avoid leaking names) -----------------

    def _find_pipe_tally_file_local(pn: Union[str, int]) -> Optional[str]:
        pn = str(pn)
        patterns = [
            f"**/*PipeTally*{pn}*.xlsx", f"**/*Pipe_Tally*{pn}*.xlsx",
            f"**/*PipeTally*{pn}*.csv",  f"**/*Pipe_Tally*{pn}*.csv",
            f"**/{pn}*PipeTally*.xlsx",  f"**/{pn}*PipeTally*.csv",
        ]
        for pat in patterns:
            hits = glob(pat, recursive=True)
            if hits:
                hits.sort(key=len)
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
        h, m, s = [int(x) for x in str(t).split(":")]
        return (h % 12) * 3600 + m * 60 + s

    def _nearest_band_label(seconds: int, band_labels: List[str]) -> str:
        band_labels_str = [str(x) for x in band_labels]
        band_secs = np.array([_hhmmss_to_seconds(lbl) for lbl in band_labels_str], dtype=int)
        idx = int(np.argmin(np.abs(band_secs - seconds)))
        return band_labels_str[idx]

    def _create_time_dict():
        # 12h dial, 7.5-minute bands
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
        start_time = '00:00:00'; end_time = '00:07:29'
        t = datetime.strptime(time_str, '%H:%M:%S')
        return datetime.strptime(start_time, '%H:%M:%S') <= t <= datetime.strptime(end_time, '%H:%M:%S')

    def _load_overlay_points_for_pipe_local(pn, y_band_labels, path_override=None, debug_prefix="OVERLAY DEBUG"):
        path = path_override if path_override else _find_pipe_tally_file_local(pn)
        if not path:
            print(f"{debug_prefix}: pipe {pn}: no PipeTally file found.")
            return None
        try:
            dfp = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
        except Exception as e:
            print(f"{debug_prefix}: pipe {pn}: failed reading '{path}': {e}")
            return None

        x_col    = _pick_col(dfp, ["Abs. Distance (m)", "Absolute Distance"], ["abs", "distance"]) or _pick_col(dfp, [], ["distance"])
        ori_col  = _pick_col(dfp, ["Orientation o' clock", "Orientation", "Ori"], ["ori"]) or _pick_col(dfp, [], ["orient"])
        feat_col = _pick_col(dfp, ["Feature Type", "Feature"], ["feature", "type"])
        pipe_col = _pick_col(dfp, ["Pipe Number", "Pipe"], ["pipe", "number"])
        sno_col  = _pick_col(dfp, ["s_no", "S_No", "Serial Number", "SNo"], ["s_no", "sno", "serial"])

        print(f"{debug_prefix}: pipe {pn}: file='{path}' cols: x='{x_col}', ori='{ori_col}', feature='{feat_col}', pipe='{pipe_col}', s_no='{sno_col}'")

        if x_col is None or ori_col is None:
            print(f"{debug_prefix}: pipe {pn}: missing required columns; no overlay.")
            return None

        if pipe_col is not None:
            mask = dfp[pipe_col].astype(str).str.contains(str(pn), na=False)
            if mask.any():
                dfp = dfp[mask]

        if feat_col is not None:
            ml_mask = dfp[feat_col].astype(str).str.contains("metal", case=False, na=False)
            if ml_mask.any():
                dfp = dfp[ml_mask]

        xs, ys, labels = [], [], []
        skipped_no_x, skipped_no_ori = 0, 0
        for _, row in dfp.iterrows():
            x = pd.to_numeric(row.get(x_col), errors="coerce")
            if pd.isna(x):
                skipped_no_x += 1; continue
            ori_sec = _parse_ori_to_seconds(row.get(ori_col))
            if ori_sec is None:
                skipped_no_ori += 1; continue
            y = _nearest_band_label(ori_sec, list(y_band_labels))
            if sno_col is not None:
                lbl = row.get(sno_col)
                if not pd.isna(lbl) and lbl is not None and str(lbl).strip() != "":
                    lbl = str(lbl)
                else:
                    lbl = row.get("Defect_id")
                    lbl = str(lbl) if (not pd.isna(lbl) and lbl is not None and str(lbl).strip() != "") else str(len(labels) + 1)
            else:
                lbl = row.get("Defect_id")
                lbl = str(lbl) if (not pd.isna(lbl) and lbl is not None and str(lbl).strip() != "") else str(len(labels) + 1)
            xs.append(float(x)); ys.append(str(y)); labels.append(lbl)

        print(f"{debug_prefix}: pipe {pn}: plotted={len(xs)}, skipped_no_x={skipped_no_x}, skipped_no_ori={skipped_no_ori}")
        if not xs:
            return None
        return xs, ys, labels

    # ----------------- load df & resolve pipe_number -----------------
    if df_in is None:
        if pkl_path is None:
            raise ValueError("Provide either df_in or pkl_path.")
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"{pkl_path}")
        df_in = pd.read_pickle(pkl_path)
        if not isinstance(df_in, pd.DataFrame):
            df_in = pd.DataFrame(df_in)
        if pipe_number is None:
            pipe_number = pkl_path.stem
    else:
        if pipe_number is None:
            pipe_number = "NA"

    # ----------------- pre-process -----------------
    # 1) sensors present
    expected = [f"F{i}H{j}" for i in range(1, 25) for j in range(1, 5)]
    sens_cols = [c for c in expected if c in df_in.columns]
    if not sens_cols:
        raise ValueError("No F*H* sensor columns found in the input DataFrame.")

    df_sens = pd.DataFrame(df_in, columns=sens_cols).copy()
    df_sens_raw = df_sens.copy(deep=True)

    # numeric + fill
    for col in sens_cols:
        df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    df_sens = df_sens.fillna(method='ffill').fillna(0.0)

    # percent-of-mean (raw)
    Mean1_raw = df_sens_raw.mean()
    df_raw_percent = ((df_sens_raw - Mean1_raw) / (Mean1_raw + 1e-8)) * 100

    # percent-of-mean (processed baseline)
    Mean1 = df_sens.mean()
    df_processed_percent = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100

    # sigma zeroing for processed
    sens_std = df_sens.std(axis=0, skipna=True)
    upper = Mean1 + upper_sens_mul * sens_std
    lower = Mean1 - lower_sens_mul * sens_std
    for col in sens_cols:
        mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
        df_processed_percent.loc[mask, col] = 0

    # ----------------- roll → time bands mapping -----------------
    if roll_col not in df_in.columns:
        raise KeyError(f"'{roll_col}' column not found in input DataFrame.")

    roll = pd.to_numeric(df_in[roll_col], errors='coerce').fillna(0.0) - float(initial_read)

    # build clock matrix
    d = [{"Roll_Sensor_0": pos} for pos in roll]
    upd_d = []
    for e in d:
        nd = {**e}
        for i in range(1, 96):
            nd[f"Roll_Sensor_{i}"] = e['Roll_Sensor_0'] + (3.75 * i)
        upd_d.append(nd)
    clock_df = pd.DataFrame.from_dict(upd_d).applymap(_degrees_to_hhmmss)

    # normalize time strings
    tmp = clock_df.apply(pd.to_datetime, format='%H:%M:%S')
    tmp = tmp.applymap(lambda x: x.strftime('%H:%M:%S'))
    tmp = tmp.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
    tmp = tmp.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
    tmp = tmp.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

    # choose plot matrix
    df_plot_base = df_processed_percent if plot_mode.lower().startswith("proc") else df_raw_percent
    # align time-matrix columns to sensor names
    tmp = tmp.rename(columns=dict(zip(tmp.columns, df_plot_base.columns)))

    time_dict = _create_time_dict(); bands = list(time_dict.keys())
    for _, row in tmp.iterrows():
        row_list = list(row); row_dict = dict(row); keys = list(row_dict.keys())
        ind = 0
        for _, dval in row_dict.items():
            if _check_time_range(dval):
                ind = row_list.index(dval); break
        c = 0; curr = ind
        while True:
            ck = keys[curr]
            time_dict[bands[c]].append((curr, ck, row_dict[ck]))
            c += 1
            curr = (curr + 1) % len(keys)
            if curr == ind: break

    map_ori_sens = pd.DataFrame(time_dict)
    val_ori_sens = map_ori_sens.applymap(lambda cell: cell[1])

    # rearrange into banded matrix
    test_val = val_ori_sens.copy()
    for r, e in val_ori_sens.iterrows():
        c = 0
        for _, sensor_col in e.items():
            test_val.iloc[r, c] = df_plot_base.at[r, sensor_col]
            c += 1

    # x-axis
    if oddo_col in df_in.columns:
        x_vals = (pd.to_numeric(df_in[oddo_col], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m)'
    else:
        x_vals = pd.Series(np.arange(len(df_in)))
        x_label = 'Index'

    y_bands = [str(c) for c in test_val.columns]

    # ----------------- render -----------------
    heatmap_data = test_val.T.copy()

    # ensure numeric, finite
    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_vals.round(2),
        y=y_bands,
        colorscale=colorscale,
        zmin=float(zmin),
        zmax=float(zmax),
        colorbar=dict(title="Sensor Value (%)"),
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br><b>%{y}</b><br><b>Value: %{z:.2f}%</b><extra></extra>'
    ))

    # overlays
    overlay_added = False
    if include_overlay:
        pts = _load_overlay_points_for_pipe_local(pipe_number, y_bands, str(pipetally_path) if pipetally_path else None)
        if pts is not None:
            xs, ys, labels = pts
            for x, y_band, label in zip(xs, ys, labels):
                try:
                    if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                        y_idx = y_bands.index(y_band)
                        fig.add_shape(
                            type="rect",
                            x0=x - 0.05, y0=y_idx - 0.35,
                            x1=x + 0.05, y1=y_idx + 0.35,
                            line=dict(color="black", width=2),
                            fillcolor="rgba(255,0,0,0.6)"
                        )
                        fig.add_annotation(
                            x=x, y=y_idx, text=str(label), showarrow=False,
                            font=dict(color="white", size=8, family="Arial Black"),
                            bgcolor="red", bordercolor="black", borderwidth=1
                        )
                        overlay_added = True
                except Exception:
                    # keep going; don't let a bad point kill the render
                    pass

    fig.update_layout(
        title=f"Interactive Plotly Heatmap — Pipe {pipe_number}",
        xaxis_title=x_label,
        yaxis_title="Orientation (12h bands)",
        width=1400,
        height=700,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )

    # save html
    if output_html is None:
        output_html = Path(f"heatmap{pipe_number}.html")
    else:
        output_html = Path(output_html)
        if output_html.is_dir():
            output_html = output_html / f"heatmap{pipe_number}.html"

    fig.write_html(
        str(output_html),
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

    print(f"Saved interactive heatmap: {output_html.resolve()}  | overlays: {'Yes' if overlay_added else 'None'}")

    return {
        "pipe_number": str(pipe_number),
        "html_path": output_html,
        "fig": fig,
        "df_raw_percent": df_raw_percent,
        "df_processed_percent": df_processed_percent,
        "x_vals": x_vals,
        "y_bands": y_bands,
        "map_ori_sens": map_ori_sens
    }
