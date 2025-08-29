import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

# case-insensitive F#P# like F1P1, F12P3
FP_PATTERN = re.compile(r'^F\d{1,2}P\d{1,2}$', re.IGNORECASE)


def render_fp_linechart(
    pkl_path: str,
    x_pref: str = "auto",      # auto -> ODDO2 else ODDO1 else index
    offset_step: float = 0.10,
    dtick: int = 1000
):
    df = pd.read_pickle(pkl_path)
    pipe_number = Path(pkl_path).stem

    # 1) collect F*P* columns
    candidates = [c for c in df.columns if isinstance(c, str) and FP_PATTERN.match(c.strip())]
    if not candidates:
        print(f"[pipe {pipe_number}] No F#P# columns found. Nothing to render.")
        return

    # 2) ensure numeric (coerce where possible)
    res1 = []
    for c in candidates:
        if not is_numeric_dtype(df[c]):
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().any():
                df[c] = s
        if is_numeric_dtype(df[c]):
            res1.append(c)
    if not res1:
        print(f"[pipe {pipe_number}] No numeric F#P# columns after coercion.")
        return

    # 3) forward-fill (values and any x-axis cols)
    df1 = df.fillna(method='ffill')

    # 4) choose x-axis
    x_label = "Index"
    if x_pref.lower() == "oddo1" or (x_pref == "auto" and "ODDO1" in df1.columns):
        x_vals = (pd.to_numeric(df1["ODDO1"], errors="coerce") / 1000.0).round(3)
        x_label = "Abs. Distance (m) — ODDO1"
    else:
        x_vals = df1.index
        x_label = "Index"

    # 5) MinMax scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df1[res1].to_numpy())
    df1.loc[:, res1] = scaled

    # 6) figure
    fig = go.Figure()
    for i, col in enumerate(res1):
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=df1[col] + i * offset_step,
            name=col,
            mode='lines',
            line=dict(width=1),
            hoverinfo='x+y+name',
            showlegend=True
        ))

    # 7) styling + axis titles + gridlines
    fig.update_layout(
        title="Proximity Sensor Line Chart",
        width=1600,
        height=650,
        margin=dict(l=140, b=80),
        paper_bgcolor="#ffffff",
        plot_bgcolor='rgb(255, 255, 255)',
        title_x=0.5,
        font={"family": "courier"},
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis_title=x_label,
        yaxis_title="Scaled Proximity Sensor (0–1, offset)",
    )

    # Keep titles close to axes
    fig.update_xaxes(title_standoff=8, automargin=True)
    fig.update_yaxes(title_standoff=10, automargin=True)

    # Light major gridlines
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)", gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", gridwidth=1, zeroline=False)

    # Optional: minor gridlines (uncomment if your Plotly version supports it)
    fig.update_xaxes(minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=0.5))
    fig.update_yaxes(minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", gridwidth=0.5))

    # 8) render only
    fig.show()


def main():
    ap = argparse.ArgumentParser(description="Render F#P# linechart from a single .pkl (no saving).")
    ap.add_argument("pkl", help="Path to the input .pkl file")
    ap.add_argument("--xpref", choices=["auto", "ODDO2", "ODDO1", "index"], default="auto",
                    help="X-axis preference (default: auto)")
    ap.add_argument("--offset", type=float, default=0.10,
                    help="Vertical offset per series (default: 0.10)")
    ap.add_argument("--dtick", type=int, default=1000,
                    help="X-axis dtick (default: 1000)")
    args = ap.parse_args()

    render_fp_linechart(args.pkl, x_pref=args.xpref, offset_step=args.offset, dtick=args.dtick)


if __name__ == "__main__":
    main()
