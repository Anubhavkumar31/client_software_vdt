# backend/plotting_utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import plotly.graph_objects as go
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from pages.backend.utils import get_app_dir



def plot_erf(df, view="Both", return_fig=False):
    df = df[df['Surface Location'].isin(['Internal', 'External'])].copy()

    df['Abs. Distance (m)'] = df['Abs. Distance (m)'].round(1)
    df = df.groupby(['Abs. Distance (m)', 'Surface Location'], as_index=False).agg({'ERF (ASME B31G)': 'max'})
    df.sort_values(by='Abs. Distance (m)', inplace=True)

    y_max = 1.2
    fig = go.Figure()

    if view in ["Internal", "Both"]:
        internal_df = df[df['Surface Location'] == 'Internal']
        fig.add_trace(go.Scatter(
            x=internal_df['Abs. Distance (m)'],
            y=internal_df['ERF (ASME B31G)'],
            name="Internal",
            mode="markers",
            marker=dict(symbol='triangle-up', color='steelblue', size=8),
            showlegend=True
        ))

    if view in ["External", "Both"]:
        external_df = df[df['Surface Location'] == 'External']
        fig.add_trace(go.Scatter(
            x=external_df['Abs. Distance (m)'],
            y=external_df['ERF (ASME B31G)'],
            name="External",
            mode="markers",
            marker=dict(symbol='triangle-up', color='orangered', size=8),
            showlegend=True
        ))

    fig.add_shape(
        type="line",
        x0=0,
        y0=1,
        x1=df['Abs. Distance (m)'].max(),
        y1=1,
        line=dict(color="green", width=2, dash="dash")
    )

    fig.update_layout(
        title="",
        xaxis=dict(title="Distance from Launcher (ODDO) in m", dtick=500, tickformat="~d"),
        yaxis=dict(title="ERF (ASME B31G)", range=[0, y_max], dtick=0.1, tickmode="linear"),
        height=700,
        width=1600,
        template="plotly_white"
    )

    # Save to backend/files/ERF_Abs.html
    html_path = Path(get_app_dir()) / "backend" / "files" / "ERF_Abs.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))

    return fig, str(html_path)



def plot_depth(df, view="Both", return_fig=False):
    df.columns = df.columns.str.strip()

    # Rename depth column
    df.rename(columns={col: "Depth % WT" for col in df.columns if "Depth" in col}, inplace=True)

    # Filter for Internal, External, or Both
    if view in ['Internal', 'External']:
        df = df[df['Surface Location'] == view]
    elif view == 'Both':
        df = df[df['Surface Location'].isin(['Internal', 'External'])]

    # Round and prepare distance
    df.loc[:, 'Abs. Distance (m)'] = df['Abs. Distance (m)'].round(1)


    # Keep max depth per location-distance pair
    df = df.groupby(['Abs. Distance (m)', 'Surface Location'], as_index=False).agg({'Depth % WT': 'max'})
    df.sort_values(by='Abs. Distance (m)', inplace=True)

    # Binning
    distance_edges = np.arange(0, df['Abs. Distance (m)'].max() + 1000, 1000)
    distance_bins = pd.cut(df['Abs. Distance (m)'], bins=distance_edges)

    # Only include 0–20, 20–40, 40–60, 60–80 depth ranges
    depth_edges = [0, 20, 40, 60, 80]
    depth_labels = ['0–20%', '20–40%', '40–60%', '60–80%']
    depth_bins = pd.cut(df['Depth % WT'], bins=depth_edges, labels=depth_labels)

    # Group and count
    grouped = df.groupby([distance_bins, depth_bins]).size().unstack(fill_value=0)

    # X and Y labels
    x_labels = grouped.index.categories
    y_labels = grouped.columns

    # Meshgrid for bar positions
    xpos, ypos = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dz = grouped.values.ravel()

    # Mask out zero bars
    mask = dz > 0
    xpos, ypos, zpos, dz = xpos[mask], ypos[mask], zpos[mask], dz[mask]

    dx = dy = 0.4

    # Color map for depth bins
    depth_colors = {
        '0–20%': 'steelblue',
        '20–40%': 'darkorange',
        '40–60%': 'seagreen',
        '60–80%': 'yellow'
    }
    colors = [depth_colors[y_labels[y]] for y in ypos]

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=colors)

    ax.set_xlabel('\nLauncher Distance (m)', labelpad=20)
    ax.set_ylabel('\nDepth % Bins', labelpad=20)
    ax.set_zlabel('\nCount', labelpad=10)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels([f"{int(i.left)}" for i in x_labels], rotation=30, ha='right', fontsize=10)

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)

    # Hide grid lines
    ax.xaxis._axinfo["grid"].update({"linewidth": 0})
    ax.yaxis._axinfo["grid"].update({"linewidth": 0})
    ax.zaxis._axinfo["grid"].update({"linewidth": 0})

    ax.view_init(elev=25, azim=-75)
    plt.title("3D Distribution of Depth % Across Launcher Distance", pad=30, fontsize=14)
    plt.tight_layout()

    if return_fig:
        # Save to temp PNG file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "depth_plot.png")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)  # Close to free memory
        return fig, output_path
    else:
        plt.show()


def clock_to_degrees(clock_str):
    try:
        h, m, s = map(int, str(clock_str).split(":"))
        decimal = h + m / 60 + s / 3600
        return (decimal % 12) * 30
    except:
        return None

def degrees_to_clock(deg):
    total_seconds = deg / 30 * 3600
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def plot_orientation(df, view="Both", return_fig=False):
    import os
    import plotly.graph_objects as go

    # Clean column names thoroughly
    df.columns = df.columns.str.replace(r'[\u200b\xa0]', '', regex=True).str.strip()

    # Dynamically find the Orientation column
    orientation_col = [col for col in df.columns if "Orientation" in col and "clock" in col]
    if not orientation_col:
        raise ValueError("❌ Could not find Orientation O'clock column after cleaning!")

    df['Angle (deg)'] = df[orientation_col[0]].apply(clock_to_degrees)
    df.dropna(subset=['Angle (deg)', 'Abs. Distance (m)', 'Surface Location'], inplace=True)

    fig = go.Figure()

    if view in ["Internal", "Both"]:
        internal_df = df[df['Surface Location'] == 'Internal']
        fig.add_trace(go.Scatter(
            x=internal_df['Abs. Distance (m)'],
            y=internal_df['Angle (deg)'],
            mode="markers",
            name="Internal",
            marker=dict(color="steelblue", size=6, symbol="triangle-up"),
            hovertemplate="Internal<br>Distance: %{x} m<br>Orientation: %{customdata}",
            customdata=internal_df['Angle (deg)'].apply(degrees_to_clock),
            showlegend=True
        ))

    if view in ["External", "Both"]:
        external_df = df[df['Surface Location'] == 'External']
        fig.add_trace(go.Scatter(
            x=external_df['Abs. Distance (m)'],
            y=external_df['Angle (deg)'],
            mode="markers",
            name="External",
            marker=dict(color="orangered", size=6, symbol="triangle-up"),
            hovertemplate="External<br>Distance: %{x} m<br>Orientation: %{customdata}",
            customdata=external_df['Angle (deg)'].apply(degrees_to_clock),
            showlegend=True
        ))

    fig.update_layout(
        title="",
        xaxis=dict(title="Distance from Launcher (ODDO) in m", dtick=500, tickformat="~d"),
        yaxis=dict(
            title="Circumferential Orientation (o'clock)",
            tickvals=[i * 30 for i in range(13)],
            ticktext=[f"{i:02}:00:00" for i in range(13)]
        ),
        height=700,
        width=1600,
        template="plotly_white"
    )

    html_path = os.path.abspath("orientation_plot.html")
    fig.write_html(html_path)
    return fig, html_path
