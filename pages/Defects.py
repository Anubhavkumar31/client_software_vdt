# import pandas as pd
# import plotly.graph_objects as go
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon


# def draw_3d_bar(ax, x, height, width=0.4, depth=0.3, face_colors=('#5B9BD5', '#D6E3F3', '#2F528F')):
#     # Front face
#     front = [[x, 0], [x, height], [x + width, height], [x + width, 0]]
#     ax.add_patch(Polygon(front, closed=True, facecolor=face_colors[0]))

#     # Top face
#     top = [[x, height], [x + depth, height + depth], [x + width + depth, height + depth], [x + width, height]]
#     ax.add_patch(Polygon(top, closed=True, facecolor=face_colors[1]))

#     # Side face
#     side = [[x + width, 0], [x + width, height], [x + width + depth, height + depth], [x + width + depth, depth]]
#     ax.add_patch(Polygon(side, closed=True, facecolor=face_colors[2]))

# def plot_metal_loss(df, feature_type=None, dimension_class=None, return_fig=False):
#     df.columns = df.columns.str.strip()
#     print(f"[DEBUG] Initial dataframe shape: {df.shape}")

#     df = df[df['Feature Type'].str.strip().str.lower() == 'metal loss']
#     print(f"[DEBUG] After filtering Metal Loss: {df.shape}")

#     if feature_type:
#         if feature_type == "Corrosion":
#             df = df[df['Feature Identification'].str.contains('Corrosion', case=False, na=False)]
#         elif feature_type == "MFG":
#             df = df[df['Feature Identification'].str.contains('MFG', case=False, na=False)]

#     if dimension_class and dimension_class != "ALL":
#         df = df[df['Dimension Classification'].str.contains(dimension_class, case=False, na=False)]

#     if df.empty:
#         print("No matching defects found in the file.")
#         return None

#     bin_size = 500
#     max_distance = df['Abs. Distance (m)'].max()
#     bins = list(range(0, int(max_distance) + bin_size, bin_size))
#     df['Distance Bin'] = pd.cut(df['Abs. Distance (m)'], bins=bins, right=True)

#     bin_counts = df.groupby('Distance Bin', observed=False).size().reset_index(name='Metal Loss Count')
#     bin_counts['Bin Start'] = bin_counts['Distance Bin'].apply(lambda x: int(x.left))
#     bin_counts['Label'] = bin_counts['Bin Start'].astype(str)

#     # Plotting
#     fig, ax = plt.subplots(figsize=(20, 6))  # wider to fit more bars

#     values = bin_counts['Metal Loss Count'].tolist()
#     x_labels = bin_counts['Label'].tolist()

#     for i, val in enumerate(values):
#         draw_3d_bar(ax, i, val, face_colors=('#4F81BD', '#4F81BD', '#385D8A'))

#     ax.set_xticks(np.arange(len(x_labels)) + 0.3)
#     ax.set_xticklabels(x_labels)
#     ax.set_xlabel('Distance from Launcher (m)')

#     yaxis_title = "No. of"
#     if feature_type:
#         yaxis_title += f" {feature_type}"
#     if dimension_class:
#         yaxis_title += f" {dimension_class}"
#     yaxis_title += " Metal Losses"
#     ax.set_ylabel(yaxis_title)
#     ax.set_title('Distribution of Metal Loss Defects')
#     ax.set_xlim(-0.5, len(values))
#     ax.set_ylim(0, max(values) + 2)
#     ax.grid(False)

#     plt.tight_layout()

#     # Save as PNG
#     image_path = os.path.abspath('metal_loss_3d_look_plot.png')
#     fig.savefig(image_path)

#     if return_fig:
#         return fig, image_path
#     else:
#         return image_path








# # Updated Sensor Loss Plot function
# def plot_sensor_percentage(df, return_fig=False):
#     df.columns = df.columns.str.strip()

#     # Sort values based on distance for better visualization
#     df.sort_values(by='Abs. Distance (m)', inplace=True)

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df['Abs. Distance (m)'],
#         y=df['Sensor %'],
#         mode='lines',
#         name='Sensor %',
#         line=dict(color='green', width=2)
#     ))

#     fig.update_layout(
#         # title='Sensor % vs. Absolute Distance',
#         xaxis=dict(title='Absolute Distance (m)', gridcolor='lightgray', dtick=2000),
#         yaxis=dict(
#             title='Sensor %',
#             gridcolor='lightgray',
#             range=[0, 100],  # Fixed Y-axis range
#             dtick=20         # Y-axis ticks at 0, 20, 40, ..., 100
#         ),
#         height=700,
#         width=1600,
#         template='plotly_white'
#     )

#     html_path = os.path.abspath('sensor_percentage_plot.html')
#     fig.write_html(html_path)

#     if return_fig:
#         return fig, html_path
#     else:
#         return html_path

# def plot_temperature(df, return_fig=False):
#     df.columns = df.columns.str.strip()
#     df.sort_values(by='Abs. Distance (m)', inplace=True)
#     if 'Temperature (°C)' not in df.columns:
#         temperature_profile = np.linspace(55, 50, len(df))
#         noise=np.random.uniform(low=-0.5,high=0.5, size=len(df))

#         df['Temperature (°C)'] = temperature_profile + noise


#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df['Abs. Distance (m)'],
#         y=df['Temperature (°C)'],
#         mode='lines',
#         name='Temperature Profile',
#         line=dict(color='blue', width=2)
#     ))

#     fig.update_layout(
#         # title='Temperature Level Profile',
#         xaxis=dict(title='Absolute Distance (m)', gridcolor='lightgray', dtick=2000),
#         yaxis=dict(
#             title='Temperature (°C)',
#             gridcolor='lightgray',
#             range=[0, 100],
#             dtick=10
#         ),
#         height=700,
#         width=1600,
#         template='plotly_white'
#     )

#     html_path = os.path.abspath('temperature_plot.html')
#     fig.write_html(html_path)

#     if return_fig:
#         return fig, html_path
#     else:
#         return html_path

import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def draw_3d_bar(ax, x, height, width=0.4, depth=0.3, face_colors=('#5B9BD5', '#D6E3F3', '#2F528F')):
    # Front face
    front = [[x, 0], [x, height], [x + width, height], [x + width, 0]]
    ax.add_patch(Polygon(front, closed=True, facecolor=face_colors[0]))

    # Top face
    top = [[x, height], [x + depth, height + depth], [x + width + depth, height + depth], [x + width, height]]
    ax.add_patch(Polygon(top, closed=True, facecolor=face_colors[1]))

    # Side face
    side = [[x + width, 0], [x + width, height], [x + width + depth, height + depth], [x + width + depth, depth]]
    ax.add_patch(Polygon(side, closed=True, facecolor=face_colors[2]))

# def plot_metal_loss(df, feature_type=None, dimension_class=None, return_fig=False):
#     df.columns = df.columns.str.strip()
#     print(f"[DEBUG] Initial dataframe shape: {df.shape}")
#
#     df = df[df['Feature Type'].str.strip().str.lower() == 'metal loss']
#     print(f"[DEBUG] After filtering Metal Loss: {df.shape}")
#
#     if feature_type:
#         if feature_type == "Corrosion":
#             df = df[df['Feature Identification'].str.contains('Corrosion', case=False, na=False)]
#         elif feature_type == "MFG":
#             df = df[df['Feature Identification'].str.contains('MFG', case=False, na=False)]
#
#     # Normalize column names
#     df.columns = df.columns.str.strip().str.lower()
#
#     if dimension_class and dimension_class != "ALL":
#         if "dimensions classification" in df.columns:
#             df = df[df['dimensions classification'].str.contains(dimension_class, case=False, na=False)]
#         elif "dimension classification" in df.columns:
#             df = df[df['dimension classification'].str.contains(dimension_class, case=False, na=False)]
#     if df.empty:
#         print("No matching defects found in the file.")
#         return None, None
#
#     bin_size = 500
#     max_distance = df['Abs. Distance (m)'].max()
#     bins = list(range(0, int(max_distance) + bin_size, bin_size))
#     df['Distance Bin'] = pd.cut(df['Abs. Distance (m)'], bins=bins, right=True)
#
#     bin_counts = df.groupby('Distance Bin', observed=False).size().reset_index(name='Metal Loss Count')
#     bin_counts['Bin Start'] = bin_counts['Distance Bin'].apply(lambda x: int(x.left))
#     bin_counts['Label'] = bin_counts['Bin Start'].astype(str)
#
#     # Plotting
#     fig, ax = plt.subplots(figsize=(20, 6))  # wider to fit more bars
#
#     values = bin_counts['Metal Loss Count'].tolist()
#     x_labels = bin_counts['Label'].tolist()
#
#     for i, val in enumerate(values):
#         draw_3d_bar(ax, i, val, face_colors=('#4F81BD', '#4F81BD', '#385D8A'))
#
#     ax.set_xticks(np.arange(len(x_labels)) + 0.3)
#     ax.set_xticklabels(x_labels)
#     ax.set_xlabel('Distance from Launcher (m)')
#
#     yaxis_title = "No. of"
#     if feature_type:
#         yaxis_title += f" {feature_type}"
#     if dimension_class:
#         yaxis_title += f" {dimension_class}"
#     yaxis_title += " Metal Losses"
#     ax.set_ylabel(yaxis_title)
#     ax.set_title('Distribution of Metal Loss Defects')
#     ax.set_xlim(-0.5, len(values))
#     ax.set_ylim(0, max(values) + 2)
#     ax.grid(False)
#
#     plt.tight_layout()
#
#     # Save as PNG
#     # image_path = os.path.abspath('metal_loss_3d_look_plot.png')
#     # fig.savefig(image_path)
#     #
#     # if return_fig:
#     #     return fig, image_path
#     # else:
#     #     return image_path
#     # Save as PNG since it's Matplotlib
#     image_path = os.path.abspath("metal_loss_plot.png")
#     fig.savefig(image_path, dpi=300)
#     plt.close(fig)  # free memory
#
#     return fig, image_path

def plot_metal_loss(df, feature_type=None, dimension_class=None, return_fig=False):
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    print(f"[DEBUG] Initial dataframe shape: {df.shape}")

    # Ensure "feature type" exists
    if "feature type" not in df.columns:
        print("No 'Feature Type' column found in data.")
        return None, None

    # Filter Metal Loss rows
    df = df[df["feature type"].str.strip().str.lower() == "metal loss"]
    print(f"[DEBUG] After filtering Metal Loss: {df.shape}")

    # Apply feature_type filter
    if feature_type:
        if "feature identification" in df.columns:
            if feature_type.lower() == "corrosion":
                df = df[df["feature identification"].str.contains("corrosion", case=False, na=False)]
            elif feature_type.lower() == "mfg":
                df = df[df["feature identification"].str.contains("mfg", case=False, na=False)]

    # Apply dimension_class filter
    if dimension_class and dimension_class != "ALL":
        if "dimensions classification" in df.columns:
            df = df[df["dimensions classification"].str.contains(dimension_class, case=False, na=False)]
        elif "dimension classification" in df.columns:
            df = df[df["dimension classification"].str.contains(dimension_class, case=False, na=False)]

    # Handle empty dataframe
    if df.empty:
        print("No matching defects found in the file.")
        return None, None

    # Ensure distance column exists
    if "abs. distance (m)" not in df.columns:
        print("No 'Abs. Distance (m)' column found in data.")
        return None, None

    # Binning by 500m
    bin_size = 500
    max_distance = df["abs. distance (m)"].max()
    bins = list(range(0, int(max_distance) + bin_size, bin_size))
    df["distance bin"] = pd.cut(df["abs. distance (m)"], bins=bins, right=True)

    # Count defects in bins
    bin_counts = df.groupby("distance bin", observed=False).size().reset_index(name="Metal Loss Count")
    bin_counts["Bin Start"] = bin_counts["distance bin"].apply(lambda x: int(x.left))
    bin_counts["Label"] = bin_counts["Bin Start"].astype(str)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(20, 6))
    values = bin_counts["Metal Loss Count"].tolist()
    x_labels = bin_counts["Label"].tolist()

    for i, val in enumerate(values):
        draw_3d_bar(ax, i, val, face_colors=("#4F81BD", "#4F81BD", "#385D8A"))

    ax.set_xticks(np.arange(len(x_labels)) + 0.3)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Distance from Launcher (m)")

    # Dynamic y-axis label
    yaxis_title = "No. of"
    if feature_type:
        yaxis_title += f" {feature_type}"
    if dimension_class:
        yaxis_title += f" {dimension_class}"
    yaxis_title += " Metal Losses"
    ax.set_ylabel(yaxis_title)

    ax.set_title("Distribution of Metal Loss Defects")
    ax.set_xlim(-0.5, len(values))
    ax.set_ylim(0, max(values) + 2)
    ax.grid(False)

    plt.tight_layout()

    # Save as PNG
    image_path = os.path.abspath("metal_loss_plot.png")
    fig.savefig(image_path, dpi=300)
    plt.close(fig)  # free memory

    return fig, image_path



# Updated Sensor Loss Plot function
def plot_sensor_percentage(df, return_fig=False):
    df.columns = df.columns.str.strip()

    # Sort values based on distance for better visualization
    df.sort_values(by='Abs. Distance (m)', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Abs. Distance (m)'],
        y=df['Sensor %'],
        mode='lines',
        name='Sensor %',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        # title='Sensor % vs. Absolute Distance',
        xaxis=dict(title='Absolute Distance (m)', gridcolor='lightgray', dtick=2000),
        yaxis=dict(
            title='Sensor %',
            gridcolor='lightgray',
            range=[0, 100],  # Fixed Y-axis range
            dtick=20         # Y-axis ticks at 0, 20, 40, ..., 100
        ),
        height=700,
        width=1600,
        template='plotly_white'
    )

    html_path = os.path.abspath('sensor_percentage_plot.html')
    fig.write_html(html_path)

    if return_fig:
        return fig, html_path
    else:
        return html_path

def plot_temperature(df, return_fig=False):
    df.columns = df.columns.str.strip()
    df.sort_values(by='Abs. Distance (m)', inplace=True)
    if 'Temperature (°C)' not in df.columns:
        temperature_profile = np.linspace(55, 50, len(df))
        noise=np.random.uniform(low=-0.5,high=0.5, size=len(df))

        df['Temperature (°C)'] = temperature_profile + noise


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Abs. Distance (m)'],
        y=df['Temperature (°C)'],
        mode='lines',
        name='Temperature Profile',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        # title='Temperature Level Profile',
        xaxis=dict(title='Absolute Distance (m)', gridcolor='lightgray', dtick=2000,tickformat="d"),
        yaxis=dict(
            title='Temperature (°C)',
            gridcolor='lightgray',
            range=[0, 100],
            dtick=10
        ),
        height=700,
        width=1600,
        template='plotly_white'
    )

    html_path = os.path.abspath('temperature_plot.html')
    fig.write_html(html_path)

    if return_fig:
        return fig, html_path
    else:
        return html_path