import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d


PKL_PATH = r"D:\Anubhav\softwares\client software\Data\project_oil_sample\pickle_data\150.pkl"
PIPE_NUMBER = 150


# ─────────────────────────────────────────────────────────────
#  VERSION 1 — Original style
#  Closest to your plot_clock_heatmap_t8, jet colormap, raw values,
#  no normalisation, just stripped of Qt/DB/selector stuff
# ─────────────────────────────────────────────────────────────

def plot_heatmap_original(pkl_path, pipe_number=None):
    df = pd.read_pickle(pkl_path)
    print(f"[ORIGINAL] Loaded: {pkl_path}  shape={df.shape}")

    hall_cols = [c for c in df.columns if c.startswith('F') and 'H' in c]
    df_clk = df[hall_cols].copy()
    df_clk = df_clk.apply(pd.to_numeric, errors='coerce')

    d1 = df_clk.transpose().astype(float)

    oddo_val = (pd.to_numeric(df['ODDO1'], errors='coerce') / 1000.0).round(2).tolist()

    fig, ax2 = plt.subplots(figsize=(20, 8))
    fig.subplots_adjust(bottom=0.213, left=0.077, top=0.855, right=1.000)

    heat_map_obj = sns.heatmap(d1, cmap='jet', ax=ax2, vmin=-0.01, vmax=0.07)
    heat_map_obj.set(xlabel="Index", ylabel="Clock")

    ax2.set_xticklabels(ax2.get_xticklabels(), size=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), size=9)

    ax3 = ax2.twiny()
    num_ticks = len(ax2.get_xticks())
    tick_positions = [int(i) for i in np.linspace(0, len(oddo_val) - 1, num_ticks)]
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([f'{oddo_val[i]:.2f}' for i in tick_positions], rotation=90, size=9)
    ax3.set_xlabel("Absolute Distance (m)", size=9)

    title = f"Hall-Sensor Heatmap [ORIGINAL] — Joint {pipe_number}" if pipe_number else "Hall-Sensor Heatmap [ORIGINAL]"
    ax2.set_title(title, fontsize=13, fontweight='bold', pad=40)

    plt.show()
    print("[ORIGINAL] Done.")


# ─────────────────────────────────────────────────────────────
#  VERSION 2 — Improved (median/IQR normalisation, jet colormap)
#  Removes inter-sensor DC bias, zeros dropout spikes,
#  light smoothing — anomaly should be clearly visible
# ─────────────────────────────────────────────────────────────

def plot_heatmap_improved(pkl_path, pipe_number=None):
    df = pd.read_pickle(pkl_path)
    print(f"[IMPROVED] Loaded: {pkl_path}  shape={df.shape}")

    hall_cols = [c for c in df.columns if c.startswith('F') and 'H' in c]
    df_sens = df[hall_cols].copy()
    df_sens = df_sens.apply(pd.to_numeric, errors='coerce')
    df_sens = df_sens.ffill().fillna(0.0)

    # Median/IQR normalisation — removes inter-sensor DC bias
    sensor_median = df_sens.median()
    sensor_iqr    = (df_sens.quantile(0.75) - df_sens.quantile(0.25)) + 1e-6
    df_norm = (df_sens - sensor_median) / sensor_iqr

    # Zero out dropout spikes (single-sample faults > ±4 IQR)
    df_norm = df_norm.where(df_norm.abs() <= 4.0, other=0.0)

    # Light spatial smoothing — 100 samples ≈ 0.05m
    df_norm = pd.DataFrame(
        uniform_filter1d(df_norm.values, size=100, axis=0),
        columns=df_norm.columns,
        index=df_norm.index,
    )

    df_norm = df_norm.clip(-2.0, 4.0)

    d1 = df_norm.transpose().astype(float)

    oddo_val = (pd.to_numeric(df['ODDO1'], errors='coerce') / 1000.0).round(2).tolist()

    fig, ax2 = plt.subplots(figsize=(20, 8))
    fig.subplots_adjust(bottom=0.213, left=0.077, top=0.855, right=1.000)

    heat_map_obj = sns.heatmap(d1, cmap='jet', ax=ax2, vmin=-2.0, vmax=4.0)
    heat_map_obj.set(xlabel="Index", ylabel="Sensor")

    ax2.set_xticklabels(ax2.get_xticklabels(), size=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), size=9)

    ax3 = ax2.twiny()
    num_ticks = len(ax2.get_xticks())
    tick_positions = [int(i) for i in np.linspace(0, len(oddo_val) - 1, num_ticks)]
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([f'{oddo_val[i]:.2f}' for i in tick_positions], rotation=90, size=9)
    ax3.set_xlabel("Absolute Distance (m)", size=9)

    title = f"Hall-Sensor Heatmap [IMPROVED] — Joint {pipe_number}" if pipe_number else "Hall-Sensor Heatmap [IMPROVED]"
    ax2.set_title(title, fontsize=13, fontweight='bold', pad=40)

    plt.show()
    print("[IMPROVED] Done.")


# ─────────────────────────────────────────────────────────────
#  RUN BOTH
# ─────────────────────────────────────────────────────────────

# plot_heatmap_original(PKL_PATH, PIPE_NUMBER)
plot_heatmap_improved(PKL_PATH, PIPE_NUMBER)