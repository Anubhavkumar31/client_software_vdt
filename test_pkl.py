import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


num_of_sensors = 36
minute = 720 / num_of_sensors
degree = minute / 2
pkl_path = r"D:\Anubhav\softwares\client software\client_software_vdt\test.pkl"


def plot_clock_heatmap_t8(pkl_path):
    df_new = pd.read_pickle(pkl_path)
    oddo_val = [round(elem / 1000, 2) for elem in df_new['ODDO1'].tolist()]
    index_hm = list(df_new['index'])

    df_clk = df_new[[f"{h:02}:{m:02}" for h in range(12) for m in range(0, 60, int(minute))]]
    df_clk = df_clk.apply(pd.to_numeric, errors='coerce')

    d1 = df_clk.transpose().astype(float)

    fig, ax2 = plt.subplots(figsize=(20, 8))
    fig.subplots_adjust(bottom=0.213, left=0.077, top=0.855, right=1.000)

    vmin = -0.1
    vmax = 0.3

    heat_map_obj = sns.heatmap(d1, cmap='jet', ax=ax2, vmin=vmin, vmax=vmax)
    heat_map_obj.set(xlabel="Index", ylabel="Clock")

    ax2.set_xticklabels(ax2.get_xticklabels(), size=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), size=9)

    ax3 = ax2.twiny()
    num_ticks1 = len(ax2.get_xticks())
    tick_positions1 = [int(i) for i in np.linspace(0, len(oddo_val) - 1, num_ticks1)]
    ax3.set_xticks(tick_positions1)
    ax3.set_xticklabels([f'{oddo_val[i]:.2f}' for i in tick_positions1], rotation=90, size=9)
    ax3.set_xlabel("Absolute Distance (m)", size=9)

    plt.show()

plot_clock_heatmap_t8(pkl_path)