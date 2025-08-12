import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class PipeHighlightApp:
    def __init__(self, master):
        self.master = master
        self.setup_ui()

    def setup_ui(self):
        # Load the CSV file
        file_path = resource_path("backend/files/datalog/ptal.xlsx")
        self.df = pd.read_excel(file_path)

        # Define constants
        self.CONTRACTOR = 'ZZZ'
        self.IP_TYPE = 'MFL'
        self.MEDIUM = 'Oil'
        self.TYPE_PIPE = 'ZZZ'
        self.GRADE_PIPE = 'ZZZ'

        # Known
        self.DIA = 340
        self.WT = 7.1

        # To be set
        self.DP = 3.67
        self.OP = 0
        self.MAOP = 11
        self.DF = 0.72
        self.UTS = 413.686
        self.SMYS = 2493.8

        # Calculate parameters from the DataFrame
        self.calculate_statistics()

        # Create main UI elements
        self.create_main_frame()
        self.create_general_info_section()
        self.create_statistics_section()

    def calculate_statistics(self):
        self.TOT_ANAL = self.df.shape[0]
        self.INT_ANAL = self.df[self.df['Type'] == 'Internal'].shape[0]
        self.EXT_ANAL = self.df[self.df['Type'] == 'External'].shape[0]

        self.ERF_95 = self.df[self.df['ERF (ASME B31G)'] < 0.95].shape[0]
        self.ERF_95_1 = self.df[(self.df['ERF (ASME B31G)'] >= 0.95) & (self.df['ERF (ASME B31G)'] < 1)].shape[0]
        self.ERF_1 = self.df[self.df['ERF (ASME B31G)'] >= 1].shape[0]

        self.DEP_25 = self.df[self.df['Depth % '] < 25].shape[0]
        self.DEP_25_50 = self.df[(self.df['Depth % '] >= 25) & (self.df['Depth % '] < 50)].shape[0]
        self.DEP_50_80 = self.df[(self.df['Depth % '] >= 50) & (self.df['Depth % '] < 80)].shape[0]
        self.DEP_80_100 = self.df[(self.df['Depth % '] >= 80) & (self.df['Depth % '] <= 100)].shape[0]

    def create_main_frame(self):
        self.master.title("Pipeline Highlights")
        self.master.geometry("500x500+1100+75")
        self.master.minsize(500, 500)
        self.master.maxsize(1400, 1000)
        

        self.main_frame = ttk.Frame(self.master)
        self.main_frame.grid(sticky="nsew", padx=20, pady=20)

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

    def create_general_info_section(self):
        general_info_frame = ttk.LabelFrame(self.main_frame, text="General Info", padding=(10, 5), style='Custom.TLabelframe')
        general_info_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        general_info_frame.grid_columnconfigure(0, weight=1)
        general_info_frame.grid_columnconfigure(1, weight=1)
        general_info_frame.grid_columnconfigure(2, weight=1)

        left_general_info = [
            f"Contractor: {self.CONTRACTOR}",
            f"IP Type: {self.IP_TYPE}",
            f"Medium: {self.MEDIUM}",
            f"Type of Pipe: {self.TYPE_PIPE}",
            f"Grade of Pipe: {self.GRADE_PIPE}",
        ]

        mid_general_info = [
            f"Diameter (mm): {self.DIA}",
            f"Wall Thickness (mm): {self.WT}",
            f"Design Pressure (MPa): {self.DP}",
            f"Operating Pressure (MPa): {self.OP}",
        ]

        right_general_info = [
            f"MAOP (MPa): {self.MAOP}",
            f"Design Factor: {self.DF}",
            f"UTS (MPa): {self.UTS}",
            f"SMYS (MPa): {self.SMYS}"
        ]

        for i, info in enumerate(left_general_info):
            key, value = info.split(": ")
            label = ttk.Label(general_info_frame, text=key, font=("Helvetica", 16), anchor='w')
            label.grid(row=i, column=0, sticky='w', padx=2, pady=2)

            entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
            entry.insert(0, value)
            entry.configure(state='readonly')
            entry.grid(row=i, column=1, sticky='w', padx=2, pady=2)

        for i, info in enumerate(mid_general_info):
            key, value = info.split(": ")
            label = ttk.Label(general_info_frame, text=key, font=("Helvetica", 16), anchor='w')
            label.grid(row=i, column=2, sticky='w', padx=2, pady=2)

            entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
            entry.insert(0, value)
            entry.configure(state='readonly')
            entry.grid(row=i, column=3, sticky='w', padx=2, pady=2)

        for i, info in enumerate(right_general_info):
            key, value = info.split(": ")
            label = ttk.Label(general_info_frame, text=key, font=("Helvetica", 16), anchor='w')
            label.grid(row=i, column=4, sticky='w', padx=2, pady=2)

            entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
            entry.insert(0, value)
            entry.configure(state='readonly')
            entry.grid(row=i, column=5, sticky='w', padx=2, pady=2)

    def create_statistics_section(self):
        stats_frame = ttk.Frame(self.main_frame)
        stats_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)
        stats_frame.grid_rowconfigure(0, weight=1)

        stats_labels_frame = ttk.LabelFrame(stats_frame, text="Statistics", padding=(10, 5), style='Custom.TLabelframe')
        stats_labels_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        stats_labels_frame.grid_rowconfigure(0, weight=1)

        charts_frame = ttk.Frame(stats_frame)
        charts_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        charts_frame.grid_rowconfigure(0, weight=1)

        stats_labels = [
            f"Total Anomalies Count: {self.TOT_ANAL}",
            f"Int. Anomalies Count: {self.INT_ANAL}",
            f"Ext. Anomalies Count: {self.EXT_ANAL}",
            "",  # Spacer
            "",  # Spacer
            "",  # Spacer
            "",  # Spacer
            f"ERF < 0.95: {self.ERF_95}",
            f"0.95 <= ERF < 1: {self.ERF_95_1}",
            f"1 <= ERF: {self.ERF_1}",
            "",  # Spacer
            "",  # Spacer
            "",  # Spacer
            "",  # Spacer
            f"Depth < 25%: {self.DEP_25}",
            f"25% <= Depth < 50%: {self.DEP_25_50}",
            f"50% <= Depth < 80%: {self.DEP_50_80}",
            f"80% <= Depth <= 100%: {self.DEP_80_100}",
        ]

        for i, stat in enumerate(stats_labels):
            if stat == "":
                spacer_label = ttk.Label(stats_labels_frame, text="", font=("Helvetica", 16))
                spacer_label.grid(row=i, column=0, sticky='w', padx=5, pady=5)
                continue

            key = stat.split(":")[0]
            label = ttk.Label(stats_labels_frame, text=key + ":", font=("Helvetica", 16), anchor='w')
            label.grid(row=i, column=0, sticky='w', padx=5, pady=5)

            if ":" in stat:
                value = stat.split(":")[1].strip()
                entry = ttk.Entry(stats_labels_frame, width=10, font=("Helvetica", 16))
                entry.insert(0, value)
                entry.configure(state='readonly')
                entry.grid(row=i, column=1, sticky='w', padx=5, pady=5)

        # Create pie charts using Matplotlib
        fig, axs = plt.subplots(3, 1, figsize=(3, 5))

        def autopct_func(pct):
            return f'{pct:.1f}%' if pct >= 4 else ''

        # Pie chart for Internal/External Anomalies
        labels = ['Internal ML', 'External ML']
        sizes = [self.INT_ANAL, self.EXT_ANAL]
        colors = ['red', 'blue']
        axs[0].pie(sizes, colors=colors, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
        axs[0].axis('equal')
        axs[0].legend(labels, loc='lower right', frameon=False, fontsize=3)

        # Pie chart for ERF
        labels_erf = ['ERF < 0.95', '0.95 <= ERF < 1', '1 <= ERF']
        sizes_erf = [self.ERF_95, self.ERF_95_1, self.ERF_1]
        colors_erf = ['green', 'yellow', 'red']
        axs[1].pie(sizes_erf, colors=colors_erf, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
        axs[1].axis('equal')
        axs[1].legend(labels_erf, loc='lower right', frameon=False, fontsize=3)

        # Pie chart for Depths
        labels_depth = ['Depth < 25%', '25% <= Depth < 50%', '50% <= Depth < 80%', '80% <= Depth <= 100%']
        sizes_depth = [self.DEP_25, self.DEP_25_50, self.DEP_50_80, self.DEP_80_100]
        colors_depth = ['purple', 'green', 'yellow', 'brown']
        axs[2].pie(sizes_depth, colors=colors_depth, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
        axs[2].axis('equal')
        axs[2].legend(labels_depth, loc='lower right', frameon=False, fontsize=3)

        # Display the pie charts in the charts frame
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Configure styles for LabelFrames
        style = ttk.Style()
        style.configure('Custom.TLabelframe.Label', font=("Helvetica", 20, "bold"), foreground='maroon')
        style.configure('Custom.TLabelframe', background='white')

def run_app():
    root = tk.Tk()
    app = PipeHighlightApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()