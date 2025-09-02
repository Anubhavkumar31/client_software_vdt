# # # At the very top of your Pipe_Highlights.py file, before other imports
# # import matplotlib
# # matplotlib.use('Qt5Agg')  # Use Qt-compatible backend
# # import matplotlib.pyplot as plt

# # import tkinter as tk
# # from tkinter import ttk
# # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import os
# # import sys

# # def resource_path(relative_path):
# #     """ Get absolute path to resource, works for dev and for PyInstaller """
# #     if getattr(sys, 'frozen', False):
# #         base_path = sys._MEIPASS
# #     else:
# #         base_path = os.path.abspath(".")
# #     return os.path.join(base_path, relative_path)

# # class PipeHighlightApp:
# #     def __init__(self, master, pipe_tally_df=None):
# #         print("🔍 Initializing PipeHighlightApp...")
# #         self.master = master
        
# #         # ✅ Use provided DataFrame or fallback to file loading
# #         if pipe_tally_df is not None and not pipe_tally_df.empty:
# #             self.df = pipe_tally_df.copy()  # Make a copy to avoid modifying original
# #             print(f"✅ Using loaded pipe tally data from project ({len(self.df)} rows)")
# #             print(f"📊 Available columns: {list(self.df.columns)}")
# #         else:
# #             print("⚠️ No DataFrame provided, trying fallback file...")
# #             try:
# #                 file_path = resource_path("backend/files/datalog/ptal.xlsx")
# #                 if os.path.exists(file_path):
# #                     self.df = pd.read_excel(file_path)
# #                     print("⚠️ Loading pipe tally from fallback file")
# #                 else:
# #                     print(f"❌ Fallback file not found: {file_path}")
# #                     self.df = pd.DataFrame()
# #             except Exception as e:
# #                 print(f"❌ Error loading fallback file: {e}")
# #                 self.df = pd.DataFrame()
        
# #         print("🔍 About to setup UI...")
# #         try:
# #             self.setup_ui()
# #             print("✅ UI setup completed successfully!")
# #         except Exception as e:
# #             print(f"❌ Error during UI setup: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             # Create minimal UI on error
# #             self.create_error_ui(str(e))

# #     def create_error_ui(self, error_msg):
# #         """Create a simple error display UI"""
# #         self.master.title("Pipeline Highlights - Error")
# #         self.master.geometry("500x300")
        
# #         error_frame = ttk.Frame(self.master)
# #         error_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
# #         error_label = ttk.Label(error_frame, text="Error Loading Pipeline Highlights", 
# #                                font=("Arial", 16, "bold"), foreground="red")
# #         error_label.pack(pady=10)
        
# #         error_text = tk.Text(error_frame, wrap=tk.WORD, height=10)
# #         error_text.pack(fill=tk.BOTH, expand=True, pady=5)
# #         error_text.insert(tk.END, f"Error Details:\n{error_msg}")
# #         error_text.config(state=tk.DISABLED)

# #     def setup_ui(self):
# #         print("🔍 Starting UI setup...")
        
# #         # Define constants
# #         self.CONTRACTOR = 'ZZZ'
# #         self.IP_TYPE = 'MFL'
# #         self.MEDIUM = 'Oil'
# #         self.TYPE_PIPE = 'ZZZ'
# #         self.GRADE_PIPE = 'ZZZ'

# #         # Known values
# #         self.DIA = 340
# #         self.WT = 7.1

# #         # To be set values
# #         self.DP = 3.67
# #         self.OP = 0
# #         self.MAOP = 11
# #         self.DF = 0.72
# #         self.UTS = 413.686
# #         self.SMYS = 2493.8

# #         print("🔍 About to calculate statistics...")
# #         self.calculate_statistics()
        
# #         print("🔍 About to create main frame...")
# #         self.create_main_frame()
        
# #         print("🔍 About to create general info section...")
# #         self.create_general_info_section()
        
# #         print("🔍 About to create statistics section...")
# #         self.create_statistics_section()
        
# #         print("✅ All UI components created successfully!")

# #     def calculate_statistics(self):
# #         """Calculate statistics from the loaded DataFrame"""
# #         print("🔍 Calculating statistics...")
        
# #         if self.df.empty:
# #             print("⚠️ DataFrame is empty, using default values")
# #             self.TOT_ANAL = 0
# #             self.INT_ANAL = 0
# #             self.EXT_ANAL = 0
# #             self.ERF_95 = 0
# #             self.ERF_95_1 = 0
# #             self.ERF_1 = 0
# #             self.DEP_25 = 0
# #             self.DEP_25_50 = 0
# #             self.DEP_50_80 = 0
# #             self.DEP_80_100 = 0
# #             return

# #         try:
# #             # Total anomalies
# #             self.TOT_ANAL = len(self.df)
# #             print(f"📊 Total anomalies: {self.TOT_ANAL}")
            
# #             # Internal/External anomalies (handle different column name variations)
# #             type_col = None
# #             for col in ['Type', 'Feature Type', 'Anomaly Type']:
# #                 if col in self.df.columns:
# #                     type_col = col
# #                     break
            
# #             if type_col:
# #                 type_series = self.df[type_col].astype(str).str.lower()
# #                 self.INT_ANAL = len(self.df[type_series.str.contains('internal', na=False)])
# #                 self.EXT_ANAL = len(self.df[type_series.str.contains('external', na=False)])
# #                 print(f"📊 Internal: {self.INT_ANAL}, External: {self.EXT_ANAL} (using column: {type_col})")
# #             else:
# #                 self.INT_ANAL = 0
# #                 self.EXT_ANAL = 0
# #                 print("⚠️ No Type column found")

# #             # ERF statistics (handle different column name variations)
# #             erf_col = None
# #             for col in ['ERF (ASME B31G)', 'ERF', 'Engineering Risk Factor']:
# #                 if col in self.df.columns:
# #                     erf_col = col
# #                     break
            
# #             if erf_col:
# #                 erf_data = pd.to_numeric(self.df[erf_col], errors='coerce').dropna()
# #                 if len(erf_data) > 0:
# #                     self.ERF_95 = len(erf_data[erf_data < 0.95])
# #                     self.ERF_95_1 = len(erf_data[(erf_data >= 0.95) & (erf_data < 1)])
# #                     self.ERF_1 = len(erf_data[erf_data >= 1])
# #                     print(f"📊 ERF stats calculated from {len(erf_data)} valid values (using column: {erf_col})")
# #                 else:
# #                     self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
# #                     print("⚠️ No valid ERF data found")
# #             else:
# #                 self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
# #                 print("⚠️ No ERF column found")

# #             # Depth statistics (handle different column name variations)
# #             depth_col = None
# #             for col in ['Depth %', 'Depth % ', 'Depth Percentage']:
# #                 if col in self.df.columns:
# #                     depth_col = col
# #                     break
            
# #             if depth_col:
# #                 depth_data = pd.to_numeric(self.df[depth_col], errors='coerce').dropna()
# #                 if len(depth_data) > 0:
# #                     self.DEP_25 = len(depth_data[depth_data < 25])
# #                     self.DEP_25_50 = len(depth_data[(depth_data >= 25) & (depth_data < 50)])
# #                     self.DEP_50_80 = len(depth_data[(depth_data >= 50) & (depth_data < 80)])
# #                     self.DEP_80_100 = len(depth_data[(depth_data >= 80) & (depth_data <= 100)])
# #                     print(f"📊 Depth stats calculated from {len(depth_data)} valid values (using column: {depth_col})")
# #                 else:
# #                     self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0
# #                     print("⚠️ No valid depth data found")
# #             else:
# #                 self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0
# #                 print("⚠️ No Depth column found")

# #             print("✅ Statistics calculation completed successfully")

# #         except Exception as e:
# #             print(f"❌ Error calculating statistics: {e}")
# #             # Set default values on error
# #             self.TOT_ANAL = len(self.df) if not self.df.empty else 0
# #             self.INT_ANAL = self.EXT_ANAL = 0
# #             self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
# #             self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0

# #     def create_main_frame(self):
# #         print("🔍 Creating main frame...")
# #         self.master.title("Pipeline Highlights")
# #         self.master.geometry("500x500+1100+75")
# #         self.master.minsize(500, 500)
# #         self.master.maxsize(1400, 1000)

# #         self.main_frame = ttk.Frame(self.master)
# #         self.main_frame.grid(sticky="nsew", padx=20, pady=20)

# #         # Configure grid weights
# #         self.master.grid_rowconfigure(0, weight=1)
# #         self.master.grid_columnconfigure(0, weight=1)

# #         self.main_frame.grid_rowconfigure(1, weight=1)
# #         self.main_frame.grid_columnconfigure(0, weight=1)
        
# #         print("✅ Main frame created and configured")

# #     def create_general_info_section(self):
# #         print("🔍 Creating general info section...")
        
# #         general_info_frame = ttk.LabelFrame(self.main_frame, text="General Info", padding=(10, 5))
# #         general_info_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

# #         # Configure grid for 6 columns (3 pairs of label-entry)
# #         for i in range(6):
# #             general_info_frame.grid_columnconfigure(i, weight=1)

# #         left_general_info = [
# #             ("Contractor", self.CONTRACTOR),
# #             ("IP Type", self.IP_TYPE),
# #             ("Medium", self.MEDIUM),
# #             ("Type of Pipe", self.TYPE_PIPE),
# #             ("Grade of Pipe", self.GRADE_PIPE),
# #         ]

# #         mid_general_info = [
# #             ("Diameter (mm)", self.DIA),
# #             ("Wall Thickness (mm)", self.WT),
# #             ("Design Pressure (MPa)", self.DP),
# #             ("Operating Pressure (MPa)", self.OP),
# #         ]

# #         right_general_info = [
# #             ("MAOP (MPa)", self.MAOP),
# #             ("Design Factor", self.DF),
# #             ("UTS (MPa)", self.UTS),
# #             ("SMYS (MPa)", self.SMYS)
# #         ]

# #         # Create left column
# #         for i, (key, value) in enumerate(left_general_info):
# #             label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
# #             label.grid(row=i, column=0, sticky='w', padx=2, pady=2)

# #             entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
# #             entry.insert(0, str(value))
# #             entry.configure(state='readonly')
# #             entry.grid(row=i, column=1, sticky='w', padx=2, pady=2)

# #         # Create middle column
# #         for i, (key, value) in enumerate(mid_general_info):
# #             label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
# #             label.grid(row=i, column=2, sticky='w', padx=2, pady=2)

# #             entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
# #             entry.insert(0, str(value))
# #             entry.configure(state='readonly')
# #             entry.grid(row=i, column=3, sticky='w', padx=2, pady=2)

# #         # Create right column
# #         for i, (key, value) in enumerate(right_general_info):
# #             label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
# #             label.grid(row=i, column=4, sticky='w', padx=2, pady=2)

# #             entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
# #             entry.insert(0, str(value))
# #             entry.configure(state='readonly')
# #             entry.grid(row=i, column=5, sticky='w', padx=2, pady=2)

# #         print("✅ General info section created")

# #     def create_statistics_section(self):
# #         print("🔍 Creating statistics section...")
        
# #         stats_frame = ttk.Frame(self.main_frame)
# #         stats_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

# #         stats_frame.grid_columnconfigure(0, weight=1)
# #         stats_frame.grid_columnconfigure(1, weight=1)
# #         stats_frame.grid_rowconfigure(0, weight=1)

# #         # Statistics labels frame
# #         stats_labels_frame = ttk.LabelFrame(stats_frame, text="Statistics", padding=(10, 5))
# #         stats_labels_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# #         # Charts frame
# #         charts_frame = ttk.Frame(stats_frame)
# #         charts_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
# #         charts_frame.grid_rowconfigure(0, weight=1)

# #         # Statistics data
# #         stats_data = [
# #             ("Total Anomalies Count", self.TOT_ANAL),
# #             ("Int. Anomalies Count", self.INT_ANAL),
# #             ("Ext. Anomalies Count", self.EXT_ANAL),
# #             ("", ""),  # Spacer
# #             ("ERF < 0.95", self.ERF_95),
# #             ("0.95 <= ERF < 1", self.ERF_95_1),
# #             ("1 <= ERF", self.ERF_1),
# #             ("", ""),  # Spacer
# #             ("Depth < 25%", self.DEP_25),
# #             ("25% <= Depth < 50%", self.DEP_25_50),
# #             ("50% <= Depth < 80%", self.DEP_50_80),
# #             ("80% <= Depth <= 100%", self.DEP_80_100),
# #         ]

# #         # Create statistics labels and entries
# #         for i, (key, value) in enumerate(stats_data):
# #             if key == "":
# #                 # Spacer
# #                 spacer_label = ttk.Label(stats_labels_frame, text="", font=("Helvetica", 16))
# #                 spacer_label.grid(row=i, column=0, sticky='w', padx=5, pady=5)
# #                 continue

# #             # Label
# #             label = ttk.Label(stats_labels_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
# #             label.grid(row=i, column=0, sticky='w', padx=5, pady=5)

# #             # Entry
# #             entry = ttk.Entry(stats_labels_frame, width=10, font=("Helvetica", 16))
# #             entry.insert(0, str(value))
# #             entry.configure(state='readonly')
# #             entry.grid(row=i, column=1, sticky='w', padx=5, pady=5)

# #         # Create pie charts
# #         try:
# #             print("🔍 Creating pie charts...")
            
# #             # Set matplotlib backend
# #             import matplotlib
# #             matplotlib.use('TkAgg')
            
# #             fig, axs = plt.subplots(3, 1, figsize=(3, 5))

# #             def autopct_func(pct):
# #                 return f'{pct:.1f}%' if pct >= 4 else ''

# #             # Pie chart for Internal/External Anomalies
# #             labels = ['Internal ML', 'External ML']
# #             sizes = [self.INT_ANAL, self.EXT_ANAL]
# #             colors = ['red', 'blue']
            
# #             if sum(sizes) > 0:
# #                 axs[0].pie(sizes, colors=colors, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
# #             else:
# #                 axs[0].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[0].transAxes)
# #             axs[0].axis('equal')
# #             axs[0].legend(labels, loc='lower right', frameon=False, fontsize=3)

# #             # Pie chart for ERF
# #             labels_erf = ['ERF < 0.95', '0.95 <= ERF < 1', '1 <= ERF']
# #             sizes_erf = [self.ERF_95, self.ERF_95_1, self.ERF_1]
# #             colors_erf = ['green', 'yellow', 'red']
            
# #             if sum(sizes_erf) > 0:
# #                 axs[1].pie(sizes_erf, colors=colors_erf, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
# #             else:
# #                 axs[1].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[1].transAxes)
# #             axs[1].axis('equal')
# #             axs[1].legend(labels_erf, loc='lower right', frameon=False, fontsize=3)

# #             # Pie chart for Depths
# #             labels_depth = ['Depth < 25%', '25% <= Depth < 50%', '50% <= Depth < 80%', '80% <= Depth <= 100%']
# #             sizes_depth = [self.DEP_25, self.DEP_25_50, self.DEP_50_80, self.DEP_80_100]
# #             colors_depth = ['purple', 'green', 'yellow', 'brown']
            
# #             if sum(sizes_depth) > 0:
# #                 axs[2].pie(sizes_depth, colors=colors_depth, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
# #             else:
# #                 axs[2].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[2].transAxes)
# #             axs[2].axis('equal')
# #             axs[2].legend(labels_depth, loc='lower right', frameon=False, fontsize=3)

# #             # Adjust layout
# #             plt.tight_layout()

# #             # Display the pie charts in the charts frame
# #             canvas = FigureCanvasTkAgg(fig, master=charts_frame)
# #             canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
# #             canvas.draw()

# #             print("✅ Pie charts created successfully")

# #         except Exception as e:
# #             print(f"❌ Error creating pie charts: {e}")
# #             import traceback
# #             traceback.print_exc()
            
# #             # Create a simple fallback display
# #             fallback_label = ttk.Label(charts_frame, text="Charts unavailable\n(Check console for details)", 
# #                                      font=("Helvetica", 12), justify='center')
# #             fallback_label.pack(expand=True)

# #         # Configure styles for LabelFrames
# #         try:
# #             style = ttk.Style()
# #             style.configure('Custom.TLabelframe.Label', font=("Helvetica", 20, "bold"), foreground='maroon')
# #             style.configure('Custom.TLabelframe', background='white')
# #         except Exception as e:
# #             print(f"⚠️ Warning: Could not configure styles: {e}")

# #         print("✅ Statistics section created")

# # def run_app(pipe_tally_df=None):
# #     """Main function to run the Pipeline Highlights application"""
# #     print("🚀 Starting Pipeline Highlights App...")
    
# #     try:
# #         root = tk.Tk()
# #         app = PipeHighlightApp(root, pipe_tally_df=pipe_tally_df)
# #         print("🎯 Starting mainloop...")
# #         root.mainloop()
# #         print("✅ App closed normally")
# #     except Exception as e:
# #         print(f"❌ Critical error in run_app: {e}")
# #         import traceback
# #         traceback.print_exc()

# # # Debug version for testing
# # def run_app_debug(pipe_tally_df=None):
# #     """Debug version with minimal UI for testing"""
# #     print("🚀 Starting DEBUG Pipeline Highlights App...")
    
# #     root = tk.Tk()
# #     root.title("Pipeline Highlights - DEBUG")
# #     root.geometry("400x300")
    
# #     main_frame = ttk.Frame(root)
# #     main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
# #     # Test label
# #     test_label = ttk.Label(main_frame, text="Pipeline Highlights - DEBUG MODE", font=("Arial", 16))
# #     test_label.pack(pady=10)
    
# #     # Data info
# #     if pipe_tally_df is not None and not pipe_tally_df.empty:
# #         data_label = ttk.Label(main_frame, text=f"✅ Data loaded: {len(pipe_tally_df)} rows", font=("Arial", 12))
# #         data_label.pack(pady=5)
        
# #         cols_label = ttk.Label(main_frame, text=f"📊 Columns: {len(pipe_tally_df.columns)}", font=("Arial", 10))
# #         cols_label.pack(pady=5)
        
# #         # Show some column names
# #         col_names = ', '.join(list(pipe_tally_df.columns)[:5])
# #         if len(pipe_tally_df.columns) > 5:
# #             col_names += "..."
# #         cols_detail = ttk.Label(main_frame, text=f"Sample columns: {col_names}", font=("Arial", 8))
# #         cols_detail.pack(pady=5)
# #     else:
# #         data_label = ttk.Label(main_frame, text="❌ No data loaded", font=("Arial", 12))
# #         data_label.pack(pady=5)
    
# #     # Test button
# #     def test_button_click():
# #         print("✅ Button clicked - UI is working!")
# #         if pipe_tally_df is not None:
# #             print(f"Data shape: {pipe_tally_df.shape}")
# #             print(f"Columns: {list(pipe_tally_df.columns)}")
    
# #     test_button = ttk.Button(main_frame, text="Test Data Access", command=test_button_click)
# #     test_button.pack(pady=10)
    
# #     # Launch full app button
# #     def launch_full_app():
# #         root.destroy()
# #         run_app(pipe_tally_df)
    
# #     full_app_button = ttk.Button(main_frame, text="Launch Full App", command=launch_full_app)
# #     full_app_button.pack(pady=5)
    
# #     print("🎯 Starting debug mainloop...")
# #     root.mainloop()
# #     print("✅ Debug app closed normally")

# # if __name__ == "__main__":
# #     run_app()



# # ✅ Set backend at module level BEFORE other imports (or remove entirely)
# import matplotlib
# # matplotlib.use('Qt5Agg')  # Optional: Use Qt-compatible backend

# import tkinter as tk
# from tkinter import ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import sys

# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     if getattr(sys, 'frozen', False):
#         base_path = sys._MEIPASS
#     else:
#         base_path = os.path.abspath(".")
#     return os.path.join(base_path, relative_path)

# class PipeHighlightApp:
#     def __init__(self, master, pipe_tally_df=None):
#         print("🔍 Initializing PipeHighlightApp...")
#         self.master = master
        
#         # ✅ Use provided DataFrame or fallback to file loading
#         if pipe_tally_df is not None and not pipe_tally_df.empty:
#             self.df = pipe_tally_df.copy()  # Make a copy to avoid modifying original
#             print(f"✅ Using loaded pipe tally data from project ({len(self.df)} rows)")
#             print(f"📊 Available columns: {list(self.df.columns)}")
#         else:
#             print("⚠️ No DataFrame provided, trying fallback file...")
#             try:
#                 file_path = resource_path("backend/files/datalog/ptal.xlsx")
#                 if os.path.exists(file_path):
#                     self.df = pd.read_excel(file_path)
#                     print("⚠️ Loading pipe tally from fallback file")
#                 else:
#                     print(f"❌ Fallback file not found: {file_path}")
#                     self.df = pd.DataFrame()
#             except Exception as e:
#                 print(f"❌ Error loading fallback file: {e}")
#                 self.df = pd.DataFrame()
        
#         print("🔍 About to setup UI...")
#         try:
#             self.setup_ui()
#             print("✅ UI setup completed successfully!")
#         except Exception as e:
#             print(f"❌ Error during UI setup: {e}")
#             import traceback
#             traceback.print_exc()
#             # Create minimal UI on error
#             self.create_error_ui(str(e))

#     def create_error_ui(self, error_msg):
#         """Create a simple error display UI"""
#         self.master.title("Pipeline Highlights - Error")
#         self.master.geometry("500x300")
        
#         error_frame = ttk.Frame(self.master)
#         error_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
#         error_label = ttk.Label(error_frame, text="Error Loading Pipeline Highlights", 
#                                font=("Arial", 16, "bold"), foreground="red")
#         error_label.pack(pady=10)
        
#         error_text = tk.Text(error_frame, wrap=tk.WORD, height=10)
#         error_text.pack(fill=tk.BOTH, expand=True, pady=5)
#         error_text.insert(tk.END, f"Error Details:\n{error_msg}")
#         error_text.config(state=tk.DISABLED)

#     def setup_ui(self):
#         print("🔍 Starting UI setup...")
        
#         # Define constants
#         self.CONTRACTOR = 'ZZZ'
#         self.IP_TYPE = 'MFL'
#         self.MEDIUM = 'Oil'
#         self.TYPE_PIPE = 'ZZZ'
#         self.GRADE_PIPE = 'ZZZ'

#         # Known values
#         self.DIA = 340
#         self.WT = 7.1

#         # To be set values
#         self.DP = 3.67
#         self.OP = 0
#         self.MAOP = 11
#         self.DF = 0.72
#         self.UTS = 413.686
#         self.SMYS = 2493.8

#         print("🔍 About to calculate statistics...")
#         self.calculate_statistics()
        
#         print("🔍 About to create main frame...")
#         self.create_main_frame()
        
#         print("🔍 About to create general info section...")
#         self.create_general_info_section()
        
#         print("🔍 About to create statistics section...")
#         self.create_statistics_section()
        
#         print("✅ All UI components created successfully!")

#     def calculate_statistics(self):
#         """Calculate statistics from the loaded DataFrame"""
#         print("🔍 Calculating statistics...")
        
#         if self.df.empty:
#             print("⚠️ DataFrame is empty, using default values")
#             self.TOT_ANAL = 0
#             self.INT_ANAL = 0
#             self.EXT_ANAL = 0
#             self.ERF_95 = 0
#             self.ERF_95_1 = 0
#             self.ERF_1 = 0
#             self.DEP_25 = 0
#             self.DEP_25_50 = 0
#             self.DEP_50_80 = 0
#             self.DEP_80_100 = 0
#             return

#         try:
#             # Total anomalies
#             self.TOT_ANAL = len(self.df)
#             print(f"📊 Total anomalies: {self.TOT_ANAL}")
            
#             # Internal/External anomalies (handle different column name variations)
#             type_col = None
#             for col in ['Type', 'Feature Type', 'Anomaly Type']:
#                 if col in self.df.columns:
#                     type_col = col
#                     break
            
#             if type_col:
#                 type_series = self.df[type_col].astype(str).str.lower()
#                 self.INT_ANAL = len(self.df[type_series.str.contains('internal', na=False)])
#                 self.EXT_ANAL = len(self.df[type_series.str.contains('external', na=False)])
#                 print(f"📊 Internal: {self.INT_ANAL}, External: {self.EXT_ANAL} (using column: {type_col})")
#             else:
#                 self.INT_ANAL = 0
#                 self.EXT_ANAL = 0
#                 print("⚠️ No Type column found")

#             # ERF statistics (handle different column name variations)
#             erf_col = None
#             for col in ['ERF (ASME B31G)', 'ERF', 'Engineering Risk Factor']:
#                 if col in self.df.columns:
#                     erf_col = col
#                     break
            
#             if erf_col:
#                 erf_data = pd.to_numeric(self.df[erf_col], errors='coerce').dropna()
#                 if len(erf_data) > 0:
#                     self.ERF_95 = len(erf_data[erf_data < 0.95])
#                     self.ERF_95_1 = len(erf_data[(erf_data >= 0.95) & (erf_data < 1)])
#                     self.ERF_1 = len(erf_data[erf_data >= 1])
#                     print(f"📊 ERF stats calculated from {len(erf_data)} valid values (using column: {erf_col})")
#                 else:
#                     self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
#                     print("⚠️ No valid ERF data found")
#             else:
#                 self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
#                 print("⚠️ No ERF column found")

#             # Depth statistics (handle different column name variations)
#             depth_col = None
#             for col in ['Depth %', 'Depth % ', 'Depth Percentage']:
#                 if col in self.df.columns:
#                     depth_col = col
#                     break
            
#             if depth_col:
#                 depth_data = pd.to_numeric(self.df[depth_col], errors='coerce').dropna()
#                 if len(depth_data) > 0:
#                     self.DEP_25 = len(depth_data[depth_data < 25])
#                     self.DEP_25_50 = len(depth_data[(depth_data >= 25) & (depth_data < 50)])
#                     self.DEP_50_80 = len(depth_data[(depth_data >= 50) & (depth_data < 80)])
#                     self.DEP_80_100 = len(depth_data[(depth_data >= 80) & (depth_data <= 100)])
#                     print(f"📊 Depth stats calculated from {len(depth_data)} valid values (using column: {depth_col})")
#                 else:
#                     self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0
#                     print("⚠️ No valid depth data found")
#             else:
#                 self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0
#                 print("⚠️ No Depth column found")

#             print("✅ Statistics calculation completed successfully")

#         except Exception as e:
#             print(f"❌ Error calculating statistics: {e}")
#             # Set default values on error
#             self.TOT_ANAL = len(self.df) if not self.df.empty else 0
#             self.INT_ANAL = self.EXT_ANAL = 0
#             self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
#             self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0

#     def create_main_frame(self):
#         print("🔍 Creating main frame...")
#         self.master.title("Pipeline Highlights")
#         self.master.geometry("500x500+1100+75")
#         self.master.minsize(500, 500)
#         self.master.maxsize(1400, 1000)

#         self.main_frame = ttk.Frame(self.master)
#         self.main_frame.grid(sticky="nsew", padx=20, pady=20)

#         # Configure grid weights
#         self.master.grid_rowconfigure(0, weight=1)
#         self.master.grid_columnconfigure(0, weight=1)

#         self.main_frame.grid_rowconfigure(1, weight=1)
#         self.main_frame.grid_columnconfigure(0, weight=1)
        
#         print("✅ Main frame created and configured")

#     def create_general_info_section(self):
#         print("🔍 Creating general info section...")
        
#         general_info_frame = ttk.LabelFrame(self.main_frame, text="General Info", padding=(10, 5))
#         general_info_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

#         # Configure grid for 6 columns (3 pairs of label-entry)
#         for i in range(6):
#             general_info_frame.grid_columnconfigure(i, weight=1)

#         left_general_info = [
#             ("Contractor", self.CONTRACTOR),
#             ("IP Type", self.IP_TYPE),
#             ("Medium", self.MEDIUM),
#             ("Type of Pipe", self.TYPE_PIPE),
#             ("Grade of Pipe", self.GRADE_PIPE),
#         ]

#         mid_general_info = [
#             ("Diameter (mm)", self.DIA),
#             ("Wall Thickness (mm)", self.WT),
#             ("Design Pressure (MPa)", self.DP),
#             ("Operating Pressure (MPa)", self.OP),
#         ]

#         right_general_info = [
#             ("MAOP (MPa)", self.MAOP),
#             ("Design Factor", self.DF),
#             ("UTS (MPa)", self.UTS),
#             ("SMYS (MPa)", self.SMYS)
#         ]

#         # Create left column
#         for i, (key, value) in enumerate(left_general_info):
#             label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
#             label.grid(row=i, column=0, sticky='w', padx=2, pady=2)

#             entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
#             entry.insert(0, str(value))
#             entry.configure(state='readonly')
#             entry.grid(row=i, column=1, sticky='w', padx=2, pady=2)

#         # Create middle column
#         for i, (key, value) in enumerate(mid_general_info):
#             label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
#             label.grid(row=i, column=2, sticky='w', padx=2, pady=2)

#             entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
#             entry.insert(0, str(value))
#             entry.configure(state='readonly')
#             entry.grid(row=i, column=3, sticky='w', padx=2, pady=2)

#         # Create right column
#         for i, (key, value) in enumerate(right_general_info):
#             label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
#             label.grid(row=i, column=4, sticky='w', padx=2, pady=2)

#             entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
#             entry.insert(0, str(value))
#             entry.configure(state='readonly')
#             entry.grid(row=i, column=5, sticky='w', padx=2, pady=2)

#         print("✅ General info section created")

#     def create_statistics_section(self):
#         print("🔍 Creating statistics section...")
        
#         stats_frame = ttk.Frame(self.main_frame)
#         stats_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

#         stats_frame.grid_columnconfigure(0, weight=1)
#         stats_frame.grid_columnconfigure(1, weight=1)
#         stats_frame.grid_rowconfigure(0, weight=1)

#         # Statistics labels frame
#         stats_labels_frame = ttk.LabelFrame(stats_frame, text="Statistics", padding=(10, 5))
#         stats_labels_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

#         # Charts frame
#         charts_frame = ttk.Frame(stats_frame)
#         charts_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
#         charts_frame.grid_rowconfigure(0, weight=1)

#         # Statistics data
#         stats_data = [
#             ("Total Anomalies Count", self.TOT_ANAL),
#             ("Int. Anomalies Count", self.INT_ANAL),
#             ("Ext. Anomalies Count", self.EXT_ANAL),
#             ("", ""),  # Spacer
#             ("ERF < 0.95", self.ERF_95),
#             ("0.95 <= ERF < 1", self.ERF_95_1),
#             ("1 <= ERF", self.ERF_1),
#             ("", ""),  # Spacer
#             ("Depth < 25%", self.DEP_25),
#             ("25% <= Depth < 50%", self.DEP_25_50),
#             ("50% <= Depth < 80%", self.DEP_50_80),
#             ("80% <= Depth <= 100%", self.DEP_80_100),
#         ]

#         # Create statistics labels and entries
#         for i, (key, value) in enumerate(stats_data):
#             if key == "":
#                 # Spacer
#                 spacer_label = ttk.Label(stats_labels_frame, text="", font=("Helvetica", 16))
#                 spacer_label.grid(row=i, column=0, sticky='w', padx=5, pady=5)
#                 continue

#             # Label
#             label = ttk.Label(stats_labels_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
#             label.grid(row=i, column=0, sticky='w', padx=5, pady=5)

#             # Entry
#             entry = ttk.Entry(stats_labels_frame, width=10, font=("Helvetica", 16))
#             entry.insert(0, str(value))
#             entry.configure(state='readonly')
#             entry.grid(row=i, column=1, sticky='w', padx=5, pady=5)

#         # ✅ FIXED: Create pie charts without setting backend
#         try:
#             print("🔍 Creating pie charts...")
            
#             # ❌ REMOVED: matplotlib.use('TkAgg') - this was causing the error
#             # ✅ Let matplotlib use whatever backend is already running
            
#             # Clear any previous figures to avoid conflicts
#             plt.clf()
            
#             fig, axs = plt.subplots(3, 1, figsize=(3, 5))

#             def autopct_func(pct):
#                 return f'{pct:.1f}%' if pct >= 4 else ''

#             # Pie chart for Internal/External Anomalies
#             labels = ['Internal ML', 'External ML']
#             sizes = [self.INT_ANAL, self.EXT_ANAL]
#             colors = ['red', 'blue']
            
#             if sum(sizes) > 0:
#                 axs[0].pie(sizes, colors=colors, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
#             else:
#                 axs[0].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[0].transAxes)
#             axs[0].axis('equal')
#             axs[0].legend(labels, loc='lower right', frameon=False, fontsize=3)

#             # Pie chart for ERF
#             labels_erf = ['ERF < 0.95', '0.95 <= ERF < 1', '1 <= ERF']
#             sizes_erf = [self.ERF_95, self.ERF_95_1, self.ERF_1]
#             colors_erf = ['green', 'yellow', 'red']
            
#             if sum(sizes_erf) > 0:
#                 axs[1].pie(sizes_erf, colors=colors_erf, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
#             else:
#                 axs[1].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[1].transAxes)
#             axs[1].axis('equal')
#             axs[1].legend(labels_erf, loc='lower right', frameon=False, fontsize=3)

#             # Pie chart for Depths
#             labels_depth = ['Depth < 25%', '25% <= Depth < 50%', '50% <= Depth < 80%', '80% <= Depth <= 100%']
#             sizes_depth = [self.DEP_25, self.DEP_25_50, self.DEP_50_80, self.DEP_80_100]
#             colors_depth = ['purple', 'green', 'yellow', 'brown']
            
#             if sum(sizes_depth) > 0:
#                 axs[2].pie(sizes_depth, colors=colors_depth, autopct=autopct_func, startangle=90, textprops={'fontsize': 4})
#             else:
#                 axs[2].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[2].transAxes)
#             axs[2].axis('equal')
#             axs[2].legend(labels_depth, loc='lower right', frameon=False, fontsize=3)

#             # Adjust layout
#             plt.tight_layout()

#             # Display the pie charts in the charts frame
#             canvas = FigureCanvasTkAgg(fig, master=charts_frame)
#             canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#             canvas.draw()

#             print("✅ Pie charts created successfully")

#         except Exception as e:
#             print(f"❌ Error creating pie charts: {e}")
#             import traceback
#             traceback.print_exc()
            
#             # Create a simple fallback display with statistics
#             fallback_frame = ttk.Frame(charts_frame)
#             fallback_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            
#             fallback_title = ttk.Label(fallback_frame, text="Statistics Summary", 
#                                      font=("Helvetica", 14, "bold"))
#             fallback_title.pack(pady=5)
            
#             # Show key statistics as text
#             stats_text = f"""Total Anomalies: {self.TOT_ANAL}
# Internal: {self.INT_ANAL} | External: {self.EXT_ANAL}

# ERF Distribution:
# • < 0.95: {self.ERF_95}
# • 0.95-1.0: {self.ERF_95_1}  
# • ≥ 1.0: {self.ERF_1}

# Depth Distribution:
# • < 25%: {self.DEP_25}
# • 25-50%: {self.DEP_25_50}
# • 50-80%: {self.DEP_50_80}
# • 80-100%: {self.DEP_80_100}"""
            
#             fallback_label = ttk.Label(fallback_frame, text=stats_text, 
#                                      font=("Courier", 10), justify='left')
#             fallback_label.pack(pady=10)

#         # Configure styles for LabelFrames
#         try:
#             style = ttk.Style()
#             style.configure('Custom.TLabelframe.Label', font=("Helvetica", 20, "bold"), foreground='maroon')
#             style.configure('Custom.TLabelframe', background='white')
#         except Exception as e:
#             print(f"⚠️ Warning: Could not configure styles: {e}")

#         print("✅ Statistics section created")

# def run_app(pipe_tally_df=None):
#     """Main function to run the Pipeline Highlights application"""
#     print("🚀 Starting Pipeline Highlights App...")
    
#     try:
#         root = tk.Tk()
#         app = PipeHighlightApp(root, pipe_tally_df=pipe_tally_df)
#         print("🎯 Starting mainloop...")
#         root.mainloop()
#         print("✅ App closed normally")
#     except Exception as e:
#         print(f"❌ Critical error in run_app: {e}")
#         import traceback
#         traceback.print_exc()

# # Debug version for testing
# def run_app_debug(pipe_tally_df=None):
#     """Debug version with minimal UI for testing"""
#     print("🚀 Starting DEBUG Pipeline Highlights App...")
    
#     root = tk.Tk()
#     root.title("Pipeline Highlights - DEBUG")
#     root.geometry("400x300")
    
#     main_frame = ttk.Frame(root)
#     main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
#     # Test label
#     test_label = ttk.Label(main_frame, text="Pipeline Highlights - DEBUG MODE", font=("Arial", 16))
#     test_label.pack(pady=10)
    
#     # Data info
#     if pipe_tally_df is not None and not pipe_tally_df.empty:
#         data_label = ttk.Label(main_frame, text=f"✅ Data loaded: {len(pipe_tally_df)} rows", font=("Arial", 12))
#         data_label.pack(pady=5)
        
#         cols_label = ttk.Label(main_frame, text=f"📊 Columns: {len(pipe_tally_df.columns)}", font=("Arial", 10))
#         cols_label.pack(pady=5)
        
#         # Show some column names
#         col_names = ', '.join(list(pipe_tally_df.columns)[:5])
#         if len(pipe_tally_df.columns) > 5:
#             col_names += "..."
#         cols_detail = ttk.Label(main_frame, text=f"Sample columns: {col_names}", font=("Arial", 8))
#         cols_detail.pack(pady=5)
#     else:
#         data_label = ttk.Label(main_frame, text="❌ No data loaded", font=("Arial", 12))
#         data_label.pack(pady=5)
    
#     # Test button
#     def test_button_click():
#         print("✅ Button clicked - UI is working!")
#         if pipe_tally_df is not None:
#             print(f"Data shape: {pipe_tally_df.shape}")
#             print(f"Columns: {list(pipe_tally_df.columns)}")
    
#     test_button = ttk.Button(main_frame, text="Test Data Access", command=test_button_click)
#     test_button.pack(pady=10)
    
#     # Launch full app button
#     def launch_full_app():
#         root.destroy()
#         run_app(pipe_tally_df)
    
#     full_app_button = ttk.Button(main_frame, text="Launch Full App", command=launch_full_app)
#     full_app_button.pack(pady=5)
    
#     print("🎯 Starting debug mainloop...")
#     root.mainloop()
#     print("✅ Debug app closed normally")

# if __name__ == "__main__":
#     run_app()


# ✅ Set backend at module level (optional, can be removed)
import matplotlib
# matplotlib.use('Qt5Agg')  # Optional: Use Qt-compatible backend

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
    def __init__(self, master, pipe_tally_df=None):
        print("🔍 Initializing PipeHighlightApp...")
        self.master = master
        
        # ✅ Use provided DataFrame or fallback to file loading
        if pipe_tally_df is not None and not pipe_tally_df.empty:
            self.df = pipe_tally_df.copy()  # Make a copy to avoid modifying original
            print(f"✅ Using loaded pipe tally data from project ({len(self.df)} rows)")
            print(f"📊 Available columns: {list(self.df.columns)}")
        else:
            print("⚠️ No DataFrame provided, trying fallback file...")
            try:
                file_path = resource_path("backend/files/datalog/ptal.xlsx")
                if os.path.exists(file_path):
                    self.df = pd.read_excel(file_path)
                    print("⚠️ Loading pipe tally from fallback file")
                else:
                    print(f"❌ Fallback file not found: {file_path}")
                    self.df = pd.DataFrame()
            except Exception as e:
                print(f"❌ Error loading fallback file: {e}")
                self.df = pd.DataFrame()
        
        print("🔍 About to setup UI...")
        try:
            self.setup_ui()
            print("✅ UI setup completed successfully!")
        except Exception as e:
            print(f"❌ Error during UI setup: {e}")
            import traceback
            traceback.print_exc()
            # Create minimal UI on error
            self.create_error_ui(str(e))

    def create_error_ui(self, error_msg):
        """Create a simple error display UI"""
        self.master.title("Pipeline Highlights - Error")
        self.master.geometry("500x300")
        
        error_frame = ttk.Frame(self.master)
        error_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        error_label = ttk.Label(error_frame, text="Error Loading Pipeline Highlights", 
                               font=("Arial", 16, "bold"), foreground="red")
        error_label.pack(pady=10)
        
        error_text = tk.Text(error_frame, wrap=tk.WORD, height=10)
        error_text.pack(fill=tk.BOTH, expand=True, pady=5)
        error_text.insert(tk.END, f"Error Details:\n{error_msg}")
        error_text.config(state=tk.DISABLED)

    def setup_ui(self):
        print("🔍 Starting UI setup...")
        
        # Define constants
        self.CONTRACTOR = 'ZZZ'
        self.IP_TYPE = 'MFL'
        self.MEDIUM = 'Oil'
        self.TYPE_PIPE = 'ZZZ'
        self.GRADE_PIPE = 'ZZZ'

        # Known values
        self.DIA = 340
        self.WT = 7.1

        # To be set values
        self.DP = 3.67
        self.OP = 0
        self.MAOP = 11
        self.DF = 0.72
        self.UTS = 413.686
        self.SMYS = 2493.8

        print("🔍 About to calculate statistics...")
        self.calculate_statistics()
        
        print("🔍 About to create main frame...")
        self.create_main_frame()
        
        print("🔍 About to create general info section...")
        self.create_general_info_section()
        
        print("🔍 About to create statistics section...")
        self.create_statistics_section()
        
        print("✅ All UI components created successfully!")

    def calculate_statistics(self):
        """Calculate statistics from the loaded DataFrame"""
        print("🔍 Calculating statistics...")
        
        if self.df.empty:
            print("⚠️ DataFrame is empty, using default values")
            self.TOT_ANAL = 0
            self.INT_ANAL = 0
            self.EXT_ANAL = 0
            self.ERF_95 = 0
            self.ERF_95_1 = 0
            self.ERF_1 = 0
            self.DEP_25 = 0
            self.DEP_25_50 = 0
            self.DEP_50_80 = 0
            self.DEP_80_100 = 0
            return

        try:
            # Total anomalies
            self.TOT_ANAL = len(self.df)
            print(f"📊 Total anomalies: {self.TOT_ANAL}")
            
            # Internal/External anomalies (handle different column name variations)
            type_col = None
            for col in ['Type', 'Feature Type', 'Anomaly Type']:
                if col in self.df.columns:
                    type_col = col
                    break
            
            if type_col:
                type_series = self.df[type_col].astype(str).str.lower()
                self.INT_ANAL = len(self.df[type_series.str.contains('internal', na=False)])
                self.EXT_ANAL = len(self.df[type_series.str.contains('external', na=False)])
                print(f"📊 Internal: {self.INT_ANAL}, External: {self.EXT_ANAL} (using column: {type_col})")
            else:
                self.INT_ANAL = 0
                self.EXT_ANAL = 0
                print("⚠️ No Type column found")

            # ERF statistics (handle different column name variations)
            erf_col = None
            for col in ['ERF (ASME B31G)', 'ERF', 'Engineering Risk Factor']:
                if col in self.df.columns:
                    erf_col = col
                    break
            
            if erf_col:
                erf_data = pd.to_numeric(self.df[erf_col], errors='coerce').dropna()
                if len(erf_data) > 0:
                    self.ERF_95 = len(erf_data[erf_data < 0.95])
                    self.ERF_95_1 = len(erf_data[(erf_data >= 0.95) & (erf_data < 1)])
                    self.ERF_1 = len(erf_data[erf_data >= 1])
                    print(f"📊 ERF stats calculated from {len(erf_data)} valid values (using column: {erf_col})")
                else:
                    self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
                    print("⚠️ No valid ERF data found")
            else:
                self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
                print("⚠️ No ERF column found")

            # Depth statistics (handle different column name variations)
            depth_col = None
            for col in ['Depth %', 'Depth % ', 'Depth Percentage']:
                if col in self.df.columns:
                    depth_col = col
                    break
            
            if depth_col:
                depth_data = pd.to_numeric(self.df[depth_col], errors='coerce').dropna()
                if len(depth_data) > 0:
                    self.DEP_25 = len(depth_data[depth_data < 25])
                    self.DEP_25_50 = len(depth_data[(depth_data >= 25) & (depth_data < 50)])
                    self.DEP_50_80 = len(depth_data[(depth_data >= 50) & (depth_data < 80)])
                    self.DEP_80_100 = len(depth_data[(depth_data >= 80) & (depth_data <= 100)])
                    print(f"📊 Depth stats calculated from {len(depth_data)} valid values (using column: {depth_col})")
                else:
                    self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0
                    print("⚠️ No valid depth data found")
            else:
                self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0
                print("⚠️ No Depth column found")

            print("✅ Statistics calculation completed successfully")

        except Exception as e:
            print(f"❌ Error calculating statistics: {e}")
            # Set default values on error
            self.TOT_ANAL = len(self.df) if not self.df.empty else 0
            self.INT_ANAL = self.EXT_ANAL = 0
            self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
            self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0

    def create_main_frame(self):
        print("🔍 Creating main frame...")
        self.master.title("Pipeline Highlights")
        # ✅ MUCH LARGER WINDOW: Increased from 500x500 to 1200x900
        self.master.geometry("1200x900+100+50")
        self.master.minsize(800, 600)
        self.master.maxsize(1600, 1200)

        self.main_frame = ttk.Frame(self.master)
        self.main_frame.grid(sticky="nsew", padx=20, pady=20)

        # Configure grid weights
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        print("✅ Main frame created and configured")

    def create_general_info_section(self):
        print("🔍 Creating general info section...")
        
        general_info_frame = ttk.LabelFrame(self.main_frame, text="General Info", padding=(10, 5))
        general_info_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Configure grid for 6 columns (3 pairs of label-entry)
        for i in range(6):
            general_info_frame.grid_columnconfigure(i, weight=1)

        left_general_info = [
            ("Contractor", self.CONTRACTOR),
            ("IP Type", self.IP_TYPE),
            ("Medium", self.MEDIUM),
            ("Type of Pipe", self.TYPE_PIPE),
            ("Grade of Pipe", self.GRADE_PIPE),
        ]

        mid_general_info = [
            ("Diameter (mm)", self.DIA),
            ("Wall Thickness (mm)", self.WT),
            ("Design Pressure (MPa)", self.DP),
            ("Operating Pressure (MPa)", self.OP),
        ]

        right_general_info = [
            ("MAOP (MPa)", self.MAOP),
            ("Design Factor", self.DF),
            ("UTS (MPa)", self.UTS),
            ("SMYS (MPa)", self.SMYS)
        ]

        # Create left column
        for i, (key, value) in enumerate(left_general_info):
            label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
            label.grid(row=i, column=0, sticky='w', padx=2, pady=2)

            entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
            entry.insert(0, str(value))
            entry.configure(state='readonly')
            entry.grid(row=i, column=1, sticky='w', padx=2, pady=2)

        # Create middle column
        for i, (key, value) in enumerate(mid_general_info):
            label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
            label.grid(row=i, column=2, sticky='w', padx=2, pady=2)

            entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
            entry.insert(0, str(value))
            entry.configure(state='readonly')
            entry.grid(row=i, column=3, sticky='w', padx=2, pady=2)

        # Create right column
        for i, (key, value) in enumerate(right_general_info):
            label = ttk.Label(general_info_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
            label.grid(row=i, column=4, sticky='w', padx=2, pady=2)

            entry = ttk.Entry(general_info_frame, width=15, font=("Helvetica", 16))
            entry.insert(0, str(value))
            entry.configure(state='readonly')
            entry.grid(row=i, column=5, sticky='w', padx=2, pady=2)

        print("✅ General info section created")

    def create_statistics_section(self):
        print("🔍 Creating statistics section...")
        
        stats_frame = ttk.Frame(self.main_frame)
        stats_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=2)  # ✅ Give more space to charts
        stats_frame.grid_rowconfigure(0, weight=1)

        # Statistics labels frame
        stats_labels_frame = ttk.LabelFrame(stats_frame, text="Statistics", padding=(10, 5))
        stats_labels_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Charts frame
        charts_frame = ttk.Frame(stats_frame)
        charts_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        charts_frame.grid_rowconfigure(0, weight=1)

        # Statistics data
        stats_data = [
            ("Total Anomalies Count", self.TOT_ANAL),
            ("Int. Anomalies Count", self.INT_ANAL),
            ("Ext. Anomalies Count", self.EXT_ANAL),
            ("", ""),  # Spacer
            ("ERF < 0.95", self.ERF_95),
            ("0.95 <= ERF < 1", self.ERF_95_1),
            ("1 <= ERF", self.ERF_1),
            ("", ""),  # Spacer
            ("Depth < 25%", self.DEP_25),
            ("25% <= Depth < 50%", self.DEP_25_50),
            ("50% <= Depth < 80%", self.DEP_50_80),
            ("80% <= Depth <= 100%", self.DEP_80_100),
        ]

        # Create statistics labels and entries
        for i, (key, value) in enumerate(stats_data):
            if key == "":
                spacer_label = ttk.Label(stats_labels_frame, text="", font=("Helvetica", 16))
                spacer_label.grid(row=i, column=0, sticky='w', padx=5, pady=5)
                continue

            label = ttk.Label(stats_labels_frame, text=f"{key}:", font=("Helvetica", 16), anchor='w')
            label.grid(row=i, column=0, sticky='w', padx=5, pady=5)

            entry = ttk.Entry(stats_labels_frame, width=10, font=("Helvetica", 16))
            entry.insert(0, str(value))
            entry.configure(state='readonly')
            entry.grid(row=i, column=1, sticky='w', padx=5, pady=5)

        # ✅ MUCH LARGER PIE CHARTS - REMOVED matplotlib.use('TkAgg')
        try:
            print("🔍 Creating LARGE pie charts...")
            
            # Clear any previous figures
            plt.clf()
            
            # ✅ INCREASED SIZE: Much larger figure (was 3x5, now 8x12)
            fig, axs = plt.subplots(3, 1, figsize=(8, 12))

            def autopct_func(pct):
                return f'{pct:.1f}%' if pct >= 4 else ''

            # Pie chart for Internal/External Anomalies
            labels = ['Internal ML', 'External ML']
            sizes = [self.INT_ANAL, self.EXT_ANAL]
            colors = ['red', 'blue']
            
            if sum(sizes) > 0:
                # ✅ LARGER TEXT: Increased font sizes
                wedges, texts, autotexts = axs[0].pie(
                    sizes, 
                    colors=colors, 
                    autopct=autopct_func, 
                    startangle=90, 
                    textprops={'fontsize': 12}  # ✅ Increased from 4 to 12
                )
                # ✅ Make percentage text bold and larger
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(14)
            else:
                axs[0].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=axs[0].transAxes, fontsize=16)
            
            axs[0].axis('equal')
            axs[0].legend(labels, loc='upper right', frameon=True, fontsize=10)
            axs[0].set_title('Internal vs External Anomalies', fontsize=16, fontweight='bold', pad=10)

            # Pie chart for ERF
            labels_erf = ['ERF < 0.95', '0.95 <= ERF < 1', '1 <= ERF']
            sizes_erf = [self.ERF_95, self.ERF_95_1, self.ERF_1]
            colors_erf = ['green', 'yellow', 'red']
            
            if sum(sizes_erf) > 0:
                wedges, texts, autotexts = axs[1].pie(
                    sizes_erf, 
                    colors=colors_erf, 
                    autopct=autopct_func, 
                    startangle=90, 
                    textprops={'fontsize': 12}
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(14)
            else:
                axs[1].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=axs[1].transAxes, fontsize=16)
            
            axs[1].axis('equal')
            axs[1].legend(labels_erf, loc='upper right', frameon=True, fontsize=10)
            axs[1].set_title('ERF Distribution', fontsize=16, fontweight='bold', pad=0)

            # Pie chart for Depths
            labels_depth = ['Depth < 25%', '25% <= Depth < 50%', '50% <= Depth < 80%', '80% <= Depth <= 100%']
            sizes_depth = [self.DEP_25, self.DEP_25_50, self.DEP_50_80, self.DEP_80_100]
            colors_depth = ['purple', 'green', 'yellow', 'brown']
            
            if sum(sizes_depth) > 0:
                wedges, texts, autotexts = axs[2].pie(
                    sizes_depth, 
                    colors=colors_depth, 
                    autopct=autopct_func, 
                    startangle=90, 
                    textprops={'fontsize': 12}
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(14)
            else:
                axs[2].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=axs[2].transAxes, fontsize=16)
            
            axs[2].axis('equal')
            axs[2].legend(labels_depth, loc='upper right', frameon=True, fontsize=10)
            axs[2].set_title('Depth Distribution', fontsize=16, fontweight='bold', pad=0)

            # ✅ BETTER LAYOUT: Adjust spacing between subplots
            plt.tight_layout(pad=3.0)

            # Display the pie charts in the charts frame
            canvas = FigureCanvasTkAgg(fig, master=charts_frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            canvas.draw()

            print("✅ LARGE pie charts created successfully")

        except Exception as e:
            print(f"❌ Error creating pie charts: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback display with larger text
            fallback_frame = ttk.Frame(charts_frame)
            fallback_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            
            fallback_title = ttk.Label(fallback_frame, text="Statistics Summary", 
                                     font=("Helvetica", 18, "bold"))
            fallback_title.pack(pady=10)
            
            stats_text = f"""Total Anomalies: {self.TOT_ANAL}
Internal: {self.INT_ANAL} | External: {self.EXT_ANAL}

ERF Distribution:
• < 0.95: {self.ERF_95}
• 0.95-1.0: {self.ERF_95_1}  
• ≥ 1.0: {self.ERF_1}

Depth Distribution:
• < 25%: {self.DEP_25}
• 25-50%: {self.DEP_25_50}
• 50-80%: {self.DEP_50_80}
• 80-100%: {self.DEP_80_100}"""
            
            fallback_label = ttk.Label(fallback_frame, text=stats_text, 
                                     font=("Courier", 14), justify='left')
            fallback_label.pack(pady=10)

        print("✅ Statistics section created")

def run_app(pipe_tally_df=None):
    """Main function to run the Pipeline Highlights application"""
    print("🚀 Starting Pipeline Highlights App...")
    
    try:
        root = tk.Tk()
        app = PipeHighlightApp(root, pipe_tally_df=pipe_tally_df)
        print("🎯 Starting mainloop...")
        root.mainloop()
        print("✅ App closed normally")
    except Exception as e:
        print(f"❌ Critical error in run_app: {e}")
        import traceback
        traceback.print_exc()

# Debug version for testing
def run_app_debug(pipe_tally_df=None):
    """Debug version with minimal UI for testing"""
    print("🚀 Starting DEBUG Pipeline Highlights App...")
    
    root = tk.Tk()
    root.title("Pipeline Highlights - DEBUG")
    root.geometry("400x300")
    
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Test label
    test_label = ttk.Label(main_frame, text="Pipeline Highlights - DEBUG MODE", font=("Arial", 16))
    test_label.pack(pady=10)
    
    # Data info
    if pipe_tally_df is not None and not pipe_tally_df.empty:
        data_label = ttk.Label(main_frame, text=f"✅ Data loaded: {len(pipe_tally_df)} rows", font=("Arial", 12))
        data_label.pack(pady=5)
        
        cols_label = ttk.Label(main_frame, text=f"📊 Columns: {len(pipe_tally_df.columns)}", font=("Arial", 10))
        cols_label.pack(pady=5)
        
        # Show some column names
        col_names = ', '.join(list(pipe_tally_df.columns)[:5])
        if len(pipe_tally_df.columns) > 5:
            col_names += "..."
        cols_detail = ttk.Label(main_frame, text=f"Sample columns: {col_names}", font=("Arial", 8))
        cols_detail.pack(pady=5)
    else:
        data_label = ttk.Label(main_frame, text="❌ No data loaded", font=("Arial", 12))
        data_label.pack(pady=5)
    
    # Test button
    def test_button_click():
        print("✅ Button clicked - UI is working!")
        if pipe_tally_df is not None:
            print(f"Data shape: {pipe_tally_df.shape}")
            print(f"Columns: {list(pipe_tally_df.columns)}")
    
    test_button = ttk.Button(main_frame, text="Test Data Access", command=test_button_click)
    test_button.pack(pady=10)
    
    # Launch full app button
    def launch_full_app():
        root.destroy()
        run_app(pipe_tally_df)
    
    full_app_button = ttk.Button(main_frame, text="Launch Full App", command=launch_full_app)
    full_app_button.pack(pady=5)
    
    print("🎯 Starting debug mainloop...")
    root.mainloop()
    print("✅ Debug app closed normally")

if __name__ == "__main__":
    run_app()
