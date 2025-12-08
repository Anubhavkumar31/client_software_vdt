import os
import re
import sys
import hashlib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# from pages.Pipe_Highlights import run_app
# Optional modern theming
_USING_BOOTSTRAP = False
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
    _USING_BOOTSTRAP = True
except Exception:
    pass

from PIL import Image, ImageTk, ImageGrab
import pandas as pd
from fpdf import FPDF
import mysql.connector

from utils import resource_path


# --------------------------- Small Utilities -----------------------------

def safe_parse_time(clock_orientation):
    """Safely parse time-like values from strings / Timestamp / datetime into (hour, minute)."""
    if isinstance(clock_orientation, pd.Timestamp):
        return clock_orientation.hour, clock_orientation.minute
    if hasattr(clock_orientation, "hour") and hasattr(clock_orientation, "minute"):
        return clock_orientation.hour, clock_orientation.minute
    if isinstance(clock_orientation, str):
        try:
            s = clock_orientation.strip()
            s = s.replace("o' clock", "o clock").replace("oâ€™ clock", "o clock")
            if ":" in s:
                h, m = s.split(":")[0:2]
                return int(float(h)), int(float(m))
            m = re.search(r"(\d+)", s)
            if m:
                return int(float(m.group(1))), 0
        except Exception:
            pass
    return 0, 0

def md5_image(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()

def info(msg: str):
    print(msg)

# --------------------------- Main App Class -----------------------------

class PipelineApp:
    def __init__(self, root, pipe_tally=None):
        self.root = root
        self.pipe_tally = pipe_tally
        print(self.pipe_tally)
        self.root.title("Pipeline Scheme Report & Pipe Number Visualizer")
        try:
            self.root.iconbitmap(resource_path('pipeline_schema/LOGO-withoutbg.ico'))
        except Exception:
            pass

        self.root.geometry("1200x800+150+60")
        self.root.minsize(1100, 700)

        # DB
        self.conn = None
        self.cursor = None

        # Data
        self.data = pd.DataFrame()
        self.chunks = []          # list[pd.DataFrame] grouped by 'Pipe Number'
        self.current_page = 0     # slot paging (0-based)
        self.chunks_per_page = 100

        # Canvas state
        self.canvas_scale = 1.0
        self._tooltip_win = None
        self._pan_start = None
        self.pipe_item_map = {}

        # Styling
        self._init_style()

        # Layout
        self._build_top_toolbar()
        self._build_body()
        self._build_statusbar()

        # Branding
        self._load_logo()

        # DB connect
        self._connect_db()

        # Excel load (robust)
        self._read_excel()

        # Populate & render
        self._make_chunks()
        self._populate_slot_menu()
        self.display_page(0)
        self._update_status("Ready.")

    # ------------------------ Style & Layout ----------------------------

    def _init_style(self):
        if not _USING_BOOTSTRAP:
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure("Toolbar.TFrame", background="#f7f7fa")
            style.configure("Toolbar.TButton", padding=6)
            style.configure("Status.TFrame", background="#f1f1f4")
            style.configure("Status.TLabel", background="#f1f1f4")
            style.configure("Card.TLabelframe", background="white")
            style.configure("Card.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
            style.configure("Card.TFrame", background="white")

    def _build_top_toolbar(self):
        self.toolbar = ttk.Frame(self.root, style="Toolbar.TFrame")
        self.toolbar.pack(side="top", fill="x")

        # ttk.Label(self.toolbar, text="Run ID:").pack(side="left", padx=(10, 5), pady=8)
        # self.run_id_var = tk.StringVar()
        # ttk.Entry(self.toolbar, textvariable=self.run_id_var, width=18).pack(side="left", padx=(0, 8), pady=8)

        # ttk.Button(self.toolbar, text="Fetch", command=self.get_data, style="Toolbar.TButton").pack(side="left", padx=(0, 12), pady=8)

        ttk.Label(self.toolbar, text="Slot:").pack(side="left", padx=(4,4))
        self.slot_var = tk.StringVar()
        self.slot_menu = ttk.Combobox(self.toolbar, textvariable=self.slot_var, state="readonly", width=16)
        self.slot_menu.bind("<<ComboboxSelected>>", self._on_slot_select)
        self.slot_menu.pack(side="left", padx=(0, 12), pady=8)

        # ttk.Button(self.toolbar, text="Reset Zoom", command=self._reset_zoom, style="Toolbar.TButton").pack(side="left", padx=(0, 6))
        # ttk.Button(self.toolbar, text="Reload Excel", command=self._reload_excel, style="Toolbar.TButton").pack(side="left", padx=(0, 6))
        # ttk.Button(self.toolbar, text="Export PDF", command=self.save_as_pdf, style="Toolbar.TButton").pack(side="left", padx=(0, 6))

        ttk.Label(self.toolbar, text="").pack(side="left", expand=True)
        if _USING_BOOTSTRAP:
            ttk.Label(self.toolbar, text="Theme via ttkbootstrap").pack(side="right", padx=10)

    def _build_body(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # Project Info
        # self.tab_info = ttk.Frame(self.notebook)
        # self.notebook.add(self.tab_info, text="Project Info")

        # info_frame = ttk.LabelFrame(self.tab_info, text="Project Details", style="Card.TLabelframe")
        # info_frame.pack(fill="x", padx=16, pady=(16, 8))

        # self.client_var = tk.StringVar()
        # self.pipeline_name_var = tk.StringVar()
        # self.report_date_var = tk.StringVar()
        # self._grid_form_row(info_frame, 0, "Client:", self.client_var)
        # self._grid_form_row(info_frame, 1, "Pipeline Name:", self.pipeline_name_var)
        # self._grid_form_row(info_frame, 2, "Report Date:", self.report_date_var)

        # Visualize tab
        self.tab_viz = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_viz, text="Visualize")
        self.tab_viz.columnconfigure(1, weight=1)
        self.tab_viz.rowconfigure(0, weight=3)
        self.tab_viz.rowconfigure(1, weight=3)

        # Canvas area
        mid = ttk.Frame(self.tab_viz)
        mid.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=(12, 6))
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        self.main_canvas = tk.Canvas(mid, bg="white", highlightthickness=0)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.main_canvas.yview); yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(mid, orient=tk.HORIZONTAL, command=self.main_canvas.xview); xscroll.grid(row=1, column=0, sticky="ew")
        self.main_canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.canvas_frame = ttk.Frame(self.main_canvas, style="Card.TFrame")
        self.canvas_window = self.main_canvas.create_window((0,0), window=self.canvas_frame, anchor="nw")
        self.canvas_frame.bind("<Configure>", lambda e: self._update_canvas_scrollregion())

        self.pipe_canvas = tk.Canvas(self.canvas_frame, width=1800, height=1300, bg="white", highlightthickness=1, highlightbackground="#e6e6eb")
        self.pipe_canvas.pack(padx=12, pady=12)

        # Canvas interactions
        self.pipe_canvas.bind("<MouseWheel>", self._on_mousewheel_zoom)     # Windows
        self.pipe_canvas.bind("<Button-4>", self._on_mousewheel_zoom)       # Linux
        self.pipe_canvas.bind("<Button-5>", self._on_mousewheel_zoom)       # Linux
        self.pipe_canvas.bind("<ButtonPress-2>", self._start_pan)
        self.pipe_canvas.bind("<B2-Motion>", self._on_pan)
        self.pipe_canvas.bind("<ButtonRelease-2>", self._end_pan)
        self.pipe_canvas.bind("<Motion>", self._on_canvas_hover)
        self.pipe_canvas.bind("<Leave>", lambda e: self._hide_tooltip())

        # Bottom table
        bottom = ttk.Frame(self.tab_viz)
        bottom.grid(row=1, column=1, sticky="nsew", padx=(0, 12), pady=(0, 12))
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(bottom, columns=("Joint","Length","WT","Feature","Depth%","Type","Clock","DistUS(m)"), show="headings", height=8)
        for col, w in [("Joint",80),("Length",80),("WT",60),("Feature",140),("Depth%",80),("Type",80),("Clock",100),("DistUS(m)",90)]:
            self.tree.heading(col, text=col); self.tree.column(col, width=w, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(bottom, orient=tk.VERTICAL, command=self.tree.yview); tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscroll=tree_scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Export tab
        # self.tab_export = ttk.Frame(self.notebook)
        # self.notebook.add(self.tab_export, text="Export")
        # exf = ttk.LabelFrame(self.tab_export, text="Export Options", style="Card.TLabelframe")
        # exf.pack(fill="x", padx=16, pady=16)
        # ttk.Button(exf, text="Export Current View to PDF", command=self.save_as_pdf).pack(side="left", padx=8, pady=10)
        # ttk.Button(exf, text="Export Slot Range to PDF", command=self._export_range_dialog).pack(side="left", padx=8, pady=10)
        # ttk.Label(self.tab_export, text="Hint: Export uses high-res screenshots to preserve exact canvas layout.", foreground="#666").pack(anchor="w", padx=20)

    def _grid_form_row(self, parent, row, label, var):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=(12, 6), pady=10)
        ttk.Entry(parent, textvariable=var, width=60).grid(row=row, column=1, sticky="w", padx=(0, 12), pady=10)

    def _build_statusbar(self):
        self.statusbar = ttk.Frame(self.root, style="Status.TFrame")
        self.statusbar.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(self.statusbar, text="...", style="Status.TLabel")
        self.status_label.pack(side="left", padx=10, pady=4)
        self.progress = ttk.Progressbar(self.statusbar, mode="determinate", length=180)
        self.progress.pack(side="right", padx=10, pady=4)

    def _update_status(self, msg: str):
        self.status_label.config(text=msg)
        self.statusbar.update_idletasks()

    # -------------------------- App Wiring -----------------------------

    def _load_logo(self):
        try:
            png_path = resource_path('pipeline_schema/LOGO-withoutbg.png')
            img = Image.open(png_path).convert("RGBA").resize((210, 110))
            self.logo_img = ImageTk.PhotoImage(img)
            # lframe = ttk.LabelFrame(self.tab_info, text="Brand", style="Card.TLabelframe")
            lframe.pack(fill="x", padx=16, pady=(8,16))
            ttk.Label(lframe, image=self.logo_img).pack(pady=8)
        except Exception:
            pass

    def _connect_db(self):
        try:
            self.conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='anubhav',
                database='gmfldesktop12'
            )
            self.cursor = self.conn.cursor(buffered=True)
            self._update_status("DB connected.")
        except mysql.connector.Error as err:
            self.conn = None; self.cursor = None
            self._update_status(f"DB error: {err}")

    # ----------------------- Excel: ROBUST LOADER -----------------------

    @staticmethod
    def _normalize_col(col: str) -> str:
        s = str(col)
        s = s.replace("\n", " ")
        s = s.replace("â€™", "'")
        s = s.replace("Â°", "")
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace(".", "")  # "No." -> "No"
        s = s.replace("o' clock", "o clock")  # normalize apostrophe
        return s

    def _map_to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map many header variants to the canonical names the app expects.
        Also build a metres column for plotting.
        """
        cols_norm = {c: self._normalize_col(c) for c in df.columns}

        # Build a reverse dict: norm -> original
        norm_to_orig = {}
        for orig, norm in cols_norm.items():
            norm_to_orig.setdefault(norm, orig)

        def pick(*norm_keys):
            for k in norm_keys:
                if k in norm_to_orig:
                    return norm_to_orig[k]
            return None

        out = df.copy()

        # Pipe Number
        col_pipe_num = pick("pipe number", "pipe no", "pipeno", "pipe", "pipe #", "pipe#")
        if col_pipe_num is not None:
            out.rename(columns={col_pipe_num: "Pipe Number"}, inplace=True)

        # LENGTH COLUMNS
        col_len_m  = pick("pipe length (m)", "length (m)", "joint length (m)", "length m")
        col_len_mm = pick("pipe length (mm)", "length (mm)", "joint length (mm)", "pipe length mm", "lengthmm", "pipe length(mm)")

        if col_len_mm is not None:
            out.rename(columns={col_len_mm: "Pipe Length (mm)"}, inplace=True)
            out["Pipe Length (mm)"] = pd.to_numeric(out["Pipe Length (mm)"], errors="coerce")
            out["Pipe Length (m)"]  = out["Pipe Length (mm)"] / 1000.0
        elif col_len_m is not None:
            out.rename(columns={col_len_m: "Pipe Length (m)"}, inplace=True)
            out["Pipe Length (m)"]  = pd.to_numeric(out["Pipe Length (m)"], errors="coerce")
            out["Pipe Length (mm)"] = out["Pipe Length (m)"] * 1000.0

        # Wall thickness â†’ WT (mm)
        col_wt = pick("wt (mm)", "wall thickness (mm)", "wt", "wall thickness", "thickness (mm)")
        if col_wt is not None:
            out.rename(columns={col_wt: "WT (mm)"}, inplace=True)

        # Feature Type
        col_ftype = pick("feature type", "featuretype", "type of feature")
        if col_ftype is not None:
            out.rename(columns={col_ftype: "Feature Type"}, inplace=True)

        # Feature Identification
        col_fid = pick("feature identification", "feature id", "feature")
        if col_fid is not None:
            out.rename(columns={col_fid: "Feature Identification"}, inplace=True)

        # Depth %
        col_depth = pick("depth %", "depth%", "depth (percent)", "depth percentage")
        if col_depth is not None:
            out.rename(columns={col_depth: "Depth %"}, inplace=True)

        # Type (Internal/External)
        col_type = pick("type", "corrosion type", "int/ext", "internal/external")
        if col_type is not None:
            out.rename(columns={col_type: "Type"}, inplace=True)

        # Orientation o clock
        col_clock = pick("orientation o' clock", "orientation", "clock position", "oclock", "o clock")
        if col_clock is not None:
            out.rename(columns={col_clock: "Orientation o' clock"}, inplace=True)

        # Distance to U/S GW(m)
        col_dist = pick(
            "distance to u/s gw(m)", "distance to u/s (m)", "distance to u/s m",
            "distance to upstream gw (m)", "distance to upstream (m)", "distance to us (m)"
        )
        if col_dist is not None:
            out.rename(columns={col_dist: "Distance to U/S GW(m)"}, inplace=True)

        if "Pipe Length (m)" not in out.columns and "Pipe Length (mm)" in out.columns:
            out["Pipe Length (m)"] = pd.to_numeric(out["Pipe Length (mm)"], errors="coerce") / 1000.0

        return out

    def _read_excel(self):
        # constants_file = os.path.join(self.project_root, "constants.xlsx")
        # default_path = os.path.join(self.pipe_tally, "pipetally_main", "Pipe_Tally_8inch.xlsx")
        default_path = "D:\pickle_6\pipetally_main\Pipe_Tally_8inch.xlsx"
        # if self.pipe_tally and os.path.isfile(self.pipe_tally):
        #     path = self.pipe_tally
        # else:
        #     # your existing default/fallback
        #     default_path = resource_path(r"F:\work_new\client_software\test_data_cs\pickle7\pipetally_main\Pipe_Tally_8incsdh.xlsx")
        #     path = default_path if os.path.exists(default_path) else filedialog.askopenfilename(
        #         title="Select Pipe Tally Excel",
        #         filetypes=[("Excel", "*.xlsx *.xls")]
        #     )
        #     if not path:
        #         self.data = pd.DataFrame()
        #         self._update_status("Excel not selected.")
        #         return

        path = default_path
        if not os.path.exists(path):
            path = filedialog.askopenfilename(
                title="Select Pipe Tally Excel",
                filetypes=[("Excel", "*.xlsx *.xls")]
            )
            if not path:
                self.data = pd.DataFrame()
                self._update_status("Excel not selected.")
                return

        self._update_status("Reading Excel...")
        self.progress.start(10)

        loaded = None
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
            for sheet in xls.sheet_names:
                for header_row in range(0, 5):
                    try:
                        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", header=header_row)
                        if df is None or df.empty:
                            continue
                        df = df.loc[:, ~df.columns.astype(str).str.match(r'^\s*$')]
                        df2 = self._map_to_canonical(df)
                        if "Pipe Number" in df2.columns:
                            loaded = df2
                            raise StopIteration
                    except Exception:
                        continue
        except StopIteration:
            pass
        except Exception as e:
            loaded = None
            info(f"Excel read error: {e}")

        if loaded is None:
            try:
                df = pd.read_excel(path, engine="openpyxl")
                loaded = self._map_to_canonical(df)
            except Exception as e:
                loaded = None
                info(f"Fallback read error: {e}")

        self.progress.stop()

        if loaded is None or loaded.empty:
            self.data = pd.DataFrame()
            self._update_status("Excel read error: could not detect headers.")
            messagebox.showerror("Excel", "Could not parse the Excel headers.\nTry opening the file and ensuring there is a 'Pipe Number' column.")
            return

        for col, default in [
            ("Pipe Number", None),
            ("Pipe Length (mm)", 0.0),
            ("Pipe Length (m)", 0.0),
            ("WT (mm)", 0.0),
            ("Feature Type", ""),
            ("Feature Identification", ""),
            ("Depth %", 0.0),
            ("Type", ""),
            ("Orientation o' clock", ""),
            ("Distance to U/S GW(m)", 0.0),
        ]:
            if col not in loaded.columns:
                loaded[col] = default

        for col in ["Pipe Length (mm)", "Pipe Length (m)", "WT (mm)", "Depth %", "Distance to U/S GW(m)"]:
            loaded[col] = pd.to_numeric(loaded[col], errors="coerce")

        loaded["Orientation o' clock"] = loaded["Orientation o' clock"].astype(str).str.replace("â€™", "'").str.replace("o' clock", "o clock")

        self.data = loaded
        self._update_status("Excel loaded.")

    def _reload_excel(self):
        self._read_excel()
        self._make_chunks()
        self._populate_slot_menu()
        self.display_page(0)

    def _make_chunks(self):
        self.chunks = []
        if self.data.empty:
            return
        if 'Pipe Number' not in self.data.columns:
            messagebox.showerror("Missing Column", "Column 'Pipe Number' not found in Excel.")
            return
        pn = pd.to_numeric(self.data["Pipe Number"], errors="coerce")
        data2 = self.data.copy()
        data2["__pn__"] = pn
        for _, grp in data2.groupby("__pn__", dropna=True):
            self.chunks.append(grp.drop(columns="__pn__", errors="ignore"))

    def _populate_slot_menu(self):
        n = len(self.chunks)
        slots = [f"{i+1}-{min(i+100, n)}" for i in range(0, n, 100)] or ["Empty"]
        self.slot_menu["values"] = slots
        self.slot_menu.set(slots[0])

    def _on_slot_select(self, _evt=None):
        sel = self.slot_var.get().strip()
        if not sel or sel == "Empty":
            return
        start = int(sel.split("-")[0])
        page = (start - 1) // self.chunks_per_page
        self.display_page(page)

    def get_data(self):
        if self.cursor is None:
            messagebox.showwarning("DB", "Database connection not available.")
            return
        runid = self.run_id_var.get().strip()
        # if not runid:
        #     # messagebox.showinfo("Run ID", "Enter a Run ID first.")
        #     return
        try:
            self._update_status("Fetching from DB...")
            self.progress.start(10)
            self.cursor.execute("SELECT Pipeline_owner FROM mfldesktop.projectdetail WHERE runid = %s", (runid,))
            d = self.cursor.fetchone()
            self.client_var.set(d[0] if d else "No data")

            self.cursor.execute("SELECT Pipeline_Name FROM mfldesktop.projectdetail WHERE runid = %s", (runid,))
            d = self.cursor.fetchone()
            self.pipeline_name_var.set(d[0] if d else "No data")

            self.cursor.execute("SELECT Report_date FROM mfldesktop.projectdetail WHERE runid = %s", (runid,))
            d = self.cursor.fetchone()
            self.report_date_var.set(str(d[0]) if d else "No data")
            self._update_status("DB fetch complete.")
        except mysql.connector.Error as err:
            messagebox.showerror("DB Error", str(err))
            self._update_status(f"DB error: {err}")
        finally:
            self.progress.stop()

    # ------------------------ Visualization ----------------------------

    def display_page(self, page_number=0):
        self.current_page = page_number
        self._update_status(f"Rendering page {page_number+1}...")
        self.pipe_canvas.delete("all")
        self.pipe_item_map.clear()

        # Layout
        start_x, start_y = 60, 140
        rect_w, rect_h = 120, 56
        gap_x, gap_y = 26, 120
        per_row = 10

        # Left-side labels
        for i in range(10):
            y0 = start_y + i*gap_y
            self.pipe_canvas.create_text(start_x, y0 - 42, text="Joint Number:", font=("Segoe UI", 9), anchor="w")
            self.pipe_canvas.create_text(start_x, y0 - 26, text="Length (m):",  font=("Segoe UI", 8), anchor="w")
            self.pipe_canvas.create_text(start_x, y0 - 10, text="WT (mm):",     font=("Segoe UI", 8), anchor="w")

        start_chunk = page_number * self.chunks_per_page
        end_chunk = min(start_chunk + self.chunks_per_page, len(self.chunks))
        view_chunks = self.chunks[start_chunk:end_chunk]

        # Reset table
        for i in self.tree.get_children():
            self.tree.delete(i)

        for idx, chunk in enumerate(view_chunks):
            row = idx // per_row
            col = idx % per_row
            x1 = start_x + (col + 1) * (rect_w + gap_x)
            y1 = start_y + row * gap_y
            x2, y2 = x1 + rect_w, y1 + rect_h

            # --- choose a robust length for plotting ---
            length_m_series = pd.to_numeric(chunk.get('Pipe Length (m)', pd.Series(dtype=float)), errors='coerce')
            joint_length = float(length_m_series.median(skipna=True)) if length_m_series.notna().any() else 0.0

            dtus_series = pd.to_numeric(chunk.get('Distance to U/S GW(m)', pd.Series(dtype=float)), errors='coerce')
            dtus_max = float(dtus_series.max()) if dtus_series.notna().any() else 0.0

            # If the reported joint length is tiny or < max distance, fall back to distances
            length_plot = joint_length
            if length_plot <= 0.5 or (dtus_max > 0 and dtus_max > length_plot * 1.1):
                length_plot = max(dtus_max, 0.5)

            wall_thickness = float(pd.to_numeric(chunk.get('WT (mm)', pd.Series([0])), errors='coerce').median(skipna=True))
            pipe_number = chunk.get('Pipe Number', pd.Series([None])).max()

            # Box color by severity
            severity_color = "#e8fff0"
            if 'Feature Type' in chunk.columns and not chunk['Feature Type'].isna().all():
                ml = chunk[chunk['Feature Type'].astype(str).str.contains('metal loss', case=False, na=False)]
                if not ml.empty:
                    depth = pd.to_numeric(ml.get('Depth %', pd.Series([0])), errors='coerce').fillna(0).max()
                    if depth > 50:
                        severity_color = "#ffecec"
                    elif depth >= 20:
                        severity_color = "#eef3ff"

            self.pipe_canvas.create_rectangle(x1, y1, x2, y2, outline="#333", fill=severity_color)
            # label shows best known joint length (fallback to length_plot if needed)
            label_length = joint_length if joint_length > 0.5 else length_plot
            self.pipe_canvas.create_text(x1 + rect_w//2, y1 - 42, text=f"{pipe_number}", font=("Segoe UI", 9))
            self.pipe_canvas.create_text(x1 + rect_w//2, y1 - 26, text=f"{label_length:.2f}", font=("Segoe UI", 8))
            self.pipe_canvas.create_text(x1 + rect_w//2, y1 - 10, text=f"{wall_thickness:.2f}", font=("Segoe UI", 8))

            defects_here = []

            # Bends
            if 'Feature Identification' in chunk.columns:
                bend = chunk[chunk['Feature Identification'].astype(str).str.contains('bend', case=False, na=False)]
                for _, rowd in bend.iterrows():
                    dtus  = pd.to_numeric(rowd.get('Distance to U/S GW(m)'), errors='coerce')
                    clock = rowd.get("Orientation o' clock")
                    if pd.notna(dtus) and length_plot > 0 and pd.notna(clock):
                        hour, minute = safe_parse_time(clock)
                        angle = (hour % 12 + minute/60) * 30
                        bx = x1 + (dtus * (rect_w / max(length_plot, 1e-9)))
                        by = y1 + (rect_h / 180) * angle if 0 <= angle <= 180 else y2 - ((rect_h / 180) * (angle - 180))
                        self.pipe_canvas.create_text(bx, by, text="*", font=("Segoe UI", 16), fill="#333")
                        defects_here.append(("Bend", None, None, clock, dtus))

            # Metal Loss
            if 'Feature Type' in chunk.columns:
                chunk2 = chunk.dropna(subset=['Feature Type'])
                ml = chunk2[chunk2['Feature Type'].astype(str).str.contains('metal loss', case=False, na=False)]
                for _, rowd in ml.iterrows():
                    dtus   = pd.to_numeric(rowd.get('Distance to U/S GW(m)'), errors='coerce')
                    clock  = rowd.get("Orientation o' clock")
                    depth  = pd.to_numeric(rowd.get('Depth %'), errors='coerce')
                    typ    = rowd.get('Type')
                    if pd.notna(dtus) and length_plot > 0 and pd.notna(clock):
                        hour, minute = safe_parse_time(clock)
                        angle = (hour % 12 + minute/60) * 30
                        mx = x1 + (dtus * (rect_w / max(length_plot, 1e-9)))
                        my = y1 + (rect_h / 180) * angle if 0 <= angle <= 180 else y2 - ((rect_h / 180) * (angle - 180))
                        r = 4
                        if pd.notna(depth):
                            if depth > 50:  outline, fill = "red", "red"
                            elif depth > 20: outline, fill = "blue", "blue"
                            else:            outline, fill = "green", "green"
                        else:
                            outline, fill = "#222", ""
                        if str(typ).lower().startswith("int"):
                            self.pipe_canvas.create_oval(mx-r, my-r, mx+r, my+r, outline=outline, width=2)
                        else:
                            self.pipe_canvas.create_oval(mx-r, my-r, mx+r, my+r, fill=fill, outline="")
                        defects_here.append(("Metal Loss", float(depth) if pd.notna(depth) else None, typ, clock, float(dtus)))
                        self.tree.insert("", "end", values=(pipe_number, f"{label_length:.2f}", f"{wall_thickness:.2f}",
                                                            "Metal Loss", f"{'' if pd.isna(depth) else depth:.1f}",
                                                            typ, str(clock), f"{'' if pd.isna(dtus) else dtus:.2f}"))

            self.pipe_item_map[(x1, y1, x2, y2)] = {
                "pipe_no": pipe_number,
                "length": label_length,
                "wt": wall_thickness,
                "defects": defects_here
            }

        # Update scroll region (include legend width)
        bbox = self.pipe_canvas.bbox("all")
        if bbox:
            x1, y1, x2, y2 = bbox
            # Add extra margin so legend isn’t cut off
            self.pipe_canvas.configure(scrollregion=(x1, y1, x2 + 200, y2 + 100))

        self._update_status("Rendered.")

        # ------------------ Add Legend ------------------
        # ------------------ Add Legend (placed neatly on right side) ------------------
        # Find bounding box of all joints
        bbox = self.pipe_canvas.bbox("all")
        if bbox:
            min_x, min_y, max_x, max_y = bbox
        else:
            min_x, min_y, max_x, max_y = 60, 120, 1500, 800

        # Legend position: on the right side, beside the last column
        legend_x = max_x + 40  # a bit to the right of the last joint
        legend_y = min_y + 40  # aligned with the top row
        w, h = 150, 130

        # Background
        self.pipe_canvas.create_rectangle(
            legend_x, legend_y,
            legend_x + w, legend_y + h,
            outline="#888", fill="#fdfdfd"
        )

        # Title
        self.pipe_canvas.create_text(
            legend_x + w / 2, legend_y + 12,
            text="Legend", font=("Segoe UI", 9, "bold")
        )

        # Items
        y = legend_y + 26
        dy = 14
        font_small = ("Segoe UI", 8)

        # Metal Loss
        items = [
            ("green", "Metal Loss < 20%"),
            ("blue", "Metal Loss 20–50%"),
            ("red", "Metal Loss > 50%"),
        ]
        for color, text in items:
            self.pipe_canvas.create_oval(legend_x + 8, y, legend_x + 14, y + 6, fill=color, outline="")
            self.pipe_canvas.create_text(legend_x + 22, y + 3, text=text, anchor="w", font=font_small)
            y += dy

        # Bend
        self.pipe_canvas.create_text(legend_x + 10, y + 3, text="*", font=("Segoe UI", 10))
        self.pipe_canvas.create_text(legend_x + 22, y + 3, text="Bend", anchor="w", font=font_small)
        y += dy

        # Severity Boxes
        boxes = [
            ("#e8fff0", "Normal Joint"),
            ("#eef3ff", "Medium Severity"),
            ("#ffecec", "High Severity"),
        ]
        for fill, label in boxes:
            self.pipe_canvas.create_rectangle(legend_x + 8, y - 3, legend_x + 18, y + 7, fill=fill, outline="#333")
            self.pipe_canvas.create_text(legend_x + 24, y + 2, text=label, anchor="w", font=font_small)
            y += dy
        # ----------------------------------------------------------------------

    # --------------------- Canvas Helpers & Events ----------------------

    def _apply_filter_to_table(self):  # kept for compatibility (no-op filter UI removed)
        pass

    def _jump_to_pipe(self):
        target = self.jump_pipe_var.get().strip()
        if not target: return
        try:
            target = int(float(target))
        except Exception:
            messagebox.showinfo("Jump", "Enter a valid pipe number.")
            return
        for idx, ch in enumerate(self.chunks):
            p = ch.get('Pipe Number', pd.Series([None])).max()
            if pd.notna(p) and int(float(p)) == target:
                page = idx // self.chunks_per_page
                self.display_page(page)
                return
        messagebox.showinfo("Jump", f"Pipe {target} not found.")

    def _on_tree_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel: return
        vals = self.tree.item(sel[0], "values")
        if not vals: return
        try:
            pno = int(float(vals[0]))
            self.jump_pipe_var.set(str(pno))
            self._jump_to_pipe()
        except Exception:
            pass

    def _update_canvas_scrollregion(self):
        bbox = self.main_canvas.bbox("all")
        if bbox:
            x1, y1, x2, y2 = bbox
            self.main_canvas.configure(scrollregion=(x1, y1, x2 + 100, y2 + 100))

    def _on_mousewheel_zoom(self, event):
        # ctrl + wheel to zoom
        if (event.state & 0x0004) == 0:
            return
        factor = 1.1 if (event.delta > 0 or getattr(event, 'num', 0) == 4) else 0.9
        self.canvas_scale *= factor
        self.pipe_canvas.scale("all", 0, 0, factor, factor)
        w = self.pipe_canvas.winfo_width() * factor
        h = self.pipe_canvas.winfo_height() * factor
        self.pipe_canvas.config(width=int(w), height=int(h))

    def _reset_zoom(self):
        self.canvas_scale = 1.0
        self.display_page(self.current_page)

    def _start_pan(self, event):
        self._pan_start = (event.x, event.y)
        self.main_canvas.scan_mark(event.x, event.y)

    def _on_pan(self, event):
        if self._pan_start is not None:
            self.main_canvas.scan_dragto(event.x, event.y, gain=1)

    def _end_pan(self, _event):
        self._pan_start = None

    def _on_canvas_hover(self, event):
        x = self.pipe_canvas.canvasx(event.x)
        y = self.pipe_canvas.canvasy(event.y)
        for (x1, y1, x2, y2), payload in self.pipe_item_map.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                tip = f"Pipe #{payload['pipe_no']}\nLength: {payload['length']:.2f} m | WT: {payload['wt']:.2f} mm"
                if payload['defects']:
                    tip += "\nDefects:"
                    for d in payload['defects']:
                        if d[0] == "Metal Loss":
                            tip += f"\nâ€¢ Metal Loss {'' if d[1] is None else d[1]}% ({d[2]}), {d[3]}, US {d[4]}m"
                        elif d[0] == "Bend":
                            tip += f"\nâ€¢ Bend @ {d[3]} ({d[4]}m)"
                self._show_tooltip(event.x_root, event.y_root, tip)
                return
        self._hide_tooltip()

    def _show_tooltip(self, rx, ry, text):
        self._hide_tooltip()
        tw = tk.Toplevel(self.root)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{rx+14}+{ry+12}")
        frm = ttk.Frame(tw, style="Card.TFrame")
        ttk.Label(frm, text=text, justify="left").pack(padx=8, pady=6)
        frm.pack()
        self._tooltip_win = tw

    def _hide_tooltip(self):
        if self._tooltip_win is not None:
            try:
                self._tooltip_win.destroy()
            except Exception:
                pass
            self._tooltip_win = None

    # -------------------------- Export (PDF) ----------------------------

    def save_as_pdf(self):
        try:
            out_path = filedialog.asksaveasfilename(
                title="Save PDF",
                defaultextension=".pdf",
                filetypes=[("PDF Files", "*.pdf")],
                initialfile="Pipeline_Scheme_Report.pdf"
            )
            if not out_path:
                return

            self.root.update()
            self._update_status("Capturing for PDF...")
            self.progress.start(10)

            total_h = self.canvas_frame.winfo_height()
            viewport_h = self.root.winfo_height()
            steps, pos, step_px = [], 0, max(600, int(viewport_h * 0.75))
            while pos < total_h:
                steps.append(pos); pos += step_px

            images, seen = [], set()
            for pos in steps:
                frac = pos / max(total_h, 1)
                self.main_canvas.yview_moveto(frac)
                self.root.update()
                img = ImageGrab.grab(include_layered_windows=True)
                hsh = md5_image(img)
                if hsh not in seen:
                    images.append(img); seen.add(hsh)

            pdf = FPDF(unit='mm', format='A4')
            img_w, img_h = 210, 297
            for i in range(0, len(images), 2):
                pdf.add_page()
                for j in range(2):
                    if i + j < len(images):
                        tmp = images[i + j]
                        tmp_path = f"_tmp_pdf_{i+j}.png"
                        tmp.save(tmp_path)
                        y_pos = (img_h / 2) * j
                        pdf.image(tmp_path, x=0, y=y_pos, w=img_w, h=img_h / 2)
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

            pdf.output(out_path)
            self._update_status(f"Saved: {out_path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            self._update_status(f"Export failed: {e}")
        finally:
            self.progress.stop()

    def _export_range_dialog(self):
        dlg = tk.Toplevel(self.root); dlg.title("Export Slot Range"); dlg.resizable(False, False)
        ttk.Label(dlg, text="From slot (page #):").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        ttk.Label(dlg, text="To slot (page #):").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        v1 = tk.StringVar(value=str(self.current_page + 1))
        v2 = tk.StringVar(value=str(self.current_page + 1))
        ttk.Entry(dlg, textvariable=v1, width=8).grid(row=0, column=1, padx=10, pady=10)
        ttk.Entry(dlg, textvariable=v2, width=8).grid(row=1, column=1, padx=10, pady=10)
        def go():
            try:
                p1 = max(1, int(v1.get())); p2 = max(1, int(v2.get()))
            except Exception:
                messagebox.showinfo("Export", "Enter valid integers."); return
            dlg.destroy(); self._export_range(p1-1, p2-1)
        ttk.Button(dlg, text="Export", command=go).grid(row=2, column=0, columnspan=2, pady=(0, 12))

    def _export_range(self, pstart, pend):
        try:
            out_path = filedialog.asksaveasfilename(
                title="Save PDF",
                defaultextension=".pdf",
                filetypes=[("PDF Files", "*.pdf")],
                initialfile=f"Pipeline_Scheme_Report_slots_{pstart+1}_{pend+1}.pdf"
            )
            if not out_path:
                return

            self.root.update()
            self._update_status("Capturing pages...")
            self.progress.start(10)
            pdf = FPDF(unit='mm', format='A4')
            img_w, img_h = 210, 297
            tmp_files = []

            def capture():
                self.root.update()
                img = ImageGrab.grab(include_layered_windows=True)
                pth = f"_tmp_range_{len(tmp_files)}.png"
                img.save(pth); tmp_files.append(pth)

            for p in range(pstart, pend+1):
                if p < 0 or p > (len(self.chunks)-1)//self.chunks_per_page: continue
                self.display_page(p)
                self.main_canvas.yview_moveto(0.0); capture()
                self.main_canvas.yview_moveto(0.5); capture()

            for i in range(0, len(tmp_files), 2):
                pdf.add_page()
                for j in range(2):
                    if i + j < len(tmp_files):
                        pth = tmp_files[i + j]
                        y_pos = (img_h / 2) * j
                        pdf.image(pth, x=0, y=y_pos, w=img_w, h=img_h / 2)

            pdf.output(out_path)
            for pth in tmp_files:
                try: os.remove(pth)
                except Exception: pass
            self._update_status(f"Saved: {out_path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            self._update_status(f"Export failed: {e}")
        finally:
            self.progress.stop()

    # ---------------------- Cleanup & App Run ---------------------------

    def close(self):
        try:
            if self.cursor is not None: self.cursor.close()
            if self.conn is not None and self.conn.is_connected(): self.conn.close()
        except Exception:
            pass
        self.root.destroy()

# ----------------------------- Entrypoint -------------------------------

def create_window(pipe_tally=None):
    root = tb.Window(themename="cosmo") if _USING_BOOTSTRAP else tk.Tk()
    app = PipelineApp(root, pipe_tally=pipe_tally)     # <-- pass it in
    root.protocol("WM_DELETE_WINDOW", app.close)
    return root

def main():
    root = create_window()
    root.mainloop()

if __name__ == "__main__":
    main()

def run_app(pipe_tally=None):
    print(pipe_tally)
    root = create_window(pipe_tally=pipe_tally)        # <-- pass it in
    root.mainloop()