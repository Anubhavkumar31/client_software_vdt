import datetime
import os
import io
import sys
import math
import time
import traceback
import tempfile

import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import pandas as pd
from PIL import ImageGrab, Image, ImageTk
import img2pdf

import win32api
import win32print


class Digsheet:
    """
    One big class version of your Digsheet UI.

    - Same logic / UI as your previous standalone script.
    - Can be run:
        1) from CLI:  python this_file.py <pipe_tally.pkl> <project_root>
        2) from another script: Digsheet(pipe_tally_file=..., project_root=...)
    """

    # ---- Section IDs and names ----
    SECTION_MAP = {
        1: "Client Description",
        2: "Feature Location on Pipe",
        3: "Comment",
        4: "Feature Description",
        5: "Pipe Location",
    }

    SECTION_THRESHOLDS = {
        "Client Description":       (0, 0, 175, 40),
        "Feature Location on Pipe": (5, 32, 170, 93),
        "Comment":                  (0, 85, 175, 120),
        "Feature Description":      (0, 110, 175, 170),
        "Pipe Location":            (0, 107, 175, 220),
    }

    def __init__(self, pipe_tally_file=None, project_root=None):
        """
        pipe_tally_file: path to pipe_tally.pkl (or xlsx converted to pkl).
        project_root   : folder containing constants.xlsx / constants.csv, etc.
        """
        # ---- external inputs ----
        self.pipe_tally_file = pipe_tally_file
        self.project_root = project_root

        # core state
        self.df = None
        self.batch_cancelled = False
        self.scrollable_active = False

        # TK root
        self.root = tk.Tk()
        self.root.title("Digsheet")
        self.root.state("zoomed")
        self.root.resizable(False, False)
        self.root.configure(bg="white")

        # progress holder handle
        self.progress_frame_ref = None

        # icons
        self.valve_img = None
        self.bend_img = None
        self.flange_img = None
        self.flowtee_img = None
        self.magnet_img = None

        # place-holders for widgets
        self.button_frame = None
        self.input_frame = None
        self.toolbar = None
        self.group1 = None
        self.group2 = None
        self.group3 = None

        self.preview_holder = None
        self._preview_placeholder_ref = None

        self.container = None
        self.canvas = None
        self.scrollbar = None
        self.scrollable_frame = None

        self.client_desc_frame = None
        self.main_frame = None
        self.comment_frame = None
        self.feature_desc_frame = None
        self.third_frame = None

        self.pipe_canvas1 = None
        self.pipe_canvas = None

        self.defect_entry = None

        # TK variables
        self.pipe_id_var = tk.StringVar(master=self.root)
        self.length_var = tk.StringVar(master=self.root)
        self.wt_var = tk.StringVar(master=self.root)
        self.latitude_var = tk.StringVar(master=self.root)
        self.longitude_var = tk.StringVar(master=self.root)
        self.altitude_var = tk.StringVar(master=self.root)

        self.client_var = tk.StringVar(master=self.root)
        self.pipeline_name_var = tk.StringVar(master=self.root)
        self.pipeline_section_var = tk.StringVar(master=self.root)

        # feature labels dict
        self.feature_labels = {}

        # for pipe location layout
        self.mid_x = 0
        self.mid_y = 0

        # ---- build everything ----
        self._setup_style()
        self._build_right_panel()
        self._build_icons()
        self._build_scroll_canvas()
        self._build_main_blocks()
        self._build_pipe_location_static()

        # show preview placeholder
        self._show_preview_placeholder()

    # ======================================================================
    #  ROOT / STYLE / LAYOUT
    # ======================================================================

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass

        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor="white",
            background="deepskyblue",
            thickness=25,
            bordercolor="white",
            lightcolor="deepskyblue",
            darkcolor="deepskyblue",
        )

        style.configure(
            "Floating.Vertical.TScrollbar",
            troughcolor="white",
            background="gray60",
            bordercolor="white",
            lightcolor="gray60",
            darkcolor="gray60",
        )

    def _build_right_panel(self):
        """Build the right side panel (toolbar + preview)."""
        screen_w = self.root.winfo_screenwidth()
        button_panel_w = (screen_w / 2) - 150

        self.button_frame = tk.Frame(self.root, bg="white", width=button_panel_w)
        self.button_frame.pack(side="right", fill="y", padx=50, pady=0, anchor="n")
        self.button_frame.pack_propagate(False)

        self.input_frame = tk.Frame(self.button_frame, bg="white")
        self.input_frame.pack(side="top", fill="both", expand=True, pady=(8, 0))

        # toolbar
        self.toolbar = tk.Frame(self.input_frame, bg="white")
        self.toolbar.pack(side="top", fill="x", pady=(0, 8))

        # Group1
        self.group1 = tk.LabelFrame(
            self.toolbar,
            text="",
            bg="white",
            fg="gray40",
            relief="groove",
            bd=1,
            padx=6,
            pady=4,
        )
        self.group1.pack(side="left", padx=(0, 10))

        tk.Label(self.group1, text="Enter Defect S.no:", bg="white").pack(
            side="left", padx=(2, 6)
        )
        self.defect_entry = tk.Entry(self.group1, width=8)
        self.defect_entry.pack(side="left", padx=(0, 6))

        tk.Button(self.group1, text="Load", command=self.on_load_click).pack(
            side="left", padx=3
        )
        tk.Button(self.group1, text="Save Current", command=self.open_save_dialog).pack(
            side="left", padx=3
        )
        tk.Button(
            self.group1, text="Print current", command=self.print_all_sections_dialog
        ).pack(side="left", padx=3)

        # Group2
        self.group2 = tk.Frame(self.toolbar, bg="white")
        self.group2.pack(side="left", padx=2)
        tk.Button(
            self.group2, text="Batch Export", command=self.open_batch_dialog_new
        ).pack(side="left")

        # Group3
        self.group3 = tk.Frame(self.toolbar, bg="white")
        self.group3.pack(side="left", padx=2)
        tk.Button(
            self.group3, text="MultiPreview", command=self.open_preview_dialog
        ).pack(side="left")
        tk.Button(self.group3, text="Reset", command=self.reset_ui).pack(
            side="left", padx=3
        )

        # preview holder
        self.preview_holder = tk.Frame(
            self.input_frame,
            bg="white",
            highlightbackground="#e8e8e8",
            highlightthickness=3,
        )
        self.preview_holder.pack(side="top", fill="both", expand=True, pady=(8, 0))

    def _build_icons(self):
        """Load valve/bend/flange/flowtee/magnet icons."""
        try:
            icon_path = os.getcwd() + "/Components/"+ "/dig/"+"/digsheet_icon/"
            self.valve_img = ImageTk.PhotoImage(
                Image.open(icon_path + "valve.png").resize((18, 18))
            )
            self.bend_img = ImageTk.PhotoImage(
                Image.open(icon_path + "bend.png").resize((18, 18))
            )
            self.flange_img = ImageTk.PhotoImage(
                Image.open(icon_path + "flange.png").resize((18, 18))
            )
            self.flowtee_img = ImageTk.PhotoImage(
                Image.open(icon_path + "flowtee.png").resize((18, 18))
            )
            self.magnet_img = ImageTk.PhotoImage(
                Image.open(icon_path + "magnet.png").resize((18, 18))
            )
        except Exception as e:
            print("Image loading error:", e)
            self.valve_img = self.bend_img = self.flange_img = self.flowtee_img = self.magnet_img = None

    def _build_scroll_canvas(self):
        """Create scrollable canvas for the main digsheet area."""
        self.container = tk.Frame(self.root)
        self.container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.container, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        # vertical scrollbar (floating)
        self.scrollbar = tk.Scrollbar(
            self.canvas, orient="vertical", command=self.canvas.yview, width=9
        )

        def _yscroll_set(lo, hi):
            self.scrollbar.set(lo, hi)
            try:
                lo_f, hi_f = float(lo), float(hi)
            except Exception:
                lo_f, hi_f = 0.0, 1.0

            if hi_f - lo_f >= 0.999:
                self.scrollbar.place_forget()
            else:
                self.scrollbar.place(
                    in_=self.canvas,
                    relx=1.0,
                    x=-8,
                    rely=0.5,
                    anchor="e",
                    relheight=0.98,
                )

        self.canvas.configure(yscrollcommand=_yscroll_set)

        # scrollable frame
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # mousewheel
        def _on_mousewheel(event):
            if event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                self.canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(3, "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", _on_mousewheel)
        self.canvas.bind_all("<Button-5>", _on_mousewheel)

    # ======================================================================
    #  MAIN BLOCKS (Client, Feature on Pipe, Comment, Feature Description, Pipe Location)
    # ======================================================================

    def _build_main_blocks(self):
        """Build all left side content blocks inside scrollable_frame."""

        # -------- Client Description block --------
        self.client_desc_frame = tk.Frame(
            self.scrollable_frame,
            bg="white",
            padx=5,
            pady=2,
            highlightbackground="black",
            highlightthickness=1,
        )
        self.client_desc_frame.pack(fill="both", padx=(15, 15), pady=(5, 0))

        tk.Label(
            self.client_desc_frame,
            text="Client Description:",
            bg="white",
            fg="deepskyblue",
            font=("Arial", 10, "bold"),
        ).pack(side="top", pady=(2, 6))

        left_frame = tk.Frame(self.client_desc_frame, bg="white")
        left_frame.pack(side="left", fill="both", expand=True)

        left_frame.grid_columnconfigure(0, weight=0, minsize=130)
        left_frame.grid_columnconfigure(1, weight=1)

        fields_top = [
            ("Client", self.client_var),
            ("Pipeline Name", self.pipeline_name_var),
            ("Pipeline Section", self.pipeline_section_var),
        ]

        for r, (txt, var) in enumerate(fields_top):
            tk.Label(
                left_frame,
                text=f"{txt}:",
                bg="white",
                anchor="w",
                font=("Arial", 9),
            ).grid(row=r, column=0, sticky="w", padx=(10, 6), pady=(2, 2))

            tk.Label(
                left_frame,
                textvariable=var,  # 🔥 automatically updates
                bg="white",
                anchor="w",
                font=("Arial", 9),
            ).grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=(2, 2))

        left_frame.grid_columnconfigure(0, weight=0)
        left_frame.grid_columnconfigure(1, weight=1)

        try:
            icon_path_vdt = os.getcwd() +  "/Components/"+ "/dig/"+"/digsheet_icon/" + "vdt-logo.png"
            logo_img = Image.open(
                icon_path_vdt
            ).resize((100, 100))
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_lbl = tk.Label(self.client_desc_frame, image=logo_tk, bg="white")
            logo_lbl.place(relx=1.0, rely=0.5, anchor="e", x=-10)
            self.client_desc_frame.logo_ref = logo_tk
        except Exception as e:
            print("Logo load failed:", e)

        # -------- Feature Location on Pipe + Pipe Description --------
        self.main_frame = tk.Frame(self.scrollable_frame, bg="white")
        self.main_frame.pack(pady=5, fill="x", padx=10)

        # left (Feature Location on Pipe)
        feature_frame = tk.Frame(
            self.main_frame,
            bg="white",
            padx=5,
            pady=5,
            highlightbackground="black",
            highlightthickness=1,
        )
        feature_frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(
            feature_frame,
            text="Feature Location on Pipe:",
            bg="white",
            fg="deepskyblue",
            font=("Arial", 10, "bold"),
        ).pack(pady=(0, 5))

        self.pipe_canvas1 = tk.Canvas(
            feature_frame, width=360, height=160, bg="white", highlightthickness=0
        )
        self.pipe_canvas1.pack()

        # right (Pipe Description)
        desc_frame = tk.Frame(
            self.main_frame,
            bg="white",
            padx=5,
            pady=5,
            highlightbackground="black",
            highlightthickness=1,
        )
        desc_frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(
            desc_frame,
            text="Pipe Description:",
            bg="white",
            fg="deepskyblue",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=0, columnspan=5, padx=5, pady=(0, 5), sticky="ew")

        fields = [
            ("Pipe Number", self.pipe_id_var),
            ("Pipe Length (m)", self.length_var),
            ("WT (mm)", self.wt_var),
            ("Latitude", self.latitude_var),
            ("Longitude", self.longitude_var),
            ("Altitude (m)", self.altitude_var),
        ]

        for i, (label, var) in enumerate(fields, start=1):
            tk.Label(
                desc_frame,
                text=label + ":",
                bg="white",
                anchor="w",
                font=("Arial", 9),
            ).grid(row=i, column=0, sticky="w", padx=(5, 2), pady=(2, 2))

            tk.Label(
                desc_frame,
                textvariable=var,
                bg="white",
                anchor="w",
                font=("Arial", 9),
            ).grid(row=i, column=1, sticky="w", padx=(2, 10), pady=(2, 2))

        for col in range(2):
            desc_frame.grid_columnconfigure(col, weight=1)

        # -------- Comment block --------
        self.comment_frame = tk.Frame(
            self.scrollable_frame,
            bg="white",
            padx=5,
            pady=2,
            highlightbackground="black",
            highlightthickness=1,
        )
        self.comment_frame.pack(fill="both", padx=(15, 15), pady=(5, 5))

        tk.Label(
            self.comment_frame,
            text="Comment:",
            bg="white",
            fg="deepskyblue",
            font=("Arial", 10, "bold"),
        ).pack(side="top", pady=(0, 5))

        self.comment_placeholder = tk.Label(
            self.comment_frame,
            text="",
            bg="white",
            anchor="w",
            justify="left",
            font=("Arial", 9),
        )
        self.comment_placeholder.pack(fill="both", expand=True, padx=10, pady=20)

        # -------- Feature Description block --------
        self.feature_desc_frame = tk.Frame(
            self.scrollable_frame,
            bg="white",
            padx=5,
            pady=2,
            highlightbackground="black",
            highlightthickness=1,
        )
        self.feature_desc_frame.pack(fill="both", padx=15)

        for col in range(5):
            self.feature_desc_frame.grid_columnconfigure(col, weight=1)
        self.feature_desc_frame.grid_columnconfigure(2, minsize=80)

        section_title = tk.Label(
            self.feature_desc_frame,
            text="Feature Description:",
            bg="white",
            fg="deepskyblue",
            font=("Arial", 10, "bold"),
            anchor="center",
            justify="center",
        )
        section_title.grid(row=0, column=0, columnspan=5, pady=(0, 5), sticky="ew")

        left_fields = [
            "Feature",
            "Feature type",
            "Anomaly dimension class",
            "Surface Location",
            "Remaining wall thickness (mm)",
            "ERF",
            "Safe pressure (kg/cm²)",
        ]
        right_fields = [
            "Absolute Distance (m)",
            "Length (mm)",
            "Width (mm)",
            "Max. Depth(%)",
            "Orientation(hr:min)",
            "Latitude",
            "Longitude",
        ]

        label_padx = (5, 2)
        value_padx = (2, 10)

        for i, label_text in enumerate(left_fields):
            tk.Label(
                self.feature_desc_frame,
                text=label_text + ":",
                bg="white",
                anchor="w",
                font=("Arial", 9),
            ).grid(row=i + 1, column=0, sticky="w", padx=label_padx, pady=2)

            lab = tk.Label(
                self.feature_desc_frame,
                text="",
                bg="white",
                anchor="w",
                font=("Arial", 9),
            )
            lab.grid(row=i + 1, column=1, sticky="w", padx=value_padx, pady=2)
            self.feature_labels[label_text] = lab

        for i, label_text in enumerate(right_fields):
            tk.Label(
                self.feature_desc_frame,
                text=label_text + ":",
                bg="white",
                anchor="w",
                font=("Arial", 9),
            ).grid(row=i + 1, column=3, sticky="w", padx=label_padx, pady=2)

            lab = tk.Label(
                self.feature_desc_frame,
                text="",
                bg="white",
                anchor="w",
                font=("Arial", 9),
            )
            lab.grid(row=i + 1, column=4, sticky="w", padx=value_padx, pady=2)
            self.feature_labels[label_text] = lab

        # -------- Third (Pipe Location) block --------
        self.third_frame = tk.Frame(
            self.scrollable_frame,
            bg="white",
            padx=10,
            pady=10,
            highlightbackground="black",
            highlightthickness=1,
        )
        self.third_frame.pack(fill="both", padx=15, pady=4)

        tk.Label(
            self.third_frame,
            text="Pipe Location:",
            bg="white",
            fg="deepskyblue",
            font=("Arial", 9, "bold"),
        ).grid(row=0, column=0, columnspan=5, sticky="ew")

        self.pipe_canvas = tk.Canvas(
            self.third_frame, width=650, height=370, bg="white", highlightthickness=0
        )
        self.pipe_canvas.grid(row=1, column=0, columnspan=5)

        self.pipe_canvas.update()
        canvas_width = self.pipe_canvas.winfo_width()
        canvas_height = self.pipe_canvas.winfo_height()
        self.mid_x = int(canvas_width / 2)
        self.mid_y = int(canvas_height / 2)

        for col in range(5):
            self.third_frame.grid_columnconfigure(col, weight=1)

    def _build_pipe_location_static(self):
        """Draw static texts, lines, boxes in the Pipe Location canvas."""
        mid_x = self.mid_x
        mid_y = self.mid_y

        self.pipe_canvas.create_line(mid_x, 30, mid_x, mid_y + 150, arrow=tk.FIRST)

        self.pipe_canvas.create_text(
            mid_x, 5, text="Upstream Weld", font=("Arial", 10)
        )

        labels = ["Abs. Dist.:", "Latitude:", "Longitude:"]
        for i, label in enumerate(labels):
            self.pipe_canvas.create_text(
                mid_x - 320,
                mid_y - 145 + i * 15,
                text=label,
                font=("Arial", 9),
                anchor="w",
            )
            self.pipe_canvas.create_text(
                mid_x - 320,
                mid_y - 30 + i * 15,
                text=label,
                font=("Arial", 9),
                anchor="w",
            )

        for y in [mid_y - 100, mid_y - 60, mid_y + 20, mid_y + 60]:
            self.pipe_canvas.create_line(
                mid_x - 320, y, mid_x + 320, y, width=2
            )

        self.pipe_canvas.create_text(
            mid_x - 310, mid_y - 80, text="U/S", font=("Arial", 9, "bold"), fill="blue"
        )
        self.pipe_canvas.create_text(
            mid_x + 310, mid_y - 80, text="D/S", font=("Arial", 9, "bold"), fill="blue"
        )

        self.pipe_canvas.create_text(
            mid_x - 310,
            mid_y + 40,
            text="L",
            font=("Arial", 9, "bold"),
            fill="deepskyblue",
        )
        self.pipe_canvas.create_text(
            mid_x + 310,
            mid_y + 40,
            text="R",
            font=("Arial", 9, "bold"),
            fill="deepskyblue",
        )

        pipe_info = ["Pipe No:", "Pipe Length(m):", "WT(mm):"]
        for i, label in enumerate(pipe_info):
            self.pipe_canvas.create_text(
                mid_x - 320,
                mid_y + 75 + i * 15,
                text=label,
                font=("Arial", 9),
                anchor="w",
            )

        self.pipe_canvas.create_text(
            mid_x - 315,
            mid_y + 145,
            text="FLOW",
            font=("Arial", 9),
            fill="deepskyblue",
            anchor="w",
        )
        self.pipe_canvas.create_line(
            mid_x - 270,
            mid_y + 160,
            mid_x - 320,
            mid_y + 160,
            arrow=tk.FIRST,
            width=1,
        )

        for i in range(6):
            x1 = mid_x - 240 + i * 80
            x2 = x1 + 80
            self.pipe_canvas.create_rectangle(
                x1, mid_y + 120, x2, mid_y + 180, width=1
            )

    # ======================================================================
    #  CORE ACTIONS (Load, Reset, Fetch Data)
    # ======================================================================

    def load_pipe_tally(self, pipe_tally_file):
        import pandas as pd

        try:
            if pipe_tally_file.endswith(".csv"):
                df = pd.read_csv(pipe_tally_file)
            elif pipe_tally_file.endswith(".xlsx"):
                df = pd.read_excel(pipe_tally_file)
            else:
                raise ValueError("Unsupported file format")

            return df

        except Exception as e:
            print(f"Error loading pipe_tally: {e}")
            traceback.print_exc()
            import sys
            sys.exit(1)

    def on_load_click(self):
        """Main load button: load pipe_tally from file / argv, then refresh the UI for current defect."""
        try:
            if self.pipe_tally_file is None or self.project_root is None:
                # fallback: CLI style
                if len(sys.argv) > 2:
                    self.pipe_tally_file = sys.argv[1]
                    self.project_root = sys.argv[2]
                else:
                    print("No pipe_tally file / project_root provided.")
                    return

            pipe_tally_file = self.pipe_tally_file
            project_root = self.project_root

            csv_path = os.path.join(project_root, "/constants/", "constants.csv")
            xlsx_path = os.path.join(project_root, "/constants/", "constants.xlsx")
            constants_file = csv_path if os.path.exists(csv_path) else xlsx_path
            print(f"constants_file path: {constants_file}")

            self.df = self.load_pipe_tally(pipe_tally_file)

            const_df = pd.read_excel(constants_file, dtype=str)

            import re

            def _norm(s: str) -> str:
                s = re.sub(r"[^A-Za-z0-9]+", " ", str(s))
                return "_".join(s.strip().upper().split())

            colmap = {_norm(c): c for c in const_df.columns}

            def _first_val(*aliases):
                for a in aliases:
                    key = _norm(a)
                    if key in colmap:
                        ser = (
                            const_df[colmap[key]]
                            .dropna()
                            .astype(str)
                            .str.strip()
                        )
                        if not ser.empty:
                            return ser.iloc[0]
                return ""

            print("[constants] columns:", list(const_df.columns))
            print(
                "[constants] picked:",
                "CLIENT->",
                colmap.get("CLIENT_NAME_DESCRIPTION"),
                "PIPELINE_NAME->",
                colmap.get("PIPELINE_NAME_DESCRIPTION"),
                "PIPELINE_SECTION->",
                colmap.get("PIPELINE_SECTION_DESCRIPTION"),
            )

            self.client_var.set(_first_val("CLIENT_NAME_DESCRIPTION"))
            self.pipeline_name_var.set(_first_val("PIPELINE_NAME_DESCRIPTION"))
            self.pipeline_section_var.set(
                _first_val("PIPELINE_SECTION_DESCRIPTION")
            )

        except Exception as e:
            print(f"Error in on_load_click: {e}")

        if self.df is None:
            messagebox.showwarning(
                "Missing Excel File",
                "Please load an Excel file before loading defect data.",
            )
            return

        # everything below is your original "draw dynamic stuff" code
        self._after_load_draw_all()

    def _after_load_draw_all(self):
        """The big dynamic drawing block that was inside on_load_click."""
        df = self.df
        pipe_canvas = self.pipe_canvas
        mid_x = self.mid_x
        mid_y = self.mid_y

        self.fetch_data()

        pipe_canvas.delete("upstream_text")
        pipe_canvas.delete("flange_text")
        pipe_canvas.delete("us_arrow")
        pipe_canvas.delete("ds_arrow")
        pipe_canvas.delete("bend_text")
        pipe_canvas.delete("pipe_icon")

        weld_info = self.get_dynamic_weld_and_feature_data()
        if not weld_info:
            return

        upstream_weld_dist = weld_info["upstream_weld"]
        features_upstream = weld_info["features_upstream"]
        features_downstream = weld_info["features_downstream"]
        bends_upstream = weld_info.get("bends_upstream", [])
        bends_downstream = weld_info.get("bends_downstream", [])

        pipe_canvas.create_text(
            mid_x,
            20,
            text=f"{upstream_weld_dist:.2f}(m)",
            font=("Arial", 10),
            tags="upstream_text",
        )

        feature_slots = [
            {
                "x": mid_x - 190,
                "arrow_x": mid_x - 200,
                "text_x": mid_x - 160,
                "source": features_upstream[::-1],
                "index": 1,
            },
            {
                "x": mid_x - 90,
                "arrow_x": mid_x - 100,
                "text_x": mid_x - 60,
                "source": features_upstream[::-1],
                "index": 0,
            },
            {
                "x": mid_x + 110,
                "arrow_x": mid_x + 120,
                "text_x": mid_x + 80,
                "source": features_downstream,
                "index": 0,
            },
            {
                "x": mid_x + 210,
                "arrow_x": mid_x + 220,
                "text_x": mid_x + 180,
                "source": features_downstream,
                "index": 1,
            },
        ]

        for slot in feature_slots:
            x = slot["x"]
            arrow_x = slot["arrow_x"]
            text_x = slot["text_x"]
            source = slot["source"]
            idx = slot["index"]

            try:
                feature = source[idx]
                name = feature.get("name", "")
                dist_val = feature.get("distance", "")
                lat = feature.get("lat", "")
                lon = feature.get("long", "")

                dist = f"{dist_val}(m)" if pd.notna(dist_val) else ""
                lat = lat if pd.notna(lat) else ""
                lon = lon if pd.notna(lon) else ""

                pipe_canvas.create_text(
                    x,
                    mid_y - 160,
                    text=name,
                    font=("Arial", 10),
                    tags="flange_text",
                )
                pipe_canvas.create_text(
                    x,
                    mid_y - 145,
                    text=dist,
                    font=("Arial", 9),
                    tags="flange_text",
                )
                pipe_canvas.create_text(
                    x,
                    mid_y - 130,
                    text=lat,
                    font=("Arial", 9),
                    tags="flange_text",
                )
                pipe_canvas.create_text(
                    x,
                    mid_y - 115,
                    text=lon,
                    font=("Arial", 9),
                    tags="flange_text",
                )

                arrow_val = round(
                    abs(float(upstream_weld_dist) - float(dist_val)), 2
                )
                pipe_canvas.create_line(
                    arrow_x,
                    mid_y - 95,
                    arrow_x,
                    mid_y - 65,
                    arrow=tk.FIRST,
                    fill="deepskyblue",
                    width=2,
                    tags="us_arrow",
                )
                pipe_canvas.create_text(
                    text_x,
                    mid_y - 80,
                    text=f"{arrow_val}(m)",
                    font=("Arial", 9),
                    tags="us_arrow",
                )
            except Exception:
                continue

        bend_slots = [
            {
                "source": bends_upstream[::-1],
                "index": 2,
                "x_name": mid_x - 230,
                "x_dist": mid_x - 230,
                "x_lat": mid_x - 221,
                "x_lon": mid_x - 221,
                "tri_x": mid_x - 200,
                "arrow_text_x": mid_x - 215,
            },
            {
                "source": bends_upstream[::-1],
                "index": 1,
                "x_name": mid_x - 140,
                "x_dist": mid_x - 140,
                "x_lat": mid_x - 135,
                "x_lon": mid_x - 135,
                "tri_x": mid_x - 110,
                "arrow_text_x": mid_x - 125,
            },
            {
                "source": bends_upstream[::-1],
                "index": 0,
                "x_name": mid_x - 50,
                "x_dist": mid_x - 50,
                "x_lat": mid_x - 50,
                "x_lon": mid_x - 50,
                "tri_x": mid_x - 20,
                "arrow_text_x": mid_x - 35,
            },
            {
                "source": bends_downstream,
                "index": 0,
                "x_name": mid_x + 55,
                "x_dist": mid_x + 55,
                "x_lat": mid_x + 50,
                "x_lon": mid_x + 50,
                "tri_x": mid_x + 110,
                "arrow_text_x": mid_x + 30,
            },
            {
                "source": bends_downstream,
                "index": 1,
                "x_name": mid_x + 155,
                "x_dist": mid_x + 155,
                "x_lat": mid_x + 150,
                "x_lon": mid_x + 150,
                "tri_x": mid_x + 210,
                "arrow_text_x": mid_x + 130,
            },
            {
                "source": bends_downstream,
                "index": 2,
                "x_name": mid_x + 255,
                "x_dist": mid_x + 255,
                "x_lat": mid_x + 250,
                "x_lon": mid_x + 250,
                "tri_x": mid_x + 310,
                "arrow_text_x": mid_x + 230,
            },
        ]

        def draw_triangle(x, y):
            self.pipe_canvas.create_polygon(
                x - 42.5,
                y - 20,
                x - 50,
                y + 18,
                x - 35,
                y + 18,
                fill="deepskyblue",
                outline="gray",
                width=1,
                tags="us_arrow",
            )

        for slot in bend_slots:
            src = slot["source"]
            idx = slot["index"]
            try:
                bend = src[idx]
                name = bend.get("name", "")
                dist_val = bend.get("distance", "")
                lat = bend.get("lat", "")
                lon = bend.get("long", "")

                dist = f"{dist_val}(m)" if pd.notna(dist_val) else ""
                lat = lat if pd.notna(lat) else ""
                lon = lon if pd.notna(lon) else ""

                pipe_canvas.create_text(
                    slot["x_name"],
                    mid_y - 45,
                    text=name,
                    font=("Arial", 10),
                    tags="bend_text",
                )
                pipe_canvas.create_text(
                    slot["x_dist"],
                    mid_y - 30,
                    text=dist,
                    font=("Arial", 9),
                    tags="bend_text",
                )
                pipe_canvas.create_text(
                    slot["x_lat"],
                    mid_y - 15,
                    text=lat,
                    font=("Arial", 9),
                    tags="bend_text",
                )
                pipe_canvas.create_text(
                    slot["x_lon"],
                    mid_y,
                    text=lon,
                    font=("Arial", 9),
                    tags="bend_text",
                )

                draw_triangle(slot["tri_x"], mid_y + 40)
                arrow_val = round(
                    abs(float(upstream_weld_dist) - float(dist_val)), 2
                )
                pipe_canvas.create_text(
                    slot["arrow_text_x"],
                    mid_y + 35,
                    text=f"{arrow_val}",
                    font=("Arial", 9),
                    tags="us_arrow",
                )
                pipe_canvas.create_text(
                    slot["arrow_text_x"],
                    mid_y + 45,
                    text="(m)",
                    font=("Arial", 9),
                    tags="us_arrow",
                )
            except Exception:
                continue

        try:
            s_no = int(self.defect_entry.get())
            defect_row = df[df.iloc[:, 0] == s_no]
            if defect_row.empty:
                messagebox.showwarning(
                    "Warning", f"No defect found for S.No {s_no}"
                )
                return
            pipe_num_defect = int(defect_row.iloc[0, 3])
        except Exception:
            messagebox.showerror("Error", "Invalid or missing defect S.No.")
            return

        target_pipe_numbers = [pipe_num_defect + i for i in range(-3, 3)]
        pipe_data_list = []

        for pno in target_pipe_numbers:
            match = df[df.iloc[:, 3] == pno]
            if not match.empty:
                row = match.iloc[0]
                pipe_no = row[3] if pd.notna(row[3]) else ""
                pipe_len = f"{round(float(row[4]), 3)}" if pd.notna(row[4]) else ""
                pipe_wt = f"{round(float(row[11]), 1)}" if pd.notna(row[11]) else ""
                pipe_data_list.append((str(pipe_no), pipe_len, pipe_wt))
            else:
                pipe_data_list.append(("", "", ""))

        pipe_positions = [-210, -140, -60, 20, 110, 180]
        for i, (pnum, plen, pwt) in enumerate(pipe_data_list):
            px = pipe_positions[i]
            pipe_canvas.create_text(
                mid_x + px,
                mid_y + 75,
                text=pnum,
                font=("Arial", 9),
                anchor="w",
                tags="us_arrow",
            )
            pipe_canvas.create_text(
                mid_x + px,
                mid_y + 90,
                text=plen,
                font=("Arial", 9),
                anchor="w",
                tags="us_arrow",
            )
            pipe_canvas.create_text(
                mid_x + px,
                mid_y + 105,
                text=pwt,
                font=("Arial", 9),
                anchor="w",
                tags="us_arrow",
            )

        try:
            defect_row = defect_row.iloc[0]
            upstream_dist = (
                f"{round(float(defect_row.iloc[2]), 2)}"
                if pd.notna(defect_row.iloc[2])
                else ""
            )
            clock_pos = (
                f"{(defect_row.iloc[8])}"
                if pd.notna(defect_row.iloc[8])
                else ""
            )
            pipe_len = (
                f"{round((defect_row.iloc[4]), 3)}"
                if pd.notna(defect_row.iloc[4])
                else ""
            )

            if pipe_len and upstream_dist:
                pipe_length = float(pipe_len)
                upstream = float(upstream_dist)
                clock_angle = self.hms_to_angle(clock_pos)

                box_x_start = mid_x
                box_x_end = mid_x + 80
                box_y_top = mid_y + 120
                box_y_bottom = mid_y + 190

                if upstream < pipe_length / 3:
                    defect_x = box_x_start + 15
                elif upstream < 2 * pipe_length / 3:
                    defect_x = (box_x_start + box_x_end) / 2
                else:
                    defect_x = box_x_end - 15

                if 0 <= clock_angle <= 60 or 300 < clock_angle <= 360:
                    defect_y = box_y_top + 10
                elif (
                    60 < clock_angle <= 120
                    or 240 <= clock_angle <= 300
                ):
                    defect_y = (box_y_top + box_y_bottom) / 2
                else:
                    defect_y = box_y_bottom - 10

                if 0 <= clock_angle <= 180:
                    pipe_canvas.create_rectangle(
                        defect_x - 3,
                        defect_y - 3,
                        defect_x + 3,
                        defect_y + 3,
                        fill="orange",
                        outline="black",
                        tags="us_arrow",
                    )
                else:
                    pipe_canvas.create_rectangle(
                        defect_x - 3,
                        defect_y - 3,
                        defect_x + 3,
                        defect_y + 3,
                        outline="orange",
                        width=2,
                        tags="us_arrow",
                    )
        except Exception as e:
            print("Bottom pipe defect box drawing error:", e)
            traceback.print_exc()

        pipe_box_centers = [
            (mid_x - 200, mid_y + 155),
            (mid_x - 120, mid_y + 155),
            (mid_x - 40, mid_y + 155),
            (mid_x + 40, mid_y + 155),
            (mid_x + 120, mid_y + 155),
            (mid_x + 200, mid_y + 155),
        ]

        for i, pipe_num in enumerate(target_pipe_numbers):
            matching_rows = df[df.iloc[:, 3] == pipe_num]
            if not matching_rows.empty:
                found_features = []
                for _, row in matching_rows.iterrows():
                    feature_text = str(row.iloc[5]).lower()
                    if "valve" in feature_text and "valve" not in found_features:
                        found_features.append("valve")
                    if "flow" in feature_text or "tee" in feature_text:
                        if "flowtee" not in found_features:
                            found_features.append("flowtee")
                    if "flange" in feature_text and "flange" not in found_features:
                        found_features.append("flange")
                    if "bend" in feature_text and "bend" not in found_features:
                        found_features.append("bend")
                    if "magnet" in feature_text and "magnet" not in found_features:
                        found_features.append("magnet")

                cx, cy = pipe_box_centers[i]
                spacing = 22

                for j, feat in enumerate(found_features):
                    offset_y = (
                        cy
                        - (len(found_features) - 1) * spacing // 2
                        + j * spacing
                    )

                    if feat == "valve" and self.valve_img:
                        pipe_canvas.create_image(
                            cx, offset_y, image=self.valve_img, tags="pipe_icon"
                        )
                    elif feat == "flowtee" and self.flowtee_img:
                        pipe_canvas.create_image(
                            cx, offset_y, image=self.flowtee_img, tags="pipe_icon"
                        )
                    elif feat == "flange" and self.flange_img:
                        pipe_canvas.create_image(
                            cx, offset_y, image=self.flange_img, tags="pipe_icon"
                        )
                    elif feat == "bend" and self.bend_img:
                        pipe_canvas.create_image(
                            cx, offset_y, image=self.bend_img, tags="pipe_icon"
                        )
                    elif feat == "magnet" and self.magnet_img:
                        pipe_canvas.create_image(
                            cx, offset_y, image=self.magnet_img, tags="pipe_icon"
                        )

    def reset_ui(self):
        """Return the app to its just-opened state."""
        self.batch_cancelled = False

        try:
            self.defect_entry.delete(0, tk.END)
        except Exception:
            pass

        for var in (
            self.pipe_id_var,
            self.length_var,
            self.wt_var,
            self.latitude_var,
            self.longitude_var,
            self.altitude_var,
            self.client_var,
            self.pipeline_name_var,
            self.pipeline_section_var,
        ):
            try:
                var.set("")
            except Exception:
                pass

        for lbl in self.feature_labels.values():
            try:
                lbl.config(text="")
            except Exception:
                pass

        try:
            self.comment_placeholder.config(text="")
        except Exception:
            pass

        try:
            self.pipe_canvas1.delete("all")
        except Exception:
            pass

        for tag in (
            "upstream_text",
            "flange_text",
            "us_arrow",
            "ds_arrow",
            "bend_text",
            "pipe_icon",
        ):
            try:
                self.pipe_canvas.delete(tag)
            except Exception:
                pass

        try:
            self._clear_preview_holder()
        except Exception:
            pass

        try:
            if self.progress_frame_ref and self.progress_frame_ref.winfo_exists():
                self.progress_frame_ref.destroy()
            self.progress_frame_ref = None
        except Exception:
            pass

        try:
            self.canvas.yview_moveto(0.0)
        except Exception:
            pass

        print("[reset] UI returned to initial state.")

    def reset_left_panel(self):
        """Reset only the main (left) digsheet area, keep right panel."""
        for var in (
            self.pipe_id_var,
            self.length_var,
            self.wt_var,
            self.latitude_var,
            self.longitude_var,
            self.altitude_var,
            self.client_var,
            self.pipeline_name_var,
            self.pipeline_section_var,
        ):
            try:
                var.set("")
            except Exception:
                pass

        for lbl in self.feature_labels.values():
            try:
                lbl.config(text="")
            except Exception:
                pass

        try:
            self.comment_placeholder.config(text="")
        except Exception:
            pass

        try:
            self.pipe_canvas1.delete("all")
        except Exception:
            pass

        for tag in (
            "upstream_text",
            "flange_text",
            "us_arrow",
            "ds_arrow",
            "bend_text",
            "pipe_icon",
        ):
            try:
                self.pipe_canvas.delete(tag)
            except Exception:
                pass

        try:
            self.canvas.yview_moveto(0.0)
        except Exception:
            pass

        print("[reset] Left panel cleared (preview kept).")

    # ======================================================================
    #  SAVE / PRINT / CAPTURE HELPERS
    # ======================================================================

    def open_save_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Save")
        dlg.geometry("300x160+520+260")
        dlg.configure(bg="white")
        dlg.grab_set()

        tk.Label(
            dlg, text="Save as:", bg="white", font=("Segoe UI", 11, "bold")
        ).pack(pady=(12, 6))

        mode_var = tk.StringVar(value="png")
        opts = tk.Frame(dlg, bg="white")
        opts.pack(pady=4)
        tk.Radiobutton(
            opts,
            text="PNG (image)",
            variable=mode_var,
            value="png",
            bg="white",
        ).grid(row=0, column=0, padx=10)
        tk.Radiobutton(
            opts,
            text="PDF (single page)",
            variable=mode_var,
            value="pdf",
            bg="white",
        ).grid(row=0, column=1, padx=10)

        def do_save():
            dlg.destroy()
            if mode_var.get() == "png":
                self.capture_sections(1, 5)
            else:
                self.save_all_sections_as_pdf()

        btns = tk.Frame(dlg, bg="white")
        btns.pack(pady=14)
        tk.Button(btns, text="Save", command=do_save).grid(row=0, column=0, padx=10)
        tk.Button(btns, text="Cancel", command=dlg.destroy).grid(
            row=0, column=1, padx=10
        )

    def print_all_sections_dialog(self):
        merged = self.capture_sections_image(1, 5)
        if merged is None:
            messagebox.showerror("Error", "No sections captured")
            return

        temp_img = tempfile.mktemp(suffix=".png")
        merged.save(temp_img, "PNG")

        def get_printers():
            printers = [
                p[2]
                for p in win32print.EnumPrinters(
                    win32print.PRINTER_ENUM_LOCAL
                    | win32print.PRINTER_ENUM_CONNECTIONS
                )
            ]
            return printers

        def send_to_printer(printer_name, file_path):
            try:
                win32api.ShellExecute(
                    0, "print", file_path, f'"{printer_name}"', ".", 0
                )
                messagebox.showinfo(
                    "Print", f"Sent to printer: {printer_name}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to print:\n{e}")

        def print_selected():
            selection = printer_combo.get()
            if not selection:
                messagebox.showwarning(
                    "Warning", "Please select a printer"
                )
                return
            send_to_printer(selection, temp_img)
            dialog.destroy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Print Report")
        dialog.geometry("400x200")
        dialog.configure(bg="white")
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Select a Printer",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="black",
        ).pack(pady=(15, 10))

        printers = get_printers()
        printer_combo = ttk.Combobox(
            dialog, values=printers, state="readonly", width=40
        )
        if printers:
            printer_combo.current(0)
        printer_combo.pack(pady=10)

        button_frame = tk.Frame(dialog, bg="white")
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Print", command=print_selected).grid(
            row=0, column=0, padx=10
        )
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(
            row=0, column=1, padx=10
        )

        dialog.mainloop()

    def get_section_coords(self):
        self.root.update_idletasks()
        sections = {
            "Client Description": self.client_desc_frame,
            "Feature Location on Pipe": self.main_frame,
            "Comment": self.comment_frame,
            "Feature Description": self.feature_desc_frame,
            "Pipe Location": self.third_frame,
        }
        coords = {}
        for name, frame in sections.items():
            if frame is None:
                continue
            x0 = frame.winfo_rootx()
            y0 = frame.winfo_rooty()
            x1 = x0 + frame.winfo_width()
            y1 = y0 + frame.winfo_height()
            coords[name] = (x0, y0, x1, y1)
        return coords

    def capture_sections(self, section_start=1, section_end=5):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialfile="",
        )
        if not filepath:
            return

        images = []
        for section_id in range(section_start, section_end + 1):
            if section_id not in self.SECTION_MAP:
                continue

            if section_id in [1, 2, 3, 4]:
                self.canvas.yview_moveto(0.0)
            elif section_id == 5:
                self.canvas.yview_moveto(1.0)
            self.root.update()
            time.sleep(0.4)

            coords = self.get_section_coords()
            name = self.SECTION_MAP[section_id]
            if name not in coords:
                continue

            x0, y0, x1, y1 = coords[name]
            dx0, dy0, dx1, dy1 = self.SECTION_THRESHOLDS.get(
                name, (0, 0, 0, 0)
            )
            bbox = (x0 + dx0, y0 + dy0, x1 + dx1, y1 + dy1)
            img = ImageGrab.grab(bbox=bbox)
            images.append(img)

        if not images:
            messagebox.showerror("Error", "No sections were captured.")
            return

        widths = [im.width for im in images]
        heights = [im.height for im in images]
        max_w = max(widths)
        total_h = sum(heights)

        merged = Image.new("RGB", (max_w, total_h), "white")
        y_offset = 0
        for im in images:
            if im.width != max_w:
                im = im.resize((max_w, im.height))
            merged.paste(im, (0, y_offset))
            y_offset += im.height

        merged.save(filepath)
        messagebox.showinfo(
            "Saved!", f"All sections saved successfully:\n{filepath}"
        )
        print(f"✅ Combined image saved to {filepath}")

    def capture_sections_image(self, section_start=1, section_end=5):
        images = []
        for section_id in range(section_start, section_end + 1):
            if section_id not in self.SECTION_MAP:
                continue

            self.canvas.yview_moveto(0.0 if section_id in [1, 2, 3, 4] else 1.0)
            self.root.update()
            time.sleep(0.4)

            coords = self.get_section_coords()
            name = self.SECTION_MAP[section_id]
            if name not in coords:
                continue

            x0, y0, x1, y1 = coords[name]
            dx0, dy0, dx1, dy1 = self.SECTION_THRESHOLDS.get(
                name, (0, 0, 0, 0)
            )
            bbox = (x0 + dx0, y0 + dy0, x1 + dx1, y1 + dy1)
            img = ImageGrab.grab(bbox=bbox).convert("RGB")
            images.append(img)

        if not images:
            return None

        max_w = max(im.width for im in images)
        total_h = sum(im.height for im in images)
        merged = Image.new("RGB", (max_w, total_h), "white")
        y = 0
        for im in images:
            if im.width != max_w:
                im = im.resize((max_w, im.height))
            merged.paste(im, (0, y))
            y += im.height
        return merged

    def upscale_image(self, img, target_dpi=600, base_dpi=96, scale_limit=2.0):
        scale = target_dpi / base_dpi
        if scale > scale_limit:
            scale = scale_limit
        new_size = (int(img.width * scale), int(img.height * scale))
        return img.resize(new_size, Image.LANCZOS), target_dpi

    def save_all_sections_as_pdf(self):
        merged = self.capture_sections_image(1, 5)
        if merged is None:
            messagebox.showerror("Error", "No sections were captured.")
            return

        pdf_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile="",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not pdf_path:
            return

        merged, dpi = self.upscale_image(merged, target_dpi=300, base_dpi=96)

        buf = io.BytesIO()
        merged.save(buf, format="PNG", dpi=(dpi, dpi))
        buf.seek(0)

        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(buf.getvalue()))

        messagebox.showinfo(
            "Saved!", f"High-quality PDF created:\n{pdf_path}"
        )

    # ======================================================================
    #  FETCH / DRAW PIPE
    # ======================================================================

    def hms_to_angle(self, hms):
        if isinstance(hms, str):
            h, m, s = map(int, hms.split(":"))
        else:
            h, m, s = hms.hour, hms.minute, hms.second
        angle = (h % 12) * 30 + m * 0.5 + s * (0.5 / 60)
        return angle

    def draw_pipe(self, pipe_length, upstream, clock):
        self.pipe_canvas1.delete("all")
        width, height = 320, 120
        x0, y0 = 40, 30
        x1, y1 = x0 + width, y0 + height
        mid_x = (x0 + x1) // 2
        mid_y = (y0 + y1) // 2
        radius = height // 2 - 10

        self.pipe_canvas1.create_oval(
            x0, y0, x0 + 40, y1, outline="black", width=2
        )
        self.pipe_canvas1.create_oval(
            x1 - 40, y0, x1, y1, outline="black", width=2
        )
        self.pipe_canvas1.create_line(
            x0 + 20, y0, x1 - 20, y0, fill="black", width=2
        )
        self.pipe_canvas1.create_line(
            x0 + 20, y1, x1 - 20, y1, fill="black", width=2
        )

        self.pipe_canvas1.create_line(
            x0, mid_y - 5, x1, mid_y - 5, fill="black", dash=(3, 3)
        )

        self.pipe_canvas1.create_text(
            x0 - 20, y0 + 10, text="12", anchor="w", font=("Arial", 10)
        )
        self.pipe_canvas1.create_text(
            x0 + 25, mid_y + 5, text="3", anchor="w", font=("Arial", 10)
        )
        self.pipe_canvas1.create_text(
            x0 - 17, y1 - 5, text="6", anchor="w", font=("Arial", 10)
        )
        self.pipe_canvas1.create_text(
            x0 - 10, mid_y + 5, text="9", anchor="e", font=("Arial", 10)
        )

        try:
            upstream = float(upstream) if upstream else 0.0
            pipe_length = float(pipe_length) if pipe_length else 0.0
            remaining = round(pipe_length - upstream, 2)
        except Exception:
            upstream = 0.0
            remaining = 0.0

        try:
            arrow_y = y0 - 15
            scale_factor = 0.85
            arrow_length_total = (x1 - x0) * scale_factor
            offset = ((x1 - x0) - arrow_length_total) / 2
            arrow_start_x = x0 + offset
            arrow_end_x = x1 - offset

            arrow1_length = (
                (upstream / pipe_length) * arrow_length_total
                if pipe_length > 0
                else arrow_length_total / 2
            )
            arrow2_length = arrow_length_total - arrow1_length

            arrow1_start = arrow_start_x
            arrow1_end = arrow1_start + arrow1_length
            self.pipe_canvas1.create_line(
                arrow1_start,
                arrow_y,
                arrow1_end,
                arrow_y,
                arrow=tk.LAST,
            )
            self.pipe_canvas1.create_line(
                arrow1_end,
                arrow_y,
                arrow1_start,
                arrow_y,
                arrow=tk.LAST,
            )
            self.pipe_canvas1.create_text(
                (arrow1_start + arrow1_end) / 2,
                arrow_y - 10,
                text=f"{round(upstream, 2)} m",
                font=("Arial", 10),
            )

            arrow2_start = arrow1_end
            arrow2_end = arrow_end_x
            self.pipe_canvas1.create_line(
                arrow2_start,
                arrow_y,
                arrow2_end,
                arrow_y,
                arrow=tk.LAST,
            )
            self.pipe_canvas1.create_line(
                arrow2_end,
                arrow_y,
                arrow2_start,
                arrow_y,
                arrow=tk.LAST,
            )
            self.pipe_canvas1.create_text(
                (arrow2_start + arrow2_end) / 2,
                arrow_y - 10,
                text=f"{remaining} m",
                font=("Arial", 10),
            )

            angle_deg = self.hms_to_angle(clock)
            angle_rad = math.radians(angle_deg)

            radius_y = radius
            center_y = mid_y

            defect_x = arrow1_start + (
                (upstream / pipe_length) * arrow_length_total
                if pipe_length > 0
                else arrow_length_total / 2
            )
            adjusted_radius = radius * 0.8
            defect_y = center_y - int(adjusted_radius * math.cos(angle_rad))

            if 0 <= angle_deg <= 180:
                self.pipe_canvas1.create_rectangle(
                    defect_x - 4,
                    defect_y - 4,
                    defect_x + 4,
                    defect_y + 4,
                    fill="orange",
                    outline="black",
                )
            else:
                self.pipe_canvas1.create_rectangle(
                    defect_x - 4,
                    defect_y - 4,
                    defect_x + 4,
                    defect_y + 4,
                    outline="orange",
                    width=2,
                )

            self.pipe_canvas1.create_line(
                defect_x - 5,
                defect_y,
                defect_x - 5,
                y0,
                arrow=tk.LAST,
                fill="black",
                width=1.5,
            )
        except Exception as e:
            print("Drawing error:", e)

    def fetch_data(self):
        """Fill left-side text labels from df based on current S.no."""
        if self.df is None:
            return
        df = self.df
        try:
            s_no = int(self.defect_entry.get())
            row = df[df.iloc[:, 0] == s_no]
            if row.empty:
                messagebox.showerror("Error", "Defect number not found!")
                return
            row = row.iloc[0]

            self.pipe_id_var.set(str(row.iloc[3]))
            self.length_var.set(str(row.iloc[4]))
            self.wt_var.set(str(row.iloc[11]))

            lat_col = next(
                (c for c in df.columns if c.strip().lower() == "latitude"), None
            )
            lon_col = next(
                (c for c in df.columns if c.strip().lower() == "longitude"), None
            )
            alt_col = next(
                (c for c in df.columns if c.strip().lower() == "altitude"), None
            )

            self.latitude_var.set(
                str(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else ""
            )
            self.longitude_var.set(
                str(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else ""
            )
            self.altitude_var.set(
                str(row[alt_col]) if alt_col and pd.notna(row[alt_col]) else ""
            )

            upstream = row.iloc[2]
            clock_raw = row.iloc[8]
            self.draw_pipe(row.iloc[4], upstream, clock_raw)

            columns_clean = {
                col.strip().lower().replace(" ", ""): col for col in df.columns
            }
            latitude_col = columns_clean.get("latitude", None)
            longitude_col = columns_clean.get("longitude", None)

            excel_mapping = {
                "Feature": 5,
                "Feature type": 6,
                "Anomaly dimension class": 7,
                "Surface Location": 14,
                "Remaining wall thickness (mm)": None,
                "ERF": 15,
                "Safe pressure (kg/cm²)": 16,
                "Absolute Distance (m)": 1,
                "Length (mm)": 9,
                "Width (mm)": 10,
                "Max. Depth(%)": 12,
                "Orientation(hr:min)": 8,
                "Latitude": None,
                "Longitude": None,
            }

            for label, col_index in excel_mapping.items():
                if col_index is not None:
                    value = row.iloc[col_index] if col_index < len(row) else ""

                    if label in ["Length (mm)", "Width (mm)", "Max. Depth(%)"]:
                        try:
                            value = (
                                int(float(value)) if pd.notna(value) else ""
                            )
                        except Exception:
                            value = ""
                    elif label == "ERF":
                        try:
                            value = (
                                f"{float(value):.3f}"
                                if pd.notna(value)
                                else ""
                            )
                        except Exception:
                            value = ""
                    elif label == "Orientation(hr:min)":
                        try:
                            if isinstance(value, str) and ":" in value:
                                value = ":".join(value.split(":")[:2])
                            elif isinstance(value, datetime.time):
                                value = value.strftime("%H:%M")
                            else:
                                value = str(value)
                        except Exception:
                            value = ""

                    self.feature_labels[label].config(text=str(value))

            try:
                wt = float(row.iloc[11])
                max_depth = float(row.iloc[12])
                remaining_wt = round(wt - (wt * max_depth / 100), 1)
            except Exception:
                remaining_wt = ""

            self.feature_labels["Remaining wall thickness (mm)"].config(
                text=str(remaining_wt)
            )

            lat_val = row.get(latitude_col, "") if latitude_col else ""
            lon_val = row.get(longitude_col, "") if longitude_col else ""
            self.feature_labels["Latitude"].config(text=str(lat_val))
            self.feature_labels["Longitude"].config(text=str(lon_val))

        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid S.no")

    def get_dynamic_weld_and_feature_data(self):
        try:
            feature_keywords = ["flange", "valve", "flow tee", "magnet"]
            df = self.df
            if df is None:
                return None

            s_no = int(self.defect_entry.get())
            row = df[df.iloc[:, 0] == s_no]
            if row.empty:
                messagebox.showerror("Error", "Defect number not found!")
                return
            row = row.iloc[0]
            upstream_value = float(row.iloc[2])
            absolute_value = float(row.iloc[1])
            upstream_weld = round(abs(absolute_value - upstream_value), 2)

            defect_idx = df[df.iloc[:, 0] == s_no].index[0]
            defect_row = df.loc[defect_idx]
            defect_distance = float(defect_row.iloc[1])

            lat_col = next(
                (c for c in df.columns if c.strip().lower() == "latitude"), None
            )
            lon_col = next(
                (c for c in df.columns if c.strip().lower() == "longitude"), None
            )

            features_upstream = []
            features_downstream = []
            bends_upstream = []
            bends_downstream = []

            for i in range(defect_idx - 1, -1, -1):
                row = df.loc[i]
                feature_name = str(row.iloc[5]).strip().lower()
                if any(f in feature_name for f in feature_keywords):
                    features_upstream.append(
                        {
                            "name": str(row.iloc[5]),
                            "distance": round(float(row.iloc[1]), 2),
                            "lat": str(row[lat_col])
                            if lat_col and pd.notna(row[lat_col])
                            else "",
                            "long": str(row[lon_col])
                            if lon_col and pd.notna(row[lon_col])
                            else "",
                        }
                    )
                    if len(features_upstream) == 2:
                        break

            for i in range(defect_idx + 1, len(df)):
                row = df.loc[i]
                feature_name = str(row.iloc[5]).strip().lower()
                if any(f in feature_name for f in feature_keywords):
                    features_downstream.append(
                        {
                            "name": str(row.iloc[5]),
                            "distance": round(float(row.iloc[1]), 2),
                            "lat": str(row[lat_col])
                            if lat_col and pd.notna(row[lat_col])
                            else "",
                            "long": str(row[lon_col])
                            if lon_col and pd.notna(row[lon_col])
                            else "",
                        }
                    )
                    if len(features_downstream) == 2:
                        break

            for i in range(defect_idx - 1, -1, -1):
                row = df.loc[i]
                feature_name = str(row.iloc[5]).strip().lower()
                if "bend" in feature_name:
                    bends_upstream.append(
                        {
                            "name": str(row.iloc[5]),
                            "distance": round(float(row.iloc[1]), 2),
                            "lat": str(row[lat_col])
                            if lat_col and pd.notna(row[lat_col])
                            else "",
                            "long": str(row[lon_col])
                            if lon_col and pd.notna(row[lon_col])
                            else "",
                        }
                    )
                    if len(bends_upstream) == 3:
                        break

            for i in range(defect_idx + 1, len(df)):
                row = df.loc[i]
                feature_name = str(row.iloc[5]).strip().lower()
                if "bend" in feature_name:
                    bends_downstream.append(
                        {
                            "name": str(row.iloc[5]),
                            "distance": round(float(row.iloc[1]), 2),
                            "lat": str(row[lat_col])
                            if lat_col and pd.notna(row[lat_col])
                            else "",
                            "long": str(row[lon_col])
                            if lon_col and pd.notna(row[lon_col])
                            else "",
                        }
                    )
                    if len(bends_downstream) == 3:
                        break

            return {
                "upstream_weld": upstream_weld,
                "features_upstream": features_upstream,
                "features_downstream": features_downstream,
                "bends_upstream": bends_upstream,
                "bends_downstream": bends_downstream,
            }
        except Exception:
            return {
                "upstream_weld": "",
                "features_upstream": "",
                "features_downstream": "",
                "bends_upstream": "",
                "bends_downstream": "",
            }

    # ======================================================================
    #  Preview / Batch Export UI (right panel)
    # ======================================================================

    def _clear_preview_holder(self):
        try:
            self.preview_holder.unbind_all("<Left>")
            self.preview_holder.unbind_all("<Right>")
        except Exception:
            pass

        for w in self.preview_holder.winfo_children():
            w.destroy()

    def _show_preview_placeholder(self, msg="No previews yet.\nUse MultiPreview to generate."):
        self._clear_preview_holder()
        self._preview_placeholder_ref = tk.Label(
            self.preview_holder,
            text=msg,
            bg="white",
            fg="gray50",
            font=("Segoe UI", 11, "bold"),
            justify="center",
        )
        self._preview_placeholder_ref.place(relx=0.5, rely=0.5, anchor="center")

    def _start_panel_progress(self, total, title="Generating previews"):
        self._clear_preview_holder()

        prog_frame = tk.Frame(
            self.preview_holder,
            bg="white",
            highlightbackground="#e8e8e8",
            highlightthickness=1,
        )
        prog_frame.pack(side="top", fill="x", padx=8, pady=8)

        tk.Label(
            prog_frame,
            text=title,
            bg="white",
            fg="deepskyblue",
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=(10, 6))

        status_lbl = tk.Label(
            prog_frame, text=f"0 / {total}", bg="white", font=("Segoe UI", 10)
        )
        status_lbl.pack(pady=(0, 8))

        bar_wrap = tk.Frame(prog_frame, bg="white")
        bar_wrap.pack(pady=(0, 12))

        prog_var = tk.IntVar(value=0)
        prog_bar = ttk.Progressbar(
            bar_wrap,
            maximum=total,
            variable=prog_var,
            length=320,
            mode="determinate",
            style="Custom.Horizontal.TProgressbar",
        )
        prog_bar.pack()

        def _update(done):
            prog_var.set(done)
            status_lbl.config(text=f"{done} / {total}")
            prog_frame.update_idletasks()

        def _finish():
            prog_frame.destroy()
            self.preview_holder.update_idletasks()

        return _update, _finish

    def open_preview_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Multi Preview")
        dialog.geometry("340x280+500+200")
        dialog.configure(bg="white")
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Select defects to preview",
            bg="white",
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=10)

        range_frame = tk.Frame(dialog, bg="white")
        range_frame.pack(pady=5)

        tk.Label(range_frame, text="Start ID:", bg="white").grid(
            row=0, column=0, padx=5
        )
        start_var = tk.StringVar()
        tk.Entry(range_frame, textvariable=start_var, width=8).grid(
            row=0, column=1, padx=5
        )

        tk.Label(range_frame, text="End ID:", bg="white").grid(
            row=0, column=2, padx=5
        )
        end_var = tk.StringVar()
        tk.Entry(range_frame, textvariable=end_var, width=8).grid(
            row=0, column=3, padx=5
        )

        tk.Label(
            dialog,
            text="OR Enter IDs (comma-separated):",
            bg="white",
        ).pack(pady=(15, 2))
        custom_var = tk.StringVar()
        tk.Entry(dialog, textvariable=custom_var, width=30).pack(pady=2)

        mode_var = tk.StringVar(value="png")
        mode_frame = tk.Frame(dialog, bg="white")
        mode_frame.pack(pady=10)

        tk.Label(mode_frame, text="Preview as:", bg="white").grid(
            row=0, column=0, padx=5
        )
        tk.Radiobutton(
            mode_frame, text="PNG", variable=mode_var, value="png", bg="white"
        ).grid(row=0, column=1, padx=5)
        tk.Radiobutton(
            mode_frame, text="PDF", variable=mode_var, value="pdf", bg="white"
        ).grid(row=0, column=2, padx=5)

        def run_preview():
            ids = []
            try:
                if start_var.get() and end_var.get():
                    s, e = int(start_var.get()), int(end_var.get())
                    ids.extend(range(s, e + 1))
                if custom_var.get():
                    for part in custom_var.get().split(","):
                        part = part.strip()
                        if part:
                            ids.append(int(part))
                if not ids:
                    messagebox.showwarning(
                        "Multi Preview", "Please enter a range or some IDs."
                    )
                    return
                ids = sorted(set(ids))
                dialog.destroy()
                self.root.after(
                    200,
                    lambda: self.batch_preview(
                        ids,
                        mode_var.get(),
                        embed=(mode_var.get().lower() == "png"),
                    ),
                )
            except ValueError:
                messagebox.showerror(
                    "Error", "Invalid input. Please use numbers only."
                )

        tk.Button(dialog, text="Preview", command=run_preview).pack(pady=15)
        tk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)

    def open_batch_dialog_new(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Export")
        dialog.geometry("360x280+500+200")
        dialog.configure(bg="white")
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Select defects to export",
            bg="white",
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=10)

        range_frame = tk.Frame(dialog, bg="white")
        range_frame.pack(pady=5)

        tk.Label(range_frame, text="Start ID:", bg="white").grid(
            row=0, column=0, padx=5
        )
        start_var = tk.StringVar()
        tk.Entry(range_frame, textvariable=start_var, width=8).grid(
            row=0, column=1, padx=5
        )

        tk.Label(range_frame, text="End ID:", bg="white").grid(
            row=0, column=2, padx=5
        )
        end_var = tk.StringVar()
        tk.Entry(range_frame, textvariable=end_var, width=8).grid(
            row=0, column=3, padx=5
        )

        tk.Label(
            dialog,
            text="OR Enter IDs (comma-separated):",
            bg="white",
        ).pack(pady=(15, 2))
        custom_var = tk.StringVar()
        tk.Entry(dialog, textvariable=custom_var, width=32).pack(pady=2)

        mode_var = tk.StringVar(value="pdf")
        mode_frame = tk.Frame(dialog, bg="white")
        mode_frame.pack(pady=12)
        tk.Label(mode_frame, text="Export as:", bg="white").grid(
            row=0, column=0, padx=(0, 8)
        )
        tk.Radiobutton(
            mode_frame,
            text="PDF (one multi-page file)",
            variable=mode_var,
            value="pdf",
            bg="white",
        ).grid(row=0, column=1, padx=6)
        tk.Radiobutton(
            mode_frame,
            text="PNG (one file per defect)",
            variable=mode_var,
            value="png",
            bg="white",
        ).grid(row=0, column=2, padx=6)

        def run_export():
            ids = []
            try:
                if start_var.get() and end_var.get():
                    s, e = int(start_var.get()), int(end_var.get())
                    ids.extend(range(s, e + 1))
                if custom_var.get():
                    for part in custom_var.get().split(","):
                        part = part.strip()
                        if part:
                            ids.append(int(part))
                if not ids:
                    messagebox.showwarning(
                        "Batch Export", "Please enter a range or some IDs."
                    )
                    return

                ids = sorted(set(ids))
                export_mode = mode_var.get()

                dialog.destroy()
                self.root.after(
                    200, lambda: self.batch_export_with_ui(ids, export_mode)
                )
            except ValueError:
                messagebox.showerror(
                    "Error", "Invalid input. Please use numbers only."
                )

        btns = tk.Frame(dialog, bg="white")
        btns.pack(pady=16)
        tk.Button(btns, text="Export", command=run_export).grid(
            row=0, column=0, padx=10
        )
        tk.Button(btns, text="Cancel", command=dialog.destroy).grid(
            row=0, column=1, padx=10
        )

    def _show_preview_in_panel(self, images):
        self._clear_preview_holder()

        header = tk.Frame(self.preview_holder, bg="white")
        header.pack(fill="x", pady=(8, 6))

        current_idx = tk.IntVar(value=0)
        page_lbl = tk.Label(
            header, text="", bg="white", font=("Segoe UI", 10, "bold")
        )
        page_lbl.pack(side="left", padx=8)

        body = tk.Frame(self.preview_holder, bg="white")
        body.pack(fill="both", expand=True)

        vbar = tk.Scrollbar(body, orient="vertical")
        hbar = tk.Scrollbar(body, orient="horizontal")
        canvas_prev = tk.Canvas(
            body,
            bg="white",
            highlightthickness=0,
            yscrollcommand=vbar.set,
            xscrollcommand=hbar.set,
        )
        vbar.config(command=canvas_prev.yview)
        hbar.config(command=canvas_prev.xview)
        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")
        canvas_prev.pack(side="left", fill="both", expand=True)

        img_refs = []

        def render(idx):
            dno, im = images[idx]
            avail_w = max(1, canvas_prev.winfo_width() - 10)
            avail_h = max(1, canvas_prev.winfo_height() - 10)
            r = min(avail_w / im.width, avail_h / im.height, 1.0)
            new_w, new_h = int(im.width * r), int(im.height * r)
            im_resized = im.resize((new_w, new_h), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(im_resized)
            img_refs.clear()
            img_refs.append(tk_img)

            canvas_prev.delete("all")
            canvas_prev.create_image(0, 0, image=tk_img, anchor="nw")
            canvas_prev.config(scrollregion=(0, 0, new_w, new_h))
            page_lbl.config(text=f"S. No {dno}  ({idx+1}/{len(images)})")

        def _nav(delta):
            i = current_idx.get() + delta
            if 0 <= i < len(images):
                current_idx.set(i)
                render(i)

        self.preview_holder.bind_all("<Left>", lambda e: _nav(-1))
        self.preview_holder.bind_all("<Right>", lambda e: _nav(+1))

        def save_current():
            idx = current_idx.get()
            dno, im = images[idx]
            p = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=f"digsheet_{dno}.png",
                filetypes=[("PNG", "*.png")],
            )
            if p:
                im.save(p, "PNG")
                messagebox.showinfo("Saved", f"Saved {p}")

        def save_all():
            folder = filedialog.askdirectory()
            if not folder:
                return
            for dno, im in images:
                im.save(os.path.join(folder, f"digsheet_{dno}.png"), "PNG")
            messagebox.showinfo(
                "Saved", f"Exported {len(images)} PNGs to:\n{folder}"
            )

        tk.Button(header, text="Next ⟶", command=lambda: _nav(+1)).pack(
            side="right", padx=4
        )
        tk.Button(header, text="⟵ Prev", command=lambda: _nav(-1)).pack(
            side="right", padx=4
        )

        tk.Button(
            header, text="💾 Save Current", command=save_current
        ).pack(side="right", padx=8)
        tk.Button(header, text="💾 Save All", command=save_all).pack(
            side="right", padx=4
        )

        canvas_prev.bind("<Configure>", lambda e: render(current_idx.get()))
        render(0)

    def batch_preview(self, defect_ids, mode="png", embed=False):
        if not defect_ids:
            messagebox.showwarning("Preview", "No defect IDs provided.")
            return

        update_prog, finish_prog = self._start_panel_progress(
            len(defect_ids), title="Generating previews"
        )

        images = []
        done = 0
        for dno in defect_ids:
            try:
                self.defect_entry.delete(0, tk.END)
                self.defect_entry.insert(0, str(dno))
                self.on_load_click()
                self.root.update()
                time.sleep(0.3)

                merged = self.capture_sections_image(1, 5)
                if merged:
                    images.append((dno, merged))
            except Exception as e:
                print(f"[Preview error] Defect {dno}: {e}")
            finally:
                done += 1
                update_prog(done)

        finish_prog()

        if not images:
            self._show_preview_placeholder(
                "No previews generated.\nCheck your IDs and try again."
            )
            messagebox.showerror("Preview", "No images generated.")
            return

        if str(mode).lower() == "pdf":
            tmp_paths = []
            try:
                for dno, im in images:
                    tmp_path = os.path.join(
                        tempfile.gettempdir(), f"_preview_{dno}.png"
                    )
                    im.save(tmp_path)
                    tmp_paths.append(tmp_path)

                pdf_path = os.path.join(
                    tempfile.gettempdir(), "preview.pdf"
                )
                with open(pdf_path, "wb") as f:
                    f.write(img2pdf.convert(tmp_paths))

                os.startfile(pdf_path)
            finally:
                for p in tmp_paths:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            return

        if str(mode).lower() == "png" and embed:
            self.reset_left_panel()
            self._show_preview_in_panel(images)
            return

        # (optional) old separate window behaviour could be re-added if you still want it.

    def batch_export_with_ui(self, defect_ids, output_mode="pdf", output_path=None):
        self.batch_cancelled = False

        if not defect_ids:
            messagebox.showwarning("Batch Export", "No defect IDs provided.")
            return

        if not output_path:
            if output_mode == "pdf":
                output_path = filedialog.asksaveasfilename(
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf")],
                )
            else:
                output_path = filedialog.askdirectory()
            if not output_path:
                return

        self.progress_frame_ref = tk.Frame(
            self.input_frame, bg="white", relief="solid", bd=1
        )
        self.progress_frame_ref.pack(side="top", fill="x", pady=12)
        progress_frame = self.progress_frame_ref

        tk.Label(
            progress_frame,
            text="Batch Export Progress",
            bg="white",
            fg="deepskyblue",
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=10)

        status_lbl = tk.Label(
            progress_frame, text="Starting...", bg="white", font=("Segoe UI", 10)
        )
        status_lbl.pack(pady=5)

        def cancel_process():
            self.batch_cancelled = True
            status_lbl.config(text="❌ Cancel requested...")

        cancel_btn = tk.Button(
            progress_frame, text="Cancel", command=cancel_process, bg="red", fg="white"
        )
        cancel_btn.pack(pady=10)

        bar_frame = tk.Frame(progress_frame, bg="white")
        bar_frame.pack(pady=10)

        prog_var = tk.IntVar()
        prog_bar = ttk.Progressbar(
            bar_frame,
            maximum=len(defect_ids),
            variable=prog_var,
            length=120,
            mode="determinate",
            style="Custom.Horizontal.TProgressbar",
        )
        prog_bar.pack()
        self.root.update()

        images = []
        for idx, dno in enumerate(defect_ids, start=1):
            if self.batch_cancelled:
                status_lbl.config(text="❌ Cancelled")
                break

            try:
                self.defect_entry.delete(0, tk.END)
                self.defect_entry.insert(0, str(dno))
                self.on_load_click()
                self.root.update()
                time.sleep(0.4)

                merged = self.capture_sections_image(1, 5)
                if merged is None:
                    continue

                if output_mode == "png":
                    out_file = os.path.join(
                        output_path, f"digsheet_{dno}.png"
                    )
                    merged.save(out_file, "PNG")
                else:
                    temp_path = f"_tmp_{dno}.png"
                    merged.save(temp_path)
                    images.append(temp_path)

                prog_var.set(idx)
                status_lbl.config(
                    text=f"✅ Saved {idx}/{len(defect_ids)}"
                )
                self.root.update()

            except Exception as e:
                print(f"Error on defect {dno}: {e}")

        if not self.batch_cancelled:
            if output_mode == "pdf" and images:
                with open(output_path, "wb") as f:
                    f.write(img2pdf.convert(images))
                for p in images:
                    os.remove(p)
            status_lbl.config(text="✔ Completed")
            messagebox.showinfo(
                "Batch Export Completed",
                f"Your files have been saved successfully.\n\nLocation:\n{output_path}",
            )
        else:
            for p in images:
                if os.path.exists(p):
                    os.remove(p)

        self.root.after(2000, progress_frame.destroy)

    # ======================================================================
    #  PUBLIC API
    # ======================================================================

    def run(self):
        self.root.mainloop()



def dig_run(self):
    pipe_tally = self.pipetally_dir
    project = self.project_root

    try:
        print("pipetally path for digsheet defect based", pipe_tally)
        print("project root for digsheet defect based", project)
    except Exception as e:
        print("error defect based digsheet: ", e)
        traceback.print_exc()

    app = Digsheet(pipe_tally_file=pipe_tally, project_root=project)
    app.run()

# ----------------------------------------------------------------------
#  CLI entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pipe_tally = "D:\Anubhav\softwares\client software\pickle9 - Copy\pipetally_main\Pipe_Tally_12inch_new (1).xlsx"
    project = "D:\Anubhav\softwares\client software\pickle9 - Copy\pipetally_main"
    if len(sys.argv) > 2:
        pipe_tally = sys.argv[1]
        project = sys.argv[2]
    app = Digsheet(pipe_tally_file=pipe_tally, project_root=project)
    app.run()
