
import datetime
import os
import sys
import re
import math
import pickle
import traceback
import io
import tempfile
import time

import pandas as pd
import img2pdf
from PIL import Image, ImageTk, ImageGrab

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

import win32api
import win32print


class DigsheetABS:
    # ---- Section IDs and names (stable API) ----
    SECTION_MAP = {
        1: "Client Description",  # NEW
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

    ABS_COL_CANDIDATES = [
        "Abs. Distance (m)",
        "Absolute Distance",
        "Absolute_Distance",
    ]

    def __init__(self, pkl_path, abs_raw, project_root=None, parent=None):
        self.parent = parent
        # self.config = getattr(self.parent, "config", None)
        # self.project_name = getattr(self.parent, "project_name", None)
        # self.runid = getattr(self.parent, "runid", None)
        # print(f"prject and runid name in DigSHEET: {self.project_name} : {self.runid}")
        #
        # print("PARENT:", self.parent)
        # print("PARENT.PARENT:", getattr(self.parent, "parent", None))
        # print("CONFIG:", self.config)
        self.pkl_path = pkl_path
        self.abs_raw = abs_raw
        self.project_root = project_root

        self.ABS_VALUE = self._abs_to_float(abs_raw)

        # state vars
        self.scrollable_active = False
        self.df = None
        self.icons = {}
        self.feature_labels = {}

        # 1️⃣ Create ROOT FIRST
        self.root = tk.Tk()
        self.root.title("Digsheet")
        self.root.state('zoomed')
        self.root.resizable(False, False)
        self.root.configure(bg="white")

        # 2️⃣ Now safe to create Tk variables (StringVar)
        self.pipe_id_var = tk.StringVar(master=self.root)
        self.length_var = tk.StringVar(master=self.root)
        self.wt_var = tk.StringVar(master=self.root)
        self.latitude_var = tk.StringVar(master=self.root)
        self.longitude_var = tk.StringVar(master=self.root)

        self.client_var = tk.StringVar(master=self.root)
        self.pipeline_name_var = tk.StringVar(master=self.root)
        self.pipeline_section_var = tk.StringVar(master=self.root)

        # container refs
        self.button_frame = None
        self.input_frame = None
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

        # 3️⃣ Build UI AFTER variables exist
        self.build_ui()

        # 4️⃣ Load constants
        # if self.project_root:
        #     client_val, pipe_val, section_val = self.load_constants(self.project_root)
        #     self.client_var.set(client_val)
        #     self.pipeline_name_var.set(pipe_val)
        #     self.pipeline_section_var.set(section_val)
        # else:
        #     print("[DEBUG] No PROJECT_ROOT passed")

        client_val, pipe_val, section_val = self.load_constants_from_file()
        self.client_var.set(client_val)
        self.pipeline_name_var.set(pipe_val)
        self.pipeline_section_var.set(section_val)
        # 5️⃣ Initialize with ABS once all widgets exist
        self.root.after(0, self.initialize_by_abs)

    def load_constants_from_file(self):
        from pathlib import Path
        import pandas as pd

        if not self.project_root:
            print("⚠️ No project_root provided")
            return None, None, None

        constants_dir = Path(self.project_root) / "constants"

        if not constants_dir.exists():
            print(f"⚠️ constants folder not found: {constants_dir}")
            return None, None, None

        # Try Excel first, then CSV
        file_path = None
        for name in ["constants.xlsx", "constants.xls", "constants.csv"]:
            p = constants_dir / name
            if p.exists():
                file_path = p
                break

        if not file_path:
            print("⚠️ No constants file found")
            return None, None, None

        try:
            # Read file
            if file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            if df.empty:
                print("⚠️ constants file is empty")
                return None, None, None

            # Take first row
            row = df.iloc[0]

            client_val = row.get("CLIENT_NAME_DESCRIPTION")
            pipe_val = row.get("PIPELINE_NAME_DESCRIPTION")
            section_val = row.get("PIPELINE_SECTION_DESCRIPTION")

            return client_val, pipe_val, section_val

        except Exception as e:
            import traceback
            print("❌ Failed to read constants file")
            print(traceback.format_exc())
            return None, None, None

    # --------------------------------------
    # Public entry point
    # --------------------------------------
    def run(self):
        self.root.mainloop()

    # --------------------------------------
    # Helpers
    # --------------------------------------
    @staticmethod
    def _abs_to_float(s):
        """Extract first number (handles '123.4 (m)' etc.). Returns float or None."""
        if s is None:
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(s))
        return float(m.group(0)) if m else None

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r'[^A-Za-z0-9]+', '', str(s)).upper()

    def load_pipe_tally_from_pickle(self, pkl_path):
        print(f"pickle path : {pkl_path}")
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Pickle did not contain a pandas DataFrame.")
        return df

    # --------------------------------------
    # UI construction
    # --------------------------------------
    def build_ui(self):
        # root
        # self.root = tk.Tk()
        # self.root.title("Digsheet")
        # self.root.state('zoomed')
        # self.root.resizable(False, False)
        # self.root.configure(bg="white")

        # right button panel
        screen_w = self.root.winfo_screenwidth()
        button_panel_w = (screen_w / 2) - 150
        self.button_frame = tk.Frame(self.root, bg="white", width=button_panel_w)
        self.button_frame.pack(side="right", fill="y", padx=50, pady=0, anchor="n")
        self.button_frame.pack_propagate(False)

        self.input_frame = tk.Frame(self.button_frame, bg="white")
        self.input_frame.pack(side="top", fill="both", expand=True, pady=(8, 0))

        # load icons
        self.load_icons()

        # scrollable canvas container
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas, bg="white")

        # configure scrolling
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # build all content sections
        self.build_client_description_section()
        self.build_main_section()
        self.build_comment_section()
        self.build_feature_description_section()
        self.build_pipe_location_section()

        # buttons (Save image / PDF / Print)
        self.build_action_buttons()

    def load_icons(self):
        try:
            icon_path = os.getcwd() + "/dig" + "/digsheet_icon/"
            self.icons["valve"] = ImageTk.PhotoImage(Image.open(icon_path + "valve.png").resize((18, 18)))
            self.icons["bend"] = ImageTk.PhotoImage(Image.open(icon_path + "bend.png").resize((18, 18)))
            self.icons["flange"] = ImageTk.PhotoImage(Image.open(icon_path + "flange.png").resize((18, 18)))
            self.icons["flowtee"] = ImageTk.PhotoImage(Image.open(icon_path + "flowtee.png").resize((18, 18)))
            self.icons["magnet"] = ImageTk.PhotoImage(Image.open(icon_path + "magnet.png").resize((18, 18)))
        except Exception as e:
            print("Image loading error:", e)
            self.icons["valve"] = self.icons["bend"] = self.icons["flange"] = \
                self.icons["flowtee"] = self.icons["magnet"] = None

    # scrolling handlers
    def _on_mousewheel(self, event):
        if event.delta:  # Windows / MacOS
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:  # Linux up
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:  # Linux down
            self.canvas.yview_scroll(3, "units")

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        canvas_height = self.canvas.winfo_height()
        frame_height = self.scrollable_frame.winfo_height()
        if frame_height > canvas_height:
            self.scrollbar.pack(side="right", fill="y")
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            self.scrollable_active = True
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self.canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        else:
            self.scrollbar.pack_forget()
            self.canvas.configure(yscrollcommand=None)
            self.scrollable_active = False
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

    # --------------------------------------
    # Sections
    # --------------------------------------
    def build_client_description_section(self):
        self.client_desc_frame = tk.Frame(
            self.scrollable_frame, bg="white", padx=5, pady=2,
            highlightbackground="black", highlightthickness=1
        )
        self.client_desc_frame.pack(fill="both", padx=(15, 15), pady=(5, 0))

        tk.Label(
            self.client_desc_frame, text="Client Description:", bg="white",
            fg="deepskyblue", font=("Arial", 10, "bold")
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
                left_frame, text=f"{txt}:", bg="white", anchor="w",
                font=("Arial", 9)
            ).grid(row=r, column=0, sticky="w", padx=(10, 6), pady=(2, 2))

            tk.Entry(
                left_frame, textvariable=var, width=40, bg="white",
                bd=0, highlightthickness=0, relief="flat"
            ).grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=(2, 2))

        left_frame.grid_columnconfigure(0, weight=0)
        left_frame.grid_columnconfigure(1, weight=1)

        # logo
        try:
            icon_path_vdt = os.path.join(
                os.getcwd(),
                # "Components",
                "dig",
                "digsheet_icon",
                "vdt-logo.png"
            )
            logo_img = Image.open(
                icon_path_vdt
            ).resize((100, 100))
            logo_tk = ImageTk.PhotoImage(logo_img)
            logo_lbl = tk.Label(self.client_desc_frame, image=logo_tk, bg="white")
            logo_lbl.place(relx=1.0, rely=0.5, anchor="e", x=-10)
            self.client_desc_frame.logo_ref = logo_tk  # prevent GC
        except Exception as e:
            print("Logo load failed:", e)

    def build_main_section(self):
        self.main_frame = tk.Frame(self.scrollable_frame, bg="white")
        self.main_frame.pack(pady=5, fill="x", padx=10)

        # Feature on Pipe
        feature_frame = tk.Frame(
            self.main_frame, bg="white", padx=5, pady=5,
            highlightbackground="black", highlightthickness=1
        )
        feature_frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(
            feature_frame, text="Feature Location on Pipe:", bg="white",
            fg="deepskyblue", font=("Arial", 10, "bold")
        ).pack(pady=(0, 5))

        self.pipe_canvas1 = tk.Canvas(
            feature_frame, width=360, height=160,
            bg="white", highlightthickness=0
        )
        self.pipe_canvas1.pack()

        # Pipe description
        desc_frame = tk.Frame(
            self.main_frame, bg="white", padx=5, pady=5,
            highlightbackground="black", highlightthickness=1
        )
        desc_frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(
            desc_frame, text="Pipe Description:", bg="white", fg="deepskyblue",
            font=("Arial", 10, "bold")
        ).grid(row=0, column=0, columnspan=5, padx=5, pady=(0, 5), sticky="ew")

        fields = [
            ("Pipe Number", self.pipe_id_var),
            ("Pipe Length (m)", self.length_var),
            ("WT (mm)", self.wt_var),
            ("Latitude", self.latitude_var),
            ("Longitude", self.longitude_var),
        ]

        for i, (label, var) in enumerate(fields, start=1):
            tk.Label(
                desc_frame, text=label + ":", bg="white",
                anchor="w", font=("Arial", 9)
            ).grid(row=i, column=0, sticky="w", padx=(5, 2), pady=(2, 2))

            tk.Label(
                desc_frame, textvariable=var, bg="white",
                anchor="w", font=("Arial", 9)
            ).grid(row=i, column=1, sticky="w", padx=(2, 10), pady=(2, 2))

        for col in range(2):
            desc_frame.grid_columnconfigure(col, weight=1)

    def build_comment_section(self):
        self.comment_frame = tk.Frame(
            self.scrollable_frame, bg="white", padx=5, pady=2,
            highlightbackground="black", highlightthickness=1
        )
        self.comment_frame.pack(fill="both", padx=(15, 15), pady=(5, 5))

        tk.Label(
            self.comment_frame, text="Comment:", bg="white",
            fg="deepskyblue", font=("Arial", 10, "bold")
        ).pack(side="top", pady=(0, 5))

        comment_placeholder = tk.Label(
            self.comment_frame, text="", bg="white", anchor="w",
            justify="left", font=("Arial", 9)
        )
        comment_placeholder.pack(fill="both", expand=True, padx=10, pady=20)

    def build_feature_description_section(self):
        self.feature_desc_frame = tk.Frame(
            self.scrollable_frame, bg="white", padx=5, pady=2,
            highlightbackground="black", highlightthickness=1
        )
        self.feature_desc_frame.pack(fill="both", padx=15)

        self.feature_labels = {}

        left_fields = [
            "Feature", "Feature type", "Anomaly dimension class",
            "Surface Location", "Remaining wall thickness (mm)",
            "ERF", "Safe pressure (kg/cm²)"
        ]
        right_fields = [
            "Absolute Distance (m)", "Length (mm)", "Width (mm)",
            "Max. Depth(%)", "Orientation(hr:min)",
            "Latitude", "Longitude"
        ]

        for col in range(5):
            self.feature_desc_frame.grid_columnconfigure(col, weight=1)
        self.feature_desc_frame.grid_columnconfigure(2, minsize=80)

        section_title = tk.Label(
            self.feature_desc_frame, text="Feature Description:",
            bg="white", fg="deepskyblue",
            font=("Arial", 10, "bold"), anchor="center", justify="center"
        )
        section_title.grid(row=0, column=0, columnspan=5, pady=(0, 5), sticky="ew")

        label_padx = (5, 2)
        value_padx = (2, 10)

        for i, label_text in enumerate(left_fields):
            tk.Label(
                self.feature_desc_frame,
                text=label_text + ":", bg="white", anchor="w",
                font=("Arial", 9)
            ).grid(row=i+1, column=0, sticky="w", padx=label_padx, pady=2)
            label = tk.Label(
                self.feature_desc_frame, text="", bg="white",
                anchor="w", font=("Arial", 9)
            )
            label.grid(row=i+1, column=1, sticky="w", padx=value_padx, pady=2)
            self.feature_labels[label_text] = label

        for i, label_text in enumerate(right_fields):
            tk.Label(
                self.feature_desc_frame,
                text=label_text + ":", bg="white", anchor="w",
                font=("Arial", 9)
            ).grid(row=i+1, column=3, sticky="w", padx=label_padx, pady=2)
            label = tk.Label(
                self.feature_desc_frame, text="", bg="white",
                anchor="w", font=("Arial", 9)
            )
            label.grid(row=i+1, column=4, sticky="w", padx=value_padx, pady=2)
            self.feature_labels[label_text] = label

    def build_pipe_location_section(self):
        self.third_frame = tk.Frame(
            self.scrollable_frame, bg="white", padx=10, pady=10,
            highlightbackground="black", highlightthickness=1
        )
        self.third_frame.pack(fill="both", padx=15, pady=4)

        tk.Label(
            self.third_frame, text="Pipe Location:", bg="white",
            fg="deepskyblue", font=("Arial", 9, "bold")
        ).grid(row=0, column=0, columnspan=5, sticky="ew")

        self.pipe_canvas = tk.Canvas(
            self.third_frame, width=650, height=370,
            bg="white", highlightthickness=0
        )
        self.pipe_canvas.grid(row=1, column=0, columnspan=5)
        self.pipe_canvas.update()

        for col in range(5):
            self.third_frame.grid_columnconfigure(col, weight=1)

        canvas_width = self.pipe_canvas.winfo_width()
        canvas_height = self.pipe_canvas.winfo_height()
        mid_x = int(canvas_width / 2)
        mid_y = int(canvas_height / 2)

        self.pipe_canvas.create_line(mid_x, 30, mid_x, mid_y + 150, arrow=tk.FIRST)
        self.pipe_canvas.create_text(mid_x, 5, text="Upstream Weld", font=("Arial", 10))

        labels = ["Abs. Dist.:", "Latitude:", "Longitude:"]
        for i, label in enumerate(labels):
            self.pipe_canvas.create_text(
                mid_x - 320, mid_y - 145 + i * 15,
                text=label, font=("Arial", 9), anchor="w"
            )
            self.pipe_canvas.create_text(
                mid_x - 320, mid_y - 30 + i * 15,
                text=label, font=("Arial", 9), anchor="w"
            )

        for y in [mid_y - 100, mid_y - 60, mid_y + 20, mid_y + 60]:
            self.pipe_canvas.create_line(mid_x - 320, y, mid_x + 320, y, width=2)

        self.pipe_canvas.create_text(mid_x - 310, mid_y - 80, text="U/S",
                                     font=("Arial", 9, "bold"), fill="blue")
        self.pipe_canvas.create_text(mid_x + 310, mid_y - 80, text="D/S",
                                     font=("Arial", 9, "bold"), fill="blue")
        self.pipe_canvas.create_text(mid_x - 310, mid_y + 40, text="L",
                                     font=("Arial", 9, "bold"), fill="deepskyblue")
        self.pipe_canvas.create_text(mid_x + 310, mid_y + 40, text="R",
                                     font=("Arial", 9, "bold"), fill="deepskyblue")

        pipe_info = ["Pipe No:", "Pipe Length(m):", "WT(mm):"]
        for i, label in enumerate(pipe_info):
            self.pipe_canvas.create_text(
                mid_x - 320, mid_y + 75 + i * 15,
                text=label, font=("Arial", 9), anchor="w"
            )

        self.pipe_canvas.create_text(
            mid_x - 315, mid_y + 145, text="FLOW",
            font=("Arial", 9), fill="deepskyblue", anchor="w"
        )
        self.pipe_canvas.create_line(
            mid_x - 270, mid_y + 160,
            mid_x - 320, mid_y + 160,
            arrow=tk.FIRST, width=1
        )

        for i in range(6):
            x1 = mid_x - 240 + i * 80
            x2 = x1 + 80
            self.pipe_canvas.create_rectangle(
                x1, mid_y + 120, x2, mid_y + 180, width=1
            )

    def build_action_buttons(self):
        tk.Button(
            self.input_frame,
            text="Save as image",
            command=lambda: self.capture_sections(1, 5)
        ).grid(row=5, column=0, columnspan=2, pady=5)

        tk.Button(
            self.input_frame,
            text="Save as PDF",
            command=self.save_all_sections_as_pdf
        ).grid(row=6, column=0, columnspan=2, pady=5)

        tk.Button(
            self.input_frame,
            text="Print",
            command=self.print_all_sections_dialog
        ).grid(row=7, column=0, columnspan=2, pady=5)

    # --------------------------------------
    # Constants loader
    # --------------------------------------
    def load_constants(self, project_root):
        if not project_root:
            print("no project root")
            return "", "", ""
        csv_path = os.path.join(project_root, "constants.csv")
        xlsx_path = os.path.join(project_root, "constants.xlsx")

        constants_file = None
        if os.path.exists(csv_path):
            constants_file = csv_path
        elif os.path.exists(xlsx_path):
            constants_file = xlsx_path

        print(f"constant files : {constants_file}")

        if not constants_file:
            return "", "", ""

        if constants_file.endswith(".xlsx"):
            const_df = pd.read_excel(constants_file, dtype=str)
        else:
            const_df = pd.read_csv(constants_file, dtype=str)

        colmap = {self._normalize(c): c for c in const_df.columns}

        def first_val(*keys):
            for k in keys:
                norm = self._normalize(k)
                if norm in colmap:
                    ser = const_df[colmap[norm]].dropna().astype(str).str.strip()
                    if not ser.empty:
                        return ser.iloc[0]
            return ""

        client = first_val("CLIENT_NAME_DESCRIPTION", "CLIENT")
        pipe = first_val("PIPELINE_NAME_DESCRIPTION", "PIPELINE")
        section = first_val("PIPELINE_SECTION_DESCRIPTION", "SECTION")
        return client, pipe, section

    def load_constants_from_db(self):
        with self.config.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    IFNULL(NULLIF(Pipeline_owner, ''), 'NA'),
                    IFNULL(NULLIF(Pipeline_name, ''), 'NA'),
                    IFNULL(NULLIF(Pipeline_section, ''), 'NA')
                FROM projectdetail
                WHERE ProjectName=%s AND runid=%s
                """,
                (self.project_name, self.runid)
            )

            client_name, pipeline_name, pipeline_section = cursor.fetchone()

        return client_name, pipeline_name, pipeline_section

    # --------------------------------------
    # Capture / print / PDF
    # --------------------------------------
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
            initialfile="all_sections.png"
        )
        if not filepath:
            return

        images = []
        for section_id in range(section_start, section_end + 1):
            if section_id not in self.SECTION_MAP:
                print(f"⚠️ Section ID {section_id} not in SECTION_MAP, skipping.")
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
                print(f"⚠️ Section {name} frame not found, skipping.")
                continue

            x0, y0, x1, y1 = coords[name]
            dx0, dy0, dx1, dy1 = self.SECTION_THRESHOLDS.get(name, (0, 0, 0, 0))
            bbox = (x0 + dx0, y0 + dy0, x1 + dx1, y1 + dy1)
            print(f"📸 Capturing {name} @ {bbox}")
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
        messagebox.showinfo("Saved!", f"All sections saved successfully:\n{filepath}")
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
            dx0, dy0, dx1, dy1 = self.SECTION_THRESHOLDS.get(name, (0, 0, 0, 0))
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
            initialfile="all_sections.pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not pdf_path:
            return

        merged, dpi = self.upscale_image(merged, target_dpi=300, base_dpi=96)
        buf = io.BytesIO()
        merged.save(buf, format="PNG", dpi=(dpi, dpi))
        buf.seek(0)

        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(buf.getvalue()))

        messagebox.showinfo("Saved!", f"High-quality PDF created:\n{pdf_path}")

    def print_all_sections_dialog(self):
        merged = self.capture_sections_image(1, 5)
        if merged is None:
            messagebox.showerror("Error", "No sections captured")
            return

        temp_img = tempfile.mktemp(suffix=".png")
        merged.save(temp_img, "PNG")

        def get_printers():
            printers = [
                p[2] for p in win32print.EnumPrinters(
                    win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
                )
            ]
            return printers

        def send_to_printer(printer_name, file_path):
            try:
                win32api.ShellExecute(
                    0, "print", file_path, f'"{printer_name}"', ".", 0
                )
                messagebox.showinfo("Print", f"Sent to printer: {printer_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to print:\n{e}")

        def print_selected():
            selection = printer_combo.get()
            if not selection:
                messagebox.showwarning("Warning", "Please select a printer")
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
            fg="black"
        ).pack(pady=(15, 10))

        printers = get_printers()
        printer_combo = ttk.Combobox(dialog, values=printers, state="readonly", width=40)
        if printers:
            printer_combo.current(0)
        printer_combo.pack(pady=10)

        button_frame = tk.Frame(dialog, bg="white")
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Print", command=print_selected).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=10)

        dialog.mainloop()

    # --------------------------------------
    # Geometry helpers
    # --------------------------------------
    def save_individual_section(self, section_id):
        if section_id not in self.SECTION_MAP:
            messagebox.showerror("Error", "Invalid section ID!")
            return

        name = self.SECTION_MAP[section_id]
        all_coords = self.get_section_coords()
        if name not in all_coords:
            messagebox.showerror("Error", f"Section '{name}' not found.")
            return

        x0, y0, x1, y1 = all_coords[name]
        dx0, dy0, dx1, dy1 = self.SECTION_THRESHOLDS.get(name, (0, 0, 0, 0))

        ax0 = x0 + dx0
        ay0 = y0 + dy0
        ax1 = x1 + dx1
        ay1 = y1 + dy1

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        ax0 = max(0, min(ax0, screen_w - 1))
        ay0 = max(0, min(ay0, screen_h - 1))
        ax1 = max(1, min(ax1, screen_w))
        ay1 = max(1, ay1)

        if ax1 <= ax0:
            ax1 = min(screen_w, ax0 + 1)
        if ay1 <= ay0:
            ay1 = min(screen_h, ay0 + 1)

        bbox = (ax0, ay0, ax1, ay1)
        print(f"[{name}] bbox={bbox}")

        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")],
                initialfile=f"{name.replace(' ', '_')}.png"
            )
            if not filepath:
                return

            img = ImageGrab.grab(bbox=bbox)
            img.save(filepath)
            messagebox.showinfo("Saved!", f"{name} saved successfully:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save {name}:\n{e}")

    # --------------------------------------
    # Geometry + drawing
    # --------------------------------------
    @staticmethod
    def hms_to_angle(hms):
        if isinstance(hms, str):
            try:
                parts = [int(p) for p in hms.split(":")]
                while len(parts) < 3:
                    parts.append(0)
                h, m, s = parts[:3]
            except Exception:
                h, m, s = 0, 0, 0
        elif isinstance(hms, datetime.time):
            h, m, s = hms.hour, hms.minute, hms.second
        else:
            h, m, s = 0, 0, 0
        angle = (h % 12) * 30 + m * 0.5 + s * (0.5 / 60.0)
        return angle

    def draw_pipe(self, pipe_canvas1, pipe_length, upstream, clock):
        pipe_canvas1.delete("all")
        width, height = 320, 120
        x0, y0 = 40, 30
        x1, y1 = x0 + width, y0 + height
        mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
        radius = height // 2 - 10

        pipe_canvas1.create_oval(x0, y0, x0 + 40, y1, outline="black", width=2)
        pipe_canvas1.create_oval(x1 - 40, y0, x1, y1, outline="black", width=2)
        pipe_canvas1.create_line(x0 + 20, y0, x1 - 20, y0, fill="black", width=2)
        pipe_canvas1.create_line(x0 + 20, y1, x1 - 20, y1, fill="black", width=2)
        pipe_canvas1.create_line(x0, mid_y - 5, x1, mid_y - 5, fill="black", dash=(3, 3))

        pipe_canvas1.create_text(x0 - 20, y0 + 10, text="12", anchor="w", font=("Arial", 10))
        pipe_canvas1.create_text(x0 + 25, mid_y + 5, text="3", anchor="w", font=("Arial", 10))
        pipe_canvas1.create_text(x0 - 17, y1 - 5, text="6", anchor="w", font=("Arial", 10))
        pipe_canvas1.create_text(x0 - 10, mid_y + 5, text="9", anchor="e", font=("Arial", 10))

        try:
            upstream_val = float(upstream) if upstream else 0.0
            pipe_length_val = float(pipe_length) if pipe_length else 0.0
            remaining = round(pipe_length_val - upstream_val, 2)
        except Exception:
            upstream_val = 0.0
            remaining = 0.0
            pipe_length_val = 0.0

        arrow_y = y0 - 15
        scale_factor = 0.85
        arrow_length_total = (x1 - x0) * scale_factor
        offset = ((x1 - x0) - arrow_length_total) / 2
        arrow_start_x = x0 + offset
        arrow_end_x = x1 - offset

        arrow1_length = (upstream_val / pipe_length_val) * arrow_length_total if pipe_length_val > 0 else arrow_length_total / 2
        arrow2_length = arrow_length_total - arrow1_length

        arrow1_start = arrow_start_x
        arrow1_end = arrow1_start + arrow1_length
        pipe_canvas1.create_line(arrow1_start, arrow_y, arrow1_end, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_line(arrow1_end, arrow_y, arrow1_start, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_text(
            (arrow1_start + arrow1_end) / 2, arrow_y - 10,
            text=f"{round(upstream_val, 2)} m", font=("Arial", 10)
        )

        arrow2_start = arrow1_end
        arrow2_end = arrow_end_x
        pipe_canvas1.create_line(arrow2_start, arrow_y, arrow2_end, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_line(arrow2_end, arrow_y, arrow2_start, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_text(
            (arrow2_start + arrow2_end) / 2, arrow_y - 10,
            text=f"{remaining} m", font=("Arial", 10)
        )

        angle_deg = self.hms_to_angle(clock)
        angle_rad = math.radians(angle_deg)
        center_y = mid_y
        defect_x = arrow1_start + (upstream_val / pipe_length_val) * arrow_length_total if pipe_length_val else (arrow1_start + arrow_length_total / 2.0)
        adjusted_radius = radius * 0.80
        defect_y = center_y - int(adjusted_radius * math.cos(angle_rad))

        if 0 <= angle_deg <= 180:
            pipe_canvas1.create_rectangle(
                defect_x - 4, defect_y - 4, defect_x + 4, defect_y + 4,
                fill="orange", outline="black"
            )
        else:
            pipe_canvas1.create_rectangle(
                defect_x - 4, defect_y - 4, defect_x + 4, defect_y + 4,
                outline="orange", width=2
            )

        pipe_canvas1.create_line(
            defect_x - 5, defect_y, defect_x - 5, y0,
            arrow=tk.LAST, fill="black", width=1.5
        )

    # --------------------------------------
    # ABS column / row selection
    # --------------------------------------
    def pick_abs_column(self, _df):
        cols = list(_df.columns)
        for name in self.ABS_COL_CANDIDATES:
            if name in cols:
                return name
        norm = {c.strip().lower().replace(" ", "").replace(".", ""): c for c in cols}
        for key in ["absdistance(m)", "absolutedistance", "absolute_distance"]:
            if key in norm:
                return norm[key]
        return None

    def find_row_index_by_abs(self, _df, target_abs, tol=0.5):
        col = self.pick_abs_column(_df)
        if not col:
            raise KeyError("Could not find the Absolute Distance column.")
        s = pd.to_numeric(_df[col], errors="coerce")
        if s.isna().all():
            raise ValueError("Absolute Distance column could not be parsed to numbers.")
        diffs = (s - float(target_abs)).abs()
        idx = diffs.idxmin()
        # optional tolerance logic can be added here if needed
        return idx

    # --------------------------------------
    # Populate UI from row
    # --------------------------------------
    def apply_row_to_ui(self, row, defect_idx):
        self.pipe_id_var.set(str(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else "")
        self.length_var.set(str(row.iloc[4]) if len(row) > 4 and pd.notna(row.iloc[4]) else "")
        self.wt_var.set(str(row.iloc[11]) if len(row) > 11 and pd.notna(row.iloc[11]) else "")

        columns_clean = {c.strip().lower().replace(" ", ""): c for c in self.df.columns}
        lat_c = columns_clean.get("latitude", None)
        lon_c = columns_clean.get("longitude", None)
        self.latitude_var.set(str(row[lat_c]) if lat_c and pd.notna(row[lat_c]) else "")
        self.longitude_var.set(str(row[lon_c]) if lon_c and pd.notna(row[lon_c]) else "")

        upstream = row.iloc[2] if len(row) > 2 and pd.notna(row.iloc[2]) else 0
        clock_raw = row.iloc[8] if len(row) > 8 else "00:00"
        self.draw_pipe(
            self.pipe_canvas1,
            row.iloc[4] if len(row) > 4 else 0,
            upstream,
            clock_raw
        )

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
            "Longitude": None
        }

        for label, col_index in excel_mapping.items():
            if col_index is not None and col_index < len(row):
                value = row.iloc[col_index]
                if label in ["Length (mm)", "Width (mm)", "Max. Depth(%)"]:
                    try:
                        value = int(float(value)) if pd.notna(value) else ""
                    except Exception:
                        value = ""
                elif label == "ERF":
                    try:
                        value = f"{float(value):.3f}" if pd.notna(value) else ""
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
                self.feature_labels[label].config(text=str(value) if value is not None else "")

        try:
            wt = float(row.iloc[11]) if len(row) > 11 and pd.notna(row.iloc[11]) else None
            max_depth = float(row.iloc[12]) if len(row) > 12 and pd.notna(row.iloc[12]) else None
            if wt is not None and max_depth is not None:
                remaining_wt = round(wt - (wt * max_depth / 100.0), 1)
            else:
                remaining_wt = ""
        except Exception:
            remaining_wt = ""
        self.feature_labels["Remaining wall thickness (mm)"].config(text=str(remaining_wt))

        lat_val = row[lat_c] if lat_c and pd.notna(row[lat_c]) else ""
        lon_val = row[lon_c] if lon_c and pd.notna(row[lon_c]) else ""
        self.feature_labels["Latitude"].config(text=str(lat_val))
        self.feature_labels["Longitude"].config(text=str(lon_val))

        # bottom block drawing
        self.redraw_bottom_block(row, defect_idx)

    def redraw_bottom_block(self, row, defect_idx):
        self.pipe_canvas.delete("upstream_text")
        self.pipe_canvas.delete("flange_text")
        self.pipe_canvas.delete("us_arrow")
        self.pipe_canvas.delete("ds_arrow")
        self.pipe_canvas.delete("bend_text")
        self.pipe_canvas.delete("pipe_icon")

        columns_clean = {c.strip().lower().replace(" ", ""): c for c in self.df.columns}
        lat_c = columns_clean.get("latitude", None)
        lon_c = columns_clean.get("longitude", None)

        try:
            abs_val = float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else None
            up_val = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else None
        except Exception:
            abs_val = None
            up_val = None

        upstream_weld = round(abs(abs_val - up_val), 2) if (abs_val is not None and up_val is not None) else 0.0

        canvas_width = self.pipe_canvas.winfo_width()
        canvas_height = self.pipe_canvas.winfo_height()
        mid_x = int(canvas_width / 2)
        mid_y = int(canvas_height / 2)

        self.pipe_canvas.create_text(mid_x, 20, text=f"{upstream_weld:.2f}(m)",
                                     font=("Arial", 10), tags="upstream_text")

        feature_keywords = ["flange", "valve", "flow tee", "magnet"]

        features_upstream = []
        features_downstream = []
        bends_upstream = []
        bends_downstream = []

        for i in range(defect_idx - 1, -1, -1):
            r = self.df.loc[i]
            fname = str(r.iloc[5]).strip().lower() if len(r) > 5 else ""
            if any(f in fname for f in feature_keywords):
                features_upstream.append({
                    "name": str(r.iloc[5]),
                    "distance": round(float(r.iloc[1]), 2) if pd.notna(r.iloc[1]) else "",
                    "lat": str(r[lat_c]) if lat_c and pd.notna(r[lat_c]) else "",
                    "long": str(r[lon_c]) if lon_c and pd.notna(r[lon_c]) else ""
                })
                if len(features_upstream) == 2:
                    break

        for i in range(defect_idx + 1, len(self.df)):
            r = self.df.loc[i]
            fname = str(r.iloc[5]).strip().lower() if len(r) > 5 else ""
            if any(f in fname for f in feature_keywords):
                features_downstream.append({
                    "name": str(r.iloc[5]),
                    "distance": round(float(r.iloc[1]), 2) if pd.notna(r.iloc[1]) else "",
                    "lat": str(r[lat_c]) if lat_c and pd.notna(r[lat_c]) else "",
                    "long": str(r[lon_c]) if lon_c and pd.notna(r[lon_c]) else ""
                })
                if len(features_downstream) == 2:
                    break

        for i in range(defect_idx - 1, -1, -1):
            r = self.df.loc[i]
            fname = str(r.iloc[5]).strip().lower() if len(r) > 5 else ""
            if "bend" in fname:
                bends_upstream.append({
                    "name": str(r.iloc[5]),
                    "distance": round(float(r.iloc[1]), 2) if pd.notna(r.iloc[1]) else "",
                    "lat": str(r[lat_c]) if lat_c and pd.notna(r[lat_c]) else "",
                    "long": str(r[lon_c]) if lon_c and pd.notna(r[lon_c]) else ""
                })
                if len(bends_upstream) == 3:
                    break

        for i in range(defect_idx + 1, len(self.df)):
            r = self.df.loc[i]
            fname = str(r.iloc[5]).strip().lower() if len(r) > 5 else ""
            if "bend" in fname:
                bends_downstream.append({
                    "name": str(r.iloc[5]),
                    "distance": round(float(r.iloc[1]), 2) if pd.notna(r.iloc[1]) else "",
                    "lat": str(r[lat_c]) if lat_c and pd.notna(r[lat_c]) else "",
                    "long": str(r[lon_c]) if lon_c and pd.notna(r[lon_c]) else ""
                })
                if len(bends_downstream) == 3:
                    break

        feature_slots = [
            {"x": mid_x - 190, "arrow_x": mid_x - 200, "text_x": mid_x - 160, "source": features_upstream[::-1], "index": 1},
            {"x": mid_x - 90,  "arrow_x": mid_x - 100, "text_x": mid_x - 60,  "source": features_upstream[::-1], "index": 0},
            {"x": mid_x + 110, "arrow_x": mid_x + 120, "text_x": mid_x + 80,  "source": features_downstream,      "index": 0},
            {"x": mid_x + 210, "arrow_x": mid_x + 220, "text_x": mid_x + 180, "source": features_downstream,      "index": 1},
        ]
        for slot in feature_slots:
            src = slot["source"]
            idx = slot["index"]
            try:
                feat = src[idx]
            except Exception:
                continue
            name_f = feat.get("name", "")
            dist_val = feat.get("distance", "")
            lat = feat.get("lat", "")
            lon = feat.get("long", "")

            self.pipe_canvas.create_text(
                slot["x"], mid_y - 160, text=name_f,
                font=("Arial", 10), tags="flange_text"
            )
            self.pipe_canvas.create_text(
                slot["x"], mid_y - 145,
                text=f"{dist_val}(m)" if dist_val != "" else "",
                font=("Arial", 9), tags="flange_text"
            )
            self.pipe_canvas.create_text(
                slot["x"], mid_y - 130, text=lat,
                font=("Arial", 9), tags="flange_text"
            )
            self.pipe_canvas.create_text(
                slot["x"], mid_y - 115, text=lon,
                font=("Arial", 9), tags="flange_text"
            )

            try:
                arrow_val = round(abs(float(upstream_weld) - float(dist_val)), 2)
            except Exception:
                arrow_val = ""
            self.pipe_canvas.create_line(
                slot["arrow_x"], mid_y - 95,
                slot["arrow_x"], mid_y - 65,
                arrow=tk.FIRST, fill="deepskyblue", width=2, tags="us_arrow"
            )
            self.pipe_canvas.create_text(
                slot["text_x"], mid_y - 80,
                text=f"{arrow_val}(m)" if arrow_val != "" else "",
                font=("Arial", 9), tags="us_arrow"
            )

        bend_slots = [
            {"source": bends_upstream[::-1], "index": 2, "x_name": mid_x - 230, "x_dist": mid_x - 230, "x_lat": mid_x - 235, "x_lon": mid_x - 235, "tri_x": mid_x - 200, "arrow_text_x": mid_x - 215},
            {"source": bends_upstream[::-1], "index": 1, "x_name": mid_x - 140, "x_dist": mid_x - 140, "x_lat": mid_x - 135, "x_lon": mid_x - 135, "tri_x": mid_x - 110, "arrow_text_x": mid_x - 125},
            {"source": bends_upstream[::-1], "index": 0, "x_name": mid_x - 50,  "x_dist": mid_x - 50,  "x_lat": mid_x - 35,  "x_lon": mid_x - 35,  "tri_x": mid_x - 20,  "arrow_text_x": mid_x - 35},
            {"source": bends_downstream,     "index": 0, "x_name": mid_x + 55,  "x_dist": mid_x + 55,  "x_lat": mid_x + 50,  "x_lon": mid_x + 50,  "tri_x": mid_x + 110, "arrow_text_x": mid_x + 30},
            {"source": bends_downstream,     "index": 1, "x_name": mid_x + 155, "x_dist": mid_x + 155, "x_lat": mid_x + 150, "x_lon": mid_x + 150, "tri_x": mid_x + 210, "arrow_text_x": mid_x + 130},
            {"source": bends_downstream,     "index": 2, "x_name": mid_x + 255, "x_dist": mid_x + 255, "x_lat": mid_x + 250, "x_lon": mid_x + 250, "tri_x": mid_x + 310, "arrow_text_x": mid_x + 230},
        ]

        def draw_triangle(x, y):
            self.pipe_canvas.create_polygon(
                x - 42.5, y - 20,
                x - 50,   y + 18,
                x - 35,   y + 18,
                fill="deepskyblue", outline="gray", width=1, tags="us_arrow"
            )

        for slot in bend_slots:
            src = slot["source"]
            idx = slot["index"]
            try:
                bend = src[idx]
            except Exception:
                continue
            name_b = bend.get("name", "")
            dist_val = bend.get("distance", "")
            lat = bend.get("lat", "")
            lon = bend.get("long", "")

            self.pipe_canvas.create_text(
                slot["x_name"], mid_y - 45, text=name_b,
                font=("Arial", 10), tags="bend_text"
            )
            self.pipe_canvas.create_text(
                slot["x_dist"], mid_y - 30,
                text=f"{dist_val}(m)" if dist_val != "" else "",
                font=("Arial", 9), tags="bend_text"
            )
            self.pipe_canvas.create_text(
                slot["x_lat"], mid_y - 15, text=lat,
                font=("Arial", 9), tags="bend_text"
            )
            self.pipe_canvas.create_text(
                slot["x_lon"], mid_y, text=lon,
                font=("Arial", 9), tags="bend_text"
            )

            draw_triangle(slot["tri_x"], mid_y + 40)
            try:
                arrow_val = round(abs(float(upstream_weld) - float(dist_val)), 2)
            except Exception:
                arrow_val = ""
            self.pipe_canvas.create_text(
                slot["arrow_text_x"], mid_y + 35,
                text=f"{arrow_val}", font=("Arial", 9),
                tags="us_arrow"
            )
            self.pipe_canvas.create_text(
                slot["arrow_text_x"], mid_y + 45,
                text="(m)", font=("Arial", 9),
                tags="us_arrow"
            )

        try:
            pipe_num_defect = int(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else None
        except Exception:
            pipe_num_defect = None

        if pipe_num_defect is not None:
            target_pipe_numbers = [pipe_num_defect + i for i in range(-3, 3)]
        else:
            target_pipe_numbers = [None] * 6

        pipe_data_list = []
        for pno in target_pipe_numbers:
            if pno is None:
                pipe_data_list.append(("", "", ""))
                continue
            match = self.df[self.df.iloc[:, 3] == pno]
            if not match.empty:
                r = match.iloc[0]
                pnum = r.iloc[3] if pd.notna(r.iloc[3]) else ""
                plen = f"{round(float(r.iloc[4]), 3)}" if pd.notna(r.iloc[4]) else ""
                pwt = f"{round(float(r.iloc[11]), 1)}" if pd.notna(r.iloc[11]) else ""
                pipe_data_list.append((str(pnum), plen, pwt))
            else:
                pipe_data_list.append(("", "", ""))

        pipe_positions = [-210, -140, -60, 20, 110, 180]
        for i, (pnum, plen, pwt) in enumerate(pipe_data_list):
            px = pipe_positions[i]
            self.pipe_canvas.create_text(
                mid_x + px, mid_y + 75, text=pnum,
                font=("Arial", 9), anchor="w", tags="us_arrow"
            )
            self.pipe_canvas.create_text(
                mid_x + px, mid_y + 90, text=plen,
                font=("Arial", 9), anchor="w", tags="us_arrow"
            )
            self.pipe_canvas.create_text(
                mid_x + px, mid_y + 105, text=pwt,
                font=("Arial", 9), anchor="w", tags="us_arrow"
            )

        try:
            upstream_dist = float(row.iloc[2]) if pd.notna(row.iloc[2]) else None
            clock_pos = row.iloc[8] if len(row) > 8 else "00:00"
            pipe_len = float(row.iloc[4]) if pd.notna(row.iloc[4]) else None

            if pipe_len is not None and upstream_dist is not None:
                clock_angle = self.hms_to_angle(clock_pos)

                box_x_start = mid_x
                box_x_end = mid_x + 80
                box_y_top = mid_y + 120
                box_y_bottom = mid_y + 190

                if upstream_dist < pipe_len / 3:
                    defect_x = box_x_start + 15
                elif upstream_dist < 2 * pipe_len / 3:
                    defect_x = (box_x_start + box_x_end) / 2
                else:
                    defect_x = box_x_end - 15

                if 0 <= clock_angle <= 60 or 300 < clock_angle <= 360:
                    defect_y = box_y_top + 10
                elif 60 < clock_angle <= 120 or 240 <= clock_angle <= 300:
                    defect_y = (box_y_top + box_y_bottom) / 2
                else:
                    defect_y = box_y_bottom - 10

                if 0 <= clock_angle <= 180:
                    self.pipe_canvas.create_rectangle(
                        defect_x - 3, defect_y - 3, defect_x + 3, defect_y + 3,
                        fill="orange", outline="black", tags="us_arrow"
                    )
                else:
                    self.pipe_canvas.create_rectangle(
                        defect_x - 3, defect_y - 3, defect_x + 3, defect_y + 3,
                        outline="orange", width=2, tags="us_arrow"
                    )
        except Exception as e:
            print("Bottom pipe defect box drawing error:", e)
            traceback.print_exc()

        pipe_box_centers = [
            (mid_x - 200, mid_y + 155),
            (mid_x - 120, mid_y + 155),
            (mid_x - 40,  mid_y + 155),
            (mid_x + 40,  mid_y + 155),
            (mid_x + 120, mid_y + 155),
            (mid_x + 200, mid_y + 155),
        ]
        for i, pipe_num in enumerate(target_pipe_numbers):
            if pipe_num is None:
                continue
            matching_rows = self.df[self.df.iloc[:, 3] == pipe_num]
            if matching_rows.empty:
                continue

            found_features = []
            for _, rr in matching_rows.iterrows():
                ftxt = str(rr.iloc[5]).lower() if len(rr) > 5 else ""
                if "valve" in ftxt and "valve" not in found_features:
                    found_features.append("valve")
                if "flow" in ftxt or "tee" in ftxt:
                    if "flowtee" not in found_features:
                        found_features.append("flowtee")
                if "flange" in ftxt and "flange" not in found_features:
                    found_features.append("flange")
                if "bend" in ftxt and "bend" not in found_features:
                    found_features.append("bend")
                if "magnet" in ftxt and "magnet" not in found_features:
                    found_features.append("magnet")

            cx, cy = pipe_box_centers[i]
            spacing = 22
            for j, feat in enumerate(found_features):
                offset_y = cy - ((len(found_features) - 1) * spacing // 2) + (j * spacing)
                img = self.icons.get(feat)
                if img is not None:
                    self.pipe_canvas.create_image(cx, offset_y, image=img, tags="pipe_icon")

    # --------------------------------------
    # Bootstrap
    # --------------------------------------
    def initialize_by_abs(self):
        try:
            self.df = self.load_pipe_tally_from_pickle(self.pkl_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load pickled DataFrame:\n{e}")
            return

        if self.ABS_VALUE is None:
            messagebox.showwarning(
                "No Absolute Distance",
                "No Absolute Distance was provided to digsheet_abs.py"
            )
            return

        try:
            defect_idx = self.find_row_index_by_abs(self.df, self.ABS_VALUE, tol=0.5)
            if defect_idx is None:
                messagebox.showwarning(
                    "Not found",
                    f"No row matched Absolute Distance {self.abs_raw}."
                )
                return
            row = self.df.loc[defect_idx]
            self.feature_labels["Absolute Distance (m)"].config(
                text=str(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else str(self.ABS_VALUE)
            )
            self.apply_row_to_ui(row, defect_idx)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to populate digsheet by ABS:\n{e}"
            )


# --------------------------------------
# CLI bootstrap (same semantics as original)
# --------------------------------------
def parse_cli_args():
    PKL_PATH = None
    ABS_RAW = None
    PROJECT_ROOT = None

    if len(sys.argv) >= 3 and not sys.argv[1].startswith("--"):
        # positional args: <pkl> <abs> [project_root]
        PKL_PATH = sys.argv[1]
        ABS_RAW = sys.argv[2]
        PROJECT_ROOT = sys.argv[3] if len(sys.argv) >= 4 else None
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--pkl", required=True)
        parser.add_argument("--abs_str", required=True)
        parser.add_argument("--project_root")
        args = parser.parse_args()
        PKL_PATH, ABS_RAW, PROJECT_ROOT = args.pkl, args.abs_str, args.project_root

    return PKL_PATH, ABS_RAW, PROJECT_ROOT


def run_digsheet(self, pkl, abs_str, project_root):
    app = DigsheetABS(pkl, abs_str, project_root, parent=self)
    app.run()


if __name__ == "__main__":
    pkl, abs_str, project_root = parse_cli_args()
    app = DigsheetABS(pkl, abs_str, project_root)
    app.run()

