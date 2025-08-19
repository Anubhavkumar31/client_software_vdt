# digsheet_abs.py
# Identical styling to your original digsheet; only the data source is different:
# - Loads DataFrame from a pickle path passed by main.py
# - Locates the row by Absolute Distance (tolerant numeric match)
# - Populates the UI with that row (no typing needed)

import datetime
import os
import io
import sys
import re
import math
import time
import pickle
import traceback
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import ImageGrab, Image, ImageTk
import pandas as pd

# -------------------- CLI args: support positional (<pkl> <abs>) or flags --------------------
PKL_PATH = None
ABS_RAW = None

if len(sys.argv) >= 3 and not sys.argv[1].startswith('--'):
    # Positional call from main.py:  python digsheet_abs.py <pipe_tally.pkl> <abs_text>
    PKL_PATH = sys.argv[1]
    ABS_RAW  = sys.argv[2]
else:
    # Optional flags for manual testing
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True, help="Path to pickled pipe_tally DataFrame")
    p.add_argument("--abs_str", required=True, help="Absolute Distance value (string or number)")
    a = p.parse_args()
    PKL_PATH = a.pkl
    ABS_RAW  = a.abs_str

def _abs_to_float(s):
    """Extract first number (handles '123.4 (m)' etc.). Returns float or None."""
    if s is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

ABS_VALUE = _abs_to_float(ABS_RAW)

def load_pipe_tally_from_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Pickle did not contain a pandas DataFrame.")
    return df

# -------------------- Globals used by UI / scrolling / drawing --------------------
scrollable_active = False
df = None  # will hold the DataFrame
icons = {}
feature_labels = {}

# -------------------- Tk window (same styling as original) --------------------
root = tk.Tk()
root.title("Digsheet")

# open maximized (keeps title bar) + lock resizing (same as original)
root.state('zoomed')
root.resizable(False, False)
root.configure(bg="white")

# — Right-side button panel —
button_frame = tk.Frame(root, bg="white")
button_frame.pack(side="right", fill="y", padx=50, pady=100)
input_frame = tk.Frame(button_frame, bg="white")
input_frame.pack(pady=20)

# Load small icons (same paths as your original)
try:
    icon_path = os.getcwd() + "/dig" + "/digsheet_icon/"
    icons["valve"]   = ImageTk.PhotoImage(Image.open(icon_path + "valve.png").resize((18, 18)))
    icons["bend"]    = ImageTk.PhotoImage(Image.open(icon_path + "bend.png").resize((18, 18)))
    icons["flange"]  = ImageTk.PhotoImage(Image.open(icon_path + "flange.png").resize((18, 18)))
    icons["flowtee"] = ImageTk.PhotoImage(Image.open(icon_path + "flowtee.png").resize((18, 18)))
    icons["magnet"]  = ImageTk.PhotoImage(Image.open(icon_path + "magnet.png").resize((18, 18)))
except Exception as e:
    print("Image loading error:", e)
    icons["valve"] = icons["bend"] = icons["flange"] = icons["flowtee"] = icons["magnet"] = None

# --- Scrollable canvas container (same as original) ---
container = tk.Frame(root)
container.pack(fill="both", expand=True)

canvas = tk.Canvas(container, bg="white")
canvas.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

scrollable_frame = tk.Frame(canvas, bg="white")

def _on_mousewheel(event):
    if event.delta:  # Windows / MacOS
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    elif event.num == 4:  # Linux up
        canvas.yview_scroll(-3, "units")
    elif event.num == 5:  # Linux down
        canvas.yview_scroll(3, "units")

def on_frame_configure(event):
    global scrollable_active
    canvas.configure(scrollregion=canvas.bbox("all"))
    canvas_height = canvas.winfo_height()
    frame_height = scrollable_frame.winfo_height()
    if frame_height > canvas_height:
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollable_active = True
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)
    else:
        scrollbar.pack_forget()
        canvas.configure(yscrollcommand=None)
        scrollable_active = False
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

scrollable_frame.bind("<Configure>", on_frame_configure)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# -------------------- Variables / helpers (same look) --------------------
pipe_id_var   = tk.StringVar()
length_var    = tk.StringVar()
wt_var        = tk.StringVar()
latitude_var  = tk.StringVar()
longitude_var = tk.StringVar()

def save_as_image():
    root.update_idletasks()
    x0 = root.winfo_rootx()
    y0 = root.winfo_rooty()
    x1 = x0 + root.winfo_width()
    y1 = y0 + root.winfo_height()
    x2 = feature_desc_frame.winfo_rootx()
    y2 = feature_desc_frame.winfo_rooty()
    x2_end = x2 + feature_desc_frame.winfo_width()
    y2_end = y2 + feature_desc_frame.winfo_height()
    x3 = third_frame.winfo_rootx()
    y3 = third_frame.winfo_rooty()
    x4 = x3 + third_frame.winfo_width()
    y4 = y3 + third_frame.winfo_height()

    if not scrollable_active:
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG Image", "*.png")])
        if not filepath:
            return
        img = ImageGrab.grab(bbox=(x0, y0, x1 - 660, y1 + 220))
        img.save(filepath)
        messagebox.showinfo("Saved!", f"File saved successfully:\n{filepath}")
    else:
        try:
            filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG Image", "*.png")])
            if not filepath:
                return
            x_left = min(x2, x3) - 5
            x_right = max(x2 + feature_desc_frame.winfo_width(), x4 + 10)

            canvas.yview_moveto(0.0)
            root.update(); time.sleep(0.3)
            img_top = ImageGrab.grab(bbox=(x_left, y0, x_right, y2_end + 5))

            canvas.yview_moveto(1.0)
            root.update(); time.sleep(0.6)
            img_bot = ImageGrab.grab(bbox=(x_left, y3 - 95, x_right, y4 - 85))

            if img_top.width != img_bot.width:
                img_bot = img_bot.resize((img_top.width, img_bot.height))

            total_height = img_top.height + img_bot.height
            merged = Image.new("RGB", (img_top.width, total_height), "white")
            merged.paste(img_top, (0, 0))
            merged.paste(img_bot, (0, img_top.height))
            merged.save(filepath)
            messagebox.showinfo("Saved!", f"Image saved successfully:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save merged image:\n{e}")

def hms_to_angle(hms):
    if isinstance(hms, str):
        try:
            parts = [int(p) for p in hms.split(":")]
            while len(parts) < 3:
                parts.append(0)
            h, m, s = parts[:3]
        except:
            h, m, s = 0, 0, 0
    elif isinstance(hms, datetime.time):
        h, m, s = hms.hour, hms.minute, hms.second
    else:
        h, m, s = 0, 0, 0
    angle = (h % 12) * 30 + m * 0.5 + s * (0.5 / 60.0)
    return angle

def draw_pipe(pipe_canvas1, pipe_length, upstream, clock):
    pipe_canvas1.delete("all")
    width, height = 320, 120
    x0, y0 = 40, 30
    x1, y1 = x0 + width, y0 + height
    mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
    radius = height // 2 - 10

    # Pipe outline
    pipe_canvas1.create_oval(x0, y0, x0 + 40, y1, outline="black", width=2)
    pipe_canvas1.create_oval(x1 - 40, y0, x1, y1, outline="black", width=2)
    pipe_canvas1.create_line(x0 + 20, y0, x1 - 20, y0, fill="black", width=2)
    pipe_canvas1.create_line(x0 + 20, y1, x1 - 20, y1, fill="black", width=2)
    pipe_canvas1.create_line(x0, mid_y - 5, x1, mid_y - 5, fill="black", dash=(3, 3))

    pipe_canvas1.create_text(x0 - 20, y0 + 10, text="12", anchor="w", font=("Arial", 10))
    pipe_canvas1.create_text(x0 + 25, mid_y + 5, text="3",  anchor="w", font=("Arial", 10))
    pipe_canvas1.create_text(x0 - 17, y1 - 5,  text="6",  anchor="w", font=("Arial", 10))
    pipe_canvas1.create_text(x0 - 10, mid_y + 5, text="9",  anchor="e", font=("Arial", 10))

    try:
        upstream = float(upstream) if upstream else 0.0
        pipe_length = float(pipe_length) if pipe_length else 0.0
        remaining = round(pipe_length - upstream, 2)
    except:
        upstream = 0.0
        remaining = 0.0

    # Arrows on top
    arrow_y = y0 - 15
    scale_factor = 0.85
    arrow_length_total = (x1 - x0) * scale_factor
    offset = ((x1 - x0) - arrow_length_total) / 2
    arrow_start_x = x0 + offset
    arrow_end_x = x1 - offset

    arrow1_length = (upstream / pipe_length) * arrow_length_total if pipe_length > 0 else arrow_length_total / 2
    arrow2_length = arrow_length_total - arrow1_length

    # Upstream arrow
    arrow1_start = arrow_start_x
    arrow1_end   = arrow1_start + arrow1_length
    pipe_canvas1.create_line(arrow1_start, arrow_y, arrow1_end, arrow_y, arrow=tk.LAST)
    pipe_canvas1.create_line(arrow1_end, arrow_y, arrow1_start, arrow_y, arrow=tk.LAST)
    pipe_canvas1.create_text((arrow1_start + arrow1_end) / 2, arrow_y - 10,
                             text=f"{round(upstream, 2)} m", font=("Arial", 10))
    # Remaining arrow
    arrow2_start = arrow1_end
    arrow2_end   = arrow_end_x
    pipe_canvas1.create_line(arrow2_start, arrow_y, arrow2_end, arrow_y, arrow=tk.LAST)
    pipe_canvas1.create_line(arrow2_end, arrow_y, arrow2_start, arrow_y, arrow=tk.LAST)
    pipe_canvas1.create_text((arrow2_start + arrow2_end) / 2, arrow_y - 10,
                             text=f"{remaining} m", font=("Arial", 10))

    # Defect marker by clock
    angle_deg = hms_to_angle(clock)
    angle_rad = math.radians(angle_deg)
    center_y = mid_y
    defect_x = arrow1_start + (upstream / pipe_length) * arrow_length_total if pipe_length else (arrow1_start + arrow_length_total / 2.0)
    adjusted_radius = radius * 0.80
    defect_y = center_y - int(adjusted_radius * math.cos(angle_rad))

    if 0 <= angle_deg <= 180:
        pipe_canvas1.create_rectangle(defect_x - 4, defect_y - 4, defect_x + 4, defect_y + 4,
                                      fill="orange", outline="black")
    else:
        pipe_canvas1.create_rectangle(defect_x - 4, defect_y - 4, defect_x + 4, defect_y + 4,
                                      outline="orange", width=2)

    pipe_canvas1.create_line(defect_x - 5, defect_y, defect_x - 5, y0, arrow=tk.LAST, fill="black", width=1.5)

# -------------------- Main content (same layout) --------------------
main_frame = tk.Frame(scrollable_frame, bg="white")
main_frame.pack(pady=5, fill="x", padx=10)

# --- Feature on Pipe (Left Box) ---
feature_frame = tk.Frame(main_frame, bg="white", padx=5, pady=5,
                         highlightbackground="black", highlightthickness=1)
feature_frame.pack(side="left", fill="both", expand=True, padx=5)

tk.Label(feature_frame, text="Feature Location on Pipe:", bg="white", fg="deepskyblue",
         font=("Arial", 10, "bold")).pack(pady=(0, 5))
pipe_canvas1 = tk.Canvas(feature_frame, width=360, height=160, bg="white", highlightthickness=0)
pipe_canvas1.pack()

# --- Pipe Description (Right Box) ---
desc_frame = tk.Frame(main_frame, bg="white", padx=5, pady=5,
                      highlightbackground="black", highlightthickness=1)
desc_frame.pack(side="left", fill="both", expand=True, padx=5)

tk.Label(desc_frame, text="Pipe Description:", bg="white", fg="deepskyblue",
         font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=5, pady=(0, 10), sticky="ew")

fields = [
    ("Pipe Number", pipe_id_var),
    ("Pipe Length (m)", length_var),
    ("WT (mm)", wt_var),
    ("Latitude", latitude_var),
    ("Longitude", longitude_var),
]
for i, (label, var) in enumerate(fields):
    tk.Label(desc_frame, text=label + ":", bg="white").grid(row=i + 1, column=0, sticky="e", pady=3)
    tk.Entry(desc_frame, textvariable=var, width=25, bg="white", bd=0, highlightthickness=0,
             relief="flat").grid(row=i + 1, column=1, pady=3)

for col in range(2):
    desc_frame.grid_columnconfigure(col, weight=1)

# --- Feature Description UI Block (center) ---
feature_desc_frame = tk.Frame(scrollable_frame, bg="white", padx=5, pady=2,
                              highlightbackground="black", highlightthickness=1)
feature_desc_frame.pack(fill="both", padx=15)

left_fields  = ["Feature", "Feature type", "Anomaly dimension class", "Surface Location",
                "Remaining wall thickness (mm)", "ERF", "Safe pressure (kg/cm²)"]
right_fields = ["Absolute Distance (m)", "Length (mm)", "Width (mm)", "Max. Depth(%)",
                "Orientation(hr:min)", "Latitude", "Longitude"]

for col in range(5):
    feature_desc_frame.grid_columnconfigure(col, weight=1)
feature_desc_frame.grid_columnconfigure(2, minsize=80)  # spacer

section_title = tk.Label(feature_desc_frame, text="Feature Description:", bg="white",
                         fg="deepskyblue", font=("Arial", 10, "bold"),
                         anchor="center", justify="center")
section_title.grid(row=0, column=0, columnspan=5, pady=(0, 5), sticky="ew")

label_padx = (5, 2)
value_padx = (2, 10)

for i, label_text in enumerate(left_fields):
    tk.Label(feature_desc_frame, text=label_text + ":", bg="white", anchor="w", font=("Arial", 9)).grid(
        row=i+1, column=0, sticky="w", padx=label_padx, pady=2)
    lbl = tk.Label(feature_desc_frame, text="", bg="white", anchor="w", font=("Arial", 9))
    lbl.grid(row=i+1, column=1, sticky="w", padx=value_padx, pady=2)
    feature_labels[label_text] = lbl

for i, label_text in enumerate(right_fields):
    tk.Label(feature_desc_frame, text=label_text + ":", bg="white", anchor="w", font=("Arial", 9)).grid(
        row=i+1, column=3, sticky="w", padx=label_padx, pady=2)
    lbl = tk.Label(feature_desc_frame, text="", bg="white", anchor="w", font=("Arial", 9))
    lbl.grid(row=i+1, column=4, sticky="w", padx=value_padx, pady=2)
    feature_labels[label_text] = lbl

# --- Third Block (bottom) ---
third_frame = tk.Frame(scrollable_frame, bg="white", padx=10, pady=10,
                       highlightbackground="black", highlightthickness=1)
third_frame.pack(fill="both", padx=15, pady=4)

tk.Label(third_frame, text="Pipe Location:", bg="white", fg="deepskyblue",
         font=("Arial", 9, "bold")).grid(row=0, column=0, columnspan=5, sticky="ew")

pipe_canvas = tk.Canvas(third_frame, width=650, height=370, bg="white", highlightthickness=0)
pipe_canvas.grid(row=1, column=0, columnspan=5)
pipe_canvas.update()
canvas_width = pipe_canvas.winfo_width()
canvas_height = pipe_canvas.winfo_height()
for col in range(5):
    third_frame.grid_columnconfigure(col, weight=1)

mid_x = int(canvas_width/2)
mid_y = int(canvas_height/2)

# Upstream weld info label areas etc. (static lines same as original)
pipe_canvas.create_text(mid_x, 5, text="Upstream Weld", font=("Arial", 10))

labels = ["Abs. Dist.:", "Latitude:", "Longitude:"]
for i, label in enumerate(labels):
    pipe_canvas.create_text(mid_x - 320, mid_y - 145 + i * 15, text=label, font=("Arial", 9), anchor="w")
    pipe_canvas.create_text(mid_x - 320, mid_y - 30 + i * 15,  text=label, font=("Arial", 9), anchor="w")

for y in [mid_y - 100, mid_y - 60, mid_y + 20, mid_y + 60]:
    pipe_canvas.create_line(mid_x - 320, y, mid_x + 320, y, width=2)

pipe_canvas.create_text(mid_x - 310, mid_y - 80, text="U/S", font=("Arial", 9, "bold"), fill="blue")
pipe_canvas.create_text(mid_x + 310, mid_y - 80, text="D/S", font=("Arial", 9, "bold"), fill="blue")
pipe_canvas.create_text(mid_x - 310, mid_y + 40, text="L", font=("Arial", 9, "bold"), fill="deepskyblue")
pipe_canvas.create_text(mid_x + 310, mid_y + 40, text="R", font=("Arial", 9, "bold"), fill="deepskyblue")

pipe_info = ["Pipe No:", "Pipe Length(m):", "WT(mm):"]
for i, label in enumerate(pipe_info):
    pipe_canvas.create_text(mid_x - 320, mid_y + 75 + i * 15, text=label, font=("Arial", 9), anchor="w")

pipe_canvas.create_text(mid_x - 315, mid_y + 145, text="FLOW", font=("Arial", 9), fill="deepskyblue", anchor="w")
pipe_canvas.create_line(mid_x - 270, mid_y + 160, mid_x - 320, mid_y + 160, arrow=tk.FIRST, width=1)

for i in range(6):
    x1 = mid_x - 240 + i * 80
    x2 = x1 + 80
    pipe_canvas.create_rectangle(x1, mid_y + 120, x2, mid_y + 180, width=1)

# --- Right side inputs (keep styling same; not used for ABS flow, but we keep them) ---
tk.Label(input_frame, text="Enter Defect S.no:", bg="white").grid(row=1, column=0, sticky="e", pady=5)
defect_entry = tk.Entry(input_frame, width=10)
defect_entry.grid(row=1, column=1, pady=5)
tk.Button(input_frame, text="Load (ABS)", command=lambda: None).grid(row=2, column=0, columnspan=2, pady=5)
tk.Button(input_frame, text="Save as Image", command=save_as_image).grid(row=3, column=0, columnspan=2, pady=5)

# -------------------- ABS column detection + row selection --------------------
ABS_COL_CANDIDATES = [
    "Abs. Distance (m)",
    "Absolute Distance",
    "Absolute_Distance",
]

def pick_abs_column(_df):
    cols = list(_df.columns)
    for name in ABS_COL_CANDIDATES:
        if name in cols:
            return name
    norm = {c.strip().lower().replace(" ", "").replace(".", ""): c for c in cols}
    for key in ["absdistance(m)", "absolutedistance", "absolute_distance"]:
        if key in norm:
            return norm[key]
    return None

def find_row_index_by_abs(_df, target_abs, tol=0.5):
    """
    Return index (label) of the row whose Absolute Distance is closest to target_abs.
    If you want strict equality, set tol=0 and require exact match before fallback.
    """
    col = pick_abs_column(_df)
    if not col:
        raise KeyError("Could not find the Absolute Distance column.")
    s = pd.to_numeric(_df[col], errors="coerce")
    if s.isna().all():
        raise ValueError("Absolute Distance column could not be parsed to numbers.")
    diffs = (s - float(target_abs)).abs()
    idx = diffs.idxmin()
    # Uncomment to enforce strict tolerance:
    # if diffs.loc[idx] > tol:
    #     return None
    return idx

# -------------------- Populate UI from a given row (mirrors original logic) --------------------
def apply_row_to_ui(row, defect_idx):
    """
    Takes a pandas Series 'row' (selected by ABS) and the row index (defect_idx),
    and fills the entire UI: top boxes, feature labels, bottom pipe canvas.
    This mirrors your original 'fetch_data' + 'on_load_click' logic.
    """
    # --- Top right fields ---
    pipe_id_var.set(str(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else "")
    length_var.set(str(row.iloc[4]) if len(row) > 4 and pd.notna(row.iloc[4]) else "")
    wt_var.set(str(row.iloc[11]) if len(row) > 11 and pd.notna(row.iloc[11]) else "")

    # lat/lon by name
    columns_clean = {c.strip().lower().replace(" ", ""): c for c in df.columns}
    lat_c = columns_clean.get("latitude", None)
    lon_c = columns_clean.get("longitude", None)
    latitude_var.set(str(row[lat_c]) if lat_c and pd.notna(row[lat_c]) else "")
    longitude_var.set(str(row[lon_c]) if lon_c and pd.notna(row[lon_c]) else "")

    # Draw pipe (left box)
    upstream = row.iloc[2] if len(row) > 2 and pd.notna(row.iloc[2]) else 0
    clock_raw = row.iloc[8] if len(row) > 8 else "00:00"
    draw_pipe(pipe_canvas1, row.iloc[4] if len(row) > 4 else 0, upstream, clock_raw)

    # --- Fill center feature labels (mapping like your original) ---
    excel_mapping = {
        "Feature": 5,
        "Feature type": 6,
        "Anomaly dimension class": 7,
        "Surface Location": 14,
        "Remaining wall thickness (mm)": None,  # computed below
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
            # formatting like original
            if label in ["Length (mm)", "Width (mm)", "Max. Depth(%)"]:
                try:
                    value = int(float(value)) if pd.notna(value) else ""
                except:
                    value = ""
            elif label == "ERF":
                try:
                    value = f"{float(value):.3f}" if pd.notna(value) else ""
                except:
                    value = ""
            elif label == "Orientation(hr:min)":
                try:
                    if isinstance(value, str) and ":" in value:
                        value = ":".join(value.split(":")[:2])
                    elif isinstance(value, datetime.time):
                        value = value.strftime("%H:%M")
                    else:
                        value = str(value)
                except:
                    value = ""
            feature_labels[label].config(text=str(value) if value is not None else "")
        # latitude/longitude handled separately below

    # Remaining wall thickness
    try:
        wt = float(row.iloc[11]) if len(row) > 11 and pd.notna(row.iloc[11]) else None
        max_depth = float(row.iloc[12]) if len(row) > 12 and pd.notna(row.iloc[12]) else None
        if wt is not None and max_depth is not None:
            remaining_wt = round(wt - (wt * max_depth / 100.0), 1)
        else:
            remaining_wt = ""
    except:
        remaining_wt = ""
    feature_labels["Remaining wall thickness (mm)"].config(text=str(remaining_wt))

    # Latitude/Longitude (center labels)
    lat_val = row[lat_c] if lat_c and pd.notna(row[lat_c]) else ""
    lon_val = row[lon_c] if lon_c and pd.notna(row[lon_c]) else ""
    feature_labels["Latitude"].config(text=str(lat_val))
    feature_labels["Longitude"].config(text=str(lon_val))

    # ---------------- Bottom block: features/bends around the defect ----------------
    pipe_canvas.delete("upstream_text")
    pipe_canvas.delete("flange_text")
    pipe_canvas.delete("us_arrow")
    pipe_canvas.delete("ds_arrow")
    pipe_canvas.delete("bend_text")
    pipe_canvas.delete("pipe_icon")

    # distances
    abs_val = float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else None
    up_val  = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else None
    upstream_weld = round(abs(abs_val - up_val), 2) if (abs_val is not None and up_val is not None) else 0.0
    pipe_canvas.create_text(mid_x, 20, text=f"{upstream_weld:.2f}(m)", font=("Arial", 10), tags="upstream_text")

    feature_keywords = ["flange", "valve", "flow tee", "magnet"]

    # Build up/down lists around defect_idx
    features_upstream = []
    features_downstream = []
    bends_upstream = []
    bends_downstream = []

    # Upstream features
    for i in range(defect_idx - 1, -1, -1):
        r = df.loc[i]
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

    # Downstream features
    for i in range(defect_idx + 1, len(df)):
        r = df.loc[i]
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

    # Upstream bends
    for i in range(defect_idx - 1, -1, -1):
        r = df.loc[i]
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

    # Downstream bends
    for i in range(defect_idx + 1, len(df)):
        r = df.loc[i]
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

    # FEATURES (2 upstream + 2 downstream)
    feature_slots = [
        {"x": mid_x - 190, "arrow_x": mid_x - 200, "text_x": mid_x - 160, "source": features_upstream[::-1], "index": 1},
        {"x": mid_x - 90,  "arrow_x": mid_x - 100, "text_x": mid_x - 60,  "source": features_upstream[::-1], "index": 0},
        {"x": mid_x + 110, "arrow_x": mid_x + 120, "text_x": mid_x + 80,  "source": features_downstream,      "index": 0},
        {"x": mid_x + 210, "arrow_x": mid_x + 220, "text_x": mid_x + 180, "source": features_downstream,      "index": 1},
    ]
    for slot in feature_slots:
        src = slot["source"]; idx = slot["index"]
        try:
            feat = src[idx]
        except:
            continue
        name = feat.get("name", "")
        dist_val = feat.get("distance", "")
        lat = feat.get("lat", "")
        lon = feat.get("long", "")

        pipe_canvas.create_text(slot["x"], mid_y - 160, text=name, font=("Arial", 10), tags="flange_text")
        pipe_canvas.create_text(slot["x"], mid_y - 145, text=f"{dist_val}(m)" if dist_val != "" else "", font=("Arial", 9), tags="flange_text")
        pipe_canvas.create_text(slot["x"], mid_y - 130, text=lat, font=("Arial", 9), tags="flange_text")
        pipe_canvas.create_text(slot["x"], mid_y - 115, text=lon, font=("Arial", 9), tags="flange_text")

        try:
            arrow_val = round(abs(float(upstream_weld) - float(dist_val)), 2)
        except:
            arrow_val = ""
        pipe_canvas.create_line(slot["arrow_x"], mid_y - 95, slot["arrow_x"], mid_y - 65,
                                arrow=tk.FIRST, fill="deepskyblue", width=2, tags="us_arrow")
        pipe_canvas.create_text(slot["text_x"], mid_y - 80, text=f"{arrow_val}(m)" if arrow_val != "" else "",
                                font=("Arial", 9), tags="us_arrow")

    # BENDS (3 upstream + 3 downstream)
    bend_slots = [
        {"source": bends_upstream[::-1], "index": 2, "x_name": mid_x - 230, "x_dist": mid_x - 230, "x_lat": mid_x - 235, "x_lon": mid_x - 235, "tri_x": mid_x - 200, "arrow_text_x": mid_x - 215},
        {"source": bends_upstream[::-1], "index": 1, "x_name": mid_x - 140, "x_dist": mid_x - 140, "x_lat": mid_x - 135, "x_lon": mid_x - 135, "tri_x": mid_x - 110, "arrow_text_x": mid_x - 125},
        {"source": bends_upstream[::-1], "index": 0, "x_name": mid_x - 50,  "x_dist": mid_x - 50,  "x_lat": mid_x - 35,  "x_lon": mid_x - 35,  "tri_x": mid_x - 20,  "arrow_text_x": mid_x - 35},
        {"source": bends_downstream,     "index": 0, "x_name": mid_x + 55,  "x_dist": mid_x + 55,  "x_lat": mid_x + 50,  "x_lon": mid_x + 50,  "tri_x": mid_x + 110, "arrow_text_x": mid_x + 30},
        {"source": bends_downstream,     "index": 1, "x_name": mid_x + 155, "x_dist": mid_x + 155, "x_lat": mid_x + 150, "x_lon": mid_x + 150, "tri_x": mid_x + 210, "arrow_text_x": mid_x + 130},
        {"source": bends_downstream,     "index": 2, "x_name": mid_x + 255, "x_dist": mid_x + 255, "x_lat": mid_x + 250, "x_lon": mid_x + 250, "tri_x": mid_x + 310, "arrow_text_x": mid_x + 230},
    ]
    def draw_triangle(x, y):
        pipe_canvas.create_polygon(
            x - 42.5, y - 20,
            x - 50,   y + 18,
            x - 35,   y + 18,
            fill="deepskyblue", outline="gray", width=1, tags="us_arrow"
        )
    for slot in bend_slots:
        src = slot["source"]; idx = slot["index"]
        try:
            bend = src[idx]
        except:
            continue
        name = bend.get("name", "")
        dist_val = bend.get("distance", "")
        lat = bend.get("lat", "")
        lon = bend.get("long", "")

        pipe_canvas.create_text(slot["x_name"], mid_y - 45, text=name, font=("Arial", 10), tags="bend_text")
        pipe_canvas.create_text(slot["x_dist"], mid_y - 30, text=f"{dist_val}(m)" if dist_val != "" else "", font=("Arial", 9), tags="bend_text")
        pipe_canvas.create_text(slot["x_lat"],  mid_y - 15, text=lat, font=("Arial", 9), tags="bend_text")
        pipe_canvas.create_text(slot["x_lon"],  mid_y,      text=lon, font=("Arial", 9), tags="bend_text")

        draw_triangle(slot["tri_x"], mid_y + 40)
        try:
            arrow_val = round(abs(float(upstream_weld) - float(dist_val)), 2)
        except:
            arrow_val = ""
        pipe_canvas.create_text(slot["arrow_text_x"], mid_y + 35, text=f"{arrow_val}", font=("Arial", 9), tags="us_arrow")
        pipe_canvas.create_text(slot["arrow_text_x"], mid_y + 45, text="(m)", font=("Arial", 9), tags="us_arrow")

    # --- 6 pipe boxes info (3 before, current, 2 after) ---
    try:
        pipe_num_defect = int(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else None
    except:
        pipe_num_defect = None

    target_pipe_numbers = []
    if pipe_num_defect is not None:
        target_pipe_numbers = [pipe_num_defect + i for i in range(-3, 3)]
    else:
        # fallback, just leave empty boxes
        target_pipe_numbers = [None]*6

    pipe_data_list = []
    for pno in target_pipe_numbers:
        if pno is None:
            pipe_data_list.append(("", "", ""))
            continue
        match = df[df.iloc[:, 3] == pno]
        if not match.empty:
            r = match.iloc[0]
            pnum = r.iloc[3] if pd.notna(r.iloc[3]) else ""
            plen = f"{round(float(r.iloc[4]), 3)}" if pd.notna(r.iloc[4]) else ""
            pwt  = f"{round(float(r.iloc[11]), 1)}" if pd.notna(r.iloc[11]) else ""
            pipe_data_list.append((str(pnum), plen, pwt))
        else:
            pipe_data_list.append(("", "", ""))

    pipe_positions = [-210, -140, -60, 20, 110, 180]
    for i, (pnum, plen, pwt) in enumerate(pipe_data_list):
        px = pipe_positions[i]
        pipe_canvas.create_text(mid_x + px, mid_y + 75,  text=pnum, font=("Arial", 9), anchor="w", tags="us_arrow")
        pipe_canvas.create_text(mid_x + px, mid_y + 90,  text=plen, font=("Arial", 9), anchor="w", tags="us_arrow")
        pipe_canvas.create_text(mid_x + px, mid_y + 105, text=pwt,  font=("Arial", 9), anchor="w", tags="us_arrow")

    # Defect marker in box 4
    try:
        upstream_dist = float(row.iloc[2]) if pd.notna(row.iloc[2]) else None
        clock_pos     = row.iloc[8] if len(row) > 8 else "00:00"
        pipe_len      = float(row.iloc[4]) if pd.notna(row.iloc[4]) else None

        if pipe_len is not None and upstream_dist is not None:
            clock_angle = hms_to_angle(clock_pos)

            box_x_start = mid_x
            box_x_end   = mid_x + 80
            box_y_top    = mid_y + 120
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
                pipe_canvas.create_rectangle(defect_x - 3, defect_y - 3, defect_x + 3, defect_y + 3,
                                             fill="orange", outline="black", tags="us_arrow")
            else:
                pipe_canvas.create_rectangle(defect_x - 3, defect_y - 3, defect_x + 3, defect_y + 3,
                                             outline="orange", width=2, tags="us_arrow")
    except Exception as e:
        print("Bottom pipe defect box drawing error:", e)
        traceback.print_exc()

    # Place icons for features inside 6 pipe boxes
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
        matching_rows = df[df.iloc[:, 3] == pipe_num]
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
            img = icons.get(feat)
            if img is not None:
                pipe_canvas.create_image(cx, offset_y, image=img, tags="pipe_icon")

# -------------------- Bootstrap by ABS on startup --------------------
def initialize_by_abs():
    global df
    try:
        df = load_pipe_tally_from_pickle(PKL_PATH)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load pickled DataFrame:\n{e}")
        return

    if ABS_VALUE is None:
        messagebox.showwarning("No Absolute Distance",
                               "No Absolute Distance was provided to digsheet_abs.py")
        return

    try:
        defect_idx = find_row_index_by_abs(df, ABS_VALUE, tol=0.5)
        if defect_idx is None:
            messagebox.showwarning("Not found",
                                   f"No row matched Absolute Distance {ABS_RAW}.")
            return
        row = df.loc[defect_idx]
        # (Optional) Show the resolved ABS in the center label to make it obvious
        feature_labels["Absolute Distance (m)"].config(
            text=str(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else str(ABS_VALUE)
        )
        apply_row_to_ui(row, defect_idx)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to populate digsheet by ABS:\n{e}")

# Run once the widgets are ready
root.after(0, initialize_by_abs)

root.mainloop()
