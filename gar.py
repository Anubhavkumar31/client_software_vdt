import datetime
import os, io
import tkinter as tk
import traceback
from tkinter import messagebox, filedialog
import pandas as pd
import math
from PIL import ImageGrab, Image, ImageTk
import time
import pickle
import sys
# from main import pipe_tally
batch_cancelled = False

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




scrollable_active = False

root = tk.Tk()
from tkinter import ttk

style = ttk.Style()
style.theme_use("default")   # make sure we’re on a style we can override

style.configure(
    "Custom.Horizontal.TProgressbar",
    troughcolor="white",
    background="deepskyblue",
    thickness=25,   # make height ~ same as width (square)
    bordercolor="white",
    lightcolor="deepskyblue",
    darkcolor="deepskyblue"
)



root.title("Digsheet")

# open maximized (keeps the title‐bar with minimize/close)
root.state('zoomed')
# but lock down resizing so they can’t shrink it
root.resizable(False, False)

root.configure(bg="white")



def on_load_click():
    global df
    try:
        # Check if the argument (pipe_tally file) is passed
        # if len(sys.argv) > 1:
            pipe_tally_file = r"F:\work_new\client_software\test_data_cs\pickle7\pipetally_main\Pipe_Tally_8inch.xlsx"
            # project_root = sys.argv[2]
            # csv_path  = os.path.join(project_root, "constants.csv")
            # xlsx_path = os.path.join(project_root, "constants.xlsx")
            constants_file = r"F:\work_new\client_software\test_data_cs\pickle7\constants.xlsx" 
            print(f"constants_file path: {constants_file}")
            pipe_tally = load_pipe_tally(pipe_tally_file)  # Deserialize the pipe_tally
            df = pipe_tally
            # --- Read constants and populate Client / Pipeline fields ---
            # --- Read constants and populate Client / Pipeline fields ---
            const_df = pd.read_excel(constants_file, dtype=str)

            import re
            def _norm(s: str) -> str:
                # collapse any mix of spaces/underscores/newlines into ONE underscore
                s = re.sub(r'[^A-Za-z0-9]+', ' ', str(s))     # turn junk into spaces
                return '_'.join(s.strip().upper().split())    # collapse to single underscores

            colmap = {_norm(c): c for c in const_df.columns}

            def _first_val(*aliases):
                for a in aliases:
                    key = _norm(a)
                    if key in colmap:
                        ser = const_df[colmap[key]].dropna().astype(str).str.strip()
                        if not ser.empty:
                            return ser.iloc[0]
                return ""

            # (Optional) see what we matched
            print("[constants] columns:", list(const_df.columns))
            print("[constants] picked:",
                "CLIENT->", colmap.get("CLIENT_NAME_DESCRIPTION"),
                "PIPELINE_NAME->", colmap.get("PIPELINE_NAME_DESCRIPTION"),
                "PIPELINE_SECTION->", colmap.get("PIPELINE_SECTION_DESCRIPTION"))

            client_var.set(_first_val("CLIENT_NAME_DESCRIPTION"))
            pipeline_name_var.set(_first_val("PIPELINE_NAME_DESCRIPTION"))
            pipeline_section_var.set(_first_val("PIPELINE_SECTION_DESCRIPTION"))



        # else:
            # print("No pipe_tally file provide or maybe proect_root")
    except Exception as e:
        print(f"Error in on_load_click: {e}")
    if 'df' not in globals() or df is None:
        messagebox.showwarning("Missing Excel File", "Please load an Excel file before loading defect data.")
        return
    fetch_data()
    pipe_canvas.delete("upstream_text")
    pipe_canvas.delete("flange_text")
    pipe_canvas.delete("us_arrow")
    pipe_canvas.delete("ds_arrow")
    pipe_canvas.delete("bend_text")
    pipe_canvas.delete("pipe_icon")

    weld_info = get_dynamic_weld_and_feature_data()
    if not weld_info:
        return

    upstream_weld_dist = weld_info["upstream_weld"]
    features_upstream = weld_info["features_upstream"]
    features_downstream = weld_info["features_downstream"]
    bends_upstream = weld_info.get("bends_upstream", [])
    bends_downstream = weld_info.get("bends_downstream", [])

    pipe_canvas.create_text(mid_x, 20, text=f"{upstream_weld_dist:.2f}(m)", font=("Arial", 10), tags="upstream_text")

    # FEATURE Display Logic (2 Up + 2 Down)
    feature_slots = [
        {"x": mid_x - 190, "arrow_x": mid_x - 200, "text_x": mid_x - 160, "source": features_upstream[::-1], "index": 1},
        {"x": mid_x - 90,  "arrow_x": mid_x - 100,  "text_x": mid_x - 60,  "source": features_upstream[::-1], "index": 0},
        {"x": mid_x + 110, "arrow_x": mid_x + 120, "text_x": mid_x + 80, "source": features_downstream,      "index": 0},
        {"x": mid_x + 210, "arrow_x": mid_x + 220, "text_x": mid_x + 180, "source": features_downstream,     "index": 1},
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

            pipe_canvas.create_text(x, mid_y - 160, text=name, font=("Arial", 10), tags="flange_text")
            pipe_canvas.create_text(x, mid_y - 145, text=dist, font=("Arial", 9), tags="flange_text")
            pipe_canvas.create_text(x, mid_y - 130, text=lat, font=("Arial", 9), tags="flange_text")
            pipe_canvas.create_text(x, mid_y - 115, text=lon, font=("Arial", 9), tags="flange_text")

            arrow_val = round(abs(float(upstream_weld_dist) - float(dist_val)), 2)
            pipe_canvas.create_line(arrow_x, mid_y - 95, arrow_x, mid_y - 65, arrow=tk.FIRST, fill="deepskyblue", width=2, tags="us_arrow")
            pipe_canvas.create_text(text_x, mid_y - 80, text=f"{arrow_val}(m)", font=("Arial", 9), tags="us_arrow")
        except:
            continue  # Skip if out of bounds or missing

    # BEND Display Logic (3 Up + 3 Down)
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

            pipe_canvas.create_text(slot["x_name"], mid_y - 45, text=name, font=("Arial", 10), tags="bend_text")
            pipe_canvas.create_text(slot["x_dist"], mid_y - 30, text=dist, font=("Arial", 9), tags="bend_text")
            pipe_canvas.create_text(slot["x_lat"], mid_y - 15, text=lat, font=("Arial", 9), tags="bend_text")
            pipe_canvas.create_text(slot["x_lon"], mid_y,        text=lon, font=("Arial", 9), tags="bend_text")

            draw_triangle(slot["tri_x"], mid_y + 40)
            arrow_val = round(abs(float(upstream_weld_dist) - float(dist_val)), 2)
            pipe_canvas.create_text(slot["arrow_text_x"], mid_y + 35, text=f"{arrow_val}", font=("Arial", 9), tags="us_arrow")
            pipe_canvas.create_text(slot["arrow_text_x"], mid_y + 45, text="(m)", font=("Arial", 9), tags="us_arrow")
        except:
            continue

    try:
        s_no = int(defect_entry.get())
        defect_row = df[df.iloc[:, 0] == s_no]
        if defect_row.empty:
            messagebox.showwarning("Warning", f"No defect found for S.No {s_no}")
            return
        pipe_num_defect = int(defect_row.iloc[0, 3])
    except:
        messagebox.showerror("Error", "Invalid or missing defect S.No.")
        return

    # Define target pipe numbers: 3 before to 2 after current
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

    # X positions for each pipe block (centered alignment maintained)
    pipe_positions = [-210, -140, -60, 20, 110, 180]
    for i, (pnum, plen, pwt) in enumerate(pipe_data_list):
        px = pipe_positions[i]
        pipe_canvas.create_text(mid_x + px, mid_y + 75, text=pnum, font=("Arial", 9), anchor="w", tags="us_arrow")
        pipe_canvas.create_text(mid_x + px, mid_y + 90, text=plen, font=("Arial", 9), anchor="w", tags="us_arrow")
        pipe_canvas.create_text(mid_x + px, mid_y + 105, text=pwt, font=("Arial", 9), anchor="w", tags="us_arrow")

    # Draw defect marker in 4th pipe (box 4)
    try:
        defect_row = defect_row.iloc[0]
        upstream_dist = f"{round(float(defect_row.iloc[2]), 2)}" if pd.notna(defect_row.iloc[2]) else ""
        clock_pos = f"{(defect_row.iloc[8])}" if pd.notna(defect_row.iloc[8]) else ""
        pipe_len = f"{round((defect_row.iloc[4]), 3)}" if pd.notna(defect_row.iloc[4]) else ""

        if pipe_len and upstream_dist:
            pipe_length = float(pipe_len)
            upstream = float(upstream_dist)
            clock_angle = hms_to_angle(clock_pos)

            # Box 4 boundaries
            box_x_start = mid_x
            box_x_end = mid_x + 80
            box_y_top = mid_y + 120
            box_y_bottom = mid_y + 190

            # Horizontal position
            if upstream < pipe_length / 3:
                defect_x = box_x_start + 15
            elif upstream < 2 * pipe_length / 3:
                defect_x = (box_x_start + box_x_end) / 2
            else:
                defect_x = box_x_end - 15

            # Vertical position
            if 0 <= clock_angle <= 60 or 300 < clock_angle <= 360:
                defect_y = box_y_top + 10
            elif 60 < clock_angle <= 120 or 240 <= clock_angle <= 300:
                defect_y = (box_y_top + box_y_bottom) / 2
            else:
                defect_y = box_y_bottom - 10

            # Draw defect box with fill logic
            if 0 <= clock_angle <= 180:
                pipe_canvas.create_rectangle(defect_x - 3, defect_y - 3, defect_x + 3, defect_y + 3,
                                            fill="orange", outline="black", tags="us_arrow")
            else:
                pipe_canvas.create_rectangle(defect_x - 3, defect_y - 3, defect_x + 3, defect_y + 3,
                                            outline="orange", width=2, tags="us_arrow")
    except Exception as e:
        print("Bottom pipe defect box drawing error:", e)
        traceback.print_exc()

    # Center positions of the 6 pipe boxes (from left to right)
    pipe_box_centers = [
        (mid_x - 200, mid_y + 155),  # pipe1
        (mid_x - 120, mid_y + 155),  # pipe2
        (mid_x - 40, mid_y + 155),   # pipe3
        (mid_x + 40, mid_y + 155),   # pipe4
        (mid_x + 120, mid_y + 155),  # pipe5
        (mid_x + 200, mid_y + 155)   # pipe6
    ]

    # Loop over each of the 6 pipe boxes
    for i, pipe_num in enumerate(target_pipe_numbers):
        matching_rows = df[df.iloc[:, 3] == pipe_num]

        if not matching_rows.empty:
            found_features = []  # store matched feature types for this pipe

            # Check all matching rows for features in column 5
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

            # Now place icons with vertical spacing
            cx, cy = pipe_box_centers[i]
            spacing = 22  # vertical spacing between icons

            for j, feat in enumerate(found_features):
                offset_y = cy - ((len(found_features) - 1) * spacing // 2) + (j * spacing)

                if feat == "valve":
                    pipe_canvas.create_image(cx, offset_y, image=valve_img, tags="pipe_icon")
                elif feat == "flowtee":
                    pipe_canvas.create_image(cx, offset_y, image=flowtee_img, tags="pipe_icon")
                elif feat == "flange":
                    pipe_canvas.create_image(cx, offset_y, image=flange_img, tags="pipe_icon")
                elif feat == "bend":
                    pipe_canvas.create_image(cx, offset_y, image=bend_img, tags="pipe_icon")
                elif feat == "magnet":
                    pipe_canvas.create_image(cx, offset_y, image=magnet_img, tags="pipe_icon")

# — Right-side button panel (fixed width) —
screen_w = root.winfo_screenwidth()
BUTTON_PANEL_W = (screen_w/2) - 150  # ← adjust this number to make the panel wider/narrower
button_frame = tk.Frame(root, bg="white", width=BUTTON_PANEL_W)
button_frame.pack(side="right", fill="y", padx=50, pady=0, anchor="n")

# keep the frame from shrinking to fit its children
button_frame.pack_propagate(False)

# inner container for your controls
input_frame = tk.Frame(button_frame, bg="white")
input_frame.pack(side="top", fill="both", expand=True, pady=(8,0))



# ===== TOP TOOLBAR (one row, three groups) =====
toolbar = tk.Frame(input_frame, bg="white")
toolbar.pack(side="top", fill="x", pady=(0, 8))

# -- Group 1 (bordered): Enter + Load + Save + Print
group1 = tk.LabelFrame(
    toolbar,
    text="",              # no title, just a hairline border
    bg="white",
    fg="gray40",
    relief="groove",
    bd=1,
    padx=6, pady=4
)
group1.pack(side="left", padx=(0, 10))

def open_save_dialog():
    """Ask whether to save current digsheet as PNG or PDF, then call existing flows."""
    dlg = tk.Toplevel(root)
    dlg.title("Save")
    dlg.geometry("300x160+520+260")
    dlg.configure(bg="white")
    dlg.grab_set()

    tk.Label(dlg, text="Save as:", bg="white",
             font=("Segoe UI", 11, "bold")).pack(pady=(12, 6))

    mode_var = tk.StringVar(value="png")
    opts = tk.Frame(dlg, bg="white")
    opts.pack(pady=4)
    tk.Radiobutton(opts, text="PNG (image)", variable=mode_var, value="png", bg="white").grid(row=0, column=0, padx=10)
    tk.Radiobutton(opts, text="PDF (single page)", variable=mode_var, value="pdf", bg="white").grid(row=0, column=1, padx=10)

    def do_save():
        dlg.destroy()
        if mode_var.get() == "png":
            # uses your existing PNG saver (asks for filename and saves stitched image)
            capture_sections(1, 5)
        else:
            # uses your existing PDF saver (asks for filename and writes PDF)
            save_all_sections_as_pdf()

    btns = tk.Frame(dlg, bg="white")
    btns.pack(pady=14)
    tk.Button(btns, text="Save", command=do_save).grid(row=0, column=0, padx=10)
    tk.Button(btns, text="Cancel", command=dlg.destroy).grid(row=0, column=1, padx=10)


import tempfile
import os
import win32api
import win32print
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image

def print_all_sections_dialog():
    merged = capture_sections_image(1, 5)  # your function to get stitched image
    if merged is None:
        messagebox.showerror("Error", "No sections captured")
        return

    # Save stitched image to a temporary PNG
    temp_img = tempfile.mktemp(suffix=".png")
    merged.save(temp_img, "PNG")  # we don't enforce fit-to-paper here

    # --- Printer chooser dialog ---
    def get_printers():
        printers = [
            p[2] for p in win32print.EnumPrinters(
                win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
            )
        ]
        return printers

    def send_to_printer(printer_name, file_path):
        try:
            # use "print" instead of "printto" for broader compatibility
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

    # Create a nice modal dialog
    dialog = tk.Toplevel()
    dialog.title("Print Report")
    dialog.geometry("400x200")
    dialog.configure(bg="white")
    dialog.grab_set()  # make modal

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
        printer_combo.current(0)  # default select first
    printer_combo.pack(pady=10)

    button_frame = tk.Frame(dialog, bg="white")
    button_frame.pack(pady=20)

    ttk.Button(button_frame, text="Print", command=print_selected).grid(row=0, column=0, padx=10)
    ttk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=10)

    dialog.mainloop()



def open_batch_dialog_new():
    """Open dialog to pick IDs AND choose export format (PDF/PNG)."""
    dialog = tk.Toplevel(root)
    dialog.title("Batch Export")
    dialog.geometry("360x280+500+200")
    dialog.configure(bg="white")
    dialog.grab_set()

    tk.Label(dialog, text="Select defects to export",
             bg="white", font=("Segoe UI", 11, "bold")).pack(pady=10)

    # --- Range fields ---
    range_frame = tk.Frame(dialog, bg="white")
    range_frame.pack(pady=5)

    tk.Label(range_frame, text="Start ID:", bg="white").grid(row=0, column=0, padx=5)
    start_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=start_var, width=8).grid(row=0, column=1, padx=5)

    tk.Label(range_frame, text="End ID:", bg="white").grid(row=0, column=2, padx=5)
    end_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=end_var, width=8).grid(row=0, column=3, padx=5)

    # --- OR custom IDs ---
    tk.Label(dialog, text="OR Enter IDs (comma-separated):",
             bg="white").pack(pady=(15, 2))
    custom_var = tk.StringVar()
    tk.Entry(dialog, textvariable=custom_var, width=32).pack(pady=2)

    # --- Export mode choice ---
    mode_var = tk.StringVar(value="pdf")
    mode_frame = tk.Frame(dialog, bg="white")
    mode_frame.pack(pady=12)
    tk.Label(mode_frame, text="Export as:", bg="white").grid(row=0, column=0, padx=(0,8))
    tk.Radiobutton(mode_frame, text="PDF (one multi-page file)",
                   variable=mode_var, value="pdf", bg="white").grid(row=0, column=1, padx=6)
    tk.Radiobutton(mode_frame, text="PNG (one file per defect)",
                   variable=mode_var, value="png", bg="white").grid(row=0, column=2, padx=6)

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
                messagebox.showwarning("Batch Export", "Please enter a range or some IDs.")
                return

            ids = sorted(set(ids))
            export_mode = mode_var.get()

            dialog.destroy()
            root.after(200, lambda: batch_export_with_ui(ids, export_mode))

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please use numbers only.")

    btns = tk.Frame(dialog, bg="white")
    btns.pack(pady=16)
    tk.Button(btns, text="Export", command=run_export).grid(row=0, column=0, padx=10)
    tk.Button(btns, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=10)


def open_preview_dialog():
    """Dialog to collect defect range or custom IDs for preview."""
    dialog = tk.Toplevel(root)
    dialog.title("Multi Preview")
    dialog.geometry("340x280+500+200")
    dialog.configure(bg="white")
    dialog.grab_set()

    tk.Label(dialog, text="Select defects to preview",
             bg="white", font=("Segoe UI", 11, "bold")).pack(pady=10)

    # --- Range fields ---
    range_frame = tk.Frame(dialog, bg="white")
    range_frame.pack(pady=5)

    tk.Label(range_frame, text="Start ID:", bg="white").grid(row=0, column=0, padx=5)
    start_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=start_var, width=8).grid(row=0, column=1, padx=5)

    tk.Label(range_frame, text="End ID:", bg="white").grid(row=0, column=2, padx=5)
    end_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=end_var, width=8).grid(row=0, column=3, padx=5)

    # --- OR custom IDs ---
    tk.Label(dialog, text="OR Enter IDs (comma-separated):", bg="white").pack(pady=(15, 2))
    custom_var = tk.StringVar()
    tk.Entry(dialog, textvariable=custom_var, width=30).pack(pady=2)

    # --- Preview Mode (PNG or PDF) ---
    mode_var = tk.StringVar(value="png")
    mode_frame = tk.Frame(dialog, bg="white")
    mode_frame.pack(pady=10)

    tk.Label(mode_frame, text="Preview as:", bg="white").grid(row=0, column=0, padx=5)
    tk.Radiobutton(mode_frame, text="PNG", variable=mode_var, value="png", bg="white").grid(row=0, column=1, padx=5)
    tk.Radiobutton(mode_frame, text="PDF", variable=mode_var, value="pdf", bg="white").grid(row=0, column=2, padx=5)

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
                messagebox.showwarning("Multi Preview", "Please enter a range or some IDs.")
                return
            ids = sorted(set(ids))
            dialog.destroy()
            root.after(
                        200,
                        lambda: batch_preview(
                            ids,
                            mode_var.get(),
                            embed=(mode_var.get().lower() == "png")  # PNG previews show inside right panel
                        )
                    )

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please use numbers only.")

    # --- Buttons ---
    tk.Button(dialog, text="Preview", command=run_preview).pack(pady=15)
    tk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)


def _clear_preview_holder():
    # unbind the global arrow keys used by the embedded viewer
    try:
        preview_holder.unbind_all("<Left>")
        preview_holder.unbind_all("<Right>")
    except Exception:
        pass
    for w in preview_holder.winfo_children():
        w.destroy()
    try: 
        _show_preview_placeholder()
    except: 
        pass

def reset_ui():
    """Return the app to its just-opened state."""
    global batch_cancelled, progress_frame_ref
    batch_cancelled = False

    # 1) inputs
    try: defect_entry.delete(0, tk.END)
    except: pass

    # 2) StringVars
    for var in (
        pipe_id_var, length_var, wt_var, latitude_var, longitude_var, altitude_var,
        client_var, pipeline_name_var, pipeline_section_var
    ):
        try: var.set("")
        except: pass

    # 3) Feature Description labels
    for lbl in feature_labels.values():
        try: lbl.config(text="")
        except: pass

    # 4) Comment area
    try: comment_placeholder.config(text="")
    except: pass

    # 5) Small “Feature Location on Pipe” drawing
    try: pipe_canvas1.delete("all")
    except: pass

    # 6) Dynamic overlays on the big Pipe Location canvas
    for tag in ("upstream_text", "flange_text", "us_arrow", "ds_arrow", "bend_text", "pipe_icon"):
        try: pipe_canvas.delete(tag)
        except: pass
    # (keeps the static grid/labels you drew at startup)

    # 7) Close right-side preview/progress
    try: _clear_preview_holder()
    except: pass
    try:
        if progress_frame_ref and progress_frame_ref.winfo_exists():
            progress_frame_ref.destroy()
        progress_frame_ref = None
    except: pass

    # 8) Scroll to top
    try: canvas.yview_moveto(0.0)
    except: pass

    print("[reset] UI returned to initial state.")


def reset_left_panel():
    """Reset only the main (left) digsheet area, keep the right preview intact."""
    # Don’t touch: preview_holder, progress panels, toolbar, defect_entry, etc.

    # 1) StringVars shown in the left panels
    for var in (pipe_id_var, length_var, wt_var, latitude_var, longitude_var,altitude_var,
                client_var, pipeline_name_var, pipeline_section_var):
        try: var.set("")
        except: pass

    # 2) Feature Description labels
    for lbl in feature_labels.values():
        try: lbl.config(text="")
        except: pass

    # 3) Comment area
    try: comment_placeholder.config(text="")
    except: pass

    # 4) Small “Feature Location on Pipe” drawing (top-left canvas)
    try: pipe_canvas1.delete("all")
    except: pass

    # 5) Big “Pipe Location” canvas — remove only dynamic overlays
    for tag in ("upstream_text", "flange_text", "us_arrow", "ds_arrow", "bend_text", "pipe_icon"):
        try: pipe_canvas.delete(tag)
        except: pass

    # 6) Scroll main canvas to top so it looks fresh
    try: canvas.yview_moveto(0.0)
    except: pass

    print("[reset] Left panel cleared (preview kept).")



tk.Label(group1, text="Enter Defect S.no:", bg="white").pack(side="left", padx=(2, 6))
defect_entry = tk.Entry(group1, width=8)
defect_entry.pack(side="left", padx=(0, 6))

tk.Button(group1, text="Load",  command=on_load_click).pack(side="left", padx=3)
tk.Button(group1, text="Save Current",  command=open_save_dialog).pack(side="left", padx=3)
tk.Button(group1, text="Print current", command=print_all_sections_dialog).pack(side="left", padx=3)


# -- Group 2 (plain): Batch Export
group2 = tk.Frame(toolbar, bg="white")
group2.pack(side="left", padx=2)
tk.Button(group2, text="Batch Export", command=open_batch_dialog_new).pack(side="left")

# -- Group 3 (plain): Multi Preview
group3 = tk.Frame(toolbar, bg="white")
group3.pack(side="left", padx=2)
tk.Button(group3, text="MultiPreview", command=open_preview_dialog).pack(side="left")
tk.Button(group3, text="Reset", command=reset_ui).pack(side="left", padx=3)
# --- Embedded preview area (below the toolbar buttons) ---
preview_holder = tk.Frame(
    input_frame, bg="white",
    highlightbackground="#e8e8e8", highlightthickness=3
)
preview_holder.pack(side="top", fill="both", expand=True, pady=(8, 0))

_preview_placeholder_ref = None

def _show_preview_placeholder(msg="No previews yet.\nUse MultiPreview to generate."):
    """Wipe the preview area and show a centered placeholder message."""
    global _preview_placeholder_ref
    _clear_preview_holder()
    _preview_placeholder_ref = tk.Label(
        preview_holder,
        text=msg,
        bg="white",
        fg="gray50",
        font=("Segoe UI", 11, "bold"),
        justify="center"
    )
    # Center it inside preview_holder and keep it centered on resize
    _preview_placeholder_ref.place(relx=0.5, rely=0.5, anchor="center")

_show_preview_placeholder()

try:
    icon_path = os.getcwd() + "/dig" + "/digsheet_icon/"
    valve_img = ImageTk.PhotoImage(Image.open(icon_path + "valve.png").resize((18, 18)))
    bend_img = ImageTk.PhotoImage(Image.open(icon_path + "bend.png").resize((18, 18)))
    flange_img = ImageTk.PhotoImage(Image.open(icon_path + "flange.png").resize((18, 18)))
    flowtee_img = ImageTk.PhotoImage(Image.open(icon_path + "flowtee.png").resize((18, 18)))
    magnet_img = ImageTk.PhotoImage(Image.open(icon_path + "magnet.png").resize((18, 18)))
except Exception as e:
    print("Image loading error:", e)
    valve_img = bend_img = flange_img = flowtee_img = magnet_img = None

# --- Step 1: Create a Canvas inside a container Frame ---
container = tk.Frame(root)
container.pack(fill="both", expand=True)

canvas = tk.Canvas(container, bg="white")
canvas.pack(side="left", fill="both", expand=True)


def _yscroll_set(lo, hi):
    """Custom yscrollcommand: update thumb AND float the bar when needed."""
    scrollbar.set(lo, hi)
    try:
        lo_f, hi_f = float(lo), float(hi)
    except Exception:
        lo_f, hi_f = 0.0, 1.0

    # if everything fits (no scrolling), hide it
    if hi_f - lo_f >= 0.999:
        scrollbar.place_forget()
    else:
        # place inside canvas, 8px in from the right edge, vertically centered
        # adjust x to change the 'threshold' from the digsheet right side
        scrollbar.place(in_=canvas, relx=1.0, x=-8, rely=0.5, anchor="e", relheight=0.98)

canvas.configure(yscrollcommand=_yscroll_set)



# --- Step 3: Create a scrollable frame inside the canvas ---
scrollable_frame = tk.Frame(canvas, bg="white")

from tkinter import ttk

# a slimmer, pill-ish scrollbar (optional)
ttk.Style().configure(
    "Floating.Vertical.TScrollbar",
    troughcolor="white",
    background="gray60",
    bordercolor="white",
    lightcolor="gray60",
    darkcolor="gray60"
)

# use classic tk.Scrollbar so we can set width
scrollbar = tk.Scrollbar(
    canvas,
    orient="vertical",
    command=canvas.yview,
    width=9,          # ← thickness in pixels
)




def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))


scrollable_frame.bind("<Configure>", on_frame_configure)

# Embed the scrollable frame window in canvas
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

def _on_mousewheel(event):
    if event.delta:  # Windows / MacOS
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    elif event.num == 4:  # Linux scroll up
        canvas.yview_scroll(-3, "units")
    elif event.num == 5:  # Linux scroll down
        canvas.yview_scroll(3, "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)       # Windows/Mac
canvas.bind_all("<Button-4>", _on_mousewheel)          # Linux scroll up
canvas.bind_all("<Button-5>", _on_mousewheel)          # Linux scroll down

def load_pipe_tally(pipe_tally_file):
    try:
        ext = os.path.splitext(pipe_tally_file)[1].lower()
        if ext in (".xlsx", ".xls"):
            # Excel → DataFrame
            return pd.read_excel(pipe_tally_file)
        elif ext == ".csv":
            return pd.read_csv(pipe_tally_file)
        elif ext in (".pkl", ".pickle"):
            # Real pickle → use pandas helper (or pickle.load)
            return pd.read_pickle(pipe_tally_file)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        print(f"Error loading pipe_tally ({pipe_tally_file}): {e}")
        messagebox.showerror("Load Error", f"Could not load: {pipe_tally_file}\n\n{e}")
        return None

# Variables
pipe_id_var = tk.StringVar()
length_var = tk.StringVar()
wt_var = tk.StringVar()
latitude_var = tk.StringVar()
longitude_var = tk.StringVar()
altitude_var = tk.StringVar()



# NEW: Client description variables
client_var = tk.StringVar()
pipeline_name_var = tk.StringVar()
pipeline_section_var = tk.StringVar()

from PIL import ImageGrab

import io
import img2pdf
from tkinter import filedialog, messagebox
import tempfile, os







def upscale_image(img, target_dpi=600, base_dpi=96, scale_limit=2.0):
    """
    Upscale moderately (max scale_limit ×), and set DPI metadata.
    This avoids bloated file sizes while improving sharpness.
    """
    scale = target_dpi / base_dpi
    if scale > scale_limit:
        scale = scale_limit  # cap scaling (e.g. 2× max)

    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS), target_dpi

def save_all_sections_as_pdf():
    merged = capture_sections_image(1, 5)
    if merged is None:
        messagebox.showerror("Error", "No sections were captured.")
        return

    pdf_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        initialfile="",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not pdf_path:
        return

    # --- upscale moderately & embed DPI ---
    merged, dpi = upscale_image(merged, target_dpi=300, base_dpi=96)

    buf = io.BytesIO()
    merged.save(buf, format="PNG", dpi=(dpi, dpi))  # embed DPI metadata
    buf.seek(0)

    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(buf.getvalue()))

    messagebox.showinfo("Saved!", f"High-quality PDF created:\n{pdf_path}")




def get_section_coords():
    root.update_idletasks()
    sections = {
        "Client Description": client_desc_frame,
        "Feature Location on Pipe": main_frame,
        "Comment": comment_frame,
        "Feature Description": feature_desc_frame,
        "Pipe Location": third_frame,
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







def print_section_coords():
    """Debug: print the coords in terminal."""
    coords = get_section_coords()
    for name, (x0, y0, x1, y1) in coords.items():
        print(f"{name} -> x_start={x0}, y_start={y0}, x_end={x1}, y_end={y1}")

def save_individual_section(section_id):
    """
    Save a screenshot of a specific section using its own threshold tuple.
      section_id: 1=Feature Location on Pipe, 2=Feature Description, 3=Pipe Location, 4=Client Description
    Thresholds come from SECTION_THRESHOLDS[name] as (dx0, dy0, dx1, dy1).
    """
    if section_id not in SECTION_MAP:
        messagebox.showerror("Error", "Invalid section ID!")
        return

    name = SECTION_MAP[section_id]
    all_coords = get_section_coords()
    if name not in all_coords:
        messagebox.showerror("Error", f"Section '{name}' not found.")
        return

    # base coords
    x0, y0, x1, y1 = all_coords[name]

    # per-section thresholds (pixels)
    dx0, dy0, dx1, dy1 = SECTION_THRESHOLDS.get(name, (0, 0, 0, 0))

    # apply adjustments
    ax0 = x0 + dx0
    ay0 = y0 + dy0
    ax1 = x1 + dx1
    ay1 = y1 + dy1

    # clamp to screen bounds and ensure valid bbox
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    ax0 = max(0, min(ax0, screen_w - 1))
    ay0 = max(0, min(ay0, screen_h - 1))
    ax1 = max(1, min(ax1, screen_w))
    print(f"ax1 for {section_id} is {ax1}")
    ay1 = max(1, ay1)

    if ax1 <= ax0:
        ax1 = min(screen_w, ax0 + 1)
    if ay1 <= ay0:
        ay1 = min(screen_h, ay0 + 1)

    bbox = (ax0, ay0, ax1, ay1)
    print(f"[{name}] base=({x0},{y0},{x1},{y1}) thresholds=({dx0},{dy0},{dx1},{dy1}) -> bbox={bbox}")

    try:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialfile=f"{name.replace(' ', '_')}.png"
        )
        if not filepath:
            return
        print(f"[{name}] base=({x0},{y0},{x1},{y1}) thresholds=({dx0},{dy0},{dx1},{dy1}) -> adjusted=({ax0},{ay0},{ax1},{ay1}) screen_h={root.winfo_screenheight()}")

        img = ImageGrab.grab(bbox=bbox)
        img.save(filepath)
        messagebox.showinfo("Saved!", f"{name} saved successfully:\n{filepath}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save {name}:\n{e}")
from PIL import ImageGrab, Image
import time
from tkinter import filedialog, messagebox

def capture_sections(section_start=1, section_end=5):
    """
    Capture sections in a given ID range and merge them vertically.
    If section_id is 1–4 → scroll to top, if 5 → scroll to bottom.
    """
    # --- Ask user where to save ---
    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png")],
        initialfile=""
    )
    if not filepath:
        return  # user cancelled

    images = []

    for section_id in range(section_start, section_end + 1):
        if section_id not in SECTION_MAP:
            print(f"⚠️ Section ID {section_id} not in SECTION_MAP, skipping.")
            continue

        # Scroll logic
        if section_id in [1, 2, 3, 4]:
            canvas.yview_moveto(0.0)
        elif section_id == 5:
            canvas.yview_moveto(1.0)
        root.update()
        time.sleep(0.4)  # let GUI settle

        # Get coordinates for this section
        coords = get_section_coords()
        name = SECTION_MAP[section_id]
        if name not in coords:
            print(f"⚠️ Section {name} frame not found, skipping.")
            continue

        x0, y0, x1, y1 = coords[name]
        dx0, dy0, dx1, dy1 = SECTION_THRESHOLDS.get(name, (0, 0, 0, 0))
        bbox = (x0 + dx0, y0 + dy0, x1 + dx1, y1 + dy1)

        print(f"📸 Capturing {name} @ {bbox}")
        img = ImageGrab.grab(bbox=bbox)
        images.append(img)

    if not images:
        messagebox.showerror("Error", "No sections were captured.")
        return

    # Merge vertically
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

    # Save final stitched image
    merged.save(filepath)
    messagebox.showinfo("Saved!", f"All sections saved successfully:\n{filepath}")
    print(f"✅ Combined image saved to {filepath}")
 


import io
def capture_sections_image(section_start=1, section_end=5):
    images = []
    for section_id in range(section_start, section_end + 1):
        if section_id not in SECTION_MAP:
            continue
        canvas.yview_moveto(0.0 if section_id in [1,2,3,4] else 1.0)
        root.update(); time.sleep(0.4)

        coords = get_section_coords()
        name = SECTION_MAP[section_id]
        if name not in coords: 
            continue

        x0, y0, x1, y1 = coords[name]
        dx0, dy0, dx1, dy1 = SECTION_THRESHOLDS.get(name, (0,0,0,0))
        bbox = (x0+dx0, y0+dy0, x1+dx1, y1+dy1)
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





import img2pdf
import os
from tkinter import filedialog, messagebox
from PIL import ImageGrab

def save_as_pdf_high_quality():
    """Save digsheet as high-quality PDF using img2pdf"""
    filepath = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    if not filepath:
        return

    if not scrollable_active:
        # Non-scrollable case
        root.update_idletasks()
        x0 = root.winfo_rootx()
        y0 = root.winfo_rooty()
        x1 = x0 + root.winfo_width()
        y1 = y0 + root.winfo_height()
        
        # Capture screenshot
        img = ImageGrab.grab(bbox=(x0, y0, x1 - 660, y1 + 220))
        temp_png = filepath + "_temp.png"
        img.save(temp_png)
        
        try:
            with open(filepath, "wb") as f:
                f.write(img2pdf.convert(temp_png))
            messagebox.showinfo("Saved!", f"High-quality PDF saved:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PDF:\n{e}")
        finally:
            if os.path.exists(temp_png):
                os.remove(temp_png)
    else:
        # Scrollable case - same merging logic as before
        try:
            root.update_idletasks()
            x2 = feature_desc_frame.winfo_rootx()
            y0 = root.winfo_rooty()
            y2 = feature_desc_frame.winfo_rooty()
            y2_end = y2 + feature_desc_frame.winfo_height()
            x3 = third_frame.winfo_rootx()
            y3 = third_frame.winfo_rooty()
            x4 = x3 + third_frame.winfo_width()
            y4 = y3 + third_frame.winfo_height()
            
            x_left = min(x2, x3) - 5
            x_right = max(x2 + feature_desc_frame.winfo_width(), x4 + 10)

            # Capture top section (includes Client Description)
            canvas.yview_moveto(0.0)
            root.update(); time.sleep(0.3)
            img_top = ImageGrab.grab(bbox=(x_left, y0, x_right, y2_end + 5))

            # Capture bottom section
            canvas.yview_moveto(1.0)
            root.update(); time.sleep(0.6)
            img_bot = ImageGrab.grab(bbox=(x_left, y3 - 95, x_right, y4 - 85))

            # Resize and merge
            if img_top.width != img_bot.width:
                img_bot = img_bot.resize((img_top.width, img_bot.height))

            total_height = img_top.height + img_bot.height
            merged = Image.new("RGB", (img_top.width, total_height), "white")
            merged.paste(img_top, (0, 0))
            merged.paste(img_bot, (0, img_top.height))
            
            # Save merged image temporarily
            temp_png = filepath + "_temp.png"
            merged.save(temp_png)
            
            # Convert to PDF using img2pdf
            with open(filepath, "wb") as f:
                f.write(img2pdf.convert(temp_png))
            messagebox.showinfo("Saved!", f"High-quality PDF saved:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save merged PDF:\n{e}")
        finally:
            if os.path.exists(temp_png):
                os.remove(temp_png)

def hms_to_angle(hms):
    if isinstance(hms, str):
        h, m, s = map(int, hms.split(":"))
    else:  # Assume it's a datetime.time object
        h, m, s = hms.hour, hms.minute, hms.second

    angle = (h % 12) * 30 + m * 0.5 + s * (0.5 / 60)
    return angle

# Drawing Function
def draw_pipe(pipe_canvas1, pipe_length, upstream, clock):
    pipe_canvas1.delete("all")
    width, height = 320, 120
    x0, y0 = 40, 30
    x1, y1 = x0 + width, y0 + height
    mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
    radius = height // 2 - 10

    # Pipe shape
    pipe_canvas1.create_oval(x0, y0, x0 + 40, y1, outline="black", width=2)  # Front
    pipe_canvas1.create_oval(x1 - 40, y0, x1, y1, outline="black", width=2)  # Back
    pipe_canvas1.create_line(x0 + 20, y0, x1 - 20, y0, fill="black", width=2)
    pipe_canvas1.create_line(x0 + 20, y1, x1 - 20, y1, fill="black", width=2)

    # Horizontal center dashed line
    pipe_canvas1.create_line(x0, mid_y - 5, x1, mid_y - 5, fill="black", dash=(3, 3))

    # Clock positions
    pipe_canvas1.create_text(x0 - 20, y0 + 10, text="12", anchor="w", font=("Arial", 10))     # moved above
    pipe_canvas1.create_text(x0 + 25, mid_y + 5, text="3", anchor="w", font=("Arial", 10))       # moved right
    pipe_canvas1.create_text(x0 - 17, y1 - 5, text="6", anchor="w", font=("Arial", 10))      # moved below
    pipe_canvas1.create_text(x0 - 10, mid_y + 5, text="9", anchor="e", font=("Arial", 10))       # moved left

    try:
        upstream = float(upstream) if upstream else 0.0
        pipe_length = float(pipe_length) if pipe_length else 0.0
        remaining = round(pipe_length - upstream, 2)
    except:
        upstream = 0.0
        remaining = 0.0

    # Arrow dimensions
    try:
        # Arrow positions
        # Shorter arrows (reduce total length by 10%)
        arrow_y = y0 - 15
        scale_factor = 0.85  # 90% of pipe width
        arrow_length_total = (x1 - x0) * scale_factor
        offset = ((x1 - x0) - arrow_length_total) / 2
        arrow_start_x = x0 + offset
        arrow_end_x = x1 - offset

        arrow1_length = (upstream / pipe_length) * arrow_length_total if pipe_length > 0 else arrow_length_total / 2
        arrow2_length = arrow_length_total - arrow1_length

        # Arrow 1: Upstream
        arrow1_start = arrow_start_x
        arrow1_end = arrow1_start + arrow1_length
        pipe_canvas1.create_line(arrow1_start, arrow_y, arrow1_end, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_line(arrow1_end, arrow_y, arrow1_start, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_text((arrow1_start + arrow1_end) / 2, arrow_y - 10, text=f"{round(upstream, 2)} m", font=("Arial", 10))

        # Arrow 2: Remaining
        arrow2_start = arrow1_end
        arrow2_end = arrow_end_x
        pipe_canvas1.create_line(arrow2_start, arrow_y, arrow2_end, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_line(arrow2_end, arrow_y, arrow2_start, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_text((arrow2_start + arrow2_end) / 2, arrow_y - 10, text=f"{remaining} m", font=("Arial", 10))

        # Defect marker position
        angle_deg = hms_to_angle(clock)
        angle_rad = math.radians(angle_deg)

        # Ellipse setup
        radius_y = radius  # vertical radius of pipe
        center_y = mid_y   # vertical midpoint of the pipe

        # Now apply clock angle to find X and Y offset from center
        defect_x = arrow1_start + (upstream / pipe_length) * arrow_length_total
        adjusted_radius = radius * 0.80  # You can experiment with 0.90–0.95
        defect_y = center_y - int(adjusted_radius * math.cos(angle_rad))

        # Color fill if in front (0–180), border if back (180–360)
        if 0 <= angle_deg <= 180:
            pipe_canvas1.create_rectangle(defect_x - 4, defect_y - 4, defect_x + 4, defect_y + 4, fill="orange", outline="black")
        else:
            pipe_canvas1.create_rectangle(defect_x - 4, defect_y - 4, defect_x + 4, defect_y + 4, outline="orange", width=2)

        # Vertical arrow from pipe bottom to just below defect
        arrow_bottom = y1 - 5
        pipe_canvas1.create_line(
            defect_x - 5, defect_y,
            defect_x - 5, y0,
            arrow=tk.LAST, fill="black", width=1.5
        )
    except Exception as e:
        print("Drawing error:", e)

# Fetch function
def fetch_data():
    try:
        s_no = int(defect_entry.get())
        row = df[df.iloc[:, 0] == s_no]
        if row.empty:
            messagebox.showerror("Error", "Defect number not found!")
            return
        row = row.iloc[0]
        pipe_id_var.set(str(row.iloc[3]))     # Pipe Number
        length_var.set(str(row.iloc[4]))      # Pipe Length
        wt_var.set(str(row.iloc[11]))         # WT

        lat_col = next((c for c in df.columns if c.strip().lower() == "latitude"), None)
        lon_col = next((c for c in df.columns if c.strip().lower() == "longitude"), None)
        alt_col = next((c for c in df.columns if c.strip().lower() == "altitude"), None)
        latitude_var .set(str(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else "")
        longitude_var.set(str(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else "")
        altitude_var.set(str(row[alt_col]) if alt_col and pd.notna(row[alt_col]) else "")

        # Draw pipe
        upstream = row.iloc[2]
        clock_raw = row.iloc[8]
        draw_pipe(pipe_canvas1, row.iloc[4], upstream, clock_raw)

        columns_clean = {col.strip().lower().replace(" ", ""): col for col in df.columns}
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
        "Absolute Distance (m)":1,
        "Length (mm)": 9,
        "Width (mm)": 10,
        "Max. Depth(%)": 12,
        "Orientation(hr:min)": 8,
        "Latitude": None,
        "Longitude": None
        }

        for label, col_index in excel_mapping.items():
            if col_index is not None:
                value = row.iloc[col_index] if col_index < len(row) else ""

                # Format based on label
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

                feature_labels[label].config(text=str(value))

        # --- Remaining Wall Thickness Calculation ---
        try:
            wt = float(row.iloc[11])
            max_depth = float(row.iloc[12])
            remaining_wt = round(wt - (wt * max_depth / 100), 1)
        except:
            remaining_wt = ""

        feature_labels["Remaining wall thickness (mm)"].config(text=str(remaining_wt))
        # --- Handle Latitude / Longitude gracefully ---
        lat_val = row.get(latitude_col, "") if latitude_col else ""
        lon_val = row.get(longitude_col, "") if longitude_col else ""
        feature_labels["Latitude"].config(text=str(lat_val))
        feature_labels["Longitude"].config(text=str(lon_val))

    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid S.no")


from PIL import Image, ImageTk

# ---------- Client Description (at top) ----------
client_desc_frame = tk.Frame(
    scrollable_frame, bg="white", padx=5, pady=2,
    highlightbackground="black", highlightthickness=1
)
client_desc_frame.pack(fill="both", padx=(15, 15), pady=(5,0))

# Title centered across the whole frame
tk.Label(
    client_desc_frame, text="Client Description:", bg="white",
    fg="deepskyblue", font=("Arial", 10, "bold")
).pack(side="top", pady=(2, 6))   # pack centers by default

# Left side (labels + entries) in a sub-frame
left_frame = tk.Frame(client_desc_frame, bg="white")
left_frame.pack(side="left", fill="both", expand=True)

left_frame.grid_columnconfigure(0, weight=0, minsize=130)  # fixed label column
left_frame.grid_columnconfigure(1, weight=1)               # entry column grows


# Variables
client_var = tk.StringVar()
pipeline_name_var = tk.StringVar()
pipeline_section_var = tk.StringVar()

# Fields
fields_top = [
    ("Client", client_var),
    ("Pipeline Name", pipeline_name_var),
    ("Pipeline Section", pipeline_section_var),
]

for r, (txt, var) in enumerate(fields_top):
    tk.Label(
        left_frame, text=f"{txt}:", bg="white", anchor="w", font=("Arial", 9)
    ).grid(row=r, column=0, sticky="w", padx=(10, 6), pady=(2, 2))

    tk.Entry(
        left_frame, textvariable=var, width=40, bg="white",
        bd=0, highlightthickness=0, relief="flat"
    ).grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=(2, 2))


# Column sizing
left_frame.grid_columnconfigure(0, weight=0)  # labels minimal
left_frame.grid_columnconfigure(1, weight=1)  # entries expand

# --- Logo: right edge, vertically centered ---
try:
    logo_img = Image.open(r"F:\work_new\client_software\PIE_dv_new\ui\icons\vdt-logo.png").resize((100, 100))
    logo_tk = ImageTk.PhotoImage(logo_img)
    logo_lbl = tk.Label(client_desc_frame, image=logo_tk, bg="white")
    logo_lbl.place(relx=1.0, rely=0.5, anchor="e", x=-10)  # right side, y-center, 10px right padding
    client_desc_frame.logo_ref = logo_tk  # prevent garbage collection
except Exception as e:
    print("Logo load failed:", e)


# Main content
main_frame = tk.Frame(scrollable_frame, bg="white")
main_frame.pack(pady=5, fill="x", padx=10)

# --- Feature on Pipe (Left Box) ---
feature_frame = tk.Frame(main_frame, bg="white", padx=5, pady=5, highlightbackground="black", highlightthickness=1)
feature_frame.pack(side="left", fill="both", expand=True, padx=5)

# Title inside Feature on Pipe box
tk.Label(feature_frame, text="Feature Location on Pipe:", bg="white", fg="deepskyblue", font=("Arial", 10, "bold")).pack(pady=(0, 5))
pipe_canvas1 = tk.Canvas(feature_frame, width=360, height=160, bg="white", highlightthickness=0)
pipe_canvas1.pack()

# --- Pipe Description (Right Box) ---
desc_frame = tk.Frame(main_frame, bg="white", padx=5, pady=5, highlightbackground="black", highlightthickness=1)
desc_frame.pack(side="left", fill="both", expand=True, padx=5)

# Title inside Pipe Description box
# tk.Label(desc_frame, text="Pipe Description:", bg="white", fg="deepskyblue",
#         font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=5, padx=5,pady=5, sticky="ew")

tk.Label(desc_frame, text="Pipe Description:", bg="white", fg="deepskyblue",
         font=("Arial", 10, "bold")
).grid(row=0, column=0, columnspan=5, padx=5, pady=(0, 5), sticky="ew")



# Layout fields
fields = [
    ("Pipe Number", pipe_id_var),
    ("Pipe Length (m)", length_var),
    ("WT (mm)", wt_var),
    ("Latitude", latitude_var),
    ("Longitude", longitude_var),
    ("Altitude (m)", altitude_var),
]



for i, (label, var) in enumerate(fields, start=1):  # start at row 1
    tk.Label(desc_frame, text=label + ":", bg="white", anchor="w", font=("Arial", 9))\
      .grid(row=i, column=0, sticky="w", padx=(5, 2), pady=(2, 2))
    tk.Label(desc_frame, textvariable=var, bg="white", anchor="w", font=("Arial", 9))\
      .grid(row=i, column=1, sticky="w", padx=(2, 10), pady=(2, 2))

# Ensure the grid expands to center-align contents
for col in range(2):  # Adjust columns 0 and 1
    desc_frame.grid_columnconfigure(col, weight=1)

# ---------- Comment Section (blank for now) ----------
comment_frame = tk.Frame(
    scrollable_frame, bg="white", padx=5, pady=2,
    highlightbackground="black", highlightthickness=1
)
comment_frame.pack(fill="both", padx=(15, 15), pady=(5,5))
# Title centered across the whole frame
tk.Label(
    comment_frame, text="Comment:", bg="white",
    fg="deepskyblue", font=("Arial", 10, "bold")
).pack(side="top", pady=(0, 5))

# Placeholder area (kept blank for now, will fill dynamically later)
comment_placeholder = tk.Label(
    comment_frame, text="", bg="white", anchor="w", justify="left",
    font=("Arial", 9)
)
comment_placeholder.pack(fill="both", expand=True, padx=10, pady=20)

# -------------------------------------------------------------------------------------------

# --- Feature Description UI Block ---
feature_desc_frame = tk.Frame(scrollable_frame, bg="white", padx=5, pady=2, highlightbackground="black", highlightthickness=1)
feature_desc_frame.pack(fill="both", padx=15)

# Labels dictionary to update from fetch_data()
feature_labels = {}

# Label names
left_fields = ["Feature", "Feature type", "Anomaly dimension class", "Surface Location",
            "Remaining wall thickness (mm)", "ERF", "Safe pressure (kg/cm²)"]
right_fields = ["Absolute Distance (m)", "Length (mm)", "Width (mm)", "Max. Depth(%)",
                "Orientation(hr:min)", "Latitude", "Longitude"]

# Give all columns proper weights to allow full expansion
for col in range(5):  # 0 to 4
    feature_desc_frame.grid_columnconfigure(col, weight=1)

# Configure grid columns for spacing and balance
feature_desc_frame.grid_columnconfigure(2, minsize=80)  # Spacer

# Title centered inside entire frame (spanning all columns)
section_title = tk.Label(feature_desc_frame, text="Feature Description:", bg="white", fg="deepskyblue", font=("Arial", 10, "bold"), anchor="center", justify="center")
section_title.grid(row=0, column=0, columnspan=5, pady=(0, 5), sticky="ew")

# Padding configuration
label_padx = (5, 2)
value_padx = (2, 10)

# Left fields
for i, label_text in enumerate(left_fields):
    tk.Label(feature_desc_frame, text=label_text + ":", bg="white", anchor="w", font=("Arial", 9)).grid(
        row=i+1, column=0, sticky="w", padx=label_padx, pady=2)
    label = tk.Label(feature_desc_frame, text="", bg="white", anchor="w", font=("Arial", 9))
    label.grid(row=i+1, column=1, sticky="w", padx=value_padx, pady=2)
    feature_labels[label_text] = label

# Right fields
for i, label_text in enumerate(right_fields):
    tk.Label(feature_desc_frame, text=label_text + ":", bg="white", anchor="w", font=("Arial", 9)).grid(
        row=i+1, column=3, sticky="w", padx=label_padx, pady=2)
    label = tk.Label(feature_desc_frame, text="", bg="white", anchor="w", font=("Arial", 9))
    label.grid(row=i+1, column=4, sticky="w", padx=value_padx, pady=2)
    feature_labels[label_text] = label

# --- Third Block Setup ---
third_frame = tk.Frame(scrollable_frame, bg="white", padx=10, pady=10, highlightbackground="black", highlightthickness=1)
third_frame.pack(fill="both", padx=15, pady=4)

# Title for the Third Block
tk.Label(third_frame, text="Pipe Location:", bg="white", fg="deepskyblue",
        font=("Arial", 9, "bold")).grid(row=0, column=0, columnspan=5,  sticky="ew")

# --- Sub-blocks Representation ---
pipe_canvas = tk.Canvas(third_frame, width=650, height=370, bg="white", highlightthickness=0)
pipe_canvas.grid(row=1, column=0, columnspan=5)
pipe_canvas.update()
canvas_width = pipe_canvas.winfo_width()
canvas_height = pipe_canvas.winfo_height()

# Ensure grid expansion
for col in range(5):  # Column expansion
    third_frame.grid_columnconfigure(col, weight=1)

# Midpoint
mid_x = int(canvas_width/2)
mid_y = int(canvas_height/2)

# Central vertical line
pipe_canvas.create_line(mid_x, 30, mid_x, mid_y + 150, arrow=tk.FIRST)


def read_constants_file(path):
    """Return (client, pipeline_name, pipeline_section) from constants CSV/XLSX."""
    try:
        # read CSV or Excel
        if path.lower().endswith((".csv", ".txt")):
            const_df = pd.read_csv(path)
        else:
            const_df = pd.read_excel(path)

        if const_df.empty:
            return "", "", ""

        # case-insensitive column lookup (handles the 'PIPLELINE_NAME' typo)
        cols = {c.strip().upper(): c for c in const_df.columns}

        row = const_df.iloc[0]
        def val(key):
            col = cols.get(key)
            v = row[col] if col in row else ""
            return "" if (pd.isna(v) or v is None) else str(v)

        client = val("CLIENT")
        pipeline_name = val("PIPELINE_NAME")   # as you named it
        pipeline_section = val("PIPELINE_SECTION")

        return client, pipeline_name, pipeline_section
    except Exception as e:
        print(f"Failed to read constants file '{path}': {e}")
        return "", "", ""



def get_dynamic_weld_and_feature_data():
    """
    Given a defect number and the dataframe,
    return upstream weld and absolute distance for dynamic use.
    """
    try:
        feature_keywords = ["flange", "valve", "flow tee", "magnet"]

        s_no = int(defect_entry.get())
        row = df[df.iloc[:, 0] == s_no]
        if row.empty:
            messagebox.showerror("Error", "Defect number not found!")
            return
        row = row.iloc[0]
        upstream_value = float(row.iloc[2])
        absolute_value = float(row.iloc[1])
        upstream_weld = round(abs(absolute_value - upstream_value), 2)

        # Get index of defect row
        defect_idx = df[df.iloc[:, 0] == s_no].index[0]
        defect_row = df.loc[defect_idx]
        defect_distance = float(defect_row.iloc[1])

        # Get column name references
        lat_col = next((c for c in df.columns if c.strip().lower() == "latitude"), None)
        lon_col = next((c for c in df.columns if c.strip().lower() == "longitude"), None)

        features_upstream = []
        features_downstream = []
        bends_upstream = []
        bends_downstream = []

        # Upstream (reverse search)
        for i in range(defect_idx - 1, -1, -1):
            row = df.loc[i]
            feature_name = str(row.iloc[5]).strip().lower()
            if any(f in feature_name for f in feature_keywords):
                features_upstream.append({
                    "name": str(row.iloc[5]),
                    "distance": round(float(row.iloc[1]), 2),
                    "lat": str(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else "",
                    "long": str(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else ""
                })
                if len(features_upstream) == 2:
                    break

        # Downstream (forward search)
        for i in range(defect_idx + 1, len(df)):
            row = df.loc[i]
            feature_name = str(row.iloc[5]).strip().lower()
            if any(f in feature_name for f in feature_keywords):
                features_downstream.append({
                    "name": str(row.iloc[5]),
                    "distance": round(float(row.iloc[1]), 2),
                    "lat": str(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else "",
                    "long": str(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else ""
                })
                if len(features_downstream) == 2:
                    break

        # Upstream bends
        for i in range(defect_idx - 1, -1, -1):
            row = df.loc[i]
            feature_name = str(row.iloc[5]).strip().lower()
            if "bend" in feature_name:
                bends_upstream.append({
                    "name": str(row.iloc[5]),
                    "distance": round(float(row.iloc[1]), 2),
                    "lat": str(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else "",
                    "long": str(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else ""
                })
                if len(bends_upstream) == 3:
                    break

        # Downstream bends
        for i in range(defect_idx + 1, len(df)):
            row = df.loc[i]
            feature_name = str(row.iloc[5]).strip().lower()
            if "bend" in feature_name:
                bends_downstream.append({
                    "name": str(row.iloc[5]),
                    "distance": round(float(row.iloc[1]), 2),
                    "lat": str(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else "",
                    "long": str(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else ""
                })
                if len(bends_downstream) == 3:
                    break

        return {
            "upstream_weld": upstream_weld,
            "features_upstream": features_upstream,
            "features_downstream": features_downstream,
            "bends_upstream": bends_upstream,
            "bends_downstream": bends_downstream
        }
    except:
        return {
            "upstream_weld": "",
            "features_upstream": "",
            "features_downstream": "",
            "bends_upstream": "",
            "bends_downstream": ""
        }


def load_constants_and_fill(constants_path):
    # Read CSV or Excel
    ext = os.path.splitext(constants_path)[1].lower()
    if ext in [".csv", ".txt"]:
        cdf = pd.read_csv(constants_path)
    else:
        cdf = pd.read_excel(constants_path)

    if cdf.empty:
        print("constants file is empty")
        return

    # Normalize headers: lowercase, trim, remove spaces/underscores
    def norm(s): 
        return str(s).strip().lower().replace(" ", "").replace("_", "")
    norm_map = {norm(c): c for c in cdf.columns}

    # Find actual column names regardless of format
    col_client   = norm_map.get("client")
    col_pname    = norm_map.get("pipelinename")
    col_psection = norm_map.get("pipelinesection")

    # Debug if anything missing
    if not col_client:   print("CLIENT column not found")
    if not col_pname:    print("PIPELINE_NAME column not found")
    if not col_psection: print("PIPELINE_SECTION column not found")

    # Use first row (or adjust if you want a specific row)
    row0 = cdf.iloc[0]

    client_var.set("" if not col_client   else str(row0[col_client]))
    pipeline_name_var.set("" if not col_pname    else str(row0[col_pname]))
    pipeline_section_var.set("" if not col_psection else str(row0[col_psection]))


def batch_export(defect_ids, output_mode="pdf", output_path=None):
    """
    Export multiple digsheets in one go.

    Args:
        defect_ids (list[int]): list of defect numbers (S.no) to export
        output_mode (str): "pdf" (multi-page) or "png" (separate images)
        output_path (str): path for output file/folder
    """
    if not defect_ids:
        messagebox.showwarning("Batch Export", "No defect IDs provided.")
        return

    # Ask where to save
    if not output_path:
        if output_mode == "pdf":
            output_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")]
            )
        else:
            output_path = filedialog.askdirectory()
        if not output_path:
            return

    images = []

    for idx, dno in enumerate(defect_ids, start=1):
        try:
            # Load defect into UI
            defect_entry.delete(0, tk.END)
            defect_entry.insert(0, str(dno))
            on_load_click()          # refresh UI with defect
            root.update()
            time.sleep(0.4)          # let UI settle

            # Capture digsheet
            merged = capture_sections_image(1, 5)
            if merged is None:
                print(f"❌ Skipped {dno}, no capture.")
                continue

            if output_mode == "png":
                # Save each as PNG
                out_file = os.path.join(output_path, f"digsheet_{dno}.png")
                merged.save(out_file, "PNG")
                print(f"✅ Saved {out_file}")
            else:
                # For PDF, keep temp images
                temp_path = f"_tmp_{dno}.png"
                merged.save(temp_path)
                images.append(temp_path)
                print(f"✅ Captured {dno} for PDF")

        except Exception as e:
            print(f"Error on defect {dno}: {e}")

    # Combine into single PDF
    if output_mode == "pdf" and images:
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(images))
        for path in images:
            os.remove(path)
        messagebox.showinfo("Batch Export", f"Multi-page PDF created:\n{output_path}")
        print(f"📄 PDF saved at {output_path}")

    elif output_mode == "png":
        messagebox.showinfo("Batch Export", f"Saved {len(defect_ids)} PNG files to:\n{output_path}")


from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
import time, tempfile, os
from pathlib import Path
import img2pdf


def _start_panel_progress(total, title="Generating previews"):
    """
    Render a progress panel inside the right 'preview_holder'.
    Returns (update_func, finish_func).
    - update_func(done) -> updates bar + 'done/total' label
    - finish_func() -> removes the panel
    """
    _clear_preview_holder()  # wipe the preview area first

    prog_frame = tk.Frame(preview_holder, bg="white", highlightbackground="#e8e8e8", highlightthickness=1)
    prog_frame.pack(side="top", fill="x", padx=8, pady=8)

    tk.Label(
        prog_frame, text=title, bg="white", fg="deepskyblue",
        font=("Segoe UI", 11, "bold")
    ).pack(pady=(10, 6))

    status_lbl = tk.Label(prog_frame, text=f"0 / {total}", bg="white", font=("Segoe UI", 10))
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
        style="Custom.Horizontal.TProgressbar"  # you already defined this style
    )
    prog_bar.pack()

    def _update(done):
        prog_var.set(done)
        status_lbl.config(text=f"{done} / {total}")
        prog_frame.update_idletasks()

    def _finish():
        prog_frame.destroy()
        preview_holder.update_idletasks()

    return _update, _finish



def batch_preview(defect_ids, mode="png", embed=False):
    if not defect_ids:
        messagebox.showwarning("Preview", "No defect IDs provided.")
        return

    # show progress bar inside the preview panel
    update_prog, finish_prog = _start_panel_progress(len(defect_ids), title="Generating previews")

    images = []
    done = 0
    for dno in defect_ids:
        try:
            defect_entry.delete(0, tk.END)
            defect_entry.insert(0, str(dno))
            on_load_click()
            root.update()
            time.sleep(0.3)

            merged = capture_sections_image(1, 5)
            if merged:
                images.append((dno, merged))
        except Exception as e:
            print(f"[Preview error] Defect {dno}: {e}")
        finally:
            done += 1
            update_prog(done)  # update the in-panel progress

    # remove progress UI
    finish_prog()

    if not images:
        _show_preview_placeholder("No previews generated.\nCheck your IDs and try again.")
        messagebox.showerror("Preview", "No images generated.")
        return

    # PDF preview/export path (unchanged, opens system viewer)
    if str(mode).lower() == "pdf":
        import tempfile, os, img2pdf
        tmp_paths = []
        try:
            for dno, im in images:
                tmp_path = os.path.join(tempfile.gettempdir(), f"_preview_{dno}.png")
                im.save(tmp_path)
                tmp_paths.append(tmp_path)

            pdf_path = os.path.join(tempfile.gettempdir(), "preview.pdf")
            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert(tmp_paths))

            os.startfile(pdf_path)  # Windows
        finally:
            for p in tmp_paths:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
        return

    # embedded PNG preview in the right panel (with arrow keys)
    if str(mode).lower() == "png" and embed:
        # auto-reset only the left digsheet area, keep preview panel
        reset_left_panel()
        _show_preview_in_panel(images)
        return


    # fallback: separate window (kept as-is for non-embed cases)
    win = tk.Toplevel(root)
    win.title("Multi Preview")
    win.geometry("1000x750")
    win.configure(bg="black")

    current_idx = tk.IntVar(value=0)
    zoom_factor = tk.DoubleVar(value=1.0)

    canvas = tk.Canvas(win, bg="black", highlightthickness=0, cursor="arrow")
    canvas.pack(fill="both", expand=True)

    img_refs = []
    img_id = None
    pan_start = None

    def update_cursor():
        canvas.config(cursor="hand2")

    def show_image(idx):
        nonlocal img_id
        if idx < 0 or idx >= len(images):
            return
        current_idx.set(idx)

        dno, im = images[idx]
        win_w, win_h = win.winfo_width(), win.winfo_height()

        ratio = zoom_factor.get() if zoom_factor.get() != 1.0 else min(win_w / im.width, win_h / im.height)
        new_w, new_h = int(im.width * ratio), int(im.height * ratio)
        im_resized = im.resize((new_w, new_h), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(im_resized)
        img_refs.clear(); img_refs.append(tk_img)

        canvas.delete("all")
        img_id = canvas.create_image(win_w // 2, win_h // 2, image=tk_img, anchor="center")
        page_lbl.config(text=f"Defect {dno} ({idx+1}/{len(images)}) | Zoom: {int(ratio*100)}%")
        update_cursor()

    def prev_img():
        idx = current_idx.get() - 1
        if idx >= 0:
            zoom_factor.set(1.0)
            show_image(idx)

    def next_img():
        idx = current_idx.get() + 1
        if idx < len(images):
            zoom_factor.set(1.0)
            show_image(idx)

    def save_now():
        folder = filedialog.askdirectory(title="Select folder to save PNGs")
        if not folder:
            return
        for dno, im in images:
            im.save(os.path.join(folder, f"digsheet_{dno}.png"), "PNG")
        messagebox.showinfo("Saved", f"Exported {len(images)} PNGs successfully to:\n{folder}")

    def save_current():
        idx = current_idx.get()
        if 0 <= idx < len(images):
            dno, im = images[idx]
            fp = filedialog.asksaveasfilename(
                title=f"Save defect {dno} as PNG",
                defaultextension=".png",
                initialfile=f"digsheet_{dno}.png",
                filetypes=[("PNG files", "*.png")])
            if fp:
                im.save(fp, "PNG")
                messagebox.showinfo("Saved", f"Saved current defect {dno} to:\n{fp}")

    def on_mousewheel(event):
        zoom_factor.set(zoom_factor.get() * (1.25 if event.delta > 0 else 0.8))
        show_image(current_idx.get())

    def start_pan(event):
        nonlocal pan_start
        pan_start = (event.x, event.y)

    def do_pan(event):
        nonlocal pan_start
        if pan_start and img_id:
            dx, dy = event.x - pan_start[0], event.y - pan_start[1]
            canvas.move(img_id, dx, dy)
            pan_start = (event.x, event.y)

    def end_pan(event):
        nonlocal pan_start
        pan_start = None

    controls = tk.Frame(win, bg="black")
    controls.pack(side="bottom", fill="x", pady=5)

    left_frame = tk.Frame(controls, bg="black")
    left_frame.pack(side="left", padx=10)
    tk.Button(left_frame, text="⟵ Prev", command=prev_img).pack(side="left", padx=5)
    page_lbl = tk.Label(left_frame, text="", bg="black", fg="white", font=("Segoe UI", 11, "bold"))
    page_lbl.pack(side="left", padx=10)
    tk.Button(left_frame, text="Next ⟶", command=next_img).pack(side="left", padx=5)

    right_frame = tk.Frame(controls, bg="black")
    right_frame.pack(side="right", padx=10)
    tk.Button(right_frame, text="💾 Save Current", command=save_current).pack(side="left", padx=5)
    tk.Button(right_frame, text="💾 Save All", command=save_now).pack(side="left", padx=5)
    tk.Button(right_frame, text="❌ Close", command=win.destroy).pack(side="left", padx=5)

    win.bind("<Configure>", lambda e: show_image(current_idx.get()) if zoom_factor.get() == 1.0 else None)
    win.bind("<Left>", lambda e: prev_img())
    win.bind("<Right>", lambda e: next_img())
    win.bind("<MouseWheel>", on_mousewheel)
    win.bind("<Button-4>", lambda e: zoom_factor.set(zoom_factor.get() * 1.25) or show_image(current_idx.get()))
    win.bind("<Button-5>", lambda e: zoom_factor.set(zoom_factor.get() * 0.8) or show_image(current_idx.get()))
    canvas.bind("<ButtonPress-1>", start_pan)
    canvas.bind("<B1-Motion>", do_pan)
    canvas.bind("<ButtonRelease-1>", end_pan)

    show_image(0)
    win.mainloop()





def open_batch_dialog(output_mode="pdf"):
    """Open a small window to collect range or custom IDs."""
    dialog = tk.Toplevel(root)
    dialog.title("Batch Export")
    dialog.geometry("320x220+500+200")  # center-ish
    dialog.configure(bg="white")
    dialog.grab_set()   # modal

    # --- Instructions ---
    tk.Label(dialog, text="Select defects to export", bg="white", font=("Segoe UI", 11, "bold")).pack(pady=10)

    # --- Range fields ---
    range_frame = tk.Frame(dialog, bg="white")
    range_frame.pack(pady=5)

    tk.Label(range_frame, text="Start ID:", bg="white").grid(row=0, column=0, padx=5)
    start_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=start_var, width=8).grid(row=0, column=1, padx=5)

    tk.Label(range_frame, text="End ID:", bg="white").grid(row=0, column=2, padx=5)
    end_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=end_var, width=8).grid(row=0, column=3, padx=5)

    # --- OR custom IDs ---
    tk.Label(dialog, text="OR Enter IDs (comma-separated):", bg="white").pack(pady=(15, 2))
    custom_var = tk.StringVar()
    tk.Entry(dialog, textvariable=custom_var, width=30).pack(pady=2)

    def run_export():
        ids = []
        try:
            # Collect range
            if start_var.get() and end_var.get():
                s, e = int(start_var.get()), int(end_var.get())
                ids.extend(range(s, e + 1))

            # Collect custom
            if custom_var.get():
                for part in custom_var.get().split(","):
                    part = part.strip()
                    if part:
                        ids.append(int(part))

            if not ids:
                messagebox.showwarning("Batch Export", "Please enter a range or some IDs.")
                return

            ids = sorted(set(ids))  # remove duplicates

            # ✅ Close selection dialog BEFORE export begins
            dialog.destroy()

            # Now call batch export (this may open file save dialog next)
            root.after(200, lambda: batch_export_with_ui(ids, output_mode))


        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please use numbers only.")

    # --- Buttons ---
    button_frame = tk.Frame(dialog, bg="white")
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Export", command=run_export).grid(row=0, column=0, padx=10)
    tk.Button(button_frame, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=10)
from tkinter import ttk




# --- Preview placeholder (when nothing generated yet) ---

def _show_preview_in_panel(images):
    """
    images: list of (dno, PIL.Image)
    Renders a simple viewer with Prev/Next and Save buttons inside preview_holder.
    """
    from PIL import ImageTk, Image
    _clear_preview_holder()

    # header with controls
    header = tk.Frame(preview_holder, bg="white")
    header.pack(fill="x", pady=(8, 6))

    current_idx = tk.IntVar(value=0)
    page_lbl = tk.Label(header, text="", bg="white", font=("Segoe UI", 10, "bold"))
    page_lbl.pack(side="left", padx=8)

    body = tk.Frame(preview_holder, bg="white")
    body.pack(fill="both", expand=True)

    vbar = tk.Scrollbar(body, orient="vertical")
    hbar = tk.Scrollbar(body, orient="horizontal")
    canvas_prev = tk.Canvas(body, bg="white", highlightthickness=0,
                            yscrollcommand=vbar.set, xscrollcommand=hbar.set)
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
        r = min(avail_w / im.width, avail_h / im.height, 1.0)  # don’t upscale
        new_w, new_h = int(im.width * r), int(im.height * r)
        im_resized = im.resize((new_w, new_h), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(im_resized)
        img_refs.clear(); img_refs.append(tk_img)

        canvas_prev.delete("all")
        canvas_prev.create_image(0, 0, image=tk_img, anchor="nw")
        canvas_prev.config(scrollregion=(0, 0, new_w, new_h))
        page_lbl.config(text=f"S. No {dno}  ({idx+1}/{len(images)})")

    def _nav(delta):
        i = current_idx.get() + delta
        if 0 <= i < len(images):
            current_idx.set(i)
            render(i)
        # Keyboard navigation
    preview_holder.bind_all("<Left>", lambda e: _nav(-1))
    preview_holder.bind_all("<Right>", lambda e: _nav(+1))


    def save_current():
        idx = current_idx.get()
        dno, im = images[idx]
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile=f"digsheet_{dno}.png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            im.save(p, "PNG")
            messagebox.showinfo("Saved", f"Saved {p}")

    def save_all():
        folder = filedialog.askdirectory()
        if not folder: return
        for dno, im in images:
            im.save(os.path.join(folder, f"digsheet_{dno}.png"), "PNG")
        messagebox.showinfo("Saved", f"Exported {len(images)} PNGs to:\n{folder}")
    tk.Button(header, text="Next ⟶", command=lambda: _nav(+1)).pack(side="right", padx=4)
    tk.Button(header, text="⟵ Prev", command=lambda: _nav(-1)).pack(side="right", padx=4)
    
    tk.Button(header, text="💾 Save Current", command=save_current).pack(side="right", padx=8)
    tk.Button(header, text="💾 Save All", command=save_all).pack(side="right", padx=4)

    canvas_prev.bind("<Configure>", lambda e: render(current_idx.get()))
    render(0)

# keep a global handle so Reset can close any progress panel
progress_frame_ref = None





def batch_export_with_ui(defect_ids, output_mode="pdf", output_path=None):
    global batch_cancelled
    batch_cancelled = False

    if not defect_ids:
        messagebox.showwarning("Batch Export", "No defect IDs provided.")
        return

    # Ask save location
    if not output_path:
        if output_mode == "pdf":
            output_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")]
            )
        else:
            output_path = filedialog.askdirectory()
        if not output_path:
            return


    global progress_frame_ref
    progress_frame_ref = tk.Frame(input_frame, bg="white", relief="solid", bd=1)
    progress_frame_ref.pack(side="top", fill="x", pady=12)
    progress_frame = progress_frame_ref  # keep rest of the function unchanged




    tk.Label(progress_frame, text="Batch Export Progress",
             bg="white", fg="deepskyblue", font=("Segoe UI", 11, "bold")).pack(pady=10)

    status_lbl = tk.Label(progress_frame, text="Starting...", bg="white", font=("Segoe UI", 10))
    status_lbl.pack(pady=5)

    def cancel_process():
        global batch_cancelled
        batch_cancelled = True
        status_lbl.config(text="❌ Cancel requested...")

    cancel_btn = tk.Button(progress_frame, text="Cancel", command=cancel_process, bg="red", fg="white")
    cancel_btn.pack(pady=10)

    # --- Square Progress Bar (below buttons) ---
    bar_frame = tk.Frame(progress_frame, bg="white")
    bar_frame.pack(pady=10)

    prog_var = tk.IntVar()
    prog_bar = ttk.Progressbar(
        bar_frame,
        maximum=len(defect_ids),
        variable=prog_var,
        length=120,   # width
        mode="determinate",
        style="Custom.Horizontal.TProgressbar"
    )
    prog_bar.pack()

    root.update()

    images = []
    for idx, dno in enumerate(defect_ids, start=1):
        if batch_cancelled:
            status_lbl.config(text="❌ Cancelled")
            break

        try:
            # Load defect
            defect_entry.delete(0, tk.END)
            defect_entry.insert(0, str(dno))
            on_load_click()
            root.update()
            time.sleep(0.4)

            # Capture
            merged = capture_sections_image(1, 5)
            if merged is None:
                continue

            if output_mode == "png":
                out_file = os.path.join(output_path, f"digsheet_{dno}.png")
                merged.save(out_file, "PNG")
            else:
                temp_path = f"_tmp_{dno}.png"
                merged.save(temp_path)
                images.append(temp_path)

            prog_var.set(idx)
            status_lbl.config(text=f"✅ Saved {idx}/{len(defect_ids)}")
            root.update()

        except Exception as e:
            print(f"Error on defect {dno}: {e}")

    # Finalize
    if not batch_cancelled:
        if output_mode == "pdf" and images:
            with open(output_path, "wb") as f:
                f.write(img2pdf.convert(images))
            for p in images:
                os.remove(p)
        # ✅ Final clean message
        status_lbl.config(text="✔ Completed")
        messagebox.showinfo(
                            "Batch Export Completed",
                            f"Your files have been saved successfully.\n\nLocation:\n{output_path}"
                        )

    else:
        # Clean up temp files if cancelled
        for p in images:
            if os.path.exists(p):
                os.remove(p)

    # Auto close progress panel after 2s
    root.after(2000, progress_frame.destroy)

def parse_multi_input(text):
    ids = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                ids.extend(range(start, end+1))
            except:
                pass
        elif part.isdigit():
            ids.append(int(part))
    return sorted(set(ids))


# Upstream weld info
pipe_canvas.create_text(mid_x, 5, text="Upstream Weld", font=("Arial", 10))

# --- Feature Info Blocks (Flange + Bend) ---
labels = ["Abs. Dist.:", "Latitude:", "Longitude:"]
for i, label in enumerate(labels):
    pipe_canvas.create_text(mid_x - 320, mid_y - 145 + i * 15, text=label, font=("Arial", 9), anchor="w")
    pipe_canvas.create_text(mid_x - 320, mid_y - 30 + i * 15, text=label, font=("Arial", 9), anchor="w")

# --- Horizontal Lines ---
for y in [mid_y - 100, mid_y - 60, mid_y + 20, mid_y + 60]:
    pipe_canvas.create_line(mid_x - 320, y, mid_x + 320, y, width=2)

# --- U/S and D/S Labels ---
pipe_canvas.create_text(mid_x - 310, mid_y - 80, text="U/S", font=("Arial", 9, "bold"), fill="blue")
pipe_canvas.create_text(mid_x + 310, mid_y - 80, text="D/S", font=("Arial", 9, "bold"), fill="blue")

# --- L and R Labels ---
pipe_canvas.create_text(mid_x - 310, mid_y + 40, text="L", font=("Arial", 9, "bold"), fill="deepskyblue")
pipe_canvas.create_text(mid_x + 310, mid_y + 40, text="R", font=("Arial", 9, "bold"), fill="deepskyblue")

# --- Pipe Lengths Block ---
pipe_info = ["Pipe No:", "Pipe Length(m):", "WT(mm):"]
for i, label in enumerate(pipe_info):
    pipe_canvas.create_text(mid_x - 320, mid_y + 75 + i * 15, text=label, font=("Arial", 9), anchor="w")

# --- Flow Text and Arrow ---
pipe_canvas.create_text(mid_x - 315, mid_y + 145, text="FLOW", font=("Arial", 9), fill="deepskyblue", anchor="w")
pipe_canvas.create_line(mid_x - 270, mid_y + 160, mid_x - 320, mid_y + 160, arrow=tk.FIRST, width=1)

# --- Pipe Boxes ---
for i in range(6):
    x1 = mid_x - 240 + i * 80
    x2 = x1 + 80
    pipe_canvas.create_rectangle(x1, mid_y + 120, x2, mid_y + 180, width=1)

root.mainloop()