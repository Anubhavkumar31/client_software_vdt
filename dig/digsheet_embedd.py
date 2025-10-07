













# dig_sheet.py  —  embeddable or standalone Digsheet UI
# ------------------------------------------------------
# Usage (embedded):
#   import dig_sheet
#   dig_sheet.mount_into(parent_frame, project_root=..., pipe_tally_file=...)
#
# Usage (standalone):
#   python dig_sheet.py  <pipe_tally.pkl> <project_root>

import datetime
import io
import math
import os
import pickle
import sys
import tempfile
import time
import traceback
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import pandas as pd
from PIL import Image, ImageGrab, ImageTk
import img2pdf

# ------- Global embedding toggles / handles -------
EMBEDDED = False          # set True when mounted via mount_into()
ROOT_CONTAINER = None     # parent container to render into
root = None               # the toplevel window (winfo_toplevel)

# ------- Section IDs and capture thresholds -------
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

# ------- State / UI globals used across functions -------
batch_cancelled = False
scrollable_active = False
progress_frame_ref = None

# widgets we need to reference elsewhere
BUTTON_PANEL_W = None
button_frame = input_frame = toolbar = None
defect_entry = None
preview_holder = None
container = canvas = scrollable_frame = scrollbar = None
client_desc_frame = main_frame = comment_frame = feature_desc_frame = third_frame = None
pipe_canvas1 = pipe_canvas = None
mid_x = mid_y = None

# data vars
pipe_id_var = tk.StringVar()
length_var = tk.StringVar()
wt_var = tk.StringVar()
latitude_var = tk.StringVar()
longitude_var = tk.StringVar()
altitude_var = tk.StringVar()

client_var = tk.StringVar()
pipeline_name_var = tk.StringVar()
pipeline_section_var = tk.StringVar()

# icons
valve_img = bend_img = flange_img = flowtee_img = magnet_img = None

# feature labels map
feature_labels = {}

# dataframe
df = None


# ---------- Utility / data loading ----------
def load_pipe_tally(pipe_tally_file):
    try:
        with open(pipe_tally_file, "rb") as f:
            pipe_tally = pickle.load(f)
        return pipe_tally
    except Exception as e:
        print(f"Error loading pipe_tally: {e}")
        sys.exit(1)


# ---------- Canvas scroll & helpers ----------
def _yscroll_set(lo, hi):
    """Slim floating scrollbar that only appears when needed."""
    scrollbar.set(lo, hi)
    try:
        lo_f, hi_f = float(lo), float(hi)
    except Exception:
        lo_f, hi_f = 0.0, 1.0

    if hi_f - lo_f >= 0.999:
        scrollbar.place_forget()
    else:
        scrollbar.place(in_=canvas, relx=1.0, x=-8, rely=0.5, anchor="e", relheight=0.98)


def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))


def _on_mousewheel(event):
    if event.delta:  # Windows / MacOS
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    elif event.num == 4:  # Linux up
        canvas.yview_scroll(-3, "units")
    elif event.num == 5:  # Linux down
        canvas.yview_scroll(3, "units")


# ---------- File I/O: Save / Print / Capture ----------
def upscale_image(img, target_dpi=600, base_dpi=96, scale_limit=2.0):
    scale = target_dpi / base_dpi
    if scale > scale_limit:
        scale = scale_limit
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS), target_dpi


def capture_sections_image(section_start=1, section_end=5):
    images = []
    for section_id in range(section_start, section_end + 1):
        if section_id not in SECTION_MAP:
            continue
        canvas.yview_moveto(0.0 if section_id in [1, 2, 3, 4] else 1.0)
        root.update(); time.sleep(0.4)

        coords = get_section_coords()
        name = SECTION_MAP[section_id]
        if name not in coords:
            continue

        x0, y0, x1, y1 = coords[name]
        dx0, dy0, dx1, dy1 = SECTION_THRESHOLDS.get(name, (0, 0, 0, 0))
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

    merged, dpi = upscale_image(merged, target_dpi=300, base_dpi=96)
    buf = io.BytesIO()
    merged.save(buf, format="PNG", dpi=(dpi, dpi))
    buf.seek(0)

    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(buf.getvalue()))

    messagebox.showinfo("Saved!", f"High-quality PDF created:\n{pdf_path}")


def open_save_dialog():
    dlg = tk.Toplevel(root)
    dlg.title("Save")
    dlg.geometry("300x160+520+260")
    dlg.configure(bg="white")
    dlg.grab_set()

    tk.Label(dlg, text="Save as:", bg="white",
             font=("Segoe UI", 11, "bold")).pack(pady=(12, 6))

    mode_var = tk.StringVar(value="png")
    opts = tk.Frame(dlg, bg="white"); opts.pack(pady=4)
    tk.Radiobutton(opts, text="PNG (image)", variable=mode_var, value="png", bg="white").grid(row=0, column=0, padx=10)
    tk.Radiobutton(opts, text="PDF (single page)", variable=mode_var, value="pdf", bg="white").grid(row=0, column=1, padx=10)

    def do_save():
        dlg.destroy()
        if mode_var.get() == "png":
            capture_sections(1, 5)
        else:
            save_all_sections_as_pdf()

    btns = tk.Frame(dlg, bg="white"); btns.pack(pady=14)
    tk.Button(btns, text="Save", command=do_save).grid(row=0, column=0, padx=10)
    tk.Button(btns, text="Cancel", command=dlg.destroy).grid(row=0, column=1, padx=10)


# Windows-only print dialog (kept from your code)
def print_all_sections_dialog():
    try:
        import win32api, win32print
    except Exception:
        messagebox.showerror("Print", "Printing requires pywin32 on Windows.")
        return

    merged = capture_sections_image(1, 5)
    if merged is None:
        messagebox.showerror("Error", "No sections captured")
        return

    temp_img = tempfile.mktemp(suffix=".png")
    merged.save(temp_img, "PNG")

    def get_printers():
        return [p[2] for p in win32print.EnumPrinters(
            win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)]

    def send_to_printer(printer_name, file_path):
        try:
            win32api.ShellExecute(0, "print", file_path, f'"{printer_name}"', ".", 0)
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

    dialog = tk.Toplevel(root)
    dialog.title("Print Report")
    dialog.geometry("400x200")
    dialog.configure(bg="white")
    dialog.grab_set()

    tk.Label(dialog, text="Select a Printer",
             font=("Segoe UI", 12, "bold"), bg="white").pack(pady=(15, 10))
    printers = get_printers()
    printer_combo = ttk.Combobox(dialog, values=printers, state="readonly", width=40)
    if printers:
        printer_combo.current(0)
    printer_combo.pack(pady=10)
    btns = tk.Frame(dialog, bg="white"); btns.pack(pady=20)
    ttk.Button(btns, text="Print", command=print_selected).grid(row=0, column=0, padx=10)
    ttk.Button(btns, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=10)
    dialog.mainloop()


def capture_sections(section_start=1, section_end=5):
    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png")],
        initialfile=""
    )
    if not filepath:
        return

    images = []
    for section_id in range(section_start, section_end + 1):
        if section_id not in SECTION_MAP:
            continue
        if section_id in [1, 2, 3, 4]:
            canvas.yview_moveto(0.0)
        else:
            canvas.yview_moveto(1.0)
        root.update(); time.sleep(0.4)

        coords = get_section_coords()
        name = SECTION_MAP[section_id]
        if name not in coords:
            continue

        x0, y0, x1, y1 = coords[name]
        dx0, dy0, dx1, dy1 = SECTION_THRESHOLDS.get(name, (0, 0, 0, 0))
        bbox = (x0 + dx0, y0 + dy0, x1 + dx1, y1 + dy1)
        img = ImageGrab.grab(bbox=bbox)
        images.append(img)

    if not images:
        messagebox.showerror("Error", "No sections were captured.")
        return

    max_w = max(im.width for im in images)
    total_h = sum(im.height for im in images)
    merged = Image.new("RGB", (max_w, total_h), "white")
    y_offset = 0
    for im in images:
        if im.width != max_w:
            im = im.resize((max_w, im.height))
        merged.paste(im, (0, y_offset))
        y_offset += im.height

    merged.save(filepath)
    messagebox.showinfo("Saved!", f"All sections saved successfully:\n{filepath}")


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


# ---------- Domain logic / drawing ----------
def hms_to_angle(hms):
    if isinstance(hms, str):
        parts = [int(p) for p in hms.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, s = parts[:3]
    else:
        h, m, s = hms.hour, hms.minute, hms.second
    return (h % 12) * 30 + m * 0.5 + s * (0.5 / 60)


def draw_pipe(pipe_canvas1, pipe_length, upstream, clock):
    pipe_canvas1.delete("all")
    width, height = 320, 120
    x0, y0 = 40, 30
    x1, y1 = x0 + width, y0 + height
    mid_x_local, mid_y_local = (x0 + x1) // 2, (y0 + y1) // 2
    radius = height // 2 - 10

    # shape
    pipe_canvas1.create_oval(x0, y0, x0 + 40, y1, outline="black", width=2)
    pipe_canvas1.create_oval(x1 - 40, y0, x1, y1, outline="black", width=2)
    pipe_canvas1.create_line(x0 + 20, y0, x1 - 20, y0, fill="black", width=2)
    pipe_canvas1.create_line(x0 + 20, y1, x1 - 20, y1, fill="black", width=2)
    pipe_canvas1.create_line(x0, mid_y_local - 5, x1, mid_y_local - 5, fill="black", dash=(3, 3))

    pipe_canvas1.create_text(x0 - 20, y0 + 10, text="12", anchor="w", font=("Arial", 10))
    pipe_canvas1.create_text(x0 + 25, mid_y_local + 5, text="3", anchor="w", font=("Arial", 10))
    pipe_canvas1.create_text(x0 - 17, y1 - 5, text="6", anchor="w", font=("Arial", 10))
    pipe_canvas1.create_text(x0 - 10, mid_y_local + 5, text="9", anchor="e", font=("Arial", 10))

    try:
        upstream_f = float(upstream) if upstream else 0.0
        pipe_length_f = float(pipe_length) if pipe_length else 0.0
        remaining = round(max(pipe_length_f - upstream_f, 0.0), 2)
    except Exception:
        upstream_f = 0.0
        remaining = 0.0

    try:
        arrow_y = y0 - 15
        scale_factor = 0.85
        arrow_length_total = (x1 - x0) * scale_factor
        offset = ((x1 - x0) - arrow_length_total) / 2
        arrow_start_x = x0 + offset
        arrow_end_x = x1 - offset

        arrow1_length = (upstream_f / pipe_length_f) * arrow_length_total if pipe_length_f > 0 else arrow_length_total / 2
        arrow2_length = arrow_length_total - arrow1_length

        # Upstream
        a1s = arrow_start_x
        a1e = a1s + arrow1_length
        pipe_canvas1.create_line(a1s, arrow_y, a1e, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_line(a1e, arrow_y, a1s, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_text((a1s + a1e) / 2, arrow_y - 10, text=f"{round(upstream_f, 2)} m", font=("Arial", 10))

        # Remaining
        a2s = a1e
        a2e = arrow_end_x
        pipe_canvas1.create_line(a2s, arrow_y, a2e, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_line(a2e, arrow_y, a2s, arrow_y, arrow=tk.LAST)
        pipe_canvas1.create_text((a2s + a2e) / 2, arrow_y - 10, text=f"{remaining} m", font=("Arial", 10))

        # defect box
        angle_deg = hms_to_angle(clock)
        angle_rad = math.radians(angle_deg)
        center_y = mid_y_local
        defect_x = arrow_start_x + (upstream_f / pipe_length_f) * arrow_length_total if pipe_length_f > 0 else arrow_start_x
        adjusted_radius = radius * 0.80
        defect_y = center_y - int(adjusted_radius * math.cos(angle_rad))
        if 0 <= angle_deg <= 180:
            pipe_canvas1.create_rectangle(defect_x - 4, defect_y - 4, defect_x + 4, defect_y + 4, fill="orange", outline="black")
        else:
            pipe_canvas1.create_rectangle(defect_x - 4, defect_y - 4, defect_x + 4, defect_y + 4, outline="orange", width=2)

        pipe_canvas1.create_line(defect_x - 5, defect_y, defect_x - 5, y0, arrow=tk.LAST, fill="black", width=1.5)
    except Exception as e:
        print("Drawing error:", e)


def fetch_data():
    global df
    try:
        s_no = int(defect_entry.get())
        row = df[df.iloc[:, 0] == s_no]
        if row.empty:
            messagebox.showerror("Error", "Defect number not found!")
            return
        row = row.iloc[0]
        pipe_id_var.set(str(row.iloc[3]))
        length_var.set(str(row.iloc[4]))
        wt_var.set(str(row.iloc[11]))

        lat_col = next((c for c in df.columns if c.strip().lower() == "latitude"), None)
        lon_col = next((c for c in df.columns if c.strip().lower() == "longitude"), None)
        alt_col = next((c for c in df.columns if c.strip().lower() == "altitude"), None)

        latitude_var .set(str(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else "")
        longitude_var.set(str(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else "")
        altitude_var .set(str(row[alt_col]) if alt_col and pd.notna(row[alt_col]) else "")

        upstream = row.iloc[2]
        clock_raw = row.iloc[8]
        draw_pipe(pipe_canvas1, row.iloc[4], upstream, clock_raw)

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
            if col_index is not None:
                value = row.iloc[col_index] if col_index < len(row) else ""
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

        # Remaining wall thickness
        try:
            wt = float(row.iloc[11])
            max_depth = float(row.iloc[12])
            remaining_wt = round(wt - (wt * max_depth / 100), 1)
        except:
            remaining_wt = ""
        feature_labels["Remaining wall thickness (mm)"].config(text=str(remaining_wt))

        # Lat/Long copy into feature_labels too (if desired)
        feature_labels["Latitude"].config(text=str(latitude_var.get()))
        feature_labels["Longitude"].config(text=str(longitude_var.get()))

    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid S.no")


def get_dynamic_weld_and_feature_data():
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

        defect_idx = df[df.iloc[:, 0] == s_no].index[0]
        lat_col = next((c for c in df.columns if c.strip().lower() == "latitude"), None)
        lon_col = next((c for c in df.columns if c.strip().lower() == "longitude"), None)

        features_upstream, features_downstream = [], []
        bends_upstream, bends_downstream = [], []

        # upstream features
        for i in range(defect_idx - 1, -1, -1):
            r = df.loc[i]
            name = str(r.iloc[5]).strip().lower()
            if any(f in name for f in feature_keywords):
                features_upstream.append({
                    "name": str(r.iloc[5]),
                    "distance": round(float(r.iloc[1]), 2),
                    "lat": str(r[lat_col]) if lat_col and pd.notna(r[lat_col]) else "",
                    "long": str(r[lon_col]) if lon_col and pd.notna(r[lon_col]) else ""
                })
                if len(features_upstream) == 2:
                    break

        # downstream features
        for i in range(defect_idx + 1, len(df)):
            r = df.loc[i]
            name = str(r.iloc[5]).strip().lower()
            if any(f in name for f in feature_keywords):
                features_downstream.append({
                    "name": str(r.iloc[5]),
                    "distance": round(float(r.iloc[1]), 2),
                    "lat": str(r[lat_col]) if lat_col and pd.notna(r[lat_col]) else "",
                    "long": str(r[lon_col]) if lon_col and pd.notna(r[lon_col]) else ""
                })
                if len(features_downstream) == 2:
                    break

        # bends up
        for i in range(defect_idx - 1, -1, -1):
            r = df.loc[i]
            name = str(r.iloc[5]).strip().lower()
            if "bend" in name:
                bends_upstream.append({
                    "name": str(r.iloc[5]),
                    "distance": round(float(r.iloc[1]), 2),
                    "lat": str(r[lat_col]) if lat_col and pd.notna(r[lat_col]) else "",
                    "long": str(r[lon_col]) if lon_col and pd.notna(r[lon_col]) else ""
                })
                if len(bends_upstream) == 3:
                    break

        # bends down
        for i in range(defect_idx + 1, len(df)):
            r = df.loc[i]
            name = str(r.iloc[5]).strip().lower()
            if "bend" in name:
                bends_downstream.append({
                    "name": str(r.iloc[5]),
                    "distance": round(float(r.iloc[1]), 2),
                    "lat": str(r[lat_col]) if lat_col and pd.notna(r[lat_col]) else "",
                    "long": str(r[lon_col]) if lon_col and pd.notna(r[lon_col]) else ""
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
    except Exception:
        return {
            "upstream_weld": "",
            "features_upstream": "",
            "features_downstream": "",
            "bends_upstream": "",
            "bends_downstream": ""
        }


# ---------- Top-level actions ----------
def on_load_click():
    global df
    try:
        # Load from argv if provided (standalone flow)
        # argv: [script, pipe_tally.pkl, project_root]
        if len(sys.argv) > 1 and not EMBEDDED:
            pipe_tally_file = sys.argv[1]
            project_root = sys.argv[2]
            csv_path  = os.path.join(project_root, "constants.csv")
            xlsx_path = os.path.join(project_root, "constants.xlsx")
            constants_file = csv_path if os.path.exists(csv_path) else xlsx_path
            print(f"constants_file path: {constants_file}")

            pipe_tally = load_pipe_tally(pipe_tally_file)
            df = pipe_tally

            # constants → client fields
            const_df = pd.read_excel(constants_file, dtype=str)
            import re
            def _norm(s: str) -> str:
                s = re.sub(r'[^A-Za-z0-9]+', ' ', str(s))
                return '_'.join(s.strip().upper().split())
            colmap = {_norm(c): c for c in const_df.columns}
            def _first_val(*aliases):
                for a in aliases:
                    key = _norm(a)
                    if key in colmap:
                        ser = const_df[colmap[key]].dropna().astype(str).str.strip()
                        if not ser.empty:
                            return ser.iloc[0]
                return ""
            print("[constants] columns:", list(const_df.columns))
            client_var.set(_first_val("CLIENT_NAME_DESCRIPTION"))
            pipeline_name_var.set(_first_val("PIPELINE_NAME_DESCRIPTION"))
            pipeline_section_var.set(_first_val("PIPELINE_SECTION_DESCRIPTION"))
        # else: in embedded mode, assume df is filled by your app or earlier call
    except Exception as e:
        print(f"Error in on_load_click: {e}")

    if 'df' not in globals() or df is None:
        messagebox.showwarning("Missing Excel File", "Please load an Excel file before loading defect data.")
        return

    fetch_data()
    # Clear dynamic overlays on big pipe canvas
    for tag in ("upstream_text", "flange_text", "us_arrow", "ds_arrow", "bend_text", "pipe_icon"):
        try: pipe_canvas.delete(tag)
        except: pass

    # dynamic top/bottom visuals (kept from your code, shortened)
    weld_info = get_dynamic_weld_and_feature_data()
    if not weld_info:
        return

    upstream_weld_dist = weld_info["upstream_weld"]
    features_upstream  = weld_info["features_upstream"]
    features_downstream = weld_info["features_downstream"]
    bends_upstream     = weld_info.get("bends_upstream", [])
    bends_downstream   = weld_info.get("bends_downstream", [])

    pipe_canvas.create_text(mid_x, 20, text=f"{upstream_weld_dist:.2f}(m)", font=("Arial", 10), tags="upstream_text")

    # features 2 up + 2 down (condensed drawing)
    slots = [
        {"x": mid_x - 190, "arrow_x": mid_x - 200, "text_x": mid_x - 160, "src": features_upstream[::-1], "i": 1},
        {"x": mid_x - 90,  "arrow_x": mid_x - 100, "text_x": mid_x - 60,  "src": features_upstream[::-1], "i": 0},
        {"x": mid_x + 110, "arrow_x": mid_x + 120, "text_x": mid_x + 80,  "src": features_downstream,      "i": 0},
        {"x": mid_x + 210, "arrow_x": mid_x + 220, "text_x": mid_x + 180, "src": features_downstream,      "i": 1},
    ]
    for sl in slots:
        try:
            f = sl["src"][sl["i"]]
            name = f.get("name", ""); dist_val = f.get("distance", "")
            lat = f.get("lat", ""); lon = f.get("long", "")
            dist_txt = f"{dist_val}(m)" if pd.notna(dist_val) else ""
            pipe_canvas.create_text(sl["x"], mid_y - 160, text=name, font=("Arial", 10), tags="flange_text")
            pipe_canvas.create_text(sl["x"], mid_y - 145, text=dist_txt, font=("Arial", 9), tags="flange_text")
            pipe_canvas.create_text(sl["x"], mid_y - 130, text=lat, font=("Arial", 9), tags="flange_text")
            pipe_canvas.create_text(sl["x"], mid_y - 115, text=lon, font=("Arial", 9), tags="flange_text")
            arrow_val = round(abs(float(upstream_weld_dist) - float(dist_val)), 2)
            pipe_canvas.create_line(sl["arrow_x"], mid_y - 95, sl["arrow_x"], mid_y - 65, arrow=tk.FIRST,
                                    fill="deepskyblue", width=2, tags="us_arrow")
            pipe_canvas.create_text(sl["text_x"], mid_y - 80, text=f"{arrow_val}(m)", font=("Arial", 9), tags="us_arrow")
        except Exception:
            pass

    # bends 3 up + 3 down
    def tri(x, y):
        pipe_canvas.create_polygon(x - 42.5, y - 20, x - 50, y + 18, x - 35, y + 18,
                                   fill="deepskyblue", outline="gray", width=1, tags="us_arrow")
    bslots = [
        {"src": bends_upstream[::-1], "i": 2, "xn": mid_x - 230, "xd": mid_x - 230, "xlat": mid_x - 235, "xlon": mid_x - 235, "tx": mid_x - 200, "ax": mid_x - 215},
        {"src": bends_upstream[::-1], "i": 1, "xn": mid_x - 140, "xd": mid_x - 140, "xlat": mid_x - 135, "xlon": mid_x - 135, "tx": mid_x - 110, "ax": mid_x - 125},
        {"src": bends_upstream[::-1], "i": 0, "xn": mid_x - 50,  "xd": mid_x - 50,  "xlat": mid_x - 35,  "xlon": mid_x - 35,  "tx": mid_x - 20,  "ax": mid_x - 35},
        {"src": bends_downstream,     "i": 0, "xn": mid_x + 55,  "xd": mid_x + 55,  "xlat": mid_x + 50,  "xlon": mid_x + 50,  "tx": mid_x + 110, "ax": mid_x + 30},
        {"src": bends_downstream,     "i": 1, "xn": mid_x + 155, "xd": mid_x + 155, "xlat": mid_x + 150, "xlon": mid_x + 150, "tx": mid_x + 210, "ax": mid_x + 130},
        {"src": bends_downstream,     "i": 2, "xn": mid_x + 255, "xd": mid_x + 255, "xlat": mid_x + 250, "xlon": mid_x + 250, "tx": mid_x + 310, "ax": mid_x + 230},
    ]
    for sl in bslots:
        try:
            b = sl["src"][sl["i"]]
            name = b.get("name", ""); dist_val = b.get("distance", "")
            lat = b.get("lat", ""); lon = b.get("long", "")
            pipe_canvas.create_text(sl["xn"], mid_y - 45, text=name, font=("Arial", 10), tags="bend_text")
            pipe_canvas.create_text(sl["xd"], mid_y - 30, text=f"{dist_val}(m)", font=("Arial", 9), tags="bend_text")
            pipe_canvas.create_text(sl["xlat"], mid_y - 15, text=lat, font=("Arial", 9), tags="bend_text")
            pipe_canvas.create_text(sl["xlon"], mid_y, text=lon, font=("Arial", 9), tags="bend_text")
            tri(sl["tx"], mid_y + 40)
            arrow_val = round(abs(float(upstream_weld_dist) - float(dist_val)), 2)
            pipe_canvas.create_text(sl["ax"], mid_y + 35, text=f"{arrow_val}", font=("Arial", 9), tags="us_arrow")
            pipe_canvas.create_text(sl["ax"], mid_y + 45, text="(m)", font=("Arial", 9), tags="us_arrow")
        except Exception:
            pass

    # bottom 6 pipe boxes + icons + defect rect kept from your code (unchanged layout)
    try:
        s_no = int(defect_entry.get())
        defect_row = df[df.iloc[:, 0] == s_no]
        if defect_row.empty:
            messagebox.showwarning("Warning", f"No defect found for S.No {s_no}")
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
            pipe_wt  = f"{round(float(row[11]), 1)}" if pd.notna(row[11]) else ""
            pipe_data_list.append((str(pipe_no), pipe_len, pipe_wt))
        else:
            pipe_data_list.append(("", "", ""))

    pipe_positions = [-210, -140, -60, 20, 110, 180]
    for i, (pnum, plen, pwt) in enumerate(pipe_data_list):
        px = pipe_positions[i]
        pipe_canvas.create_text(mid_x + px, mid_y + 75, text=pnum, font=("Arial", 9), anchor="w", tags="us_arrow")
        pipe_canvas.create_text(mid_x + px, mid_y + 90, text=plen, font=("Arial", 9), anchor="w", tags="us_arrow")
        pipe_canvas.create_text(mid_x + px, mid_y + 105, text=pwt, font=("Arial", 9), anchor="w", tags="us_arrow")

    # defect marker in 4th box
    try:
        defect_row = defect_row.iloc[0]
        upstream_dist = f"{round(float(defect_row.iloc[2]), 2)}" if pd.notna(defect_row.iloc[2]) else ""
        clock_pos = f"{(defect_row.iloc[8])}" if pd.notna(defect_row.iloc[8]) else ""
        pipe_len = f"{round((defect_row.iloc[4]), 3)}" if pd.notna(defect_row.iloc[4]) else ""

        if pipe_len and upstream_dist:
            pipe_length = float(pipe_len)
            upstream = float(upstream_dist)
            clock_angle = hms_to_angle(clock_pos)

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

    # center points of the 6 pipe boxes
    pipe_box_centers = [
        (mid_x - 200, mid_y + 155),
        (mid_x - 120, mid_y + 155),
        (mid_x - 40,  mid_y + 155),
        (mid_x + 40,  mid_y + 155),
        (mid_x + 120, mid_y + 155),
        (mid_x + 200, mid_y + 155)
    ]

    # icons per pipe
    for i, pipe_num in enumerate(target_pipe_numbers):
        matching_rows = df[df.iloc[:, 3] == pipe_num]
        if not matching_rows.empty:
            found = []
            for _, rowx in matching_rows.iterrows():
                feature_text = str(rowx.iloc[5]).lower()
                if "valve" in feature_text and "valve" not in found:
                    found.append("valve")
                if "flow" in feature_text or "tee" in feature_text:
                    if "flowtee" not in found:
                        found.append("flowtee")
                if "flange" in feature_text and "flange" not in found:
                    found.append("flange")
                if "bend" in feature_text and "bend" not in found:
                    found.append("bend")
                if "magnet" in feature_text and "magnet" not in found:
                    found.append("magnet")

            cx, cy = pipe_box_centers[i]
            spacing = 22
            for j, feat in enumerate(found):
                oy = cy - ((len(found) - 1) * spacing // 2) + (j * spacing)
                if feat == "valve"   and valve_img:  pipe_canvas.create_image(cx, oy, image=valve_img,  tags="pipe_icon")
                if feat == "flowtee" and flowtee_img:pipe_canvas.create_image(cx, oy, image=flowtee_img,tags="pipe_icon")
                if feat == "flange"  and flange_img: pipe_canvas.create_image(cx, oy, image=flange_img, tags="pipe_icon")
                if feat == "bend"    and bend_img:   pipe_canvas.create_image(cx, oy, image=bend_img,   tags="pipe_icon")
                if feat == "magnet"  and magnet_img: pipe_canvas.create_image(cx, oy, image=magnet_img, tags="pipe_icon")


# ---------- Batch export / preview ----------
def _start_panel_progress(total, title="Generating previews"):
    _clear_preview_holder()
    prog_frame = tk.Frame(preview_holder, bg="white", highlightbackground="#e8e8e8", highlightthickness=1)
    prog_frame.pack(side="top", fill="x", padx=8, pady=8)

    tk.Label(prog_frame, text=title, bg="white", fg="deepskyblue",
             font=("Segoe UI", 11, "bold")).pack(pady=(10, 6))
    status_lbl = tk.Label(prog_frame, text=f"0 / {total}", bg="white", font=("Segoe UI", 10))
    status_lbl.pack(pady=(0, 8))

    bar_wrap = tk.Frame(prog_frame, bg="white"); bar_wrap.pack(pady=(0, 12))
    prog_var = tk.IntVar(value=0)
    prog_bar = ttk.Progressbar(
        bar_wrap, maximum=total, variable=prog_var, length=320, mode="determinate",
        style="Custom.Horizontal.TProgressbar"
    )
    prog_bar.pack()

    def _update(done):
        prog_var.set(done); status_lbl.config(text=f"{done} / {total}"); prog_frame.update_idletasks()
    def _finish():
        prog_frame.destroy(); preview_holder.update_idletasks()
    return _update, _finish


def _show_preview_placeholder(msg="No previews yet.\nUse MultiPreview to generate."):
    _clear_preview_holder()
    lbl = tk.Label(preview_holder, text=msg, bg="white", fg="gray50",
                   font=("Segoe UI", 11, "bold"), justify="center")
    lbl.place(relx=0.5, rely=0.5, anchor="center")


def _clear_preview_holder():
    try:
        preview_holder.unbind_all("<Left>")
        preview_holder.unbind_all("<Right>")
    except Exception:
        pass
    for w in preview_holder.winfo_children():
        w.destroy()


def _show_preview_in_panel(images):
    from PIL import ImageTk
    _clear_preview_holder()

    header = tk.Frame(preview_holder, bg="white"); header.pack(fill="x", pady=(8, 6))
    current_idx = tk.IntVar(value=0)
    page_lbl = tk.Label(header, text="", bg="white", font=("Segoe UI", 10, "bold"))
    page_lbl.pack(side="left", padx=8)

    body = tk.Frame(preview_holder, bg="white"); body.pack(fill="both", expand=True)
    vbar = tk.Scrollbar(body, orient="vertical")
    hbar = tk.Scrollbar(body, orient="horizontal")
    canvas_prev = tk.Canvas(body, bg="white", highlightthickness=0,
                            yscrollcommand=vbar.set, xscrollcommand=hbar.set)
    vbar.config(command=canvas_prev.yview); hbar.config(command=canvas_prev.xview)
    vbar.pack(side="right", fill="y"); hbar.pack(side="bottom", fill="x")
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
        img_refs.clear(); img_refs.append(tk_img)
        canvas_prev.delete("all")
        canvas_prev.create_image(0, 0, image=tk_img, anchor="nw")
        canvas_prev.config(scrollregion=(0, 0, new_w, new_h))
        page_lbl.config(text=f"S. No {dno}  ({idx+1}/{len(images)})")

    def _nav(delta):
        i = current_idx.get() + delta
        if 0 <= i < len(images):
            current_idx.set(i); render(i)

    preview_holder.bind_all("<Left>",  lambda e: _nav(-1))
    preview_holder.bind_all("<Right>", lambda e: _nav(+1))

    def save_current():
        idx = current_idx.get()
        dno, im = images[idx]
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile=f"digsheet_{dno}.png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            im.save(p, "PNG"); messagebox.showinfo("Saved", f"Saved {p}")

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


def batch_preview(defect_ids, mode="png", embed=False):
    if not defect_ids:
        messagebox.showwarning("Preview", "No defect IDs provided.")
        return

    update_prog, finish_prog = _start_panel_progress(len(defect_ids), title="Generating previews")
    images = []
    done = 0
    for dno in defect_ids:
        try:
            defect_entry.delete(0, tk.END)
            defect_entry.insert(0, str(dno))
            on_load_click()
            root.update(); time.sleep(0.3)
            merged = capture_sections_image(1, 5)
            if merged:
                images.append((dno, merged))
        except Exception as e:
            print(f"[Preview error] Defect {dno}: {e}")
        finally:
            done += 1; update_prog(done)

    finish_prog()

    if not images:
        _show_preview_placeholder("No previews generated.\nCheck your IDs and try again.")
        messagebox.showerror("Preview", "No images generated.")
        return

    if str(mode).lower() == "pdf":
        tmp_paths = []
        try:
            for dno, im in images:
                tmp_path = os.path.join(tempfile.gettempdir(), f"_preview_{dno}.png")
                im.save(tmp_path); tmp_paths.append(tmp_path)
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

    if str(mode).lower() == "png" and embed:
        reset_left_panel()
        _show_preview_in_panel(images)
        return

    # window viewer fallback (kept but omitted here for brevity – panel preview covers most needs)
    _show_preview_in_panel(images)


def batch_export_with_ui(defect_ids, output_mode="pdf", output_path=None):
    global batch_cancelled, progress_frame_ref
    batch_cancelled = False

    if not defect_ids:
        messagebox.showwarning("Batch Export", "No defect IDs provided.")
        return

    if not output_path:
        if output_mode == "pdf":
            output_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                       filetypes=[("PDF files", "*.pdf")])
        else:
            output_path = filedialog.askdirectory()
        if not output_path:
            return

    progress_frame_ref = tk.Frame(input_frame, bg="white", relief="solid", bd=1)
    progress_frame_ref.pack(side="top", fill="x", pady=12)
    progress_frame = progress_frame_ref

    tk.Label(progress_frame, text="Batch Export Progress",
             bg="white", fg="deepskyblue", font=("Segoe UI", 11, "bold")).pack(pady=10)
    status_lbl = tk.Label(progress_frame, text="Starting...", bg="white", font=("Segoe UI", 10))
    status_lbl.pack(pady=5)

    def cancel_process():
        global batch_cancelled
        batch_cancelled = True
        status_lbl.config(text="❌ Cancel requested...")

    tk.Button(progress_frame, text="Cancel", command=cancel_process, bg="red", fg="white").pack(pady=10)
    bar_frame = tk.Frame(progress_frame, bg="white"); bar_frame.pack(pady=10)

    prog_var = tk.IntVar()
    prog_bar = ttk.Progressbar(bar_frame, maximum=len(defect_ids), variable=prog_var,
                               length=120, mode="determinate", style="Custom.Horizontal.TProgressbar")
    prog_bar.pack()
    root.update()

    images = []
    for idx, dno in enumerate(defect_ids, start=1):
        if batch_cancelled:
            status_lbl.config(text="❌ Cancelled")
            break
        try:
            defect_entry.delete(0, tk.END)
            defect_entry.insert(0, str(dno))
            on_load_click()
            root.update(); time.sleep(0.4)
            merged = capture_sections_image(1, 5)
            if merged is None:
                continue

            if output_mode == "png":
                out_file = os.path.join(output_path, f"digsheet_{dno}.png")
                merged.save(out_file, "PNG")
            else:
                temp_path = os.path.join(tempfile.gettempdir(), f"_tmp_{dno}.png")
                merged.save(temp_path); images.append(temp_path)

            prog_var.set(idx)
            status_lbl.config(text=f"✅ Saved {idx}/{len(defect_ids)}")
            root.update()
        except Exception as e:
            print(f"Error on defect {dno}: {e}")

    if not batch_cancelled:
        if output_mode == "pdf" and images:
            with open(output_path, "wb") as f:
                f.write(img2pdf.convert(images))
            for p in images:
                try: os.remove(p)
                except: pass
        status_lbl.config(text="✔ Completed")
        messagebox.showinfo("Batch Export Completed",
                            f"Your files have been saved successfully.\n\nLocation:\n{output_path}")
    else:
        for p in images:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

    root.after(2000, progress_frame.destroy)


def open_batch_dialog_new():
    dialog = tk.Toplevel(root)
    dialog.title("Batch Export")
    dialog.geometry("360x280+500+200")
    dialog.configure(bg="white")
    dialog.grab_set()

    tk.Label(dialog, text="Select defects to export",
             bg="white", font=("Segoe UI", 11, "bold")).pack(pady=10)

    range_frame = tk.Frame(dialog, bg="white"); range_frame.pack(pady=5)
    tk.Label(range_frame, text="Start ID:", bg="white").grid(row=0, column=0, padx=5)
    start_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=start_var, width=8).grid(row=0, column=1, padx=5)
    tk.Label(range_frame, text="End ID:", bg="white").grid(row=0, column=2, padx=5)
    end_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=end_var, width=8).grid(row=0, column=3, padx=5)

    tk.Label(dialog, text="OR Enter IDs (comma-separated):", bg="white").pack(pady=(15, 2))
    custom_var = tk.StringVar()
    tk.Entry(dialog, textvariable=custom_var, width=32).pack(pady=2)

    mode_var = tk.StringVar(value="pdf")
    mode_frame = tk.Frame(dialog, bg="white"); mode_frame.pack(pady=12)
    tk.Label(mode_frame, text="Export as:", bg="white").grid(row=0, column=0, padx=(0,8))
    tk.Radiobutton(mode_frame, text="PDF (one multi-page file)", variable=mode_var, value="pdf", bg="white").grid(row=0, column=1, padx=6)
    tk.Radiobutton(mode_frame, text="PNG (one file per defect)", variable=mode_var, value="png", bg="white").grid(row=0, column=2, padx=6)

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
            dialog.destroy()
            root.after(200, lambda: batch_export_with_ui(ids, mode_var.get()))
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please use numbers only.")

    btns = tk.Frame(dialog, bg="white"); btns.pack(pady=16)
    tk.Button(btns, text="Export", command=run_export).grid(row=0, column=0, padx=10)
    tk.Button(btns, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=10)


def open_preview_dialog():
    dialog = tk.Toplevel(root)
    dialog.title("Multi Preview")
    dialog.geometry("340x280+500+200")
    dialog.configure(bg="white")
    dialog.grab_set()

    tk.Label(dialog, text="Select defects to preview",
             bg="white", font=("Segoe UI", 11, "bold")).pack(pady=10)

    range_frame = tk.Frame(dialog, bg="white"); range_frame.pack(pady=5)
    tk.Label(range_frame, text="Start ID:", bg="white").grid(row=0, column=0, padx=5)
    start_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=start_var, width=8).grid(row=0, column=1, padx=5)
    tk.Label(range_frame, text="End ID:", bg="white").grid(row=0, column=2, padx=5)
    end_var = tk.StringVar()
    tk.Entry(range_frame, textvariable=end_var, width=8).grid(row=0, column=3, padx=5)

    tk.Label(dialog, text="OR Enter IDs (comma-separated):", bg="white").pack(pady=(15, 2))
    custom_var = tk.StringVar()
    tk.Entry(dialog, textvariable=custom_var, width=30).pack(pady=2)

    mode_var = tk.StringVar(value="png")
    mode_frame = tk.Frame(dialog, bg="white"); mode_frame.pack(pady=10)
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
            root.after(200, lambda: batch_preview(ids, mode_var.get(), embed=(mode_var.get().lower() == "png")))
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please use numbers only.")

    tk.Button(dialog, text="Preview", command=run_preview).pack(pady=15)
    tk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)


def reset_ui():
    global batch_cancelled, progress_frame_ref
    batch_cancelled = False
    try: defect_entry.delete(0, tk.END)
    except: pass

    for var in (pipe_id_var, length_var, wt_var, latitude_var, longitude_var, altitude_var,
                client_var, pipeline_name_var, pipeline_section_var):
        try: var.set("")
        except: pass

    for lbl in feature_labels.values():
        try: lbl.config(text="")
        except: pass

    try: comment_placeholder.config(text="")
    except: pass

    try: pipe_canvas1.delete("all")
    except: pass

    for tag in ("upstream_text", "flange_text", "us_arrow", "ds_arrow", "bend_text", "pipe_icon"):
        try: pipe_canvas.delete(tag)
        except: pass

    try: _clear_preview_holder()
    except: pass
    try:
        if progress_frame_ref and progress_frame_ref.winfo_exists():
            progress_frame_ref.destroy()
        progress_frame_ref = None
    except: pass

    try: canvas.yview_moveto(0.0)
    except: pass


def reset_left_panel():
    for var in (pipe_id_var, length_var, wt_var, latitude_var, longitude_var, altitude_var,
                client_var, pipeline_name_var, pipeline_section_var):
        try: var.set("")
        except: pass
    for lbl in feature_labels.values():
        try: lbl.config(text="")
        except: pass
    try: comment_placeholder.config(text="")
    except: pass
    try: pipe_canvas1.delete("all")
    except: pass
    for tag in ("upstream_text", "flange_text", "us_arrow", "ds_arrow", "bend_text", "pipe_icon"):
        try: pipe_canvas.delete(tag)
        except: pass
    try: canvas.yview_moveto(0.0)
    except: pass


# ---------- UI builder (moved/centralized) ----------
def _build_ui(project_root=None):
    global BUTTON_PANEL_W, button_frame, input_frame, toolbar, defect_entry
    global preview_holder, container, canvas, scrollable_frame, scrollbar
    global client_desc_frame, main_frame, comment_frame, feature_desc_frame, third_frame
    global pipe_canvas1, pipe_canvas, mid_x, mid_y
    global valve_img, bend_img, flange_img, flowtee_img, magnet_img
    global comment_placeholder

    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Custom.Horizontal.TProgressbar",
        troughcolor="white",
        background="deepskyblue",
        thickness=25,
        bordercolor="white",
        lightcolor="deepskyblue",
        darkcolor="deepskyblue"
    )

    if not EMBEDDED:
        root.title("Digsheet")
        root.state('zoomed')
        root.resizable(False, False)
    root.configure(bg="white")

    # --- Right panel (fixed width) ---
    screen_w = root.winfo_screenwidth()
    BUTTON_PANEL_W = (screen_w/2) - 150
    button_frame = tk.Frame(ROOT_CONTAINER, bg="white", width=BUTTON_PANEL_W)
    button_frame.pack(side="right", fill="y", padx=50, pady=0, anchor="n")
    button_frame.pack_propagate(False)

    input_frame = tk.Frame(button_frame, bg="white")
    input_frame.pack(side="top", fill="both", expand=True, pady=(8,0))

    toolbar = tk.Frame(input_frame, bg="white"); toolbar.pack(side="top", fill="x", pady=(0, 8))

    # Group 1
    group1 = tk.LabelFrame(toolbar, text="", bg="white", fg="gray40", relief="groove", bd=1, padx=6, pady=4)
    group1.pack(side="left", padx=(0, 10))
    tk.Label(group1, text="Enter Defect S.no:", bg="white").pack(side="left", padx=(2, 6))
    defect_entry = tk.Entry(group1, width=8); defect_entry.pack(side="left", padx=(0, 6))
    tk.Button(group1, text="Load",  command=on_load_click).pack(side="left", padx=3)
    tk.Button(group1, text="Save Current",  command=open_save_dialog).pack(side="left", padx=3)
    tk.Button(group1, text="Print current", command=print_all_sections_dialog).pack(side="left", padx=3)

    # Group 2
    group2 = tk.Frame(toolbar, bg="white"); group2.pack(side="left", padx=2)
    tk.Button(group2, text="Batch Export", command=open_batch_dialog_new).pack(side="left")

    # Group 3
    group3 = tk.Frame(toolbar, bg="white"); group3.pack(side="left", padx=2)
    tk.Button(group3, text="MultiPreview", command=open_preview_dialog).pack(side="left")
    tk.Button(group3, text="Reset", command=reset_ui).pack(side="left", padx=3)

    # Embedded preview holder
    global _preview_placeholder_ref
    preview_holder = tk.Frame(input_frame, bg="white", highlightbackground="#e8e8e8", highlightthickness=3)
    preview_holder.pack(side="top", fill="both", expand=True, pady=(8, 0))
    _show_preview_placeholder()

    # Icons
    try:
        icon_path = os.getcwd() + "/dig" + "/digsheet_icon/"
        valve_img  = ImageTk.PhotoImage(Image.open(icon_path + "valve.png").resize((18, 18)))
        bend_img   = ImageTk.PhotoImage(Image.open(icon_path + "bend.png").resize((18, 18)))
        flange_img = ImageTk.PhotoImage(Image.open(icon_path + "flange.png").resize((18, 18)))
        flowtee_img= ImageTk.PhotoImage(Image.open(icon_path + "flowtee.png").resize((18, 18)))
        magnet_img = ImageTk.PhotoImage(Image.open(icon_path + "magnet.png").resize((18, 18)))
    except Exception as e:
        print("Image loading error:", e)
        valve_img = bend_img = flange_img = flowtee_img = magnet_img = None

    # Scrollable digsheet area (left)
    container = tk.Frame(ROOT_CONTAINER); container.pack(fill="both", expand=True)
    canvas = tk.Canvas(container, bg="white"); canvas.pack(side="left", fill="both", expand=True)

    # slim scrollbar
    global scrollbar
    scrollbar = tk.Scrollbar(canvas, orient="vertical", command=canvas.yview, width=9)
    canvas.configure(yscrollcommand=_yscroll_set)

    # inner scrollable frame
    global scrollable_active
    scrollable_active = True
    global scrollable_frame
    scrollable_frame = tk.Frame(canvas, bg="white")
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    scrollable_frame.bind("<Configure>", on_frame_configure)
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)

    # ---------- Client Description ----------
    global comment_placeholder
    global logo_lbl  # keep reference if image loads

    client_desc_frame = tk.Frame(scrollable_frame, bg="white", padx=5, pady=2,
                                 highlightbackground="black", highlightthickness=1)
    client_desc_frame.pack(fill="both", padx=(15, 15), pady=(5,0))

    tk.Label(client_desc_frame, text="Client Description:", bg="white",
             fg="deepskyblue", font=("Arial", 10, "bold")).pack(side="top", pady=(2, 6))

    left_frame = tk.Frame(client_desc_frame, bg="white"); left_frame.pack(side="left", fill="both", expand=True)
    left_frame.grid_columnconfigure(0, weight=0, minsize=130)
    left_frame.grid_columnconfigure(1, weight=1)

    fields_top = [("Client", client_var), ("Pipeline Name", pipeline_name_var), ("Pipeline Section", pipeline_section_var)]
    for r, (txt, var) in enumerate(fields_top):
        tk.Label(left_frame, text=f"{txt}:", bg="white", anchor="w", font=("Arial", 9))\
            .grid(row=r, column=0, sticky="w", padx=(10, 6), pady=(2, 2))
        tk.Entry(left_frame, textvariable=var, width=40, bg="white", bd=0, highlightthickness=0, relief="flat")\
            .grid(row=r, column=1, sticky="ew", padx=(0, 10), pady=(2, 2))

    # logo (optional)
    try:
        # change this path if needed
        logo_img = Image.open(r"F:\work_new\client_software\PIE_dv_new\ui\icons\vdt-logo.png").resize((100, 100))
        logo_tk = ImageTk.PhotoImage(logo_img)
        logo_lbl = tk.Label(client_desc_frame, image=logo_tk, bg="white")
        logo_lbl.place(relx=1.0, rely=0.5, anchor="e", x=-10)
        client_desc_frame.logo_ref = logo_tk
    except Exception as e:
        print("Logo load failed:", e)

    # ---------- Main (Feature on Pipe + Pipe Description) ----------
    main_frame = tk.Frame(scrollable_frame, bg="white"); main_frame.pack(pady=5, fill="x", padx=10)

    feature_frame = tk.Frame(main_frame, bg="white", padx=5, pady=5, highlightbackground="black", highlightthickness=1)
    feature_frame.pack(side="left", fill="both", expand=True, padx=5)
    tk.Label(feature_frame, text="Feature Location on Pipe:", bg="white", fg="deepskyblue",
             font=("Arial", 10, "bold")).pack(pady=(0, 5))
    pipe_canvas1 = tk.Canvas(feature_frame, width=360, height=160, bg="white", highlightthickness=0)
    pipe_canvas1.pack()

    desc_frame = tk.Frame(main_frame, bg="white", padx=5, pady=5, highlightbackground="black", highlightthickness=1)
    desc_frame.pack(side="left", fill="both", expand=True, padx=5)
    tk.Label(desc_frame, text="Pipe Description:", bg="white", fg="deepskyblue",
             font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=5, padx=5, pady=(0, 5), sticky="ew")

    fields = [
        ("Pipe Number", pipe_id_var),
        ("Pipe Length (m)", length_var),
        ("WT (mm)", wt_var),
        ("Latitude", latitude_var),
        ("Longitude", longitude_var),
        ("Altitude (m)", altitude_var),
    ]
    for i, (label, var) in enumerate(fields, start=1):
        tk.Label(desc_frame, text=label + ":", bg="white", anchor="w", font=("Arial", 9))\
            .grid(row=i, column=0, sticky="w", padx=(5, 2), pady=(2, 2))
        tk.Label(desc_frame, textvariable=var, bg="white", anchor="w", font=("Arial", 9))\
            .grid(row=i, column=1, sticky="w", padx=(2, 10), pady=(2, 2))
    for col in range(2): desc_frame.grid_columnconfigure(col, weight=1)

    # ---------- Comment ----------
    global comment_frame
    comment_frame = tk.Frame(scrollable_frame, bg="white", padx=5, pady=2,
                             highlightbackground="black", highlightthickness=1)
    comment_frame.pack(fill="both", padx=(15, 15), pady=(5,5))
    tk.Label(comment_frame, text="Comment:", bg="white", fg="deepskyblue",
             font=("Arial", 10, "bold")).pack(side="top", pady=(0, 5))
    comment_placeholder = tk.Label(comment_frame, text="", bg="white", anchor="w", justify="left", font=("Arial", 9))
    comment_placeholder.pack(fill="both", expand=True, padx=10, pady=20)

    # ---------- Feature Description ----------
    global feature_desc_frame
    feature_desc_frame = tk.Frame(scrollable_frame, bg="white", padx=5, pady=2,
                                  highlightbackground="black", highlightthickness=1)
    feature_desc_frame.pack(fill="both", padx=15)

    for col in range(5):
        feature_desc_frame.grid_columnconfigure(col, weight=1)
    feature_desc_frame.grid_columnconfigure(2, minsize=80)

    section_title = tk.Label(feature_desc_frame, text="Feature Description:", bg="white", fg="deepskyblue",
                             font=("Arial", 10, "bold"), anchor="center", justify="center")
    section_title.grid(row=0, column=0, columnspan=5, pady=(0, 5), sticky="ew")

    left_fields  = ["Feature", "Feature type", "Anomaly dimension class", "Surface Location",
                    "Remaining wall thickness (mm)", "ERF", "Safe pressure (kg/cm²)"]
    right_fields = ["Absolute Distance (m)", "Length (mm)", "Width (mm)", "Max. Depth(%)",
                    "Orientation(hr:min)", "Latitude", "Longitude"]

    label_padx = (5, 2); value_padx = (2, 10)
    for i, label_text in enumerate(left_fields):
        tk.Label(feature_desc_frame, text=label_text + ":", bg="white", anchor="w", font=("Arial", 9))\
            .grid(row=i+1, column=0, sticky="w", padx=label_padx, pady=2)
        lbl = tk.Label(feature_desc_frame, text="", bg="white", anchor="w", font=("Arial", 9))
        lbl.grid(row=i+1, column=1, sticky="w", padx=value_padx, pady=2)
        feature_labels[label_text] = lbl

    for i, label_text in enumerate(right_fields):
        tk.Label(feature_desc_frame, text=label_text + ":", bg="white", anchor="w", font=("Arial", 9))\
            .grid(row=i+1, column=3, sticky="w", padx=label_padx, pady=2)
        lbl = tk.Label(feature_desc_frame, text="", bg="white", anchor="w", font=("Arial", 9))
        lbl.grid(row=i+1, column=4, sticky="w", padx=value_padx, pady=2)
        feature_labels[label_text] = lbl

    # ---------- Pipe Location (big canvas) ----------
    global third_frame
    third_frame = tk.Frame(scrollable_frame, bg="white", padx=10, pady=10,
                           highlightbackground="black", highlightthickness=1)
    third_frame.pack(fill="both", padx=15, pady=4)

    tk.Label(third_frame, text="Pipe Location:", bg="white", fg="deepskyblue",
             font=("Arial", 9, "bold")).grid(row=0, column=0, columnspan=5, sticky="ew")
    global pipe_canvas
    pipe_canvas = tk.Canvas(third_frame, width=650, height=370, bg="white", highlightthickness=0)
    pipe_canvas.grid(row=1, column=0, columnspan=5)
    pipe_canvas.update()

    for col in range(5):
        third_frame.grid_columnconfigure(col, weight=1)

    # midpoints
    global mid_x, mid_y
    canvas_width  = pipe_canvas.winfo_width()
    canvas_height = pipe_canvas.winfo_height()
    mid_x = int(canvas_width/2)
    mid_y = int(canvas_height/2)

    pipe_canvas.create_line(mid_x, 30, mid_x, mid_y + 150, arrow=tk.FIRST)


# ---------- Public API for embedding ----------
def mount_into(parent, *, project_root=None, pipe_tally_file=None, data_df=None):
    """
    Render the digsheets UI inside `parent` (a Frame in your main window).
    You may pass either:
      - data_df                      (already-loaded pandas DataFrame), OR
      - pipe_tally_file + project_root (same as argv flow)
    """
    global EMBEDDED, ROOT_CONTAINER, root, df
    EMBEDDED = True
    ROOT_CONTAINER = parent
    root = parent.winfo_toplevel()

    # data setup
    if data_df is not None:
        df = data_df
    elif pipe_tally_file and project_root:
        # mimic argv load once
        try:
            tmp_argv = sys.argv[:]
            sys.argv = [sys.argv[0], pipe_tally_file, project_root]
            on_load_click()
        finally:
            sys.argv = tmp_argv
    else:
        # no data yet; user can press Load (your app can populate later)
        pass

    _build_ui(project_root=project_root)
    return True  # simple okay flag


# ---------- Standalone ----------
def _standalone_main():
    global EMBEDDED, ROOT_CONTAINER, root, df
    EMBEDDED = False
    root = tk.Tk()
    ROOT_CONTAINER = root

    # if argv provided, on_load_click will read them
    _build_ui()
    root.mainloop()


# ---------- Optional helpers exposed ----------
def open_batch_dialog(output_mode="pdf"):
    return open_batch_dialog_new()


# ---------- Run ----------
if __name__ == "__main__":
    _standalone_main()
