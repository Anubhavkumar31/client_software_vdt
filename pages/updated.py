import datetime
import os, io
import tkinter as tk
import traceback
from tkinter import messagebox, filedialog
import pandas as pd
import math
from PIL import ImageGrab, Image, ImageTk
import time

root = tk.Tk()
root.title("Digsheet")

# open maximized (keeps the title‐bar with minimize/close)
root.state('zoomed')
# but lock down resizing so they can’t shrink it
root.resizable(False, False)

root.configure(bg="white")

# — Right-side button panel —
button_frame = tk.Frame(root, bg="white")
button_frame.pack(side="right", fill="y", padx=50, pady=100)
input_frame = tk.Frame(button_frame, bg="white")
input_frame.pack(pady=20)

def load_excel():
    global df
    file_path = filedialog.askopenfilename(
        title="Select Excel File",
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )
    if file_path:
        try:
            df = pd.read_excel(file_path, skiprows=3)
            messagebox.showinfo("Success", f"Excel loaded successfully:\n{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Excel file.\n{str(e)}")

try:
    icon_path = os.getcwd() + "/digsheet_icon/"
    valve_img = ImageTk.PhotoImage(Image.open(icon_path + "valve.png").resize((18, 18)))
    bend_img = ImageTk.PhotoImage(Image.open(icon_path + "bend.png").resize((18, 18)))
    flange_img = ImageTk.PhotoImage(Image.open(icon_path + "flange.png").resize((18, 18)))
    flowtee_img = ImageTk.PhotoImage(Image.open(icon_path + "flowtee.png").resize((18, 18)))
    magnet_img = ImageTk.PhotoImage(Image.open(icon_path + "magnet.png").resize((18, 18)))
except Exception as e:
    print("Image loading error:", e)
    valve_img = bend_img = flange_img = flowtee_img = magnet_img = None





# Title
# tk.Label(root, text="DIGSHEET GENERATOR", font=("Arial", 12, "bold"), bg="white").pack()

# --- Step 1: Create a Canvas inside a container Frame ---
container = tk.Frame(root)
container.pack(fill="both", expand=True)

canvas = tk.Canvas(container, bg="white")
canvas.pack(side="left", fill="both", expand=True)

# # --- Step 2: Add a vertical scrollbar linked to the canvas ---
# scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
# scrollbar.pack(side="right", fill="y")

# canvas.configure(yscrollcommand=scrollbar.set)

# --- Step 3: Create a scrollable frame inside the canvas ---
scrollable_frame = tk.Frame(canvas, bg="white")

# Connect canvas and inner frame
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

scrollable_frame.bind("<Configure>", on_frame_configure)

# Embed the scrollable frame window in canvas
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# # OPTIONAL: Mouse wheel scrolling (Windows & Linux)
# def _on_mousewheel(event):
#     canvas.yview_scroll(int(-1*(event.delta/120)), "units")

# canvas.bind_all("<MouseWheel>", _on_mousewheel)



# Variables
pipe_id_var = tk.StringVar()
length_var = tk.StringVar()
wt_var = tk.StringVar()
latitude_var = tk.StringVar()
longitude_var = tk.StringVar()

from PIL import ImageGrab
from PIL import ImageGrab

# def save_as_image():
#     # ensure everything’s rendered
#     root.update_idletasks()
    
#     # get the root window’s screen position and size
#     x0 = root.winfo_rootx()
#     y0 = root.winfo_rooty()
#     x1 = x0 + root.winfo_width()
#     y1 = y0 + root.winfo_height()

#     # ask where to save
#     filepath = filedialog.asksaveasfilename(
#         defaultextension=".png",
#         filetypes=[("PNG Image", "*.png"), ("PDF File", "*.pdf")]
#     )
#     if not filepath:
#         return

#     # grab exactly the window area
#     img = ImageGrab.grab(bbox=(x0, y0, x1-660, y1+220))

#     # save as PNG or PDF
#     if filepath.lower().endswith(".pdf"):
#         img.save(filepath, "PDF", resolution=100.0)
#     else:
#         img.save(filepath)

def save_as_image():
    # ensure everything’s rendered
    root.update_idletasks()
    
    # get the root window’s screen position and size
    x0 = root.winfo_rootx()
    y0 = root.winfo_rooty()
    x1 = x0 + root.winfo_width()
    y1 = y0 + root.winfo_height()

    # ask where to save
    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("PDF File", "*.pdf")]
    )
    if not filepath:
        return

    # grab exactly the window area
    img = ImageGrab.grab(bbox=(x0, y0, x1-660, y1+220))

    try:
        # save as PNG or PDF
        if filepath.lower().endswith(".pdf"):
            img.save(filepath, "PDF", resolution=100.0)
        else:
            img.save(filepath)
        
        # <-- Here’s your success popup:
        messagebox.showinfo("Saved!", f"File saved successfully:\n{filepath}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save file:\n{e}")


# def save_as_image():
#     """Capture frames and save as image - Part 1: feature_frame + desc_frame + feature_desc_frame"""
#     from PIL import Image
#
#     # Update layout before capture
#     root.update_idletasks()
#
#     # Ask for save path
#     filepath = filedialog.asksaveasfilename(
#         defaultextension=".png",
#         filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
#     )
#     if not filepath:
#         return
#
#     try:
#         # ===== PART 1: Capture top section (main_frame + feature_desc_frame) =====
#
#         # Force update all frames to ensure proper sizing
#         main_frame.update_idletasks()
#         feature_desc_frame.update_idletasks()
#         feature_frame.update_idletasks()
#         desc_frame.update_idletasks()
#
#         # Get main_frame coordinates (contains feature_frame and desc_frame side by side)
#         main_x = main_frame.winfo_rootx()
#         main_y = main_frame.winfo_rooty()
#         main_width = main_frame.winfo_width()
#         main_height = main_frame.winfo_height()
#
#         # Get feature_desc_frame coordinates
#         feature_x = feature_desc_frame.winfo_rootx()
#         feature_y = feature_desc_frame.winfo_rooty()
#         feature_width = feature_desc_frame.winfo_width()
#         feature_height = feature_desc_frame.winfo_height()
#
#         # Calculate combined bounding box with proper margins
#         part1_x = min(main_x, feature_x) - 5  # Add small margin
#         part1_y = min(main_y, feature_y) - 5  # Add small margin
#         part1_w = max(main_x + main_width, feature_x + feature_width) + 5  # Add margin
#         part1_h = max(main_y + main_height, feature_y + feature_height) + 5  # Add margin
#
#         # Debug print to check dimensions
#         print(f"Main frame: x={main_x}, y={main_y}, w={main_width}, h={main_height}")
#         print(f"Feature desc frame: x={feature_x}, y={feature_y}, w={feature_width}, h={feature_height}")
#         print(f"Capture area: x={part1_x}, y={part1_y}, w={part1_w}, h={part1_h}")
#
#         # Capture the first part
#         part1_image = ImageGrab.grab(bbox=(part1_x, part1_y, part1_w, part1_h))
#
#         # Save the image
#         part1_image.save(filepath)
#
#         messagebox.showinfo("Success", f"Saved Part 1 image as: {filepath}")
#
#     except Exception as e:
#         messagebox.showerror("Error", f"Failed to save image: {str(e)}")
#         import traceback
#         traceback.print_exc()

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
        # latitude_var.set(str(row.get("Latitude", "")) if "Latitude" in row else "")
        # longitude_var.set(str(row.get("Longitude", "")) if "Longitude" in row else "")

        lat_col = next((c for c in df.columns if c.strip().lower() == "latitude"), None)
        lon_col = next((c for c in df.columns if c.strip().lower() == "longitude"), None)
        latitude_var .set(str(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else "")
        longitude_var.set(str(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else "")

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

# tk.Button(input_frame, text="Save as Image", command=save_as_image).grid(row=0, column=5, padx=20)

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
tk.Label(desc_frame, text="Pipe Description:", bg="white", fg="deepskyblue",
         font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=5, pady=(0, 10), sticky="ew")

# Layout fields
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

# Ensure the grid expands to center-align contents
for col in range(2):  # Adjust columns 0 and 1
    desc_frame.grid_columnconfigure(col, weight=1)

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

# pipe_canvas.create_text(mid_x, mid_y, text='mid')

# Central vertical line
pipe_canvas.create_line(mid_x, 30, mid_x, mid_y + 150, arrow=tk.FIRST)

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

def on_load_click():
    global df
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

    # ========================
    # FEATURE Display Logic (2 Up + 2 Down) - Shortened
    # ========================
    feature_slots = [
        {"x": mid_x - 190, "arrow_x": mid_x - 200, "text_x": mid_x - 160, "source": features_upstream[::-1], "index": 1},  # flange1
        {"x": mid_x - 90,  "arrow_x": mid_x - 100,  "text_x": mid_x - 60,  "source": features_upstream[::-1], "index": 0},  # flange2
        {"x": mid_x + 110, "arrow_x": mid_x + 120, "text_x": mid_x + 80, "source": features_downstream,      "index": 0},  # flange3
        {"x": mid_x + 210, "arrow_x": mid_x + 220, "text_x": mid_x + 180, "source": features_downstream,     "index": 1},  # flange4
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

            # Draw feature text
            pipe_canvas.create_text(x, mid_y - 160, text=name, font=("Arial", 10), tags="flange_text")
            pipe_canvas.create_text(x, mid_y - 145, text=dist, font=("Arial", 9), tags="flange_text")
            pipe_canvas.create_text(x, mid_y - 130, text=lat, font=("Arial", 9), tags="flange_text")
            pipe_canvas.create_text(x, mid_y - 115, text=lon, font=("Arial", 9), tags="flange_text")

            # Draw arrow and distance
            arrow_val = round(abs(float(upstream_weld_dist) - float(dist_val)), 2)
            pipe_canvas.create_line(arrow_x, mid_y - 95, arrow_x, mid_y - 65, arrow=tk.FIRST, fill="deepskyblue", width=2, tags="us_arrow")
            pipe_canvas.create_text(text_x, mid_y - 80, text=f"{arrow_val}(m)", font=("Arial", 9), tags="us_arrow")
        except:
            continue  # Skip if out of bounds or missing

    # ========================
    # BEND Display Logic (3 Up + 3 Down) - Shortened
    # ========================
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

            # Draw bend text
            pipe_canvas.create_text(slot["x_name"], mid_y - 45, text=name, font=("Arial", 10), tags="bend_text")
            pipe_canvas.create_text(slot["x_dist"], mid_y - 30, text=dist, font=("Arial", 9), tags="bend_text")
            pipe_canvas.create_text(slot["x_lat"], mid_y - 15, text=lat, font=("Arial", 9), tags="bend_text")
            pipe_canvas.create_text(slot["x_lon"], mid_y,        text=lon, font=("Arial", 9), tags="bend_text")

            # Draw triangle + arrow text
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
        # Get all rows in df where column 3 matches pipe_num
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

# tk.Button(input_frame, text="Load", command=on_load_click).grid(row=0, column=4, padx=15)
# Sidebar controls
tk.Button(input_frame, text="Load Excel", command=load_excel)\
    .grid(row=0, column=0, columnspan=2, pady=5)
tk.Label(input_frame, text="Enter Defect S.no:", bg="white")\
    .grid(row=1, column=0, sticky="e", pady=5)
defect_entry = tk.Entry(input_frame, width=10)
defect_entry.grid(row=1, column=1, pady=5)
tk.Button(input_frame, text="Load", command=on_load_click)\
    .grid(row=2, column=0, columnspan=2, pady=5)
tk.Button(input_frame, text="Save as Image", command=save_as_image)\
    .grid(row=3, column=0, columnspan=2, pady=5)


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

