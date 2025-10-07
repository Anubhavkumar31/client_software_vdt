# defect_table.py
# Modern, fast defect table using Tkinter + ttkbootstrap

import tkinter as tk
from tkinter import ttk
from datetime import datetime

try:
    import ttkbootstrap as tb
except ImportError:
    tb = None

COLUMNS_DEFAULT = (
    "Defect_id","Abs. Distance (m)","Distance to U/S GW(m)","Pipe Number",
    "Pipe Length (mm)","Feature Type","Feature Identification",
    "Dimensions Classification","Orientation o' clock","WT (mm)","Length (mm)",
    "Width (mm)","Depth %","Depth (mm)","Location",
    "ERF (ASME B31G)","Psafe (ASME B31G) Barg","Latitude","Longitude",
    "Altitude","Comment"
)

def _format(v):
    if v is None: return ""
    if isinstance(v, float): return f"{v:.6g}"
    return str(v)

def run_defect_table(rows, *, title="Defect Table", columns=COLUMNS_DEFAULT):
    """
    rows: list[dict] — keys can be a superset; we'll display `columns` in order
    columns: tuple[str] — ordered headers to show
    """
    # Create window (bootstrap if available)
    root = tb.Window(themename="cosmo") if tb else tk.Tk()
    root.title(title)
    root.geometry("1280x700+200+80")

    # ---- Top bar (search, counts, export)
    top = ttk.Frame(root, padding=(12, 10))
    top.pack(side="top", fill="x")

    search_var = tk.StringVar()
    ttk.Label(top, text="Search:").pack(side="left")
    search_entry = ttk.Entry(top, textvariable=search_var, width=32)
    search_entry.pack(side="left", padx=(6, 12))

    count_var = tk.StringVar(value=f"{len(rows)} rows")
    ttk.Label(top, textvariable=count_var).pack(side="left")

    def do_export_csv():
        import csv, os
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"defects_{ts}.csv"
        path = os.path.abspath(name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(columns)
            for r in filtered_rows:
                w.writerow([_format(r.get(c, "")) for c in columns])
        msg.set(f"Saved: {path}")

    ttk.Button(top, text="Export CSV", command=do_export_csv).pack(side="right")

    # ---- Treeview
    wrap = ttk.Frame(root, padding=(12, 0, 12, 12))
    wrap.pack(side="top", fill="both", expand=True)

    tree = ttk.Treeview(wrap, columns=columns, show="headings", height=20)
    tree.pack(side="left", fill="both", expand=True)

    # Scrollbars
    vs = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
    hs = ttk.Scrollbar(root, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
    vs.pack(side="left", fill="y")
    hs.pack(side="bottom", fill="x")

    # Headings + sensible widths
    for col in columns:
        tree.heading(col, text=col, anchor="center")
        width = 120
        if col in ("Feature Identification","Dimensions Classification", "Comment"): width = 200
        if col in ("Defect_id", "Depth %", "Depth (mm)", "WT (mm)", "Length (mm)", "Width (mm)"): width = 110
        if col in ("Abs. Distance (m)", "Distance to U/S GW(m)"): width = 140
        tree.column(col, width=width, anchor="center", stretch=True)

    # Row striping + highlight for deeper defects
    tree.tag_configure("even", background="#f7f9fe")
    tree.tag_configure("odd",  background="#ffffff")
    tree.tag_configure("deep", background="#dbe7ff")  # accent for Depth % >= 30 (change as you like)

    # Filtering
    filtered_rows = list(rows)

    def refill():
        tree.delete(*tree.get_children())
        for i, r in enumerate(filtered_rows):
            vals = [_format(r.get(c, "")) for c in columns]
            tags = ("even" if i % 2 == 0 else "odd",)
            try:
                d = float(r.get("Depth %", 0) or 0)
                if d >= 30:  # tweak threshold
                    tags = tags + ("deep",)
            except Exception:
                pass
            tree.insert("", "end", values=vals, tags=tags)
        count_var.set(f"{len(filtered_rows)} rows")

    def apply_search(*_):
        q = search_var.get().strip().lower()
        if not q:
            filtered_rows[:] = rows
        else:
            def hit(rd):
                for c in columns:
                    v = rd.get(c, "")
                    if q in str(v).lower():
                        return True
                return False
            filtered_rows[:] = [rd for rd in rows if hit(rd)]
        refill()

    search_var.trace_add("write", apply_search)

    # right-click: copy selected rows
    def copy_selected(event=None):
        sel = tree.selection()
        if not sel: return
        lines = [",".join(columns)]
        for iid in sel:
            vals = tree.item(iid, "values")
            lines.append(",".join(map(str, vals)))
        root.clipboard_clear()
        root.clipboard_append("\n".join(lines))

    tree.bind("<Control-c>", copy_selected)
    tree.bind("<Control-C>", copy_selected)

    # status line
    msg = tk.StringVar(value="")
    ttk.Label(root, textvariable=msg, anchor="w", padding=(12, 6)).pack(side="bottom", fill="x")

    # initial fill
    refill()
    search_entry.focus_set()
    root.mainloop()
