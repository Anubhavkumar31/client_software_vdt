# # workers/pipe_loader.py
# from PyQt6.QtCore import QThread, pyqtSignal
# import os
# import traceback
# import pandas as pd
# from utils.assets import load_html_assets, load_pipe_tally_data

# class PipeLoaderWorker(QThread):
#     """
#     Worker for loading project assets and pipe tally data in the background.
#     Emits:
#         progress(str)   - textual progress updates
#         finished(dict)  - loaded assets
#         error(str)      - error message
#     """
#     progress = pyqtSignal(str)
#     finished = pyqtSignal(dict, pd.DataFrame)
#     error = pyqtSignal(str)

#     def __init__(self, project_dir: str, tally_file: str | None = None, parent=None):
#         super().__init__(parent)
#         self.project_dir = project_dir
#         self.tally_file = tally_file

#     def run(self):
#         try:
#             self.progress.emit("🔍 Loading project assets...")

#             # Load HTML assets (heatmaps, line plots, 3D, proximity)
#             assets = load_html_assets(self.project_dir)
#             self.progress.emit(f"✅ Found {sum(len(v) for v in assets.values())} HTML assets")

#             # Load pipe tally data if available
#             df = pd.DataFrame()
#             if self.tally_file and os.path.exists(self.tally_file):
#                 self.progress.emit("📊 Loading pipe tally file...")
#                 df = load_pipe_tally_data(self.tally_file)
#                 self.progress.emit(f"✅ Loaded tally with {len(df)} rows")

#             self.finished.emit(assets, df)

#         except Exception as e:
#             tb = traceback.format_exc()
#             self.error.emit(f"PipeLoaderWorker error: {e}\n{tb}")


# workers/pipe_loader.py
from __future__ import annotations

import os
import re
import glob
import traceback
import pandas as pd

from PyQt6.QtCore import QThread, pyqtSignal

from config.constants import HTML_ASSET_PATTERNS, PIPE_TALLY_PATTERNS
from utils.data_processing import process_table_data
import time
from glob import glob


class PipeLoaderWorker(QThread):
    # Signals for communication
    progress_updated = pyqtSignal(int, str)  # progress %, message
    data_loaded = pyqtSignal(object)  # pandas DataFrame
    assets_loaded = pyqtSignal(dict)  # asset paths dictionary
    table_data_ready = pyqtSignal(object)  # processed table data
    error_occurred = pyqtSignal(str)  # error message
    time_estimate = pyqtSignal(float)  # estimated time remaining

    def __init__(self, pkl_path, project_root, pipe_idx):
        super().__init__()
        self.pkl_path = pkl_path
        self.project_root = project_root
        self.pipe_idx = pipe_idx
        self.start_time = None

    def run(self):
        try:
            self.start_time = time.time()
            total_steps = 6

            # Step 1: Load pickle data
            self.progress_updated.emit(10, "Loading pipe data...")
            df = pd.read_pickle(self.pkl_path)
            self.data_loaded.emit(df)
            self._update_time_estimate(1, total_steps)
            print(f"Loaded pickle with {len(df)} rows")

            # Step 2: Find pipe directory
            self.progress_updated.emit(25, "Locating asset files...")
            pipe_dir = self._find_pipe_directory()
            self._update_time_estimate(2, total_steps)

            # Step 3: Load HTML assets
            self.progress_updated.emit(40, "Loading chart assets...")
            assets = self._load_html_assets(pipe_dir)
            self.assets_loaded.emit(assets)
            self._update_time_estimate(3, total_steps)

            # Step 4: Load pipe tally data
            self.progress_updated.emit(60, "Processing pipe tally...")
            table_data = self._load_pipe_tally_data(pipe_dir)
            self._update_time_estimate(4, total_steps)

            # Step 5: Process table data
            self.progress_updated.emit(80, "Preparing table data...")
            if table_data is not None:
                processed_data = self._process_table_data(table_data)
                self.table_data_ready.emit(processed_data)
            else:
                self.table_data_ready.emit(None)
            self._update_time_estimate(5, total_steps)

            # Step 6: Complete
            self.progress_updated.emit(100, "Loading complete!")
            self._update_time_estimate(6, total_steps)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _update_time_estimate(self, current_step, total_steps):
        elapsed = time.time() - self.start_time
        if current_step > 0:
            avg_time_per_step = elapsed / current_step
            remaining_steps = total_steps - current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            self.time_estimate.emit(estimated_remaining)

    def _find_pipe_directory(self):
        # Look for pipe directories inside pipes_data subfolder
        pipes_data_dir = os.path.join(self.project_root, "pipes_data")
        if not os.path.isdir(pipes_data_dir):
            print(f"[Warning] pipes_data directory not found in {self.project_root}")
            return None
        
        candidates = [
            os.path.join(pipes_data_dir, f"pipe_{self.pipe_idx}"),
            os.path.join(pipes_data_dir, f"pipe-{self.pipe_idx}"),
            os.path.join(pipes_data_dir, f"Pipe_{self.pipe_idx}"),
        ]
        return next((d for d in candidates if os.path.isdir(d)), None)


    def _load_html_assets(self, pipe_dir):
        if not pipe_dir:
            return {}

        def pick_one(patterns, exclude=None):
            exclude = exclude or []
            hits = []
            for pat in patterns:
                hits.extend(glob(os.path.join(pipe_dir, pat)))
            hits = [h for h in hits if not any(ex in os.path.basename(h).lower() for ex in (exclude or []))]
            exact = [h for h in hits if re.search(rf'{re.escape(str(self.pipe_idx))}\b', os.path.basename(h))]
            return exact[0] if exact else (hits[0] if hits else None)

        return {
            'hmap': pick_one(["*heatmap*.html"], exclude=["raw", "box"]),
            'hmap_r': pick_one(["*heatmap*raw*.html", "*raw*heatmap*.html"]),
            'heatmap_box': pick_one(["*heatmap*box*.html", "*box*heatmap*.html"]),
            'lplot': pick_one(["*lineplot*.html", "*line*.html"], exclude=["raw"]),
            'lplot_r': pick_one(["*lineplot*raw*.html", "*line*raw*.html"]),
            'pipe3d': pick_one(["*pipe3d*.html", "pipe3d*.html"]),
            'prox_linechart': pick_one(["proximity_linechart*.html", "*proximity_linechart*.html"])
        }

    def _load_pipe_tally_data(self, pipe_dir):
        if not pipe_dir:
            return None

        def pick_one(patterns, exclude=None):
            exclude = exclude or []
            hits = []
            for pat in patterns:
                hits.extend(glob(os.path.join(pipe_dir, pat)))
            hits = [h for h in hits if not any(ex in os.path.basename(h).lower() for ex in (exclude or []))]
            exact = [h for h in hits if re.search(rf'{re.escape(str(self.pipe_idx))}\b', os.path.basename(h))]
            return exact[0] if exact else (hits[0] if hits else None)

        pipe_tally_csv = pick_one([f"*PipeTally{self.pipe_idx}.csv", f"*PipeTally{self.pipe_idx}.xlsx"])
        if pipe_tally_csv:
            try:
                if pipe_tally_csv.lower().endswith(".csv"):
                    df = pd.read_csv(pipe_tally_csv)
                else:
                    df = pd.read_excel(pipe_tally_csv)
                return df
            except Exception:
                pass

        # Fallback to defects.csv
        ds_csv = pick_one(["*defectS*.csv", "*defects*.csv"])
        if ds_csv:
            try:
                return pd.read_csv(ds_csv)
            except Exception:
                pass

        return None

    def _process_table_data(self, df):
        if df is None or df.empty:
            return None

        # Check if this is a PipeTally file (has Feature Type column) or defects.csv
        if "Feature Type" in df.columns:
            # Filter Metal Loss defects
            original_count = len(df)
            df = df[df["Feature Type"].astype(str).str.strip().str.lower() == "metal loss"]

            if df.empty:
                return None

            # Round numeric columns
            numeric_columns = [
                'Depth %', 'Depth (mm)', 'ERF (ASME B31G)', 'Psafe (ASME B31G) Barg',
                'Abs. Distance (m)', 'Distance to U/S GW(m)', 'Length (mm)',
                'Width (mm)', 'WT (mm)', 'Pipe Length (mm)'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').round(3)

        return df