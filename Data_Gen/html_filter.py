
# '''
# This code will input the raw pkl file( {x}.pkl ) and generate out the plots to:
# Pipe_{x}  with ending format as <Name_{x}.html>
# '''

# import os
# # Limit intra-process math threads to avoid over-subscription
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

# import pandas as pd
# from pathlib import Path
# import numpy as np
# import re
# import plotly.graph_objects as go
# from scipy.signal import savgol_filter
# from datetime import datetime
# from joblib import Parallel, delayed
# import warnings
# from pandas.api.types import is_numeric_dtype
# from sklearn.preprocessing import MinMaxScaler

# import plotly.express as px
# from typing import Optional, Union, List
# from glob import glob

# warnings.filterwarnings("ignore", category=FutureWarning)

# # -------------------- CONFIG --------------------
# INITIAL_READ = 0.0      # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
# UPPER_SENS_MUL = 1
# LOWER_SENS_MUL = 3
# # -1 / 0 / None / "auto" => auto (CPU-1, at least 1). Or set an int, e.g. 4
# WORKERS = 4
# # ------------------------------------------------


# def pre_process_data(datafile, pipe_number, output_folder):

#     datafile_original = datafile.copy(deep=True)
#     # Create DataFrame from raw data
#     df_new_tab9 = pd.DataFrame(
#         datafile,
#         columns=[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)]
#     )

#     df_new_tab10 = df_new_tab9.copy()
#     sensor_columns = df_new_tab9.columns.tolist()

#     # Denoising using Savitzky-Golay filter
#     window_length = 15
#     polyorder = 2
#     for col in sensor_columns:
#         data = df_new_tab9[col].values
#         time_index = np.arange(len(df_new_tab9))
#         trend = np.polyval(np.polyfit(time_index, data, 2), time_index)
#         data_denoised = savgol_filter(data - trend, window_length, polyorder)
#         df_new_tab9[col] = data_denoised

#     df_raw_straight = df_new_tab9.copy()
    
#     # Setting bounds and applying conditions
#     sens_mean = df_new_tab9.abs().mean()
#     standard_deviation = df_new_tab9.std(axis=0, skipna=True)

#     mean_plus_sigma = sens_mean + UPPER_SENS_MUL * standard_deviation
#     mean_negative_sigma = sens_mean - LOWER_SENS_MUL * standard_deviation

#     # Apply noise filtering to zero-out in-bound values
#     for col in df_new_tab9.columns:
#         if col in mean_plus_sigma.index and col in mean_negative_sigma.index:
#             df_new_tab9[col] = np.where(
#                 (df_new_tab9[col] >= mean_negative_sigma[col]) &
#                 (df_new_tab9[col] <= mean_plus_sigma[col]),
#                 0,
#                 df_new_tab9[col]
#             )

#     initial_read = INITIAL_READ
#     roll = datafile['ROLL'] - initial_read
#     # odoData = datafile['ODDO1']  # kept if needed later

#     def degrees_to_hours_minutes2(degrees):
#         if (degrees < 0):
#             degrees = degrees % 360
#         elif degrees >= 360:
#             degrees %= 360
#         degrees_per_second = 360 / (12 * 60 * 60)
#         total_seconds = degrees / degrees_per_second
#         hours = int(total_seconds // 3600)
#         minutes = int((total_seconds % 3600) // 60)
#         seconds = int(total_seconds % 60)
#         return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

#     def add_sensor_keys(d):
#         for e in d:
#             new_dict = {**e}
#             for i in range(1, 96):
#                 new_dict[f'Roll_Sensor_{i}'] = e['Roll_Sensor_0'] + (3.75 * i)
#             yield new_dict

#     def check_time_range(time_str):
#         start_time = '00:00:00'
#         end_time = '00:07:29'
#         time_to_check = datetime.strptime(time_str, '%H:%M:%S')
#         start_time_dt = datetime.strptime(start_time, '%H:%M:%S')
#         end_time_dt = datetime.strptime(end_time, '%H:%M:%S')
#         return start_time_dt <= time_to_check <= end_time_dt

#     d = []
#     for pos in roll:
#         d.append({f"Roll_Sensor_0": pos})

#     upd_d = list(add_sensor_keys(d))
#     oriData = pd.DataFrame.from_dict(data=upd_d)
#     clockData = oriData.applymap(degrees_to_hours_minutes2)

#     test_clockData = clockData.copy()
#     test_clockData = test_clockData.apply(pd.to_datetime, format='%H:%M:%S')
#     test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M:%S'))
#     test_clockData = test_clockData.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
#     test_clockData = test_clockData.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
#     test_clockData = test_clockData.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

#     test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_new_tab9.columns)))

#     def create_time_dict():
#         time_ranges_2 = [
#             '00:00:00', '00:07:30', '00:15:00', '00:22:30', '00:30:00', '00:37:30', '00:45:00', '00:52:30', '01:00:00', '01:07:30', 
#             '01:15:00', '01:22:30', '01:30:00', '01:37:30', '01:45:00', '01:52:30', '02:00:00', '02:07:30', '02:15:00', '02:22:30', 
#             '02:30:00', '02:37:30', '02:45:00', '02:52:30', '03:00:00', '03:07:30', '03:15:00', '03:22:30', '03:30:00', '03:37:30', 
#             '03:45:00', '03:52:30', '04:00:00', '04:07:30', '04:15:00', '04:22:30', '04:30:00', '04:37:30', '04:45:00', '04:52:30', 
#             '05:00:00', '05:07:30', '05:15:00', '05:22:30', '05:30:00', '05:37:30', '05:45:00', '05:52:30', '06:00:00', '06:07:30', 
#             '06:15:00', '06:22:30', '06:30:00', '06:37:30', '06:45:00', '06:52:30', '07:00:00', '07:07:30', '07:15:00', '07:22:30', 
#             '07:30:00', '07:37:30', '07:45:00', '07:52:30', '08:00:00', '08:07:30', '08:15:00', '08:22:30', '08:30:00', '08:37:30', 
#             '08:45:00', '08:52:30', '09:00:00', '09:07:30', '09:15:00', '09:22:30', '09:30:00', '09:37:30', '09:45:00', '09:52:30', 
#             '10:00:00', '10:07:30', '10:15:00', '10:22:30', '10:30:00', '10:37:30', '10:45:00', '10:52:30', '11:00:00', '11:07:30', 
#             '11:15:00', '11:22:30', '11:30:00', '11:37:30', '11:45:00', '11:52:30'
#         ]
#         return {key: [] for key in time_ranges_2}

#     time_dict_1 = create_time_dict()
#     rang = list(time_dict_1.keys())

#     for _, row in test_clockData.iterrows():
#         xl = list(row)
#         xd = dict(row)
#         xkeys = list(xd.keys())
#         c = 0
#         for _, dval in xd.items():
#             if check_time_range(dval):
#                 ind = xl.index(dval)
#                 _ = xl[ind:] + xl[:ind]  # not used later but kept for clarity
#                 break

#         curr = ind
#         while True:
#             ck = xkeys[curr]
#             time_dict_1[rang[c]].append((curr, ck, xd[ck]))
#             c += 1
#             curr = (curr + 1) % len(xkeys)
#             if curr == ind:
#                 break

#     map_ori_sens = pd.DataFrame(time_dict_1)

#     val_ori_sens = map_ori_sens.copy()

#     def extract_string(cell):
#         return cell[1]

#     val_ori_sens = val_ori_sens.applymap(extract_string)

#     test_val = val_ori_sens.copy()

#     for r, e in val_ori_sens.iterrows():
#         c = 0
#         for _, tup_value in e.items():
#             cell_v = df_new_tab9.at[r, tup_value]
#             test_val.iloc[r, c] = cell_v
#             c += 1

#     # map_val_sens is created but not used later; keeping for parity
#     map_val_sens = pd.DataFrame(index=test_val.index, columns=test_val.columns)
#     for column in test_val.columns:
#         for i in range(test_val.shape[0]):
#             map_value = map_ori_sens.loc[i, column]
#             test_value = test_val.loc[i, column]
#             map_val_sens.loc[i, column] = (*map_value, test_value)
    
#     create_plots(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original)
#     return datafile


# def _find_pipe_tally_file(pipe_number: Union[str, int], folder_path: str) -> Optional[str]:
#     """Look for PipeTally{pipe_number}.csv in the pipe folder."""
#     pn = str(pipe_number)
#     # Look specifically in the pipe folder for PipeTally{pipe_number}.csv
#     tally_path = f"{folder_path}/PipeTally{pn}.csv"
#     if os.path.exists(tally_path):
#         return tally_path
    
#     # Fallback: look for any PipeTally file in the folder
#     patterns = [
#         f"{folder_path}/*PipeTally*{pn}*.csv",
#         f"{folder_path}/*PipeTally*.csv",
#         f"{folder_path}/*Pipe_Tally*.csv"
#     ]
#     for pat in patterns:
#         hits = glob(pat)
#         if hits:
#             return hits[0]
#     return None

# def _pick_col(df: pd.DataFrame, preferred: List[str], tokens: List[str]) -> Optional[str]:
#     """Pick a column by exact name (case-insensitive) or by 'contains all tokens'."""
#     cols = list(df.columns)
#     low = {c.lower(): c for c in cols}

#     for name in preferred:
#         nlow = name.lower()
#         if nlow in low:
#             return low[nlow]

#     for c in cols:
#         cl = c.lower()
#         if all(t in cl for t in tokens):
#             return c
#     return None

# def _parse_ori_to_seconds(v) -> Optional[int]:
#     """Convert '8', '8.5', '8:30', '08:30:00' → seconds on a 12h dial."""
#     if v is None or (isinstance(v, float) and np.isnan(v)):
#         return None

#     if isinstance(v, (int, float)):
#         h = int(v) % 12
#         m = int(round((float(v) - int(v)) * 60))
#         return h * 3600 + m * 60

#     s = str(v).strip().lower()
#     s = re.sub(r"[^0-9:.]", "", s)
#     if not s:
#         return None

#     if ":" in s:
#         parts = s.split(":")
#         try:
#             h = int(parts[0]) % 12
#             m = int(parts[1]) if len(parts) > 1 else 0
#             sec = int(parts[2]) if len(parts) > 2 else 0
#             return h * 3600 + m * 60 + sec
#         except Exception:
#             return None

#     try:
#         f = float(s)
#         h = int(f) % 12
#         m = int(round((f - int(f)) * 60))
#         return h * 3600 + m * 60
#     except Exception:
#         pass

#     try:
#         h = int(s) % 12
#         return h * 3600
#     except Exception:
#         return None

# def _hhmmss_to_seconds(t: str) -> int:
#     h, m, s = [int(x) for x in str(t).split(":")]
#     return (h % 12) * 3600 + m * 60 + s

# def _nearest_band_label(seconds: int, band_labels: List[str]) -> str:
#     band_labels_str = [str(x) for x in band_labels]
#     band_secs = np.array([_hhmmss_to_seconds(lbl) for lbl in band_labels_str], dtype=int)
#     idx = int(np.argmin(np.abs(band_secs - seconds)))
#     return band_labels_str[idx]

# def _load_overlay_points_for_pipe(pipe_number, y_band_labels, folder_path, *, debug_prefix="OVERLAY DEBUG"):
#     """Return (xs, ys, labels) for overlay markers if a PipeTally file is found.
#        Only plot rows where Feature Type == 'Metal Loss' (case-insensitive).
#        Skip any row missing usable x or orientation.
#     """
#     path = _find_pipe_tally_file(pipe_number, folder_path)
#     if not path:
#         print(f"{debug_prefix}: pipe {pipe_number}: no PipeTally file found.")
#         return None

#     try:
#         df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
#     except Exception as e:
#         print(f"{debug_prefix}: pipe {pipe_number}: failed reading '{path}': {e}")
#         return None

#     x_col    = _pick_col(df, ["Abs. Distance (m)", "Absolute Distance"], ["abs", "distance"]) \
#             or _pick_col(df, [], ["distance"])
#     ori_col  = _pick_col(df, ["Orientation o' clock", "Orientation", "Ori"], ["ori"]) \
#             or _pick_col(df, [], ["orient"])
#     feat_col = _pick_col(df, ["Feature Type", "Feature"], ["feature", "type"])
#     pipe_col = _pick_col(df, ["Pipe Number", "Pipe"], ["pipe", "number"])
#     sno_col  = _pick_col(df, ["s_no", "S_No", "Serial Number", "SNo"], ["s_no", "sno", "serial"])

#     print(f"{debug_prefix}: pipe {pipe_number}: file='{path}' cols: x='{x_col}', ori='{ori_col}', feature='{feat_col}', pipe='{pipe_col}', s_no='{sno_col}'")

#     # Must have x and orientation columns to plot
#     if x_col is None or ori_col is None:
#         print(f"{debug_prefix}: pipe {pipe_number}: missing required columns; no overlay.")
#         return None

#     # If there is a 'Pipe Number' column, restrict to this pipe (best-effort)
#     if pipe_col is not None:
#         mask = df[pipe_col].astype(str).str.contains(str(pipe_number), na=False)
#         if mask.any():
#             df = df[mask]

#     total_rows = len(df)

#     # Feature filtering: ONLY "Metal Loss" (case-insensitive, tolerant to spacing)
#     # If feat_col doesn't exist, we bail (since we must only plot Metal Loss)
#     if feat_col is None:
#         print(f"{debug_prefix}: pipe {pipe_number}: no feature column; overlays disabled (Metal Loss only).")
#         return None

#     feat_series = df[feat_col].astype(str).str.strip().str.lower()
#     metal_loss_mask = feat_series.str.fullmatch(r"metal\s*loss", case=False, na=False)
#     df = df[metal_loss_mask]

#     after_filter = len(df)

#     xs, ys, labels = [], [], []
#     skipped_no_x, skipped_no_ori, skipped_other = 0, 0, 0

#     for _, row in df.iterrows():
#         # x must be a valid number
#         x = pd.to_numeric(row.get(x_col), errors="coerce")
#         if pd.isna(x):
#             skipped_no_x += 1
#             continue

#         # orientation must parse
#         ori_sec = _parse_ori_to_seconds(row.get(ori_col))
#         if ori_sec is None:
#             skipped_no_ori += 1
#             continue

#         # find nearest y-band
#         y = _nearest_band_label(int(ori_sec), list(y_band_labels))
#         if y not in y_band_labels:
#             skipped_other += 1
#             continue

#         # label: prefer s_no; fallback to Defect_id; fallback to running index
#         if sno_col is not None:
#             lbl = row.get(sno_col)
#             lbl = str(lbl).strip() if (lbl is not None and str(lbl).strip() != "" and not pd.isna(lbl)) else None
#         else:
#             lbl = None

#         if lbl is None:
#             lbl2 = row.get("Defect_id")
#             lbl = str(lbl2).strip() if (lbl2 is not None and str(lbl2).strip() != "" and not pd.isna(lbl2)) else str(len(labels) + 1)

#         xs.append(float(x))
#         ys.append(str(y))
#         labels.append(lbl)

#     print(
#         f"{debug_prefix}: pipe {pipe_number}: total_rows={total_rows}, "
#         f"after_feature='Metal Loss'={after_filter}, plotted={len(xs)}, "
#         f"skipped_no_x={skipped_no_x}, skipped_no_ori={skipped_no_ori}, skipped_other={skipped_other}"
#     )

#     if not xs:
#         return None
#     return xs, ys, labels



# def pre_process_for_interactive_heatmap(df_in: pd.DataFrame, datafile: pd.DataFrame, test_val: pd.DataFrame, map_ori_sens: pd.DataFrame):
#     """Process data specifically for interactive heatmap - percentage-based."""
    
#     # Get sensor columns (use raw data, not the processed df_new_tab9)
#     expected = [f"F{i}H{j}" for i in range(1, 25) for j in range(1, 5)]
#     sens_cols = [c for c in expected if c in datafile.columns]
#     if not sens_cols:
#         raise ValueError("No F*H* sensor columns found in the input DataFrame.")

#     df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
#     df_sens_raw = df_sens.copy(deep=True)  # Keep raw copy for CSV export
#     df_mean_cols = df_sens_raw[[f"F{i}H{j}" for i in range(1, 25) for j in range(1, 5)]]
#     Mean1 = df_mean_cols.mean()
#     df_raw_plot = ((df_mean_cols - Mean1)/Mean1)*100
    
#     # Convert all sensor data to numeric
#     for col in sens_cols:
#         df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    
#     # Fill NaN values with forward fill, then zeros
#     df_sens = df_sens.fillna(method='ffill').fillna(0.0)
    
#     # Calculate mean for percentage normalization
#     Mean1 = df_sens.mean()
    
#     # Normalize as percentage deviation from mean
#     df_sens_normalized = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100
    
#     # Zero out values above mean threshold (different from main processing)
#     for col in sens_cols:
#         df_sens_normalized.loc[df_sens_normalized[col] > Mean1[col], col] = 0
    
#     # Apply additional filtering if needed
#     if UPPER_SENS_MUL > 0 and LOWER_SENS_MUL > 0:
#         sens_std = df_sens.std(axis=0, skipna=True)
#         upper = Mean1 + UPPER_SENS_MUL * sens_std
#         lower = Mean1 - LOWER_SENS_MUL * sens_std
        
#         for col in sens_cols:
#             mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
#             df_sens_normalized.loc[mask, col] = 0

#     # Rearrange data to match test_val structure using map_ori_sens
#     df_plot_rearranged = test_val.copy()
#     for r in range(len(test_val)):
#         for c, band in enumerate(test_val.columns):
#             # Get the sensor mapping from map_ori_sens
#             sensor_info = map_ori_sens.iloc[r, c]
#             if isinstance(sensor_info, tuple) and len(sensor_info) >= 2:
#                 sensor_col = sensor_info[1]  # sensor column name
#                 if sensor_col in df_sens_normalized.columns:
#                     df_plot_rearranged.iloc[r, c] = df_sens_normalized.loc[r, sensor_col]
    
#     return df_plot_rearranged, df_raw_plot

# def save_interactive_heatmap(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10):
#     """Create and save interactive heatmap with overlays."""
    
#     # Use the specialized preprocessing for interactive heatmap
#     df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)
    
#     # Get x-axis values
#     if 'ODDO1' in datafile.columns:
#         x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
#         x_label = 'Absolute Distance (m)'
#     else:
#         x_vals = pd.Series(np.arange(len(datafile)))
#         x_label = 'Index'
    
#     # Get y-band labels
#     y_bands = [str(c) for c in test_val.columns]
    
#     # Create heatmap data - transpose so y_bands are on y-axis
#     heatmap_data = df_raw_plot.T
    
#     # Ensure all data is numeric
#     for col in heatmap_data.columns:
#         heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
#     heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    
#     # Replace infinite values
#     if not np.isfinite(heatmap_data.values).all():
#         heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    
#     # Create the interactive heatmap using Plotly
#     fig = go.Figure(data=go.Heatmap(
#         z=heatmap_data.values,
#         x=x_vals.round(2),
#         y=y_bands,
#         colorscale='jet',
#         zmin=-3,
#         zmax=8,
#         colorbar=dict(title="Sensor Value (%)"),
#         hoverongaps=False,
#         hovertemplate='<b>%{x}</b><br>' +
#                      '<b>%{y}</b><br>' +
#                      '<b>Value: %{z:.2f}%</b>' +
#                      '<extra></extra>'
#     ))
    
#     # Add overlay points with FIXED SIZE SQUARES
#     overlay_added = False
#     pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
#     if pts is not None:
#         xs, ys, labels = pts
        
#         for x, y_band, label in zip(xs, ys, labels):
#             if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
#                 y_idx = y_bands.index(y_band)
                
#                 # Fixed size square
#                 fig.add_shape(
#                     type="rect",
#                     x0=x - 0.05, y0=y_idx - 0.35,
#                     x1=x + 0.05, y1=y_idx + 0.35,
#                     line=dict(color="black", width=2),
#                     fillcolor="rgba(255,0,0,0.6)"
#                 )
                
#                 # Add label annotation
#                 fig.add_annotation(
#                     x=x, y=y_idx,
#                     text=label,
#                     showarrow=False,
#                     font=dict(color="white", size=8, family="Arial Black"),
#                     bgcolor="red",
#                     bordercolor="black",
#                     borderwidth=1
#                 )
#                 overlay_added = True
    
#     # Update layout
#     fig.update_layout(
#         title=f"Interactive Plotly Heatmap — Pipe {pipe_number}",
#         xaxis_title=x_label,
#         yaxis_title="Orientation (12h bands)",
#         width=1400,
#         height=700,
#         font=dict(size=12),
#         xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#         yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
#     )
    
#     # Save the interactive heatmap
#     fig.write_html(
#         f'{folder_path}/heatmap{pipe_number}.html',
#         full_html=True,
#         config={
#             'displayModeBar': True,
#             'displaylogo': False,
#             'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
#             'toImageButtonOptions': {
#                 'format': 'png',
#                 'filename': f'heatmap_pipe_{pipe_number}',
#                 'height': 700,
#                 'width': 1400,
#                 'scale': 2
#             }
#         },
#         include_plotlyjs=True,
#         auto_open=False
#     )
    
#     print(f"Saved interactive heatmap: {folder_path}/interactive_heatmap{pipe_number}.html")
#     print(f"Overlays: {'Yes' if overlay_added else 'None found'}")


# def create_plots(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original):
#     folder_path = f'{output_folder}/Pipe_{pipe_number}'
#     os.makedirs(folder_path, exist_ok=True)

#     # Heatmap
#     # save_heatmap(test_val, datafile, map_ori_sens, folder_path, pipe_number)

#     # MultilinePlot (offset stack of sensors)
#     save_lineplot(folder_path, test_val, datafile, pipe_number)

#     # 3D Pipe 
#     save_pipe3d(test_val, test_val, folder_path, pipe_number)

#     # inside create_plots(...) or wherever you save other charts:
#     save_proximity_linechart(folder_path, datafile, pipe_number)

#     save_interactive_heatmap(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10)


# def save_heatmap(test_val, datafile, map_ori_sens, folder_path, pipe_number):
#     fighm = go.Figure(data=go.Heatmap(
#         z=test_val.T,
#         y=test_val.columns,
#         x=(datafile['ODDO1'] / 1000).round(2),
#         colorscale='jet',
#         hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
#         text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],
#     ))
#     fighm.update_layout(
#         xaxis_title='Absolute Distance (m)',
#         height=500,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20)
#     )
#     fighm.write_html(f'{folder_path}/heatmap{pipe_number}.html', auto_open=False)


# # case-insensitive F#P# like F1P1, F12P3
# FP_PATTERN = re.compile(r'^F\d{1,2}P\d{1,2}$', re.IGNORECASE)

# def save_proximity_linechart(
#     folder_path: str,
#     datafile: pd.DataFrame,
#     pipe_number,
#     *,
#     offset_step: float = 0.10,
#     dtick: int = 1000,
#     x_pref: str = "auto"  # "auto" -> ODDO1 if available, else index
# ):
#     """
#     Proximity linechart:
#       - selects columns matching F#P# (case-insensitive),
#       - forward-fills data,
#       - MinMax scales each series to [0,1],
#       - offsets each series by `offset_step`,
#       - X-axis = ODDO1 (meters) if present (or if x_pref='oddo1'), else index,
#       - saves HTML to {folder_path}/proximity_linechart{pipe_number}.html
#     """
#     df = datafile.copy()

#     # 1) collect F*P* columns
#     candidates = [c for c in df.columns if isinstance(c, str) and FP_PATTERN.match(c.strip())]
#     if not candidates:
#         print(f"No F#P# columns found for pipe {pipe_number}. Skipping proximity linechart.")
#         return

#     # 2) ensure numeric (coerce where possible)
#     res_cols = []
#     for c in candidates:
#         if not is_numeric_dtype(df[c]):
#             coerced = pd.to_numeric(df[c], errors='coerce')
#             if coerced.notna().any():
#                 df[c] = coerced
#         if is_numeric_dtype(df[c]):
#             res_cols.append(c)
#     if not res_cols:
#         print(f"No numeric F#P# columns for pipe {pipe_number}. Skipping proximity linechart.")
#         return

#     # 3) forward-fill
#     df1 = df.fillna(method='ffill')

#     # 4) choose x-axis (ODDO1 -> index)
#     if x_pref.lower() == "oddo1" or (x_pref == "auto" and "ODDO1" in df1.columns):
#         x_vals = (pd.to_numeric(df1["ODDO1"], errors="coerce") / 1000.0).round(3)
#         x_label = "Abs. Distance (m) — ODDO1"
#     else:
#         x_vals = df1.index
#         x_label = "Index"

#     # 5) MinMax scale selected columns
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(df1[res_cols].to_numpy())
#     df1.loc[:, res_cols] = scaled

#     # 6) figure with offsets
#     fig = go.Figure()
#     for i, col in enumerate(res_cols):
#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=df1[col] + i * offset_step,
#             name=col,
#             mode='lines',
#             line=dict(width=1),
#             hoverinfo='x+y+name',
#             showlegend=True
#         ))

#     # 7) styling + axis titles + gridlines
#     fig.update_layout(
#         title="Proximity Sensor Line Chart",
#         width=1600,
#         height=650,
#         margin=dict(l=140, b=80),
#         paper_bgcolor="#ffffff",
#         plot_bgcolor='rgb(255, 255, 255)',
#         title_x=0.5,
#         font={"family": "courier"},
#         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
#         xaxis_title=x_label,
#         yaxis_title="Scaled Proximity Sensor (0–1, offset)",
#     )
#     # keep titles close to axes
#     fig.update_xaxes(title_standoff=8, automargin=True, dtick=dtick)
#     fig.update_yaxes(title_standoff=10, automargin=True)

#     # light major gridlines
#     fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)", gridwidth=1, zeroline=False)
#     fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", gridwidth=1, zeroline=False)

#     # optional minor gridlines (comment out if your Plotly build lacks support)
#     fig.update_xaxes(minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=0.5))
#     fig.update_yaxes(minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", gridwidth=0.5))

#     # 8) save EXACTLY as requested
#     fig.write_html(f'{folder_path}/proximity_linechart{pipe_number}.html', auto_open=False)

#     print(f"Saved {folder_path}/proximity_linechart{pipe_number}.html")



# def save_heatmap_raw(folder_path, df_raw_straight, dataf, pipe_number):
#     initial_read = INITIAL_READ
#     roll = dataf['ROLL'] - initial_read

#     def degrees_to_hours_minutes2(degrees):
#         if (degrees < 0):
#             degrees = degrees % 360
#         elif degrees >= 360:
#             degrees %= 360
#         degrees_per_second = 360 / (12 * 60 * 60)
#         total_seconds = degrees / degrees_per_second
#         hours = int(total_seconds // 3600)
#         minutes = int((total_seconds % 3600) // 60)
#         seconds = int(total_seconds % 60)
#         return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

#     def add_sensor_keys(d):
#         for e in d:
#             new_dict = {**e}
#             for i in range(1, 96):
#                 new_dict[f'Roll_Sensor_{i}'] = e['Roll_Sensor_0'] + (3.75 * i)
#             yield new_dict

#     def check_time_range(time_str):
#         start_time = '00:00:00'
#         end_time = '00:07:29'
#         time_to_check = datetime.strptime(time_str, '%H:%M:%S')
#         start_time_dt = datetime.strptime(start_time, '%H:%M:%S')
#         end_time_dt = datetime.strptime(end_time, '%H:%M:%S')
#         return start_time_dt <= time_to_check <= end_time_dt

#     d = []
#     for pos in roll:
#         d.append({f"Roll_Sensor_0": pos})

#     upd_d = list(add_sensor_keys(d))
#     oriData = pd.DataFrame.from_dict(data=upd_d)
#     clockData = oriData.applymap(degrees_to_hours_minutes2)

#     test_clockData = clockData.copy()
#     test_clockData = test_clockData.apply(pd.to_datetime, format='%H:%M:%S')
#     test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M:%S'))
#     test_clockData = test_clockData.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
#     test_clockData = test_clockData.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
#     test_clockData = test_clockData.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

#     pat1 = re.compile(r'^F\d+H\d+$')   # F%H%
#     sel_col = ['ROLL', 'ODDO1']
#     fil_col = [col for col in dataf.columns if (pat1.match(col) or col in sel_col)]
#     refData = dataf[fil_col]
#     sensD = refData[refData.columns[2:]]

#     test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_raw_straight.columns)))

#     def create_time_dict():
#         time_ranges_2 = [
#             '00:00:00', '00:07:30', '00:15:00', '00:22:30', '00:30:00', '00:37:30', '00:45:00', '00:52:30', '01:00:00', '01:07:30', 
#             '01:15:00', '01:22:30', '01:30:00', '01:37:30', '01:45:00', '01:52:30', '02:00:00', '02:07:30', '02:15:00', '02:22:30', 
#             '02:30:00', '02:37:30', '02:45:00', '02:52:30', '03:00:00', '03:07:30', '03:15:00', '03:22:30', '03:30:00', '03:37:30', 
#             '03:45:00', '03:52:30', '04:00:00', '04:07:30', '04:15:00', '04:22:30', '04:30:00', '04:37:30', '04:45:00', '04:52:30', 
#             '05:00:00', '05:07:30', '05:15:00', '05:22:30', '05:30:00', '05:37:30', '05:45:00', '05:52:30', '06:00:00', '06:07:30', 
#             '06:15:00', '06:22:30', '06:30:00', '06:37:30', '06:45:00', '06:52:30', '07:00:00', '07:07:30', '07:15:00', '07:22:30', 
#             '07:30:00', '07:37:30', '07:45:00', '07:52:30', '08:00:00', '08:07:30', '08:15:00', '08:22:30', '08:30:00', '08:37:30', 
#             '08:45:00', '08:52:30', '09:00:00', '09:07:30', '09:15:00', '09:22:30', '09:30:00', '09:37:30', '09:45:00', '09:52:30', 
#             '10:00:00', '10:07:30', '10:15:00', '10:22:30', '10:30:00', '10:37:30', '10:45:00', '10:52:30', '11:00:00', '11:07:30', 
#             '11:15:00', '11:22:30', '11:30:00', '11:37:30', '11:45:00', '11:52:30'
#         ]
#         return {key: [] for key in time_ranges_2}

#     time_dict_1 = create_time_dict()
#     rang = list(time_dict_1.keys())

#     for _, row in test_clockData.iterrows():
#         xl = list(row)
#         xd = dict(row)
#         xkeys = list(xd.keys())
#         c = 0
#         for _, dval in xd.items():
#             if check_time_range(dval):
#                 ind = xl.index(dval)
#                 _ = xl[ind:] + xl[:ind]
#                 break

#         curr = ind
#         while True:
#             ck = xkeys[curr]
#             time_dict_1[rang[c]].append((curr, ck, xd[ck]))
#             c += 1
#             curr = (curr + 1) % len(xkeys)
#             if curr == ind:
#                 break

#     map_ori_sens = pd.DataFrame(time_dict_1)
#     val_ori_sens = map_ori_sens.copy().applymap(lambda cell: cell[1])
#     test_val = val_ori_sens.copy()

#     for r, e in val_ori_sens.iterrows():
#         c = 0
#         for _, tup_value in e.items():
#             cell_v = sensD.at[r, tup_value]
#             test_val.iloc[r, c] = cell_v
#             c += 1
    
#     figraw = go.Figure(data=go.Heatmap(
#         z=test_val.T,
#         y=test_val.columns,
#         x=(dataf['ODDO1'] / 1000).round(2),
#         colorscale='jet',
#         hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
#         text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],
#         zmin=-12000,
#         zmax=30000
#     ))

#     figraw.update_layout(
#         height=300,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20)
#     )

#     figraw.write_html(f'{folder_path}/heatmap_raw{pipe_number}.html', auto_open=False)


# def save_lineplot(folder_path, test_val, datafile, pipe_number):
#     figmlp = go.Figure()
#     offset_step = 1200
#     for idx, col in enumerate(test_val.columns):
#         y_data = test_val[col].values
#         offset_y_data = y_data + (idx * offset_step)
#         figmlp.add_trace(go.Scatter(
#             x=(datafile['ODDO1'] / 1000).round(2),
#             y=offset_y_data,
#             mode='lines',
#             name=col,
#             line=dict(width=1),
#             hoverinfo='x+y+name',
#             showlegend=False
#         ))

#     figmlp.update_layout(
#         template='plotly_white',
#         height=900,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20),
#         yaxis=dict(
#             tickmode='array',
#             tickvals=[idx * offset_step for idx in range(len(test_val.columns))],
#             ticktext=test_val.columns,
#             tickfont=dict(size=8),
#         )
#     )

#     figmlp.write_html(f'{folder_path}/lineplot{pipe_number}.html', auto_open=False)


# def save_lineplot_raw(folder_path, test_val, pipe_number):
#     figmlpraw = go.Figure()
#     for _, col in enumerate(test_val.columns):
#         y_data = test_val[col]
#         figmlpraw.add_trace(go.Scatter(
#             x=test_val.index,
#             y=y_data,
#             mode='lines',
#             name=col,
#             line=dict(width=1),
#             hoverinfo='x+y+name',
#             showlegend=False
#         ))

#     figmlpraw.update_layout(
#         xaxis_title='Counter',
#         template='plotly_white',
#         height=300,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20)
#     )

#     figmlpraw.write_html(f'{folder_path}/lineplot_raw{pipe_number}.html', auto_open=False)


# def save_pipe3d(data, data_cp, folder_path, pipe_number):
#     if not isinstance(data, np.ndarray):
#         data = np.asarray(data)  # test_val

#     num_rows, num_cols = data.shape

#     theta = np.linspace(0, 2 * np.pi, num_cols)
#     z = np.linspace(0, 1, num_rows)
#     theta, z = np.meshgrid(theta, z)

#     radius = 109.5   # OD = 219mm, R = OD/2
#     odometer = num_rows

#     # Cartesian Coords
#     x = odometer * z
#     y = radius * np.cos(theta)
#     zc = radius * np.sin(theta)

#     fig = go.Figure(data=[go.Surface(
#         x=x,
#         y=zc,
#         z=y,
#         surfacecolor=data,
#         colorscale='jet',
#         customdata=data_cp
#     )])

#     camera = dict(eye=dict(x=0., y=5, z=0.), up=dict(x=0, y=1, z=0))

#     odometer_start = 0
#     odometer_end = odometer

#     fig.add_trace(go.Scatter3d(
#         x=[odometer_start, odometer_end], y=[radius, radius], z=[0, 0],
#         text=["3"], mode='text', textposition="middle center",
#         marker=dict(size=0), name="3pm",
#         textfont=dict(size=20, color="#61090c")
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=[odometer_start, odometer_end], y=[-radius, -radius], z=[0, 0],
#         text=["9"], mode='text', textposition="middle center",
#         marker=dict(size=0), name="9pm",
#         textfont=dict(size=20, color="#61090c")
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=[odometer_start, odometer_end], y=[0, 0], z=[radius, radius],
#         text=["6"], mode='text', textposition="middle center",
#         marker=dict(size=0), name="6pm",
#         textfont=dict(size=20, color="#61090c")
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=[odometer_start, odometer_end], y=[0, 0], z=[-radius, -radius],
#         text=["12"], mode='text', textposition="middle center",
#         marker=dict(size=0), name="12pm",
#         textfont=dict(size=20, color="#61090c")
#     ))
   
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='Odometer',
#             yaxis_title='Radial Length',
#             zaxis_title='Radial Length',
#             aspectmode='data',
#             aspectratio=dict(x=1, y=1, z=0.5),
#             camera=camera
#         ),
#         height=500,
#         width=1500,
#         title='Pipe Visualization',
#         margin=dict(l=20, r=20, t=50, b=20),
#     )

#     fig.write_html(f'{folder_path}/pipe3d{pipe_number}.html', auto_open=False)


# # -------------------- PARALLEL HELPERS --------------------
# def _resolve_workers(workers):
#     if workers in (None, 0, -1, "auto"):
#         cpu = os.cpu_count() or 1
#         return max(1, cpu - 1)
#     if isinstance(workers, int):
#         cpu = os.cpu_count() or 1
#         return max(1, min(workers, cpu))
#     return 1


# def _process_one_pkl(pkl_path, output_folder):
#     try:
#         pipe_number = Path(pkl_path).stem
#         pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
#         pipe_folder.mkdir(exist_ok=True)

#         data = pd.read_pickle(pkl_path)
#         dfile = pre_process_data(data, pipe_number, output_folder)

#         # Save the Excel
#         xlsx_path = pipe_folder / f"Pipe_{pipe_number}.xlsx"
#         dfile.to_excel(xlsx_path, index=False)

#         return f"Processed {os.path.basename(pkl_path)} and saved to {pipe_folder}"
#     except Exception as e:
#         return f"Error loading {os.path.basename(pkl_path)}: {e}"


# def create_html_and_csv_from_pkl(
#     pkl_folder='pipes3',
#     output_folder='Client_Pipes',
#     output_callback=None,
#     workers=WORKERS
# ):
#     Path(output_folder).mkdir(exist_ok=True)

#     # collect .pkl file paths
#     pkl_paths = [
#         str(Path(pkl_folder) / f)
#         for f in os.listdir(pkl_folder)
#         if f.lower().endswith('.pkl')
#     ]

#     if not pkl_paths:
#         msg = f"No .pkl files found in {pkl_folder}"
#         if output_callback: output_callback(msg)
#         else: print(msg)
#         return

#     n_jobs = _resolve_workers(workers)

#     # fan out work across processes
#     results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
#         delayed(_process_one_pkl)(p, output_folder) for p in pkl_paths
#     )

#     # report
#     for msg in results:
#         if output_callback: output_callback(msg)
#         else: print(msg)


# # -------------------- MAIN --------------------
# if __name__ == "__main__":
#     import time
#     st = time.time()
#     # override cores at runtime if needed, e.g., workers=4
#     create_html_and_csv_from_pkl(workers=WORKERS)
#     print(f'Total time: {time.time()-st} seconds')




























'''
This code will input the raw pkl file( {x}.pkl ) and generate out the plots to:
Pipe_{x}  with ending format as <Name_{x}.html>
'''

import os
# Limit intra-process math threads to avoid over-subscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
from pathlib import Path
import numpy as np
import re
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import datetime
from joblib import Parallel, delayed
import warnings
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
from typing import Optional, Union, List
from glob import glob

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------ Plotly export options (keep interactive + smaller HTML) ------------
PLOTLY_JS_MODE   = "directory"   # "directory" (offline, small HTML) or "cdn" (requires internet)
PLOTLY_COMPRESS  = True          # compress embedded data
HTML_DEFAULT_W   = "100%"        # let the container control size
HTML_DEFAULT_H   = 700           # or "100%" if you want full-height wrappers

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d"
    ],
    # , "zoomIn2d", "zoomOut2d",
    #     "pan2d", "toImage", "hoverCompareCartesian", "hoverClosestCartesian"
    "displayModeBar": True, 
    "scrollZoom": True,
}

def write_plotly_html(fig, out_path: str):
    common = dict(
        include_plotlyjs=PLOTLY_JS_MODE,
        full_html=True,
        config=PLOTLY_CONFIG,
        auto_open=False,
        default_width=HTML_DEFAULT_W,
        default_height=HTML_DEFAULT_H,
    )
    try:
        fig.write_html(out_path, compress_data=PLOTLY_COMPRESS, **common)
    except TypeError:
        fig.write_html(out_path, **common)


# -------------------- CONFIG --------------------
INITIAL_READ = 0.0      # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3
# -1 / 0 / None / "auto" => auto (CPU-1, at least 1). Or set an int, e.g. 4
WORKERS = 4
# ------------------------------------------------


def pre_process_data(datafile, pipe_number, output_folder):

    datafile_original = datafile.copy(deep=True)
    # Create DataFrame from raw data
    df_new_tab9 = pd.DataFrame(
        datafile,
        columns=[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)]
    )

    df_new_tab10 = df_new_tab9.copy()
    sensor_columns = df_new_tab9.columns.tolist()

    # Denoising using Savitzky-Golay filter
    window_length = 15
    polyorder = 2
    for col in sensor_columns:
        data = df_new_tab9[col].values
        time_index = np.arange(len(df_new_tab9))
        trend = np.polyval(np.polyfit(time_index, data, 2), time_index)
        data_denoised = savgol_filter(data - trend, window_length, polyorder)
        df_new_tab9[col] = data_denoised

    df_raw_straight = df_new_tab9.copy()
    
    # Setting bounds and applying conditions
    sens_mean = df_new_tab9.abs().mean()
    standard_deviation = df_new_tab9.std(axis=0, skipna=True)

    mean_plus_sigma = sens_mean + UPPER_SENS_MUL * standard_deviation
    mean_negative_sigma = sens_mean - LOWER_SENS_MUL * standard_deviation

    # Apply noise filtering to zero-out in-bound values
    for col in df_new_tab9.columns:
        if col in mean_plus_sigma.index and col in mean_negative_sigma.index:
            df_new_tab9[col] = np.where(
                (df_new_tab9[col] >= mean_negative_sigma[col]) &
                (df_new_tab9[col] <= mean_plus_sigma[col]),
                0,
                df_new_tab9[col]
            )

    initial_read = INITIAL_READ
    roll = datafile['ROLL'] - initial_read
    # odoData = datafile['ODDO1']  # kept if needed later

    def degrees_to_hours_minutes2(degrees):
        if (degrees < 0):
            degrees = degrees % 360
        elif degrees >= 360:
            degrees %= 360
        degrees_per_second = 360 / (12 * 60 * 60)
        total_seconds = degrees / degrees_per_second
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def add_sensor_keys(d):
        for e in d:
            new_dict = {**e}
            for i in range(1, 96):
                new_dict[f'Roll_Sensor_{i}'] = e['Roll_Sensor_0'] + (3.75 * i)
            yield new_dict

    def check_time_range(time_str):
        start_time = '00:00:00'
        end_time = '00:07:29'
        time_to_check = datetime.strptime(time_str, '%H:%M:%S')
        start_time_dt = datetime.strptime(start_time, '%H:%M:%S')
        end_time_dt = datetime.strptime(end_time, '%H:%M:%S')
        return start_time_dt <= time_to_check <= end_time_dt

    d = []
    for pos in roll:
        d.append({f"Roll_Sensor_0": pos})

    upd_d = list(add_sensor_keys(d))
    oriData = pd.DataFrame.from_dict(data=upd_d)
    clockData = oriData.applymap(degrees_to_hours_minutes2)

    test_clockData = clockData.copy()
    test_clockData = test_clockData.apply(pd.to_datetime, format='%H:%M:%S')
    test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M:%S'))
    test_clockData = test_clockData.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
    test_clockData = test_clockData.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
    test_clockData = test_clockData.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

    test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_new_tab9.columns)))

    def create_time_dict():
        time_ranges_2 = [
            '00:00:00', '00:07:30', '00:15:00', '00:22:30', '00:30:00', '00:37:30', '00:45:00', '00:52:30', '01:00:00', '01:07:30', 
            '01:15:00', '01:22:30', '01:30:00', '01:37:30', '01:45:00', '01:52:30', '02:00:00', '02:07:30', '02:15:00', '02:22:30', 
            '02:30:00', '02:37:30', '02:45:00', '02:52:30', '03:00:00', '03:07:30', '03:15:00', '03:22:30', '03:30:00', '03:37:30', 
            '03:45:00', '03:52:30', '04:00:00', '04:07:30', '04:15:00', '04:22:30', '04:30:00', '04:37:30', '04:45:00', '04:52:30', 
            '05:00:00', '05:07:30', '05:15:00', '05:22:30', '05:30:00', '05:37:30', '05:45:00', '05:52:30', '06:00:00', '06:07:30', 
            '06:15:00', '06:22:30', '06:30:00', '06:37:30', '06:45:00', '06:52:30', '07:00:00', '07:07:30', '07:15:00', '07:22:30', 
            '07:30:00', '07:37:30', '07:45:00', '07:52:30', '08:00:00', '08:07:30', '08:15:00', '08:22:30', '08:30:00', '08:37:30', 
            '08:45:00', '08:52:30', '09:00:00', '09:07:30', '09:15:00', '09:22:30', '09:30:00', '09:37:30', '09:45:00', '09:52:30', 
            '10:00:00', '10:07:30', '10:15:00', '10:22:30', '10:30:00', '10:37:30', '10:45:00', '10:52:30', '11:00:00', '11:07:30', 
            '11:15:00', '11:22:30', '11:30:00', '11:37:30', '11:45:00', '11:52:30'
        ]
        return {key: [] for key in time_ranges_2}

    time_dict_1 = create_time_dict()
    rang = list(time_dict_1.keys())

    for _, row in test_clockData.iterrows():
        xl = list(row)
        xd = dict(row)
        xkeys = list(xd.keys())
        c = 0
        for _, dval in xd.items():
            if check_time_range(dval):
                ind = xl.index(dval)
                _ = xl[ind:] + xl[:ind]  # not used later but kept for clarity
                break

        curr = ind
        while True:
            ck = xkeys[curr]
            time_dict_1[rang[c]].append((curr, ck, xd[ck]))
            c += 1
            curr = (curr + 1) % len(xkeys)
            if curr == ind:
                break

    map_ori_sens = pd.DataFrame(time_dict_1)

    val_ori_sens = map_ori_sens.copy()

    def extract_string(cell):
        return cell[1]

    val_ori_sens = val_ori_sens.applymap(extract_string)

    test_val = val_ori_sens.copy()

    for r, e in val_ori_sens.iterrows():
        c = 0
        for _, tup_value in e.items():
            cell_v = df_new_tab9.at[r, tup_value]
            test_val.iloc[r, c] = cell_v
            c += 1

    # map_val_sens is created but not used later; keeping for parity
    map_val_sens = pd.DataFrame(index=test_val.index, columns=test_val.columns)
    for column in test_val.columns:
        for i in range(test_val.shape[0]):
            map_value = map_ori_sens.loc[i, column]
            test_value = test_val.loc[i, column]
            map_val_sens.loc[i, column] = (*map_value, test_value)
    
    create_plots(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original)
    return datafile


def _find_pipe_tally_file(pipe_number: Union[str, int], folder_path: str) -> Optional[str]:
    """Look for PipeTally{pipe_number}.csv in the pipe folder."""
    pn = str(pipe_number)
    # Look specifically in the pipe folder for PipeTally{pipe_number}.csv
    tally_path = f"{folder_path}/PipeTally{pn}.csv"
    if os.path.exists(tally_path):
        return tally_path
    
    # Fallback: look for any PipeTally file in the folder
    patterns = [
        f"{folder_path}/*PipeTally*{pn}*.csv",
        f"{folder_path}/*PipeTally*.csv",
        f"{folder_path}/*Pipe_Tally*.csv"
    ]
    for pat in patterns:
        hits = glob(pat)
        if hits:
            return hits[0]
    return None

def _pick_col(df: pd.DataFrame, preferred: List[str], tokens: List[str]) -> Optional[str]:
    """Pick a column by exact name (case-insensitive) or by 'contains all tokens'."""
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}

    for name in preferred:
        nlow = name.lower()
        if nlow in low:
            return low[nlow]

    for c in cols:
        cl = c.lower()
        if all(t in cl for t in tokens):
            return c
    return None

def _parse_ori_to_seconds(v) -> Optional[int]:
    """Convert '8', '8.5', '8:30', '08:30:00' → seconds on a 12h dial."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None

    if isinstance(v, (int, float)):
        h = int(v) % 12
        m = int(round((float(v) - int(v)) * 60))
        return h * 3600 + m * 60

    s = str(v).strip().lower()
    s = re.sub(r"[^0-9:.]", "", s)
    if not s:
        return None

    if ":" in s:
        parts = s.split(":")
        try:
            h = int(parts[0]) % 12
            m = int(parts[1]) if len(parts) > 1 else 0
            sec = int(parts[2]) if len(parts) > 2 else 0
            return h * 3600 + m * 60 + sec
        except Exception:
            return None

    try:
        f = float(s)
        h = int(f) % 12
        m = int(round((f - int(f)) * 60))
        return h * 3600 + m * 60
    except Exception:
        pass

    try:
        h = int(s) % 12
        return h * 3600
    except Exception:
        return None

def _hhmmss_to_seconds(t: str) -> int:
    h, m, s = [int(x) for x in str(t).split(":")]
    return (h % 12) * 3600 + m * 60 + s

def _nearest_band_label(seconds: int, band_labels: List[str]) -> str:
    band_labels_str = [str(x) for x in band_labels]
    band_secs = np.array([_hhmmss_to_seconds(lbl) for lbl in band_labels_str], dtype=int)
    idx = int(np.argmin(np.abs(band_secs - seconds)))
    return band_labels_str[idx]

def _load_overlay_points_for_pipe(pipe_number, y_band_labels, folder_path, *, debug_prefix="OVERLAY DEBUG"):
    """Return (xs, ys, labels) for overlay markers if a PipeTally file is found.
       Only plot rows where Feature Type == 'Metal Loss' (case-insensitive).
       Skip any row missing usable x or orientation.
    """
    path = _find_pipe_tally_file(pipe_number, folder_path)
    if not path:
        print(f"{debug_prefix}: pipe {pipe_number}: no PipeTally file found.")
        return None

    try:
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    except Exception as e:
        print(f"{debug_prefix}: pipe {pipe_number}: failed reading '{path}': {e}")
        return None

    x_col    = _pick_col(df, ["Abs. Distance (m)", "Absolute Distance"], ["abs", "distance"]) \
            or _pick_col(df, [], ["distance"])
    ori_col  = _pick_col(df, ["Orientation o' clock", "Orientation", "Ori"], ["ori"]) \
            or _pick_col(df, [], ["orient"])
    feat_col = _pick_col(df, ["Feature Type", "Feature"], ["feature", "type"])
    pipe_col = _pick_col(df, ["Pipe Number", "Pipe"], ["pipe", "number"])
    sno_col  = _pick_col(df, ["s_no", "S_No", "Serial Number", "SNo"], ["s_no", "sno", "serial"])

    print(f"{debug_prefix}: pipe {pipe_number}: file='{path}' cols: x='{x_col}', ori='{ori_col}', feature='{feat_col}', pipe='{pipe_col}', s_no='{sno_col}'")

    # Must have x and orientation columns to plot
    if x_col is None or ori_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: missing required columns; no overlay.")
        return None

    # If there is a 'Pipe Number' column, restrict to this pipe (best-effort)
    if pipe_col is not None:
        mask = df[pipe_col].astype(str).str.contains(str(pipe_number), na=False)
        if mask.any():
            df = df[mask]

    total_rows = len(df)

    # Feature filtering: ONLY "Metal Loss" (case-insensitive, tolerant to spacing)
    # If feat_col doesn't exist, we bail (since we must only plot Metal Loss)
    if feat_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: no feature column; overlays disabled (Metal Loss only).")
        return None

    feat_series = df[feat_col].astype(str).str.strip().str.lower()
    metal_loss_mask = feat_series.str.fullmatch(r"metal\s*loss", case=False, na=False)
    df = df[metal_loss_mask]

    after_filter = len(df)

    xs, ys, labels = [], [], []
    skipped_no_x, skipped_no_ori, skipped_other = 0, 0, 0

    for _, row in df.iterrows():
        # x must be a valid number
        x = pd.to_numeric(row.get(x_col), errors="coerce")
        if pd.isna(x):
            skipped_no_x += 1
            continue

        # orientation must parse
        ori_sec = _parse_ori_to_seconds(row.get(ori_col))
        if ori_sec is None:
            skipped_no_ori += 1
            continue

        # find nearest y-band
        y = _nearest_band_label(int(ori_sec), list(y_band_labels))
        if y not in y_band_labels:
            skipped_other += 1
            continue

        # label: prefer s_no; fallback to Defect_id; fallback to running index
        if sno_col is not None:
            lbl = row.get(sno_col)
            lbl = str(lbl).strip() if (lbl is not None and str(lbl).strip() != "" and not pd.isna(lbl)) else None
        else:
            lbl = None

        if lbl is None:
            lbl2 = row.get("Defect_id")
            lbl = str(lbl2).strip() if (lbl2 is not None and str(lbl2).strip() != "" and not pd.isna(lbl2)) else str(len(labels) + 1)

        xs.append(float(x))
        ys.append(str(y))
        labels.append(lbl)

    print(
        f"{debug_prefix}: pipe {pipe_number}: total_rows={total_rows}, "
        f"after_feature='Metal Loss'={after_filter}, plotted={len(xs)}, "
        f"skipped_no_x={skipped_no_x}, skipped_no_ori={skipped_no_ori}, skipped_other={skipped_other}"
    )

    if not xs:
        return None
    return xs, ys, labels



def pre_process_for_interactive_heatmap(df_in: pd.DataFrame, datafile: pd.DataFrame, test_val: pd.DataFrame, map_ori_sens: pd.DataFrame):
    """Process data specifically for interactive heatmap - percentage-based."""
    
    # Get sensor columns (use raw data, not the processed df_new_tab9)
    expected = [f"F{i}H{j}" for i in range(1, 25) for j in range(1, 5)]
    sens_cols = [c for c in expected if c in datafile.columns]
    if not sens_cols:
        raise ValueError("No F*H* sensor columns found in the input DataFrame.")

    df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
    df_sens_raw = df_sens.copy(deep=True)  # Keep raw copy for CSV export
    df_mean_cols = df_sens_raw[[f"F{i}H{j}" for i in range(1, 25) for j in range(1, 5)]]
    Mean1 = df_mean_cols.mean()
    df_raw_plot = ((df_mean_cols - Mean1)/Mean1)*100
    
    # Convert all sensor data to numeric
    for col in sens_cols:
        df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    
    # Fill NaN values with forward fill, then zeros
    df_sens = df_sens.fillna(method='ffill').fillna(0.0)
    
    # Calculate mean for percentage normalization
    Mean1 = df_sens.mean()
    
    # Normalize as percentage deviation from mean
    df_sens_normalized = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100
    
    # Zero out values above mean threshold (different from main processing)
    for col in sens_cols:
        df_sens_normalized.loc[df_sens_normalized[col] > Mean1[col], col] = 0
    
    # Apply additional filtering if needed
    if UPPER_SENS_MUL > 0 and LOWER_SENS_MUL > 0:
        sens_std = df_sens.std(axis=0, skipna=True)
        upper = Mean1 + UPPER_SENS_MUL * sens_std
        lower = Mean1 - LOWER_SENS_MUL * sens_std
        
        for col in sens_cols:
            mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
            df_sens_normalized.loc[mask, col] = 0

    # Rearrange data to match test_val structure using map_ori_sens
    df_plot_rearranged = test_val.copy()
    for r in range(len(test_val)):
        for c, band in enumerate(test_val.columns):
            # Get the sensor mapping from map_ori_sens
            sensor_info = map_ori_sens.iloc[r, c]
            if isinstance(sensor_info, tuple) and len(sensor_info) >= 2:
                sensor_col = sensor_info[1]  # sensor column name
                if sensor_col in df_sens_normalized.columns:
                    df_plot_rearranged.iloc[r, c] = df_sens_normalized.loc[r, sensor_col]
    
    return df_plot_rearranged, df_raw_plot

def save_interactive_heatmap(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10):
    """Create and save interactive heatmap with overlays."""
    
    # Use the specialized preprocessing for interactive heatmap
    df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)
    
    # Get x-axis values
    if 'ODDO1' in datafile.columns:
        x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m)'
    else:
        x_vals = pd.Series(np.arange(len(datafile)))
        x_label = 'Index'
    
    # Get y-band labels
    y_bands = [str(c) for c in test_val.columns]
    
    # Create heatmap data - transpose so y_bands are on y-axis
    heatmap_data = df_raw_plot.T
    
    # Ensure all data is numeric
    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    
    # Replace infinite values
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    
    heatmap_data = heatmap_data.astype("float32").round(3)
    
    # Create the interactive heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=x_vals.round(2),
        y=y_bands,
        colorscale='jet',
        zmin=-3,
        zmax=8,
        colorbar=dict(title="Sensor Value (%)"),
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>' +
                     '<b>%{y}</b><br>' +
                     '<b>Value: %{z:.2f}%</b>' +
                     '<extra></extra>'
    ))
    
    # Add overlay points with FIXED SIZE SQUARES
    overlay_added = False
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
    if pts is not None:
        xs, ys, labels = pts
        
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                
                # Fixed size square
                fig.add_shape(
                    type="rect",
                    x0=x - 0.05, y0=y_idx - 0.35,
                    x1=x + 0.05, y1=y_idx + 0.35,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(255,0,0,0.6)"
                )
                
                # Add label annotation
                fig.add_annotation(
                    x=x, y=y_idx,
                    text=label,
                    showarrow=False,
                    font=dict(color="white", size=8, family="Arial Black"),
                    bgcolor="red",
                    bordercolor="black",
                    borderwidth=1
                )
                overlay_added = True
    
    # Update layout
    fig.update_layout(
        title=f"Interactive Plotly Heatmap — Pipe {pipe_number}",
        xaxis_title=x_label,
        yaxis_title="Orientation (12h bands)",
        width=1400,
        height=700,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    # Save the interactive heatmap
    write_plotly_html(fig, f'{folder_path}/heatmap{pipe_number}.html')

    
    print(f"Saved interactive heatmap: {folder_path}/interactive_heatmap{pipe_number}.html")
    print(f"Overlays: {'Yes' if overlay_added else 'None found'}")


def create_plots(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original):
    folder_path = f'{output_folder}/Pipe_{pipe_number}'
    os.makedirs(folder_path, exist_ok=True)

    # Heatmap
    # save_heatmap(test_val, datafile, map_ori_sens, folder_path, pipe_number)

    # MultilinePlot (offset stack of sensors)
    save_lineplot(folder_path, test_val, datafile, pipe_number)

    # 3D Pipe 
    save_pipe3d(test_val, test_val, folder_path, pipe_number)

    # inside create_plots(...) or wherever you save other charts:
    save_proximity_linechart(folder_path, datafile, pipe_number)

    save_interactive_heatmap(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10)


def save_heatmap(test_val, datafile, map_ori_sens, folder_path, pipe_number):
    fighm = go.Figure(data=go.Heatmap(
        z=test_val.T,
        y=test_val.columns,
        x=(datafile['ODDO1'] / 1000).round(2),
        colorscale='jet',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],
    ))
    fighm.update_layout(
        xaxis_title='Absolute Distance (m)',
        height=500,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    write_plotly_html(fighm, f'{folder_path}/heatmap{pipe_number}.html')



# case-insensitive F#P# like F1P1, F12P3
FP_PATTERN = re.compile(r'^F\d{1,2}P\d{1,2}$', re.IGNORECASE)

def save_proximity_linechart(
    folder_path: str,
    datafile: pd.DataFrame,
    pipe_number,
    *,
    offset_step: float = 0.10,
    dtick: int = 1000,
    x_pref: str = "auto"  # "auto" -> ODDO1 if available, else index
):
    """
    Proximity linechart:
      - selects columns matching F#P# (case-insensitive),
      - forward-fills data,
      - MinMax scales each series to [0,1],
      - offsets each series by `offset_step`,
      - X-axis = ODDO1 (meters) if present (or if x_pref='oddo1'), else index,
      - saves HTML to {folder_path}/proximity_linechart{pipe_number}.html
    """
    df = datafile.copy()

    # 1) collect F*P* columns
    candidates = [c for c in df.columns if isinstance(c, str) and FP_PATTERN.match(c.strip())]
    if not candidates:
        print(f"No F#P# columns found for pipe {pipe_number}. Skipping proximity linechart.")
        return

    # 2) ensure numeric (coerce where possible)
    res_cols = []
    for c in candidates:
        if not is_numeric_dtype(df[c]):
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().any():
                df[c] = coerced
        if is_numeric_dtype(df[c]):
            res_cols.append(c)
    if not res_cols:
        print(f"No numeric F#P# columns for pipe {pipe_number}. Skipping proximity linechart.")
        return

    # 3) forward-fill
    df1 = df.fillna(method='ffill')

    # 4) choose x-axis (ODDO1 -> index)
    if x_pref.lower() == "oddo1" or (x_pref == "auto" and "ODDO1" in df1.columns):
        x_vals = (pd.to_numeric(df1["ODDO1"], errors="coerce") / 1000.0).round(3)
        x_label = "Abs. Distance (m) — ODDO1"
    else:
        x_vals = df1.index
        x_label = "Index"

    # 5) MinMax scale selected columns
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df1[res_cols].to_numpy())
    df1.loc[:, res_cols] = scaled

    # 6) figure with offsets
    fig = go.Figure()
    for i, col in enumerate(res_cols):
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=df1[col] + i * offset_step,
            name=col,
            mode='lines',
            line=dict(width=1),
            hoverinfo='x+y+name',
            showlegend=True
        ))

    # 7) styling + axis titles + gridlines
    fig.update_layout(
        title="Proximity Sensor Line Chart",
        width=1600,
        height=650,
        margin=dict(l=140, b=80),
        paper_bgcolor="#ffffff",
        plot_bgcolor='rgb(255, 255, 255)',
        title_x=0.5,
        font={"family": "courier"},
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis_title=x_label,
        yaxis_title="Scaled Proximity Sensor (0–1, offset)",
    )
    # keep titles close to axes
    fig.update_xaxes(title_standoff=8, automargin=True, dtick=dtick)
    fig.update_yaxes(title_standoff=10, automargin=True)

    # light major gridlines
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)", gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", gridwidth=1, zeroline=False)

    # optional minor gridlines (comment out if your Plotly build lacks support)
    fig.update_xaxes(minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=0.5))
    fig.update_yaxes(minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", gridwidth=0.5))

    # 8) save EXACTLY as requested
    write_plotly_html(fig, f'{folder_path}/proximity_linechart{pipe_number}.html')

    print(f"Saved {folder_path}/proximity_linechart{pipe_number}.html")



def save_heatmap_raw(folder_path, df_raw_straight, dataf, pipe_number):
    initial_read = INITIAL_READ
    roll = dataf['ROLL'] - initial_read

    def degrees_to_hours_minutes2(degrees):
        if (degrees < 0):
            degrees = degrees % 360
        elif degrees >= 360:
            degrees %= 360
        degrees_per_second = 360 / (12 * 60 * 60)
        total_seconds = degrees / degrees_per_second
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def add_sensor_keys(d):
        for e in d:
            new_dict = {**e}
            for i in range(1, 96):
                new_dict[f'Roll_Sensor_{i}'] = e['Roll_Sensor_0'] + (3.75 * i)
            yield new_dict

    def check_time_range(time_str):
        start_time = '00:00:00'
        end_time = '00:07:29'
        time_to_check = datetime.strptime(time_str, '%H:%M:%S')
        start_time_dt = datetime.strptime(start_time, '%H:%M:%S')
        end_time_dt = datetime.strptime(end_time, '%H:%M:%S')
        return start_time_dt <= time_to_check <= end_time_dt

    d = []
    for pos in roll:
        d.append({f"Roll_Sensor_0": pos})

    upd_d = list(add_sensor_keys(d))
    oriData = pd.DataFrame.from_dict(data=upd_d)
    clockData = oriData.applymap(degrees_to_hours_minutes2)

    test_clockData = clockData.copy()
    test_clockData = test_clockData.apply(pd.to_datetime, format='%H:%M:%S')
    test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M:%S'))
    test_clockData = test_clockData.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
    test_clockData = test_clockData.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
    test_clockData = test_clockData.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

    pat1 = re.compile(r'^F\d+H\d+$')   # F%H%
    sel_col = ['ROLL', 'ODDO1']
    fil_col = [col for col in dataf.columns if (pat1.match(col) or col in sel_col)]
    refData = dataf[fil_col]
    sensD = refData[refData.columns[2:]]

    test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_raw_straight.columns)))

    def create_time_dict():
        time_ranges_2 = [
            '00:00:00', '00:07:30', '00:15:00', '00:22:30', '00:30:00', '00:37:30', '00:45:00', '00:52:30', '01:00:00', '01:07:30', 
            '01:15:00', '01:22:30', '01:30:00', '01:37:30', '01:45:00', '01:52:30', '02:00:00', '02:07:30', '02:15:00', '02:22:30', 
            '02:30:00', '02:37:30', '02:45:00', '02:52:30', '03:00:00', '03:07:30', '03:15:00', '03:22:30', '03:30:00', '03:37:30', 
            '03:45:00', '03:52:30', '04:00:00', '04:07:30', '04:15:00', '04:22:30', '04:30:00', '04:37:30', '04:45:00', '04:52:30', 
            '05:00:00', '05:07:30', '05:15:00', '05:22:30', '05:30:00', '05:37:30', '05:45:00', '05:52:30', '06:00:00', '06:07:30', 
            '06:15:00', '06:22:30', '06:30:00', '06:37:30', '06:45:00', '06:52:30', '07:00:00', '07:07:30', '07:15:00', '07:22:30', 
            '07:30:00', '07:37:30', '07:45:00', '07:52:30', '08:00:00', '08:07:30', '08:15:00', '08:22:30', '08:30:00', '08:37:30', 
            '08:45:00', '08:52:30', '09:00:00', '09:07:30', '09:15:00', '09:22:30', '09:30:00', '09:37:30', '09:45:00', '09:52:30', 
            '10:00:00', '10:07:30', '10:15:00', '10:22:30', '10:30:00', '10:37:30', '10:45:00', '10:52:30', '11:00:00', '11:07:30', 
            '11:15:00', '11:22:30', '11:30:00', '11:37:30', '11:45:00', '11:52:30'
        ]
        return {key: [] for key in time_ranges_2}

    time_dict_1 = create_time_dict()
    rang = list(time_dict_1.keys())

    for _, row in test_clockData.iterrows():
        xl = list(row)
        xd = dict(row)
        xkeys = list(xd.keys())
        c = 0
        for _, dval in xd.items():
            if check_time_range(dval):
                ind = xl.index(dval)
                _ = xl[ind:] + xl[:ind]
                break

        curr = ind
        while True:
            ck = xkeys[curr]
            time_dict_1[rang[c]].append((curr, ck, xd[ck]))
            c += 1
            curr = (curr + 1) % len(xkeys)
            if curr == ind:
                break

    map_ori_sens = pd.DataFrame(time_dict_1)
    val_ori_sens = map_ori_sens.copy().applymap(lambda cell: cell[1])
    test_val = val_ori_sens.copy()

    for r, e in val_ori_sens.iterrows():
        c = 0
        for _, tup_value in e.items():
            cell_v = sensD.at[r, tup_value]
            test_val.iloc[r, c] = cell_v
            c += 1
    
    figraw = go.Figure(data=go.Heatmap(
        z=test_val.T,
        y=test_val.columns,
        x=(dataf['ODDO1'] / 1000).round(2),
        colorscale='jet',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],
        zmin=-12000,
        zmax=30000
    ))

    figraw.update_layout(
        height=300,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    write_plotly_html(figraw, f'{folder_path}/heatmap_raw{pipe_number}.html')



def save_lineplot(folder_path, test_val, datafile, pipe_number):
    figmlp = go.Figure()
    offset_step = 1200
    for idx, col in enumerate(test_val.columns):
        y_data = test_val[col].values
        offset_y_data = y_data + (idx * offset_step)
        figmlp.add_trace(go.Scatter(
            x=(datafile['ODDO1'] / 1000).round(2),
            y=offset_y_data,
            mode='lines',
            name=col,
            line=dict(width=1),
            hoverinfo='x+y+name',
            showlegend=False
        ))

    figmlp.update_layout(
        template='plotly_white',
        height=900,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(
            tickmode='array',
            tickvals=[idx * offset_step for idx in range(len(test_val.columns))],
            ticktext=test_val.columns,
            tickfont=dict(size=8),
        )
    )

    write_plotly_html(figmlp, f'{folder_path}/lineplot{pipe_number}.html')




def save_lineplot_raw(folder_path, test_val, pipe_number):
    figmlpraw = go.Figure()
    for _, col in enumerate(test_val.columns):
        y_data = test_val[col]
        figmlpraw.add_trace(go.Scatter(
            x=test_val.index,
            y=y_data,
            mode='lines',
            name=col,
            line=dict(width=1),
            hoverinfo='x+y+name',
            showlegend=False
        ))

    figmlpraw.update_layout(
        xaxis_title='Counter',
        template='plotly_white',
        height=300,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    write_plotly_html(figmlpraw, f'{folder_path}/lineplot_raw{pipe_number}.html')



def save_pipe3d(data, data_cp, folder_path, pipe_number):
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)  # test_val

    # ↓ add: subsample big grids
    if data.shape[0] > 1500:
        data = data[::2, :]          # halve rows
    if data.shape[1] > 128:
        data = data[:, ::1]          # keep cols (or ::2 if needed)

    num_rows, num_cols = data.shape

    theta = np.linspace(0, 2 * np.pi, num_cols)
    z = np.linspace(0, 1, num_rows)
    theta, z = np.meshgrid(theta, z)

    radius = 109.5   # OD = 219mm, R = OD/2
    odometer = num_rows

    # Cartesian Coords
    x = odometer * z
    y = radius * np.cos(theta)
    zc = radius * np.sin(theta)

    fig = go.Figure(data=[go.Surface(
        x=x,
        y=zc,
        z=y,
        surfacecolor=data,
        colorscale='jet',
        customdata=data_cp
    )])

    camera = dict(eye=dict(x=0., y=5, z=0.), up=dict(x=0, y=1, z=0))

    odometer_start = 0
    odometer_end = odometer

    fig.add_trace(go.Scatter3d(
        x=[odometer_start, odometer_end], y=[radius, radius], z=[0, 0],
        text=["3"], mode='text', textposition="middle center",
        marker=dict(size=0), name="3pm",
        textfont=dict(size=20, color="#61090c")
    ))
    fig.add_trace(go.Scatter3d(
        x=[odometer_start, odometer_end], y=[-radius, -radius], z=[0, 0],
        text=["9"], mode='text', textposition="middle center",
        marker=dict(size=0), name="9pm",
        textfont=dict(size=20, color="#61090c")
    ))
    fig.add_trace(go.Scatter3d(
        x=[odometer_start, odometer_end], y=[0, 0], z=[radius, radius],
        text=["6"], mode='text', textposition="middle center",
        marker=dict(size=0), name="6pm",
        textfont=dict(size=20, color="#61090c")
    ))
    fig.add_trace(go.Scatter3d(
        x=[odometer_start, odometer_end], y=[0, 0], z=[-radius, -radius],
        text=["12"], mode='text', textposition="middle center",
        marker=dict(size=0), name="12pm",
        textfont=dict(size=20, color="#61090c")
    ))
   
    fig.update_layout(
        scene=dict(
            xaxis_title='Odometer',
            yaxis_title='Radial Length',
            zaxis_title='Radial Length',
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=camera
        ),
        height=500,
        width=1500,
        title='Pipe Visualization',
        margin=dict(l=20, r=20, t=50, b=20),
    )

    write_plotly_html(fig, f'{folder_path}/pipe3d{pipe_number}.html')



# -------------------- PARALLEL HELPERS --------------------
def _resolve_workers(workers):
    if workers in (None, 0, -1, "auto"):
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1)
    if isinstance(workers, int):
        cpu = os.cpu_count() or 1
        return max(1, min(workers, cpu))
    return 1


def _process_one_pkl(pkl_path, output_folder):
    try:
        pipe_number = Path(pkl_path).stem
        pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
        pipe_folder.mkdir(exist_ok=True)

        data = pd.read_pickle(pkl_path)
        dfile = pre_process_data(data, pipe_number, output_folder)

        # Save the Excel
        xlsx_path = pipe_folder / f"Pipe_{pipe_number}.xlsx"
        dfile.to_excel(xlsx_path, index=False)

        return f"Processed {os.path.basename(pkl_path)} and saved to {pipe_folder}"
    except Exception as e:
        return f"Error loading {os.path.basename(pkl_path)}: {e}"


def create_html_and_csv_from_pkl(
    pkl_folder='pipes3',
    output_folder='Client_Pipes',
    output_callback=None,
    workers=WORKERS
):
    Path(output_folder).mkdir(exist_ok=True)

    # collect .pkl file paths
    pkl_paths = [
        str(Path(pkl_folder) / f)
        for f in os.listdir(pkl_folder)
        if f.lower().endswith('.pkl')
    ]

    if not pkl_paths:
        msg = f"No .pkl files found in {pkl_folder}"
        if output_callback: output_callback(msg)
        else: print(msg)
        return

    n_jobs = _resolve_workers(workers)

    # fan out work across processes
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_process_one_pkl)(p, output_folder) for p in pkl_paths
    )

    # report
    for msg in results:
        if output_callback: output_callback(msg)
        else: print(msg)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    import time
    st = time.time()
    # override cores at runtime if needed, e.g., workers=4
    create_html_and_csv_from_pkl(workers=WORKERS)
    print(f'Total time: {time.time()-st} seconds')
