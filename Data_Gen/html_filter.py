# import os
# # Limit intra-process math threads to avoid over-subscription
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
#
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
#
# import plotly.express as px
# from typing import Optional, Union, List
# from glob import glob
# from datetime import datetime, timedelta
# import traceback
#
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# # ------------ Plotly export options (keep interactive + smaller HTML) ------------
# PLOTLY_JS_MODE   = "directory"   # "directory" (offline, small HTML) or "cdn" (requires internet)
# PLOTLY_COMPRESS  = True          # compress embedded data
# HTML_DEFAULT_W   = "100%"        # let the container control size
# HTML_DEFAULT_H   = 500           # or "100%" if you want full-height wrappers
#
# PLOTLY_CONFIG = {
#     "displaylogo": False,
#     "modeBarButtonsToRemove": [
#         "lasso2d", "select2d", "autoScale2d"
#     ],
#     # , "zoomIn2d", "zoomOut2d",
#     #     "pan2d", "toImage", "hoverCompareCartesian", "hoverClosestCartesian"
#     "displayModeBar": True,
#     "scrollZoom": False,
# }
#
# def write_plotly_html(fig, out_path: str):
#     common = dict(
#         include_plotlyjs=PLOTLY_JS_MODE,
#         full_html=True,
#         config=PLOTLY_CONFIG,
#         auto_open=False,
#         default_width=HTML_DEFAULT_W,
#         default_height=HTML_DEFAULT_H,
#     )
#     try:
#         fig.write_html(out_path, compress_data=PLOTLY_COMPRESS, **common)
#     except TypeError:
#         fig.write_html(out_path, **common)
#
#
# # -------------------- CONFIG --------------------
# INITIAL_READ = 0.0      # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
# UPPER_SENS_MUL = 1
# LOWER_SENS_MUL = 3
# # -1 / 0 / None / "auto" => auto (CPU-1, at least 1). Or set an int, e.g. 4
# WORKERS = 8
# # ------------------------------------------------
#
#
# # def pre_process_data(datafile, pipe_number, output_folder, total_sensors, column_names, minute_sensors, degree_sensors, sensor_type):
# #
# #     datafile_original = datafile.copy(deep=True)
# #     df_new_tab9 = pd.DataFrame(
# #         datafile,
# #         columns=column_names
# #     )
# #
# #     df_new_tab10 = df_new_tab9.copy()
# #     sensor_columns = df_new_tab9.columns.tolist()
# #
# #     # Denoising using Savitzky-Golay filter
# #     window_length = 15
# #     polyorder = 2
# #     for col in sensor_columns:
# #         data = df_new_tab9[col].values
# #         time_index = np.arange(len(df_new_tab9))
# #         trend = np.polyval(np.polyfit(time_index, data, 2), time_index)
# #         data_denoised = savgol_filter(data - trend, window_length, polyorder)
# #         df_new_tab9[col] = data_denoised
# #
# #     df_raw_straight = df_new_tab9.copy()
# #
# #     # Setting bounds and applying conditions
# #     sens_mean = df_new_tab9.abs().mean()
# #     standard_deviation = df_new_tab9.std(axis=0, skipna=True)
# #
# #     mean_plus_sigma = sens_mean + UPPER_SENS_MUL * standard_deviation
# #     mean_negative_sigma = sens_mean - LOWER_SENS_MUL * standard_deviation
# #
# #     # Apply noise filtering to zero-out in-bound values
# #     for col in df_new_tab9.columns:
# #         if col in mean_plus_sigma.index and col in mean_negative_sigma.index:
# #             df_new_tab9[col] = np.where(
# #                 (df_new_tab9[col] >= mean_negative_sigma[col]) &
# #                 (df_new_tab9[col] <= mean_plus_sigma[col]),
# #                 0,
# #                 df_new_tab9[col]
# #             )
# #
# #     initial_read = INITIAL_READ
# #     roll = datafile['ROLL'] - initial_read
# #
# #     def degrees_to_hours_minutes2(degrees):
# #         if (degrees < 0):
# #             degrees = degrees % 360
# #         elif degrees >= 360:
# #             degrees %= 360
# #         degrees_per_second = 360 / (12 * 60 * 60)
# #         total_seconds = degrees / degrees_per_second
# #         hours = int(total_seconds // 3600)
# #         minutes = int((total_seconds % 3600) // 60)
# #         seconds = int(total_seconds % 60)
# #         return f"{hours:02d}:{minutes:02d}"
# #
# #     def add_sensor_keys(d):
# #         for e in d:
# #             new_dict = {**e}
# #             for i in range(1, total_sensors):
# #                 new_dict[f'Roll_Sensor_{i}'] = e['Roll_Sensor_0'] + (degree_sensors * i)
# #             yield new_dict
# #
# #     def check_time_range(time_str):
# #         start_time = list(time_dict_1.keys())[0]
# #         end_time_dt = datetime.strptime(list(time_dict_1.keys())[1], '%H:%M') - timedelta(seconds=1)
# #         end_time = list(time_dict_1.keys())[1]
# #         time_to_check = datetime.strptime(time_str, '%H:%M')
# #         start_time_dt = datetime.strptime(start_time, '%H:%M')
# #         return start_time_dt <= time_to_check <= end_time_dt
# #
# #     d = []
# #     for pos in roll:
# #         d.append({f"Roll_Sensor_0": pos})
# #
# #     upd_d = list(add_sensor_keys(d))
# #     oriData = pd.DataFrame.from_dict(data=upd_d)
# #     clockData = oriData.applymap(degrees_to_hours_minutes2)
# #
# #     test_clockData = clockData.copy()
# #
# #     # Parse flexibly with mixed formats (works for both HH:MM and HH:MM:SS)
# #     test_clockData = test_clockData.apply(pd.to_datetime, format='mixed')
# #
# #     # Now format the datetime objects to strings 'HH:MM' dropping seconds
# #     test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M'))
# #     test_clockData = test_clockData.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
# #     test_clockData = test_clockData.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
# #     test_clockData = test_clockData.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)
# #
# #     test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_new_tab9.columns)))
# #
# #     def create_time_dict():
# #         time_list = [timedelta(minutes=i * minute_sensors) for i in range(total_sensors)]
# #         time_ranges_2 = [(datetime.min + t).strftime('%H:%M') for t in time_list]
# #         return {key: [] for key in time_ranges_2}
# #
# #     time_dict_1 = create_time_dict()
# #     rang = list(time_dict_1.keys())
# #
# #     for _, row in test_clockData.iterrows():
# #         xl = list(row)
# #         xd = dict(row)
# #         xkeys = list(xd.keys())
# #         c = 0
# #         for _, dval in xd.items():
# #             if check_time_range(dval):
# #                 ind = xl.index(dval)
# #                 _ = xl[ind:] + xl[:ind]  # not used later but kept for clarity
# #                 break
# #
# #         curr = ind
# #         while True:
# #             ck = xkeys[curr]
# #             time_dict_1[rang[c]].append((curr, ck, xd[ck]))
# #             c += 1
# #             curr = (curr + 1) % len(xkeys)
# #             if curr == ind:
# #                 break
# #
# #     map_ori_sens = pd.DataFrame(time_dict_1)
# #
# #     val_ori_sens = map_ori_sens.copy()
# #
# #     def extract_string(cell):
# #         return cell[1]
# #
# #     val_ori_sens = val_ori_sens.applymap(extract_string)
# #
# #     test_val = val_ori_sens.copy()
# #
# #     for r, e in val_ori_sens.iterrows():
# #         c = 0
# #         for _, tup_value in e.items():
# #             cell_v = df_new_tab9.at[r, tup_value]
# #             test_val.iloc[r, c] = cell_v
# #             c += 1
# #
# #     map_val_sens = pd.DataFrame(index=test_val.index, columns=test_val.columns)
# #     for column in test_val.columns:
# #         for i in range(test_val.shape[0]):
# #             map_value = map_ori_sens.loc[i, column]
# #             test_value = test_val.loc[i, column]
# #             map_val_sens.loc[i, column] = (*map_value, test_value)
# #     if sensor_type == "Hall":
# #         create_plots_hall(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original)
# #     else:
# #         create_plots_proximity(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original)
# #     return datafile, df_new_tab9, datafile_original, test_val, map_ori_sens, df_new_tab10
#
# def pre_process_data(
#     pkl_path,
#     datafile,
#     pipe_number,
#     output_folder,
#     total_sensors,
#     column_names,
#     minute_sensors,
#     degree_sensors,
#     sensor_type,
#     debug=False
# ):
#     import time
#
#     # -----------------------------
#     # Debug helper
#     # -----------------------------
#     dbg_start = time.time()
#
#     def dbg(msg):
#         if debug:
#             elapsed = round(time.time() - dbg_start, 2)
#             print(
#                 f"[PIPE {pipe_number} | {sensor_type}] {msg} | {elapsed}s",
#                 flush=True
#             )
#
#     dbg("START pre_process_data")
#
#     # -----------------------------
#     # Initial copies
#     # -----------------------------
#     datafile_original = datafile.copy(deep=True)
#     dbg("1: datafile_original copy complete")
#
#     df_new_tab9 = pd.DataFrame(
#         datafile,
#         columns=column_names
#     )
#     dbg(f"2: df_new_tab9 created | shape={df_new_tab9.shape}")
#
#     df_new_tab10 = df_new_tab9.copy()
#     dbg("3: df_new_tab10 copy complete")
#
#     sensor_columns = df_new_tab9.columns.tolist()
#     dbg(f"4: sensor_columns loaded | count={len(sensor_columns)}")
#
#     # -----------------------------
#     # Denoising using Savitzky-Golay filter
#     # -----------------------------
#     window_length = 15
#     polyorder = 2
#
#     dbg("5: starting denoise loop")
#
#     for idx, col in enumerate(sensor_columns):
#         if debug and idx % 10 == 0:
#             dbg(f"5.{idx}: denoising {col}")
#
#         data = df_new_tab9[col].values
#         time_index = np.arange(len(df_new_tab9))
#
#         trend = np.polyval(
#             np.polyfit(time_index, data, 2),
#             time_index
#         )
#
#         data_denoised = savgol_filter(
#             data - trend,
#             window_length,
#             polyorder
#         )
#
#         df_new_tab9[col] = data_denoised
#
#     dbg("6: denoise loop complete")
#
#     df_raw_straight = df_new_tab9.copy()
#     dbg("7: df_raw_straight copy complete")
#
#     # -----------------------------
#     # Setting bounds and applying conditions
#     # -----------------------------
#     dbg("8: calculating means/std")
#
#     sens_mean = df_new_tab9.abs().mean()
#     standard_deviation = df_new_tab9.std(axis=0, skipna=True)
#
#     mean_plus_sigma = sens_mean + UPPER_SENS_MUL * standard_deviation
#     mean_negative_sigma = sens_mean - LOWER_SENS_MUL * standard_deviation
#
#     dbg("9: applying noise filtering")
#
#     for idx, col in enumerate(df_new_tab9.columns):
#         if debug and idx % 20 == 0:
#             dbg(f"9.{idx}: filtering {col}")
#
#         if col in mean_plus_sigma.index and col in mean_negative_sigma.index:
#             df_new_tab9[col] = np.where(
#                 (df_new_tab9[col] >= mean_negative_sigma[col]) &
#                 (df_new_tab9[col] <= mean_plus_sigma[col]),
#                 0,
#                 df_new_tab9[col]
#             )
#
#     dbg("10: noise filtering complete")
#
#     # -----------------------------
#     # Roll prep
#     # -----------------------------
#     initial_read = INITIAL_READ
#     roll = datafile['ROLL'] - initial_read
#
#     dbg("11: roll normalization complete")
#
#     def degrees_to_hours_minutes2(degrees):
#         if degrees < 0:
#             degrees = degrees % 360
#         elif degrees >= 360:
#             degrees %= 360
#
#         degrees_per_second = 360 / (12 * 60 * 60)
#         total_seconds = degrees / degrees_per_second
#
#         hours = int(total_seconds // 3600)
#         minutes = int((total_seconds % 3600) // 60)
#
#         return f"{hours:02d}:{minutes:02d}"
#
#     def add_sensor_keys(d):
#         for e in d:
#             new_dict = {**e}
#
#             for i in range(1, total_sensors):
#                 new_dict[f'Roll_Sensor_{i}'] = (
#                     e['Roll_Sensor_0'] + (degree_sensors * i)
#                 )
#
#             yield new_dict
#
#     def check_time_range(time_str):
#         start_time = list(time_dict_1.keys())[0]
#
#         end_time_dt = (
#             datetime.strptime(
#                 list(time_dict_1.keys())[1],
#                 '%H:%M'
#             ) - timedelta(seconds=1)
#         )
#
#         time_to_check = datetime.strptime(time_str, '%H:%M')
#         start_time_dt = datetime.strptime(start_time, '%H:%M')
#
#         return start_time_dt <= time_to_check <= end_time_dt
#
#     # -----------------------------
#     # Build roll sensor map
#     # -----------------------------
#     dbg("12: building roll sensor seed list")
#
#     d = []
#
#     for idx, pos in enumerate(roll):
#         if debug and idx % 5000 == 0:
#             dbg(f"12.{idx}: roll seed progress")
#
#         d.append({'Roll_Sensor_0': pos})
#
#     dbg("13: expanding sensor keys")
#
#     upd_d = list(add_sensor_keys(d))
#
#     dbg("14: creating oriData dataframe")
#
#     oriData = pd.DataFrame.from_dict(data=upd_d)
#
#     dbg(f"15: oriData created | shape={oriData.shape}")
#
#     dbg("16: converting degrees to clock")
#
#     clockData = oriData.applymap(degrees_to_hours_minutes2)
#
#     dbg("17: clockData created")
#
#     test_clockData = clockData.copy()
#
#     dbg("18: parsing datetime mixed format")
#
#     test_clockData = test_clockData.apply(
#         pd.to_datetime,
#         format='mixed'
#     )
#
#     dbg("19: formatting datetime strings")
#
#     test_clockData = test_clockData.applymap(
#         lambda x: x.strftime('%H:%M')
#     )
#
#     test_clockData = test_clockData.applymap(
#         lambda x: x.replace('23:', '11:')
#         if isinstance(x, str) and x.startswith('23:')
#         else x
#     )
#
#     test_clockData = test_clockData.applymap(
#         lambda x: x.replace('22:', '10:')
#         if isinstance(x, str) and x.startswith('22:')
#         else x
#     )
#
#     test_clockData = test_clockData.applymap(
#         lambda x: x.replace('12:', '00:')
#         if isinstance(x, str) and x.startswith('12:')
#         else x
#     )
#
#     dbg("20: renaming clockData columns")
#
#     test_clockData = test_clockData.rename(
#         columns=dict(
#             zip(
#                 test_clockData.columns,
#                 df_new_tab9.columns
#             )
#         )
#     )
#
#     # -----------------------------
#     # Time dict
#     # -----------------------------
#     dbg("21: creating time dictionary")
#
#     def create_time_dict():
#         time_list = [
#             timedelta(minutes=i * minute_sensors)
#             for i in range(total_sensors)
#         ]
#
#         time_ranges_2 = [
#             (datetime.min + t).strftime('%H:%M')
#             for t in time_list
#         ]
#
#         return {key: [] for key in time_ranges_2}
#
#     time_dict_1 = create_time_dict()
#     rang = list(time_dict_1.keys())
#
#     dbg("22: remapping sensor timeline")
#
#     for row_idx, row in test_clockData.iterrows():
#         if debug and row_idx % 2000 == 0:
#             dbg(f"22.{row_idx}: timeline remap progress")
#
#         xl = list(row)
#         xd = dict(row)
#         xkeys = list(xd.keys())
#         c = 0
#
#         for _, dval in xd.items():
#             if check_time_range(dval):
#                 ind = xl.index(dval)
#                 break
#
#         curr = ind
#
#         while True:
#             ck = xkeys[curr]
#
#             time_dict_1[rang[c]].append(
#                 (curr, ck, xd[ck])
#             )
#
#             c += 1
#             curr = (curr + 1) % len(xkeys)
#
#             if curr == ind:
#                 break
#
#     dbg("23: timeline remap complete")
#
#     # -----------------------------
#     # Build mapped dataframes
#     # -----------------------------
#     map_ori_sens = pd.DataFrame(time_dict_1)
#
#     dbg(f"24: map_ori_sens created | shape={map_ori_sens.shape}")
#
#     val_ori_sens = map_ori_sens.copy()
#
#     def extract_string(cell):
#         return cell[1]
#
#     dbg("25: extracting sensor labels")
#
#     val_ori_sens = val_ori_sens.applymap(extract_string)
#
#     test_val = val_ori_sens.copy()
#
#     # -----------------------------
#     # STAGE 26 DEBUG
#     # -----------------------------
#     dbg("26: mapping sensor values")
#
#     total_rows_26 = val_ori_sens.shape[0]
#     total_cols_26 = val_ori_sens.shape[1]
#     total_iter_26 = total_rows_26 * total_cols_26
#
#     dbg(
#         f"26.DEBUG SUMMARY | "
#         f"Rows={total_rows_26:,} | "
#         f"Cols={total_cols_26:,} | "
#         f"Total Inner Iterations={total_iter_26:,} "
#         f"({round(total_iter_26 / 1_000_000, 2)}M)"
#     )
#
#     for r, e in val_ori_sens.iterrows():
#         # if debug:
#             # dbg(
#             #     f"26.OUTER LOOP | "
#             #     f"Row {r+1}/{total_rows_26:,} | "
#             #     f"Inner loop size={len(e):,}"
#             # )
#
#         c = 0
#
#         for inner_idx, (_, tup_value) in enumerate(e.items()):
#             # if debug and inner_idx == 0:
#                 # dbg(
#                 #     f"26.INNER START | "
#                 #     f"Row={r+1} | "
#                 #     f"Columns this row={len(e):,}"
#                 # )
#
#             cell_v = df_new_tab9.at[r, tup_value]
#             test_val.iloc[r, c] = cell_v
#             c += 1
#
#     dbg("27: creating map_val_sens")
#
#     map_val_sens = pd.DataFrame(
#         index=test_val.index,
#         columns=test_val.columns
#     )
#     total_rows_27 = test_val.shape[0]
#     total_cols_27 = test_val.shape[1]
#     total_iter_27 = total_rows_27 * total_cols_27
#
#     dbg(
#         f"27.DEBUG SUMMARY | "
#         f"Rows={total_rows_27:,} | "
#         f"Cols={total_cols_27:,} | "
#         f"Total Iterations={total_iter_27:,} "
#         f"({round(total_iter_27 / 1_000_000, 2)}M)"
#     )
#     for col_idx, column in enumerate(test_val.columns):
#         if debug and col_idx % 20 == 0:
#             dbg(f"27.{col_idx}: map_val_sens column progress")
#
#         for i in range(test_val.shape[0]):
#             map_value = map_ori_sens.loc[i, column]
#             test_value = test_val.loc[i, column]
#
#             map_val_sens.loc[i, column] = (
#                 *map_value,
#                 test_value
#             )
#
#     dbg("28: plotting stage")
#
#     # -----------------------------
#     # Plotting
#     # -----------------------------
#     if sensor_type == "Hall":
#         create_plots_hall(
#             pkl_path,
#             df_new_tab9,
#             df_raw_straight,
#             datafile,
#             test_val,
#             map_ori_sens,
#             pipe_number,
#             output_folder,
#             df_new_tab10,
#             datafile_original
#         )
#
#     else:
#         create_plots_proximity(
#             pkl_path,
#             df_new_tab9,
#             df_raw_straight,
#             datafile,
#             test_val,
#             map_ori_sens,
#             pipe_number,
#             output_folder,
#             df_new_tab10,
#             datafile_original
#         )
#
#     dbg("29: COMPLETE pre_process_data")
#
#     return (
#         datafile,
#         df_new_tab9,
#         datafile_original,
#         test_val,
#         map_ori_sens,
#         df_new_tab10
#     )
#
#
# def _find_pipe_tally_file(pipe_number: Union[str, int], folder_path: str) -> Optional[str]:
#     """Look for PipeTally{pipe_number}.csv in the pipe folder."""
#     pn = str(pipe_number)
#     # Look specifically in the pipe folder for PipeTally{pipe_number}.csv
#     tally_path = f"{folder_path}/PipeTally{pn}.csv"
#     if os.path.exists(tally_path):
#         return tally_path
#
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
#
# def _pick_col(df: pd.DataFrame, preferred: List[str], tokens: List[str]) -> Optional[str]:
#     """Pick a column by exact name (case-insensitive) or by 'contains all tokens'."""
#     cols = list(df.columns)
#     low = {c.lower(): c for c in cols}
#
#     for name in preferred:
#         nlow = name.lower()
#         if nlow in low:
#             return low[nlow]
#
#     for c in cols:
#         cl = c.lower()
#         if all(t in cl for t in tokens):
#             return c
#     return None
#
# def _parse_ori_to_seconds(v) -> Optional[int]:
#     """Convert '8', '8.5', '8:30', '08:30:00' → seconds on a 12h dial."""
#     if v is None or (isinstance(v, float) and np.isnan(v)):
#         return None
#
#     if isinstance(v, (int, float)):
#         h = int(v) % 12
#         m = int(round((float(v) - int(v)) * 60))
#         return h * 3600 + m * 60
#
#     s = str(v).strip().lower()
#     s = re.sub(r"[^0-9:.]", "", s)
#     if not s:
#         return None
#
#     if ":" in s:
#         parts = s.split(":")
#         try:
#             h = int(parts[0]) % 12
#             m = int(parts[1]) if len(parts) > 1 else 0
#             sec = int(parts[2]) if len(parts) > 2 else 0
#             return h * 3600 + m * 60 + sec
#         except Exception:
#             return None
#
#     try:
#         f = float(s)
#         h = int(f) % 12
#         m = int(round((f - int(f)) * 60))
#         return h * 3600 + m * 60
#     except Exception:
#         pass
#
#     try:
#         h = int(s) % 12
#         return h * 3600
#     except Exception:
#         return None
#
# def _hhmmss_to_seconds(t: str) -> int:
#     parts = str(t).split(":")
#     if len(parts) == 3:
#         h, m, s = [int(x) for x in parts]
#     elif len(parts) == 2:
#         h, m = [int(x) for x in parts]
#         s = 0
#     else:
#         raise ValueError(f"Invalid time format: {t}")
#     return (h % 12) * 3600 + m * 60 + s
#
#
# def _nearest_band_label(seconds: int, band_labels: List[str]) -> str:
#     band_labels_str = [str(x) for x in band_labels]
#     band_secs = np.array([_hhmmss_to_seconds(lbl) for lbl in band_labels_str], dtype=int)
#     idx = int(np.argmin(np.abs(band_secs - seconds)))
#     return band_labels_str[idx]
#
# def _load_overlay_points_for_pipe(pipe_number, y_band_labels, folder_path, *, debug_prefix="OVERLAY DEBUG"):
#     """Return (xs, ys, labels) for overlay markers if a PipeTally file is found.
#        Only plot rows where Feature Type == 'Metal Loss' (case-insensitive).
#        Skip any row missing usable x or orientation.
#     """
#     path = _find_pipe_tally_file(pipe_number, folder_path)
#     if not path:
#         print(f"{debug_prefix}: pipe {pipe_number}: no PipeTally file found.")
#         return None
#
#     try:
#         df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
#     except Exception as e:
#         print(f"{debug_prefix}: pipe {pipe_number}: failed reading '{path}': {e}")
#         return None
#
#     x_col    = _pick_col(df, ["Abs. Distance (m)", "Absolute Distance"], ["abs", "distance"]) \
#             or _pick_col(df, [], ["distance"])
#     ori_col  = _pick_col(df, ["Orientation o' clock", "Orientation", "Ori"], ["ori"]) \
#             or _pick_col(df, [], ["orient"])
#     feat_col = _pick_col(df, ["Feature Type", "Feature"], ["feature", "type"])
#     pipe_col = _pick_col(df, ["Pipe Number", "Pipe"], ["pipe", "number"])
#     sno_col  = _pick_col(df, ["s_no", "S_No", "Serial Number", "SNo"], ["s_no", "sno", "serial"])
#
#     # print(f"{debug_prefix}: pipe {pipe_number}: file='{path}' cols: x='{x_col}', ori='{ori_col}', feature='{feat_col}', pipe='{pipe_col}', s_no='{sno_col}'")
#
#     # Must have x and orientation columns to plot
#     if x_col is None or ori_col is None:
#         print(f"{debug_prefix}: pipe {pipe_number}: missing required columns; no overlay.")
#         return None
#
#     # If there is a 'Pipe Number' column, restrict to this pipe (best-effort)
#     if pipe_col is not None:
#         mask = df[pipe_col].astype(str).str.contains(str(pipe_number), na=False)
#         if mask.any():
#             df = df[mask]
#
#     total_rows = len(df)
#
#     # Feature filtering: ONLY "Metal Loss" (case-insensitive, tolerant to spacing)
#     # If feat_col doesn't exist, we bail (since we must only plot Metal Loss)
#     if feat_col is None:
#         print(f"{debug_prefix}: pipe {pipe_number}: no feature column; overlays disabled (Metal Loss only).")
#         return None
#
#     feat_series = df[feat_col].astype(str).str.strip().str.lower()
#     metal_loss_mask = feat_series.str.fullmatch(r"metal\s*loss", case=False, na=False)
#     df = df[metal_loss_mask]
#
#     after_filter = len(df)
#
#     xs, ys, labels = [], [], []
#     skipped_no_x, skipped_no_ori, skipped_other = 0, 0, 0
#
#     for _, row in df.iterrows():
#         # x must be a valid number
#         x = pd.to_numeric(row.get(x_col), errors="coerce")
#         if pd.isna(x):
#             skipped_no_x += 1
#             continue
#
#         # orientation must parse
#         ori_sec = _parse_ori_to_seconds(row.get(ori_col))
#         if ori_sec is None:
#             skipped_no_ori += 1
#             continue
#
#         # find nearest y-band
#         y = _nearest_band_label(int(ori_sec), list(y_band_labels))
#         if y not in y_band_labels:
#             skipped_other += 1
#             continue
#
#         # label: prefer s_no; fallback to Defect_id; fallback to running index
#         if sno_col is not None:
#             lbl = row.get(sno_col)
#             lbl = str(lbl).strip() if (lbl is not None and str(lbl).strip() != "" and not pd.isna(lbl)) else None
#         else:
#             lbl = None
#
#         if lbl is None:
#             lbl2 = row.get("Defect_id")
#             lbl = str(lbl2).strip() if (lbl2 is not None and str(lbl2).strip() != "" and not pd.isna(lbl2)) else str(len(labels) + 1)
#
#         xs.append(float(x))
#         ys.append(str(y))
#         labels.append(lbl)
#
#     print(
#         f"\n{debug_prefix}: pipe {pipe_number}: total_rows={total_rows}, "
#         f"after_feature='Metal Loss'={after_filter}, plotted={len(xs)}, "
#         f"skipped_no_x={skipped_no_x}, skipped_no_ori={skipped_no_ori}, skipped_other={skipped_other}"
#     )
#
#     if not xs:
#         return None
#     return xs, ys, labels
#
#
#
# def pre_process_for_interactive_heatmap(df_in: pd.DataFrame, datafile: pd.DataFrame, test_val: pd.DataFrame, map_ori_sens: pd.DataFrame):
#     """Process data specifically for interactive heatmap - percentage-based."""
#
#     # Get sensor columns (use raw data, not the processed df_new_tab9)
#     sens_cols = df_in.columns.tolist()
#     print(f"Sensor columns in datafile for preprocess_heatmap_hall_sensor: {sens_cols}")
#     if not sens_cols:
#         raise ValueError("No F*H* sensor columns found in the input DataFrame.")
#
#     df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
#     df_sens_raw = df_sens.copy(deep=True)  # Keep raw copy for CSV export
#     df_mean_cols = df_sens_raw
#     # print(list(df_mean_cols.columns))
#     Mean1 = df_mean_cols.mean()
#     df_raw_plot = ((df_mean_cols - Mean1)/Mean1)*100
#
#     # Convert all sensor data to numeric
#     for col in sens_cols:
#         df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
#
#     # Fill NaN values with forward fill, then zeros
#     df_sens = df_sens.fillna(method='ffill').fillna(0.0)
#
#     # Calculate mean for percentage normalization
#     Mean1 = df_sens.mean()
#
#     # Normalize as percentage deviation from mean
#     df_sens_normalized = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100
#
#     # Zero out values above mean threshold (different from main processing)
#     for col in sens_cols:
#         df_sens_normalized.loc[df_sens_normalized[col] > Mean1[col], col] = 0
#
#     # Apply additional filtering if needed
#     if UPPER_SENS_MUL > 0 and LOWER_SENS_MUL > 0:
#         sens_std = df_sens.std(axis=0, skipna=True)
#         upper = Mean1 + UPPER_SENS_MUL * sens_std
#         lower = Mean1 - LOWER_SENS_MUL * sens_std
#
#         for col in sens_cols:
#             mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
#             df_sens_normalized.loc[mask, col] = 0
#
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
#
#     return df_plot_rearranged, df_raw_plot
#
# def save_interactive_heatmap(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10):
#     """Create and save interactive heatmap with overlays."""
#
#
#     # Use the specialized preprocessing for interactive heatmap
#     df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)
#
#     # Get x-axis values
#     if 'ODDO1' in datafile.columns:
#         x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
#         x_label = 'Absolute Distance (m) --- Hall Sensor Heatmap'
#     else:
#         x_vals = pd.Series(np.arange(len(datafile)))
#         x_label = 'Index'
#
#     # Get y-band labels
#     y_bands = [str(c) for c in test_val.columns]
#
#     # Create heatmap data - transpose so y_bands are on y-axis
#     heatmap_data = df_raw_plot.T
#
#     # Ensure all data is numeric
#     for col in heatmap_data.columns:
#         heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
#     heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
#
#     # Replace infinite values
#     if not np.isfinite(heatmap_data.values).all():
#         heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
#
#     heatmap_data = heatmap_data.astype("float32").round(3)
#
#     # Create the interactive heatmap using Plotly
#     fig = go.Figure(data=go.Heatmap(
#         z=heatmap_data.values,
#         x=x_vals.round(2),
#         y=y_bands,
#         colorscale='jet',
#         zmin=-3,
#         zmax=8,
#         # colorbar=dict(title="Sensor Value (%)"),
#         showscale=False,
#         hoverongaps=False,
#         hovertemplate='<b>%{x}</b><br>' +
#                      '<b>%{y}</b><br>' +
#                      '<b>Value: %{z:.2f}%</b>' +
#                      '<extra></extra>'
#     ))
#
#     # Add overlay points with FIXED SIZE SQUARES
#     overlay_added = False
#     pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
#     if pts is not None:
#         xs, ys, labels = pts
#
#         for x, y_band, label in zip(xs, ys, labels):
#             if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
#                 y_idx = y_bands.index(y_band)
#
#                 # Fixed size square
#                 fig.add_shape(
#                     type="rect",
#                     x0=x - 0.05, y0=y_idx - 0.35,
#                     x1=x + 0.05, y1=y_idx + 0.35,
#                     line=dict(color="black", width=2),
#                     fillcolor="rgba(255,0,0,0.6)"
#                 )
#
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
#
#     # Update layout
#     fig.update_layout(
#         title=dict(
#         text=f"Hall-Sensor Heatmap — Joint Number {pipe_number}",  # 👈 chart title
#         x=0.5,        # center
#         xanchor="center",
#         font=dict(size=18, family="Arial Black")),  # customize font,
#         xaxis_title=x_label,
#         yaxis_title=" ",
#         width=1500,
#         height=500,
#         font=dict(size=12),
#         xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#         yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
#     )
#     fig.update_yaxes(autorange="reversed")
#
#     # Save the interactive heatmap
#     write_plotly_html(fig, f'{folder_path}/hallsensor_heatmap{pipe_number}.html')
#
#
#     print(f"Saved hallsensor heatmap: {folder_path}/hallsensor_heatmap{pipe_number}.html")
#     print(f"Overlays: {'Yes' if overlay_added else 'None found'}")
#
# def save_interactive_heatmap_proximity(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10):
#     """Create and save interactive heatmap with overlays."""
#     print("RUNNING PROXIMITY HEATMAP GENERATION \n")
#     # # case-insensitive F#P# like F1P1, F12P3
#     pattern = re.compile(r'^F\d+P\d+$', re.IGNORECASE)
#     datafile_original = datafile.copy(deep=True)
#
#     pattern = re.compile(r'^F\d+P\d+$')
#     matching_columns = [col for col in datafile_original.columns if pattern.match(col)]
#     df_new_tab9 = pd.DataFrame(
#         datafile,
#         columns=matching_columns
#     )
#     # print("doing processing with datacolms: ", df_new_tab9.columns)
#     df_new_tab10 = df_new_tab9.copy()
#
#     # Use the specialized preprocessing for interactive heatmap
#     df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)
#
#     # Get x-axis values
#     if 'ODDO1' in datafile.columns:
#         x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
#         x_label = 'Absolute Distance (m) --- Proximity Sensor Heatmap'
#     else:
#         x_vals = pd.Series(np.arange(len(datafile)))
#         x_label = 'Index'
#
#     # Get y-band labels
#     y_bands = [str(c) for c in test_val.columns]
#
#     # Create heatmap data - transpose so y_bands are on y-axis
#     heatmap_data = df_raw_plot.T
#
#     # Ensure all data is numeric
#     for col in heatmap_data.columns:
#         heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
#     heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
#
#     # Replace infinite values
#     if not np.isfinite(heatmap_data.values).all():
#         heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
#
#     heatmap_data = heatmap_data.astype("float32").round(3)
#
#     # Create the interactive heatmap using Plotly
#     fig = go.Figure(data=go.Heatmap(
#         z=heatmap_data.values,
#         x=x_vals.round(2),
#         y=y_bands,
#         colorscale='jet',
#         zmin=-3,
#         zmax=8,
#         # colorbar=dict(title="Sensor Value (%)"),
#         showscale=False,
#         hoverongaps=False,
#         hovertemplate='<b>%{x}</b><br>' +
#                      '<b>%{y}</b><br>' +
#                      '<b>Value: %{z:.2f}%</b>' +
#                      '<extra></extra>'
#     ))
#
#     # Add overlay points with FIXED SIZE SQUARES
#     overlay_added = False
#     # pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
#     pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
#     if pts is not None:
#         xs, ys, labels = pts
#
#         for x, y_band, label in zip(xs, ys, labels):
#             if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
#                 y_idx = y_bands.index(y_band)
#
#                 # Fixed size square
#                 fig.add_shape(
#                     type="rect",
#                     x0=x - 0.05, y0=y_idx - 0.35,
#                     x1=x + 0.05, y1=y_idx + 0.35,
#                     line=dict(color="black", width=2),
#                     fillcolor="rgba(255,0,0,0.6)"
#                 )
#
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
#
#     # Update layout
#     fig.update_layout(
#         title=dict(
#         text=f"Proximity-Sensor Heatmap — Joint Number {pipe_number}",  # 👈 chart title
#         x=0.5,        # center
#         xanchor="center",
#         font=dict(size=18, family="Arial Black")),  # customize font,
#         xaxis_title=x_label,
#         yaxis_title=" ",
#         width=1500,
#         height=500,
#         font=dict(size=12),
#         xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
#         yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
#     )
#     fig.update_yaxes(autorange="reversed")
#     # Save the interactive heatmap
#     write_plotly_html(fig, f'{folder_path}/proximity_heatmap{pipe_number}.html')
#
#
#     print(f"Saved proximity heatmap: {folder_path}/proximity_heatmap{pipe_number}.html")
#     print(f"Overlays: {'Yes' if overlay_added else 'None found'}")
#
#
# def create_plots_hall(pkl_path, df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original):
#     folder_path = f'{output_folder}/Pipe_{pipe_number}'
#     os.makedirs(folder_path, exist_ok=True)
#
#     # MultilinePlot (offset stack of sensors)
#     save_lineplot(pkl_path, folder_path, test_val, datafile, pipe_number)
#
#     # 3D Pipe
#     save_pipe3d(test_val, test_val, folder_path, pipe_number, pkl_path)
#
#     # inside create_plots(...) or wherever you save other charts:
#     # save_proximity_linechart(folder_path, datafile, pipe_number)
#
#     save_interactive_heatmap(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10)
#
#     # save_interactive_heatmap_proximity(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10)
#
#
# def create_plots_proximity(pkl_path, df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder,df_new_tab10, datafile_original):
#     folder_path = f'{output_folder}/Pipe_{pipe_number}'
#     os.makedirs(folder_path, exist_ok=True)
#
#     # inside create_plots(...) or wherever you save other charts:
#     save_proximity_linechart(folder_path, datafile, pipe_number)
#     save_interactive_heatmap_proximity(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number,df_new_tab10)
#
#
#
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
#     write_plotly_html(fighm, f'{folder_path}/heatmap{pipe_number}.html')
#
#
#
# # case-insensitive F#P# like F1P1, F12P3
# FP_PATTERN = re.compile(r'^F\d+P\d+$', re.IGNORECASE)
#
#
# # def save_proximity_linechart(
# #     folder_path: str,
# #     datafile: pd.DataFrame,
# #     pipe_number,
# #     *,
# #     offset_step: float = 0.10,
# #     dtick: int = 1000,
# #     x_pref: str = "auto"  # "auto" -> ODDO1 if available, else index
# # ):
# #     """
# #     Proximity linechart:
# #       - selects columns matching F#P# (case-insensitive),
# #       - forward-fills data,
# #       - MinMax scales each series to [0,1],
# #       - offsets each series by `offset_step`,
# #       - X-axis = ODDO1 (meters) if present (or if x_pref='oddo1'), else index,
# #       - saves HTML to {folder_path}/proximity_linechart{pipe_number}.html
# #       - ADDS defect markers/labels as vertical lines using PipeTally overlays
# #     """
# #     df = datafile.copy()
# #
# #     # 1) collect F*P* columns
# #     candidates = [c for c in df.columns if isinstance(c, str) and FP_PATTERN.match(c.strip())]
# #     if not candidates:
# #         print(f"No F#P# columns found for pipe {pipe_number}. Skipping proximity linechart.")
# #         return
# #
# #     # 2) ensure numeric (coerce where possible)
# #     res_cols = []
# #     for c in candidates:
# #         if not is_numeric_dtype(df[c]):
# #             coerced = pd.to_numeric(df[c], errors='coerce')
# #             if coerced.notna().any():
# #                 df[c] = coerced
# #         if is_numeric_dtype(df[c]):
# #             res_cols.append(c)
# #     if not res_cols:
# #         print(f"No numeric F#P# columns for pipe {pipe_number}. Skipping proximity linechart.")
# #         return
# #
# #     # 3) forward-fill
# #     df1 = df.fillna(method='ffill')
# #
# #     # 4) choose x-axis (ODDO1 -> index)
# #     if x_pref.lower() == "oddo1" or (x_pref == "auto" and "ODDO1" in df1.columns):
# #         x_vals = (pd.to_numeric(df1["ODDO1"], errors="coerce") / 1000.0).round(3)
# #         x_label = "Abs. Distance (m) — ODDO1"
# #     else:
# #         x_vals = df1.index
# #         x_label = "Index"
# #
# #     # 5) MinMax scale selected columns
# #     scaler = MinMaxScaler()
# #     scaled = scaler.fit_transform(df1[res_cols].to_numpy())
# #     df1.loc[:, res_cols] = scaled
# #
# #     # 6) figure with offsets
# #     fig = go.Figure()
# #     for i, col in enumerate(res_cols):
# #         fig.add_trace(go.Scatter(
# #             x=x_vals,
# #             y=df1[col] + i * offset_step,
# #             name=col,
# #             mode='lines',
# #             line=dict(width=1),
# #             hoverinfo='x+y+name',
# #             showlegend=False
# #         ))
# #
# #     # 7) styling + axis titles + gridlines
# #     fig.update_layout(
# #         title=dict(
# #         text=f"Proximity-Sensor Lineplot — Joint Number {pipe_number}",  # 👈 chart title
# #         x=0.5,        # center
# #         xanchor="center",
# #         font=dict(size=18, family="Arial Black")),  # customize font,
# #         width=1500,
# #         height=650,
# #         margin=dict(l=10, b=20),
# #         paper_bgcolor="#ffffff",
# #         plot_bgcolor='rgb(255, 255, 255)',
# #         title_x=0.5,
# #         font={"family": "courier"},
# #         # legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
# #         xaxis_title=x_label,
# #         # yaxis_title="Scaled Proximity Sensor (0–1, offset)",
# #     )
# #     # keep titles close to axes
# #     fig.update_xaxes(title_standoff=8, automargin=True, dtick=dtick)
# #     fig.update_yaxes(title_standoff=10, automargin=True)
# #
# #     # light major gridlines
# #     fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)", gridwidth=1, zeroline=False)
# #     fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", gridwidth=1, zeroline=False)
# #
# #     # optional minor gridlines (comment out if your Plotly build lacks support)
# #     fig.update_xaxes(minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", gridwidth=0.5))
# #     fig.update_yaxes(minor=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", gridwidth=0.5))
# #
# #     # 8) add defect markers/labels as vertical lines (PipeTally overlays)
# #     try:
# #         x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
# #     except Exception:
# #         x_min, x_max = (float(df1.index.min()), float(df1.index.max()))
# #
# #     # We only need xs & labels here; pass a small dummy band list to satisfy the function
# #     y_bands_dummy = ["00:00", "06:00"]
# #     pts = _load_overlay_points_for_pipe(pipe_number, y_bands_dummy, folder_path)
# #     if pts is not None:
# #         xs, _, labels = pts
# #
# #         # keep only xs within chart window
# #         xs_labels = [(float(x), str(lbl)) for x, lbl in zip(xs, labels) if x_min <= float(x) <= x_max]
# #
# #         # group identical x's so we draw one line per location, stacking labels
# #         from collections import defaultdict
# #         at_x = defaultdict(list)
# #         for x, lbl in xs_labels:
# #             at_x[x].append(lbl)
# #
# #         for x, lbls in at_x.items():
# #             # vertical guide line
# #             fig.add_shape(
# #                 type="line",
# #                 x0=x, x1=x,
# #                 y0=0, y1=1,
# #                 xref="x",  yref="paper",
# #                 line=dict(color="black", width=1, dash="dot")
# #             )
# #             # label at top margin
# #             fig.add_annotation(
# #                 x=x, y=1.02, xref="x", yref="paper",
# #                 text=", ".join(lbls),
# #                 showarrow=False,
# #                 bgcolor="red",
# #                 bordercolor="black",
# #                 borderwidth=1,
# #                 font=dict(color="white", size=10, family="Arial Black"),
# #                 align="center"
# #             )
# #
# #     # 9) save EXACTLY as requested
# #     write_plotly_html(fig, f'{folder_path}/proximity_linechart{pipe_number}.html')
# #
# #     print(f"Saved {folder_path}/proximity_linechart{pipe_number}.html")
#
# def save_proximity_linechart(
#     folder_path: str,
#     datafile: pd.DataFrame,
#     pipe_number,
#     *,
#     offset_step: float = 0.10,
#     dtick: int = 1000,
#     x_pref: str = "auto"
# ):
#     import numpy as np
#     import pandas as pd
#     from sklearn.preprocessing import MinMaxScaler
#     from scipy.signal import lfilter
#     import plotly.graph_objects as go
#     from collections import defaultdict
#
#     df = datafile.copy()
#
#     # 1) collect F*P* columns
#     candidates = [c for c in df.columns if isinstance(c, str) and FP_PATTERN.match(c.strip())]
#     if not candidates:
#         print(f"No F#P# columns found for pipe {pipe_number}. Skipping.")
#         return
#
#     # 2) ensure numeric
#     res_cols = []
#     for c in candidates:
#         if not is_numeric_dtype(df[c]):
#             coerced = pd.to_numeric(df[c], errors='coerce')
#             if coerced.notna().any():
#                 df[c] = coerced
#         if is_numeric_dtype(df[c]):
#             res_cols.append(c)
#
#     if not res_cols:
#         print(f"No numeric F#P# columns for pipe {pipe_number}. Skipping.")
#         return
#
#     # 3) forward fill
#     df1 = df.fillna(method='ffill')
#
#     # 4) X-axis
#     if x_pref.lower() == "oddo1" or (x_pref == "auto" and "ODDO1" in df1.columns):
#         x_vals = (pd.to_numeric(df1["ODDO1"], errors="coerce") / 1000.0).round(3)
#         x_label = "Abs. Distance (m) — ODDO1"
#     else:
#         x_vals = df1.index
#         x_label = "Index"
#
#     # ------------------ ✅ PERFECT LOGIC START ------------------
#
#     # 5) MinMax scaling
#     scaler = MinMaxScaler()
#     scaled_values = scaler.fit_transform(df1[res_cols])
#     for i, col in enumerate(res_cols):
#         df1[col] = scaled_values[:, i]
#
#     # 6) smoothing + offset
#     n = 15
#     b = [1.0 / n] * n
#     a = 1
#
#     offsets = [round(i * offset_step, 3) for i in range(len(res_cols))]
#
#     fig = go.Figure()
#
#     for i, col in enumerate(res_cols):
#         yy = lfilter(b, a, df1[col])
#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=yy + offsets[i],
#             mode='lines',
#             line=dict(width=1),
#             hoverinfo='x+y+name',
#             showlegend=False,
#             name=col
#         ))
#
#     # ------------------ ✅ PERFECT LOGIC END ------------------
#
#     # 7) layout
#     fig.update_layout(
#         title=dict(
#             text=f"Proximity-Sensor Lineplot — Joint Number {pipe_number}",
#             x=0.5,
#             xanchor="center",
#             font=dict(size=18, family="Arial Black")
#         ),
#         width=1500,
#         height=500,
#         margin=dict(l=20, r=20, t=50, b=20),
#         template='plotly_white',
#         showlegend=False,
#         xaxis_title=x_label,
#     )
#
#
#     # 8) clean X ticks (no congestion)
#     num_ticks = 12
#     tick_positions = np.linspace(0, len(x_vals) - 1, num_ticks).astype(int)
#
#     if hasattr(x_vals, "iloc"):
#         tickvals = [x_vals.iloc[i] for i in tick_positions]
#     else:
#         tickvals = [x_vals[i] for i in tick_positions]
#
#     ticktext = [f"{v:.3f}" for v in tickvals]
#
#     fig.update_xaxes(
#         tickvals=tickvals,
#         ticktext=ticktext,
#         tickfont=dict(size=10),
#         tickangle=0,
#         showgrid=True,
#         gridcolor="rgba(0,0,0,0.10)"
#     )
#
#     fig.update_yaxes(
#         showgrid=True,
#         gridcolor="rgba(0,0,0,0.12)"
#     )
#     fig.update_yaxes(autorange="reversed")
#     # ------------------ OVERLAYS (UNCHANGED) ------------------
#
#     try:
#         x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
#     except Exception:
#         x_min, x_max = (float(df1.index.min()), float(df1.index.max()))
#
#     y_bands_dummy = ["00:00", "06:00"]
#     pts = _load_overlay_points_for_pipe(pipe_number, y_bands_dummy, folder_path)
#
#     if pts is not None:
#         xs, _, labels = pts
#
#         xs_labels = [(float(x), str(lbl)) for x, lbl in zip(xs, labels) if x_min <= float(x) <= x_max]
#
#         at_x = defaultdict(list)
#         for x, lbl in xs_labels:
#             at_x[x].append(lbl)
#
#         for x, lbls in at_x.items():
#             fig.add_shape(
#                 type="line",
#                 x0=x, x1=x,
#                 y0=0, y1=1,
#                 xref="x", yref="paper",
#                 line=dict(color="black", width=1, dash="dot")
#             )
#
#             fig.add_annotation(
#                 x=x, y=1.02, xref="x", yref="paper",
#                 text=", ".join(lbls),
#                 showarrow=False,
#                 bgcolor="red",
#                 bordercolor="black",
#                 borderwidth=1,
#                 font=dict(color="white", size=10, family="Arial Black"),
#                 align="center"
#             )
#
#     # 9) save
#     write_plotly_html(fig, f'{folder_path}/proximity_linechart{pipe_number}.html')
#
#     print(f"Saved {folder_path}/proximity_linechart{pipe_number}.html")
#
# # def save_lineplot(folder_path, test_val, datafile, pipe_number):
# #     figmlp = go.Figure()
# #     offset_step = 1200
# #     for idx, col in enumerate(test_val.columns):
# #         y_data = test_val[col].values
# #         offset_y_data = y_data + (idx * offset_step)
# #         figmlp.add_trace(go.Scatter(
# #             x=(datafile['ODDO1'] / 1000).round(2),
# #             y=offset_y_data,
# #             mode='lines',
# #             name=col,
# #             line=dict(width=1),
# #             hoverinfo='x+y+name',
# #             showlegend=False
# #         ))
# #
# #     # ----- DEFECT LABELS on hall-sensor stacked line plot -----
# #     try:
# #         x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
# #         y_bands = list(test_val.columns)  # orientation band labels (tick text)
# #         pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
# #         if pts is not None:
# #             xs, ys, labels = pts
# #             for x, y_band, label in zip(xs, ys, labels):
# #                 if y_band not in y_bands:
# #                     continue
# #                 # ensure x within range
# #                 if not (np.nanmin(x_vals) <= x <= np.nanmax(x_vals)):
# #                     continue
# #                 y_idx = y_bands.index(y_band)
# #                 y_pos = y_idx * offset_step
# #
# #                 # tiny marker at the band baseline
# #                 figmlp.add_trace(go.Scatter(
# #                     x=[x], y=[y_pos],
# #                     mode="markers",
# #                     marker=dict(size=8, line=dict(width=1, color="black")),
# #                     showlegend=False,
# #                     hoverinfo="skip"
# #                 ))
# #
# #                 # label with small arrow
# #                 figmlp.add_annotation(
# #                     x=x, y=y_pos,
# #                     text=str(label),
# #                     showarrow=True,
# #                     arrowhead=2, arrowsize=1, arrowwidth=1,
# #                     ax=0, ay=-20,  # nudge label above the baseline
# #                     bgcolor="red",
# #                     bordercolor="black",
# #                     font=dict(color="white", size=10, family="Arial Black")
# #                 )
# #     except Exception as e:
# #         print(f"Overlay labels on lineplot failed: {e}")
# #
# #     figmlp.update_layout(
# #         template='plotly_white',
# #         height=500,
# #         width=1500,
# #         margin=dict(l=20, r=20, t=50, b=20),
# #         title=dict(
# #         text=f"Hall-Sensor Lineplot — Joint Number {pipe_number}",  # 👈 chart title
# #         x=0.5,        # center
# #         xanchor="center",
# #         font=dict(size=18, family="Arial Black")  # customize font
# #     )
# #     )
# #     max_ticks = 15
# #     total = len(test_val.columns)
# #     step = 6
# #
# #     tick_indices = list(range(0, total, step))
# #
# #     figmlp.update_yaxes(
# #         tickmode='array',
# #         tickvals=[i * offset_step for i in tick_indices],
# #         ticktext=[test_val.columns[i] for i in tick_indices],
# #         tickfont=dict(size=9),
# #         autorange="reversed"
# #     )
# #     figmlp.update_yaxes(autorange="reversed")
# #     write_plotly_html(figmlp, f'{folder_path}/lineplot{pipe_number}.html')
# def save_lineplot(pkl_path, folder_path, test_val, datafile, pipe_number):
#     import numpy as np
#     import pandas as pd
#     from scipy.signal import savgol_filter, lfilter
#     import plotly.graph_objects as go
#
#     figmlp = go.Figure()
#     offset_step = 1400
#
#     print(f"pkl path recieved for pipe number: {pipe_number} is {pkl_path}")
#     df_pipe = pd.read_pickle(pkl_path)
#
#     F_columns = 36
#     res = [f'F{i}H{j}' for i in range(1, F_columns + 1) for j in range(1, 5)]
#
#     df1 = df_pipe[res].apply(pd.to_numeric, errors='coerce')
#     x_vals = df_pipe['index']
#
#     abs_dist_vals = (df_pipe['ODDO1'] / 1000).values
#
#     x_vals_arr = x_vals.values
#
#     def dist_to_index(dist_m):
#         idx = np.argmin(np.abs(abs_dist_vals - dist_m))
#         return float(x_vals_arr[idx])
#
#     window_length = 15
#     polyorder = 2
#     n = 15
#     b = [1.0 / n] * n
#     a = 1
#
#     for i, col in enumerate(res):
#         data = df1[col].values
#         time_index = np.arange(len(df1))
#
#         coeffs = np.polyfit(time_index, data, polyorder)
#         trend = np.polyval(coeffs, time_index)
#         detrended = data - trend
#
#         smoothed = savgol_filter(detrended, window_length, polyorder)
#         offset_data = smoothed + i * offset_step
#         filtered_data = lfilter(b, a, offset_data)
#
#         figmlp.add_trace(go.Scatter(
#             x=x_vals,
#             y=filtered_data,
#             mode='lines',
#             line=dict(width=1),
#             showlegend=False,
#             name=col,
#             customdata=abs_dist_vals,
#             hovertemplate=(
#                 "<b>%{fullData.name}</b><br>"
#                 "Index: %{x}<br>"
#                 "Abs Distance: %{customdata:.2f} m<br>"
#                 "Amplitude: %{y:.1f}<extra></extra>"
#             )
#         ))
#
#     # ── x-axis tick marks ──
#     valid_mask = ~np.isnan(abs_dist_vals)
#     if valid_mask.any():
#         all_x = x_vals.values[valid_mask]
#         all_d = abs_dist_vals[valid_mask]
#
#         n_ticks = 20
#         idx = np.round(np.linspace(0, len(all_x) - 1, n_ticks)).astype(int)
#         tick_x = all_x[idx]
#         tick_d = all_d[idx]
#
#         figmlp.update_xaxes(
#             tickmode='array',
#             tickvals=tick_x.tolist(),
#             ticktext=[f"{d:.1f}m" for d in tick_d],
#             tickangle=45,
#             tickfont=dict(size=8),
#             title_text="Abs Distance (m)"
#         )
#
#     # ---------- OVERLAYS ----------
#     try:
#         y_bands = list(test_val.columns)
#         pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
#
#         print(f"DEBUG: pts = {pts}")
#         print(f"DEBUG: y_bands sample = {y_bands[:5]}")
#         print(f"DEBUG: x_vals range = {float(np.nanmin(x_vals)):.2f} to {float(np.nanmax(x_vals)):.2f}")
#         print(f"DEBUG: abs_dist_vals range = {float(np.nanmin(abs_dist_vals)):.2f} to {float(np.nanmax(abs_dist_vals)):.2f}")
#
#         if pts is not None:
#             xs, ys, labels = pts
#             print(f"DEBUG: {len(xs)} overlay points loaded")
#             print(f"DEBUG: xs sample    = {xs[:3]}")
#             print(f"DEBUG: ys sample    = {ys[:3]}")
#             print(f"DEBUG: labels sample= {labels[:3]}")
#
#             for x_dist, y_band, label in zip(xs, ys, labels):
#                 print(f"\nDEBUG: processing x_dist={x_dist:.2f}, y_band='{y_band}', label='{label}'")
#
#                 if y_band not in y_bands:
#                     print(f"  → SKIPPED: y_band '{y_band}' not in y_bands")
#                     continue
#
#                 x = dist_to_index(x_dist)
#                 print(f"  → mapped x_dist={x_dist:.2f}m to index x={x:.2f}")
#
#                 if not (np.nanmin(x_vals) <= x <= np.nanmax(x_vals)):
#                     print(f"  → SKIPPED: x={x:.2f} out of range [{float(np.nanmin(x_vals)):.2f}, {float(np.nanmax(x_vals)):.2f}]")
#                     continue
#
#                 y_idx = y_bands.index(y_band)
#                 y_pos = y_idx * offset_step
#                 print(f"  → PLOTTING at x={x:.2f}, y_pos={y_pos}, y_idx={y_idx}")
#
#                 figmlp.add_trace(go.Scatter(
#                     x=[x],
#                     y=[y_pos],
#                     mode="markers",
#                     marker=dict(size=8, color="red", line=dict(width=1, color="black")),
#                     showlegend=False,
#                     name=f"{label} @ {x_dist:.2f}m",
#                     hovertemplate=f"<b>{label}</b><br>Abs Dist: {x_dist:.2f} m<br>Band: {y_band}<extra></extra>"
#                 ))
#
#                 figmlp.add_annotation(
#                     x=x, y=y_pos,
#                     text=str(label),
#                     showarrow=True, arrowhead=2,
#                     arrowsize=1, arrowwidth=1,
#                     ax=0, ay=-20,
#                     bgcolor="red", bordercolor="black",
#                     font=dict(color="white", size=10, family="Arial Black")
#                 )
#
#     except Exception as e:
#         import traceback
#         print(f"Overlay labels on lineplot failed: {e}")
#         traceback.print_exc()
#
#     # ---------- LAYOUT ----------
#     figmlp.update_layout(
#         template='plotly_white',
#         height=500, width=1500,
#         margin=dict(l=20, r=20, t=50, b=20),
#         showlegend=False,
#         title=dict(
#             text=f"Hall-Sensor Lineplot — Joint Number {pipe_number}",
#             x=0.5, xanchor="center",
#             font=dict(size=18, family="Arial Black")
#         )
#     )
#
#     # ---------- Y AXIS ----------
#     step = 6
#     total = len(res)
#     tick_indices = list(range(0, total, step))
#
#     figmlp.update_yaxes(
#         tickmode='array',
#         tickvals=[i * offset_step for i in tick_indices],
#         ticktext=[res[i] for i in tick_indices],
#         tickfont=dict(size=9),
#         autorange="reversed"
#     )
#
#     write_plotly_html(figmlp, f'{folder_path}/lineplot{pipe_number}.html')
#
#
#
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
#
#     figmlpraw.update_layout(
#         xaxis_title='Counter',
#         template='plotly_white',
#         height=300,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20)
#     )
#
#     write_plotly_html(figmlpraw, f'{folder_path}/lineplot_raw{pipe_number}.html')
#
#
#
# # def save_pipe3d(data, data_cp, folder_path, pipe_number):
# #     if not isinstance(data, np.ndarray):
# #         data = np.asarray(data)  # test_val
# #
# #     # ↓ add: subsample big grids
# #     if data.shape[0] > 1500:
# #         data = data[::2, :]          # halve rows
# #     if data.shape[1] > 128:
# #         data = data[:, ::1]          # keep cols (or ::2 if needed)
# #
# #     num_rows, num_cols = data.shape
# #
# #     theta = np.linspace(0, 2 * np.pi, num_cols)
# #     z = np.linspace(0, 1, num_rows)
# #     theta, z = np.meshgrid(theta, z)
# #
# #     radius = 109.5   # OD = 219mm, R = OD/2
# #     odometer = num_rows
# #
# #     # Cartesian Coords
# #     x = odometer * z
# #     y = radius * np.cos(theta)
# #     zc = radius * np.sin(theta)
# #
# #     fig = go.Figure(data=[go.Surface(
# #         x=x,
# #         y=zc,
# #         z=y,
# #         surfacecolor=data,
# #         colorscale='jet',
# #         customdata=data_cp
# #     )])
# #
# #     camera = dict(eye=dict(x=0., y=5, z=0.), up=dict(x=0, y=1, z=0))
# #
# #     odometer_start = 0
# #     odometer_end = odometer
# #
# #     fig.add_trace(go.Scatter3d(
# #         x=[odometer_start, odometer_end], y=[radius, radius], z=[0, 0],
# #         text=["3"], mode='text', textposition="middle center",
# #         marker=dict(size=0), name="3pm",
# #         textfont=dict(size=20, color="#61090c")
# #     ))
# #     fig.add_trace(go.Scatter3d(
# #         x=[odometer_start, odometer_end], y=[-radius, -radius], z=[0, 0],
# #         text=["9"], mode='text', textposition="middle center",
# #         marker=dict(size=0), name="9pm",
# #         textfont=dict(size=20, color="#61090c")
# #     ))
# #     fig.add_trace(go.Scatter3d(
# #         x=[odometer_start, odometer_end], y=[0, 0], z=[radius, radius],
# #         text=["6"], mode='text', textposition="middle center",
# #         marker=dict(size=0), name="6pm",
# #         textfont=dict(size=20, color="#61090c")
# #     ))
# #     fig.add_trace(go.Scatter3d(
# #         x=[odometer_start, odometer_end], y=[0, 0], z=[-radius, -radius],
# #         text=["12"], mode='text', textposition="middle center",
# #         marker=dict(size=0), name="12pm",
# #         textfont=dict(size=20, color="#61090c")
# #     ))
# #
# #     fig.update_layout(
# #         scene=dict(
# #             xaxis_title='Odometer',
# #             yaxis_title='Radial Length',
# #             zaxis_title='Radial Length',
# #             aspectmode='data',
# #             aspectratio=dict(x=1, y=1, z=0.5),
# #             camera=camera
# #         ),
# #         height=500,
# #         width=1500,
# #         title='Pipe Visualization',
# #         margin=dict(l=20, r=20, t=50, b=20),
# #     )
# #
# #     write_plotly_html(fig, f'{folder_path}/pipe3d{pipe_number}.html')
# def save_pipe3d(data, data_cp, folder_path, pipe_number, pkl_path):
#     import numpy as np
#     import pandas as pd
#     import plotly.graph_objects as go
#
#     df_pipe = pd.read_pickle(pkl_path)
#     oddo_vals = (df_pipe['ODDO1'] / 1000).values
#
#     if not isinstance(data, np.ndarray):
#         data = np.asarray(data)
#
#     if data.shape[0] > 1500:
#         data = data[::2, :]
#         oddo_vals = oddo_vals[::2]
#     if data.shape[1] > 128:
#         data = data[:, ::2]
#
#     num_rows, num_cols = data.shape
#
#     if len(oddo_vals) != num_rows:
#         oddo_vals = np.interp(
#             np.linspace(0, 1, num_rows),
#             np.linspace(0, 1, len(oddo_vals)),
#             oddo_vals
#         )
#
#     theta = np.linspace(0, 2 * np.pi, num_cols)
#     theta_grid, _ = np.meshgrid(theta, np.zeros(num_rows))
#
#     radius = 109.5
#     odometer_start = float(np.nanmin(oddo_vals))
#     odometer_end   = float(np.nanmax(oddo_vals))
#     dist_range     = odometer_end - odometer_start
#
#     x  = np.outer(oddo_vals, np.ones(num_cols))
#     y  = radius * np.cos(theta_grid)
#     zc = radius * np.sin(theta_grid)
#
#     fig = go.Figure(data=[go.Surface(
#         x=x, y=y, z=zc,
#         surfacecolor=data,
#         colorscale='jet',
#         customdata=data_cp,
#         showscale=False,
#         hovertemplate='Dist: %{x:.2f} m<br>Value: %{surfacecolor:.2f}<extra></extra>'
#     )])
#
#     clock_labels = [
#         dict(y=0,       z=radius,  text="12", name="12 o'clock"),
#         dict(y=radius,  z=0,       text="3",  name="3 o'clock"),
#         dict(y=0,       z=-radius, text="6",  name="6 o'clock"),
#         dict(y=-radius, z=0,       text="9",  name="9 o'clock"),
#     ]
#
#     for cl in clock_labels:
#         fig.add_trace(go.Scatter3d(
#             x=[odometer_start, odometer_end],
#             y=[cl['y'], cl['y']],
#             z=[cl['z'], cl['z']],
#             text=[cl['text'], cl['text']],
#             mode='text',
#             textposition="middle center",
#             marker=dict(size=0),
#             name=cl['name'],
#             textfont=dict(size=16, color="#61090c"),
#             showlegend=False
#         ))
#
#     # ── aspect ratio: normalize x to same scale as y/z ──
#     # x range is dist_range (metres), y/z range is 2*radius
#     # we want x to visually appear ~6x longer than the diameter
#     x_ratio = max(4, min((dist_range / (2 * radius)) * 6, 14))
#
#     # ── camera: distance scales with x_ratio so full pipe fits ──
#     cam_dist = 1.5 + x_ratio * 0.3  # pull back more for longer pipes
#     camera = dict(
#         eye=dict(x=cam_dist, y=-cam_dist * 0.4, z=cam_dist * 0.5),
#         up=dict(x=0, y=0, z=1),
#         center=dict(x=0, y=0, z=0)
#     )
#
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='Abs Distance (m)',
#             yaxis_title='',
#             zaxis_title='',
#             aspectmode='manual',
#             aspectratio=dict(x=x_ratio, y=1, z=1),
#             camera=camera,
#             xaxis=dict(
#                 showgrid=True,
#                 range=[odometer_start, odometer_end],
#                 autorange=False
#             ),
#             yaxis=dict(showgrid=False, showticklabels=False),
#             zaxis=dict(showgrid=False, showticklabels=False),
#         ),
#         scene_dragmode='orbit',
#         height=600,
#         width=1500,
#         title=dict(
#             text=f'Pipe 3D Visualization — Joint {pipe_number}',
#             x=0.5, xanchor='center',
#             font=dict(size=18, family='Arial Black')
#         ),
#         margin=dict(l=20, r=20, t=50, b=20),
#     )
#
#     write_plotly_html(fig, f'{folder_path}/pipe3d{pipe_number}.html')
#
#
#
# # -------------------- PARALLEL HELPERS --------------------
# def _resolve_workers(workers):
#     if workers in (None, 0, -1, "auto"):
#         cpu = os.cpu_count() or 1
#         return max(1, cpu - 1)
#     if isinstance(workers, int):
#         cpu = os.cpu_count() or 1
#         return max(1, min(workers, cpu))
#     return 1
#
#
# # def _process_one_pkl(pkl_path, output_folder):
# #     try:
# #         pipe_number = Path(pkl_path).stem
# #         pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
# #         pipe_folder.mkdir(exist_ok=True)
#
# #         data = pd.read_pickle(pkl_path)
# #         total_sensors_count_hall, column_names_hall, minute_sensors_hall, degree_sensors_hall = count_pattern_minute_degree(pkl_path)
# #         print(f" total_sensors_count_hall: {total_sensors_count_hall}, column_names_hall: {column_names_hall}, minute_sensors_hall: {minute_sensors_hall}, degree_sensors_hall: {degree_sensors_hall}")
# #         dfile = pre_process_data(data, pipe_number, output_folder, total_sensors_count_hall, column_names_hall, minute_sensors_hall, degree_sensors_hall)
#
# #         # Save the Excel
# #         xlsx_path = pipe_folder / f"Pipe_{pipe_number}.xlsx"
# #         dfile.to_excel(xlsx_path, index=False)
#
# #         return f"Processed {os.path.basename(pkl_path)} and saved to {pipe_folder}"
# #     except Exception as e:
# #         print(f"Error loading {os.path.basename(pkl_path)}: {e}")
# #         traceback.print_exc()
# #         return f"Error loading {os.path.basename(pkl_path)}: {e}"
#
# import time
# from datetime import datetime
#
# def _process_one_pkl(pkl_path, output_folder):
#     start_time = time.time()
#     start_clock = datetime.now().strftime("%H:%M:%S")
#
#
#     sensor_type = ["Hall", "Proximity"]
#     try:
#         pipe_number = int(Path(pkl_path).stem)
#         print(f"\n🟢 PIPE {pipe_number} START: {start_clock}", flush=True)
#         pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
#         pipe_folder.mkdir(exist_ok=True)
#
#         data = pd.read_pickle(pkl_path)
#         hall, prox = count_pattern_minute_degree(pkl_path)
#
#         #hall sensors feature data
#         total_sensors_count_hall = hall["count"]
#         column_names_hall = hall["columns"]
#         minute_sensors_hall = hall["minute"]
#         degree_sensors_hall = hall["degree"]
#
#         #proximity sensor feature data
#         total_sensors_count_prox = prox["count"]
#         column_names_prox = prox["columns"]
#         minute_sensors_prox = prox["minute"]
#         degree_sensors_prox = prox["degree"]
#
#         # for sensor_type in sensor_type:
#         #     if sensor_type == "Hall":
#         #         print(f"HALL SENSOR DETAILS ----> total_sensors_count_hall: {total_sensors_count_hall}, column_names_hall: {column_names_hall}, minute_sensors_hall: {minute_sensors_hall}, degree_sensors_hall: {degree_sensors_hall}\n\n")
#         #         dfile = pre_process_data(data, pipe_number, output_folder, total_sensors_count_hall, column_names_hall, minute_sensors_hall, degree_sensors_hall, sensor_type)
#         #     else:
#         #         dfile, df_new_tab9, datafile_original, test_val, map_ori_sens, df_new_tab10= pre_process_data(data, pipe_number, output_folder, total_sensors_count_prox, column_names_prox, minute_sensors_prox, degree_sensors_prox, sensor_type)
#         #
#
#         # Replace your current sensor loop with this fully wrapped debug version
#
#         for current_sensor_type in sensor_type:
#             try:
#                 print(
#                     f"\n==================================================\n"
#                     f"🚀 START SENSOR TYPE: {current_sensor_type}\n"
#                     f"PIPE: {pipe_number}\n"
#                     f"==================================================",
#                     flush=True
#                 )
#
#                 # -----------------------------
#                 # HALL
#                 # -----------------------------
#                 if current_sensor_type == "Hall":
#                     print(
#                         f"📌 HALL SENSOR DETAILS:\n"
#                         f"total_sensors_count_hall: {total_sensors_count_hall}\n"
#                         f"column_count_hall: {len(column_names_hall)}\n"
#                         f"minute_sensors_hall: {minute_sensors_hall}\n"
#                         f"degree_sensors_hall: {degree_sensors_hall}\n"
#                         f"first_10_cols: {column_names_hall[:10]}\n",
#                         flush=True
#                     )
#
#                     print("➡ ABOUT TO ENTER pre_process_data(Hall)", flush=True)
#
#                     dfile = pre_process_data(
#                         pkl_path,
#                         data,
#                         pipe_number,
#                         output_folder,
#                         total_sensors_count_hall,
#                         column_names_hall,
#                         minute_sensors_hall,
#                         degree_sensors_hall,
#                         current_sensor_type,
#                         debug=True
#                     )
#
#                     print("✅ FINISHED pre_process_data(Hall)", flush=True)
#
#                 # -----------------------------
#                 # PROXIMITY
#                 # -----------------------------
#                 else:
#                     print(
#                         f"📌 PROX SENSOR DETAILS:\n"
#                         f"total_sensors_count_prox: {total_sensors_count_prox}\n"
#                         f"column_count_prox: {len(column_names_prox)}\n"
#                         f"minute_sensors_prox: {minute_sensors_prox}\n"
#                         f"degree_sensors_prox: {degree_sensors_prox}\n"
#                         f"first_10_cols: {column_names_prox[:10]}\n",
#                         flush=True
#                     )
#
#                     print("➡ ABOUT TO ENTER pre_process_data(Proximity)", flush=True)
#
#                     (
#                         dfile,
#                         df_new_tab9,
#                         datafile_original,
#                         test_val,
#                         map_ori_sens,
#                         df_new_tab10
#                     ) = pre_process_data(
#                         pkl_path,
#                         data,
#                         pipe_number,
#                         output_folder,
#                         total_sensors_count_prox,
#                         column_names_prox,
#                         minute_sensors_prox,
#                         degree_sensors_prox,
#                         current_sensor_type,
#                         debug=True
#                     )
#
#                     print("✅ FINISHED pre_process_data(Proximity)", flush=True)
#
#                 print(
#                     f"🏁 COMPLETED SENSOR TYPE: {current_sensor_type}\n",
#                     flush=True
#                 )
#
#             except Exception as e:
#                 import traceback
#
#                 print(
#                     f"\n❌ CRASH INSIDE SENSOR LOOP\n"
#                     f"PIPE: {pipe_number}\n"
#                     f"SENSOR TYPE: {current_sensor_type}\n"
#                     f"ERROR: {str(e)}\n"
#                     f"{traceback.format_exc()}\n",
#                     flush=True
#                 )
#
#                 # Optional hard stop so you instantly know where it failed
#                 raise
#         # Save the Excel
#         xlsx_path = pipe_folder / f"Pipe_{pipe_number}.xlsx"
#         dfile.to_excel(xlsx_path, index=False)
#
#         end_time = time.time()
#         end_clock = datetime.now().strftime("%H:%M:%S")
#
#         total_time = round(end_time - start_time, 2)
#
#         print(
#             f"🔴 PIPE {pipe_number} END: {end_clock} | "
#             f"TOTAL: {total_time}s ({round(total_time / 60, 2)} min)\n",
#             flush=True
#         )
#
#         return f"Processed {os.path.basename(pkl_path)} and saved to {pipe_folder}"
#     except Exception as e:
#         print(f"Error loading {os.path.basename(pkl_path)}: {e}")
#         traceback.print_exc()
#         return f"Error loading {os.path.basename(pkl_path)}: {e}"
#
#
# # def count_pattern_minute_degree(datafile_path):
# #     df = pd.read_pickle(datafile_path)
# #     pattern_hall = re.compile(r'^F\d+H\d+$')
# #     matching_columns_hall = [col for col in df.columns if pattern_hall.match(col)]
# #     count_hall = len(matching_columns_hall)
# #     minute_sensors_hall = 720 / count_hall
# #     degree_sensors_hall = minute_sensors_hall / 2
# #     return count_hall, matching_columns_hall, minute_sensors_hall , degree_sensors_hall
#
#
# def count_pattern_minute_degree(datafile_path):
#     df = pd.read_pickle(datafile_path)
#
#     pattern_hall = re.compile(r'^F\d+H\d+$', re.IGNORECASE)
#     pattern_proximity = re.compile(r'^F\d+P\d+$', re.IGNORECASE)
#     matching_columns_hall = [col for col in df.columns if pattern_hall.match(col)]
#     matching_columns_proximity = [col for col in df.columns if pattern_proximity.match(col)]
#
#     count_hall = len(matching_columns_hall)
#     minute_sensors_hall = 720 / count_hall if count_hall > 0 else None
#     degree_sensors_hall = minute_sensors_hall / 2 if count_hall > 0 else None
#
#     count_proximity = len(matching_columns_proximity)
#     minute_sensors_proximity = 720 / count_proximity if count_proximity > 0 else None
#     degree_sensors_proximity = minute_sensors_proximity / 2 if count_proximity > 0 else None
#
#
#     hall = {
#         "columns": matching_columns_hall,
#         "count": count_hall,
#         "minute": minute_sensors_hall,
#         "degree": degree_sensors_hall
#     }
#
#     proximity = {
#         "columns": matching_columns_proximity,
#         "count": count_proximity,
#         "minute": minute_sensors_proximity,
#         "degree": degree_sensors_proximity
#     }
#     print(f"hall sensor details: {hall} \n proxiity sensor details: {proximity}")
#
#
#     return hall , proximity
#
#
#
# # def create_html_and_csv_from_pkl(
# #     pkl_folder='pipes3',
# #     output_folder='Client_Pipes',
# #     output_callback=None,
# #     workers=WORKERS
# # ):
# #     Path(output_folder).mkdir(exist_ok=True)
# #
# #     # collect .pkl file paths
# #     pkl_paths = [
# #         str(Path(pkl_folder) / f)
# #         for f in os.listdir(pkl_folder)
# #         if f.lower().endswith('.pkl')
# #     ]
# #
# #     if not pkl_paths:
# #         msg = f"No .pkl files found in {pkl_folder}"
# #         if output_callback: output_callback(msg)
# #         else: print(msg)
# #         return
# #
# #     n_jobs = _resolve_workers(workers)
# #
# #     # fan out work across processes
# #     results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
# #         delayed(_process_one_pkl)(p, output_folder) for p in pkl_paths
# #     )
# #
# #     # report
# #     for msg in results:
# #         if output_callback: output_callback(msg)
# #         else: print(msg)
# def create_html_and_csv_from_pkl(
#     pkl_folder='pipes3',
#     output_folder='Client_Pipes',
#     output_callback=None,
#     workers=WORKERS
# ):
#     import os
#     import traceback
#     from pathlib import Path
#     from joblib import Parallel, delayed
#
#     # -----------------------------
#     # Ensure output folder exists
#     # -----------------------------
#     try:
#         Path(output_folder).mkdir(parents=True, exist_ok=True)
#     except Exception as e:
#         err = f"❌ Failed creating output folder '{output_folder}': {e}\n{traceback.format_exc()}"
#         if output_callback:
#             output_callback(err)
#         else:
#             print(err)
#         return
#
#     # -----------------------------
#     # Collect PKL paths
#     # -----------------------------
#     try:
#         pkl_paths = [
#             str(Path(pkl_folder) / f)
#             for f in os.listdir(pkl_folder)
#             if f.lower().endswith('.pkl')
#         ]
#     except Exception as e:
#         err = f"❌ Failed reading PKL folder '{pkl_folder}': {e}\n{traceback.format_exc()}"
#         if output_callback:
#             output_callback(err)
#         else:
#             print(err)
#         return
#
#     # -----------------------------
#     # No files found
#     # -----------------------------
#     if not pkl_paths:
#         msg = f"⚠ No .pkl files found in {pkl_folder}"
#         if output_callback:
#             output_callback(msg)
#         else:
#             print(msg)
#         return
#
#     # -----------------------------
#     # Resolve worker count
#     # -----------------------------
#     try:
#         n_jobs = _resolve_workers(workers)
#     except Exception as e:
#         err = f"❌ Worker resolution failed: {e}\n{traceback.format_exc()}"
#         if output_callback:
#             output_callback(err)
#         else:
#             print(err)
#         return
#
#     # -----------------------------
#     # Safe wrapper for each PKL
#     # -----------------------------
#     # def safe_process(pkl_path):
#     #     try:
#     #         return _process_one_pkl(pkl_path, output_folder)
#     #
#     #     except Exception as e:
#     #         return (
#     #             f"\n❌ ERROR processing file: {pkl_path}\n"
#     #             f"Reason: {str(e)}\n"
#     #             f"{traceback.format_exc()}\n"
#     #         )
#     def safe_process(pkl_path):
#         import os
#         import time
#         import traceback
#         import pandas as pd
#
#         fname = os.path.basename(pkl_path)
#         start = time.time()
#
#         try:
#             # -----------------------------
#             # Basic file load test only
#             # -----------------------------
#             print(f"\n🔍 STARTING: {fname}")
#
#             try:
#                 test_obj = pd.read_pickle(pkl_path)
#                 print(f"✅ PKL LOAD OK: {fname}")
#
#                 if hasattr(test_obj, "shape"):
#                     print(f"📏 SHAPE: {test_obj.shape}")
#
#                 if hasattr(test_obj, "columns"):
#                     print(f"📋 COLUMN COUNT: {len(test_obj.columns)}")
#
#             except Exception as load_err:
#                 return (
#                     f"\n❌ PKL LOAD FAILED: {fname}\n"
#                     f"{str(load_err)}\n"
#                     f"{traceback.format_exc()}"
#                 )
#
#             # -----------------------------
#             # Actual pipeline
#             # -----------------------------
#             result = _process_one_pkl(pkl_path, output_folder)
#
#             elapsed = round(time.time() - start, 2)
#
#             print(f"🏁 FINISHED: {fname} in {elapsed}s")
#
#             return result
#
#         except Exception as e:
#             elapsed = round(time.time() - start, 2)
#
#             return (
#                 f"\n❌ ERROR processing file: {fname}\n"
#                 f"⏱ Failed after: {elapsed}s\n"
#                 f"Reason: {str(e)}\n"
#                 f"{traceback.format_exc()}\n"
#             )
#
#     # -----------------------------
#     # Parallel processing
#     # -----------------------------
#     try:
#         results = Parallel(
#             n_jobs=n_jobs,
#             backend="loky",
#             prefer="processes"
#         )(
#             delayed(safe_process)(p) for p in pkl_paths
#         )
#
#     except Exception as e:
#         err = f"❌ Parallel execution crashed: {e}\n{traceback.format_exc()}"
#         if output_callback:
#             output_callback(err)
#         else:
#             print(err)
#         return
#
#     # -----------------------------
#     # Report all results/errors
#     # -----------------------------
#     for msg in results:
#         if output_callback:
#             output_callback(msg)
#         else:
#             print(msg)
#
# # -------------------- MAIN --------------------
# if __name__ == "__main__":
#     import time
#     st = time.time()
#     # override cores at runtime if needed, e.g., workers=4
#     create_html_and_csv_from_pkl(workers=WORKERS)
#     print(f'Total time: {time.time()-st} seconds')


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
from datetime import datetime, timedelta
import traceback

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------ Plotly export options ------------
PLOTLY_JS_MODE   = "directory"
PLOTLY_COMPRESS  = True
HTML_DEFAULT_W   = "100%"
HTML_DEFAULT_H   = 500

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "displayModeBar": True,
    "scrollZoom": False,
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
INITIAL_READ = 0.0
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3
WORKERS = 4
# ------------------------------------------------


# ============================================================
#  OPTIMIZED CORE HELPERS
# ============================================================

def _degrees_to_hhmm_vectorized(degrees_arr: np.ndarray) -> np.ndarray:
    """
    Vectorised replacement for the per-cell degrees_to_hours_minutes2 loop.
    Returns an object array of 'HH:MM' strings.
    """
    d = degrees_arr % 360                          # [0, 360)
    total_seconds = d * (12 * 3600 / 360)         # map to 12h dial
    hours   = (total_seconds // 3600).astype(int) % 12
    minutes = ((total_seconds % 3600) // 60).astype(int)

    # vectorised string formatting
    hh = np.char.zfill(hours.astype(str),   2)
    mm = np.char.zfill(minutes.astype(str), 2)
    return np.char.add(np.char.add(hh, ':'), mm)


def _fix_hour_prefix_vectorized(arr: np.ndarray) -> np.ndarray:
    """Replace 23:→11:, 22:→10:, 12:→00: in a string array (vectorised)."""
    replacements = [('23:', '11:'), ('22:', '10:'), ('12:', '00:')]
    out = arr.copy()
    for old, new in replacements:
        mask = np.char.startswith(out, old)
        if mask.any():
            out[mask] = np.char.replace(out[mask], old, new)
    return out


def _build_clock_data_fast(oriData: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised replacement for oriData.applymap(degrees_to_hours_minutes2)
    + the four applymap string-replace calls + pd.to_datetime conversion.

    Returns a DataFrame of 'HH:MM' strings — same shape as oriData.
    """
    arr = oriData.to_numpy(dtype=float)           # (n_rows, n_cols)
    flat = arr.ravel()

    hhmm_flat = _degrees_to_hhmm_vectorized(flat)
    hhmm_2d   = hhmm_flat.reshape(arr.shape)

    # fix hour prefixes row-by-row via vectorised string ops
    for col_i in range(hhmm_2d.shape[1]):
        hhmm_2d[:, col_i] = _fix_hour_prefix_vectorized(hhmm_2d[:, col_i])

    return pd.DataFrame(hhmm_2d, columns=oriData.columns, index=oriData.index)


def _remap_sensor_timeline_fast(
    test_clockData: pd.DataFrame,
    rang: list,
    total_sensors: int,
) -> tuple:
    """
    Vectorised replacement for the row-by-row Python loop that builds
    time_dict_1 (stage 22).

    Returns (map_ori_sens, val_ori_sens_labels) where:
      map_ori_sens      — DataFrame of tuples (col_idx, col_name, hhmm_str)
      val_ori_sens_labels — DataFrame of sensor-column-name strings (for stage 26 lookup)
    """
    n_rows, n_cols = test_clockData.shape
    col_names = list(test_clockData.columns)

    # Convert HH:MM → integer seconds once for every cell
    def hhmm_to_sec(hhmm_2d: np.ndarray) -> np.ndarray:
        h = hhmm_2d[:, :, :2].view(np.uint8)   # won't work on string arrays
        # simpler: parse via vectorised split
        flat = hhmm_2d.ravel()
        secs = np.array([int(s[:2]) * 60 + int(s[3:]) for s in flat], dtype=np.int32)
        return secs.reshape(hhmm_2d.shape)

    # Build a (n_rows, n_cols) int array of seconds
    raw = test_clockData.to_numpy()             # dtype=object, strings 'HH:MM'
    flat_raw = raw.ravel()
    flat_secs = np.array(
        [int(s[:2]) * 60 + int(s[3:]) for s in flat_raw],
        dtype=np.int32
    ).reshape(n_rows, n_cols)

    # Target seconds for rang[0] (first band)
    start_sec = int(rang[0][:2]) * 60 + int(rang[0][3:])

    # For each row, find the column whose value == rang[0] (or nearest)
    diff = np.abs(flat_secs - start_sec)          # (n_rows, n_cols)
    ind_arr = np.argmin(diff, axis=1)             # (n_rows,) — start column per row

    # Build the remapped label matrix using advanced indexing
    # For row r, the output column c gets col_names[(ind_arr[r] + c) % n_cols]
    col_idx_grid = (ind_arr[:, None] + np.arange(n_cols)[None, :]) % n_cols  # (n_rows, n_cols)

    # label matrix: sensor column names, shape (n_rows, n_cols)
    col_names_arr = np.array(col_names)
    label_matrix = col_names_arr[col_idx_grid]    # (n_rows, n_cols)

    # hhmm values at those positions
    row_idx = np.arange(n_rows)[:, None]
    hhmm_matrix = raw[row_idx, col_idx_grid]      # (n_rows, n_cols)

    # Build map_ori_sens as a DataFrame of tuples
    # Each cell: (col_idx, col_name, hhmm_str)
    # Store as three separate arrays for speed, merge into tuples only when needed
    map_ori_sens_data = np.empty((n_rows, n_cols), dtype=object)
    for c in range(n_cols):
        col_indices = col_idx_grid[:, c]
        for r in range(n_rows):
            map_ori_sens_data[r, c] = (
                int(col_indices[r]),
                label_matrix[r, c],
                hhmm_matrix[r, c]
            )

    map_ori_sens = pd.DataFrame(
        map_ori_sens_data,
        columns=rang[:n_cols],
        index=test_clockData.index
    )

    val_ori_sens_labels = pd.DataFrame(
        label_matrix,
        columns=rang[:n_cols],
        index=test_clockData.index
    )

    return map_ori_sens, val_ori_sens_labels


def _map_sensor_values_fast(
    val_ori_sens_labels: pd.DataFrame,
    df_new_tab9: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vectorised replacement for stage 26 (the 4.3M-iteration nested loop).

    val_ori_sens_labels[r, c] = column name in df_new_tab9 to look up for row r.
    We want test_val[r, c] = df_new_tab9.at[r, col_name].

    Strategy: for each unique column name that appears in val_ori_sens_labels,
    scatter its entire column vector into the right positions of the output matrix.
    This turns O(n_rows * n_cols) individual .at lookups into O(unique_cols) bulk
    numpy assignments.
    """
    n_rows, n_cols = val_ori_sens_labels.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)

    label_arr = val_ori_sens_labels.to_numpy()   # (n_rows, n_cols) strings

    # Convert df_new_tab9 to numpy for O(1) column access
    col_to_idx = {col: i for i, col in enumerate(df_new_tab9.columns)}
    data_mat   = df_new_tab9.to_numpy(dtype=np.float64)  # (n_rows, n_sensors)

    for c in range(n_cols):
        col_labels = label_arr[:, c]             # (n_rows,) — one sensor name per row

        # group rows by which sensor column they point to
        unique_sensors = np.unique(col_labels)
        for sensor in unique_sensors:
            if sensor not in col_to_idx:
                continue
            row_mask     = col_labels == sensor
            sensor_col_i = col_to_idx[sensor]
            out[row_mask, c] = data_mat[row_mask, sensor_col_i]

    return pd.DataFrame(
        out,
        columns=val_ori_sens_labels.columns,
        index=val_ori_sens_labels.index
    )


# ============================================================
#  PRE-PROCESS DATA  (drop-in replacement)
# ============================================================

def pre_process_data(
    pkl_path,
    datafile,
    pipe_number,
    output_folder,
    total_sensors,
    column_names,
    minute_sensors,
    degree_sensors,
    sensor_type,
    debug=False
):
    import time

    dbg_start = time.time()

    def dbg(msg):
        if debug:
            elapsed = round(time.time() - dbg_start, 2)
            print(f"[PIPE {pipe_number} | {sensor_type}] {msg} | {elapsed}s", flush=True)

    dbg("START pre_process_data")

    # ── initial copies ────────────────────────────────────────────────────────
    datafile_original = datafile.copy(deep=True)
    dbg("1: datafile_original copy complete")

    df_new_tab9 = pd.DataFrame(datafile, columns=column_names)
    dbg(f"2: df_new_tab9 created | shape={df_new_tab9.shape}")

    df_new_tab10 = df_new_tab9.copy()
    dbg("3: df_new_tab10 copy complete")

    sensor_columns = df_new_tab9.columns.tolist()
    dbg(f"4: sensor_columns loaded | count={len(sensor_columns)}")

    # ── Savitzky-Golay denoise ────────────────────────────────────────────────
    # OPTIMISATION: batch detrend + filter using matrix ops instead of per-column loop
    dbg("5: starting denoise (vectorised)")

    data_mat   = df_new_tab9.to_numpy(dtype=np.float64)          # (n, n_cols)
    time_index = np.arange(data_mat.shape[0], dtype=np.float64)

    # Fit degree-2 polynomial for every column at once via lstsq
    A      = np.column_stack([time_index**2, time_index, np.ones_like(time_index)])
    coeffs = np.linalg.lstsq(A, data_mat, rcond=None)[0]         # (3, n_cols)
    trends = A @ coeffs                                           # (n, n_cols)

    detrended = data_mat - trends
    smoothed  = np.apply_along_axis(
        lambda col: savgol_filter(col, 15, 2), axis=0, arr=detrended
    )
    df_new_tab9 = pd.DataFrame(smoothed, columns=sensor_columns, index=df_new_tab9.index)

    dbg("6: denoise complete (vectorised)")

    df_raw_straight = df_new_tab9.copy()
    dbg("7: df_raw_straight copy complete")

    # ── noise filtering ───────────────────────────────────────────────────────
    dbg("8: calculating means/std")
    sens_mean          = df_new_tab9.abs().mean()
    standard_deviation = df_new_tab9.std(axis=0, skipna=True)
    mean_plus_sigma    = sens_mean + UPPER_SENS_MUL * standard_deviation
    mean_negative_sigma= sens_mean - LOWER_SENS_MUL * standard_deviation

    dbg("9: applying noise filtering (vectorised)")
    mat   = df_new_tab9.to_numpy(dtype=np.float64)
    lower = mean_negative_sigma.to_numpy()
    upper = mean_plus_sigma.to_numpy()
    mask  = (mat >= lower) & (mat <= upper)
    mat[mask] = 0.0
    df_new_tab9 = pd.DataFrame(mat, columns=sensor_columns, index=df_new_tab9.index)
    dbg("10: noise filtering complete")

    # ── roll prep ─────────────────────────────────────────────────────────────
    initial_read = INITIAL_READ
    roll = (datafile['ROLL'] - initial_read).to_numpy(dtype=np.float64)
    dbg("11: roll normalization complete")

    # ── build oriData — vectorised ────────────────────────────────────────────
    dbg("12-15: building oriData (vectorised)")

    # Each row: [roll[r] + degree_sensors*i  for i in 0..total_sensors-1]
    offsets  = np.arange(total_sensors, dtype=np.float64) * degree_sensors  # (n_cols,)
    ori_mat  = roll[:, None] + offsets[None, :]                              # (n_rows, n_cols)

    col_labels_orig = [f'Roll_Sensor_{i}' for i in range(total_sensors)]
    oriData = pd.DataFrame(ori_mat, columns=col_labels_orig)
    dbg(f"15: oriData created | shape={oriData.shape}")

    # ── degrees → HH:MM  (vectorised, replaces applymap + 4 string-replace applymap) ──
    dbg("16-20: clockData (vectorised)")
    test_clockData = _build_clock_data_fast(oriData)
    test_clockData.columns = df_new_tab9.columns          # rename to sensor column names
    dbg("20: clockData renamed")

    # ── time dict ─────────────────────────────────────────────────────────────
    dbg("21: creating time dictionary")
    time_list     = [timedelta(minutes=i * minute_sensors) for i in range(total_sensors)]
    time_ranges_2 = [(datetime.min + t).strftime('%H:%M') for t in time_list]
    rang          = time_ranges_2                         # list of band labels

    # ── remap sensor timeline  (vectorised — replaces 29884-row Python loop) ──
    dbg("22: remapping sensor timeline (vectorised)")
    map_ori_sens, val_ori_sens_labels = _remap_sensor_timeline_fast(
        test_clockData, rang, total_sensors
    )
    dbg("23: timeline remap complete")
    dbg(f"24: map_ori_sens created | shape={map_ori_sens.shape}")

    # ── map sensor values  (vectorised — replaces 4.3M-iteration loop) ───────
    dbg("25-26: mapping sensor values (vectorised)")
    test_val = _map_sensor_values_fast(val_ori_sens_labels, df_new_tab9)
    dbg(f"26: test_val built | shape={test_val.shape}")

    # ── map_val_sens  (stage 27 — now trivially fast) ────────────────────────
    dbg("27: building map_val_sens")
    # Reconstruct map_val_sens as before (tuple of (col_idx, col_name, hhmm, value))
    # This is only used downstream if callers inspect it; build it efficiently.
    map_ori_arr  = map_ori_sens.to_numpy()               # object array of 3-tuples
    test_val_arr = test_val.to_numpy(dtype=np.float64)

    map_val_mat  = np.empty(map_ori_arr.shape, dtype=object)
    for c in range(map_ori_arr.shape[1]):
        for r in range(map_ori_arr.shape[0]):
            t = map_ori_arr[r, c]
            map_val_mat[r, c] = (*t, test_val_arr[r, c])

    map_val_sens = pd.DataFrame(
        map_val_mat, columns=map_ori_sens.columns, index=map_ori_sens.index
    )
    dbg("27: map_val_sens complete")

    # ── plotting ──────────────────────────────────────────────────────────────
    dbg("28: plotting stage")
    if sensor_type == "Hall":
        create_plots_hall(
            pkl_path, df_new_tab9, df_raw_straight, datafile,
            test_val, map_ori_sens, pipe_number, output_folder,
            df_new_tab10, datafile_original
        )
    else:
        create_plots_proximity(
            pkl_path, df_new_tab9, df_raw_straight, datafile,
            test_val, map_ori_sens, pipe_number, output_folder,
            df_new_tab10, datafile_original
        )

    dbg("29: COMPLETE pre_process_data")
    return (datafile, df_new_tab9, datafile_original, test_val, map_ori_sens, df_new_tab10)


# ============================================================
#  EVERYTHING BELOW IS UNCHANGED FROM ORIGINAL
# ============================================================

def _find_pipe_tally_file(pipe_number: Union[str, int], folder_path: str) -> Optional[str]:
    pn = str(pipe_number)
    tally_path = f"{folder_path}/PipeTally{pn}.csv"
    if os.path.exists(tally_path):
        return tally_path
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
    parts = str(t).split(":")
    if len(parts) == 3:
        h, m, s = [int(x) for x in parts]
    elif len(parts) == 2:
        h, m = [int(x) for x in parts]
        s = 0
    else:
        raise ValueError(f"Invalid time format: {t}")
    return (h % 12) * 3600 + m * 60 + s

def _nearest_band_label(seconds: int, band_labels: List[str]) -> str:
    band_labels_str = [str(x) for x in band_labels]
    band_secs = np.array([_hhmmss_to_seconds(lbl) for lbl in band_labels_str], dtype=int)
    idx = int(np.argmin(np.abs(band_secs - seconds)))
    return band_labels_str[idx]

def _load_overlay_points_for_pipe(pipe_number, y_band_labels, folder_path, *, debug_prefix="OVERLAY DEBUG"):
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

    if x_col is None or ori_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: missing required columns; no overlay.")
        return None
    if pipe_col is not None:
        mask = df[pipe_col].astype(str).str.contains(str(pipe_number), na=False)
        if mask.any():
            df = df[mask]
    total_rows = len(df)
    if feat_col is None:
        print(f"{debug_prefix}: pipe {pipe_number}: no feature column; overlays disabled.")
        return None
    feat_series = df[feat_col].astype(str).str.strip().str.lower()
    metal_loss_mask = feat_series.str.fullmatch(r"metal\s*loss", case=False, na=False)
    df = df[metal_loss_mask]
    after_filter = len(df)

    xs, ys, labels = [], [], []
    skipped_no_x, skipped_no_ori, skipped_other = 0, 0, 0

    for _, row in df.iterrows():
        x = pd.to_numeric(row.get(x_col), errors="coerce")
        if pd.isna(x):
            skipped_no_x += 1
            continue
        ori_sec = _parse_ori_to_seconds(row.get(ori_col))
        if ori_sec is None:
            skipped_no_ori += 1
            continue
        y = _nearest_band_label(int(ori_sec), list(y_band_labels))
        if y not in y_band_labels:
            skipped_other += 1
            continue
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
        f"\n{debug_prefix}: pipe {pipe_number}: total_rows={total_rows}, "
        f"after_feature='Metal Loss'={after_filter}, plotted={len(xs)}, "
        f"skipped_no_x={skipped_no_x}, skipped_no_ori={skipped_no_ori}, skipped_other={skipped_other}"
    )
    if not xs:
        return None
    return xs, ys, labels


def pre_process_for_interactive_heatmap(df_in: pd.DataFrame, datafile: pd.DataFrame, test_val: pd.DataFrame, map_ori_sens: pd.DataFrame):
    sens_cols = df_in.columns.tolist()
    print(f"Sensor columns in datafile for preprocess_heatmap_hall_sensor: {sens_cols}")
    if not sens_cols:
        raise ValueError("No sensor columns found in the input DataFrame.")

    df_sens = pd.DataFrame(datafile, columns=sens_cols).copy()
    df_sens_raw = df_sens.copy(deep=True)
    df_mean_cols = df_sens_raw
    Mean1 = df_mean_cols.mean()
    df_raw_plot = ((df_mean_cols - Mean1) / Mean1) * 100

    for col in sens_cols:
        df_sens[col] = pd.to_numeric(df_sens[col], errors='coerce')
    df_sens = df_sens.fillna(method='ffill').fillna(0.0)
    Mean1 = df_sens.mean()
    df_sens_normalized = ((df_sens - Mean1) / (Mean1 + 1e-8)) * 100

    for col in sens_cols:
        df_sens_normalized.loc[df_sens_normalized[col] > Mean1[col], col] = 0

    if UPPER_SENS_MUL > 0 and LOWER_SENS_MUL > 0:
        sens_std = df_sens.std(axis=0, skipna=True)
        upper = Mean1 + UPPER_SENS_MUL * sens_std
        lower = Mean1 - LOWER_SENS_MUL * sens_std
        for col in sens_cols:
            mask = (df_sens[col] >= lower[col]) & (df_sens[col] <= upper[col])
            df_sens_normalized.loc[mask, col] = 0

    df_plot_rearranged = test_val.copy()
    for r in range(len(test_val)):
        for c, band in enumerate(test_val.columns):
            sensor_info = map_ori_sens.iloc[r, c]
            if isinstance(sensor_info, tuple) and len(sensor_info) >= 2:
                sensor_col = sensor_info[1]
                if sensor_col in df_sens_normalized.columns:
                    df_plot_rearranged.iloc[r, c] = df_sens_normalized.loc[r, sensor_col]

    return df_plot_rearranged, df_raw_plot


def save_interactive_heatmap(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10):
    df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)

    if 'ODDO1' in datafile.columns:
        x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m) --- Hall Sensor Heatmap'
    else:
        x_vals = pd.Series(np.arange(len(datafile)))
        x_label = 'Index'

    y_bands = [str(c) for c in test_val.columns]
    heatmap_data = df_raw_plot.T

    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    heatmap_data = heatmap_data.astype("float32").round(3)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=x_vals.round(2), y=y_bands,
        colorscale='jet', zmin=-3, zmax=8, showscale=False, hoverongaps=False,
        hovertemplate='<b>%{x}</b><br><b>%{y}</b><br><b>Value: %{z:.2f}%</b><extra></extra>'
    ))

    overlay_added = False
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
    if pts is not None:
        xs, ys, labels = pts
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                fig.add_shape(type="rect",
                    x0=x-0.05, y0=y_idx-0.35, x1=x+0.05, y1=y_idx+0.35,
                    line=dict(color="black", width=2), fillcolor="rgba(255,0,0,0.6)")
                fig.add_annotation(x=x, y=y_idx, text=label, showarrow=False,
                    font=dict(color="white", size=8, family="Arial Black"),
                    bgcolor="red", bordercolor="black", borderwidth=1)
                overlay_added = True

    fig.update_layout(
        title=dict(text=f"Hall-Sensor Heatmap — Joint Number {pipe_number}",
            x=0.5, xanchor="center", font=dict(size=18, family="Arial Black")),
        xaxis_title=x_label, yaxis_title=" ", width=1500, height=500,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    fig.update_yaxes(autorange="reversed")
    write_plotly_html(fig, f'{folder_path}/hallsensor_heatmap{pipe_number}.html')
    print(f"Saved hallsensor heatmap: {folder_path}/hallsensor_heatmap{pipe_number}.html")
    print(f"Overlays: {'Yes' if overlay_added else 'None found'}")


def save_interactive_heatmap_proximity(df_new_tab9, datafile, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10):
    print("RUNNING PROXIMITY HEATMAP GENERATION \n")
    pattern = re.compile(r'^F\d+P\d+$')
    datafile_original = datafile.copy(deep=True)
    matching_columns = [col for col in datafile_original.columns if pattern.match(col)]
    df_new_tab9 = pd.DataFrame(datafile, columns=matching_columns)
    df_new_tab10 = df_new_tab9.copy()

    df_plot_rearranged, df_raw_plot = pre_process_for_interactive_heatmap(df_new_tab10, datafile, test_val, map_ori_sens)

    if 'ODDO1' in datafile.columns:
        x_vals = (pd.to_numeric(datafile['ODDO1'], errors='coerce') / 1000.0).round(2)
        x_label = 'Absolute Distance (m) --- Proximity Sensor Heatmap'
    else:
        x_vals = pd.Series(np.arange(len(datafile)))
        x_label = 'Index'

    y_bands = [str(c) for c in test_val.columns]
    heatmap_data = df_raw_plot.T

    for col in heatmap_data.columns:
        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
    heatmap_data = heatmap_data.fillna(0.0).astype(np.float64)
    if not np.isfinite(heatmap_data.values).all():
        heatmap_data = heatmap_data.replace([np.inf, -np.inf], 0.0)
    heatmap_data = heatmap_data.astype("float32").round(3)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=x_vals.round(2), y=y_bands,
        colorscale='jet', zmin=-3, zmax=8, showscale=False, hoverongaps=False,
        hovertemplate='<b>%{x}</b><br><b>%{y}</b><br><b>Value: %{z:.2f}%</b><extra></extra>'
    ))

    overlay_added = False
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
    if pts is not None:
        xs, ys, labels = pts
        for x, y_band, label in zip(xs, ys, labels):
            if x_vals.min() <= x <= x_vals.max() and y_band in y_bands:
                y_idx = y_bands.index(y_band)
                fig.add_shape(type="rect",
                    x0=x-0.05, y0=y_idx-0.35, x1=x+0.05, y1=y_idx+0.35,
                    line=dict(color="black", width=2), fillcolor="rgba(255,0,0,0.6)")
                fig.add_annotation(x=x, y=y_idx, text=label, showarrow=False,
                    font=dict(color="white", size=8, family="Arial Black"),
                    bgcolor="red", bordercolor="black", borderwidth=1)
                overlay_added = True

    fig.update_layout(
        title=dict(text=f"Proximity-Sensor Heatmap — Joint Number {pipe_number}",
            x=0.5, xanchor="center", font=dict(size=18, family="Arial Black")),
        xaxis_title=x_label, yaxis_title=" ", width=1500, height=500,
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    fig.update_yaxes(autorange="reversed")
    write_plotly_html(fig, f'{folder_path}/proximity_heatmap{pipe_number}.html')
    print(f"Saved proximity heatmap: {folder_path}/proximity_heatmap{pipe_number}.html")
    print(f"Overlays: {'Yes' if overlay_added else 'None found'}")


def create_plots_hall(pkl_path, df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder, df_new_tab10, datafile_original):
    folder_path = f'{output_folder}/Pipe_{pipe_number}'
    os.makedirs(folder_path, exist_ok=True)
    save_lineplot(pkl_path, folder_path, test_val, datafile, pipe_number)
    save_pipe3d(test_val, test_val, folder_path, pipe_number, pkl_path)
    save_interactive_heatmap(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10)


def create_plots_proximity(pkl_path, df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder, df_new_tab10, datafile_original):
    folder_path = f'{output_folder}/Pipe_{pipe_number}'
    os.makedirs(folder_path, exist_ok=True)
    save_proximity_linechart(folder_path, datafile, pipe_number)
    save_interactive_heatmap_proximity(df_new_tab9, datafile_original, test_val, map_ori_sens, folder_path, pipe_number, df_new_tab10)


def save_heatmap(test_val, datafile, map_ori_sens, folder_path, pipe_number):
    fighm = go.Figure(data=go.Heatmap(
        z=test_val.T, y=test_val.columns,
        x=(datafile['ODDO1'] / 1000).round(2),
        colorscale='jet',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],
    ))
    fighm.update_layout(xaxis_title='Absolute Distance (m)', height=500, width=1500,
                        margin=dict(l=20, r=20, t=50, b=20))
    write_plotly_html(fighm, f'{folder_path}/heatmap{pipe_number}.html')


FP_PATTERN = re.compile(r'^F\d+P\d+$', re.IGNORECASE)


def save_proximity_linechart(
    folder_path: str,
    datafile: pd.DataFrame,
    pipe_number,
    *,
    offset_step: float = 0.10,
    dtick: int = 1000,
    x_pref: str = "auto"
):
    from scipy.signal import lfilter
    from collections import defaultdict

    df = datafile.copy()
    candidates = [c for c in df.columns if isinstance(c, str) and FP_PATTERN.match(c.strip())]
    if not candidates:
        print(f"No F#P# columns found for pipe {pipe_number}. Skipping.")
        return

    res_cols = []
    for c in candidates:
        if not is_numeric_dtype(df[c]):
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().any():
                df[c] = coerced
        if is_numeric_dtype(df[c]):
            res_cols.append(c)

    if not res_cols:
        print(f"No numeric F#P# columns for pipe {pipe_number}. Skipping.")
        return

    df1 = df.fillna(method='ffill')

    if x_pref.lower() == "oddo1" or (x_pref == "auto" and "ODDO1" in df1.columns):
        x_vals = (pd.to_numeric(df1["ODDO1"], errors="coerce") / 1000.0).round(3)
        x_label = "Abs. Distance (m) — ODDO1"
    else:
        x_vals = df1.index
        x_label = "Index"

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df1[res_cols])
    for i, col in enumerate(res_cols):
        df1[col] = scaled_values[:, i]

    n = 15
    b = [1.0 / n] * n
    a = 1
    offsets = [round(i * offset_step, 3) for i in range(len(res_cols))]

    fig = go.Figure()
    for i, col in enumerate(res_cols):
        yy = lfilter(b, a, df1[col])
        fig.add_trace(go.Scatter(
            x=x_vals, y=yy + offsets[i], mode='lines',
            line=dict(width=1), hoverinfo='x+y+name',
            showlegend=False, name=col
        ))

    fig.update_layout(
        title=dict(text=f"Proximity-Sensor Lineplot — Joint Number {pipe_number}",
            x=0.5, xanchor="center", font=dict(size=18, family="Arial Black")),
        width=1500, height=500, margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_white', showlegend=False, xaxis_title=x_label,
    )

    num_ticks = 12
    tick_positions = np.linspace(0, len(x_vals) - 1, num_ticks).astype(int)
    if hasattr(x_vals, "iloc"):
        tickvals = [x_vals.iloc[i] for i in tick_positions]
    else:
        tickvals = [x_vals[i] for i in tick_positions]
    ticktext = [f"{v:.3f}" for v in tickvals]

    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickfont=dict(size=10),
                     tickangle=0, showgrid=True, gridcolor="rgba(0,0,0,0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(autorange="reversed")

    try:
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    except Exception:
        x_min, x_max = (float(df1.index.min()), float(df1.index.max()))

    y_bands_dummy = ["00:00", "06:00"]
    pts = _load_overlay_points_for_pipe(pipe_number, y_bands_dummy, folder_path)
    if pts is not None:
        xs, _, labels = pts
        xs_labels = [(float(x), str(lbl)) for x, lbl in zip(xs, labels) if x_min <= float(x) <= x_max]
        at_x = defaultdict(list)
        for x, lbl in xs_labels:
            at_x[x].append(lbl)
        for x, lbls in at_x.items():
            fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1,
                          xref="x", yref="paper",
                          line=dict(color="black", width=1, dash="dot"))
            fig.add_annotation(x=x, y=1.02, xref="x", yref="paper",
                                text=", ".join(lbls), showarrow=False,
                                bgcolor="red", bordercolor="black", borderwidth=1,
                                font=dict(color="white", size=10, family="Arial Black"),
                                align="center")

    write_plotly_html(fig, f'{folder_path}/proximity_linechart{pipe_number}.html')
    print(f"Saved {folder_path}/proximity_linechart{pipe_number}.html")


def save_lineplot(pkl_path, folder_path, test_val, datafile, pipe_number):
    from scipy.signal import savgol_filter, lfilter

    figmlp = go.Figure()
    offset_step = 1400

    print(f"pkl path received for pipe number: {pipe_number} is {pkl_path}")
    df_pipe = pd.read_pickle(pkl_path)

    F_columns = 36
    res = [f'F{i}H{j}' for i in range(1, F_columns + 1) for j in range(1, 5)]

    df1 = df_pipe[res].apply(pd.to_numeric, errors='coerce')
    x_vals = df_pipe['index']
    abs_dist_vals = (df_pipe['ODDO1'] / 1000).values
    x_vals_arr = x_vals.values

    def dist_to_index(dist_m):
        idx = np.argmin(np.abs(abs_dist_vals - dist_m))
        return float(x_vals_arr[idx])

    window_length = 15
    polyorder = 2
    n = 15
    b = [1.0 / n] * n
    a = 1

    for i, col in enumerate(res):
        data = df1[col].values
        time_index = np.arange(len(df1))
        coeffs = np.polyfit(time_index, data, polyorder)
        trend = np.polyval(coeffs, time_index)
        detrended = data - trend
        smoothed = savgol_filter(detrended, window_length, polyorder)
        offset_data = smoothed + i * offset_step
        filtered_data = lfilter(b, a, offset_data)
        figmlp.add_trace(go.Scatter(
            x=x_vals, y=filtered_data, mode='lines', line=dict(width=1),
            showlegend=False, name=col, customdata=abs_dist_vals,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Index: %{x}<br>"
                "Abs Distance: %{customdata:.2f} m<br>Amplitude: %{y:.1f}<extra></extra>"
            )
        ))

    valid_mask = ~np.isnan(abs_dist_vals)
    if valid_mask.any():
        all_x = x_vals.values[valid_mask]
        all_d = abs_dist_vals[valid_mask]
        n_ticks = 20
        idx = np.round(np.linspace(0, len(all_x) - 1, n_ticks)).astype(int)
        tick_x = all_x[idx]
        tick_d = all_d[idx]
        figmlp.update_xaxes(
            tickmode='array', tickvals=tick_x.tolist(),
            ticktext=[f"{d:.1f}m" for d in tick_d],
            tickangle=45, tickfont=dict(size=8), title_text="Abs Distance (m)"
        )

    try:
        y_bands = list(test_val.columns)
        pts = _load_overlay_points_for_pipe(pipe_number, y_bands, folder_path)
        print(f"DEBUG: pts = {pts}")
        if pts is not None:
            xs, ys, labels = pts
            for x_dist, y_band, label in zip(xs, ys, labels):
                if y_band not in y_bands:
                    continue
                x = dist_to_index(x_dist)
                if not (np.nanmin(x_vals) <= x <= np.nanmax(x_vals)):
                    continue
                y_idx = y_bands.index(y_band)
                y_pos = y_idx * offset_step
                figmlp.add_trace(go.Scatter(
                    x=[x], y=[y_pos], mode="markers",
                    marker=dict(size=8, color="red", line=dict(width=1, color="black")),
                    showlegend=False, name=f"{label} @ {x_dist:.2f}m",
                    hovertemplate=f"<b>{label}</b><br>Abs Dist: {x_dist:.2f} m<br>Band: {y_band}<extra></extra>"
                ))
                figmlp.add_annotation(
                    x=x, y=y_pos, text=str(label), showarrow=True, arrowhead=2,
                    arrowsize=1, arrowwidth=1, ax=0, ay=-20,
                    bgcolor="red", bordercolor="black",
                    font=dict(color="white", size=10, family="Arial Black")
                )
    except Exception as e:
        print(f"Overlay labels on lineplot failed: {e}")
        traceback.print_exc()

    figmlp.update_layout(
        template='plotly_white', height=500, width=1500,
        margin=dict(l=20, r=20, t=50, b=20), showlegend=False,
        title=dict(text=f"Hall-Sensor Lineplot — Joint Number {pipe_number}",
            x=0.5, xanchor="center", font=dict(size=18, family="Arial Black"))
    )

    step = 6
    total = len(res)
    tick_indices = list(range(0, total, step))
    figmlp.update_yaxes(
        tickmode='array',
        tickvals=[i * offset_step for i in tick_indices],
        ticktext=[res[i] for i in tick_indices],
        tickfont=dict(size=9), autorange="reversed"
    )

    write_plotly_html(figmlp, f'{folder_path}/lineplot{pipe_number}.html')


def save_lineplot_raw(folder_path, test_val, pipe_number):
    figmlpraw = go.Figure()
    for _, col in enumerate(test_val.columns):
        figmlpraw.add_trace(go.Scatter(
            x=test_val.index, y=test_val[col], mode='lines', name=col,
            line=dict(width=1), hoverinfo='x+y+name', showlegend=False
        ))
    figmlpraw.update_layout(
        xaxis_title='Counter', template='plotly_white',
        height=300, width=1500, margin=dict(l=20, r=20, t=50, b=20)
    )
    write_plotly_html(figmlpraw, f'{folder_path}/lineplot_raw{pipe_number}.html')


def save_pipe3d(data, data_cp, folder_path, pipe_number, pkl_path):
    df_pipe = pd.read_pickle(pkl_path)
    oddo_vals = (df_pipe['ODDO1'] / 1000).values

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.shape[0] > 1500:
        data = data[::2, :]
        oddo_vals = oddo_vals[::2]
    if data.shape[1] > 128:
        data = data[:, ::2]

    num_rows, num_cols = data.shape
    if len(oddo_vals) != num_rows:
        oddo_vals = np.interp(
            np.linspace(0, 1, num_rows),
            np.linspace(0, 1, len(oddo_vals)),
            oddo_vals
        )

    theta = np.linspace(0, 2 * np.pi, num_cols)
    theta_grid, _ = np.meshgrid(theta, np.zeros(num_rows))

    radius = 109.5
    odometer_start = float(np.nanmin(oddo_vals))
    odometer_end   = float(np.nanmax(oddo_vals))
    dist_range     = odometer_end - odometer_start

    x  = np.outer(oddo_vals, np.ones(num_cols))
    y  = radius * np.cos(theta_grid)
    zc = radius * np.sin(theta_grid)

    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=zc, surfacecolor=data,
        colorscale='jet', customdata=data_cp, showscale=False,
        hovertemplate='Dist: %{x:.2f} m<br>Value: %{surfacecolor:.2f}<extra></extra>'
    )])

    clock_labels = [
        dict(y=0,       z=radius,  text="12", name="12 o'clock"),
        dict(y=radius,  z=0,       text="3",  name="3 o'clock"),
        dict(y=0,       z=-radius, text="6",  name="6 o'clock"),
        dict(y=-radius, z=0,       text="9",  name="9 o'clock"),
    ]
    for cl in clock_labels:
        fig.add_trace(go.Scatter3d(
            x=[odometer_start, odometer_end],
            y=[cl['y'], cl['y']], z=[cl['z'], cl['z']],
            text=[cl['text'], cl['text']], mode='text',
            textposition="middle center", marker=dict(size=0),
            name=cl['name'], textfont=dict(size=16, color="#61090c"), showlegend=False
        ))

    x_ratio = max(4, min((dist_range / (2 * radius)) * 6, 14))
    cam_dist = 1.5 + x_ratio * 0.3
    camera = dict(
        eye=dict(x=0.1, y=-3.5, z=0.3),
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='Abs Distance (m)', yaxis_title='', zaxis_title='',
            aspectmode='manual', aspectratio=dict(x=x_ratio, y=1, z=1),
            camera=camera,
            xaxis=dict(showgrid=True, range=[odometer_start, odometer_end], autorange=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False),
        ),
        scene_dragmode='orbit', height=600, width=1500,
        title=dict(text=f'Pipe 3D Visualization — Joint {pipe_number}',
            x=0.5, xanchor='center', font=dict(size=18, family='Arial Black')),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    write_plotly_html(fig, f'{folder_path}/pipe3d{pipe_number}.html')


def _resolve_workers(workers):
    if workers in (None, 0, -1, "auto"):
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1)
    if isinstance(workers, int):
        cpu = os.cpu_count() or 1
        return max(1, min(workers, cpu))
    return 1


import time as _time_module

def _process_one_pkl(pkl_path, output_folder):
    start_time = _time_module.time()
    start_clock = datetime.now().strftime("%H:%M:%S")
    sensor_type = ["Hall", "Proximity"]

    try:
        pipe_number = int(Path(pkl_path).stem)
        print(f"\n🟢 PIPE {pipe_number} START: {start_clock}", flush=True)
        pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
        pipe_folder.mkdir(exist_ok=True)

        data = pd.read_pickle(pkl_path)
        hall, prox = count_pattern_minute_degree(pkl_path)

        total_sensors_count_hall = hall["count"]
        column_names_hall        = hall["columns"]
        minute_sensors_hall      = hall["minute"]
        degree_sensors_hall      = hall["degree"]

        total_sensors_count_prox = prox["count"]
        column_names_prox        = prox["columns"]
        minute_sensors_prox      = prox["minute"]
        degree_sensors_prox      = prox["degree"]

        for current_sensor_type in sensor_type:
            try:
                print(f"\n{'='*50}\n🚀 START SENSOR TYPE: {current_sensor_type}\nPIPE: {pipe_number}\n{'='*50}", flush=True)

                if current_sensor_type == "Hall":
                    print(f"📌 HALL: count={total_sensors_count_hall}, first_10={column_names_hall[:10]}", flush=True)
                    dfile = pre_process_data(
                        pkl_path, data, pipe_number, output_folder,
                        total_sensors_count_hall, column_names_hall,
                        minute_sensors_hall, degree_sensors_hall,
                        current_sensor_type, debug=True
                    )
                else:
                    print(f"📌 PROX: count={total_sensors_count_prox}, first_10={column_names_prox[:10]}", flush=True)
                    (dfile, df_new_tab9, datafile_original, test_val, map_ori_sens, df_new_tab10) = pre_process_data(
                        pkl_path, data, pipe_number, output_folder,
                        total_sensors_count_prox, column_names_prox,
                        minute_sensors_prox, degree_sensors_prox,
                        current_sensor_type, debug=True
                    )

                print(f"🏁 COMPLETED SENSOR TYPE: {current_sensor_type}\n", flush=True)

            except Exception as e:
                print(f"\n❌ CRASH\nPIPE: {pipe_number}\nSENSOR: {current_sensor_type}\n{traceback.format_exc()}", flush=True)
                raise

        xlsx_path = pipe_folder / f"Pipe_{pipe_number}.xlsx"
        dfile.to_excel(xlsx_path, index=False)

        total_time = round(_time_module.time() - start_time, 2)
        print(f"🔴 PIPE {pipe_number} END | TOTAL: {total_time}s ({round(total_time/60,2)} min)\n", flush=True)
        return f"Processed {os.path.basename(pkl_path)} and saved to {pipe_folder}"

    except Exception as e:
        print(f"Error loading {os.path.basename(pkl_path)}: {e}")
        traceback.print_exc()
        return f"Error loading {os.path.basename(pkl_path)}: {e}"


def count_pattern_minute_degree(datafile_path):
    df = pd.read_pickle(datafile_path)
    pattern_hall      = re.compile(r'^F\d+H\d+$', re.IGNORECASE)
    pattern_proximity = re.compile(r'^F\d+P\d+$', re.IGNORECASE)

    matching_columns_hall      = [col for col in df.columns if pattern_hall.match(col)]
    matching_columns_proximity = [col for col in df.columns if pattern_proximity.match(col)]

    count_hall = len(matching_columns_hall)
    minute_sensors_hall  = 720 / count_hall      if count_hall > 0 else None
    degree_sensors_hall  = minute_sensors_hall/2 if count_hall > 0 else None

    count_proximity = len(matching_columns_proximity)
    minute_sensors_proximity  = 720 / count_proximity      if count_proximity > 0 else None
    degree_sensors_proximity  = minute_sensors_proximity/2 if count_proximity > 0 else None

    hall = {"columns": matching_columns_hall, "count": count_hall,
            "minute": minute_sensors_hall, "degree": degree_sensors_hall}
    proximity = {"columns": matching_columns_proximity, "count": count_proximity,
                 "minute": minute_sensors_proximity, "degree": degree_sensors_proximity}

    print(f"hall sensor details: {hall} \n proximity sensor details: {proximity}")
    return hall, proximity


def create_html_and_csv_from_pkl(
    pkl_folder='pipes3',
    output_folder='Client_Pipes',
    output_callback=None,
    workers=WORKERS
):
    try:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        msg = f"❌ Failed creating output folder '{output_folder}': {e}"
        (output_callback or print)(msg)
        return

    try:
        pkl_paths = [str(Path(pkl_folder)/f) for f in os.listdir(pkl_folder) if f.lower().endswith('.pkl')]
    except Exception as e:
        msg = f"❌ Failed reading PKL folder '{pkl_folder}': {e}"
        (output_callback or print)(msg)
        return

    if not pkl_paths:
        msg = f"⚠ No .pkl files found in {pkl_folder}"
        (output_callback or print)(msg)
        return

    n_jobs = _resolve_workers(workers)

    def safe_process(pkl_path):
        import time, traceback, pandas as pd
        fname = os.path.basename(pkl_path)
        start = time.time()
        try:
            print(f"\n🔍 STARTING: {fname}")
            test_obj = pd.read_pickle(pkl_path)
            print(f"✅ PKL LOAD OK: {fname} | shape={getattr(test_obj,'shape','?')}")
            result = _process_one_pkl(pkl_path, output_folder)
            print(f"🏁 FINISHED: {fname} in {round(time.time()-start,2)}s")
            return result
        except Exception as e:
            return (f"\n❌ ERROR: {fname}\n⏱ {round(time.time()-start,2)}s\n{str(e)}\n{traceback.format_exc()}")

    try:
        results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(safe_process)(p) for p in pkl_paths
        )
    except Exception as e:
        msg = f"❌ Parallel execution crashed: {e}\n{traceback.format_exc()}"
        (output_callback or print)(msg)
        return

    for msg in results:
        (output_callback or print)(msg)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    import time
    st = time.time()
    create_html_and_csv_from_pkl(workers=WORKERS)
    print(f'Total time: {time.time()-st} seconds')