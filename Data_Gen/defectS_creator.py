# '''
# To create defectS.csv based on heatmap from Pipe_<Pipe number>/ heatmap<Pipe number>.html
# '''
# '''
# This code will compare the defectS{x}.csv with PipeTally{x}.csv and generate defectS{x}.csv and heatmap_box{x}.html
# '''

# import os
# import pandas as pd
# import pickle
# from pathlib import Path
# import numpy as np
# import re
# import plotly.graph_objects as go
# from scipy.signal import savgol_filter
# from datetime import datetime
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# from Data_Gen.defectSheet_util import boxing2

# INITIAL_READ = 0.0                    # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
# UPPER_SENS_MUL = 1
# LOWER_SENS_MUL = 3

# def defectSheet_process(datafile, pipe_number,output_folder):
#     # Create DataFrame from raw data
#     df_new_tab9 = pd.DataFrame(datafile, columns=[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)])
#     sensor_columns = df_new_tab9.columns.tolist()

#     # Denoising using Savitzky-Golay filter
#     window_length = 15
#     polyorder = 2
#     # for col in sensor_columns:
#     #     data = df_new_tab9[col].values
#     #     time_index = np.arange(len(df_new_tab9))
#     #     trend = np.polyval(np.polyfit(time_index, data, 2), time_index)
#     #     data_denoised = savgol_filter(data - trend, window_length, polyorder)
#     #     df_new_tab9[col] = data_denoised

#     for col in sensor_columns:
#         try:
#             # Convert to numeric, coerce invalid values to NaN
#             data = pd.to_numeric(df_new_tab9[col], errors='coerce').astype(float)
#             if data.isnull().all():
#                 print(f"Column {col} is all NaN. Skipping.")
#                 continue
            
#             data = data.fillna(0)  # Optional: Replace NaNs with 0
#             time_index = np.arange(len(data))
            
#             trend = np.polyval(np.polyfit(time_index, data, 2), time_index)
#             data_denoised = savgol_filter(data - trend, window_length, polyorder)
#             df_new_tab9[col] = data_denoised

#         except Exception as err:
#             print(f"Error processing column {col}: {err}")
#             continue

    
#     # Setting bounds and applying conditions
#     sens_mean = df_new_tab9.abs().mean()
#     standard_deviation = df_new_tab9.std(axis=0, skipna=True)

#     mean_plus_sigma = sens_mean + UPPER_SENS_MUL * standard_deviation
#     mean_negative_sigma = sens_mean - LOWER_SENS_MUL * standard_deviation

#     # Apply the noise filtering
#     for col in df_new_tab9.columns:
#         if col in mean_plus_sigma.index and col in mean_negative_sigma.index:
#             df_new_tab9[col] = np.where((df_new_tab9[col] >= mean_negative_sigma[col]) & 
#                                         (df_new_tab9[col] <= mean_plus_sigma[col]), 
#                                         0, df_new_tab9[col])
#     initial_read = INITIAL_READ                    
#     roll = datafile['ROLL']
#     roll = roll - initial_read
#     odoData = datafile['ODDO1']

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
        
#         if start_time_dt <= time_to_check <= end_time_dt:
#             return True
#         else:
#             return False

#     d= []

#     for i,pos in enumerate(roll):
#         d.append({f"Roll_Sensor_0": pos})

#     upd_d = list(add_sensor_keys(d))

#     oriData = pd.DataFrame.from_dict(data=upd_d)

#     clockData = oriData.applymap(degrees_to_hours_minutes2)

#     test_clockData = clockData.copy()
#     test_clockData = test_clockData.apply(pd.to_datetime, format='%H:%M:%S')
#     # test_clockData = test_clockData - pd.Timedelta(minutes=79)
#     test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M:%S'))
#     test_clockData = test_clockData.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
#     test_clockData = test_clockData.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
#     test_clockData = test_clockData.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

#     number_sensors = 96
#     sensor_coluns = [f'F{f}H{h}' for f in range(1, 25) for h in range(1, 5)]

#     test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_new_tab9.columns)))
#     def create_time_dict():
#         time_ranges_2 = [
#         '00:00:00', '00:07:30', '00:15:00', '00:22:30', '00:30:00', '00:37:30', '00:45:00', '00:52:30', '01:00:00', '01:07:30', 
#         '01:15:00', '01:22:30', '01:30:00', '01:37:30', '01:45:00', '01:52:30', '02:00:00', '02:07:30', '02:15:00', '02:22:30', 
#         '02:30:00', '02:37:30', '02:45:00', '02:52:30', '03:00:00', '03:07:30', '03:15:00', '03:22:30', '03:30:00', '03:37:30', 
#         '03:45:00', '03:52:30', '04:00:00', '04:07:30', '04:15:00', '04:22:30', '04:30:00', '04:37:30', '04:45:00', '04:52:30', 
#         '05:00:00', '05:07:30', '05:15:00', '05:22:30', '05:30:00', '05:37:30', '05:45:00', '05:52:30', '06:00:00', '06:07:30', 
#         '06:15:00', '06:22:30', '06:30:00', '06:37:30', '06:45:00', '06:52:30', '07:00:00', '07:07:30', '07:15:00', '07:22:30', 
#         '07:30:00', '07:37:30', '07:45:00', '07:52:30', '08:00:00', '08:07:30', '08:15:00', '08:22:30', '08:30:00', '08:37:30', 
#         '08:45:00', '08:52:30', '09:00:00', '09:07:30', '09:15:00', '09:22:30', '09:30:00', '09:37:30', '09:45:00', '09:52:30', 
#         '10:00:00', '10:07:30', '10:15:00', '10:22:30', '10:30:00', '10:37:30', '10:45:00', '10:52:30', '11:00:00', '11:07:30', 
#         '11:15:00', '11:22:30', '11:30:00', '11:37:30', '11:45:00', '11:52:30'
#         ]

#         time_dict = {key: [] for key in time_ranges_2}
        
#         return time_dict

#     time_dict_1 = create_time_dict()
#     time_dict_2 = create_time_dict()

#     rang = list(time_dict_1.keys())

#     for index,row in test_clockData.iterrows():
#         xl = list(row)
#         xd = dict(row)
#         xkeys = list(xd.keys())
#         c = 0
#         for s,d in xd.items():
#             if check_time_range(d):
#                 ind = xl.index(d)
#                 y = xl[ind:] + xl[:ind]
#                 break
        
#         curr = ind
#         # ck = xkeys[curr]
#         while True:
#             ck = xkeys[curr]
#             time_dict_1[rang[c]].append((curr,ck,xd[ck]))
#             c += 1
#             curr = (curr+1)% len(xkeys)
#             if curr == ind:
#                 break

#     map_ori_sens = pd.DataFrame(time_dict_1)

#     val_ori_sens = map_ori_sens.copy()

#     def extract_string(cell):
#         return cell[1]

#     val_ori_sens = val_ori_sens.applymap(extract_string)

#     test_val = val_ori_sens.copy()

#     for r,e in val_ori_sens.iterrows():
#         c = 0
#         for col_name, tup_value in e.items():
#             # print(r,col_name,tup_value)
#             cell_v = df_new_tab9.at[r,tup_value]
#             # print(cell_v)
#             test_val.iloc[r,c] = cell_v
#             c += 1

    
#     final_df = compare_and_merge(datafile, test_val, map_ori_sens, pipe_number,sens_mean,output_folder)

#     return final_df

# def compare_and_merge(datafile, test_val, map_ori_sens, pipe_number,sens_mean,output_folder):
#     defect_Sheet,fig_hm_box = boxing2(datafile, test_val, map_ori_sens,sens_mean)
#     if isinstance(defect_Sheet, list):
#         defect_Sheet = pd.DataFrame(defect_Sheet)

#     pipe_Tally = pd.read_csv(f'{output_folder}/Pipe_{pipe_number}/PipeTally{pipe_number}.csv')
#     pipe_Tally.columns = pipe_Tally.columns.str.strip()
#     pipe_Tally['Absolute Distance'] = pipe_Tally['Abs. Distance (m)']
#     pipe_Tally_2 = pipe_Tally.copy()
#     pipe_Tally = pipe_Tally[['WT (mm)', 'Dimensions Classification', 'Type', 'Absolute Distance']]

#     pipe_Tally_2 = pipe_Tally_2[['WT (mm)', 'Dimensions Classification', 'Type', 'Absolute Distance','Length (mm)','Width (mm)','Distance to U/S GW(m)','Depth %','Orientation o\' clock']]
#     # pipe_Tally_2['Box Number'] = pipe_Tally_2.index
#     pipe_Tally_2['Box Number'] = None 
#     def get_closest_box_number(row):
#         distance_diff = abs(defect_Sheet['Absolute Distance'] - row['Absolute Distance'])
#         closest_index = distance_diff.idxmin()
#         return defect_Sheet.loc[closest_index, 'Box Number']
    
#     pipe_Tally_2['Box Number'] = pipe_Tally_2.apply(get_closest_box_number, axis=1)

#     save_heatmap_box(fig_hm_box,pipe_number,output_folder,pipe_Tally_2,defect_Sheet)
#     # pipe_Tally['Absolute Distance'] = pipe_Tally['Absolute Distance']/1000
#     return pipe_Tally_2

# def save_heatmap_box(fighmbox,pipe_number,output_folder,pipe_Tally_2,defect_Sheet):
#     selected_defects = defect_Sheet[defect_Sheet['Box Number'].isin(pipe_Tally_2['Box Number'])]
#     # print(selected_defects)
#     for i, row in selected_defects.iterrows():
#         box_number = row.get('Box Number')  # Retrieve the Box Number
        
#         if pd.notna(box_number):  
#             start_reading = row['Distance to U/S GW(m)'] # Define x0 (start of box)
#             end_reading = row['x1']  # Define x1 (end of box)
#             start_sensor = row['y0']  # Define y0 (start of box)
#             end_sensor = row['y1']  # Define y1 (end of box)

#             # Add rectangle shape for the box
#             fighmbox.add_shape(
#                 type='rect',
#                 x0=start_reading,
#                 x1=end_reading,
#                 y0=start_sensor,
#                 y1=end_sensor,
#                 line=dict(color='black', width=1),
#                 fillcolor='rgba(255, 0, 0, 0.2)'
#             )

#             fighmbox.add_annotation(
#                 x=start_reading, 
#                 y=end_sensor+1.5,  
#                 text=f'{box_number}',  
#                 showarrow=False,
#                 font=dict(size=12, color='black'),
#                 align='center',
#                 xref='x',
#                 yref='y'
#             )
#     fighmbox.write_html(f'{output_folder}/Pipe_{pipe_number}/heatmap_box{pipe_number}.html', auto_open=False)

# def create_defectSheet_and_heatmap_box(pkl_folder='pipes3.1', output_folder='Client_Pipes',output_callback=None):
#     Path(output_folder).mkdir(exist_ok=True)

#     for pkl_file in os.listdir(pkl_folder):
#         if pkl_file.endswith('.pkl'):
#             pipe_number = pkl_file[:-4]  # Remove '.pkl' to get the number
#             pipe_folder = Path(output_folder) / f'Pipe_{pipe_number}'
#             pipe_folder.mkdir(exist_ok=True)

#             try:
#                 with open(os.path.join(pkl_folder, pkl_file), 'rb') as file:
#                     data = pd.read_pickle(file)
#             except Exception as e:
#                 print(f"Error loading {pkl_file}: {e}")
#                 continue
            
#             ds = defectSheet_process(data,pipe_number,output_folder)

#             # Save the Defect Sheet file
#             csv_file_path = pipe_folder / f'defectS{pipe_number}.csv'
#             ds.to_csv(csv_file_path, index=False)
            
#             message = f"Processed Defect Sheet from {pkl_file} and saved to {pipe_folder}"
#             if output_callback:
#                 output_callback(message)
#             else:
#                 print(message)
#             # break

# if __name__ == "__main__":
#     create_defectSheet_and_heatmap_box()


"""
To create defectS.csv based on heatmap from Pipe_<Pipe number>/heatmap<Pipe number>.html

This code will compare the defectS{x}.csv with PipeTally{x}.csv and generate
defectS{x}.csv and heatmap_box{x}.html

Now with multicore parallel processing controlled by WORKER.
"""

import os
# Avoid BLAS over-subscription (many threads per process can slow things down)
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
warnings.filterwarnings("ignore", category=FutureWarning)

from Data_Gen.defectSheet_util import boxing2

# -------------------- CONFIG --------------------
INITIAL_READ = 0.0       # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3
# -1 / 0 / None / "auto" => auto (CPU-1, at least 1). Or set an int, e.g. 4
WORKER = 4
# ------------------------------------------------


def defectSheet_process(datafile: pd.DataFrame, pipe_number, output_folder):
    # Create DataFrame from raw data (ensures exact sensor column order/coverage)
    df_new_tab9 = pd.DataFrame(
        datafile,
        columns=[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)]
    )
    sensor_columns = df_new_tab9.columns.tolist()

    # --- Denoising using Savitzky-Golay filter (robust to non-numeric cols) ---
    window_length = 15
    polyorder = 2
    for col in sensor_columns:
        try:
            data = pd.to_numeric(df_new_tab9[col], errors='coerce').astype(float)
            if data.isnull().all():
                # If all NaN leave as-is
                continue
            data = data.fillna(0.0)
            time_index = np.arange(len(data), dtype=float)
            # 2nd-order trend removal before SavGol
            trend = np.polyval(np.polyfit(time_index, data, 2), time_index)
            df_new_tab9[col] = savgol_filter((data - trend), window_length, polyorder)
        except Exception as err:
            print(f"[{pipe_number}] Denoise error in {col}: {err}")
            continue

    # --- Sigma bounds and noise filtering to zero-out in-bound values ---
    sens_mean = df_new_tab9.abs().mean()
    standard_deviation = df_new_tab9.std(axis=0, skipna=True)
    mean_plus_sigma = sens_mean + UPPER_SENS_MUL * standard_deviation
    mean_negative_sigma = sens_mean - LOWER_SENS_MUL * standard_deviation

    for col in df_new_tab9.columns:
        if col in mean_plus_sigma.index and col in mean_negative_sigma.index:
            arr = df_new_tab9[col]
            mask = (arr >= mean_negative_sigma[col]) & (arr <= mean_plus_sigma[col])
            # set in-bound values to 0
            df_new_tab9[col] = np.where(mask, 0.0, arr)

    # --- Orientation clock mapping per ROLL ---
    initial_read = INITIAL_READ
    roll = datafile['ROLL'] - initial_read
    # odoData = datafile['ODDO1']  # retained if needed elsewhere

    def degrees_to_hours_minutes2(degrees):
        if degrees < 0:
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

    # Build roll sensor degrees → HH:MM:SS map
    d = [{f"Roll_Sensor_0": pos} for pos in roll]
    upd_d = list(add_sensor_keys(d))
    oriData = pd.DataFrame.from_dict(data=upd_d)
    clockData = oriData.applymap(degrees_to_hours_minutes2)

    # Normalize times
    test_clockData = clockData.copy()
    test_clockData = test_clockData.apply(pd.to_datetime, format='%H:%M:%S')
    test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M:%S'))
    test_clockData = test_clockData.applymap(
        lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x
    )
    test_clockData = test_clockData.applymap(
        lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x
    )
    test_clockData = test_clockData.applymap(
        lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x
    )

    # Align column names with processed sensors
    test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_new_tab9.columns)))

    def create_time_dict():
        time_ranges_2 = [
            '00:00:00', '00:07:30', '00:15:00', '00:22:30', '00:30:00', '00:37:30',
            '00:45:00', '00:52:30', '01:00:00', '01:07:30', '01:15:00', '01:22:30',
            '01:30:00', '01:37:30', '01:45:00', '01:52:30', '02:00:00', '02:07:30',
            '02:15:00', '02:22:30', '02:30:00', '02:37:30', '02:45:00', '02:52:30',
            '03:00:00', '03:07:30', '03:15:00', '03:22:30', '03:30:00', '03:37:30',
            '03:45:00', '03:52:30', '04:00:00', '04:07:30', '04:15:00', '04:22:30',
            '04:30:00', '04:37:30', '04:45:00', '04:52:30', '05:00:00', '05:07:30',
            '05:15:00', '05:22:30', '05:30:00', '05:37:30', '05:45:00', '05:52:30',
            '06:00:00', '06:07:30', '06:15:00', '06:22:30', '06:30:00', '06:37:30',
            '06:45:00', '06:52:30', '07:00:00', '07:07:30', '07:15:00', '07:22:30',
            '07:30:00', '07:37:30', '07:45:00', '07:52:30', '08:00:00', '08:07:30',
            '08:15:00', '08:22:30', '08:30:00', '08:37:30', '08:45:00', '08:52:30',
            '09:00:00', '09:07:30', '09:15:00', '09:22:30', '09:30:00', '09:37:30',
            '09:45:00', '09:52:30', '10:00:00', '10:07:30', '10:15:00', '10:22:30',
            '10:30:00', '10:37:30', '10:45:00', '10:52:30', '11:00:00', '11:07:30',
            '11:15:00', '11:22:30', '11:30:00', '11:37:30', '11:45:00', '11:52:30'
        ]
        return {key: [] for key in time_ranges_2}

    time_dict_1 = create_time_dict()
    rang = list(time_dict_1.keys())

    # Build orientation-to-sensor mapping per row
    for _, row in test_clockData.iterrows():
        xl = list(row)
        xd = dict(row)
        xkeys = list(xd.keys())
        c = 0
        ind = 0
        for _, dval in xd.items():
            if check_time_range(dval):
                ind = xl.index(dval)
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

    # Resolve sensor names from mapping
    val_ori_sens = map_ori_sens.copy().applymap(lambda cell: cell[1])

    # Build the heatmap-aligned values by dereferencing into df_new_tab9
    test_val = val_ori_sens.copy()
    for r, e in val_ori_sens.iterrows():
        c = 0
        for _, sensor_name in e.items():
            test_val.iloc[r, c] = df_new_tab9.at[r, sensor_name]
            c += 1

    # Compare/merge to produce defect sheet and figure with boxes
    final_df = compare_and_merge(datafile, test_val, map_ori_sens, pipe_number, sens_mean, output_folder)
    return final_df


def compare_and_merge(datafile, test_val, map_ori_sens, pipe_number, sens_mean, output_folder):
    defect_Sheet, fig_hm_box = boxing2(datafile, test_val, map_ori_sens, sens_mean)
    if isinstance(defect_Sheet, list):
        defect_Sheet = pd.DataFrame(defect_Sheet)

    # Load PipeTally; handle missing gracefully
    tally_path = Path(output_folder) / f'Pipe_{pipe_number}' / f'PipeTally{pipe_number}.csv'
    if not tally_path.exists():
        print(f"[{pipe_number}] Missing {tally_path.name}. Saving only defect sheet overlay figure.")
        save_heatmap_box(fig_hm_box, pipe_number, output_folder, pd.DataFrame(), defect_Sheet)
        return defect_Sheet  # return at least the computed defects

    pipe_Tally = pd.read_csv(tally_path)
    pipe_Tally.columns = pipe_Tally.columns.str.strip()
    # Normalize expected column name
    if 'Absolute Distance' not in pipe_Tally.columns and 'Abs. Distance (m)' in pipe_Tally.columns:
        pipe_Tally['Absolute Distance'] = pipe_Tally['Abs. Distance (m)']

    # Keep enriched version for output/boxing annotation mapping
    cols_keep = [
        'WT (mm)', 'Dimensions Classification', 'Type', 'Absolute Distance',
        'Length (mm)', 'Width (mm)', 'Distance to U/S GW(m)', 'Depth %', "Orientation o' clock"
    ]
    pipe_Tally_2 = pipe_Tally[[c for c in cols_keep if c in pipe_Tally.columns]].copy()
    if 'Absolute Distance' not in pipe_Tally_2.columns:
        raise ValueError(f"[{pipe_number}] 'Absolute Distance' column not found in PipeTally.")

    # Assign nearest box number from defect sheet to each tally row
    pipe_Tally_2['Box Number'] = None
    if not defect_Sheet.empty and 'Absolute Distance' in defect_Sheet.columns and 'Box Number' in defect_Sheet.columns:
        def get_closest_box_number(row):
            distance_diff = abs(defect_Sheet['Absolute Distance'] - row['Absolute Distance'])
            closest_index = distance_diff.idxmin()
            return defect_Sheet.loc[closest_index, 'Box Number']
        pipe_Tally_2['Box Number'] = pipe_Tally_2.apply(get_closest_box_number, axis=1)

    save_heatmap_box(fig_hm_box, pipe_number, output_folder, pipe_Tally_2, defect_Sheet)
    return pipe_Tally_2


def save_heatmap_box(fighmbox: go.Figure, pipe_number, output_folder, pipe_Tally_2: pd.DataFrame, defect_Sheet: pd.DataFrame):
    # Draw boxes for defects that match PipeTally by Box Number (if tally provided)
    if pipe_Tally_2 is not None and not pipe_Tally_2.empty and 'Box Number' in pipe_Tally_2.columns:
        selected_defects = defect_Sheet[defect_Sheet['Box Number'].isin(pipe_Tally_2['Box Number'])]
    else:
        selected_defects = defect_Sheet

    for _, row in selected_defects.iterrows():
        box_number = row.get('Box Number')
        if pd.notna(box_number):
            start_reading = row['Distance to U/S GW(m)']  # x0
            end_reading = row['x1']                       # x1
            start_sensor = row['y0']                      # y0
            end_sensor = row['y1']                        # y1

            fighmbox.add_shape(
                type='rect',
                x0=start_reading, x1=end_reading,
                y0=start_sensor, y1=end_sensor,
                line=dict(color='black', width=1),
                fillcolor='rgba(255, 0, 0, 0.2)'
            )
            fighmbox.add_annotation(
                x=start_reading,
                y=end_sensor + 1.5,
                text=f'{box_number}',
                showarrow=False,
                font=dict(size=12, color='black'),
                align='center',
                xref='x', yref='y'
            )

    out_html = Path(output_folder) / f'Pipe_{pipe_number}' / f'heatmap_box{pipe_number}.html'
    fighmbox.write_html(str(out_html), auto_open=False)


# -------------------- PARALLEL HELPERS --------------------
def _resolve_workers(workers):
    if workers in (None, 0, -1, "auto"):
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1)
    if isinstance(workers, int):
        cpu = os.cpu_count() or 1
        return max(1, min(workers, cpu))
    return 1


def _process_one_pkl(pkl_path: str, output_folder: str):
    try:
        pipe_number = Path(pkl_path).stem
        pipe_folder = Path(output_folder) / f"Pipe_{pipe_number}"
        pipe_folder.mkdir(exist_ok=True)

        data = pd.read_pickle(pkl_path)
        ds = defectSheet_process(data, pipe_number, output_folder)

        # Save the Defect Sheet file (can be defect_Sheet or pipe_Tally_2 depending on availability)
        csv_file_path = pipe_folder / f'defectS{pipe_number}.csv'
        ds.to_csv(csv_file_path, index=False)

        return f"Processed Defect Sheet from {os.path.basename(pkl_path)} and saved to {pipe_folder}"
    except Exception as e:
        return f"Error processing {os.path.basename(pkl_path)}: {e}"


def create_defectSheet_and_heatmap_box(
    pkl_folder='pipes3.1',
    output_folder='Client_Pipes',
    output_callback=None,
    workers=WORKER
):
    Path(output_folder).mkdir(exist_ok=True)

    # Collect .pkl file paths
    pkl_paths = [
        str(Path(pkl_folder) / f)
        for f in os.listdir(pkl_folder)
        if f.lower().endswith('.pkl')
    ]

    if not pkl_paths:
        msg = f"No .pkl files found in {pkl_folder}"
        if output_callback:
            output_callback(msg)
        else:
            print(msg)
        return

    n_jobs = _resolve_workers(workers)

    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_process_one_pkl)(p, output_folder) for p in pkl_paths
    )

    for msg in results:
        if output_callback:
            output_callback(msg)
        else:
            print(msg)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Example:
    # create_defectSheet_and_heatmap_box(workers=4)
    create_defectSheet_and_heatmap_box(workers=WORKER)

