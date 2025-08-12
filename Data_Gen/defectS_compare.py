'''
This code will compare the defectS{x}.csv with PipeTally{x}.csv and generate defectS{x}.csv and heatmap_box{x}.html
'''

import os
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import re
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import datetime

from Data_Gen.defectSheet_util import boxing

INITIAL_READ = 0.0                    # At 400mm, F1H1 detects defect at 11:00 with roll 39.93

def defectSheet_process(datafile, pipe_number,output_folder):
    # Create DataFrame from raw data
    df_new_tab9 = pd.DataFrame(datafile, columns=[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)])
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
    
    # Setting bounds and applying conditions
    sens_mean = df_new_tab9.abs().mean()
    up_sen = 1.8
    lo_sen = -3.5

    up_bo = sens_mean*up_sen
    lo_bo = sens_mean*lo_sen

    for col in df_new_tab9.columns:
        if col in up_bo.index and col in lo_bo.index:
            df_new_tab9[col] = np.where((df_new_tab9[col] >= lo_bo[col]) & (df_new_tab9[col] <= up_bo[col]), 0, df_new_tab9[col])

    initial_read = INITIAL_READ                    
    roll = datafile['ROLL']
    roll = roll - initial_read
    odoData = datafile['ODDO1']

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
        
        if start_time_dt <= time_to_check <= end_time_dt:
            return True
        else:
            return False

    d= []

    for i,pos in enumerate(roll):
        d.append({f"Roll_Sensor_0": pos})

    upd_d = list(add_sensor_keys(d))

    oriData = pd.DataFrame.from_dict(data=upd_d)

    clockData = oriData.applymap(degrees_to_hours_minutes2)

    test_clockData = clockData.copy()
    test_clockData = test_clockData.apply(pd.to_datetime, format='%H:%M:%S')
    # test_clockData = test_clockData - pd.Timedelta(minutes=79)
    test_clockData = test_clockData.applymap(lambda x: x.strftime('%H:%M:%S'))
    test_clockData = test_clockData.applymap(lambda x: x.replace('23:', '11:') if isinstance(x, str) and x.startswith('23:') else x)
    test_clockData = test_clockData.applymap(lambda x: x.replace('22:', '10:') if isinstance(x, str) and x.startswith('22:') else x)
    test_clockData = test_clockData.applymap(lambda x: x.replace('12:', '00:') if isinstance(x, str) and x.startswith('12:') else x)

    number_sensors = 96
    sensor_coluns = [f'F{f}H{h}' for f in range(1, 25) for h in range(1, 5)]

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

        time_dict = {key: [] for key in time_ranges_2}
        
        return time_dict

    time_dict_1 = create_time_dict()
    time_dict_2 = create_time_dict()

    rang = list(time_dict_1.keys())

    for index,row in test_clockData.iterrows():
        xl = list(row)
        xd = dict(row)
        xkeys = list(xd.keys())
        c = 0
        for s,d in xd.items():
            if check_time_range(d):
                ind = xl.index(d)
                y = xl[ind:] + xl[:ind]
                break
        
        curr = ind
        # ck = xkeys[curr]
        while True:
            ck = xkeys[curr]
            time_dict_1[rang[c]].append((curr,ck,xd[ck]))
            c += 1
            curr = (curr+1)% len(xkeys)
            if curr == ind:
                break

    map_ori_sens = pd.DataFrame(time_dict_1)

    val_ori_sens = map_ori_sens.copy()

    def extract_string(cell):
        return cell[1]

    val_ori_sens = val_ori_sens.applymap(extract_string)

    test_val = val_ori_sens.copy()

    for r,e in val_ori_sens.iterrows():
        c = 0
        for col_name, tup_value in e.items():
            # print(r,col_name,tup_value)
            cell_v = df_new_tab9.at[r,tup_value]
            # print(cell_v)
            test_val.iloc[r,c] = cell_v
            c += 1

    
    final_df = compare_and_merge(datafile, test_val, map_ori_sens, pipe_number,output_folder)

    save_heatmap_box(final_df,test_val,datafile,map_ori_sens,pipe_number,output_folder)

    return final_df

def compare_and_merge(datafile, test_val, map_ori_sens, pipe_number,output_folder):
    # Load defect sheet and pipe tally data
    defect_Sheet = boxing(datafile, test_val, map_ori_sens)
    def_s_ori = defect_Sheet.copy()
    pipe_Tally = pd.read_csv(f'{output_folder}/Pipe_{pipe_number}/PipeTally{pipe_number}.csv')

    defect_Sheet.loc[defect_Sheet['Width'] == 0.0, 'Width'] = 2.0
    pipe_Tally['Abs. Distance (mm)'] = pipe_Tally['Abs. Distance (m)'] * 1000
    
    defect_Sheet = defect_Sheet.sort_values(by='Absolute Distance').reset_index(drop=True)

    # Prepare an empty list for the merged rows
    merged_rows = []

    # Iterate over each row in pipe_tally
    for _, pipe_row in pipe_Tally.iterrows():
        pipe_distance = pipe_row['Abs. Distance (mm)']

        # Find the row in defect_sheet with the smallest absolute difference
        if not defect_Sheet.empty:
            defect_Sheet['Difference'] = abs(defect_Sheet['Absolute Distance'] - pipe_distance)
            closest_defect_row = defect_Sheet.loc[defect_Sheet['Difference'].idxmin()]

            # Merge the rows
            combined_row = {**closest_defect_row.to_dict(), **pipe_row.to_dict()}

            # Append the combined row to merged_rows
            merged_rows.append(combined_row)

            # Remove the closest defect row from defect_sheet to avoid reuse
            defect_Sheet = defect_Sheet.drop(closest_defect_row.name)

    merged_df = pd.DataFrame(merged_rows)

    # Select and rename columns as necessary
    # final_columns = [
    #     'Box Number', 'Type', 'Width', 'Height', 'x0', 'y0', 'x1', 'y1',
    #     'StartS', 'EndS', 'Absolute Distance', 'Peak Value', 'Avg_Val',
    #     'Ori Val', 'Breadth', 'Distance to U/S GW(m)', 'Feature Type',
    #     'Dimensions  Classification', 'Orientation o\' clock', 'Length(mm)',
    #     'Width(mm)', 'WT (mm)', 'Depth % '
    # ]

    # final_df = merged_df[final_columns]

    return def_s_ori

def save_heatmap_box(df_boxes,test_val,datafile,map_ori_sens,pipe_number,output_folder):
    fighmbox = go.Figure(data=go.Heatmap(
        z=test_val.T,
        y=test_val.columns,
        x=datafile['ODDO1'],
        colorscale='jet',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns], 
        # zmin=-12000,
        # zmax=30000
    ))

    # Add rectangles and annotations for merged bounding boxes
    for _, row in df_boxes.iterrows():
        x0, x1, y0, y1 = row['x0'], row['x1'], row['y0'], row['y1']
        box_number = row['Box Number']
        color = 'red'
        # color = 'red' if row['Feature Type'] == 'Metal_Loss' else 'green'
        # fillcolor = 'rgba(255, 0, 0, 0.1)' if row['Feature Type'] == 'Metal_Loss' else 'rgba(0, 255, 0, 0.1)'
        
        fighmbox.add_shape(
            type='rect',
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color=color, width=2),
            # fillcolor=fillcolor
        )
        # tt = 'L' if row["Feature Type"] == 'Metal_Loss' else 'G'
        # Add text annotation to the top left of the box with a margin
        fighmbox.add_annotation(
            x=x0,  
            y=y1,  
            # text=f'{tt}: {box_number}',
            text='',
            showarrow=False,
            font=dict(size=12, color=color),
            align='center',
            xref='x',
            yref='y'
        )

    fighmbox.update_layout(
        xaxis_title='Odometer',
        height=500,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    fighmbox.write_html(f'{output_folder}/Pipe_{pipe_number}/heatmap_box{pipe_number}.html', auto_open=False)

def create_defectSheet(pkl_folder='pipes3', output_folder='Client_Pipe',output_callback=None):
    Path(output_folder).mkdir(exist_ok=True)

    for pkl_file in os.listdir(pkl_folder):
        if pkl_file.endswith('.pkl'):
            pipe_number = pkl_file[:-4]  # Remove '.pkl' to get the number
            pipe_folder = Path(output_folder) / f'Pipe_{pipe_number}'
            pipe_folder.mkdir(exist_ok=True)

            try:
                with open(os.path.join(pkl_folder, pkl_file), 'rb') as file:
                    data = pickle.load(file)
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue
            
            ds = defectSheet_process(data,pipe_number,output_folder)

            # Save the Defect Sheet file
            csv_file_path = pipe_folder / f'defectS{pipe_number}.csv'
            ds.to_csv(csv_file_path, index=False)

            message = f"Processed {pkl_file} and saved to {pipe_folder}"
            if output_callback:
                output_callback(message)
            else:
                print(message)
            # break

if __name__ == "__main__":
    create_defectSheet()