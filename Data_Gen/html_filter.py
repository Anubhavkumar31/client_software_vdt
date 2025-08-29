'''
This code will input the raw pkl file( {x}.pkl ) and generate out the plots to:
Pipe_{x}  with ending format as <Name_{x}.html>
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
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

INITIAL_READ = 0.0                   # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3

def pre_process_data(datafile, pipe_number,output_folder):
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

    df_raw_straight = df_new_tab9.copy()
    
    # Setting bounds and applying conditions
    sens_mean = df_new_tab9.abs().mean()
    # up_sen = 1.8
    # lo_sen = -3.5

    # up_bo = sens_mean*up_sen
    # lo_bo = sens_mean*lo_sen

    # for col in df_new_tab9.columns:
    #     if col in up_bo.index and col in lo_bo.index:
    #         df_new_tab9[col] = np.where((df_new_tab9[col] >= lo_bo[col]) & (df_new_tab9[col] <= up_bo[col]), 0, df_new_tab9[col])

    standard_deviation = df_new_tab9.std(axis=0, skipna=True)

    mean_plus_sigma = sens_mean + UPPER_SENS_MUL * standard_deviation
    mean_negative_sigma = sens_mean - LOWER_SENS_MUL * standard_deviation

    # Apply the noise filtering
    for col in df_new_tab9.columns:
        if col in mean_plus_sigma.index and col in mean_negative_sigma.index:
            df_new_tab9[col] = np.where((df_new_tab9[col] >= mean_negative_sigma[col]) & 
                                        (df_new_tab9[col] <= mean_plus_sigma[col]), 
                                        0, df_new_tab9[col])

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

    map_val_sens = pd.DataFrame(index=test_val.index, columns=test_val.columns)
    for column in test_val.columns:
        for i in range(test_val.shape[0]):
            map_value = map_ori_sens.loc[i, column]
            test_value = test_val.loc[i, column]
            map_val_sens.loc[i, column] = (*map_value, test_value)
    
    create_plots(df_new_tab9,df_raw_straight, datafile, test_val,map_ori_sens, pipe_number,output_folder)

    return datafile

def create_plots(df_new_tab9,df_raw_straight, datafile, test_val,map_ori_sens, pipe_number,output_folder):
    folder_path = f'{output_folder}/Pipe_{pipe_number}'
    os.makedirs(folder_path, exist_ok=True)

    # Heatmap
    save_heatmap(test_val, datafile,map_ori_sens, folder_path,pipe_number)

    # Heatmap Raw
    #save_heatmap_raw(folder_path,df_raw_straight,datafile,pipe_number)

    # MultilinePlot 
    save_lineplot(folder_path,test_val,datafile,pipe_number)

    # MultilinePlot Raw
    #save_lineplot_raw(folder_path,test_val,pipe_number)

    # 3D Pipe 
    save_pipe3d(test_val,test_val,folder_path,pipe_number)

def save_heatmap(test_val, datafile,map_ori_sens, folder_path,pipe_number):
    fighm = go.Figure(data=go.Heatmap(
        z=test_val.T,
        y=test_val.columns,
        # x=[test_val.index,datafile['ODDO1'].round(2)],
        x=(datafile['ODDO1']/1000).round(2),
        colorscale='jet',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns], 
        # zmin = -12000,
        # zmax = 30000
    ))
    fighm.update_layout(
        xaxis_title='Absolute Distance (m)',
        height=500,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fighm.write_html(f'{folder_path}/heatmap{pipe_number}.html', auto_open=False)


def save_heatmap_raw(folder_path,df_raw_straight,dataf,pipe_number):
    initial_read = INITIAL_READ
    roll = dataf['ROLL']
    roll = roll - initial_read
    odoData = dataf['ROLL']

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

    pat1 = re.compile(r'^F\d+H\d+$')   # F%H%
    sel_col = ['ROLL','ODDO1']
    def matches_pat(column):
        return pat1.match(column)
    fil_col = [col for col in dataf.columns if matches_pat(col) or col in sel_col]
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
            cell_v = sensD.at[r,tup_value]
            # print(cell_v)
            test_val.iloc[r,c] = cell_v
            c += 1
    
    figraw = go.Figure(data=go.Heatmap(
        z=test_val.T,
        y=test_val.columns,
        x=(dataf['ODDO1']/1000).round(2),
        colorscale='jet', 
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],  
        zmin = -12000,
        zmax = 30000
    ))

    figraw.update_layout(
        height=300,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    figraw.write_html(f'{folder_path}/heatmap_raw{pipe_number}.html', auto_open=False)

# def save_lineplot(folder_path,test_val,pipe_number):
#     figmlp = go.Figure()
#     offset_step = 1000

#     for idx, col in enumerate(test_val.columns):
#         y_data = test_val[col]
#         offset = idx * offset_step
#         figmlp.add_trace(go.Scatter(x=test_val.index, y=y_data + offset,
#                                 mode='lines',
#                                 name= col,
#                                 line=dict(width=1),
#                                 hoverinfo='x+y+name',
#                                 showlegend=False))

#     figmlp.update_layout(
#         xaxis_title='Counter',
#         template='plotly_white',
#         height=800,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20)
#     )

#     figmlp.write_html(f'{folder_path}/lineplot{pipe_number}.html', auto_open=False)

def save_lineplot(folder_path, test_val,datafile, pipe_number):
    figmlp = go.Figure()
    offset_step = 1200 
    for idx, col in enumerate(test_val.columns):
        y_data = test_val[col].values
        x_data = test_val.index
        offset_y_data = y_data + (idx * offset_step)
        figmlp.add_trace(go.Scatter(
            # x=[x_data,datafile['ODDO1'].round(2)],  
            x=(datafile['ODDO1']/1000).round(2),
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
            tickmode='array',       #
            tickvals=[idx * offset_step for idx in range(len(test_val.columns))],  
            ticktext=test_val.columns,
            tickfont=dict(
                size=8, 
            )
        )
    )

    figmlp.write_html(f'{folder_path}/lineplot{pipe_number}.html', auto_open=False)

def save_lineplot_raw(folder_path,test_val,pipe_number):
    figmlpraw = go.Figure()

    for idx, col in enumerate(test_val.columns):
        y_data = test_val[col]
        figmlpraw.add_trace(go.Scatter(x=test_val.index, y=y_data,
                                mode='lines',
                                name= col,
                                line=dict(width=1),
                                hoverinfo='x+y+name',
                                showlegend=False))

    figmlpraw.update_layout(
        xaxis_title='Counter',
        template='plotly_white',
        height=300,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    figmlpraw.write_html(f'{folder_path}/lineplot_raw{pipe_number}.html', auto_open=False)

def save_pipe3d(data, data_cp,folder_path,pipe_number):
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)                 # test_val

    num_rows, num_cols = data.shape

    theta = np.linspace(0, 2 * np.pi, num_cols)  
    z = np.linspace(0, 1, num_rows) 
    theta, z = np.meshgrid(theta, z)

    radius = 109.5   # OD = 219mm, R =OD/2
    odometer = num_rows                               # ODDO1

    # Cartesian Coords
    x = odometer * z
    y = radius * np.cos(theta)
    z = radius * np.sin(theta)

    fig = go.Figure(data=[go.Surface(
        x=x,
        y=z,
        z=y,
        # cmin=-10000,           
        # cmax=30000, 
        surfacecolor=data,  
        colorscale='jet',  
        # hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        # text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],            # map_ori_sens
        customdata=data_cp                      # test_val
    )])

    camera = dict(
        eye=dict(x=0., y=5, z=0.), 
        up=dict(x=0, y=1, z=0)  
     )

    odometer_start = 0
    odometer_end = odometer

    fig.add_trace(go.Scatter3d(
        x=[odometer_start, odometer_end], 
        y=[radius, radius], 
        z=[0, 0], 
        text=["3"],
        mode='text',
        textposition="middle center",
        marker=dict(size=0),  
        name="3pm",
        textfont=dict(
            size=20,  
            color="#61090c"  
        )
    ))
    fig.add_trace(go.Scatter3d(
        x=[odometer_start, odometer_end], 
        y=[-radius, -radius], 
        z=[0, 0], 
        text=["9"],
        mode='text',
        textposition="middle center",
        marker=dict(size=0),  
        name="9pm",
        textfont=dict(
            size=20,  
            color="#61090c"  
        )
    ))
    fig.add_trace(go.Scatter3d(
        x=[odometer_start, odometer_end], 
        y=[0, 0], 
        z=[radius, radius], 
        text=["6"],
        mode='text',
        textposition="middle center",
        marker=dict(size=0), 
        name="6pm",
        textfont=dict(
            size=20,  
            color="#61090c"  
        )
    ))
    fig.add_trace(go.Scatter3d(
        x=[odometer_start, odometer_end], 
        y=[0, 0], 
        z=[-radius, -radius], 
        text=["12"],
        mode='text',
        textposition="middle center",
        marker=dict(size=0),
        name="12pm",
        textfont=dict(
            size=20,  
            color="#61090c"  
        )
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

    fig.write_html(f'{folder_path}/pipe3d{pipe_number}.html', auto_open=False)

def create_html_and_csv_from_pkl(pkl_folder='pipes3', output_folder='Client_Pipes',output_callback=None):
    Path(output_folder).mkdir(exist_ok=True)

    for pkl_file in os.listdir(pkl_folder):
        if pkl_file.endswith('.pkl'):
            pipe_number = pkl_file[:-4]  # Remove '.pkl' to get the number
            pipe_folder = Path(output_folder) / f'Pipe_{pipe_number}'
            pipe_folder.mkdir(exist_ok=True)

            try:
                with open(os.path.join(pkl_folder, pkl_file), 'rb') as file:
                    data = pd.read_pickle(file)
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue
            
            dfile = pre_process_data(data, pipe_number,output_folder)

            # Save the CSV file
            csv_file_path = pipe_folder / f'Pipe_{pipe_number}.xlsx'
            dfile.to_excel(csv_file_path, index=False)

            message = f"Processed {pkl_file} and saved to {pipe_folder}"
            if output_callback:
                output_callback(message)
            else:
                print(message)
            # break


if __name__ == "__main__":
    import time
    st = time.time()
    create_html_and_csv_from_pkl()
    print(f'Total time: {time.time()-st} seconds')