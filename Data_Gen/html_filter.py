# '''
# This code will input the raw pkl file( {x}.pkl ) and generate out the plots to:
# Pipe_{x}  with ending format as <Name_{x}.html>
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
# import joblib
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# INITIAL_READ = 0.0                   # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
# UPPER_SENS_MUL = 1
# LOWER_SENS_MUL = 3

# def pre_process_data(datafile, pipe_number,output_folder):
#     # Create DataFrame from raw data
#     df_new_tab9 = pd.DataFrame(datafile, columns=[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)])
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
#     # up_sen = 1.8
#     # lo_sen = -3.5

#     # up_bo = sens_mean*up_sen
#     # lo_bo = sens_mean*lo_sen

#     # for col in df_new_tab9.columns:
#     #     if col in up_bo.index and col in lo_bo.index:
#     #         df_new_tab9[col] = np.where((df_new_tab9[col] >= lo_bo[col]) & (df_new_tab9[col] <= up_bo[col]), 0, df_new_tab9[col])

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

#     map_val_sens = pd.DataFrame(index=test_val.index, columns=test_val.columns)
#     for column in test_val.columns:
#         for i in range(test_val.shape[0]):
#             map_value = map_ori_sens.loc[i, column]
#             test_value = test_val.loc[i, column]
#             map_val_sens.loc[i, column] = (*map_value, test_value)
    
#     create_plots(df_new_tab9,df_raw_straight, datafile, test_val,map_ori_sens, pipe_number,output_folder)

#     return datafile

# def create_plots(df_new_tab9,df_raw_straight, datafile, test_val,map_ori_sens, pipe_number,output_folder):
#     folder_path = f'{output_folder}/Pipe_{pipe_number}'
#     os.makedirs(folder_path, exist_ok=True)

#     # Heatmap
#     save_heatmap(test_val, datafile,map_ori_sens, folder_path,pipe_number)

#     # Heatmap Raw
#     #save_heatmap_raw(folder_path,df_raw_straight,datafile,pipe_number)

#     # MultilinePlot 
#     save_lineplot(folder_path,test_val,datafile,pipe_number)

#     # MultilinePlot Raw
#     #save_lineplot_raw(folder_path,test_val,pipe_number)

#     # 3D Pipe 
#     save_pipe3d(test_val,test_val,folder_path,pipe_number)

#     # NEW: H*P* linechart
#     save_hp_linechart(folder_path, datafile, pipe_number, offset=False)  # set True for stacked/offset view


# def save_heatmap(test_val, datafile,map_ori_sens, folder_path,pipe_number):
#     fighm = go.Figure(data=go.Heatmap(
#         z=test_val.T,
#         y=test_val.columns,
#         # x=[test_val.index,datafile['ODDO1'].round(2)],
#         x=(datafile['ODDO1']/1000).round(2),
#         colorscale='jet',
#         hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
#         text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns], 
#         # zmin = -12000,
#         # zmax = 30000
#     ))
#     fighm.update_layout(
#         xaxis_title='Absolute Distance (m)',
#         height=500,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20)
#     )
#     fighm.write_html(f'{folder_path}/heatmap{pipe_number}.html', auto_open=False)

# def save_hp_linechart(folder_path, datafile, pipe_number, offset=False):
#     """
#     Build a multi-line chart from columns like H1P1, H1P2, H3P1, ...
#     and save it as linechart{pipe_number}.html in the pipe's folder.

#     Args:
#         folder_path (str): output directory for the pipe.
#         datafile (pd.DataFrame): raw dataframe (contains ODDO1 if available).
#         pipe_number (str|int): current pipe number.
#         offset (bool): if True, vertically offset each series for readability.
#     """
#     # find H#P# columns
#     hp_cols = []
#     for c in datafile.columns:
#         cs = str(c)
#         m = re.match(r'^H(\d+)P(\d+)$', cs)
#         if m:
#             h_num = int(m.group(1))
#             p_num = int(m.group(2))
#             hp_cols.append((h_num, p_num, cs))

#     if not hp_cols:
#         print(f"No H#P# columns found for pipe {pipe_number}. Skipping H*P* linechart.")
#         return

#     # sort by (H, P) numerically so H1P2 < H1P10, etc.
#     hp_cols.sort(key=lambda t: (t[0], t[1]))
#     hp_names = [t[2] for t in hp_cols]

#     # x-axis: Absolute Distance (m) if available, else index
#     if 'ODDO1' in datafile.columns:
#         x_vals = (datafile['ODDO1'] / 1000).round(2)
#         x_title = 'Absolute Distance (m)'
#     else:
#         x_vals = np.arange(len(datafile))
#         x_title = 'Index'

#     fig = go.Figure()

#     # optional vertical offset for readability
#     if offset:
#         # choose a step based on data scale
#         # robust: use IQR-ish scale across all selected columns
#         vals = pd.concat([datafile[col] for col in hp_names], axis=1)
#         q75 = np.nanpercentile(vals.values, 75)
#         q25 = np.nanpercentile(vals.values, 25)
#         step = max(1.0, (q75 - q25))  # fallback minimum
#     else:
#         step = 0.0

#     for idx, col in enumerate(hp_names):
#         y = datafile[col].values
#         if offset:
#             y = y + idx * step
#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=y,
#             mode='lines',
#             name=col,
#             line=dict(width=1),
#             hoverinfo='x+y+name',
#             showlegend=True
#         ))

#     # y tick labels (only if offset view)
#     yaxis_cfg = dict()
#     if offset:
#         yaxis_cfg.update({
#             "tickmode": "array",
#             "tickvals": [i * step for i in range(len(hp_names))],
#             "ticktext": hp_names,
#             "tickfont": {"size": 9},
#         })

#     fig.update_layout(
#         xaxis_title=x_title,
#         yaxis_title='H*P readings',
#         template='plotly_white',
#         height=700 if not offset else 900,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20),
#         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
#         yaxis=yaxis_cfg
#     )

#     fig.write_html(f'{folder_path}/linechart{pipe_number}.html', auto_open=False)



# def save_heatmap_raw(folder_path,df_raw_straight,dataf,pipe_number):
#     initial_read = INITIAL_READ
#     roll = dataf['ROLL']
#     roll = roll - initial_read
#     odoData = dataf['ROLL']

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

#     pat1 = re.compile(r'^F\d+H\d+$')   # F%H%
#     sel_col = ['ROLL','ODDO1']
#     def matches_pat(column):
#         return pat1.match(column)
#     fil_col = [col for col in dataf.columns if matches_pat(col) or col in sel_col]
#     refData = dataf[fil_col]
#     sensD = refData[refData.columns[2:]]

#     test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, df_raw_straight.columns)))
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
#             cell_v = sensD.at[r,tup_value]
#             # print(cell_v)
#             test_val.iloc[r,c] = cell_v
#             c += 1
    
#     figraw = go.Figure(data=go.Heatmap(
#         z=test_val.T,
#         y=test_val.columns,
#         x=(dataf['ODDO1']/1000).round(2),
#         colorscale='jet', 
#         hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
#         text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],  
#         zmin = -12000,
#         zmax = 30000
#     ))

#     figraw.update_layout(
#         height=300,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20)
#     )

#     figraw.write_html(f'{folder_path}/heatmap_raw{pipe_number}.html', auto_open=False)

# # def save_lineplot(folder_path,test_val,pipe_number):
# #     figmlp = go.Figure()
# #     offset_step = 1000

# #     for idx, col in enumerate(test_val.columns):
# #         y_data = test_val[col]
# #         offset = idx * offset_step
# #         figmlp.add_trace(go.Scatter(x=test_val.index, y=y_data + offset,
# #                                 mode='lines',
# #                                 name= col,
# #                                 line=dict(width=1),
# #                                 hoverinfo='x+y+name',
# #                                 showlegend=False))

# #     figmlp.update_layout(
# #         xaxis_title='Counter',
# #         template='plotly_white',
# #         height=800,
# #         width=1500,
# #         margin=dict(l=20, r=20, t=50, b=20)
# #     )

# #     figmlp.write_html(f'{folder_path}/lineplot{pipe_number}.html', auto_open=False)

# def save_lineplot(folder_path, test_val,datafile, pipe_number):
#     figmlp = go.Figure()
#     offset_step = 1200 
#     for idx, col in enumerate(test_val.columns):
#         y_data = test_val[col].values
#         x_data = test_val.index
#         offset_y_data = y_data + (idx * offset_step)
#         figmlp.add_trace(go.Scatter(
#             # x=[x_data,datafile['ODDO1'].round(2)],  
#             x=(datafile['ODDO1']/1000).round(2),
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
#             tickmode='array',       #
#             tickvals=[idx * offset_step for idx in range(len(test_val.columns))],  
#             ticktext=test_val.columns,
#             tickfont=dict(
#                 size=8, 
#             )
#         )
#     )

#     figmlp.write_html(f'{folder_path}/lineplot{pipe_number}.html', auto_open=False)

# def save_lineplot_raw(folder_path,test_val,pipe_number):
#     figmlpraw = go.Figure()

#     for idx, col in enumerate(test_val.columns):
#         y_data = test_val[col]
#         figmlpraw.add_trace(go.Scatter(x=test_val.index, y=y_data,
#                                 mode='lines',
#                                 name= col,
#                                 line=dict(width=1),
#                                 hoverinfo='x+y+name',
#                                 showlegend=False))

#     figmlpraw.update_layout(
#         xaxis_title='Counter',
#         template='plotly_white',
#         height=300,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20)
#     )

#     figmlpraw.write_html(f'{folder_path}/lineplot_raw{pipe_number}.html', auto_open=False)

# def save_pipe3d(data, data_cp,folder_path,pipe_number):
#     if not isinstance(data, np.ndarray):
#         data = np.asarray(data)                 # test_val

#     num_rows, num_cols = data.shape

#     theta = np.linspace(0, 2 * np.pi, num_cols)  
#     z = np.linspace(0, 1, num_rows) 
#     theta, z = np.meshgrid(theta, z)

#     radius = 109.5   # OD = 219mm, R =OD/2
#     odometer = num_rows                               # ODDO1

#     # Cartesian Coords
#     x = odometer * z
#     y = radius * np.cos(theta)
#     z = radius * np.sin(theta)

#     fig = go.Figure(data=[go.Surface(
#         x=x,
#         y=z,
#         z=y,
#         # cmin=-10000,           
#         # cmax=30000, 
#         surfacecolor=data,  
#         colorscale='jet',  
#         # hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
#         # text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],            # map_ori_sens
#         customdata=data_cp                      # test_val
#     )])

#     camera = dict(
#         eye=dict(x=0., y=5, z=0.), 
#         up=dict(x=0, y=1, z=0)  
#      )

#     odometer_start = 0
#     odometer_end = odometer

#     fig.add_trace(go.Scatter3d(
#         x=[odometer_start, odometer_end], 
#         y=[radius, radius], 
#         z=[0, 0], 
#         text=["3"],
#         mode='text',
#         textposition="middle center",
#         marker=dict(size=0),  
#         name="3pm",
#         textfont=dict(
#             size=20,  
#             color="#61090c"  
#         )
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=[odometer_start, odometer_end], 
#         y=[-radius, -radius], 
#         z=[0, 0], 
#         text=["9"],
#         mode='text',
#         textposition="middle center",
#         marker=dict(size=0),  
#         name="9pm",
#         textfont=dict(
#             size=20,  
#             color="#61090c"  
#         )
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=[odometer_start, odometer_end], 
#         y=[0, 0], 
#         z=[radius, radius], 
#         text=["6"],
#         mode='text',
#         textposition="middle center",
#         marker=dict(size=0), 
#         name="6pm",
#         textfont=dict(
#             size=20,  
#             color="#61090c"  
#         )
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=[odometer_start, odometer_end], 
#         y=[0, 0], 
#         z=[-radius, -radius], 
#         text=["12"],
#         mode='text',
#         textposition="middle center",
#         marker=dict(size=0),
#         name="12pm",
#         textfont=dict(
#             size=20,  
#             color="#61090c"  
#         )
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

# def create_html_and_csv_from_pkl(pkl_folder='pipes3', output_folder='Client_Pipes',output_callback=None):
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
            
#             dfile = pre_process_data(data, pipe_number,output_folder)

#             # Save the CSV file
#             csv_file_path = pipe_folder / f'Pipe_{pipe_number}.xlsx'
#             dfile.to_excel(csv_file_path, index=False)

#             message = f"Processed {pkl_file} and saved to {pipe_folder}"
#             if output_callback:
#                 output_callback(message)
#             else:
#                 print(message)
#             # break


# if __name__ == "__main__":
#     import time
#     st = time.time()
#     create_html_and_csv_from_pkl()
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
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- CONFIG --------------------
INITIAL_READ = 0.0      # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
UPPER_SENS_MUL = 1
LOWER_SENS_MUL = 3
# -1 / 0 / None / "auto" => auto (CPU-1, at least 1). Or set an int, e.g. 4
WORKERS = 4
# ------------------------------------------------


def pre_process_data(datafile, pipe_number, output_folder):
    # Create DataFrame from raw data
    df_new_tab9 = pd.DataFrame(
        datafile,
        columns=[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)]
    )
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
    
    create_plots(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder)
    return datafile


def create_plots(df_new_tab9, df_raw_straight, datafile, test_val, map_ori_sens, pipe_number, output_folder):
    folder_path = f'{output_folder}/Pipe_{pipe_number}'
    os.makedirs(folder_path, exist_ok=True)

    # Heatmap
    save_heatmap(test_val, datafile, map_ori_sens, folder_path, pipe_number)

    # MultilinePlot (offset stack of sensors)
    save_lineplot(folder_path, test_val, datafile, pipe_number)

    # 3D Pipe 
    save_pipe3d(test_val, test_val, folder_path, pipe_number)

    # inside create_plots(...) or wherever you save other charts:
    save_proximity_linechart(folder_path, datafile, pipe_number)



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
    fighm.write_html(f'{folder_path}/heatmap{pipe_number}.html', auto_open=False)


# def save_hp_linechart(folder_path, datafile, pipe_number, offset=False):
#     """
#     Build a multi-line chart from columns like H1P1, H1P2, H3P1, ...
#     and save it as linechart{pipe_number}.html in the pipe's folder.
#     """
#     # find H#P# columns
#     hp_cols = []
#     for c in datafile.columns:
#         cs = str(c)
#         m = re.match(r'^F(\d+)P(\d+)$', cs)
#         if m:
#             h_num = int(m.group(1))
#             p_num = int(m.group(2))
#             hp_cols.append((h_num, p_num, cs))

#     if not hp_cols:
#         print(f"No H#P# columns found for pipe {pipe_number}. Skipping H*P* linechart.")
#         return

#     # sort by (H, P) numerically so H1P2 < H1P10, etc.
#     hp_cols.sort(key=lambda t: (t[0], t[1]))
#     hp_names = [t[2] for t in hp_cols]

#     # x-axis: Absolute Distance (m) if available, else index
#     if 'ODDO1' in datafile.columns:
#         x_vals = (datafile['ODDO1'] / 1000).round(2)
#         x_title = 'Absolute Distance (m)'
#     else:
#         x_vals = np.arange(len(datafile))
#         x_title = 'Index'

#     fig = go.Figure()

#     # optional vertical offset for readability
#     if offset:
#         vals = pd.concat([datafile[col] for col in hp_names], axis=1)
#         q75 = np.nanpercentile(vals.values, 75)
#         q25 = np.nanpercentile(vals.values, 25)
#         step = max(1.0, (q75 - q25))
#     else:
#         step = 0.0

#     for idx, col in enumerate(hp_names):
#         y = datafile[col].values
#         if offset:
#             y = y + idx * step
#         fig.add_trace(go.Scatter(
#             x=x_vals,
#             y=y,
#             mode='lines',
#             name=col,
#             line=dict(width=1),
#             hoverinfo='x+y+name',
#             showlegend=True
#         ))

#     # y tick labels (only if offset view)
#     yaxis_cfg = dict()
#     if offset:
#         yaxis_cfg.update({
#             "tickmode": "array",
#             "tickvals": [i * step for i in range(len(hp_names))],
#             "ticktext": hp_names,
#             "tickfont": {"size": 9},
#         })

#     fig.update_layout(
#         xaxis_title=x_title,
#         yaxis_title='H*P readings',
#         template='plotly_white',
#         height=700 if not offset else 900,
#         width=1500,
#         margin=dict(l=20, r=20, t=50, b=20),
#         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
#         yaxis=yaxis_cfg
#     )

#     fig.write_html(f'{folder_path}/proximity_linechart{pipe_number}.html', auto_open=False)





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
    fig.write_html(f'{folder_path}/proximity_linechart{pipe_number}.html', auto_open=False)

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

    figraw.write_html(f'{folder_path}/heatmap_raw{pipe_number}.html', auto_open=False)


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

    figmlp.write_html(f'{folder_path}/lineplot{pipe_number}.html', auto_open=False)


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

    figmlpraw.write_html(f'{folder_path}/lineplot_raw{pipe_number}.html', auto_open=False)


def save_pipe3d(data, data_cp, folder_path, pipe_number):
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)  # test_val

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

    fig.write_html(f'{folder_path}/pipe3d{pipe_number}.html', auto_open=False)


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
