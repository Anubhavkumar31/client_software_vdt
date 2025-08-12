from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtWidgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from io import BytesIO
import mpld3
import math
from scipy.signal import savgol_filter
from scipy.ndimage import label, find_objects
from datetime import datetime

class HeatmapWindow(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)

def pre_process2(datafile):
    df_new_tab9 = pd.DataFrame(datafile,columns=[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)])
    sensor_columns = [f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)]
    df1 = df_new_tab9[[f'F{i}H{j}' for i in range(1, 25) for j in range(1, 5)]]

    window_length = 15
    polyorder = 2

    for col in sensor_columns:
        data = df1[col].values
        time_index = np.arange(len(df1))
        coefficients = np.polyfit(time_index, data, 2)
        trend = np.polyval(coefficients, time_index)
        data_dettrended = data - trend
        data_denoised = savgol_filter(data_dettrended,window_length, polyorder)
        df_new_tab9.loc[:len(df1), col] = data_denoised
    
    sens_mean = df_new_tab9.abs().mean()
    sens_mean

    up_sen = 1.8
    lo_sen = -3.5

    up_bo = sens_mean*up_sen
    lo_bo = sens_mean*lo_sen

    for col in df_new_tab9.columns:
        if col in up_bo.index and col in lo_bo.index:
            df_new_tab9[col] = np.where((df_new_tab9[col] >= lo_bo[col]) & (df_new_tab9[col] <= up_bo[col]), 0, df_new_tab9[col])
    
    initial_read = 39.93                    # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
    roll = datafile['ROLL']
    roll = roll - initial_read
    odoData = datafile['ODDO1']

    def pipe3D(data, data_cp, od, map_ori_sens):  
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
            cmin=-10000,           
            cmax=30000, 
            surfacecolor=data,  
            colorscale='jet',  
            # hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
            # text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],            # map_ori_sens
            customdata=data_cp                      # test_val
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title='Odometer',
                yaxis_title='Radial Length',
                zaxis_title='Radial Length',
                aspectmode='data',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            height = 500,
            width=1500
        )

        return fig

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

    figmlp = go.Figure()
    offset_step = 1000

    for idx, col in enumerate(test_val.columns):
        y_data = test_val[col]
        offset = idx * offset_step
        figmlp.add_trace(go.Scatter(x=test_val.index, y=y_data + offset,
                                mode='lines',
                                name= col,
                                line=dict(width=1),
                                hoverinfo='x+y+name',
                                showlegend=False))

    figmlp.update_layout(
        xaxis_title='Counter',
        template='plotly_white',
        height=800,
        width=1500
    )

    figmlp.write_html('backend/files/lineplot.html', auto_open=False)

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
        width=1500
    )

    figmlpraw.write_html('backend/files/lineplot_raw.html', auto_open=False)

    fighmraw = raw_heatmap(datafile)

    fighmraw.write_html('backend/files/heatmap_raw.html', auto_open=False)

    fighm = go.Figure(data=go.Heatmap(
        z=test_val.T,
        y=test_val.columns,
        x=datafile['ODDO1'],
        colorscale='jet',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns], 
        zmin = -12000,
        zmax = 30000
    ))

    fighm.update_layout(
        xaxis_title='Odometer',
        height=500,
        width=1500
    )


    fighm.write_html('backend/files/heatmap.html', auto_open=False)

    fig3d = pipe3D(test_val, test_val, odoData, map_ori_sens)
    fig3d.write_html('backend/files/pipe3d.html', auto_open=False)
    
    defect_sheet = boxing(datafile, test_val, map_ori_sens)
    defect_sheet.to_csv('Data_Folder',index=False, header=True)

    return defect_sheet

def boxing(datafile, test_val, map_ori_sens):
    #Box detection
    test_val = test_val.T
    def boxes_overlap(box1, box2):
        x0_1, x1_1, y0_1, y1_1 = box1
        x0_2, x1_2, y0_2, y1_2 = box2
        
        return not (x1_1 < x0_2 or x0_1 > x1_2 or y1_1 < y0_2 or y0_1 > y1_2)

    def box_contains(outer_box, inner_box):
        x0_outer, x1_outer, y0_outer, y1_outer = outer_box
        x0_inner, x1_inner, y0_inner, y1_inner = inner_box
        
        return (x0_inner >= x0_outer and x1_inner <= x1_outer and
                y0_inner >= y0_outer and y1_inner <= y1_outer)

    def merge_boxes(boxes):
        if not boxes:
            return []

        merged_boxes = []

        for box in boxes:
            new_merged_boxes = []
            merged = False
            
            for merged_box in merged_boxes:
                if boxes_overlap(box, merged_box) or box_contains(merged_box, box):
                    # Merge boxes if they overlap or if box is contained within merged_box
                    merged_box = [
                        min(box[0], merged_box[0]),  # x0
                        max(box[1], merged_box[1]),  # x1
                        min(box[2], merged_box[2]),  # y0
                        max(box[3], merged_box[3])   # y1
                    ]
                    merged = True
                new_merged_boxes.append(merged_box)
            
            if not merged:
                new_merged_boxes.append(box)
            
            merged_boxes = new_merged_boxes

        return merged_boxes

    def get_bounding_boxes(slices):
        boxes = []
        for s in slices:
            if s is not None:
                y_slice, x_slice = s
                y_start, y_end = y_slice.start, y_slice.stop
                x_start, x_end = x_slice.start, x_slice.stop
                boxes.append([x_start - 0.5, x_end - 0.5, y_start - 0.5, y_end - 0.5])
        return boxes

    def get_box_value(box, test_val2, box_type):
        x0, x1, y0, y1 = box
        sub_matrix = test_val2.iloc[int(y0):int(y1), int(x0):int(x1)]
        if sub_matrix.size == 0:
            return 0.0
        
        if box_type == 'Metal_Loss':
            return sub_matrix.max().max()
        elif box_type == 'Metal_Gain':
            return sub_matrix.min().min()

    def get_avg_value(box, test_val2, box_type):
        x0, x1, y0, y1 = box
        means_test_val2 = test_val2.mean()
        sub_matrix = test_val2.iloc[int(y0):int(y1), int(x0):int(x1)]
        if sub_matrix.size == 0:
            return 0.0
    
        return means_test_val2[int(y0)]

    def get_ori_value(box, map_ori_sens, box_type):
        x0, x1, y0, y1 = box
        sub_matrix = map_ori_sens.iloc[int(y0):int(y1), int(x0):int(x1)]
        if sub_matrix.size == 0:
            return None
    
        return sub_matrix.iloc[0,0][2]

    def map_x_coords(original_x_coords, new_x_values):
        x_map = {old: new for old, new in zip(range(len(original_x_coords)), new_x_values)}
        return [x_map[int(x)] for x in original_x_coords]

    def adjust_boxes_x(boxes, original_x_values, new_x_values):
        adjusted_boxes = []
        x_map = {old: new for old, new in zip(original_x_values, new_x_values)}
        for box in boxes:
            x0, x1, y0, y1 = box
            adjusted_boxes.append([
                x_map[int(x0)], x_map[int(x1)],
                y0, y1
            ])
        return adjusted_boxes
    
    def create_box_df(boxes, box_type):
        box_data = []
        for i, box in enumerate(boxes, start=1):
            x0, x1, y0, y1 = box
            width = x1 - x0
            height = y1 - y0
            center_x = (x0 + x1) / 2
            max_value = get_box_value(box, test_val, box_type)
            avg_val = get_avg_value(box, test_val, box_type)
            ori_val = get_ori_value(box, map_ori_sens, box_type)
            box_data.append({
                'Box Number': i,
                'Type': box_type,
                'Width': width,
                'Height': height,
                'x0': x0,
                'y0': y0,
                'x1': x1,
                'y1': y1,
                'StartS': int(y0+0.5),
                'EndS':int(y1+0.5),
                'Absolute Distance': abs(center_x),
                'Peak Value': max_value,
                'Avg_Val' : avg_val,
                'Ori Val': ori_val
            })
        return pd.DataFrame(box_data)
    
    metal_loss_mask = (test_val > 0).astype(int)
    metal_gain_mask = (test_val < 0).astype(int)

    structure2 = np.ones((3, 3), dtype=float)
    labeled_loss, num_features_loss = label(metal_loss_mask, structure2)
    labeled_gain, num_features_gain = label(metal_gain_mask, structure2)

    slices_loss = find_objects(labeled_loss)
    slices_gain = find_objects(labeled_gain)

    boxes_loss = get_bounding_boxes(slices_loss)
    boxes_gain = get_bounding_boxes(slices_gain)

    original_x_coords = list(range(len(test_val.columns)))
    mapped_x_coords = map_x_coords(original_x_coords, datafile['ODDO1'])

    adjusted_boxes_loss = adjust_boxes_x(boxes_loss, original_x_coords, mapped_x_coords)
    adjusted_boxes_gain = adjust_boxes_x(boxes_gain, original_x_coords, mapped_x_coords)
    
    merged_boxes_loss = merge_boxes(adjusted_boxes_loss)
    merged_boxes_gain = merge_boxes(adjusted_boxes_gain)

    unique_boxes_loss = []
    for box in merged_boxes_loss:
        if not any(box_contains(other, box) for other in unique_boxes_loss):
            unique_boxes_loss.append(box)

    unique_boxes_gain = []
    for box in merged_boxes_gain:
        if not any(box_contains(other, box) for other in unique_boxes_gain):
            unique_boxes_gain.append(box)

    df_boxes_loss = create_box_df(unique_boxes_loss, 'Metal_Loss')
    df_boxes_gain = create_box_df(unique_boxes_gain, 'Metal_Gain')

    # Concatenate and sort by 'x0' and 'y0'
    df_boxes = pd.concat([df_boxes_loss, df_boxes_gain])
    df_boxes = df_boxes.sort_values(by=['y0', 'x0'], ascending=[False, True])

    fighmbox = go.Figure(data=go.Heatmap(
        z=test_val,
        y=test_val.index,
        x=datafile['ODDO1'],
        colorscale='jet',
        hovertemplate='(%{x}, %{z})<br>Sensor: %{text[0]}<br>Actual Ori: %{text[2]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],
        zmin=-12000,
        zmax=30000
    ))

    # Add rectangles and annotations for merged bounding boxes
    for _, row in df_boxes.iterrows():
        x0, x1, y0, y1 = row['x0'], row['x0'] + row['Width'], row['y0'], row['y0'] + row['Height']
        box_number = row['Box Number']
        color = 'red' if row['Type'] == 'Metal_Loss' else 'green'
        fillcolor = 'rgba(255, 0, 0, 0.1)' if row['Type'] == 'Metal_Loss' else 'rgba(0, 255, 0, 0.1)'
        
        fighmbox.add_shape(
            type='rect',
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color=color, width=2),
            fillcolor=fillcolor
        )
        tt = 'L' if row["Type"] == 'Metal_Loss' else 'G'
        # Add text annotation to the top left of the box with a margin
        fighmbox.add_annotation(
            x=x0 - 0.5,  
            y=y1 + 0.5,  
            text=f'{tt}: {box_number}',
            showarrow=False,
            font=dict(size=12, color=color),
            align='center',
            xref='x',
            yref='y'
        )

    # Update layout with titles and axis labels
    fighmbox.update_layout(
        title='Defect Detection',
        xaxis_title='Odometer',
        yaxis_title='Orientation',
        yaxis=dict(scaleanchor='x'),  
        xaxis=dict(constrain='domain'),  
        autosize=False,
        height=500,
        width=1500
    )

    fighmbox.write_html('backend/files/heatmap_box.html', auto_open=False)

    #Breadth Calulation
    def breadth(start_sensor,end_sensor):
        if start_sensor == end_sensor:
            bredth = 0
            return bredth
        else:
            outer_diameter_1 = 219
            thickness_1 = 12.7
            inner_diameter_1 = outer_diameter_1 - 2 * (thickness_1)
            radius_1 = inner_diameter_1 / 2

            theta_2 = 2.36
            c_1 = math.radians(theta_2)
            theta_3 = 5.69
            d_1 = math.radians(theta_3)

            x1 = radius_1 * c_1
            y1 = radius_1 * d_1
            count = 0
            if start_sensor == end_sensor:
                bredth = 0
                return bredth
            else:
                for i in range(start_sensor, end_sensor):
                    if i % 4 == 0:
                        count = count + 1
                        k = (y1 - x1) * count
                    else:
                        pass
                try:
                    l = (end_sensor - start_sensor) * x1
                    bredth = k + l
                    return bredth
                    # bredth=bredth*0.79
                    # bredth=bredth*1.08
                except:
                    k = 0
                    l = (end_sensor - start_sensor) * x1
                    bredth = k + l
                    return bredth
    df_boxes['Breadth'] = df_boxes.apply(lambda row: breadth(row['StartS'], row['EndS']), axis=1)

    return df_boxes

def pre_process(datafile):
    pat1 = re.compile(r'^F\d+H\d+$')   # F%H%
    sel_col = ['ROLL','ODDO1']


    def matches_pat(column):
        return pat1.match(column)
    
    def find_ind_known_defect(odo):
        for index,e in datafile.iterrows(): 
            if odo == int(e[1]):
                return index


    fil_col = [col for col in datafile.columns if matches_pat(col) or col in sel_col]

    refData = datafile[fil_col]
    initial_read = 39.93                    # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
    roll = refData['ROLL']
    roll = roll - initial_read
    odoData = refData['ODDO1']
    sensD = refData[refData.columns[2:]]

    def degree_to_clock_position(degrees):
        normalized_degrees = degrees % 360

        hours = normalized_degrees // 30
        minutes = (normalized_degrees % 30) * 2

        if hours == 0:
            hours = 12  # 0 is 12:00
        formatted_hours = '{:02d}'.format(int(hours))
        formatted_minutes = '{:02d}'.format(int(minutes))
        
        clock_position = formatted_hours + ':' + formatted_minutes
        
        return clock_position

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

    odosensD = sensD.copy()
    odosensD.insert(0, 'Oddo', odoData)

    sensor_d = sensD.copy()

    fig2 = go.Figure()
    offset_step = 1500  # Adjust the offset

    for idx, col in enumerate(sensor_coluns):
        y_data = odosensD[col]
        offset = idx * offset_step
        fig2.add_trace(go.Scatter(x=odosensD['Oddo'], y=y_data + offset,
                                mode='lines',
                                line=dict(width=1),
                                hoverinfo='x+y+name',
                                showlegend=False ))

    fig2.update_layout(
        xaxis_title='Odometer, mm',
        template='plotly_white',
        height=800,
        width=1500
    )

    fig2.write_html('backend/files/lineplot.html', auto_open=False)


    fig2_1 = go.Figure()

    for idx, col in enumerate(sensor_coluns):
        y_data = odosensD[col]
        fig2_1.add_trace(go.Scatter(x=odosensD['Oddo'], y=y_data,
                                mode='lines',
                                name=col,
                                line=dict(width=1),
                                hoverinfo='x+y+name',
                                showlegend=False))

    fig2_1.update_layout(
        template='plotly_white',
        height=300,
        width=1500,
        yaxis_range=[-20000,45000]
    )
    fig2_1.write_html('backend/files/lineplot_raw.html', auto_open=False)

    html_content = pio.to_html(fig2, full_html=False)

    lineplot = html_content


    x = odosensD.iloc[:, 1:40]
    y = odosensD.iloc[:, 40:]

    new_odosensD = pd.concat([y,x], axis=1)

    data = {}

    for col in sensor_d.columns:
        col_index_in_Df2 = sensor_d.columns.get_loc(col)
        pairs = []
        
        for index, (value1, value2) in enumerate(zip(sensor_d[col], test_clockData.iloc[:, col_index_in_Df2])):
            pairs.append((value1, value2))
        
        data[col] = pairs

    new_df = pd.DataFrame(data)

    test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, sensor_d.columns)))

    def create_time_dict():
        time_ranges_1 = [
            '12:00-12:07', '12:07-12:15', '12:15-12:22', '12:22-12:30', '12:30-12:37', '12:37-12:45', '12:45-12:52', '12:52-01:00',
            '01:00-01:07', '01:07-01:15', '01:15-01:22', '01:22-01:30', '01:30-01:37', '01:37-01:45', '01:45-01:52', '01:52-02:00',
            '02:00-02:07', '02:07-02:15', '02:15-02:22', '02:22-02:30', '02:30-02:37', '02:37-02:45', '02:45-02:52', '02:52-03:00',
            '03:00-03:07', '03:07-03:15', '03:15-03:22', '03:22-03:30', '03:30-03:37', '03:37-03:45', '03:45-03:52', '03:52-04:00',
            '04:00-04:07', '04:07-04:15', '04:15-04:22', '04:22-04:30', '04:30-04:37', '04:37-04:45', '04:45-04:52', '04:52-05:00',
            '05:00-05:07', '05:07-05:15', '05:15-05:22', '05:22-05:30', '05:30-05:37', '05:37-05:45', '05:45-05:52', '05:52-06:00',
            '06:00-06:07', '06:07-06:15', '06:15-06:22', '06:22-06:30', '06:30-06:37', '06:37-06:45', '06:45-06:52', '06:52-07:00',
            '07:00-07:07', '07:07-07:15', '07:15-07:22', '07:22-07:30', '07:30-07:37', '07:37-07:45', '07:45-07:52', '07:52-08:00',
            '08:00-08:07', '08:07-08:15', '08:15-08:22', '08:22-08:30', '08:30-08:37', '08:37-08:45', '08:45-08:52', '08:52-09:00',
            '09:00-09:07', '09:07-09:15', '09:15-09:22', '09:22-09:30', '09:30-09:37', '09:37-09:45', '09:45-09:52', '09:52-10:00',
            '10:00-10:07', '10:07-10:15', '10:15-10:22', '10:22-10:30', '10:30-10:37', '10:37-10:45', '10:45-10:52', '10:52-11:00',
            '11:00-11:07', '11:07-11:15', '11:15-11:22', '11:22-11:30', '11:30-11:37', '11:37-11:45', '11:45-11:52', '11:52-12:00',
        ]

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
                # print(s,d)
                ind = xl.index(d)
                # print(ind)
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
            cell_v = sensor_d.at[r,tup_value]
            # print(cell_v)
            test_val.iloc[r,c] = cell_v
            c += 1

    odo_m = (odosensD['Oddo']/1000).round(2)


    heatmap_fin = plt.figure(figsize=(28, 3))
    plt.imshow(new_odosensD.T, cmap='turbo', aspect='auto')
    plt.xlabel('Odometer')
    plt.tight_layout()

    old_test_val = test_val.copy()
    test_val[test_val < 0] = 0

    heatmap = go.Heatmap(
        z=test_val.T,
        x=[test_val.index,odo_m],
        y=test_val.columns,
        colorscale='Turbo',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns], 
        zmin = -12000,
        zmax = 40000
    )

    layout = go.Layout(
        xaxis=dict(title='Counter/Odometer'),
        height=450,  
        width=1500,
    )

    fig3 = go.Figure(data=[heatmap], layout=layout)
    fig3.update_layout(
        coloraxis_showscale=False
    )
    fig3.update_xaxes(tickangle=90)
    fig3.write_html('backend/files/heatmap.html', auto_open=False)


    heatmap = go.Heatmap(
        z=test_val.T,
        x=[test_val.index,odo_m],
        y=test_val.columns,
        colorscale='Turbo',
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns], 
        zmin = -12000,
        zmax = 40000
    )

    layout = go.Layout(
        height=300,  
        width=1500,
    )

    fig3_1 = go.Figure(data=[heatmap], layout=layout)
    fig3_1.update_layout(
        coloraxis_showscale=False
    )
    fig3_1.update_xaxes(tickangle=90)
    fig3_1.write_html('backend/files/heatmap_raw.html', auto_open=False)

    img_data1 = BytesIO()
    heatmap_fin.savefig(img_data1, format='png')
    img_data1.seek(0)

    img_data2 = fig2.to_image(format='png')

    return (img_data1,BytesIO(img_data2))



def raw_heatmap(dataf):
    initial_read = 39.93                    # At 400mm, F1H1 detects defect at 11:00 with roll 39.93
    roll = dataf['ROLL']
    roll = roll - initial_read
    odoData = dataf['ROLL']

    def degree_to_clock_position(degrees):
        normalized_degrees = degrees % 360

        hours = normalized_degrees // 30
        minutes = (normalized_degrees % 30) * 2

        if hours == 0:
            hours = 12  # 0 is 12:00
        formatted_hours = '{:02d}'.format(int(hours))
        formatted_minutes = '{:02d}'.format(int(minutes))
        
        clock_position = formatted_hours + ':' + formatted_minutes
        
        return clock_position

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

    test_clockData = test_clockData.rename(columns=dict(zip(test_clockData.columns, sensor_coluns)))
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
        x=dataf['ODDO1'],
        colorscale='jet', 
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],  
        zmin = -12000,
        zmax = 30000
    ))

    figraw.update_layout(
        height=300,
        width=1500
    )
    
    return figraw
