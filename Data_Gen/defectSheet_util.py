'''
This code will create the raw defectS{x}.csv 
'''

import numpy as np
import pandas as pd
from scipy.ndimage import label, find_objects
import plotly.graph_objects as go
import math
from sklearn.cluster import DBSCAN

OUTER_DIA_1 = 219        # Pipe1 and Pipe2
THICKNESS = 12.7         # Pipe Thickness

DEPTH_VAL_LIMIT = 4
WIDTH_LIMIT = 10

def boxing(datafile, test_val, map_ori_sens):
    test_val = test_val.T
    def boxes_overlap(box1, box2, tolerance=5.0):
        x0_1, x1_1, y0_1, y1_1 = box1
        x0_2, x1_2, y0_2, y1_2 = box2

        return not (x1_1 + tolerance < x0_2 or x0_1 - tolerance > x1_2 or 
                    y1_1 + tolerance < y0_2 or y0_1 - tolerance > y1_2)

    def box_contains(outer_box, inner_box):
        x0_outer, x1_outer, y0_outer, y1_outer = outer_box
        x0_inner, x1_inner, y0_inner, y1_inner = inner_box
        
        return (x0_inner >= x0_outer and x1_inner <= x1_outer and
                y0_inner >= y0_outer and y1_inner <= y1_outer)

    def merge_boxes_with_clustering(boxes, eps=5.0, min_samples=1):
        box_centers = np.array([[(box[0] + box[1]) / 2, (box[2] + box[3]) / 2] for box in boxes])
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(box_centers)
        
        merged_boxes = []
        for cluster_label in set(clustering.labels_):
            if cluster_label == -1:
                continue
            
            # Get all boxes in this cluster
            cluster_boxes = [boxes[i] for i in range(len(boxes)) if clustering.labels_[i] == cluster_label]
            
            # Merge boxes into a single bounding box
            x0 = min(box[0] for box in cluster_boxes)
            x1 = max(box[1] for box in cluster_boxes)
            y0 = min(box[2] for box in cluster_boxes)
            y1 = max(box[3] for box in cluster_boxes)
            merged_boxes.append([x0, x1, y0, y1])

        return merged_boxes

    def merge_boxes_with_overlap_and_containment(boxes, tolerance=5.0):
        def boxes_overlap(box1, box2, tolerance):
            x0_1, x1_1, y0_1, y1_1 = box1
            x0_2, x1_2, y0_2, y1_2 = box2
            return not (x1_1 + tolerance < x0_2 or x0_1 - tolerance > x1_2 or y1_1 + tolerance < y0_2 or y0_1 - tolerance > y1_2)

        def box_contains(outer_box, inner_box):
            x0_outer, x1_outer, y0_outer, y1_outer = outer_box
            x0_inner, x1_inner, y0_inner, y1_inner = inner_box
            return (x0_inner >= x0_outer and x1_inner <= x1_outer and y0_inner >= y0_outer and y1_inner <= y1_outer)

        merged_boxes = []
        debug_info = []  # For storing debug information

        for box in boxes:
            x0, x1, y0, y1 = box
            merged = False
            debug_entry = {
                "box": box,
                "merged_with": None,
                "contained_in": None,
                "replaced_existing": None
            }

            for i, merged_box in enumerate(merged_boxes):
                if boxes_overlap(box, merged_box, tolerance=tolerance):
                    # Merge boxes
                    new_x0 = min(merged_box[0], x0)
                    new_x1 = max(merged_box[1], x1)
                    new_y0 = min(merged_box[2], y0)
                    new_y1 = max(merged_box[3], y1)
                    merged_boxes[i] = [new_x0, new_x1, new_y0, new_y1]
                    debug_entry["merged_with"] = merged_box
                    merged = True
                    break
                elif box_contains(merged_box, box):
                    # Skip adding this box if it is contained within an existing one
                    debug_entry["contained_in"] = merged_box
                    merged = True
                    break
                elif box_contains(box, merged_box):
                    # Replace an existing box if it is contained within the new box
                    debug_entry["replaced_existing"] = merged_box
                    merged_boxes[i] = box
                    merged = True
                    break

            if not merged:
                merged_boxes.append([x0, x1, y0, y1])

            # Add the debug entry to the log
            debug_info.append(debug_entry)

        # Print debug information
        # for entry in debug_info:
        #     print("Processing Box:", entry["box"])
        #     if entry["merged_with"]:
        #         print(f" - Merged with existing box: {entry['merged_with']}")
        #     if entry["contained_in"]:
        #         print(f" - Skipped, as it is contained in box: {entry['contained_in']}")
        #     if entry["replaced_existing"]:
        #         print(f" - Replaced existing box: {entry['replaced_existing']}")
        # print("\nFinal Merged Boxes:", merged_boxes)

        def is_contained(box1, box2):
            return (box1[0] >= box2[0] and
                    box1[1] >= box2[1] and
                    box1[2] <= box2[2] and
                    box1[3] <= box2[3])

        def is_overlapping(box1, box2):
            return (box1[0] < box2[2] and
                    box1[2] > box2[0] and
                    box1[1] < box2[3] and
                    box1[3] > box2[1])

        def create_bigger_box(box1, box2):
            return [min(box1[0], box2[0]),min(box1[1], box2[1]),max(box1[2], box2[2]),max(box1[3], box2[3])]


        def find_final_boxes(merged_boxes: list[list]):
            final_boxes = merged_boxes.copy()  # Start with the original list
            processed_boxes = []  # To store the new boxes created from overlaps
            skip_indices = set()  # Track indices of boxes to skip

            # Iterate through each pair of boxes
            for i, box1 in enumerate(merged_boxes):
                if i in skip_indices:
                    continue  # Skip already processed boxes

                has_changed = False  # Track if the box has been modified

                for j, box2 in enumerate(merged_boxes):
                    if i != j and j not in skip_indices:
                        if is_contained(box1, box2):
                            # If box1 is contained within box2, remove it
                            if box1 in final_boxes:
                                final_boxes.remove(box1)
                            skip_indices.add(i)
                            has_changed = True
                            break  # Move to the next box1

                        elif is_overlapping(box1, box2):
                            # If overlapping, create a larger box and add both to processed_boxes
                            new_box = create_bigger_box(box1, box2)
                            if new_box not in processed_boxes:
                                processed_boxes.append(new_box)
                            skip_indices.add(i)
                            skip_indices.add(j)
                            has_changed = True

                # If box1 was modified, add the updated version to final_boxes
                if has_changed and box1 not in skip_indices:
                    final_boxes.append(box1)
            
            # Merge processed boxes with final boxes
            final_boxes.extend(processed_boxes)

            # Filter out any duplicates that might have been added
            unique_final_boxes = []
            for box in final_boxes:
                if box not in unique_final_boxes:
                    unique_final_boxes.append(box)

            return unique_final_boxes, processed_boxes


        # final_boxes, filter_boxes = find_final_boxes(merged_boxes)

        return merged_boxes

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
    
    # merged_boxes_loss = merge_boxes(adjusted_boxes_loss)
    # merged_boxes_gain = merge_boxes(adjusted_boxes_gain)

    merged_boxes_loss = merge_boxes_with_overlap_and_containment(adjusted_boxes_loss, tolerance=15)
    merged_boxes_gain = merge_boxes_with_overlap_and_containment(adjusted_boxes_gain, tolerance=15)

    # merged_boxes_loss = merge_boxes_with_clustering(adjusted_boxes_loss, eps=5.0, min_samples=1)
    # merged_boxes_gain = merge_boxes_with_clustering(adjusted_boxes_gain, eps=5.0, min_samples=1)

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

    # Breadth Calculation
    def breadth(start_sensor,end_sensor):
        if start_sensor == end_sensor:
            bredth = 0
            return bredth
            # bredth=bredth*0.79
            # bredth=bredth*1.08
        else:
            outer_diameter_1 = OUTER_DIA_1
            thickness_1 = THICKNESS
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
                # bredth=bredth*0.79
                # bredth=bredth*1.08
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

def boxing2(datafile, test_val, map_ori_sens,sens_mean):
    test_val_c = test_val
    test_val = test_val.T

    data_array = test_val.values.astype(np.float64)

    # Breadth Calculation
    def breadth(start_sensor,end_sensor):
        if start_sensor == end_sensor:
            bredth = 0
            return bredth
            # bredth=bredth*0.79
            # bredth=bredth*1.08
        else:
            outer_diameter_1 = OUTER_DIA_1
            thickness_1 = THICKNESS
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
                # bredth=bredth*0.79
                # bredth=bredth*1.08
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
    # Define DFS for clustering
    def dfs(matrix, x, y, visited, cluster):
        stack = [(x, y)]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            cluster.append((cx, cy))
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1] and
                        matrix[nx, ny] != 0 and (nx, ny) not in visited):
                    stack.append((nx, ny))

    def boxes_overlap(box1, box2):
        """Check if two bounding boxes overlap."""
        return not (box1['end_row'] < box2['start_row'] or
                    box1['start_row'] > box2['end_row'] or
                    box1['end_col'] < box2['start_col'] or
                    box1['start_col'] > box2['end_col'])
    # def boxes_overlap(box1, box2, tolerance=20.0):
    #     """Check if two bounding boxes overlap with a tolerance factor."""
    #     return not (
    #         box1['end_row'] + tolerance < box2['start_row'] or
    #         box1['start_row'] - tolerance > box2['end_row'] or
    #         box1['end_col'] + tolerance < box2['start_col'] or
    #         box1['start_col'] - tolerance > box2['end_col']
    #     )

    def cluster_defects(defect_list, eps=10, min_samples=2):
        """Clusters defects based on proximity using DBSCAN."""
        positions = np.array([
            [defect["Absolute Distance"], defect["Width"]] 
            for defect in defect_list
        ])
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        
        for defect, cluster_id in zip(defect_list, clustering.labels_):
            defect["Cluster ID"] = cluster_id
        
        return defect_list

    def merge_boxes(box1, box2):
        """Merge two overlapping bounding boxes into one."""
        return {
            'start_row': min(box1['start_row'], box2['start_row']),
            'end_row': max(box1['end_row'], box2['end_row']),
            'start_col': min(box1['start_col'], box2['start_col']),
            'end_col': max(box1['end_col'], box2['end_col'])
        }

    def get_bounding_boxes(data_array):
        visited = set()
        bounding_boxes = []

        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                if data_array[i, j] != 0 and (i, j) not in visited:
                    cluster = []
                    dfs(data_array, i, j, visited, cluster)
                    if cluster:  # Check if the cluster is not empty
                        min_row = min(point[0] for point in cluster)
                        max_row = max(point[0] for point in cluster)
                        min_col = min(point[1] for point in cluster)
                        max_col = max(point[1] for point in cluster)
                        bounding_boxes.append({'start_row': min_row, 'end_row': max_row,
                                               'start_col': min_col, 'end_col': max_col})
        return bounding_boxes

    # Find clusters of connected non-zero values
    bounding_boxes = get_bounding_boxes(data_array)

    # Merge overlapping boxes
    def merge_overlapping_boxes(bounding_boxes):
        merged_boxes = []
        while bounding_boxes:
            box = bounding_boxes.pop(0)
            merged = False
            for i in range(len(merged_boxes)):
                if boxes_overlap(box, merged_boxes[i]):
                    merged_boxes[i] = merge_boxes(box, merged_boxes[i])
                    merged = True
                    break
            if not merged:
                merged_boxes.append(box)
        return merged_boxes

    merged_boxes = merge_overlapping_boxes(bounding_boxes)
    
    # Create a DataFrame from the bounding boxes
    result_df = pd.DataFrame(merged_boxes)
    df_sorted = result_df.sort_values(by='start_col')
    df_clock_hall_oddo1 = datafile['ODDO1']/1000  # Extract the ODDO1 column from datafile
    oddo1_li = list(df_clock_hall_oddo1)

    # Create heatmap
    figx112 = go.Figure(data=go.Heatmap(
        z=test_val_c.T,
        y=test_val_c.columns,
        x=(datafile['ODDO1']/1000).round(2),
        hovertemplate='(%{x}, %{z})<br>Actual Ori: %{text[2]}<br>Sensor: %{text[0]}',
        text=[[item for item in map_ori_sens[col]] for col in map_ori_sens.columns],
        # zmin = -12000,
        # zmax = 30000,
        colorscale='jet',
    ))

    # Process defects and calculate values
    finial_defect_list = []

    def process_defects(df_sorted, result):
        for i, row in df_sorted.iterrows():
            start_sensor = row['start_row']
            end_sensor = row['end_row']
            start_reading = row['start_col']
            end_reading = row['end_col']
            if start_sensor == end_sensor:
                continue

            # submatrix = result.iloc[start_reading:end_reading + 1, start_sensor:end_sensor + 1]
            try:
                submatrix = result.iloc[start_reading:end_reading + 1, start_sensor:end_sensor + 1]
                if submatrix.empty or (submatrix.values == 0).all():
                    # print(f"Skipping empty or zero submatrix for row: {row}")
                    continue
                
                submatrix = submatrix.dropna(how='all', axis=0).dropna(how='all', axis=1)
                if submatrix.empty:
                    # print(f"Skipping submatrix with all NaN values for row: {row}")
                    continue

                max_value = submatrix.max().max()
                max_column = submatrix.max().idxmax()
                max_index = submatrix.columns.get_loc(max_column)
                sub_matrix_list = list(submatrix[max_column])
                # Filter positive values
                positive_values = [val for val in sub_matrix_list if val > 0]
                if not positive_values:
                    # print(f"No positive values in submatrix for row: {row}")
                    continue
                max_val = max(positive_values)
                min_val = min(positive_values)
                # Check indices for validity
                if start_reading >= len(oddo1_li) or end_reading >= len(oddo1_li):
                    # print(f"Invalid reading indices for row: {row}")
                    continue
                base_value = sens_mean[max_index]
                absolute_distance = oddo1_li[start_reading]
                length = (oddo1_li[end_reading] - oddo1_li[start_reading]) * 1000
                width = breadth(start_sensor, end_sensor)
                absolute_distance_start = oddo1_li[start_reading]
                absolute_distance_end = oddo1_li[end_reading]
            
                if width <= 0:
                    # print(f"Invalid width for row: {row}")
                    continue
                orientation = map_ori_sens.iloc[start_reading, end_sensor][2]

                depth_val = (length / width) * (max_value / base_value)
                depth_val = (depth_val * 100) / 7.5
                
            except Exception as e:
                # print(f"Error processing defect: {e}")
                continue

            if depth_val>DEPTH_VAL_LIMIT and width>WIDTH_LIMIT:
                runid = 1
                finial_defect_list.append({
                    "Box Number": i,
                    "Distance to U/S GW(m)": absolute_distance_start,
                    "x1": absolute_distance_end,
                    "y0": start_sensor,
                    "y1": end_sensor,
                    "Absolute Distance": absolute_distance,
                    "Width": length,
                    "Breadth": width,
                    "Orientation o' clock": orientation,
                    "Depth % ": depth_val,
                    "Min_Val": min_val,
                    "Max_Val": max_val
                })
                # figx112.add_shape(
                #     type='rect',
                #     # x0=start_reading,  # Map to 'ODDO1'
                #     # x1=end_reading,    # Map to 'ODDO1'
                #     x0=absolute_distance_start,
                #     x1=absolute_distance_end,
                #     y0=start_sensor,
                #     y1=end_sensor,
                #     line=dict(color='red', width=1),
                #     fillcolor='rgba(255, 0, 0, 0.2)'
                # )
                # figx112.add_annotation(
                #     x=start_reading,  
                #     y=end_sensor,  
                #     text=f'{i}',
                #     showarrow=False,
                #     font=dict(size=12, color='red'),
                #     align='center',
                #     xref='x',
                #     yref='y'
                # )
                # print(finial_defect_list)

    process_defects(df_sorted, test_val.T)

    figx112.update_layout(
        xaxis_title='Absolute Distance (m)',
        height=500,
        width=1500,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return finial_defect_list, figx112