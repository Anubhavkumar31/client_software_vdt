def plot_sensor_heatmap(self):
    Config.print_with_time("Pre graph analysis called")
    runid = self.runid
    pipe_id = self.combo_box.currentText()
    self.Weld_id_shm = int(pipe_id)
    # lower_sensitivity = self.lower_Sensitivity_combo_box.text()
    # upper_sensitivity = self.upper_Sensitivity_combo_box.text()
    # print(type(lower_sensitivity),type(upper_sensitivity))
    with connection.cursor() as cursor:
        # query = "SELECT start_index,end_index,length FROM pipes where runid=" + str(runid) + " and id=" + str(pipe_id)
        # cursor.execute(query)
        # result = cursor.fetchone()
        # print(result)
        query = "SELECT start_index, end_index,start_oddo1,end_oddo1 FROM welds WHERE runid=%s AND id IN (%s, (SELECT MAX(id) FROM welds WHERE runid=%s AND id < %s)) ORDER BY id"
        cursor.execute(query, (self.runid, self.Weld_id_shm, self.runid, self.Weld_id_shm))
        result = cursor.fetchall()
        start_oddo1 = result[0][2]
        end_oddo1 = result[1][3]
        self.pipe_len_8 = round(end_oddo1 - start_oddo1)
        print("Weld_pipe_Length", self.pipe_len_8)

        if not result:
            Config.print_with_time("No data found for this pipe ID : ")
        else:
            """
            pkl file is found in local path 
            """
            path = Config.sensor_heatmap + self.project_name + '/' + str(self.Weld_id_shm) + '.pkl'
            if os.path.isfile(path):
                Config.print_with_time("File exist")
                df_new = pd.read_pickle(path)

                val_ori_sensVal = df_new[[f'proximity{m}' for m in range(1, 25)]]
                map_ori_sens_ind = df_new[[f'proximity{m}_x' for m in range(1, 25)]]
                map_ori_sens_ind.columns = map_ori_sens_ind.columns.str.rstrip('_x')
                mean_clock_data = val_ori_sensVal.mean().tolist()
                # print("hello1")

                self.figure1.clear()
                ax1 = self.figure1.add_subplot(111)
                ax1.figure.subplots_adjust(bottom=0.213, left=0.077, top=0.855, right=1.000)
                # print("hello3")

                df3 = ((val_ori_sensVal - mean_clock_data)/mean_clock_data) * 100
                d1 = df3.transpose().astype(float)

                heat_map_obj = sns.heatmap(d1, cmap='jet', ax=ax1, vmin=-0.015, vmax=0.06)
                # heat_map_obj = sns.heatmap(d1, cmap='jet', ax=ax1)
                # print("hello4")
                """
                Pipewise ranges have been set
                """
                oddo1_li_chm = list(df_new['ODDO1'])
                index_hm = list(df_new['index'])
                # print("index_chm", self.index_chm)

                # print("hello5")
                ax1.set_xticklabels(ax1.get_xticklabels(), size=9)
                ax1.set_yticklabels(ax1.get_yticklabels(), size=9)
                ax3 = ax1.twiny()
                oddo_val = [round(elem / 1000, 2) for elem in oddo1_li_chm]
                num_ticks1 = len(ax1.get_xticks())  # Adjust the number of ticks based on your preference
                # print(num_ticks1)
                tick_positions1 = [int(i) for i in np.linspace(0, len(oddo_val) - 1, num_ticks1)]
                # print(tick_positions1)
                ax3.set_xticks(tick_positions1)
                ax3.set_xticklabels([f'{oddo_val[i]:.2f}' for i in tick_positions1], rotation=90, size=9)
                ax3.set_xlabel("Absolute Distance (m)", size=9)
                def on_hover(event):
                    # print("hello6")
                    if event.xdata is not None and event.ydata is not None:
                        try:
                            x = int(event.xdata)
                            y = int(event.ydata)
                            index_value = index_hm[x]
                            clock_val = list(val_ori_sensVal.columns)[y]           ### It shows clock_column values ###
                            value = d1.iloc[y, x]   ### It shows real time values, like clock values at particular point ###
                            value1 = map_ori_sens_ind.transpose().iloc[y, x]
                            z = oddo1_li_chm[x]
                            self.canvas1.toolbar.set_message(f'Index={index_value:.0f},Abs.distance(m)={z/1000:.3f},Clock={clock_val},Value={value:.1f}')
                        except (IndexError, ValueError):
                            # Print a user-friendly message instead of showing an error
                            pass
                            print("Hovering outside valid data range. No data available.")
                self.figure1.canvas.mpl_connect('motion_notify_event', on_hover)
                print("hello7")
                heat_map_obj.set(xlabel="Index", ylabel="Sensors")
                self.canvas1.draw()
                print("Plotted heatmap........")

                # self.plot_sensor_heatmap_t7(df_new)
            else:
                """
                pkl file is not found than data fetch from GCP and save pkl file in local path
                """
                folder_path = Config.sensor_heatmap + self.project_name
                folder_path1 = Config.weld_pipe_pkl + self.project_name
                # print(folder_path)
                Config.print_with_time("File not exist")
                for path in [folder_path, folder_path1]:
                    try:
                        os.makedirs(path)
                        Config.print_with_time(f"Created folder: {path}")
                    except FileExistsError:
                        Config.print_with_time(f"Folder already exists: {path}")
                    except Exception as e:
                        Config.print_with_time(f"Error creating folder {path}: {e}")
                start_index, end_index = result[0][0], result[1][1]
                print(self.Weld_id_shm)
                print("start index and end index", start_index, end_index)
                Config.print_with_time("Start fetching at : ")
                query_for_start = 'SELECT index,ROLL,ODDO1,ODDO2,[proximity1, proximity2, proximity3, proximity4, proximity5, proximity6, proximity7, proximity8,proximity9, proximity10, proximity11, proximity12, proximity13, proximity14, proximity15,proximity16, proximity17, proximity18, proximity19, proximity20, proximity21, proximity22, proximity23, proximity24],PITCH,YAW FROM ' + Config.table_name + ' WHERE index>{} AND index<{} order by index'
                query_job = client.query(query_for_start.format(start_index, end_index))
                results = query_job.result()

                # query = query_generator.get_pipe(lower_sensitivity, upper_sensitivity, runid, start_index, end_index,
                #                                  self.Weld_id_shm)
                # query_job = client.query(query)
                # results = query_job.result()

                Config.print_with_time("End fetching  at : ")
                data = []
                self.index_tab7 = []
                oddo_1 = []
                oddo_2 = []
                roll1 = []
                pitch1 = []
                yaw1 = []

                Config.print_with_time("Start of conversion at : ")
                for row in results:
                    self.index_tab7.append(row[0])
                    roll1.append(row[1])
                    oddo_1.append(row[2])
                    oddo_2.append(row[3])
                    data.append(row[4])
                    pitch1.append(row[5])
                    yaw1.append(row[6])
                    """
                    Swapping the Pitch data to Roll data
                    """

                    # indexes.append(ranges(index_of_occurrences(row['frames'], 1)))

                self.oddo1_tab7 = []
                self.oddo2_tab7 = []
                self.roll_t = []
                self.pitch_t = []
                self.yaw_t = []

                """
                Reference value will be consider 
                """
                for odometer1 in oddo_1:
                    od1 = odometer1 - Config.oddo1  ###16984.2 change According to run
                    self.oddo1_tab7.append(od1)
                for odometer2 in oddo_2:
                    od2 = odometer2 - Config.oddo2  ###17690.36 change According to run
                    self.oddo2_tab7.append(od2)

                """
                Reference value will be consider
                """
                for i in roll1:
                    roll3 = i - Config.roll_value
                    self.roll_t.append(roll3)
                for j in pitch1:
                    pitch3 = j - Config.pitch_value
                    self.pitch_t.append(pitch3)
                for k in yaw1:
                    yaw3 = k - Config.yaw_value
                    self.yaw_t.append(yaw3)

                self.df_new7 = pd.DataFrame(data, columns=[f'proximity{i}' for i in range(1, 25)])

                df_elem = pd.DataFrame({"index": self.index_tab7, "ODDO1": self.oddo1_tab7, "ROLL": self.roll_t, "PITCH": self.pitch_t, "YAW": self.yaw_t})
                frames = [df_elem, self.df_new7]
                df_pipe = pd.concat(frames, axis=1, join='inner')
                # df_new.reset_index(inplace=True)
                # print("Plotted data", df_pipe)
                df_pipe.to_pickle(folder_path1 + '/' + str(self.Weld_id_shm) + '.pkl')
                Config.print_with_time("Succesfully saved to sensor pickle file")

                # df_processed = df_new_tab9.copy()
                sensor_columns = [f'proximity{i}' for i in range(1, 25)]
                df1_raw = self.df_new7[[f'proximity{i}' for i in range(1, 25)]]

                roll_dictionary = {'1': self.roll_t}
                angle = [round(i*15, 1) for i in range(0, 24)]
                # print(len(angle))

                for i in range(2, 25):
                    current_values = [round((value + angle[i - 1]), 2) for value in self.roll_t]
                    roll_dictionary['{}'.format(i)] = current_values

                clock_dictionary = {}
                for key in roll_dictionary:
                    clock_dictionary[key] = [self.degrees_to_hours_minutes(value) for value in roll_dictionary[key]]

                Roll_hr = pd.DataFrame(clock_dictionary)
                Roll_hr.columns = [f'proximity{m}' for m in range(1, 25)]

                column_means = self.df_new7.mean()
                # print("column_means", column_means)
                sensor_mean = [i_x for i_x in column_means]
                standard_deviation = self.df_new7.std(axis=0, skipna=True).tolist()

                """
                To Calculate upper thersold Value
                """
                mean_plus_1sigma = []
                for i, data1 in enumerate(sensor_mean):
                    sigma1 = data1 + (Config.positive_sigma_col) * standard_deviation[i]
                    mean_plus_1sigma.append(sigma1)
                # print("sigma1_positive",mean_plus_1sigma)

                """
                To Calculate lower thersold value
                """
                mean_negative_3sigma = []
                for i_2, data_3 in enumerate(sensor_mean):
                    sigma_3 = data_3 - (Config.negative_sigma) * standard_deviation[i_2]
                    mean_negative_3sigma.append(sigma_3)
                # print("sigma3_negative",mean_negative_3sigma)
                """
                Values above the upper threshold are considered as 1,
                values below the lower threshold are considere
                d as 1,
                and values between the upper and lower thresholds are considered as 0.
                """
                for col, data in enumerate(self.df_new7.columns):
                    self.df_new7[data] = self.df_new7[data].apply(
                        lambda x: 1 if x > mean_plus_1sigma[col] else 0)

                filtered_df1 = self.df_new7
                df1_raw.columns = filtered_df1.columns
                df1_aligned = filtered_df1.reindex(df1_raw.index)
                result_new = df1_aligned * df1_raw
                result_new = result_new.dropna()
                # print("result",result)
                result_new.reset_index(drop=True, inplace=True)

                result_raw_df = result_new.mask(result_new == 0, df1_raw)
                result_raw_df = result_raw_df.dropna()
                # print("result_raw_df",result_raw_df)
                result_raw_df.reset_index(drop=True, inplace=True)

                frames2 = [df_elem, result_raw_df]
                df_new = pd.concat(frames2, axis=1, join='inner')

                for col in Roll_hr.columns:
                    df_new[col + '_x'] = Roll_hr[col]

                df_new.to_pickle(folder_path + '/' + str(self.Weld_id_shm) + '.pkl')
                Config.print_with_time("Succesfully saved to sensor_heatmap pickle file")

                result_new_transpose = result_new.transpose()
                # print("result_new_transpose", result_new_transpose)
                data_array = result_new_transpose.values.astype(np.float64)

                def dfs(matrix, x, y, visited, cluster):
                    """Perform DFS to find clusters, but only include positive values."""
                    stack = [(x, y)]
                    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                    while stack:
                        cx, cy = stack.pop()
                        if (cx, cy) in visited:  # Ignore negative values
                            continue
                        if matrix[cx, cy] <= 0:  # Ignore negative values
                            continue
                        visited.add((cx, cy))
                        cluster.append((cx, cy))
                        for dx, dy in directions:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1] and
                                    matrix[nx, ny] > 0 and (nx, ny) not in visited):  # Only traverse positive values
                                stack.append((nx, ny))

                def do_boxes_overlap(box1, box2):
                    """Check if two bounding boxes overlap."""
                    return not (box1['end_row'] < box2['start_row'] or
                                box1['start_row'] > box2['end_row'] or
                                box1['end_col'] < box2['start_col'] or
                                box1['start_col'] > box2['end_col'])

                # Find clusters of connected non-zero values and calculate bounding boxes
                def merge_boxes(box1, box2):
                    """Merge two overlapping bounding boxes into one."""
                    return {
                        'start_row': min(box1['start_row'], box2['start_row']),
                        'end_row': max(box1['end_row'], box2['end_row']),
                        'start_col': min(box1['start_col'], box2['start_col']),
                        'end_col': max(box1['end_col'], box2['end_col'])
                    }

                visited = set()
                bounding_boxes = []
                for i in range(data_array.shape[0]):
                    for j in range(data_array.shape[1]):
                        if data_array[i, j] > 0 and (i, j) not in visited:
                            cluster = []
                            dfs(data_array, i, j, visited, cluster)
                            if cluster:  # Check if the cluster is not empty
                                min_row = min(point[0] for point in cluster)
                                max_row = max(point[0] for point in cluster)
                                min_col = min(point[1] for point in cluster)
                                max_col = max(point[1] for point in cluster)
                                bounding_boxes.append({'start_row': min_row, 'end_row': max_row,
                                                       'start_col': min_col, 'end_col': max_col})

                merged_boxes = []
                while bounding_boxes:
                    box = bounding_boxes.pop(0)
                    merged = False
                    for i in range(len(merged_boxes)):
                        if do_boxes_overlap(box, merged_boxes[i]):
                            merged_boxes[i] = merge_boxes(box, merged_boxes[i])
                            merged = True
                            break
                    if not merged:
                        merged_boxes.append(box)

                df_sorted = pd.DataFrame(merged_boxes).sort_values(by='start_col')
                oddo1_li_chm = list(df_new['ODDO1'])

                self.figure1.clear()
                ax1 = self.figure1.add_subplot(111)
                ax1.figure.subplots_adjust(bottom=0.213, left=0.077, top=0.855, right=1.000)
                # print("hello3")

                val_ori_sensVal = df_new[[f'proximity{m}' for m in range(1, 25)]]
                mean_clock_data = val_ori_sensVal.mean().tolist()
                df3 = ((val_ori_sensVal - mean_clock_data)/mean_clock_data) * 100
                d1 = df3.transpose().astype(float)

                # d1 = ((df_new_1.set_index(df_clock_index)).T).astype(float)

                # # color_ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                # #                 (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]
                # color_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                #                             (9, 10)]
                #
                # color_values = ['#82ffff', '#00CD00', '#008000', '#fd8f00', '#D98719',
                #                 '#CD661D', '#EE4000', '#FF0000', '#820202', '#000000']
                #
                # # custom_palette = sns.color_palette(color_values)
                #
                #
                # bounds = [r[0] for r in color_ranges] + [color_ranges[-1][1]]
                #
                # cmap = ListedColormap(color_values)
                #
                # norm = BoundaryNorm(bounds, cmap.N)
                #
                # heat_map_obj = sns.heatmap(self.df_new, cmap=cmap, ax=ax1, norm=norm)

                # heat_map_obj = sns.heatmap(d1, cmap='jet', ax=ax1, vmin=-10, vmax=10)
                heat_map_obj = sns.heatmap(result_new_transpose, cmap='jet', ax=ax1)
                # print("hello4")
                """
                Pipewise ranges have been set
                """
                oddo1_li_chm = list(df_new['ODDO1'])
                index_hm = list(df_new['index'])
                # print("index_chm", self.index_chm)

                # print("hello5")
                ax1.set_xticklabels(ax1.get_xticklabels(), size=9)
                ax1.set_yticklabels(ax1.get_yticklabels(), size=9)
                ax3 = ax1.twiny()
                oddo_val = [round(elem / 1000, 2) for elem in oddo1_li_chm]
                num_ticks1 = len(ax1.get_xticks())  # Adjust the number of ticks based on your preference
                # print(num_ticks1)
                tick_positions1 = [int(i) for i in np.linspace(0, len(oddo_val) - 1, num_ticks1)]
                # print(tick_positions1)
                ax3.set_xticks(tick_positions1)
                ax3.set_xticklabels([f'{oddo_val[i]:.2f}' for i in tick_positions1], rotation=90, size=9)
                ax3.set_xlabel("Absolute Distance (m)", size=9)
                def on_hover(event):
                    # print("hello6")
                    if event.xdata is not None and event.ydata is not None:
                        try:
                            x = int(event.xdata)
                            y = int(event.ydata)
                            index_value = index_hm[x]
                            clock_val = list(val_ori_sensVal.columns)[y]           ### It shows clock_column values ###
                            value = d1.iloc[y, x]   ### It shows real time values, like clock values at particular point ###
                            value1 = Roll_hr.transpose().iloc[y, x]
                            z = oddo1_li_chm[x]
                            self.canvas1.toolbar.set_message(f'Index={index_value:.0f},Abs.distance(m)={z/1000:.3f},Clock={clock_val},Value={value:.1f}')
                        except (IndexError, ValueError):
                            # Print a user-friendly message instead of showing an error
                            pass
                            print("Hovering outside valid data range. No data available.")
                self.figure1.canvas.mpl_connect('motion_notify_event', on_hover)
                print("hello7")
                heat_map_obj.set(xlabel="Index", ylabel="Sensors")

                max_submatrix_list = []
                min_submatrix_list = []
                new_boxes = []
                for _, row in df_sorted.iterrows():
                    start_sensor = row['start_row']
                    end_sensor = row['end_row']
                    start_reading = row['start_col']
                    end_reading = row['end_col']
                    if start_sensor == end_sensor:
                        pass
                    else:
                        try:
                            submatrix = result_new.iloc[start_reading:end_reading + 1, start_sensor:end_sensor + 1]
                            submatrix = submatrix.apply(pd.to_numeric, errors='coerce')  # Ensure numeric data
                            if submatrix.isnull().values.any():
                                print("Submatrix contains NaN values, skipping this iteration.")
                                continue
                            max_value = submatrix.max().max()
                            max_submatrix_list.append(max_value)
                            two_d_list = submatrix.values.tolist()
                            min_positive = min(x for row in two_d_list for x in row if x > 0)
                            min_submatrix_list.append(min_positive)
                        except Exception as e:
                            print(f"Error found 1: {str(e)}")
                            traceback.print_exc()
                            pass

                max_of_all = max(max_submatrix_list)  # Get the max of all submatrix max_values
                min_of_all = min(min_submatrix_list)
                threshold_value = round(min_of_all + (max_of_all - min_of_all) * Config.defectBox_threshold)

                finial_defect_list = []
                for _, row in df_sorted.iterrows():
                    start_sensor = row['start_row']
                    end_sensor = row['end_row']
                    start_reading = row['start_col']
                    end_reading = row['end_col']

                    # start_sensor = row['start_col']
                    # end_sensor = row['end_col']
                    # start_reading = row['start_row']
                    # end_reading = row['end_row']
                    if start_sensor == end_sensor:
                        pass
                    else:
                        try:
                            submatrix = df1_raw.iloc[start_reading:end_reading + 1, start_sensor:end_sensor + 1]
                            submatrix = submatrix.apply(pd.to_numeric, errors='coerce')  # Ensure numeric data
                            two_d_list = submatrix.values.tolist()
                            max_value = submatrix.max().max()
                            min_positive = min(x for row in two_d_list for x in row if x > 0)

                            if (threshold_value <= max_value <= max_of_all):
                                print("max_value", max_value)
                                print("min_positive", min_positive)
                                print("Max of all submatrices:", max_of_all)
                                print("Threshold Value:", threshold_value)
                                print(".....................................................")

                                depth_old = (max_value-min_positive)/min_positive*100
                                print("depth_old", depth_old)

                                max_column = submatrix.max().idxmax()
                                max_index = submatrix.columns.get_loc(max_column)
                                print("max_index", max_index)

                                base_value = sensor_mean[max_index]
                                print("base_value", base_value)

                                lenth = (oddo1_li_chm[end_reading] - oddo1_li_chm[start_reading])
                                print("length of defect", lenth)

                                counter_difference = end_reading - start_reading
                                # print("counter_difference", counter_difference)
                                divid = int(counter_difference/2)
                                center = start_reading + divid
                                factor1 = divid * Config.l_per_1
                                start1 = int(center - factor1)
                                end1 = int(center + factor1)
                                length = (oddo1_li_chm[end1] - oddo1_li_chm[start1])

                                absolute_distance = (oddo1_li_chm[start_reading])
                                print("absolute_distance", absolute_distance)

                                upstream_oddo1 = (oddo1_li_chm[start_reading] - oddo1_li_chm[0])
                                print("upstream1", upstream_oddo1)
                                width = Width_calculation(start_sensor, end_sensor)
                                print("width_new", width)

                                avg_counter = round((start_reading+end_reading)/2)
                                avg_sensor = round((start_sensor+end_sensor)/2)
                                orientation = Roll_hr.iloc[avg_counter, avg_sensor]

                                try:
                                    ################# each pipe thickness can be change #################
                                    depth_new = round((((length / width) * (max_value / base_value))*100)/Config.pipe_thickness)
                                    print("depth_new", depth_new)
                                except:
                                    depth_new = 0

                                finial_defect_list.append({"start_index": start_reading, "end_index": end_reading,
                                               "start_sensor": start_sensor, "end_sensor": end_sensor,
                                               "Absolute_distance": absolute_distance,
                                               "Upstream": upstream_oddo1,
                                               "Pipe_length": self.pipe_len_8,
                                               "Feature_type": 'Dent',
                                               "Orientation": orientation, "WT": Config.pipe_thickness,
                                               "length": length,
                                               "Width": width,
                                               "Depth_percentage": depth_old,
                                               "Depth_new_per": depth_new})
                            else:
                                pass
                        except Exception as e:
                            print(f"Error found: {str(e)}")
                            traceback.print_exc()
                            pass

                with connection.cursor() as cursor:
                    for i in finial_defect_list:
                        start_index = i['start_index']
                        end_index = i['end_index']
                        start_sensor = i['start_sensor']
                        end_sensor = i['end_sensor']
                        Absolute_distance = round(i['Absolute_distance']/1000, 3)
                        Upstream = round(i['Upstream']/1000, 3)
                        Pipe_length = round(i['Pipe_length']/1000, 3)
                        Feature_type = i['Feature_type']
                        Orientation = i['Orientation']
                        WT = i['WT']
                        length = round(i['length'], 0)
                        Width = i['Width']
                        Depth_percentage = round(i['Depth_percentage'], 1)
                        Depth_new_per = round(i['Depth_new_per'], 1)

                        """
                        Insert data into database
                        """
                        with connection.cursor() as cursor:
                            query_defect_insert = "INSERT INTO dent_table (runid,pipe_id,start_index,end_index,start_sensor,end_sensor,Absolute_distance,Upstream,Pipe_length,Feature_type,Orientation,WT,length,Width,depth_old,depth_new) VALUE(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "

                            cursor.execute(query_defect_insert, (
                                int(runid), int(self.Weld_id_shm), start_index, end_index, start_sensor, end_sensor,
                                Absolute_distance, Upstream, Pipe_length, Feature_type,
                                Orientation, WT, length, Width, Depth_percentage, Depth_new_per))
                            connection.commit()
                    print("inserted data into db____________")

                print("generating_heatmap................")
                # self.plot_sensor_heatmap_t7(df_new)

        with connection.cursor() as cursor:
            Fetch_weld_detail = "select id,pipe_id,WT,Absolute_distance,Upstream,Feature_type,Orientation,length,Width,depth_old from dent_table where runid='%s' and pipe_id='%s'"
            # Execute query.
            cursor.execute(Fetch_weld_detail, (int(self.runid), int(self.Weld_id_shm)))
            self.myTableWidget_hm.setRowCount(0)
            allSQLRows = cursor.fetchall()
            if allSQLRows:
                for row_number, row_data in enumerate(allSQLRows):
                    self.myTableWidget_hm.insertRow(row_number)
                    for column_num, data in enumerate(row_data):
                        self.myTableWidget_hm.setItem(row_number, column_num, QtWidgets.QTableWidgetItem(str(data)))
                self.myTableWidget_hm.setEditTriggers(QAbstractItemView.NoEditTriggers)
                self.myTableWidget_hm.doubleClicked.connect(self.handle_table_double_click_hm)
            else:
                # self.myTableWidget5.doubleClicked.disconnect(self.handle_table_double_click)
                Config.warning_msg("No record found", "")
        self.canvas1.draw()