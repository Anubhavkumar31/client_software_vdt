import re
from datetime import timedelta

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

num_of_sensors = 36
minute = 720 / num_of_sensors
degree = minute / 2
pkl_path = r"D:\Anubhav\softwares\client software\Data\project_oil_sample\pickle_data\150.pkl"


def show_heatmap_tab8(pkl_path):

    print("\n==============================")
    print("STEP 1: Loading PKL file")
    print("==============================")

    df = pd.read_pickle(pkl_path)

    print(f"PKL loaded successfully")
    print(f"DataFrame Shape: {df.shape}")

    # =========================
    # FIND PROXIMITY COLUMNS
    # =========================

    print("\n==============================")
    print("STEP 2: Finding proximity columns")
    print("==============================")

    prox_pattern = re.compile(r"^F\d+P\d+$")

    proximity_cols = [
        col for col in df.columns
        if prox_pattern.match(str(col))
    ]

    print(f"Total proximity columns found: {len(proximity_cols)}")
    print(proximity_cols)

    # =========================
    # KEEP REQUIRED COLUMNS
    # =========================

    print("\n==============================")
    print("STEP 3: Preparing results dataframe")
    print("==============================")

    required_cols = ["index", "ROLL", "ODDO1", "ODDO2"] + proximity_cols
    required_cols = [c for c in required_cols if c in df.columns]

    results = df[required_cols]

    print(f"Results Shape: {results.shape}")

    # =========================
    # EXISTING LOGIC
    # =========================

    print("\n==============================")
    print("STEP 4: Extracting row data")
    print("==============================")

    data = []
    index_tab8 = []
    oddo_1 = []
    oddo_2 = []
    roll1 = []

    total_rows = len(results)

    for idx, row in enumerate(results.itertuples(index=False, name=None)):

        if idx % 1000 == 0:
            print(f"Processing row {idx}/{total_rows}")

        index_tab8.append(row[0])
        roll1.append(row[1])
        oddo_1.append(row[2])
        oddo_2.append(row[3])

        # Store all proximity sensor values
        data.append(list(row[4:]))

    print("Row extraction completed")

    """
    Swapping the Pitch data to Roll data
    """

    print("\n==============================")
    print("STEP 5: Calculating adjusted values")
    print("==============================")

    oddo1_tab8 = []
    oddo2_tab8 = []
    roll_t = []

    oddo1 = 733.402
    oddo2 = 0
    roll_value = -14.64

    for odometer1 in oddo_1:
        od1 = odometer1 - oddo1
        oddo1_tab8.append(od1)

    for odometer2 in oddo_2:
        od2 = odometer2 - oddo2
        oddo2_tab8.append(od2)

    for roll2 in roll1:
        roll3 = roll2 - roll_value
        roll_t.append(roll3)

    print("Adjusted values calculated")

    # =========================
    # CREATE FINAL DATAFRAME
    # =========================

    print("\n==============================")
    print("STEP 6: Creating proximity dataframe")
    print("==============================")

    df_new_tab8 = pd.DataFrame(
        data,
        columns=[f'proximity{i}' for i in range(1, len(proximity_cols) + 1)]
    )

    print(f"Proximity DataFrame Shape: {df_new_tab8.shape}")

    df_new_tab8 = df_new_tab8.apply(pd.to_numeric, errors='coerce')

    print("Numeric conversion completed")

    # =========================
    # ROLL CALCULATION
    # =========================

    print("\n==============================")
    print("STEP 7: Running Roll_Calculation")
    print("==============================")

    map_ori_sens_ind, val_ori_sensVal = Roll_Calculation(df_new_tab8, roll_t)

    print("Roll_Calculation completed")

    # =========================
    # FINAL MERGE
    # =========================

    print("\n==============================")
    print("STEP 8: Creating final dataframe")
    print("==============================")

    df_elem = pd.DataFrame({
        "index": index_tab8,
        "ODDO1": oddo1_tab8
    })

    frames = [df_elem, val_ori_sensVal]

    df_new = pd.concat(frames, axis=1, join='inner')

    print(f"Final DataFrame Shape: {df_new.shape}")

    # =========================
    # SAVE OUTPUT
    # =========================

    output_path = r"D:\Anubhav\softwares\client software\client_software_vdt\test.pkl"

    print("\n==============================")
    print("STEP 9: Saving output")
    print("==============================")

    df_new.to_pickle(output_path)

    print(f"Output saved successfully")
    print(f"Saved to: {output_path}")

    print("\n==============================")
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("==============================")

    plot_clock_heatmap_t8(df_new)


def Roll_Calculation(df_hall, roll):

    print("\n[Roll_Calculation] Started")

    # =========================
    # NORMALIZATION
    # =========================
    print("-> Normalizing hall sensor values")

    mean1 = df_hall.mean()

    # df_hall = ((df_hall - mean1) / mean1) * 100
    for i, data in enumerate(df_hall.columns):

        df_hall[data] = (
            (df_hall[data] - mean1.iloc[i]) / mean1.iloc[i]
        ) * 100

    print("-> Normalization completed")

    # =========================
    # CREATE ROLL DICTIONARY
    # =========================
    print("-> Building roll dictionary")

    first_key_values = roll

    roll_dictionary = {'1': first_key_values}

    angle = [round(i * degree, 1) for i in range(0, num_of_sensors)]

    for i in range(2, num_of_sensors + 1):

        current_values = [
            round((value + angle[i - 1]), 2)
            for value in first_key_values
        ]

        roll_dictionary[str(i)] = current_values

    print(f"-> Roll dictionary created with {len(roll_dictionary)} sensors")

    # =========================
    # CLOCK CONVERSION
    # =========================
    print("-> Converting roll values to clock format")

    clock_dictionary = {}

    for key in roll_dictionary:

        clock_dictionary[key] = [
            degrees_to_hours_minutes(value)
            for value in roll_dictionary[key]
        ]

    Roll_hr = pd.DataFrame(clock_dictionary)

    print(f"-> Roll_hr shape: {Roll_hr.shape}")

    # =========================
    # PREPARE DATA
    # =========================
    print("-> Preparing intermediate structures")

    df_hall.reset_index(inplace=True, drop=True)

    k = (df_hall.transpose()).astype(float)
    k.reset_index(inplace=True, drop=True)

    time_list = [timedelta(minutes=i * int(minute)) for i in range(num_of_sensors)]

    clock_list = [
        (time_list[i], time_list[i + 1])
        for i in range(len(time_list) - 1)
    ]

    clock_list.append((time_list[-1], timedelta(days=1)))

    clock_df = Roll_hr.map(
        lambda x: timedelta(
            hours=int(x.split(':')[0]),
            minutes=int(x.split(':')[1])
        ) if isinstance(x, str) else x
    )

    print(f"-> clock_df shape: {clock_df.shape}")

    # =========================
    # CREATE CLOCK DATAFRAME
    # =========================
    print("-> Creating clock dataframe")

    df_clock = pd.DataFrame(
        0.0,
        index=range(clock_df.shape[0]),
        columns=[str(t) for t in time_list],
        dtype=float
    )

    df_clock.columns = [
        f"{int(t.seconds // 3600):02}:{int((t.seconds % 3600) // 60):02}"
        for t in time_list
    ]

    # =========================
    # MAIN LOOP
    # =========================
    print("-> Mapping sensor values to clock bins")

    total_rows = clock_df.shape[0]

    for i in range(total_rows):

        if i % 2000 == 0:
            print(f"   Processed {i}/{total_rows} rows")

        # Iterate over each column
        for j in range(clock_df.shape[1]):

            value = clock_df.iloc[i, j]

            # Check if the value is in any range
            for r in clock_list:

                if r[0] <= value < r[1]:

                    col_name = (
                        f"{int(r[0].seconds // 3600):02}:"
                        f"{int((r[0].seconds % 3600) // 60):02}"
                    )

                    df_clock.loc[i, col_name] = float(df_hall.iloc[i, j])

                    break

    print("-> Clock bin mapping completed")

    # =========================
    # FINAL OUTPUT
    # =========================
    print("-> Creating transpose dataframe")

    df_clock_tranpose = df_clock.T

    print(f"-> Final transpose shape: {df_clock_tranpose.shape}")

    print("[Roll_Calculation] Completed Successfully\n")

    return df_clock_tranpose, df_clock


def degrees_to_hours_minutes(degrees):
    if (degrees < 0):
        degrees = degrees % 360
    elif degrees >= 360:
        degrees %= 360
    degrees_per_second = 360 / (12 * 60 * 60)
    total_seconds = degrees / degrees_per_second
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}:{minutes:02d}"

def plot_clock_heatmap_t8(df_new, pipe_number=None):

    oddo_val = [round(elem / 1000, 2) for elem in df_new['ODDO1'].tolist()]
    index_hm = list(df_new['index'])

    df_clk = df_new[[f"{h:02}:{m:02}" for h in range(12) for m in range(0, 60, int(minute))]]
    df_clk = df_clk.apply(pd.to_numeric, errors='coerce')

    d1 = df_clk.transpose().astype(float)

    fig, ax2 = plt.subplots(figsize=(20, 8))
    fig.subplots_adjust(bottom=0.213, left=0.077, top=0.855, right=1.000)

    heat_map_obj = sns.heatmap(d1, cmap='jet', ax=ax2, vmin=-0.01, vmax=0.07)
    heat_map_obj.set(xlabel="Index", ylabel="Clock")

    ax2.set_xticklabels(ax2.get_xticklabels(), size=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), size=9)

    ax3 = ax2.twiny()
    num_ticks1 = len(ax2.get_xticks())
    tick_positions1 = [int(i) for i in np.linspace(0, len(oddo_val) - 1, num_ticks1)]
    ax3.set_xticks(tick_positions1)
    ax3.set_xticklabels([f'{oddo_val[i]:.2f}' for i in tick_positions1], rotation=90, size=9)
    ax3.set_xlabel("Absolute Distance (m)", size=9)

    plt.show()

show_heatmap_tab8(pkl_path)



