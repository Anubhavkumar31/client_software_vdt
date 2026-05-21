'''
This code will input the Pipe Tally( respective pipe numbers, x) and generate out the csv to:
Pipe_{x}  with ending format as <PipeTally{x}.csv>
'''
import numpy as np
import pandas as pd
from pathlib import Path

# p_df = pd.read_excel('backend/files/datalog/PipeT_test.xlsx')

# def create_pipe_tally(p_df, output_folder='Client_Pipes',output_callback=None):
#     Path(output_folder).mkdir(exist_ok=True)
#
#     # Group by 'Pipe Number'
#     grouped = p_df.groupby('Pipe Number')
#
#     for pipe_number, group in grouped:
#         pipe_number = int(float(pipe_number))
#         folder_path = Path(output_folder) / f'Pipe_{pipe_number}'
#         folder_path.mkdir(exist_ok=True)
#
#         # Path for the Ptally CSV file
#         csv_file_path = folder_path / f'PipeTally{pipe_number}.csv'
#         group.to_csv(csv_file_path, index=False)
#
#         message = f"Processed Pipe: {pipe_number} and saved to {folder_path}"
#         if output_callback:
#             output_callback(message)
#         else:
#             print(message)


def create_pipe_tally(p_df, output_folder='Client_Pipes', pkl_folder='pkl_files', output_callback=None):
    Path(output_folder).mkdir(exist_ok=True)

    # get pipe numbers that have a pkl file
    pkl_pipes = {int(p.stem) for p in Path(pkl_folder).glob('*.pkl')}

    if not pkl_pipes:
        print(f"[pipe_tally] No pkl files found in {pkl_folder}")
        return

    print(f"[pipe_tally] Found pkl for pipes: {sorted(pkl_pipes)}")

    grouped = p_df.groupby('Pipe Number')

    for pipe_number, group in grouped:
        pipe_number = int(float(pipe_number))

        if pipe_number not in pkl_pipes:
            print(f"[pipe_tally] Pipe {pipe_number} — no pkl found, skipping")
            continue

        folder_path = Path(output_folder) / f'Pipe_{pipe_number}'
        folder_path.mkdir(exist_ok=True)

        csv_file_path = folder_path / f'PipeTally{pipe_number}.csv'
        group.to_csv(csv_file_path, index=False)

        message = f"[pipe_tally] Pipe {pipe_number} — saved to {folder_path}"
        if output_callback:
            output_callback(message)
        else:
            print(message)

# create_pipe_tally(p_df)
