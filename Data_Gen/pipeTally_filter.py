'''
This code will input the Pipe Tally( respective pipe numbers, x) and generate out the csv to:
Pipe_{x}  with ending format as <PipeTally{x}.csv>
'''
import numpy as np
import pandas as pd
from pathlib import Path

# p_df = pd.read_excel('backend/files/datalog/PipeT_test.xlsx')

def create_pipe_tally(p_df, output_folder='Client_Pipes',output_callback=None):
    Path(output_folder).mkdir(exist_ok=True)
    
    # Group by 'Pipe Number'
    grouped = p_df.groupby('Pipe Number')
    
    for pipe_number, group in grouped:
        folder_path = Path(output_folder) / f'Pipe_{pipe_number}'
        folder_path.mkdir(exist_ok=True)
        
        # Path for the Ptally CSV file
        csv_file_path = folder_path / f'PipeTally{pipe_number}.csv'
        group.to_csv(csv_file_path, index=False)

        message = f"Processed Pipe: {pipe_number} and saved to {folder_path}"
        if output_callback:
            output_callback(message)
        else:
            print(message)

# create_pipe_tally(p_df)
