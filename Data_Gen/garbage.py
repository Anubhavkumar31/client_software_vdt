# from test_heatmap import run_heatmap_plotly

# res = run_heatmap_plotly(
#     pkl_path=r"F:\work_new\client_software\test_data_cs\35.pkl",
#     roll_col="ROLL",
#     oddo_col="ODDO1",
#     initial_read=0.0,
#     upper_sens_mul=1,
#     lower_sens_mul=3,
#     pipetally_path=r"F:\work_new\client_software\test_data_cs\picle_2\Pipe_35\PipeTally35.csv",
#     output_html=r"F:\work_new\client_software\test_data_cs\picle_2\Pipe_35",  # folder or file path
#     plot_mode="raw"  # or "processed"
# )

# print("HTML:", res["html_path"])


#!/usr/bin/env python3
"""
Convert .pkl file (pandas DataFrame) to .csv and save it.

Usage:
  python pkl_to_csv.py
"""

import pandas as pd
from pathlib import Path

def pkl_to_csv(pkl_path: str, output_csv: str = None):
    """
    Convert a pickle file to CSV.

    Parameters
    ----------
    pkl_path : str
        Path to input .pkl file (should contain a pandas DataFrame).
    output_csv : str, optional
        Path to save output CSV. If None, saves alongside the .pkl with .csv extension.
    """
    pkl_file = Path(pkl_path)
    if not pkl_file.exists():
        raise FileNotFoundError(f"File not found: {pkl_file}")

    # Load the pickle
    df = pd.read_pickle(pkl_file)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Determine CSV path
    if output_csv is None:
        output_csv = pkl_file.with_suffix(".csv")

    # Save as CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv.resolve()}")

    return output_csv


if __name__ == "__main__":
    # Example: change this path to your .pkl file
    PKL_PATH = r"F:\work_new\client_software\test_data_cs\pickle_test\2.pkl"

    # Call converter
    pkl_to_csv(PKL_PATH)
