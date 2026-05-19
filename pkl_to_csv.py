import pandas as pd

# Path to your .pkl file
pkl_path = r"D:\Anubhav\softwares\MFL_software\GMFL_12_Inch_Desktop\backend_data\data_generated\ClockDataFrames\GMFL_12inch_14_03_2026_Oil_India\150.pkl"

# Load pickle file
data = pd.read_pickle(pkl_path)

# Save as CSV
csv_path = r"D:\Anubhav\softwares\client software\Data\project_oil_sample\pickle_data\output1501.csv"
data.to_csv(csv_path, index=False)

print(f"Converted {pkl_path} -> {csv_path}")