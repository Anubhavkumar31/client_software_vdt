import pandas as pd
import numpy as np

# =========================
# PKL PATH
# =========================
pkl_path = r"D:\Anubhav\softwares\client software\client_software_vdt\test.pkl"

# =========================
# LOAD PKL
# =========================
print("\n==============================")
print("STEP 1: Loading PKL")
print("==============================")

df = pd.read_pickle(pkl_path)

print("PKL Loaded Successfully")
print(f"Shape: {df.shape}")

# =========================
# PRINT COLUMNS
# =========================
print("\n==============================")
print("STEP 2: Columns")
print("==============================")

print(df.columns.tolist())

# =========================
# FIND CLOCK COLUMNS
# =========================
print("\n==============================")
print("STEP 3: Finding Clock Columns")
print("==============================")

clock_cols = []

for col in df.columns:

    if isinstance(col, str) and ":" in col:
        clock_cols.append(col)

print(f"Total Clock Columns: {len(clock_cols)}")
print(clock_cols)

# =========================
# EXTRACT CLOCK DATA
# =========================
print("\n==============================")
print("STEP 4: Extracting Clock Data")
print("==============================")

df_clk = df[clock_cols]

print(f"Clock Data Shape: {df_clk.shape}")

# =========================
# BASIC STATS
# =========================
print("\n==============================")
print("STEP 5: Basic Statistics")
print("==============================")

print(df_clk.describe())

# =========================
# CHECK NANs
# =========================
print("\n==============================")
print("STEP 6: NaN Check")
print("==============================")

print(df_clk.isna().sum())

# =========================
# GLOBAL MIN/MAX
# =========================
print("\n==============================")
print("STEP 7: Global Min/Max")
print("==============================")

global_min = np.nanmin(df_clk.values)
global_max = np.nanmax(df_clk.values)

print(f"Global Min : {global_min}")
print(f"Global Max : {global_max}")

# =========================
# PERCENTILES
# =========================
print("\n==============================")
print("STEP 8: Percentiles")
print("==============================")

percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

vals = np.nanpercentile(df_clk.values, percentiles)

for p, v in zip(percentiles, vals):
    print(f"{p}% : {v}")

# =========================
# ABSOLUTE DISTRIBUTION
# =========================
print("\n==============================")
print("STEP 9: Absolute Value Distribution")
print("==============================")

abs_vals = np.abs(df_clk.values)

abs_percentiles = np.nanpercentile(
    abs_vals,
    [50, 75, 90, 95, 99]
)

for p, v in zip([50, 75, 90, 95, 99], abs_percentiles):
    print(f"ABS {p}% : {v}")

# =========================
# SAMPLE DATA
# =========================
print("\n==============================")
print("STEP 10: Sample Data")
print("==============================")

print(df_clk.head())

print("\n==============================")
print("ANALYSIS COMPLETE")
print("==============================")