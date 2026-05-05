import pandas as pd

# 🔹 Hardcoded file paths
input_file = r"D:\Anubhav\softwares\client software\Data\pickle9 - Copy\pipetally_main\pipetally_main_12inch_new (1).xlsx"
output_file = r"D:\Anubhav\softwares\client software\Data\pickle9 - Copy\pipetally_main\pipetally_main_123inch_new (1).xlsx"

# 🔹 Values to KEEP (unsorted input)
to_keep = [599, 546, 320, 34, 54, 81, 791, 1019, 1042, 779, 978, 580, 586, 593, 596, 603, 606]

# -----------------------------

# Load Excel
df = pd.read_excel(input_file)

# Clean column names (important)
df.columns = df.columns.str.strip().str.lower()

# (Optional) ensure numeric
df['s_no'] = pd.to_numeric(df['s_no'], errors='coerce')

# 🔥 Keep ONLY these rows
filtered_df = df[df['s_no'].isin(to_keep)]

# 🔥 Sort ascending
filtered_df = filtered_df.sort_values(by='s_no', ascending=True)

# Save output
filtered_df.to_excel(output_file, index=False)

print("✅ Done. File saved at:", output_file)