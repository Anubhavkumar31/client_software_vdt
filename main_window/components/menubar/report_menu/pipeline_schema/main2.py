import pandas as pd

# Load the Excel file and specific sheet
df = pd.read_excel(r"D:\PIE_dv_new\14inch Petrofac pipetally.xlsx", sheet_name="Pipe Tally")

# Clean up all weird characters, multiple spaces, etc.
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True).str.replace(u'\xa0', ' ')

# Print cleaned column names
print("Actual column names from file:")
for col in df.columns:
    print(repr(col))  # shows hidden spaces, non-breaking spaces, etc.

# Optional: also print as a list
print("All column names as list:")
print(df.columns.tolist())
