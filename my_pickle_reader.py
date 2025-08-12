import pandas as pd

# Use raw string (r'...') to avoid backslash issues on Windows
# df = pd.read_pickle(r'D:\test_data_cs\pickle\2.pkl')

# # Print all column names, one per line
# print("Column names in the pickle file:\n")
# for col in df.columns:
#     print(col)


print("Actual columns in pipe_Tally_2:")

# pipe_Tally_2 = pd.read_pickle(r'D:\test_data_cs\pipe_tally_8inch.xlsx')
pipe_Tally_2 = pd.read_excel(r'D:\test_data_cs\pipe_tally_8inch.xlsx')

for col in pipe_Tally_2.columns:
    print(f"'{col}'")