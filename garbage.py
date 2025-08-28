import os, pandas as pd

for f in [r"F:\work_new\client_software\test_data_cs\pickle_4\35.pkl",r"F:\work_new\client_software\test_data_cs\pickle_4\36.pkl"]:  # replace with your files
    size = os.path.getsize(f)/1024/1024
    df = pd.read_pickle(f)
    print(f, "->", df.shape, f"{size:.2f} MB")
