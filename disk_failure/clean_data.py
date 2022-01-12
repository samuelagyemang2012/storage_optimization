import pandas as pd
import numpy as np
import os

data_path = "D:/Datasets/blackbase/Q2/"
data_files = os.listdir(data_path)
df = pd.read_csv(data_path + data_files[0])
THRESHOLD = 0.80

output_path = "D:/Datasets/blackbase/clean_data/Q2/"


def remove_nan_columns(df_):
    df1 = df_.replace(r'^\s*$', np.NaN, regex=True)
    df2 = df1.dropna(how='all', axis=1)
    limit = len(df2) * THRESHOLD
    df3 = df2.dropna(thresh=limit, axis=1)
    return df3


for d in data_files:
    df = pd.read_csv(data_path + d)
    new_df = remove_nan_columns(df)
    # new_df = new_df.iloc[:, 4:]
    print(d + "-------" + str(len(new_df.columns)))
    new_df.to_csv(output_path + "clean_" + d, index=None)

print("Done")
