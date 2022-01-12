import pandas as pd
import os

training_path = "D:/Datasets/blackbase/clean_data/Q2/"
output_path = 'D:/Datasets/blackbase/merged/Q2_merged.csv'

data = os.listdir(training_path)

dfs = []

for d in data:
    df = pd.read_csv(training_path + d)
    # print(d)
    dfs.append(df)

print('Merging')
result = pd.concat(dfs)
result.to_csv(output_path, index=False)

print("Done")
