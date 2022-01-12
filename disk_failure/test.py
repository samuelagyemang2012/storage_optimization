import pandas as pd
import numpy as np

df = pd.read_csv("D:/Datasets/blackbase/merged/Q2_merged.csv")
df = df.fillna(0)
# print(df.isnull().values.any())

uniques = df['model'].unique()
print(len(uniques))
# print(df['model'].unique())
# print(len(df['model'].unique()))
#
dx = df[df['model'] == 'ST12000NM0007']  # 'TOSHIBA MQ01ABF050']
print(dx.to_csv('D:/Datasets/blackbase/disks/ST12000NM0007_test.csv'))
print(dx['failure'].value_counts())

# da = []
# for u in uniques:
#     dx = df[df['model'] == u]
#     fail = len(dx[dx['failure'] == 1])
#     ok = len(dx[dx['failure'] == 0])
#     ratio = (fail / ok) * 100
#     da.append([u, fail, ok, ratio])
#
# ndf = pd.DataFrame(da, columns=['disk', 'fail', 'ok', 'fail_ratio'])
# ndf = ndf.sort_values(by=['fail_ratio'], ascending=[False])
# ndf.to_csv('D:/Datasets/blackbase/misc/failure_rate.csv', index=None)
