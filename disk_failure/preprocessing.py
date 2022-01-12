import pandas as pd
import numpy as np

TIME_PERIOD = 30

# Step 1
# df = pd.read_csv("D:/Datasets/blackbase/merged/Q2_merged.csv")
# df = df.fillna(0)
# print(df.isnull().values.any())
# df.to_csv('D:/Datasets/blackbase/selected_data/Q2-testing_data.csv', index=False)

#############################################################################

# Step 2
# df = pd.read_csv('D:/Datasets/blackbase/disks/MQ01ABF050_test.csv')
df = pd.read_csv('D:/Datasets/blackbase/disks/ST12000NM0007_test.csv')
print('before')
print(df['failure'].value_counts())

# # drop duplicates
df = df.drop_duplicates(subset=['serial_number', 'date'])

# sort data by date and serial number
df = df.sort_values(by=['serial_number', 'date'], ascending=[True, True])

# get failed disk
df_failure = df[df['failure'] == 1]
df_failure = df_failure[['serial_number', 'date']]
df_failure = df_failure.rename(index=str, columns={"date": "failure_date"})

df_failure = df_failure.sort_values(by=['serial_number'], ascending=[True])
serial_nums = df_failure['serial_number'].to_list()

dd = df
dx = dd.groupby('serial_number').first().reset_index()

df_start = dx[dx["serial_number"].isin(serial_nums)]

df_failure['start_date'] = df_start['date'].to_list()

df_failure[['failure_date', 'start_date']] = df_failure[['failure_date', 'start_date']].apply(pd.to_datetime)
df_failure['days'] = (df_failure['failure_date'] - df_failure['start_date']).dt.days
df_failure = df_failure[df_failure['days'] < TIME_PERIOD]
# df_failure.to_csv('D:/Datasets/blackbase/testing_data/failure.csv')
new_serials = df_failure['serial_number'].to_list()
# print(df_failure)
#############################################################################

# Step 3
df['failure'] = np.where((df["serial_number"].isin(new_serials)), 1, df['failure'])
df.to_csv('D:/Datasets/blackbase/prepared_data/ST12000NM0007_test_prepared.csv', index=False)
print('after')
print(df['failure'].value_counts())

##############################################################################
