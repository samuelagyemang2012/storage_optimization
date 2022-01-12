import pandas as pd

df = pd.read_csv("D:/Datasets/clean_data/Q1_training_new.csv")

limit = len(df) * 0.8
df = df.dropna(thresh=limit, axis=1)
print(df.head())

target = df['failure']
no_failure = 0
failure = 0

# Count the number of failures and no failures
for t in target:
    if t == 0:
        no_failure += 1
    else:
        failure += 1

print("Num no failure: " + str(no_failure))
print("Num failure: " + str(failure))

# Get all failed disks
df_failure = df[df['failure'] == 1]

# Get 1000 unfailed disks
LIMIT = 1000
df_no_failure = df[df['failure'] == 0][0:LIMIT]

# Join failed and unfailed disks
result = pd.concat([df_failure, df_no_failure])

# Remove unneeded columns
result.drop(['Unnamed: 0'], axis=1, inplace=True)

# Shuffle dataframe
result = result.sample(frac=1).reset_index(drop=True)
print(result.head())

# Fill all null values with 0
result = result.fillna(0)
result.isnull().values.any()

# Save data to a csv file
result.to_csv('../disk_failure/data/Q1_training_new.csv', index=False)

features = result.iloc[:, 1:]
target_ = result['failure']
features['target'] = target_

# Pearson Correlation
p_corr_data = features.corr(method="pearson")
pcd = p_corr_data["target"].to_frame(name="Pearson Correlation").reset_index()
pcd = pcd.rename(columns={'index': 'features'})
pcd = pcd.sort_values(by=['Pearson Correlation'], ascending=False)
top_pearson = pcd.head(11)
top_pearson = top_pearson.iloc[1:, :]
top_pearson.to_csv('../disk_failure/correlation/Q1_training_new.csv/top_pearson.csv', index=None)

# Spearman Correlation
s_corr_data = features.corr(method="spearman")
scd = s_corr_data["target"].to_frame(name="Spearman Correlation").reset_index()
scd = scd.rename(columns={'index': 'features'})
scd = scd.sort_values(by=['Spearman Correlation'], ascending=False)
top_spearman = scd.head(11)
top_spearman = top_spearman.iloc[1:, :]
top_spearman.to_csv('../disk_failure/correlation/Q1_training_new.csv/top_spearman.csv', index=None)

# features = df.iloc[:, 1:]
# target = df['failure']
#
#
# def get_highly_corr_cols(dataset, threshold):
#     col_corr = set()  # Set of all the names of correlated columns
#     corr_matrix = dataset.corr()
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
#                 colname = corr_matrix.columns[i]  # getting the name of column
#                 col_corr.add(colname)
#     return col_corr
#
#
# def list_to_csv(fields, data, path):
#     rows = []
#     for d in data:
#         rows.append([d])
#
#     with open(path, 'w') as f:
#         write = csv.writer(f)
#         write.writerow(fields)
#         write.writerows(rows)
#
#
# #  Remove low variance features
# # delta = 0.01
# # n_features = features.var()
# # n_features = n_features.to_frame(name="variance").reset_index()
# # n_features = n_features.rename(columns={'index': 'features'})
# # n_features = n_features[n_features["variance"] <= delta]
# # print(n_features)
#
#
# # high corr features
# h_corr = get_highly_corr_cols(features, 0.85)
# h_corr = list(h_corr)
# list_to_csv(['features'], h_corr, "../disk_failure/correlation/high_corr_params.csv")
# numerical_features = features.drop(h_corr, axis=1)
#
# # Add target to features
# n_features = features
# n_features["target"] = target
#
# print(n_features.head())
