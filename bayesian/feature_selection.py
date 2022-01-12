import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import csv

path1 = "ceph_parameters.csv"
path2 = "rados_bench_results/params_average iops2.csv"

# For Ceph
df_iops = pd.read_csv(path2)
df_params = pd.read_csv(path1)

# Display data
print(df_params.head())
print("")
print(df_iops.head())
print("")
print(df_params["DataType"].unique())
print("Data loaded")


# Return n best features with scores for chi2 test
def get_top_features(columns, scores, n):
    df_scores = pd.DataFrame(scores)
    df_columns = pd.DataFrame(columns)
    df_features = pd.concat([df_columns, df_scores], axis=1)
    df_features.columns = ["Feature", "Score"]

    return df_features.nlargest(n, "Score")


def ANOVA_test(features, target, K):
    feature_selector = SelectKBest(score_func=f_classif, k=K)
    best_features = feature_selector.fit(features, target)
    best = get_top_features(features.columns, best_features.scores_, K)
    return best


def correlation_features_test(features, method):
    top_K = features.corr(method=method).index.tolist()
    plt.figure(figsize=(10, 10))
    g = sns.heatmap(features[top_K].corr(method=method), annot=True, cmap="RdYlGn")
    figure = g.get_figure()
    figure.savefig("graphs/" + method + "2.png")


def get_highly_corr_cols(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


def list_to_csv(fields, data, path):
    rows = []
    for d in data:
        rows.append([d])

    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)


def get_feature_intersection(top_p, top_k, top_s, all_data):
    plist = top_p['features'].tolist()
    # klist = top_k['features'].tolist()
    slist = top_s['features'].tolist()

    pset = set(plist[1:])
    # kset = set(klist[1:])
    sset = set(slist[1:])

    # s1 = pset.intersection(kset)
    s2 = list(pset.intersection(sset))

    pd.DataFrame(s2, index=None, columns=['features']).to_csv("correlation/ps_joined_params2.csv", index=False)
    s2.insert(0, "Target")
    all_data[s2].to_csv("correlation/ps_joined_params_training2.csv", index=False)


# Select categorical features
categorical_features_list = df_params[df_params["DataType"] == 'bool']["Parameters"].tolist()
categorical_features = df_iops[categorical_features_list]

# Select numerical features
numerical_features_list = df_params[df_params["DataType"] != 'bool']["Parameters"].tolist()
numerical_features = df_iops[numerical_features_list]

# All Features
all_features = df_iops[df_iops.columns[0:(len(df_iops.columns) - 1)]]

# Select target feature i.e. IOPS
target = df_iops["Average IOPS"]

# Average IOPS scatter plot
sns.set_style("darkgrid")
f = plt.figure()
f.set_figwidth(12)
f.set_figheight(5)
plt.scatter([i for i in range(1, len(target.tolist()) + 1)], target)
plt.ylabel('Average IOPS')
plt.savefig('graphs/scatter_iops2.png')

# Average IOPS line plot
sns.set_style("darkgrid")
f = plt.figure()
f.set_figwidth(12)
f.set_figheight(5)
plt.plot(target, )
plt.ylabel('Average IOPS')
plt.savefig('graphs/line_iops2.png')

print("Target scatter and line graphs saved")

# Filter by variance: Remove features with low variance
delta = 0.01
n_features = numerical_features.var()
n_features = n_features.to_frame(name="variance").reset_index()
n_features = n_features.rename(columns={'index': 'features'})
n_features = n_features[n_features["variance"] <= delta]
print(len(n_features))
n_features.to_csv("correlation/low_variance_params2.csv")

# Remove features with low variance from main df
numerical_features = numerical_features.drop(n_features["features"].tolist(), axis=1)

print("Low variance features removed")

# Remove features that highly correlate with other feature by a threshold greater than 0.85 or -0.85
h_corr = get_highly_corr_cols(numerical_features, 0.85)
h_corr = list(h_corr)
print(len(h_corr))
list_to_csv(['features'], h_corr, "correlation/high_corr_params2.csv")

print("Highly correlated features removed")

# Drop highly correlated features
numerical_features = numerical_features.drop(h_corr, axis=1)

# Add Target to features for correlation tests
new_features = numerical_features
new_features["Target"] = target

# Pearson Correlation Test
p_corr_data = new_features.corr(method="pearson")
pcd = p_corr_data["Target"].to_frame(name="Pearson Correllation").reset_index()
pcd = pcd.rename(columns={'index': 'features'})
pcd = pcd.sort_values(by=['Pearson Correllation'], ascending=False)
# Top 13 features
top_p = pcd.head(8)
correlation_features_test(numerical_features[top_p["features"]], "pearson")
# Save data
p_data = new_features[top_p["features"]]
p_data.to_csv("correlation/pearson_features2.csv", index=None)
top_p.to_csv("correlation/top_pearson_features2.csv", index=None)
print("Pearson correlation done")

# Kendall Correlation Test
k_corr_data = new_features.corr(method="kendall")
kcd = k_corr_data["Target"].to_frame(name="Kendall Correllation").reset_index()
kcd = kcd.rename(columns={'index': 'features'})
kcd = kcd.sort_values(by=['Kendall Correllation'], ascending=False)
# Top 9 features
top_k = kcd.head(9)
correlation_features_test(numerical_features[top_k["features"]], "kendall")
# Save data
k_data = new_features[top_k["features"]]
k_data.to_csv("correlation/kendall_features2.csv", index=None)
top_k.to_csv("correlation/top_kendall_features2.csv", index=None)
print("Kendall correlation done")

# Spearman Correlation Test
s_corr_data = new_features.corr(method="spearman")
scd = s_corr_data["Target"].to_frame(name="Spearman Correllation").reset_index()
scd = scd.rename(columns={'index': 'features'})
scd = scd.sort_values(by=['Spearman Correllation'], ascending=False)
# Top 12 features
top_s = scd.head(9)
correlation_features_test(numerical_features[top_s["features"]], "spearman")
# Save data
s_data = new_features[top_s["features"]]
s_data.to_csv("correlation/spearman_features2.csv", index=None)
top_s.to_csv("correlation/top_spearman_features2.csv", index=None)
print("Spearman correlation done")

get_feature_intersection(top_p, top_k, top_s, numerical_features)
print("Feature intersection complete")

print("done")
