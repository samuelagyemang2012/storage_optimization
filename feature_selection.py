import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import csv

params_path = "ceph_parameters.csv"
iops_path = "rados_bench_results/params_average latency(s).csv"
target_ = "Average Latency(s)"
K = 10

df_iops = pd.read_csv(iops_path)
df_params = pd.read_csv(params_path)


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


def correlation_test(features, target, K, method, path):
    features["Target"] = target
    top_K = features.corr(method=method).index.tolist()[0:K]
    top_K.append("Target")

    plt.figure(figsize=(20, 20))

    g = sns.heatmap(features[top_K].corr(method=method), annot=True, cmap="RdYlGn")
    figure = g.get_figure()
    figure.savefig(path)


def save(columns, data, path):
    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(columns)
        write.writerows(data)


# Select categorical features
categorical_features_list = df_params[df_params["DataType"] == 'bool']["Parameters"].tolist()
categorical_features = df_iops[categorical_features_list]

# Select numerical features
numerical_features_list = df_params[df_params["DataType"] != 'bool']["Parameters"].tolist()
numerical_features = df_iops[numerical_features_list]

# All Features
all_features = df_iops[df_iops.columns[0:(len(df_iops.columns) - 1)]]

# Select target feature i.e. IOPS
target = df_iops[target_]

# Plot Average IOPS
sns.set_style("darkgrid")
plt.plot(target)
plt.ylabel(target_)
# plt.show()
plt.savefig(target_.lower() + '.png')

# Pearson Correlation test
correlation_test(numerical_features, target, K, "pearson", "correlation/pearson_" + target_.lower() + ".png")

# Kendall Correlation test
correlation_test(numerical_features, target, K, "kendall", "correlation/kendall_" + target_.lower() + ".png")

# Spearman Correlation test
correlation_test(numerical_features, target, K, "spearman", "correlation/spearman" + target_.lower() + ".png")

# ANOVA test
anova = ANOVA_test(categorical_features, target, K)
data = []
top_features = anova["Feature"].tolist()
scores = anova["Score"].tolist()

for i, tf in enumerate(top_features):
    data.append([tf, scores[i]])

# Save data to file
save(anova.columns, data, "correlation/anova_results_"+target_.lower()+".csv")
