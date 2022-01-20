import numpy as np
from numpy import genfromtxt
import shap
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_path):
    files = os.listdir(data_path)
    data = []

    for f in files:
        print(f)
        if f.split(".")[1] == 'csv':
            d = genfromtxt(data_path + f, delimiter=",")
            data.append(d)

    return data


def load_models(model_path):
    files = os.listdir(model_path)
    models = []

    for f in files:
        if f.split(".")[1] == 'sav':
            m = pickle.load(open(model_path + f, 'rb'))
            models.append(m)

    return models


name = 'rf'
print('loading data and models')
DATA = load_data('../disk_failure/models/data/test/' + name + '/')
models = load_models('../disk_failure/models/' + name + '/')

# Feature names
df_all = pd.read_csv('D:/Datasets/blackbase/prepared_data/MQ01ABF050_train_prepared.csv')
df_p = pd.read_csv('../disk_failure/correlation/top_pearson.csv')
df_s = pd.read_csv('../disk_failure/correlation/top_spearman.csv')

all_features = df_all.columns.to_list()[6:]
pearson_features = df_p['features'].to_list()
spearman_features = df_s['features'].to_list()

features = [all_features, pearson_features, spearman_features]

print(len(all_features))
print(all_features)
print(len(pearson_features))
print(len(spearman_features))

# SHAP
# explainer = shap.TreeExplainer(models[0].best_estimator_)
# shap_values = explainer.shap_values(DATA[0])
# print(len(shap_values))
# shap.summary_plot(shap_values, DATA[0], feature_names=features[0])

# for i in range(len(models)):
#     path = "../disk_failure/SHAP/" + name + "/" + name + "_beeswarm_plot" + str(i) + ".png"
#     f = plt.figure()
#     shap_values = shap.TreeExplainer(models[i].best_estimator_).shap_values(DATA[i])
#     shap.summary_plot(shap_values[1], DATA[i], feature_names=features[i])
#
#     f.savefig(path, bbox_inches='tight', dpi=100)
#     plt.clf()
