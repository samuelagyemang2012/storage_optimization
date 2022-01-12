import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import uuid


# functions
def eval_clf(X_test_, y_test_, model):
    preds = model.predict(X_test_)
    AUC = round(roc_auc_score(y_test_.reshape(-1, ), preds), 4)
    precision = round(precision_score(y_test_.reshape(-1, ), preds), 4)
    # tn, fp, fn, tp = confusion_matrix(y_test_.argmax(axis=1), preds).ravel()
    acc = round(accuracy_score(y_test_.reshape(-1, ), preds), 4)

    return (AUC * 100), (precision * 100), (acc * 100)


def do_sampling(X, y, o, u):
    oversampler = RandomOverSampler(sampling_strategy=o)
    X, y = oversampler.fit_resample(X, y)

    undersampler = RandomUnderSampler(sampling_strategy=u)
    X, y = undersampler.fit_resample(X, y)

    return X, y


def over_sample(X, y, o):
    oversampler = RandomOverSampler(sampling_strategy=o)
    X, y = oversampler.fit_resample(X, y)

    return X, y


def smote(X, y, o, u):
    over = SMOTE(sampling_strategy=o)
    under = RandomUnderSampler(sampling_strategy=u)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    x, y = pipeline.fit_resample(X, y)
    return x, y


def scale_data(X, y, scaler):
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    return X, y


def data_to_numpy(data):
    return data.to_numpy()


def plot_ROC(y, preds, title, path):
    fpr, tpr, threshold = roc_curve(y, preds)
    plt.subplots(1, figsize=(10, 10))
    plt.title(title)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path)


# Load data
print('load data')
train_path = "D:/Datasets/blackbase/prepared_data/MQ01ABF050_train_prepared.csv"
test_path = "D:/Datasets/blackbase/prepared_data/MQ01ABF050_test_prepared.csv"

train_df = pd.read_csv(train_path, index_col=None)
test_df = pd.read_csv(test_path, index_col=None)

# Get columns for selected features
df_pearson = pd.read_csv('../disk_failure/correlation/top_pearson.csv')
df_spearman = pd.read_csv('../disk_failure/correlation/top_spearman.csv')
pearson_cols = df_pearson['features'].to_list()
spearman_cols = df_spearman['features'].to_list()

# Get features and target
print('select target and features')
all_features = train_df.iloc[:, 6:]
spearman_features = train_df[spearman_cols]
pearson_features = train_df[pearson_cols]
target = train_df.iloc[:, 5:6]

eval_features = test_df.iloc[:, 6:]
s_eval_features = eval_features[spearman_cols]
p_eval_features = eval_features[pearson_cols]
eval_target = test_df.iloc[:, 5:6]

# do sampling
################################################################################
print('do sampling')
# all_features, atarget = do_sampling(all_features, target, 0.1, 0.5)
# pearson_features, ptarget = do_sampling(pearson_features, target, 0.1, 0.5)
# spearman_features, starget = do_sampling(spearman_features, target, 0.1, 0.5)

all_features, atarget = smote(all_features, target, 0.1, 0.5)
pearson_features, ptarget = smote(pearson_features, target, 0.1, 0.5)
spearman_features, starget = smote(spearman_features, target, 0.1, 0.5)
###############################################################################

# Convert features to numpy arrays
print('convert features to numpy arrays')
all_features_ = data_to_numpy(all_features)
pearson_features_ = data_to_numpy(pearson_features)
spearman_features_ = data_to_numpy(spearman_features)

atarget_ = data_to_numpy(atarget)
ptarget_ = data_to_numpy(ptarget)
starget_ = data_to_numpy(starget)

###############################################################################
eX = data_to_numpy(eval_features)
p_eX = data_to_numpy(p_eval_features)
s_eX = data_to_numpy(s_eval_features)

ey = data_to_numpy(eval_target)
###############################################################################

# Scale Data
print('scale data')
scaler = StandardScaler()
p_scaler = StandardScaler()
s_scaler = StandardScaler()

all_features_ = scaler.fit_transform(all_features_)
pearson_features_ = p_scaler.fit_transform(pearson_features_)
spearman_features_ = s_scaler.fit_transform(spearman_features_)

# scale eval features
eX = scaler.fit_transform(eX)
p_eX = p_scaler.fit_transform(p_eX)
s_eX = s_scaler.fit_transform(s_eX)

# Split data
print('split data')
X_train, X_test, y_train, y_test = train_test_split(all_features_, atarget_, test_size=0.2, random_state=12)
pX_train, pX_test, py_train, py_test = train_test_split(pearson_features_, ptarget_, test_size=0.2, random_state=12)
sX_train, sX_test, sy_train, sy_test = train_test_split(spearman_features_, starget_, test_size=0.2, random_state=12)

# Grid Search
print('setup up model')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# XGBoost
xgb_param_grid = {
    "n_estimators": [100],  # 1000
    "max_depth": [3],  # [1, 2, 4, 8, 10],
    "eta": [0.01],  # 0.1
    "learning_rate": [0.08],  # 0.01, 0.3
    "subsample": [1],  # 0
    "colsample_bytree": [1],  # 1
    "gamma": [0]  # 1, 5
}
xgb_grid = GridSearchCV(XGBClassifier(seed=20), xgb_param_grid, verbose=0, n_jobs=-1, cv=cv)
p_xgb_grid = GridSearchCV(XGBClassifier(seed=20), xgb_param_grid, verbose=0, n_jobs=-1, cv=cv)
s_xgb_grid = GridSearchCV(XGBClassifier(seed=20), xgb_param_grid, verbose=0, n_jobs=-1, cv=cv)

# Decision Tree
tree_param_grid = {
    "max_depth": [3, 5, 7],
    "max_leaf_nodes": [5, 8, 10]
}
# tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, verbose=0, n_jobs=-1, cv=cv)
# p_tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, verbose=0, n_jobs=-1, cv=cv)
# s_tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, verbose=0, n_jobs=-1, cv=cv)

# Random Forest
rf_param_grid = {
    "n_estimators": [100],  # , 300],
    # "criterion": ['squared_error'],
    "max_depth": [5]  # , 7, 10]
}
# rf_grid = GridSearchCV(RandomForestClassifier(), rf_param_grid, verbose=0, n_jobs=-1, cv=cv)
# p_rf_grid = GridSearchCV(RandomForestClassifier(), rf_param_grid, verbose=0, n_jobs=-1, cv=cv)
# s_rf_grid = GridSearchCV(RandomForestClassifier(), rf_param_grid, verbose=0, n_jobs=-1, cv=cv)

# Fit XGB Clf
print('xgb train model')
xgb_clf_model = xgb_grid.fit(X_train, y_train)
p_xgb_clf_model = p_xgb_grid.fit(pX_train, py_train)
s_xgb_clf_model = s_xgb_grid.fit(sX_train, sy_train)

# Fit D-Tree Clf
print('decision tree train model')
# tree_clf_model = tree_grid.fit(X_train, y_train)
# p_tree_clf_model = p_tree_grid.fit(pX_train, py_train)
# s_tree_clf_model = s_tree_grid.fit(sX_train, sy_train)

# Fit RF Clf
print('random forest train model')
# rf_clf_model = rf_grid.fit(X_train, y_train)
# p_rf_clf_model = p_rf_grid.fit(pX_train, py_train)
# s_rf_clf_model = s_rf_grid.fit(sX_train, sy_train)

# Evaluate XGB Classifier
print('evaluation')

train_auc, train_prec, train_rec = eval_clf(X_test, y_test, xgb_clf_model)
eval_auc, eval_prec, eval_rec = eval_clf(eX, ey, xgb_clf_model)

p_train_auc, p_train_prec, p_train_rec = eval_clf(pX_test, py_test, p_xgb_clf_model)
p_eval_auc, p_eval_prec, p_eval_rec = eval_clf(p_eX, ey, p_xgb_clf_model)

s_train_auc, s_train_prec, s_train_rec = eval_clf(sX_test, sy_test, s_xgb_clf_model)
s_eval_auc, s_eval_prec, s_eval_rec = eval_clf(s_eX, ey, s_xgb_clf_model)

results = [['All features', eval_auc, eval_prec, eval_rec],
           ['Pearson features', p_eval_auc, p_eval_prec, p_eval_rec],
           ['Spearman features', s_eval_auc, s_eval_prec, s_eval_rec]]

results_df = pd.DataFrame(results, columns=['Feature Type', 'Eval AUC', 'Eval Prec', 'Eval Acc'])
id = uuid.uuid4()
id = str(id)[:8]
results_df.to_csv('../disk_failure/results/rf/results_' + id + ".csv")
print(results_df)

print('save roc curve')
preds = xgb_clf_model.predict(eX)
plot_ROC(ey, preds, 'ROC - XGB all', "../disk_failure/graphs/rf/all_roc_" + id + ".png")

p_preds = p_xgb_clf_model.predict(p_eX)
plot_ROC(ey, p_preds, 'ROC - XGB pearson', "../disk_failure/graphs/rf/pearson_roc_" + id + ".png")

s_preds = s_xgb_clf_model.predict(s_eX)
plot_ROC(ey, s_preds, 'ROC - XGB spearman', "../disk_failure/graphs/rf/spearman_roc_" + id + ".png")
