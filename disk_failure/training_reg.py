import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pickle

# functions

THRESHOLD = 0.5


def get_acc(X_test_, y_test_, model, scaler):
    correct_prediction = 0
    for i in range(len(X_test_)):

        prediction = model.predict(X_test_[i].reshape(1, X_test_.shape[1]))
        pred_raw = scaler.inverse_transform(prediction.reshape(1, -1))
        y_raw = scaler.inverse_transform(y_test_[i].reshape(1, -1))

        pred = round(pred_raw.item(), 2)
        y_ = round(y_raw.item(), 2)

        if pred > THRESHOLD and y_ == 1 or pred < THRESHOLD and y_ == 0:
            correct_prediction += 1

    acc = (correct_prediction / len(X_test_)) * 100
    print('Accuracy: ' + str(round(acc, 2)) + "%")


def eval_reg(X_test_, y_test_, model):
    preds = model.predict(X_test_)
    rmse = round(np.sqrt(mean_squared_error(y_test_.reshape(-1, ), preds)), 4)
    rsquared = round(r2_score(y_test_.reshape(-1, ), preds), 4)

    print("RMSE: ", rmse)
    print("R2: ", rsquared)


def eval_clf(X_test_, y_test_, model):
    preds = model.predict(X_test_)
    acc = round(accuracy_score(y_test_.reshape(-1, ), preds), 4)

    print("Accuracy: ", acc)


def do_sampling(X, y, o, u):
    oversampler = RandomOverSampler(sampling_strategy=o)
    X, y = oversampler.fit_resample(X, y)

    undersampler = RandomUnderSampler(sampling_strategy=u)
    X, y = undersampler.fit_resample(X, y)

    return X, y


def scale_data(X, y, scaler):
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    return X, y


def data_to_numpy(data):
    return data.to_numpy()


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
# all_features, target = do_sampling(all_features, target, 0.4, 0.6)
all_features, target = do_sampling(all_features, target, 0.4, 0.6)
# all_features, target = do_sampling(all_features, target, 0.4, 0.6)
###############################################################################

# Convert features to numpy arrays
print('convert features to numpy arrays')
all_features_ = data_to_numpy(all_features)
target_ = data_to_numpy(target)

###############################################################################
# eX = data_to_numpy(eval_features)
eX = data_to_numpy(eval_features)
ey = data_to_numpy(eval_target)
###############################################################################

# Scale data
print('scale data for regression')
scaler = StandardScaler()
X, y = scale_data(all_features_, target_, scaler)
reX, rey = scale_data(eX, ey, scaler)

# Split data
print('split data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# print(cX_train)
# Grid Search
print('setup up model')
cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)

# XGBoost
xgb_param_grid = {
    "n_estimators": [300],
    "max_depth": [7],  # [1, 2, 4, 8, 10],
    "eta": [0.01],
    "tree_method": ['approx'],
    "learning_rate": [0.08],
    "subsample": [0.75],
    "colsample_bytree": [1],
    "gamma": [0]
}
xgb_grid_reg = GridSearchCV(XGBRegressor(seed=20), xgb_param_grid, verbose=0, n_jobs=-1, cv=cv)

# Fit XGB Regressor
print('train model')
reg_model = xgb_grid_reg.fit(X_train, y_train)

print(reg_model.best_params_)

# Evaluate XGB Regressor
print('evaluate reg model')
eval_reg(X_test, y_test, reg_model)
eval_reg(reX, rey, reg_model)
print('')
