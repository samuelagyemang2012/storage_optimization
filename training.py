import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pickle

# Load data
path = "correlation/ps_joined_params_training.csv"
df = pd.read_csv(path)
df.head()

print("Data loaded")

# Get features and target
cols = df.columns.tolist()
features = df[cols[1:]]
target = df[cols[0]]

# Convert features and target to numpy arrays
features_ = features.to_numpy()
target_ = target.to_numpy()

# Scale data
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(features_)
y = sc_y.fit_transform(target_.reshape(-1, 1))

pickle.dump(sc_X, open("scalers/X_scaler.sav", 'wb'))
pickle.dump(sc_y, open("scalers/y_scaler.sav", 'wb'))

print("Data scaled and scalers saved")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

print("Data splitted for training")

# Grid Search
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

# SVR
svr_param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [1, 0.1, 0.01, 0.001, 0.0001],
}
svr_grid = GridSearchCV(SVR(), svr_param_grid, verbose=0, n_jobs=-1, cv=cv)

# LR
lr = LinearRegression()

# RR
rr_param_grid = {
    'n_estimators': [10, 30],
    'max_depth': [None, 2, 5, 8],
}
rr_grid = GridSearchCV(RandomForestRegressor(), rr_param_grid, verbose=0, n_jobs=-1, cv=cv)

# MLP
mlp_param_grid = {
    "max_iter": [10, 50],
    "hidden_layer_sizes": [50, 80]
}
mlp_grid = GridSearchCV(MLPRegressor(), mlp_param_grid, verbose=0, n_jobs=-1, cv=cv)

# XGBoost
xgb_param_grid = {
    "n_estimators": [10, 100, 1000],
    "max_depth": [1, 2, 4, 8, 10],
    "eta": [0.1],
    "subsample": [0.7],
    "colsample_bytree": [0.8]
}
xgb_grid = GridSearchCV(XGBRegressor(), xgb_param_grid, verbose=0, n_jobs=-1, cv=cv)

# GradientBoost
gb_param_grid = {
    "n_estimators": [10, 100, 1000],
    "criterion": ["friedman_mse", "mse"],
    "max_depth": [3, 5, 10]
}
gb_grid = GridSearchCV(GradientBoostingRegressor(), gb_param_grid, verbose=0, n_jobs=-1, cv=cv)

print("Grid search initialized")
# Fit models
# Fit Support Vector Regressor
svr_grid = svr_grid.fit(X_train, y_train.reshape(-1, ))
print(svr_grid.best_params_)

# Fit Linear Regression model
lr = lr.fit(X_train, y_train.reshape(-1, ))

# Fit Random Regressor
rr_grid = rr_grid.fit(X_train, y_train.reshape(-1, ))
print(rr_grid.best_params_)

# Fit MLP Regressor
mlp_grid = mlp_grid.fit(X_train, y_train.reshape(-1, ))
print(mlp_grid.best_params_)

# Fit XGB Regressor
xgb_grid = xgb_grid.fit(X_train, y_train.reshape(-1, ))
print(xgb_grid.best_params_)

# Fit GB Regressor
gb_grid = gb_grid.fit(X_train, y_train.reshape(-1, ))
print(gb_grid.best_params_)

print("Training Done")

# Evaluate models

# Evaluate Support Vector Regressor
svr_preds = svr_grid.predict(X_test)
svr_rmse = round(np.sqrt(mean_squared_error(y_test.reshape(-1, ), svr_preds)), 4)
pickle.dump(svr_grid, open("models/svr/svr.sav", 'wb'))
print("SVR RMSE: ", svr_rmse)

# Evaluate Linear Regression model
lr_preds = lr.predict(X_test)
lr_rmse = round(np.sqrt(mean_squared_error(y_test.reshape(-1, ), lr_preds)), 4)
pickle.dump(lr, open("models/lr/lr.sav", 'wb'))
print("LR RMSE: ", lr_rmse)

# Evaluate Random Forest Regressor
rr_preds = rr_grid.predict(X_test)
rr_rmse = round(np.sqrt(mean_squared_error(y_test.reshape(-1, ), rr_preds)), 4)
pickle.dump(rr_grid, open("models/rr/rr.sav", 'wb'))
print("RR RMSE: ", rr_rmse)

# Evaluate MLP Regressor
mlp_preds = mlp_grid.predict(X_test)
mlp_rmse = round(np.sqrt(mean_squared_error(y_test.reshape(-1, ), mlp_preds)), 4)
pickle.dump(mlp_grid, open("models/mlp/mlp.sav", 'wb'))
print("MLP RMSE: ", mlp_rmse)

# Evaluate XGB Regressor
xgb_preds = xgb_grid.predict(X_test)
xgb_rmse = round(np.sqrt(mean_squared_error(y_test.reshape(-1, ), xgb_preds)), 4)
pickle.dump(xgb_grid, open("models/xgb/xgb.sav", 'wb'))
print("XGB RMSE: ", xgb_rmse)

# Evaluate GradientBoosting Regressor
gb_preds = gb_grid.predict(X_test)
gb_rmse = round(np.sqrt(mean_squared_error(y_test.reshape(-1, ), gb_preds)), 4)
pickle.dump(gb_grid, open("models/gb/gb.sav", 'wb'))
print("GB RMSE: ", gb_rmse)

print("Evaluation Done")


def original_pred_plot(x_train_, y_train_, model, title, path):
    y_fit = model.predict(x_train_)

    y_train_ = sc_y.inverse_transform(y_train_).reshape(-1, )
    y_fit = sc_y.inverse_transform(y_fit).reshape(-1, )

    sns.set_style("darkgrid")

    f = plt.figure()
    f.set_figwidth(8)
    f.set_figheight(5)

    plt.title(title)
    plt.plot(y_train_, color="blue", label="Original")
    plt.plot(y_fit, color="red", label="Predicted")
    plt.legend()
    plt.savefig(path)
    return y_fit


def compare_original_pred(y_train_, y_preds_):
    data = []
    y_train_ = sc_y.inverse_transform(y_train_).reshape(-1, )
    # y_preds_ = sc_y.inverse_transform(y_preds_).reshape(-1,)
    for i, a in enumerate(y_train_):
        data.append([a, y_preds_[i]])
    return pd.DataFrame(data, columns=["Original", "Predicted"])


def display_info(df1, df2):
    print("Train Data")
    print(df1.head())
    print("")
    print("Test Data")
    print(df2.head())


# Train data
train_svr_yfit = original_pred_plot(X_train, y_train, svr_grid, "SVR-Train Line of Fit", "models/svr/train.png")
# Test data
test_svr_yfit = original_pred_plot(X_test, y_test, svr_grid, "SVR-Test Line of Fit", "models/svr/test.png")

# LR train and test plot (Original vs Predicted)
# Train data
train_lr_yfit = original_pred_plot(X_train, y_train, lr, "LR-Train Line of Fit", "models/lr/train.png")
# Test data
test_lr_yfit = original_pred_plot(X_test, y_test, lr, "LR-Test Line of Fit", "models/lr/test.png")

# RR train and test plot (Original vs Predicted)
# Train data
train_rr_yfit = original_pred_plot(X_train, y_train, rr_grid, "RR-Train Line of Fit", "models/rr/train.png")
# Test data
test_rr_yfit = original_pred_plot(X_test, y_test, rr_grid, "RR-Test Line of Fit", "models/rr/test.png")

# MLP train and test plot (Original vs Predicted)
# Train data
train_mlp_yfit = original_pred_plot(X_train, y_train, mlp_grid, "MLP-Train Line of Fit", "models/mlp/train.png")
# Test data
test_mlp_yfit = original_pred_plot(X_test, y_test, mlp_grid, "MLP-Test Line of Fit", "models/mlp/test.png")

# XGB train and test plot (Original vs Predicted)
train_xgb_yfit = original_pred_plot(X_train, y_train, xgb_grid, "XGB-Train Line of Fit", "models/xgb/train.png")
# Test data
test_xgb_yfit = original_pred_plot(X_test, y_test, xgb_grid, "XGB-Test Line of Fit", "models/xgb/test.png")

# GB train and test plot (Original vs Predicted)
train_gb_yfit = original_pred_plot(X_train, y_train, gb_grid, "GB-Train Line of Fit", "models/gb/train.png")
# Test data
test_gb_yfit = original_pred_plot(X_test, y_test, gb_grid, "GB-Test Line of Fit", "models/gb/test.png")

print("Plots complete")
# Plot all lines of Fit

f = plt.figure()
f.set_figwidth(25)
f.set_figheight(10)

l = [i for i in range(0, len(target))]
plt.scatter(l, target, label="Original")

svr_ = svr_grid.predict(X)
lr_ = lr.predict(X)
rr_ = rr_grid.predict(X)
mlp_ = mlp_grid.predict(X)
xgb_ = xgb_grid.predict(X)
gb_ = gb_grid.predict(X)

plt.title("Line of Fit of ML algorithms")
plt.plot(sc_y.inverse_transform(svr_), color="red", label="SVR")
plt.plot(sc_y.inverse_transform(lr_), color="green", label="LR")
plt.plot(sc_y.inverse_transform(rr_), color="black", label="RR")
plt.plot(sc_y.inverse_transform(mlp_), color="orange", label="MLP")
plt.plot(sc_y.inverse_transform(xgb_), color="magenta", label="XGB")
plt.plot(sc_y.inverse_transform(gb_), color="blue", label="GB")

plt.legend()

f.savefig("models/lines_of_fit.png")
print("Lines of fit drawn")

# Save RMSE

rmses = [["SVR", svr_rmse],
         ["LR", lr_rmse],
         ["RR", rr_rmse],
         ["MLP", mlp_rmse],
         ["XGB", xgb_rmse],
         ["GB", gb_rmse]]
df_rmses = pd.DataFrame(rmses, columns=["model", "score"])
df_rmses.to_csv("models/rmses.csv", index=None)
