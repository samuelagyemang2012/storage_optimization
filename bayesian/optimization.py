import optuna
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time

svr_results = []
lr_results = []
rr_results = []
mlp_results = []
xgb_results = []
gb_results = []

model_paths = ["models/svr/svr2.sav",
               "models/lr/lr2.sav",
               "models/rr/rr2.sav",
               "models/mlp/mlp2.sav",
               "models/xgb/xgb2.sav",
               "models/gb/gb2.sav"]

scaler_paths = ["scalers/X_scaler2.sav", "scalers/y_scaler2.sav"]

ceph_param_path = "ceph_parameters.csv"
data_path = "correlation/ps_joined_params_training2.csv"

# Load Data
params_df = pd.read_csv(ceph_param_path)
data_df = pd.read_csv(data_path)

print(params_df["DataType"].unique())


def load_pickle_data(paths):
    data = []
    for p in paths:
        with (open(p, "rb")) as openfile:
            while True:
                try:
                    data.append(pickle.load(openfile))
                except EOFError:
                    break
    return data


def filter_params(params):
    data_range = []
    for dc in params:
        a = params_df[params_df["Parameters"] == dc]
        param = a["Parameters"].item()
        datatype = a["DataType"].item()
        start_ = a["Start"].item()
        end_ = a["End"].item()

        if datatype == "uint" or datatype == "size" or datatype == "int":
            start_ = int(start_)
            end_ = int(end_)

        if datatype == "double" or datatype == "float":
            start_ = float(start_)
            end_ = float(end_)

        data_range.append([param, datatype, start_, end_])

    return data_range


# Filter required parameters

data_cols = list(data_df.columns[1:])
filtered_params = filter_params(data_cols)
print(filtered_params)

# Load models and scalers

# Load models
models = load_pickle_data(model_paths)
scalers = load_pickle_data(scaler_paths)

print(len(models))
print(len(scalers))
print('Models and scalers loaded')


def svr_objective(trial):
    trial_params = []

    for f in filtered_params:
        param = f[0]
        datatype = f[1]
        start_ = f[2]
        end_ = f[3]

        if datatype == "size" or datatype == "uint" or datatype == "int":
            parameter = trial.suggest_int(param, int(start_), int(end_))
            trial_params.append(parameter)

        if datatype == "float" or datatype == "double":
            parameter = trial.suggest_float(param, float(start_), float(end_))
            trial_params.append(parameter)

    data = [tp for tp in trial_params]
    data = np.array(data)

    X = scalers[0].transform(data.reshape(1, -1))

    # Surrogate function
    y = models[0].predict(X.reshape(1, -1))
    y = scalers[1].inverse_transform(y)

    # score
    return round(y.item())


def lr_objective(trial):
    trial_params = []

    for f in filtered_params:
        param = f[0]
        datatype = f[1]
        start_ = f[2]
        end_ = f[3]

        if datatype == "size" or datatype == "uint" or datatype == "int":
            parameter = trial.suggest_int(param, int(start_), int(end_))
            trial_params.append(parameter)

        if datatype == "float" or datatype == "double":
            parameter = trial.suggest_float(param, float(start_), float(end_))
            trial_params.append(parameter)

    data = [tp for tp in trial_params]
    data = np.array(data)

    X = scalers[0].transform(data.reshape(1, -1))

    # Surrogate function
    y = models[1].predict(X.reshape(1, -1))
    y = scalers[1].inverse_transform(y)

    # score
    return round(y.item())


def rr_objective(trial):
    trial_params = []

    for f in filtered_params:
        param = f[0]
        datatype = f[1]
        start_ = f[2]
        end_ = f[3]

        if datatype == "size" or datatype == "uint" or datatype == "int":
            parameter = trial.suggest_int(param, int(start_), int(end_))
            trial_params.append(parameter)

        if datatype == "float" or datatype == "double":
            parameter = trial.suggest_float(param, float(start_), float(end_))
            trial_params.append(parameter)

    data = [tp for tp in trial_params]
    data = np.array(data)

    X = scalers[0].transform(data.reshape(1, -1))

    # Surrogate function
    y = models[2].predict(X.reshape(1, -1))
    y = scalers[1].inverse_transform(y)

    # score
    return round(y.item())


def mlp_objective(trial):
    trial_params = []

    for f in filtered_params:
        param = f[0]
        datatype = f[1]
        start_ = f[2]
        end_ = f[3]

        if datatype == "size" or datatype == "uint" or datatype == "int":
            parameter = trial.suggest_int(param, int(start_), int(end_))
            trial_params.append(parameter)

        if datatype == "float" or datatype == "double":
            parameter = trial.suggest_float(param, float(start_), float(end_))
            trial_params.append(parameter)

    data = [tp for tp in trial_params]
    data = np.array(data)

    X = scalers[0].transform(data.reshape(1, -1))

    # Surrogate function
    y = models[3].predict(X.reshape(1, -1))
    y = scalers[1].inverse_transform(y)

    # score
    return round(y.item())


def xgb_objective(trial):
    trial_params = []

    for f in filtered_params:
        param = f[0]
        datatype = f[1]
        start_ = f[2]
        end_ = f[3]

        if datatype == "size" or datatype == "uint" or datatype == "int":
            parameter = trial.suggest_int(param, int(start_), int(end_))
            trial_params.append(parameter)

        if datatype == "float" or datatype == "double":
            parameter = trial.suggest_float(param, float(start_), float(end_))
            trial_params.append(parameter)

    data = [tp for tp in trial_params]
    data = np.array(data)

    X = scalers[0].transform(data.reshape(1, -1))

    # Surrogate function
    y = models[4].predict(X.reshape(1, -1))
    y = scalers[1].inverse_transform(y)

    # score
    return round(y.item())


def gb_objective(trial):
    trial_params = []

    for f in filtered_params:
        param = f[0]
        datatype = f[1]
        start_ = f[2]
        end_ = f[3]

        if datatype == "size" or datatype == "uint" or datatype == "int":
            parameter = trial.suggest_int(param, int(start_), int(end_))
            trial_params.append(parameter)

        if datatype == "float" or datatype == "double":
            parameter = trial.suggest_float(param, float(start_), float(end_))
            trial_params.append(parameter)

    data = [tp for tp in trial_params]
    data = np.array(data)

    X = scalers[0].transform(data.reshape(1, -1))

    # Surrogate function
    y = models[5].predict(X.reshape(1, -1))
    y = scalers[1].inverse_transform(y)

    # score
    return round(y.item())


def bayesian_opt(obj_fun, num_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(obj_fun, n_trials=num_trials)

    trial = study.best_trial
    # print('result: {}'.format(trial.value))
    # print("Best hyperparameters: {}".format(trial.params))
    # optuna.visualization.plot_optimization_history(study).show()
    return trial


def save_optimization_results(data, path):
    df = pd.DataFrame(data, index=None, columns=['score', 'parameters'])
    df.to_csv(path, index=False)


n_trials = 500
loops_ = 30

# Bayesian Optimization
# SVR model
start = time.time()
for i in range(loops_):
    svr_study = bayesian_opt(svr_objective, n_trials)
    svr_results.append([svr_study.value, svr_study.params])
save_optimization_results(svr_results, "results/svr_results2.csv")
end = time.time()
print("Taken taken: ", round((end - start) / 60, 4))
print("SVR Optimization done")
print("")

# LR model
start = time.time()
for i in range(loops_):
    lr_study = bayesian_opt(lr_objective, n_trials)
    lr_results.append([lr_study.value, lr_study.params])
save_optimization_results(lr_results, "results/lr_results2.csv")
end = time.time()
print("Taken taken: ", round((end - start) / 60, 4))
print("LR Optimization done")
print("")

# RR model
start = time.time()
for i in range(loops_):
    rr_study = bayesian_opt(rr_objective, n_trials)
    rr_results.append([rr_study.value, rr_study.params])
save_optimization_results(rr_results, "results/rr_results2.csv")
end = time.time()
print("Taken taken: ", round((end - start) / 60, 4))
print("RR Optimization done")
print("")

# MLP model
start = time.time()
for i in range(loops_):
    mlp_study = bayesian_opt(mlp_objective, n_trials)
    mlp_results.append([mlp_study.value, mlp_study.params])
save_optimization_results(mlp_results, "results/mlp_results2.csv")
end = time.time()
print("Taken taken: ", round((end - start) / 60, 4))
print("MLP Optimization done")
print("")

# XGB model
start = time.time()
for i in range(loops_):
    xgb_study = bayesian_opt(xgb_objective, n_trials)
    xgb_results.append([xgb_study.value, xgb_study.params])
save_optimization_results(xgb_results, "results/xgb_results2.csv")
end = time.time()
print("Taken taken: ", round((end - start) / 60, 4))
print("XGB Optimization done")
print("")

# GB model
start = time.time()
for i in range(loops_):
    gb_study = bayesian_opt(gb_objective, n_trials)
    gb_results.append([gb_study.value, gb_study.params])
save_optimization_results(gb_results, "results/gb_results2.csv")
end = time.time()
print("Taken taken: ", round((end - start) / 60, 4))
print("GB Optimization done")
