import pandas as pd
import json

paths = ["results/svr_results.csv", "results/lr_results.csv", "results/rr_results.csv", "results/mlp_results.csv",
         "results/xgb_results.csv", "results/gb_results.csv"]
models = ["svr", "lr", "rr", "mlp", "xgb", "gb"]
data_path = "correlation/ps_joined_params_training.csv"
results = []

data_df = pd.read_csv(data_path)
data_iops = data_df["Target"].tolist()

for i, p in enumerate(paths):
    df = pd.read_csv(p, index_col=None)
    iops = df["score"].tolist()
    max_iops = max(iops)
    best_df = df[df["score"] == max_iops].head(1)
    results.append([models[i], max_iops, best_df["parameters"].item()])

fields = ["model", "best_score", "parameters"]
final_df = pd.DataFrame(results, columns=fields, index=None, )
final_df.to_csv('results/final.csv')


def get_parameters(model_name, df_):
    data = ""
    best_params = eval(df_[df_["model"] == model_name]["parameters"].item())
    for key, value in best_params.items():
        data += key + "=" + str(value) + "\n"

    return data


best_ = get_parameters("svr", final_df)
print(best_)
