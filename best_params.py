import pandas as pd
import json

paths = ["results/svr_results2.csv", "results/lr_results2.csv", "results/rr_results2.csv", "results/mlp_results2.csv",
         "results/xgb_results2.csv", "results/gb_results2.csv"]
models = ["svr", "lr", "rr", "mlp", "xgb", "gb"]
data_path = "correlation/ps_joined_params_training2.csv"
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
final_df = pd.DataFrame(results, columns=fields, index=None)
final_df.to_csv('results/final2.csv')


def get_parameters(model_name, df_):
    data = ""
    best_params = eval(df_[df_["model"] == model_name]["parameters"].item())
    for key, value in best_params.items():
        data += key + "=" + str(value) + "\n"

    return data


for m in models:
    best_ = get_parameters(m, final_df)
    f = open("results/best_params/" + m + "2.txt", "w")
    f.write(best_)
    f.close()

print("done")
