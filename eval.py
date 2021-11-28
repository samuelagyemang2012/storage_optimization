def_res_path = "rados_bench_results/eval/default_results.txt"
# def_res_path = "rados_bench_results/rados_results2.txt"
gb_res_path = "rados_bench_results/eval/gb.txt"
lr_res_path = "rados_bench_results/eval/lr.txt"
mlp_res_path = "rados_bench_results/eval/mlp.txt"
rr_res_path = "rados_bench_results/eval/rr.txt"
svr_res_path = "rados_bench_results/eval/svr.txt"
xgb_res_path = "rados_bench_results/eval/xgb.txt"

TARGET = "Average IOPS"

def_results = open(def_res_path)
gb_results = open(gb_res_path)
lr_results = open(lr_res_path)
mlp_results = open(mlp_res_path)
rr_results = open(rr_res_path)
svr_results = open(svr_res_path)
xgb_results = open(xgb_res_path)


def get_average(file_data):
    data_list = []
    for line in file_data:
        if line[0:len(TARGET)] == TARGET:
            line = line.split(":")
            iops = int(line[1].strip())
            data_list.append(iops)
    # print(max(data_list))
    print(round(sum(data_list[0:50]) / len(data_list[0:50])))
    print("-----------------------------")


print("Default Average")
get_average(def_results)

print("GB Average")
get_average(gb_results)

print("LR Average")
get_average(lr_results)

print("MLP Average")
get_average(mlp_results)

print("RR Average")
get_average(rr_results)

print("SVR Average")
get_average(svr_results)

print("XGB Average")
get_average(xgb_results)
