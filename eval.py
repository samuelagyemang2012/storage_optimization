def_res_path = "rados_bench_results/eval/default_results.txt"
ml_res_path = "rados_bench_results/eval/ml_results.txt"

TARGET = "Average IOPS"

def_results = open(def_res_path)
ml_results = open(ml_res_path)

def_list = []
ml_list = []

for line in def_results:
    if line[0:len(TARGET)] == TARGET:
        line = line.split(":")
        iops = int(line[1].strip())
        def_list.append(iops)

for line in ml_results:
    if line[0:len(TARGET)] == TARGET:
        line = line.split(":")
        iops = int(line[1].strip())
        ml_list.append(iops)

print("Default Average")
print(sum(def_list)/len(def_list))

print("ML Average")
print(sum(ml_list)/len(ml_list))
