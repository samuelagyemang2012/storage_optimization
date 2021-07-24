import csv
import pandas as pd

"""
Reads parameter values and results generated from rados bench and saves it to a csv file
"""
gen_path = "bench_results/gen.txt"
res_path = "bench_results/rados_results.txt"
ceph_path = "ceph_parameters.csv"

NUM_PARAMS = 128

gen = open(gen_path)
res = open(res_path)
df = pd.read_csv(ceph_path)

datatypes = df["DataType"].tolist()
columns = df["Parameters"].tolist()
columns.append("Average IOPS")

params = []
iops = []


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


# Fetch params from file
for line in gen:
    params.append(line)

# Fetch iops from file
for line in res:
    if line[0:12] == "Average IOPS":
        line = line.split(":")
        iops.append(int(line[1].strip()))

# Split params into chunks of 128 bc we have 128 parameters
params = list(divide_chunks(params, NUM_PARAMS))

# Fetch parameter values for each parameter per rados test
tunings = []
for param in params:
    for p in param:
        data = p.split("=")
        tunings.append(data[1].strip().strip("\n").strip('\''))

tunings = list(divide_chunks(tunings, NUM_PARAMS))

# Convert each parameter from string to its specific type defined by ceph
for i in range(len(tunings)):
    for j in range(len(tunings[i])):
        if datatypes[j] == "uint" or datatypes[j] == "size" or datatypes[j] == "int" or datatypes[j] == "bool":
            tunings[i][j] = int(tunings[i][j])
        else:
            tunings[i][j] = float(tunings[i][j])

# Append IOPS to parameters tunings: Average IOPS is our target
for i, ip in enumerate(iops):
    tunings[i].append(ip)

# Write data to a csv file
with open("bench_results/params_iops.csv", 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)
    write.writerows(tunings)

print("Done")
