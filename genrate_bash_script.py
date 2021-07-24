import pandas as pd
from get_params_list import generate_parameter_list

"""
Generate a bash file thats randomly generate parameters based data in the ceph_parameters.csv file, 
creates a ceph.conf file and runs a rados bench test.

:returns a txt named get_rados_data.txt
"""

# Read file
path = "ceph_parameters.csv"
config_file = "/etc/ceph/ceph.conf"

df = pd.read_csv(path)

# Fetch params
params = df["Parameters"].tolist()

datatypes = df["DataType"]
starts = df["Start"]
ends = df["End"]

# print(datatypes.unique())
# Init script
script = "#!/bin/bash" + "\n" + "declare -a params" + "\n" + "\n"

# Remove _ from paramaters
# for i in range(len(params)):
#     params[i] = params[i].replace("_", " ")
script += 'echo "Generating params"' + "\n"
for i, p in enumerate(params):
    if datatypes[i] == 'double' or datatypes[i] == 'float':
        script += p + "=$(awk -v min=" + str(starts[i]) + " -v max=" + str(
            ends[i]) + " 'BEGIN{srand(); print min+rand()*(max-min+1)}')" + "\n"
        script += 'params[' + str(i) + ']=$' + p + "\n"
        script += "\n"

    if datatypes[i] == 'bool':
        script += p + '=$[ $RANDOM % 2]' + "\n"
        script += 'params[' + str(i) + ']=$' + p + "\n"
        script += "\n"

    if datatypes[i] == 'uint' or datatypes[i] == 'int' or datatypes[i] == 'size':
        script += p + '=$[ $RANDOM % ' + str(int(starts[i])) + ' + ' + str(int(ends[i])) + ']' + "\n"
        script += 'params[' + str(i) + ']=$' + p + "\n"
        script += "\n"
"""
# ceph config default
# minimal ceph.conf for 8fbf4e36-e6c6-11eb-91ce-6bc35296a5da
[global]
        fsid = 8fbf4e36-e6c6-11eb-91ce-6bc35296a5da
        mon_host = [v2:172.19.203.10:3300/0,v1:172.19.203.10:6789/0]
"""
script += 'echo "Generating ceph config"' + "\n"
script += 'echo "">' + config_file + "\n"
script += 'echo "# minimal ceph.conf for 8fbf4e36-e6c6-11eb-91ce-6bc35296a5da">>' + config_file + "\n"
script += 'echo "[global]">>' + config_file + "\n"
script += 'echo "fsid = 8fbf4e36-e6c6-11eb-91ce-6bc35296a5da">>' + config_file + "\n"
script += 'echo "mon_host = [v2:172.19.203.10:3300/0,v1:172.19.203.10:6789/0]">>' + config_file + "\n" + "\n"

script += "n=0" + "\n"
script += "while read line;do " + "\n"
# data+=$line=${params[n]}
script += "echo $line = ${params[n]}>>" + config_file + "\n"
script += "echo $line = ${params[n]}>>gen.txt" + "\n"
script += "#echo $line" + "\n"
script += "n=$((n+1))" + "\n"
script += "done <params.txt" + "\n"

script += 'echo "Rados bench-marking with params"' + "\n"
script += 'rados bench -p paraTune 10 seq >> rados_results.txt' + "\n"
script += 'echo "Done"' + "\n" + "\n"

f = open("results/get_rados_data.txt", "w")
f.write(script)

generate_parameter_list(params)
print("done")
