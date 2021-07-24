"""
Preprocess a the list of params in ceph_parameters.csv file and save it to a txt file.
"""

def generate_parameter_list(params):
    data = ""
    for i in range(len(params)):
        params[i] = params[i].replace("_", " ")
        data += params[i] + "\n"

    f = open("results/params.txt", "w")
    f.write(data)
    print("Parameters saved")
