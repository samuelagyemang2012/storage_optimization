"""
Preprocess a the list of params in ceph_parameters.csv file and save it to a txt file.
"""


def generate_parameter_list(params, path):
    data = ""
    for i in range(len(params)):
        params[i] = params[i].replace("_", " ")
        data += params[i] + "\n"

    f = open(path, "w")
    f.write(data)
    print("Parameters saved")
