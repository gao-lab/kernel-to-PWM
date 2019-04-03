import numpy
import glob

config_dict_list = [
    {"dataset":"train", "transform":"exp", "result_prefix":"exp_"},
    {"dataset":"val", "transform":"exp", "result_prefix":"exp_"},
    {"dataset":"test", "transform":"exp", "result_prefix":"exp_"},
    {"dataset":"train", "transform":"Deepbind", "result_prefix":"Deepbind_"},
    {"dataset":"val", "transform":"Deepbind", "result_prefix":"Deepbind_"},
    {"dataset":"test", "transform":"Deepbind", "result_prefix":"Deepbind_"}
]

for config_dict in config_dict_list:
    dataset = config_dict["dataset"]
    transform = config_dict["transform"]
    result_prefix = config_dict["result_prefix"]
    result_file_list = glob.glob("../data/for.02..4." + dataset + "_"   + result_prefix + "results/*.npy")
    result_file_ordered_list = sorted(result_file_list)
    result_matrix_list = [numpy.load(result_file) for result_file in result_file_ordered_list]
    result_total_matrix = numpy.concatenate(result_matrix_list, axis=0)
    numpy.save("../data/for.02..5." + dataset + "_" + transform + "_total_log_prob.npy" , result_total_matrix)
