## In this script, all tensors are assumed to have the nucleotide order "A"-"C"-"G"-"T"

import pandas
import numpy
import h5py
import sklearn.cross_validation
import re
import itertools
import datetime
import joblib


# 1. read all PWMs

all_PWMs_hdf5_filename = "../data/for.1..1.JAPSAR.CORE.2016.PWM.hdf5"
all_PWMs_hdf5_filehandle = h5py.File(all_PWMs_hdf5_filename, mode="r")
all_PWMs_name_list = all_PWMs_hdf5_filehandle.keys()
all_PWMs_matrix_list = []
for temp_PWM_name in all_PWMs_name_list:
    all_PWMs_matrix_list.append(all_PWMs_hdf5_filehandle[temp_PWM_name][:, :])

all_PWMs_hdf5_filehandle.close()


# 2. generate tensors





#for PWM_name, PWM_matrix, positive_sequence_random_seed, negative_sequence_random_seed in list(itertools.izip(all_PWMs_name_list, all_PWMs_matrix_list, range(0, len(all_PWMs_name_list)), range(0, len(all_PWMs_name_list))))[0:5]:

def single_run_for_single_PWM_function(PWM_name, PWM_matrix, positive_sequence_random_seed, negative_sequence_random_seed):
    PWM_length_int = PWM_matrix.shape[1]
    PWM_normalized_matrix = PWM_matrix / PWM_matrix.sum(axis=0)[None, :]

    sequence_length_int = 50
    sequence_position_int_vector = numpy.arange(start=0, stop=sequence_length_int, step=1, dtype=int)
    positive_sequence_count_int = 2000
    negative_sequence_count_int = 2000
    minimal_random_fragment_length_before_signal_sequence_in_positive_sequence_int = 0
    dataset_output_prefix = "../data/for.1..2.sequence.dataset."
    temp_fasta_output_prefix = "../data/for.1..2.sequence.dataset."
    
    # 2.1. generate positive sequence tensors
    numpy.random.seed(positive_sequence_random_seed)

    positive_sequence_tensor = numpy.zeros([positive_sequence_count_int, sequence_length_int, 4])
    positive_sequence_heading_random_sequence_length_int_vector = numpy.random.randint(low=minimal_random_fragment_length_before_signal_sequence_in_positive_sequence_int, high=sequence_length_int - minimal_random_fragment_length_before_signal_sequence_in_positive_sequence_int - PWM_length_int + 1, size=positive_sequence_count_int)
    positive_sequence_trailing_random_sequence_length_int_vector = sequence_length_int - PWM_length_int - positive_sequence_heading_random_sequence_length_int_vector

    for positive_sequence_index_int in range(0, positive_sequence_count_int):
        positive_sequence_index_repeat_int_vector = numpy.repeat(positive_sequence_index_int, sequence_length_int)
        positive_sequence_heading_random_sequence_length_int = positive_sequence_heading_random_sequence_length_int_vector[positive_sequence_index_int]
        positive_sequence_heading_random_sequence_nucleotide_type_int_vector = numpy.random.randint(low=0, high=4, size=positive_sequence_heading_random_sequence_length_int)
        
        positive_sequence_trailing_random_sequence_length_int = positive_sequence_trailing_random_sequence_length_int_vector[positive_sequence_index_int]
        positive_sequence_trailing_random_sequence_nucleotide_type_int_vector = numpy.random.randint(low=0, high=4, size=positive_sequence_trailing_random_sequence_length_int)
        
        positive_sequence_signal_sequence_nucleotide_type_int_vector = numpy.apply_along_axis(func1d=lambda temp_row : numpy.random.choice([0, 1, 2, 3], size=1, p=temp_row)[0], axis=0, arr=PWM_normalized_matrix)
        
        positive_sequence_whole_nucleotide_type_int_vector = numpy.concatenate([positive_sequence_heading_random_sequence_nucleotide_type_int_vector, positive_sequence_signal_sequence_nucleotide_type_int_vector, positive_sequence_trailing_random_sequence_nucleotide_type_int_vector], axis=0)
        positive_sequence_tensor[positive_sequence_index_repeat_int_vector, sequence_position_int_vector, positive_sequence_whole_nucleotide_type_int_vector] = 1 

    # 2.2. generate negative sequence tensors
    numpy.random.seed(negative_sequence_random_seed)

    negative_sequence_tensor = numpy.zeros([negative_sequence_count_int, sequence_length_int, 4])
    negative_sequence_heading_random_sequence_length_int_vector = numpy.random.randint(low=minimal_random_fragment_length_before_signal_sequence_in_positive_sequence_int, high=sequence_length_int - minimal_random_fragment_length_before_signal_sequence_in_positive_sequence_int + 1, size=negative_sequence_count_int)
    negative_sequence_trailing_random_sequence_length_int_vector = sequence_length_int - negative_sequence_heading_random_sequence_length_int_vector

    for negative_sequence_index_int in range(0, negative_sequence_count_int):
        negative_sequence_index_repeat_int_vector = numpy.repeat(negative_sequence_index_int, sequence_length_int)
        negative_sequence_heading_random_sequence_length_int = negative_sequence_heading_random_sequence_length_int_vector[negative_sequence_index_int]
        negative_sequence_heading_random_sequence_nucleotide_type_int_vector = numpy.random.randint(low=0, high=4, size=negative_sequence_heading_random_sequence_length_int)

        negative_sequence_trailing_random_sequence_length_int = negative_sequence_trailing_random_sequence_length_int_vector[negative_sequence_index_int]
        negative_sequence_trailing_random_sequence_nucleotide_type_int_vector = numpy.random.randint(low=0, high=4, size=negative_sequence_trailing_random_sequence_length_int)

        negative_sequence_whole_nucleotide_type_int_vector = numpy.concatenate([negative_sequence_heading_random_sequence_nucleotide_type_int_vector, negative_sequence_trailing_random_sequence_nucleotide_type_int_vector], axis=0)
        negative_sequence_tensor[negative_sequence_index_repeat_int_vector, sequence_position_int_vector, negative_sequence_whole_nucleotide_type_int_vector] = 1 

    sequence_tensor = numpy.concatenate([positive_sequence_tensor, negative_sequence_tensor], axis=0)
    y_vector = numpy.concatenate([numpy.repeat(1, positive_sequence_count_int), numpy.repeat(0, negative_sequence_count_int)], axis=0)

    training_and_validation_index_vector, testing_index_vector = list(sklearn.cross_validation.StratifiedKFold(y=y_vector, n_folds=4, shuffle=False))[0]

    X_for_training_and_validation_tensor = sequence_tensor[training_and_validation_index_vector, :, :]
    X_for_testing_tensor = sequence_tensor[testing_index_vector, :, :]

    y_for_training_and_validation_vector = y_vector[training_and_validation_index_vector]
    y_for_testing_vector = y_vector[testing_index_vector]

    training_index_in_training_and_validation_vector, validation_index_in_training_and_validation_vector = list(sklearn.cross_validation.StratifiedKFold(y=y_for_training_and_validation_vector, n_folds=5, shuffle=False))[0]

    X_for_training_tensor = X_for_training_and_validation_tensor[training_index_in_training_and_validation_vector, :, :]
    X_for_validation_tensor = X_for_training_and_validation_tensor[validation_index_in_training_and_validation_vector, :, :]

    y_for_training_vector = y_for_training_and_validation_vector[training_index_in_training_and_validation_vector]
    y_for_validation_vector = y_for_training_and_validation_vector[validation_index_in_training_and_validation_vector]

    dataset_output_filename = dataset_output_prefix + "__" + PWM_name + ".hdf5"
    dataset_output_filehandle = h5py.File(name=dataset_output_filename, mode="w")
    dataset_output_filehandle.create_dataset(name="X_for_training_tensor", data=X_for_training_tensor, compression="gzip")
    dataset_output_filehandle.create_dataset(name="X_for_validation_tensor", data=X_for_validation_tensor, compression="gzip")
    dataset_output_filehandle.create_dataset(name="X_for_testing_tensor", data=X_for_testing_tensor, compression="gzip")
    dataset_output_filehandle.create_dataset(name="X_for_training_and_validation_tensor", data=X_for_training_and_validation_tensor, compression="gzip")
    dataset_output_filehandle.create_dataset(name="y_for_training_vector", data=y_for_training_vector, compression="gzip")
    dataset_output_filehandle.create_dataset(name="y_for_validation_vector", data=y_for_validation_vector, compression="gzip")
    dataset_output_filehandle.create_dataset(name="y_for_testing_vector", data=y_for_testing_vector, compression="gzip")
    dataset_output_filehandle.create_dataset(name="y_for_training_and_validation_vector", data=y_for_training_and_validation_vector, compression="gzip")
    dataset_output_filehandle.create_dataset(name="PWM_matrix", data=PWM_matrix, compression="gzip")
    dataset_output_filehandle.create_dataset(name="PWM_normalized_matrix", data=PWM_normalized_matrix, compression="gzip")
    dataset_output_filehandle.close()


    dataset_name_str_list = ['training', 'validation', 'testing']
    temp_sequence_name_and_content_list_list_for_each_dataset_list = []
    for temp_X_tensor, temp_y_vector, temp_dataset_name_str in itertools.izip([X_for_training_tensor, X_for_validation_tensor, X_for_testing_tensor], [y_for_training_vector, y_for_validation_vector, y_for_testing_vector],  dataset_name_str_list):
        temp_sequence_name_and_content_list_list = []
        for sample_index_int in range(0, temp_X_tensor.shape[0]):
            sequence_length_int = int(temp_X_tensor[sample_index_int, :, :].sum())
            sequence_name_str = temp_dataset_name_str + "__" + str(sample_index_int) + "__" + str(sequence_length_int) + "__" + str(temp_y_vector[sample_index_int])
            sequence_matrix = temp_X_tensor[sample_index_int, 0:sequence_length_int, :]
            sequence_content_str = "".join([['A', 'C', 'G', 'T'][temp_index_int] for temp_index_int in numpy.argmax(sequence_matrix, axis=1)])
            temp_sequence_name_and_content_list_list.append([sequence_name_str, sequence_content_str])
        temp_sequence_name_and_content_list_list_for_each_dataset_list.append(temp_sequence_name_and_content_list_list)

    for temp_sequence_name_and_content_list_list, temp_dataset_name_str in itertools.izip(temp_sequence_name_and_content_list_list_for_each_dataset_list, dataset_name_str_list):
        temp_fasta_output_filename_str = temp_fasta_output_prefix + "__" + temp_dataset_name_str + "__" + PWM_name + ".fasta"
        with open(temp_fasta_output_filename_str, "w") as temp_fasta_output_filehandle:
            for temp_sequence_name_and_content_list in temp_sequence_name_and_content_list_list:
                temp_fasta_output_filehandle.write(">")
                temp_fasta_output_filehandle.write(temp_sequence_name_and_content_list[0])
                temp_fasta_output_filehandle.write("\n")
                temp_fasta_output_filehandle.write(temp_sequence_name_and_content_list[1])
                temp_fasta_output_filehandle.write("\n")

    print(datetime.datetime.now().strftime("%H-%M-%S") + ": Finish processing PWM " + PWM_name + " of length " + str(PWM_length_int))

    return PWM_name

parallel_object_Parallel = joblib.Parallel(n_jobs=30, backend="multiprocessing")
result = parallel_object_Parallel(map(joblib.delayed(single_run_for_single_PWM_function), all_PWMs_name_list, all_PWMs_matrix_list, range(0, len(all_PWMs_name_list)), range(0, len(all_PWMs_name_list)) ))

with open("../data/for.1..2.all.PWMs.name.txt", "w") as all_PWMs_name_filehandle:
    for PWM_name in all_PWMs_name_list:
        all_PWMs_name_filehandle.write(PWM_name)
        all_PWMs_name_filehandle.write("\n")
