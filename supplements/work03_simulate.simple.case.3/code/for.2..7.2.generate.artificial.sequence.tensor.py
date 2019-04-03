## In this script, all tensors are assumed to have the nucleotide order "A"-"C"-"G"-"T"

import pandas
import numpy
import h5py
import sklearn.cross_validation
import itertools
import datetime
import joblib
import Bio.SeqIO
import os
import stat
import sys

# 1. read CM information

CM_name_str, CM_ACC_str = sys.argv[1:3]
sequence_length_int = int(sys.argv[3])

CM_dir_str = "../data/for.2..7.simulation." + "CM.accession__" + CM_ACC_str + ".CM.name__" + CM_name_str +  ".sequence.length__" + str(sequence_length_int) + "/"


# 2. do the transformation: positive and negative fasta files -> positive and negative tensors and y vectors

print(datetime.datetime.now().strftime("%H-%M-%S") + ": transforming fasta files to tensors ")

sequence_tensor_list = []
sequence_y_vector_list = []



def transform_sequence_record_to_tensor(sequence_record_row):
    sequence_content = sequence_record_row['sequence_content']
    sequence_index = sequence_record_row.name
    #print(datetime.datetime.now().strftime("%H-%M-%S") + ": Processing sequence " + str(sequence_index) + " of length " + str(len(sequence_content)))
    sequence_character_list = list(sequence_content)
    sequence_length = len(sequence_character_list)
    sequence_position_in_feature_map_list = [character_to_position_in_feature_map_dict[sequence_character] for sequence_character in sequence_character_list]
    sequence_width_list = numpy.arange(start=0, stop=sequence_length, step=1, dtype=int)
    sequence_positions_used_list = [index for index, element in enumerate(sequence_position_in_feature_map_list) if element != None]
    sequence_position_used_in_feature_map_list = [sequence_position_in_feature_map_list[i] for i in sequence_positions_used_list]
    sequence_width_used_list = [sequence_width_list[i] for i in sequence_positions_used_list]
    sequence_index_used_list = [sequence_index] * len(sequence_position_used_in_feature_map_list)
    sequence_tensor_rightpadded_collection_tensor[sequence_index_used_list, sequence_width_used_list, sequence_position_used_in_feature_map_list] = 1
    return None


for dataset_type_str in ["positive", "negative"]:
    fasta_filename_str = CM_dir_str + "/for.2..7.1." + dataset_type_str + ".sequence.fasta"        
    fasta_parse_content_list = [(record.id, str(record.seq)) for record in Bio.SeqIO.parse(fasta_filename_str, "fasta")]
    fasta_parse_name_list = [record[0] for record in fasta_parse_content_list]
    sequence_table_DataFrame = pandas.DataFrame(
        fasta_parse_content_list,
        columns=['sequence_name', 'sequence_content']
    )
    tensor_dimension_0__sequence_count = sequence_table_DataFrame.shape[0]
    tensor_dimension_1__sequence_maximal_length = len(sequence_table_DataFrame.ix[0, 1])
    tensor_dimension_2__sequence_feature_count = 4
    sequence_tensor_rightpadded_collection_tensor = numpy.zeros([
        tensor_dimension_0__sequence_count,
        tensor_dimension_1__sequence_maximal_length,
        tensor_dimension_2__sequence_feature_count
    ])
    
    character_to_position_in_feature_map_dict = {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }

        
    
    not_used_variable_1 = sequence_table_DataFrame.apply(transform_sequence_record_to_tensor, axis=1)
    sequence_tensor_list.append(sequence_tensor_rightpadded_collection_tensor)

total_sequence_tensor = numpy.concatenate(sequence_tensor_list, axis=0)
total_sequence_y_vector = numpy.concatenate([numpy.repeat(1, sequence_tensor_list[0].shape[0]), numpy.repeat(0, sequence_tensor_list[1].shape[0])])

# 3. generate training, validation, training+validation, and testing tensors

print(datetime.datetime.now().strftime("%H-%M-%S") + ": generating training, validation, training+validation, and testing tensors ")


training_and_validation_index_in_total_vector, testing_index_in_total_vector = list(sklearn.cross_validation.StratifiedKFold(y=total_sequence_y_vector, n_folds=4, shuffle=False))[0]
    
training_and_validation_y_vector = total_sequence_y_vector[training_and_validation_index_in_total_vector]
testing_y_vector = total_sequence_y_vector[testing_index_in_total_vector]
training_and_validation_sequence_tensor = total_sequence_tensor[training_and_validation_index_in_total_vector, :, :]
testing_sequence_tensor = total_sequence_tensor[testing_index_in_total_vector, :, :]

    
    
training_index_in_training_and_validation_vector, validation_index_in_training_and_validation_vector = list(sklearn.cross_validation.StratifiedKFold(y=training_and_validation_y_vector, n_folds=5, shuffle=False))[0]
training_y_vector = training_and_validation_y_vector[training_index_in_training_and_validation_vector]
training_sequence_tensor = training_and_validation_sequence_tensor[training_index_in_training_and_validation_vector, :, :]
validation_y_vector = training_and_validation_y_vector[validation_index_in_training_and_validation_vector]
validation_sequence_tensor = training_and_validation_sequence_tensor[validation_index_in_training_and_validation_vector, :, :]

hdf5_filename_str = CM_dir_str + "/for.2..7.2.sequence.tensor.hdf5"

final_result_hdf5_filehandle = h5py.File(hdf5_filename_str, 'w')
final_result_hdf5_filehandle.create_dataset('sequence_name_list', data=fasta_parse_name_list, compression="gzip")
final_result_hdf5_filehandle.create_dataset('training_sequence_tensor', data=training_sequence_tensor, compression="gzip")
final_result_hdf5_filehandle.create_dataset('training_y_vector', data=training_y_vector, compression="gzip")
final_result_hdf5_filehandle.create_dataset('validation_sequence_tensor', data=validation_sequence_tensor, compression="gzip")
final_result_hdf5_filehandle.create_dataset('validation_y_vector', data=validation_y_vector, compression="gzip")
final_result_hdf5_filehandle.create_dataset('training_and_validation_sequence_tensor', data=training_and_validation_sequence_tensor, compression="gzip")
final_result_hdf5_filehandle.create_dataset('training_and_validation_y_vector', data=training_and_validation_y_vector, compression="gzip")
final_result_hdf5_filehandle.create_dataset('testing_sequence_tensor', data=testing_sequence_tensor, compression="gzip")
final_result_hdf5_filehandle.create_dataset('testing_y_vector', data=testing_y_vector, compression="gzip")

final_result_hdf5_filehandle.close()

# 4. generate training, validation, testing, and training+validation fasta subset files

print(datetime.datetime.now().strftime("%H-%M-%S") + ": generating training, validation, training+validation, and testing fasta subset files ")

dataset_name_str_list = ['training', 'validation', 'training_and_validation', 'testing']
temp_sequence_name_and_content_list_list_for_each_dataset_list = []
for temp_X_tensor, temp_y_vector, temp_dataset_name_str in itertools.izip([training_sequence_tensor, validation_sequence_tensor, training_and_validation_sequence_tensor, testing_sequence_tensor], [training_y_vector, validation_y_vector, training_and_validation_y_vector, testing_y_vector],  dataset_name_str_list):
    temp_sequence_name_and_content_list_list = []
    for sample_index_int in range(0, temp_X_tensor.shape[0]):
        sequence_length_int = int(temp_X_tensor[sample_index_int, :, :].sum())
        sequence_name_str = temp_dataset_name_str + "__" + str(sample_index_int) + "__" + str(sequence_length_int) + "__" + str(temp_y_vector[sample_index_int])
        sequence_matrix = temp_X_tensor[sample_index_int, 0:sequence_length_int, :]
        sequence_content_str = "".join([['A', 'C', 'G', 'T'][temp_index_int] for temp_index_int in numpy.argmax(sequence_matrix, axis=1)])
        temp_sequence_name_and_content_list_list.append([sequence_name_str, sequence_content_str])
    temp_sequence_name_and_content_list_list_for_each_dataset_list.append(temp_sequence_name_and_content_list_list)

for temp_sequence_name_and_content_list_list, temp_dataset_name_str in itertools.izip(temp_sequence_name_and_content_list_list_for_each_dataset_list, dataset_name_str_list):
    temp_fasta_output_filename_str = CM_dir_str + "for.2..7.2.sequence.fasta." + "__" + temp_dataset_name_str + ".fasta"
    with open(temp_fasta_output_filename_str, "w") as temp_fasta_output_filehandle:
        for temp_sequence_name_and_content_list in temp_sequence_name_and_content_list_list:
            temp_fasta_output_filehandle.write(">")
            temp_fasta_output_filehandle.write(temp_sequence_name_and_content_list[0])
            temp_fasta_output_filehandle.write("\n")
            temp_fasta_output_filehandle.write(temp_sequence_name_and_content_list[1])
            temp_fasta_output_filehandle.write("\n")

