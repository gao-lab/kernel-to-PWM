## Summary of steps
## 4. generate the benchmarking dataset
## TODO move this part to for.1..2.0.generate.input.for.step.7.with.smaller.base.py
## This script repeats step 1-3 and finish step 4. There's code redundancy and it should be removed in future.

import sys
import numpy
import pandas
import datetime
import keras
import theano
import scipy.signal
import h5py

motif_id_str = sys.argv[1]
input_sequence_partial_path_str = sys.argv[2]


## 1. read the motif parameters

motif_params_filename_str = "../data/for.0..2.deepbind-v0.11-linux/db/params/" + motif_id_str + ".txt"
motif_params_dict = {}
with open(motif_params_filename_str, "r") as motif_params_filehandle:
    temp_line_str = motif_params_filehandle.readline()
    while temp_line_str:
        if temp_line_str.startswith("#") != True:
            param_name_str, param_raw_value_str = temp_line_str.split("=")
            param_name_stripped_str = param_name_str.strip()
            param_raw_value_stripped_str = param_raw_value_str.strip()
            if param_raw_value_stripped_str == "":
                param_raw_value_stripped_str = "0"                
            if param_name_stripped_str in ["reverse_complement", "num_detectors", "detector_len", "has_avg_pooling", "num_hidden"]:
                ## integer scalar
                motif_params_dict[param_name_stripped_str] = int(param_raw_value_stripped_str)                
            else:
                ## float vector or 3D-tensor or 2D-tensor
                ## the correct shape of the 2D/3D-tensor cannot be determined until other relevant parameters have been read
                motif_params_dict[param_name_stripped_str] = numpy.asarray([float(temp_number_str) for temp_number_str in param_raw_value_stripped_str.split(",") ])
        temp_line_str = motif_params_filehandle.readline()

motif_params_dict["detectors_in_3D_tensor"] = numpy.reshape(a=motif_params_dict["detectors"], newshape=[motif_params_dict["num_detectors"], 4, motif_params_dict["detector_len"]], order="F")
## note that in Deepbind, the "convolution" does not reverse the kernel in practice (i.e., it's essentially a cross-correlation). Therefore for keras 2.0.3 we still need a reversion here
motif_params_dict["kernel_W"] = numpy.asarray([ numpy.fliplr( motif_params_dict["detectors_in_3D_tensor"][index_of_detector, :, :] ) for index_of_detector in range(0, motif_params_dict["num_detectors"]) ]).swapaxes(0, 2)
motif_params_dict["kernel_b"] = numpy.tile(0.0, motif_params_dict["num_detectors"])


if motif_params_dict["num_hidden"] > 0:
    motif_params_dict["weights1_in_correct_shape"] = numpy.reshape(a=motif_params_dict["weights1"], newshape=[motif_params_dict["num_hidden"], motif_params_dict["num_detectors"]], order="F").transpose()
    motif_params_dict["weights2_in_correct_shape"] = motif_params_dict["weights2"][:, None]
else:
    motif_params_dict["weights1_in_correct_shape"] = numpy.reshape(a=motif_params_dict["weights1"], newshape=[1, motif_params_dict["num_detectors"]], order="F").transpose()

## [motif_params_dict["detectors_with_correct_shape"][i, :, :, 0].argmax(axis=0) for i in range(0, motif_params_dict["num_detectors"])]
## [motif_params_dict["kernel_W"][i, :, :, 0].argmax(axis=0) for i in range(0, motif_params_dict["num_detectors"])]


## 2. read the sequences

## note that Deepbind uses 0.25 to pad both sides of sequences (with padding length being "kernel (or detector) length - 1" for each side)

input_sequence_filename_str = "../data/for.0..3.deepbind.nbt.3300.dataset/data/" + input_sequence_partial_path_str
input_sequence_raw_DataFrame = pandas.read_csv(filepath_or_buffer=input_sequence_filename_str, sep="\t", header=0)

## skip all sequences with N/n
input_sequence_DataFrame = input_sequence_raw_DataFrame.ix[input_sequence_raw_DataFrame.ix[:, "seq"].str.contains("N|n") == False, :]
input_sequence_DataFrame.loc[:, "number"] = range(0, input_sequence_DataFrame.shape[0])

input_sequence_str_list = input_sequence_DataFrame.ix[:, "seq"].tolist()
input_sequence_count_int = len(input_sequence_str_list)
input_sequence_max_length_int = numpy.max([len(temp_input_sequence) for temp_input_sequence in input_sequence_str_list])

input_sequence_tensor = numpy.zeros([
    input_sequence_count_int, input_sequence_max_length_int, 4
])

character_to_position_in_feature_map_dict = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'U': 3,
    'a': 0,
    'c': 1,
    'g': 2,
    't': 3,
    'u': 3
}

def transform_sequence_record_to_tensor(sequence_record_row):
    sequence_content = sequence_record_row['seq']
    sequence_index = sequence_record_row['number']
    sequence_character_list = list(sequence_content)
    sequence_length = len(sequence_character_list)
    sequence_position_in_feature_map_list = [character_to_position_in_feature_map_dict[sequence_character] for sequence_character in sequence_character_list]
    sequence_width_list = numpy.arange(start=0, stop=sequence_length, step=1, dtype=int)
    sequence_positions_used_list = [index for index, element in enumerate(sequence_position_in_feature_map_list) if element != None]
    sequence_position_used_in_feature_map_list = [sequence_position_in_feature_map_list[i] for i in sequence_positions_used_list]
    sequence_width_used_list = [sequence_width_list[i] for i in sequence_positions_used_list]
    sequence_index_used_list = [sequence_index] * len(sequence_position_used_in_feature_map_list)
    input_sequence_tensor[sequence_index_used_list, sequence_width_used_list, sequence_position_used_in_feature_map_list] = 1
    return None
        
not_used_variable_1 = input_sequence_DataFrame.apply(transform_sequence_record_to_tensor, axis=1)


input_sequence_padded_with_one_quarter_on_both_sides_tensor = numpy.pad(array=input_sequence_tensor, pad_width=[
    [ 0, 0 ],
    [ motif_params_dict["detector_len"] - 1, motif_params_dict["detector_len"] - 1 ],
    [ 0, 0 ]
], mode="constant", constant_values=0.25)


## 3. build the CNN model
## the CNN model serves as the basis of all models (CNN, latter regression with my PFM, and latter regression with Deepbind's PFM)


single_input_shape_tuple = input_sequence_padded_with_one_quarter_on_both_sides_tensor.shape[1:3]

## 3.1. build the direct Conv-Pooling and assign the weights

model_input = keras.layers.Input(shape=single_input_shape_tuple, dtype="float32")
model_layer_Conv1D = keras.layers.convolutional.Conv1D(
    input_shape=single_input_shape_tuple,
    filters=motif_params_dict["num_detectors"],
    kernel_size=motif_params_dict["detector_len"],
    padding='valid',
    activation='relu'
)
model_layer_GlobalMaxPooling = keras.layers.pooling.GlobalMaxPooling1D()
model_layer_GlobalAveragePooling = keras.layers.pooling.GlobalAveragePooling1D()

model_intermediate_direct_pooling_result = None 
if motif_params_dict["has_avg_pooling"] == 1:
    model_intermediate_direct_pooling_result = model_layer_GlobalAveragePooling(model_layer_Conv1D(model_input))
else:
    model_intermediate_direct_pooling_result = model_layer_GlobalMaxPooling(model_layer_Conv1D(model_input))

model_layer_Conv1D.set_weights([motif_params_dict["kernel_W"], motif_params_dict["kernel_b"]])

## 3.2. build the RC Conv-Pooling and take its max with the direct Conv-Pooling (if needed); no weights is assigned in this step

model_intermediate_final_pooling_result = None    
if motif_params_dict["reverse_complement"] == 1:
        
    ## here the 0th dimension is the batch dimension
    ## reverse: reverse the 2nd dimension
    ## complement: the 1st dimension has the order 'A, C, G, T', and thus reversing the 1st dimension is enough to make the complement
    model_layer_RC = keras.layers.core.Lambda(lambda x: keras.backend.reverse(x, axes=(1, 2)), output_shape=single_input_shape_tuple)
    model_intermediate_RC_pooling_result = None
    if motif_params_dict["has_avg_pooling"] == 1:
        model_intermediate_RC_pooling_result = model_layer_GlobalAveragePooling(model_layer_Conv1D(model_layer_RC(model_input)))
    else:
        model_intermediate_RC_pooling_result = model_layer_GlobalMaxPooling(model_layer_Conv1D(model_layer_RC(model_input)))

    model_layer_pick_larger = keras.layers.Maximum()
    model_intermediate_final_pooling_result = model_layer_pick_larger([model_intermediate_direct_pooling_result, model_intermediate_RC_pooling_result])
else:
    model_intermediate_final_pooling_result = model_intermediate_direct_pooling_result

## 3.3. build the Dense layers and assign the weights
    
model_output = None
if motif_params_dict["num_hidden"] > 0:
    model_layer_dense1 = keras.layers.core.Dense(units=motif_params_dict["num_hidden"])
    model_layer_activation_ReLU = keras.layers.core.Activation("relu")
    model_layer_dense2 = keras.layers.core.Dense(units=1)
    model_output = model_layer_dense2(model_layer_activation_ReLU(model_layer_dense1(model_intermediate_final_pooling_result)))
    model_layer_dense1.set_weights([motif_params_dict["weights1_in_correct_shape"], motif_params_dict["biases1"]])
    model_layer_dense2.set_weights([motif_params_dict["weights2_in_correct_shape"], motif_params_dict["biases2"]])
else:
    model_layer_dense1 = keras.layers.core.Dense(units=1)
    model_output = model_layer_dense1(model_intermediate_final_pooling_result)
    model_layer_dense1.set_weights([motif_params_dict["weights1_in_correct_shape"], motif_params_dict["biases1"]])

## 3.4. stitch the whole model
    
model = keras.models.Model(inputs=[model_input], outputs=model_output)

## 4. generate the benchmark dataset
## 4.1. just make the prediction

input_prediction_vector = model.predict(input_sequence_padded_with_one_quarter_on_both_sides_tensor, verbose=1).flatten()

temp_real_dataset_hdf5_filename = "../data/for.1..2.3.temp_real_dataset_for_motif_" + motif_id_str + ".hdf5"
temp_real_dataset_hdf5_filehandle = h5py.File(temp_real_dataset_hdf5_filename, "w")
temp_real_dataset_hdf5_filehandle.create_dataset(name="all_sample_index_int_vector", data=numpy.arange(0, input_sequence_padded_with_one_quarter_on_both_sides_tensor.shape[0]))
temp_real_dataset_hdf5_filehandle.create_dataset(name="all_prediction_float_vector", data=input_prediction_vector)
temp_real_dataset_hdf5_filehandle.close()
