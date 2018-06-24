## Summary of steps
## 1. read the motif parameters
## 2. read the sequences
## 3. build the CNN model
## 4. generate the benchmarking dataset (moved to for.1..2.3.generate.real.dataset.with.smaller.base.py)
## 5. generate our (exp) PFM, with base chosen arbitrarily (not using MLE)
## 6. generate Deepbind PFM
## (start of 7. calculate log-likelihood) prepare for inputs for step 7 <- this step is time-consuming and should be separated from other steps

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

## 4. (skipped)

## 5. generate our PFM
## MLE function is closed temporarily

## 5.0. pick sequences passing the threshold defined by Deepbind
## two ways to understand sequence fragments skipped by this picking (i.e., "invalid" sequence fragments):
## (1) these sequence fragments are irrelevant to the estimation of the PFM; we just picked such sequence fragments from a random trial. In other words, the estimation is based on valid sequence fragments only.
## (2) these sequence fragments score too low to be included in the estimation of the PFM; however, such process of scoring should be considered during the estimation. In other words, the estimation is based on valid sequence fragments AND invalid sequence fragments AND all full sequences.

# conv_pool_output_model = keras.models.Model(inputs=[model_input], outputs=model_intermediate_final_pooling_result)
# conv_pool_result_matrix = conv_pool_output_model.predict(input_sequence_padded_with_one_quarter_on_both_sides_tensor, verbose=1)
# count_of_valid_sequence_fragment_for_each_kernel_vector = (conv_pool_result_matrix > 0).sum(axis=0)

## 5.1. determine ln(the optimal base of logarithm) by MLE

# note that invalid sequence fragments have a score of 0, and therefore including them into the summands does not affect the final result

# total_score = conv_pool_result_matrix.sum()
# convolution_W_tensor = model_layer_Conv1D.get_weights()[0]

# def MLL_loss_function(ln_b):
#     convolution_W_exponentiated_by_base_for_logarithm_tensor = numpy.exp(ln_b * convolution_W_tensor)
#     kernel_loss_term_vector = count_of_valid_sequence_fragment_for_each_kernel_vector * numpy.log(convolution_W_exponentiated_by_base_for_logarithm_tensor.sum(axis=(0, 1)))
#     total_loss_ln = -1 * (ln_b * total_score - kernel_loss_term_vector.sum())
#     return total_loss_ln

## TODO
## something is weird about the optimizer (L-BFGS-B and SLSQP give different results)
## please check whether the optimizer works properly
## optimization_result = scipy.optimize.minimize(MLL_loss_function, 1, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True}); optimization_result

# optimization_result = scipy.optimize.minimize(MLL_loss_function, 1, method='SLSQP', jac=False, bounds=[(0, None)], options={'xtol': 1e-8, 'disp': True})

# optimization_result = scipy.optimize.minimize(MLL_loss_function, 1, method='SLSQP', jac=False, bounds=[(0, 1000)], options={'xtol': 1e-8, 'disp': True}); optimization_result
# optimization_result = scipy.optimize.minimize(MLL_loss_function, 1, method='L-BFGS-B', jac=False, bounds=[(0, None)], options={'xtol': 1e-8, 'disp': True})
# optimization_result = scipy.optimize.minimize(MLL_loss_function, 1, method='L-BFGS-B', jac=False, bounds=[(1000, 2000)], options={'xtol': 1e-8, 'disp': True}); optimization_result

# ln_of_optimal_base = optimization_result["x"][0]

ln_of_optimal_base = 5.0

## 5.2. make the transformation
    
convolution_W_and_b_list = model_layer_Conv1D.get_weights()
convolution_W_tensor = convolution_W_and_b_list[0]
convolution_b_vector = convolution_W_and_b_list[1]
exp_PFM_tensor = numpy.zeros_like(convolution_W_tensor)


def __correct_single_row_of_C_function(temp_row_vector):
    if numpy.any(numpy.isinf(temp_row_vector)) == False:
        return temp_row_vector
    else:
        temp_new_row_vector = numpy.ones_like(temp_row_vector) * 1.0e-10
        temp_new_row_vector[numpy.isinf(temp_row_vector)] = 1.0
        return temp_new_row_vector

for temp_kernel_index in numpy.arange(0, motif_params_dict["num_detectors"]):
    temp_kernel_W_matrix = convolution_W_tensor[:, :, temp_kernel_index]
    temp_kernel_b_scalar = convolution_b_vector[temp_kernel_index]
    temp_kernel_W_bias_corrected_matrix = temp_kernel_W_matrix + temp_kernel_b_scalar / (motif_params_dict["detector_len"] + 0.0)
    temp_kernel_W_bias_corrected_and_flipped_matrix = numpy.flipud(temp_kernel_W_bias_corrected_matrix)
    temp_C_matrix = numpy.exp(ln_of_optimal_base * temp_kernel_W_bias_corrected_and_flipped_matrix)
    ## here inf (infinity) might be present; if this is the case, then for each position, replace all inf's with 1 and all other numbers with 1e-10 (the choice of this infinitesimal value is arbitrary for now)
    temp_C_corrected_matrix = numpy.apply_along_axis(__correct_single_row_of_C_function , axis=1, arr=temp_C_matrix)
    temp_P_matrix = numpy.apply_along_axis(lambda temp_row_vector: temp_row_vector / numpy.sum(temp_row_vector), axis=1, arr=temp_C_corrected_matrix)
    ## here 0 might be present; replace all of them with 1e-10 (the choice of this infinitesimal value is arbitrary for now)
    temp_P_corrected_matrix = temp_P_matrix
    temp_P_corrected_matrix[temp_P_corrected_matrix == 0.0] = 1e-10
    exp_PFM_tensor[:, :, temp_kernel_index] = temp_P_corrected_matrix

## 6. generate Deepbind PFM

convolution_W_and_b_list = model_layer_Conv1D.get_weights()
convolution_W_tensor = convolution_W_and_b_list[0]
Deepbind_PFM_tensor = numpy.zeros_like(convolution_W_tensor)

## here for models with or without RC, the computation of argmax is different
## average pooling is not supported

temp_model_layer_argmax_Lambda = keras.layers.Lambda(function=lambda temp_single_convolution_result : keras.backend.argmax(temp_single_convolution_result, axis=1), output_shape=(None, motif_params_dict["num_detectors"]))
temp_model_intermediate_direct_argmax_result = temp_model_layer_argmax_Lambda(model_layer_Conv1D(model_input))
temp_model_direct_max_and_argmax_model = keras.models.Model(inputs=[model_input], outputs=[model_intermediate_direct_pooling_result, temp_model_intermediate_direct_argmax_result])
temp_direct_max_result_matrix, temp_direct_argmax_result_matrix = temp_model_direct_max_and_argmax_model.predict(input_sequence_padded_with_one_quarter_on_both_sides_tensor, verbose=1)

## we only consider models with RC
## models without RC must have average pooling and thus cannot be processed

temp_model_intermediate_RC_argmax_result = temp_model_layer_argmax_Lambda(model_layer_Conv1D(model_layer_RC(model_input)))
temp_model_RC_max_and_argmax_model = keras.models.Model(inputs=[model_input], outputs=[model_intermediate_RC_pooling_result, temp_model_intermediate_RC_argmax_result])
temp_RC_max_result_matrix, temp_RC_argmax_result_matrix = temp_model_RC_max_and_argmax_model.predict(input_sequence_padded_with_one_quarter_on_both_sides_tensor, verbose=1)

analysis_result_list_for_all_sequences_list = []

for sample_index in numpy.arange(0, input_sequence_padded_with_one_quarter_on_both_sides_tensor.shape[0]):
    for kernel_index in numpy.arange(0, motif_params_dict["num_detectors"]):
        temp_single_direct_max_float = temp_direct_max_result_matrix[sample_index, kernel_index]
        temp_single_direct_argmax_int = temp_direct_argmax_result_matrix[sample_index, kernel_index]
        temp_single_RC_max_float = temp_RC_max_result_matrix[sample_index, kernel_index]
        temp_single_RC_argmax_int = temp_RC_argmax_result_matrix[sample_index, kernel_index]
        ## tie
        temp_single_larger_sequence_str = "tie"
        temp_single_final_max_float = temp_single_direct_max_float
        temp_single_final_argmax_in_original_direction_int = temp_single_direct_argmax_int
        if temp_single_direct_max_float > temp_single_RC_max_float:
            ## direct
            temp_single_larger_sequence_str = "direct"
        elif temp_single_direct_max_float < temp_single_RC_max_float:
            ## RC
            temp_single_larger_sequence_str = "RC"
            temp_single_final_max_float = temp_single_RC_max_float
            temp_single_final_argmax_in_original_direction_int = input_sequence_padded_with_one_quarter_on_both_sides_tensor.shape[1] - temp_single_RC_argmax_int - 1 - motif_params_dict["detector_len"] + 1

        temp_single_passing_ReLU = True
        if temp_single_final_max_float <= 0:
            temp_single_passing_ReLU = False
            
        analysis_result_list_for_all_sequences_list.append([sample_index, kernel_index, temp_single_direct_max_float, temp_single_RC_max_float, temp_single_direct_argmax_int, temp_single_RC_argmax_int, temp_single_larger_sequence_str, temp_single_final_max_float, temp_single_final_argmax_in_original_direction_int, temp_single_passing_ReLU])

analysis_result_for_all_sequences_DataFrame = pandas.DataFrame(analysis_result_list_for_all_sequences_list)
analysis_result_for_all_sequences_DataFrame.columns = ['sample.index', 'kernel.index', 'direct.max', 'RC.max', 'direct.argmax', 'RC.argmax', 'comparison', 'final.max', 'final.argmax', 'passing.ReLU']

analysis_valid_result_for_all_sequences_DataFrame = analysis_result_for_all_sequences_DataFrame[analysis_result_for_all_sequences_DataFrame["passing.ReLU"] == True]


def add_sequence_fragment_to_PCM_for_single_sample_and_single_kernel(temp_row_DataFrame):
    temp_sample_index_int = temp_row_DataFrame["sample.index"]
    temp_kernel_index_int = temp_row_DataFrame["kernel.index"]
    temp_final_argmax_int = temp_row_DataFrame["final.argmax"]
    if temp_final_argmax_int > input_sequence_padded_with_one_quarter_on_both_sides_tensor.shape[1] - 1 - motif_params_dict["detector_len"] + 1:
        print(datetime.datetime.now().strftime("%H-%M-%S") + ": struck with sample " + str(temp_sample_index_int) + " and kernel " + str(temp_kernel_index_int) + " with final argmax " + str(temp_final_argmax_int))        
    temp_sequence_fragment_matrix = input_sequence_padded_with_one_quarter_on_both_sides_tensor[temp_sample_index_int, temp_final_argmax_int:(temp_final_argmax_int + motif_params_dict["detector_len"] - 1 + 1), :]
    temp_PCM_tensor[:, :, temp_kernel_index_int] = temp_PCM_tensor[:, :, temp_kernel_index_int] + temp_sequence_fragment_matrix
    if temp_sample_index_int % 10000 == 0 :
        print(datetime.datetime.now().strftime("%H-%M-%S") + ": processing sample " + str(temp_sample_index_int) + " and kernel " + str(temp_kernel_index_int))
    return None

temp_PCM_tensor = numpy.zeros([motif_params_dict["detector_len"], 4, motif_params_dict["num_detectors"]]) + 1e-5
not_used_variable = analysis_valid_result_for_all_sequences_DataFrame.apply(add_sequence_fragment_to_PCM_for_single_sample_and_single_kernel, axis=1)

Deepbind_PFM_tensor = temp_PCM_tensor / temp_PCM_tensor.sum(axis=1).reshape([motif_params_dict["detector_len"], 1, motif_params_dict["num_detectors"]])


## 7. calculate log-likelihood
## basic rules:
## (0) given the input sequence and the log form of PFM
## (1) for each position of the input sequence
##   (1.1) set up the alignment between the input sequence and the kernel, and calculate the direct probability by going through each kernel position of the alignment
##     (1.1.1) if the current position is four 0.25's: pick all 4 log-probabilities from the corresponding position of PFM, multiply them by 0.25, and add them to the total score
##     (1.1.2) else, the current position must be one hot-encoded, and pick and add to the total score the corresponding log-probability from the corresponding position of PFM
##     (1.1.3) at the end of the looping, assign the total score to the direct probability
## (2) get the maximum of the direct probability across all alignments
## (3) reverse-complementing the sequence and repeat (1) and (2) to get the maximum of the reverse complementary (RC) probability
## (4) pick the larger one from the two maxima and save it for the pair (the input sequence, the kernel)

## currently the calculation does not support PFMs with 0; all the previous PFMs have been transformed in a way such that all 0's are replaced with infinitesimal values (here we use 1.0e-10)
## this calculation does not consider those sequences not passing the ReLU activation, because it is unknown how Deepbind PFM is supposed to calculate the log-likelihood for such sequences; for the performance comparison we will only consider those sequences that pass the ReLU activation

temp_step_7_hdf5_filename = "../data/for.1..2.temp_step_7_for_motif_" + motif_id_str + ".hdf5"
temp_step_7_hdf5_filehandle = h5py.File(temp_step_7_hdf5_filename, "w")
temp_step_7_hdf5_filehandle.create_dataset(name="length_of_detector", data=motif_params_dict["detector_len"])
temp_step_7_hdf5_filehandle.create_dataset(name="input_sequence_tensor", data=input_sequence_padded_with_one_quarter_on_both_sides_tensor)
temp_step_7_hdf5_filehandle.create_dataset(name="exp_PFM_tensor", data=exp_PFM_tensor)
temp_step_7_hdf5_filehandle.create_dataset(name="Deepbind_PFM_tensor", data=Deepbind_PFM_tensor)
temp_step_7_hdf5_filehandle.create_dataset(name="valid_sample_index", data=analysis_valid_result_for_all_sequences_DataFrame["sample.index"])
temp_step_7_hdf5_filehandle.create_dataset(name="valid_kernel_index", data=analysis_valid_result_for_all_sequences_DataFrame["kernel.index"])
temp_step_7_hdf5_filehandle.close()
