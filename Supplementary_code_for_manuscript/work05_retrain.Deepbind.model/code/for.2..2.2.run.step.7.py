## Summary of steps
## 7. calculate the log-likelihoods (the rest)
## Note that for our (exp) PFM, we made the following assumption to simplify the implementation:
##    The ReLU activation in the convolution layer + global max-pooling layer is equivalent to linear activation for all samples
## This can hardly be true for ALL samples, but is very likely to be true for most samples; the final comparison result will change little if we do not make such an assumption.

## Also, note that this step is quite time-consuming.
## Each run of this script takes 20 CPU cores to maximize its speed. Reducing the number of available CPU cores will make it slower.

import h5py
import numpy
import datetime
import joblib
import sys

motif_id_str = sys.argv[1]

temp_step_7_hdf5_filename = "../data/for.2..2.temp_step_7_for_motif_" + motif_id_str + ".hdf5"
temp_step_7_hdf5_filehandle = h5py.File(temp_step_7_hdf5_filename, "r")
detector_len_int = temp_step_7_hdf5_filehandle["length_of_detector"].value
input_sequence_padded_with_one_quarter_on_both_sides_tensor = temp_step_7_hdf5_filehandle["input_sequence_tensor"].value
exp_PFM_tensor = temp_step_7_hdf5_filehandle["exp_PFM_tensor"].value
Deepbind_PFM_tensor = temp_step_7_hdf5_filehandle["Deepbind_PFM_tensor"].value
valid_sample_index_int_vector = temp_step_7_hdf5_filehandle["valid_sample_index"].value
valid_kernel_index_int_vector = temp_step_7_hdf5_filehandle["valid_kernel_index"].value
exp_PFM_ReLU_threshold_log_probability_vector = temp_step_7_hdf5_filehandle["exp_PFM_ReLU_threshold_log_probability_vector"].value
temp_step_7_hdf5_filehandle.close()

def calculate_direct_log_probability_for_single_input_sequence_and_single_PFM_with_threshold(temp_single_input_sequence_matrix, temp_single_PFM_matrix, temp_threshold):  
    temp_single_log_PFM_matrix = numpy.log(temp_single_PFM_matrix)
    log_probability_for_all_alignment_float_list = []
    
    for temp_alignment_start_position_in_sequence_int in range(0, (temp_single_input_sequence_matrix.shape[0] - 1 - detector_len_int + 1) + 1):
        log_probability_for_current_alignment_float = 0.0
        for temp_relative_position_in_PFM_int in range(0, temp_single_PFM_matrix.shape[0]):
            if temp_single_input_sequence_matrix[temp_alignment_start_position_in_sequence_int + temp_relative_position_in_PFM_int, 0] == 0.25:
                log_probability_for_current_alignment_float = log_probability_for_current_alignment_float + 0.25 * numpy.sum(temp_single_log_PFM_matrix[temp_relative_position_in_PFM_int, :])
            else:
                temp_is_one_bool_vector = (temp_single_input_sequence_matrix[temp_alignment_start_position_in_sequence_int + temp_relative_position_in_PFM_int, :] == 1.0)
                log_probability_for_current_alignment_float = log_probability_for_current_alignment_float + temp_single_log_PFM_matrix[temp_relative_position_in_PFM_int, temp_is_one_bool_vector]
        if log_probability_for_current_alignment_float < temp_threshold:
            pass
            # log_probability_for_current_alignment_float = temp_threshold
        log_probability_for_all_alignment_float_list.append(log_probability_for_current_alignment_float)
        
    return numpy.max(log_probability_for_all_alignment_float_list)


def calculate_RC_log_probability_for_single_input_sequence_and_single_PFM_with_threshold(temp_single_input_sequence_matrix, temp_single_PFM_matrix, temp_threshold):
    temp_RC_single_input_sequence_matrix = numpy.fliplr(numpy.flipud(temp_single_input_sequence_matrix))
    return calculate_direct_log_probability_for_single_input_sequence_and_single_PFM_with_threshold(temp_RC_single_input_sequence_matrix, temp_single_PFM_matrix, temp_threshold)

def calculate_final_log_probability_for_single_input_sequence_and_single_PFM_with_threshold(temp_single_input_sequence_matrix, temp_single_PFM_matrix, temp_threshold):
    direct_log_probability_float = calculate_direct_log_probability_for_single_input_sequence_and_single_PFM_with_threshold(temp_single_input_sequence_matrix, temp_single_PFM_matrix, temp_threshold)
    RC_log_probability_float = calculate_RC_log_probability_for_single_input_sequence_and_single_PFM_with_threshold(temp_single_input_sequence_matrix, temp_single_PFM_matrix, temp_threshold)
    return numpy.max([direct_log_probability_float, RC_log_probability_float])


def calculate_exp_PFM_final_log_probability_for_each_row(temp_row_index_int):
    temp_single_input_sequence_index_int = valid_sample_index_int_vector[temp_row_index_int]
    temp_single_PFM_index_int = valid_kernel_index_int_vector[temp_row_index_int]
    if temp_single_input_sequence_index_int % 1000 == 0 :
        print(datetime.datetime.now().strftime("%H-%M-%S") + ": processing sample " + str(temp_single_input_sequence_index_int) + " and kernel " + str(temp_single_PFM_index_int))
    return calculate_final_log_probability_for_single_input_sequence_and_single_PFM_with_threshold(input_sequence_padded_with_one_quarter_on_both_sides_tensor[temp_single_input_sequence_index_int, :, :], exp_PFM_tensor[:, :, temp_single_PFM_index_int], exp_PFM_ReLU_threshold_log_probability_vector[temp_single_PFM_index_int])

def calculate_Deepbind_PFM_final_log_probability_for_each_row(temp_row_index_int):
    temp_single_input_sequence_index_int = valid_sample_index_int_vector[temp_row_index_int]
    temp_single_PFM_index_int = valid_kernel_index_int_vector[temp_row_index_int]
    if temp_single_input_sequence_index_int % 1000 == 0 :
        print(datetime.datetime.now().strftime("%H-%M-%S") + ": processing sample " + str(temp_single_input_sequence_index_int) + " and kernel " + str(temp_single_PFM_index_int))
    return calculate_final_log_probability_for_single_input_sequence_and_single_PFM_with_threshold(input_sequence_padded_with_one_quarter_on_both_sides_tensor[temp_single_input_sequence_index_int, :, :], Deepbind_PFM_tensor[:, :, temp_single_PFM_index_int], -1 * numpy.infty)


## parallelize the following code

exp_PFM_log_likelihood_calculation_result_list = joblib.Parallel(n_jobs=20)(joblib.delayed(calculate_exp_PFM_final_log_probability_for_each_row)(temp_row_index_int) for temp_row_index_int in range(0, len(valid_sample_index_int_vector)))
Deepbind_PFM_log_likelihood_calculation_result_list = joblib.Parallel(n_jobs=20)(joblib.delayed(calculate_Deepbind_PFM_final_log_probability_for_each_row)(temp_row_index_int) for temp_row_index_int in range(0, len(valid_sample_index_int_vector)))


temp_step_7_hdf5_output_filename = "../data/for.2..2.2.step.7.output.for.motif." + motif_id_str + ".hdf5"
temp_step_7_hdf5_output_filehandle = h5py.File(temp_step_7_hdf5_output_filename, "w")
temp_step_7_hdf5_output_filehandle.create_dataset(name="exp_PFM_log_likelihood", data=numpy.asarray(exp_PFM_log_likelihood_calculation_result_list))
temp_step_7_hdf5_output_filehandle.create_dataset(name="Deepbind_PFM_log_likelihood", data=numpy.asarray(Deepbind_PFM_log_likelihood_calculation_result_list))
temp_step_7_hdf5_output_filehandle.close()
