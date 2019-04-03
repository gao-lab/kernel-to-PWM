import numpy
import pandas
import keras

## filter_length_axis : 0
## ACGT_axis : 1
## filter_num_axis : 2

def transform(convolution_W_tensor, convolution_b_vector=numpy.nan, conv_direction="forward", ln_of_base_of_logarithm=1):

    """
    Transform convolutional kernel into PFM.

    Args:
        convolution_W_tensor : the tensor to transform. Must be in shape of [filter_length, ACGT_axis (=4), filter_num].
        convolution_b_vector : the bias vector for the kernels. Must be in shape of [filter_num] or numpy.nan (default). If set to numpy.nan, it is assumed that the biases for all kernels are all zero.
        conv_direction : the direction of convolution used by the original CNN model. Defaults to "forward", which means that the "convolution" here is in fact the cross-correlation; i.e., kernels are not flipped along the kernel length direction before carrying out the elementwise multiplication. If set to "reverse", then the "convolution" here means the canonical convolution where this flipping is required.
        ln_of_base_of_logarithm    : the ln of base of logarithm used for transformation. Defaults to numpy.exp(1). This must be larger than 0. A base that is too big might lead to numerical overflow (producing numpy.inf) and thus lose information; similarly, a base that is too small (i.e., too negative) might lead to numerical underflow (producing 0.250000000...) and also lose some information.
    Values:
        Returns a pair (exp_PFM_tensor, exp_PFM_ReLU_threshold_log_probability_vector).
        exp_PFM_tensor : the PFM tensor transformed, with shape identical to that of convolution_W_tensor.
        constant_matrix : the corresponding constant matrix for each kernel. Must be in the shape of [filter_num, filter_length]. Specifically, `constant_matrix[i, j]` is the constant value in `conv(X, j-th-position-of-i-th-kernel) = log(Prob(X|P(j-th-position-of-i-th-kernel))) + constant`, and `constant_matrix[i, :].sum()` is the constant value in `conv(X, i-th-kernel) = log(Prob(X|P(i-th-kernel))) + constant`. This is useful when one wants to recover the convolution value from the log probabilities.
    """

    filter_length = convolution_W_tensor.shape[0]
    filter_num = convolution_W_tensor.shape[2]

    exp_PFM_tensor = numpy.zeros_like(convolution_W_tensor)
    constant_matrix = numpy.zeros([filter_num, filter_length])

    if convolution_b_vector is numpy.nan:
        convolution_b_vector = numpy.zeros(filter_num)

    for temp_kernel_index in numpy.arange(0, filter_num):
        temp_kernel_W_matrix = convolution_W_tensor[:, :, temp_kernel_index]
        temp_kernel_b_scalar = convolution_b_vector[temp_kernel_index]
        temp_kernel_W_bias_corrected_matrix = temp_kernel_W_matrix + temp_kernel_b_scalar / (filter_length + 0.0)
        temp_C_matrix = ""
        if conv_direction == "forward":
            temp_C_matrix = numpy.exp(ln_of_base_of_logarithm * temp_kernel_W_bias_corrected_matrix)
        elif conv_direction == "reverse":
            temp_kernel_W_bias_corrected_and_flipped_matrix = numpy.flipud(temp_kernel_W_bias_corrected_matrix)
            temp_C_matrix = numpy.exp(ln_of_base_of_logarithm * temp_kernel_W_bias_corrected_and_flipped_matrix)
        else:
            raise ValueError("The parameter`conv_direction` must be either 'forward' or 'reverse'.")
        ## here inf (infinity) might be present; if this is the case, then for each position, replace all inf's with 1 and all other numbers with 1e-10 (the choice of this infinitesimal value is arbitrary for now)
        temp_C_corrected_matrix = numpy.apply_along_axis(__correct_single_row_of_C_function , axis=1, arr=temp_C_matrix)
        temp_constants_vector =  numpy.log(temp_C_corrected_matrix.sum(axis=1)) / ln_of_base_of_logarithm
        temp_P_matrix = numpy.apply_along_axis(lambda temp_row_vector: temp_row_vector / numpy.sum(temp_row_vector), axis=1, arr=temp_C_corrected_matrix)
        ## here 0 might be present; replace all of them with 1e-10 (the choice of this infinitesimal value is arbitrary for now)
        temp_P_corrected_matrix = temp_P_matrix
        temp_P_corrected_matrix[temp_P_corrected_matrix == 0.0] = 1e-10
        exp_PFM_tensor[:, :, temp_kernel_index] = temp_P_corrected_matrix
        constant_matrix[temp_kernel_index, :] = temp_constants_vector

    return exp_PFM_tensor, constant_matrix


def __correct_single_row_of_C_function(temp_row_vector):
    if numpy.any(numpy.isinf(temp_row_vector)) == False:
        return temp_row_vector
    else:
        temp_new_row_vector = numpy.ones_like(temp_row_vector) * 1.0e-10
        temp_new_row_vector[numpy.isinf(temp_row_vector)] = 1.0
        return temp_new_row_vector


def transform_Deepbind(input_tensor, convolution_W_tensor, convolution_b_vector=numpy.nan, conv_direction="forward", activation="linear", threshold=0):

    """
    Transform convolutional kernel into PFM, using Deepbind's heuristic.

    Args:
        input_tensor : the input sequence tensor used for the transformation (Deepbind's heuristic requires such input tensor). Must be in shape of [input_num / batch, input_length, ACGT_axis(=4)]. N's are allowed, provided that they are expressed as [0, 0, 0, 0]; they will NOT contribute to the calculation of PWM.
        convolution_W_tensor_list : a list of tensors to transform. Must be in shape of [filter_length, ACGT_axis (=4), filter_num].
        convolution_b_vector : the bias vector for the kernels. Must be in shape of [filter_num] or numpy.nan (default). If set to numpy.nan, it is assumed that the biases for all kernels are all zero.
        conv_direction : the direction of convolution used by the original CNN model. Defaults to "forward", which means that the "convolution" here is in fact the cross-correlation; i.e., kernels are not flipped along the kernel length direction before carrying out the elementwise multiplication. If set to "reverse", then the "convolution" here means the canonical convolution where this flipping is required.
        activation : the activation used for the convolution. Defaults to "linear". Valid values are "linear" and "relu".
        threshold : the threshold used to select input sequences for PFM calculation. Only those sequences whose convolutional value (after applying the activation) LARGER THAN the threshold will be used for PFM calculation. Defaults to 0 (a popular choice for ReLU).
    Values:
        Returns the transformed PFM, with shape identical to that of convolution_W_tensor.
    """

    input_num = input_tensor.shape[0]
    input_length = input_tensor.shape[1]
    filter_length = convolution_W_tensor.shape[0]
    filter_num = convolution_W_tensor.shape[2]

    if (convolution_b_vector is numpy.nan):
        convolution_b_vector = numpy.zeros(filter_num)

    ## STEP 1. get the max and argmax values for each input sequence

    model_input = keras.layers.Input(shape=[input_length, 4], dtype="float32")
    model_layer_Conv1D =  keras.layers.convolutional.Conv1D(
        input_shape=[input_length, 4],
        filters=filter_num,
        kernel_size=filter_length,
        padding='valid',
        activation=activation
    )
    #### create layer for max value
    model_max_result = keras.layers.pooling.GlobalMaxPooling1D()(model_layer_Conv1D(model_input))
    #### create layer for argmax value
    model_layer_argmax_Lambda = keras.layers.Lambda(function=lambda single_convolution_result : keras.backend.argmax(single_convolution_result, axis=1), output_shape=(None, filter_num))
    model_argmax_result = model_layer_argmax_Lambda(model_layer_Conv1D(model_input))
    #### concatenate the output layer
    model_max_and_argmax_model = keras.models.Model(inputs=[model_input], outputs=[model_max_result, model_argmax_result])
    #### set the conv weights
    if conv_direction == "forward":
        model_layer_Conv1D.set_weights([convolution_W_tensor, convolution_b_vector])
    elif conv_direction == "reverse":
        model_layer_Conv1D.set_weights([convolution_W_tensor[::-1, :, :], convolution_b_vector])
    else:
        raise ValueError("The parameter`conv_direction` must be either 'forward' or 'reverse'.")
    #### get max and argmax values for each input sequence
    max_result_matrix, argmax_result_matrix = model_max_and_argmax_model.predict(input_tensor, verbose=1)

    ## STEP 2. pick all sequences passing the threshold
    analysis_result_list_for_all_sequences_list = []

    for sample_index in numpy.arange(0, input_num):
        for kernel_index in numpy.arange(0, filter_num):
            single_max_float = max_result_matrix[sample_index, kernel_index]
            single_argmax_int = argmax_result_matrix[sample_index, kernel_index]
            single_passing_threshold = True
            if single_max_float <= threshold:
                single_passing_threshold = False
            analysis_result_list_for_all_sequences_list.append([sample_index, kernel_index, single_max_float, single_argmax_int, single_passing_threshold])

    analysis_result_for_all_sequences_DataFrame = pandas.DataFrame(analysis_result_list_for_all_sequences_list)
    analysis_result_for_all_sequences_DataFrame.columns = ['sample.index', 'kernel.index', 'final.max', 'final.argmax', 'passing.threshold']
    analysis_valid_result_for_all_sequences_DataFrame = analysis_result_for_all_sequences_DataFrame[analysis_result_for_all_sequences_DataFrame["passing.threshold"] == True]


    if set(analysis_valid_result_for_all_sequences_DataFrame["kernel.index"]) != set(numpy.arange(0, filter_num)):
        ## give up running
        raise ValueError("Kernel incomplete: lacking kernel(s)" + repr(set(numpy.arange(0, filter_num)) - set(analysis_valid_result_for_all_sequences_DataFrame["kernel.index"])))

    ## STEP 3. stack the argmax sequence fragments and get the final PFM

    def __add_sequence_fragment_to_PCM_for_single_sample_and_single_kernel(row_DataFrame):
        sample_index_int = row_DataFrame["sample.index"]
        kernel_index_int = row_DataFrame["kernel.index"]
        final_argmax_int = row_DataFrame["final.argmax"]
        if final_argmax_int > input_length - 1 - filter_length + 1:
            print(datetime.datetime.now().strftime("%H-%M-%S") + ": met sample " + str(sample_index_int) + " and kernel " + str(kernel_index_int) + " with final argmax " + str(final_argmax_int) + ", which is too large to let the sequence cover the kernel completely")
        sequence_fragment_matrix = input_tensor[sample_index_int, final_argmax_int:(final_argmax_int + filter_length - 1 + 1), :]
        PCM_tensor[:, :, kernel_index_int] = PCM_tensor[:, :, kernel_index_int] + sequence_fragment_matrix
        return None

    PCM_tensor = numpy.zeros([filter_length, 4, filter_num]) + 1e-5
    not_used_variable = analysis_valid_result_for_all_sequences_DataFrame.apply(__add_sequence_fragment_to_PCM_for_single_sample_and_single_kernel, axis=1)
    Deepbind_PFM_tensor = PCM_tensor / PCM_tensor.sum(axis=1).reshape([filter_length, 1, filter_num])
    return Deepbind_PFM_tensor


def _______get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_matrix, single_PFM_matrix, use_threshold=False, threshold=numpy.inf * -1, return_sequence=False):

    """
    Calculate the max value of log probabilities across all possible gap-free alignments of the PFM on the input sequence. The PFM must not be longer than the input sequence, and each alignment of the PFM on the sequence can have a probability.
    Args:
        single_input_sequence_matrix : the single input sequence matrix. Must be in the shape [input_length, 4]. N's are allowed, provided that they are expressed as [0, 0, 0, 0]; they will not be included in probability calculation. If all the sequence is composed of N, and the threshold is not set to a finite number, the resulting probability will be numpy.inf * -1.
        single_PFM_matrix : the single PFM matrix. Must be in the shape [filter_length, 4]
        use_threshold : whether to thresholding the resulting log probabilities. Defaults to False.
        threshold : if use_threshold == True, the value of threshold. Log probabilities below this threshold will be replaced with this threshold. Defaults to numpy.inf * -1 (i.e., no thresholding is done in practice).
        return_sequence : Defaults to False. If set to True, the whole sequence of log probabilities will be returned.
    Value:
        A scalar denoting the max value of log probabilities.
    NOTE:
        This function will NOT check whether the columnwise sums of the PWM matrix all are 1, and whether each element is inbetween 0 and 1.
    """

    filter_length = single_PFM_matrix.shape[0]
    input_length = single_input_sequence_matrix.shape[0]

    single_log_PFM_matrix = numpy.log(single_PFM_matrix)
    log_probability_for_all_alignment_float_list = []

    for alignment_start_position_in_sequence_int in range(0, (single_input_sequence_matrix.shape[0] - 1 - filter_length + 1) + 1):
        log_probability_for_current_alignment_float = 0.0
        for relative_position_in_PFM_int in range(0, filter_length):
            if single_input_sequence_matrix[alignment_start_position_in_sequence_int + relative_position_in_PFM_int, 0] == 0.25:
                ## 0.25 region
                log_probability_for_current_alignment_float = log_probability_for_current_alignment_float + 0.25 * numpy.sum(single_log_PFM_matrix[relative_position_in_PFM_int, :])
            else:
                ## real sequence region (one hot-encoded)
                is_one_bool_vector = (single_input_sequence_matrix[alignment_start_position_in_sequence_int + relative_position_in_PFM_int, :] == 1.0)
                if is_one_bool_vector.any() == True: ## meet A, C, G, or T, not N
                    log_probability_for_current_alignment_float = log_probability_for_current_alignment_float + single_log_PFM_matrix[relative_position_in_PFM_int, is_one_bool_vector]
                else: ## meet N
                    pass ## will not use this position to calculate the probability
        if log_probability_for_current_alignment_float == 0.0: ## all the positions are N's
            log_probability_for_current_alignment_float = numpy.inf * -1
        if use_threshold:
            log_probability_for_current_alignment_float = max(threshold, log_probability_for_current_alignment_float) ## should not use numpy.max here, because the syntax is not correct
        log_probability_for_all_alignment_float_list.append(log_probability_for_current_alignment_float)

    if return_sequence == True:
        return log_probability_for_all_alignment_float_list
    else:
        return numpy.max(log_probability_for_all_alignment_float_list)




def ___________get_max_log_probs_for_sequences_and_kernels(input_sequence_tensor, PFM_tensor, use_threshold=False, threshold_vector=numpy.nan):
    """
    Iterate over all possible pairwise combination of input sequence and PFM, and for each pair of <input sequence, PFM>, calculate the max value of log probabilities across all possible gap-free alignments of the PFM on the input sequence. The PFM must not be longer than the input sequence, and each alignment of the PFM on the sequence can have a probability.
    Args:
        input_sequence_tensor : the input sequence tensor. Must be in the shape [input_num, input_length, 4]. N's are allowed, provided that they are expressed as [0, 0, 0, 0]; they will not be included in probability calculation.
        PFM_tensor : the PFM tensor. Must be in the shape [filter_length, 4, filter_num]
        use_threshold : whether to thresholding the resulting log probabilities. Defaults to False.
        threshold_vector : if use_threshold == True, the value of threshold for each PFM. Log probabilities below this threshold will be replaced with this threshold. Must be in the shape [filter_num] or be numpy.nan (default).
    Value:
        A matrix of shape [input_num, filter_num] describing the maximal log probability of each combination of input sequence and PFM.
    """
    input_num = input_sequence_tensor.shape[0]
    filter_num = PFM_tensor.shape[2]
    result_matrix = numpy.zeros([input_num, filter_num])

    if (threshold_vector is numpy.nan):
        threshold_vector = numpy.asarray([numpy.nan] * filter_num)

    for filter_index in numpy.arange(0, filter_num):
        threshold = threshold_vector[filter_index]
        single_PFM_matrix = PFM_tensor[:, :, filter_index]
        for input_index in numpy.arange(0, input_num):
            single_input_sequence_matrix = input_sequence_tensor[input_index, :, :]
            max_log_prob = get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_matrix, single_PFM_matrix, use_threshold, threshold, return_sequence=False)
            result_matrix[input_index, filter_index] = max_log_prob

    return result_matrix



def run_transform_test():
    print("""
    kernel_1 = [
        [numpy.log(0.01), numpy.log(0.03), numpy.log(0.05), numpy.log(0.91)],
        [numpy.log(10), numpy.log(20), numpy.log(30), numpy.log(40)]
    ]
    kernel_2 = [
        [numpy.log(100), numpy.log(300), numpy.log(500), numpy.log(9100)],
        [numpy.log(0.01), numpy.log(0.02), numpy.log(0.03), numpy.log(0.04)]
    ]

    convolution_W_tensor = numpy.stack([kernel_1, kernel_2], axis=2)
    convolution_b_vector = numpy.asarray([0, 1000.0])
    """)
    print("=========Resulting convolution_W_tensor==========")
    kernel_1 = [
        [numpy.log(0.01), numpy.log(0.03), numpy.log(0.05), numpy.log(0.91)],
        [numpy.log(10), numpy.log(20), numpy.log(30), numpy.log(40)]
    ]
    kernel_2 = [
        [numpy.log(100), numpy.log(300), numpy.log(500), numpy.log(9100)],
        [numpy.log(0.01), numpy.log(0.02), numpy.log(0.03), numpy.log(0.04)]
    ]
    convolution_W_tensor = numpy.stack([kernel_1, kernel_2], axis=2)
    convolution_b_vector = numpy.asarray([0, 1000.0])
    print(convolution_W_tensor)
    print("========Expected output (conv_direction == 'forward'): for each PWM, the first position should be 0.01~0.03~0.05~0.91, and the second 0.1~0.2~0.3~0.4. Bias should not affect the PWM but only affect the threshold.===============")
    result = transform(convolution_W_tensor, convolution_b_vector=convolution_b_vector, conv_direction="forward", ln_of_base_of_logarithm=1)
    print("=======Actual output========")
    print("Result shape: ", repr([result[0].shape, result[1].shape]))
    print("PWM for first kernel: ", repr(result[0][:, :, 0]))
    print("PWM for second kernel: ", repr(result[0][:, :, 0]))
    print("Bias vector: ", repr(result[1]))
    print("========Expected output (conv_direction == 'reverse'): same as above, except that the first and second positions are switched==========")
    result2 = transform(convolution_W_tensor, convolution_b_vector=convolution_b_vector, conv_direction="reverse", ln_of_base_of_logarithm=1)
    print("=======Actual output========")
    print("Result shape: ", repr([result2[0].shape, result2[1].shape]))
    print("PWM for first kernel: ", repr(result2[0][:, :, 0]))
    print("PWM for second kernel: ", repr(result2[0][:, :, 0]))
    print("Bias vector: ", repr(result2[1]))


def run_Deepbind_transform_test():
    print("""
    kernel_1 = [
        [1, 100, 1, 1],
        [100, 1, 1, 1] ## best sequence match: CA (note that convolution needs reversing before multiplication)
    ] * 1.0
    kernel_2 = [
        [-100, -1, -100, -100],
        [-1, -100  -100, -100] ## best sequence match: CA, but will always be negative
    ] * 1.0

    convolution_W_tensor = numpy.stack([kernel_1, kernel_2], axis=2)
    convolution_b_vector = numpy.asarray([0, 0])
    """)
    print("=========Resulting convolution_W_tensor==========")
    input_seq_1 = [
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ] ## TAC
    input_seq_2 = [
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ] ## TCG
    kernel_1 = [
        [100, 1, 1, 90],
        [1, 100, 1, 1] ## best sequence match: AC (note that convolution needs reversing before multiplication). Second best match: TC
    ]
    kernel_2 = [
        [0.5, -1, -1, -0.6],
        [-1, 0.5, -1, -1] ## best sequence match: AC (note that convolution needs reversing before multiplication). Second best match: TC. But the second not pass the 0 threshold
    ]
    input_tensor = numpy.stack([input_seq_1, input_seq_2], axis=0)
    convolution_W_tensor = numpy.stack([kernel_1, kernel_2], axis=2)
    convolution_b_vector = numpy.asarray([0, 0])
    print(convolution_W_tensor)
    print("========Expected output (conv_direction == 'forward', activation=='linear', threshold==0)===============")
    print("""
    first PWM:
    array([[0.5, 0. , 0. , 0.5],
           [0. , 1. , 0. , 0. ]])
    second PWM:
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.]])
    """)
    result = transform_Deepbind(input_tensor, convolution_W_tensor, convolution_b_vector=numpy.nan, conv_direction="forward", activation="linear", threshold=0) ## activation= "linear" + threhsold =0 <=> activation = "relu" and threshold is numpy.inf * -1.0
    print("=========Actual output (rounding to 2 decimals is used here to simplify the view)=========")
    print("first PWM:")
    print(result[:, :, 0].round(2))
    print("second PWM:")
    print(result[:, :, 1].round(2))
    print("========Now we reverse the input sequence (i.e., use `input_tensor[:, ::-1, :]`), and set conv_direction to 'reverse'. This guarantees that the max values are the same, but the resulting PWMs are reversed version of those above.===========")
    result2 = transform_Deepbind(input_tensor[:, ::-1, :], convolution_W_tensor, convolution_b_vector=numpy.nan, conv_direction="reverse", activation="linear", threshold=0) ## activation= "linear" + threhsold =0 <=> activation = "relu" and threshold is numpy.inf * -1.0
    print("=========Actual output (rounding to 2 decimals is used here to simplify the view)=========")
    print("first PWM:")
    print(result2[:, :, 0].round(2))
    print("second PWM:")
    print(result2[:, :, 1].round(2))

def run_single_sequence_and_single_PWM_log_probability_test():
    single_input_sequence_matrix = numpy.asarray([
        [1, 0, 0, 0], ## max score position, -0.1 + -2
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])
    single_input_sequence_2_matrix = numpy.asarray([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1], ## max score position, -0.4 + 0.0
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ])
    single_input_sequence_3_matrix = numpy.asarray([
        [0.25, 0.25, 0.25, 0.25], ## max score position, -0.25 + -2
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]
    ])
    single_PWM_matrix = numpy.asarray([
        [numpy.exp(-0.1), numpy.exp(-0.2), numpy.exp(-0.3), numpy.exp(-0.4)], ## for 0.25-region, the log probability will be 0.25 * (-0.1 + -0.2 + -0.3 + -0.4) = -0.25
        [numpy.exp(-1), numpy.exp(-2), numpy.exp(-3), numpy.exp(-4)] ## for 0.25-region, the log probability will be -2.5
    ])
    get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_matrix, single_PWM_matrix, use_threshold=False, threshold=numpy.inf * -1, return_sequence=True)
    get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_2_matrix, single_PWM_matrix, use_threshold=False, threshold=numpy.inf * -1, return_sequence=True)
    get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_3_matrix, single_PWM_matrix, use_threshold=False, threshold=numpy.inf * -1, return_sequence=True)


def run_sequences_and_PWMs_log_probability_test():
    single_input_sequence_1_matrix = numpy.asarray([
        [1, 0, 0, 0], ## max score position for both kernels, -0.1 + -2 & -0.2 + -1
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]
    ])
    single_input_sequence_2_matrix = numpy.asarray([
        [0, 0, 0, 0],
        [0, 1, 0, 0],## max score position for second kernel, -0.2 + 0.0
        [0, 0, 1, 0],
        [0, 0, 0, 1], ## max score position for first kernel, -0.4 + 0.0
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ])
    single_input_sequence_3_matrix = numpy.asarray([
        [0.25, 0.25, 0.25, 0.25], ## max score position for first kernel, -0.25 + -2
        [0, 1, 0, 0], ## max score position for second kernel, -2 + -0.3
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0.25, 0.25, 0.25, 0.25]
    ])
    single_PWM_1_matrix = numpy.asarray([
        [numpy.exp(-0.1), numpy.exp(-0.2), numpy.exp(-0.3), numpy.exp(-0.4)], ## for 0.25-region, the log probability will be 0.25 * (-0.1 + -0.2 + -0.3 + -0.4) = -0.25
        [numpy.exp(-1), numpy.exp(-2), numpy.exp(-3), numpy.exp(-4)] ## for 0.25-region, the log probability will be -2.5
    ])
    single_PWM_2_matrix = numpy.asarray([
        [numpy.exp(-1), numpy.exp(-2), numpy.exp(-3), numpy.exp(-4)], ## for 0.25-region, the log probability will be -2.5
        [numpy.exp(-0.1), numpy.exp(-0.2), numpy.exp(-0.3), numpy.exp(-0.4)] ## for 0.25-region, the log probability will be 0.25 * (-0.1 + -0.2 + -0.3 + -0.4) = -0.25
    ])
    sequence_tensor = numpy.stack([single_input_sequence_1_matrix, single_input_sequence_2_matrix, single_input_sequence_3_matrix], axis=0)
    PWM_tensor = numpy.stack([single_PWM_1_matrix, single_PWM_2_matrix], axis=2)
    get_max_log_probs_for_sequences_and_kernels(sequence_tensor, PWM_tensor, use_threshold=True, threshold_vector=numpy.asarray([numpy.inf*-1] * 2))
    get_max_log_probs_for_sequences_and_kernels(sequence_tensor, PWM_tensor, use_threshold=True, threshold_vector=numpy.asarray([100,200])) ## will become the thresholds
