import numpy



def get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_matrix, single_PFM_matrix, use_threshold=False, threshold=numpy.inf * -1, return_sequence=False, use_conv_constant_shift=False, conv_constant_vector="Not a number"):

    """
    Calculate the max value of log probabilities across all possible gap-free alignments of the PFM on the input sequence. The PFM must not be longer than the input sequence, and each alignment of the PFM on the sequence can have a probability.
    Args:
        single_input_sequence_matrix : the single input sequence matrix. Must be in the shape [input_length, 4]. N's are allowed, provided that they are expressed as [0, 0, 0, 0]; they will not be included in probability calculation. If all the sequence is composed of N, and the threshold is not set to a finite number, the resulting probability will be numpy.inf * -1.
        single_PFM_matrix : the single PFM matrix. Must be in the shape [filter_length, 4]
        use_threshold : whether to thresholding the resulting log probabilities. Defaults to False.
        threshold : if use_threshold == True, the value of threshold. Log probabilities below this threshold will be replaced with this threshold. Defaults to numpy.inf * -1 (i.e., no thresholding is done in practice).
        return_sequence : Defaults to False. If set to True, the whole sequence of log probabilities will be returned.
        use_conv_constant_shift : Defaults to False. If set to True, the log probs will be shifted by (i.e., added with) the conv constants (those constants in the equation `conv result = log prob + constant`).
        conv_constant_vector : Defaults to "Not a number" (which makes the addition err if use_conv_constant_shift==True). The conv constant vector used for shifting the log probs. Must be in the shape of [filter_length]. Because the function knows nothing about convolution, the user is supposed to provide this constant here. The actual shift added to the log prob will be "sum of all non-N positions in this vector". This is applied before thresholding.
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
        conv_constant_total = 0.0
        for relative_position_in_PFM_int in range(0, filter_length):
            if single_input_sequence_matrix[alignment_start_position_in_sequence_int + relative_position_in_PFM_int, 0] == 0.25:
                ## 0.25 region
                log_probability_for_current_alignment_float = log_probability_for_current_alignment_float + 0.25 * numpy.sum(single_log_PFM_matrix[relative_position_in_PFM_int, :])
                if use_conv_constant_shift:
                    conv_constant_total = conv_constant_total + conv_constant_vector[relative_position_in_PFM_int]
            else:
                ## real sequence region (one hot-encoded)
                is_one_bool_vector = (single_input_sequence_matrix[alignment_start_position_in_sequence_int + relative_position_in_PFM_int, :] == 1.0)
                if is_one_bool_vector.any() == True: ## meet A, C, G, or T, not N
                    log_probability_for_current_alignment_float = log_probability_for_current_alignment_float + single_log_PFM_matrix[relative_position_in_PFM_int, is_one_bool_vector][0]
                    if use_conv_constant_shift:
                        conv_constant_total = conv_constant_total + conv_constant_vector[relative_position_in_PFM_int]
                else: ## meet N
                    pass ## will not use this position to calculate the probability
        if log_probability_for_current_alignment_float == 0.0: ## all the positions are N's
            log_probability_for_current_alignment_float = numpy.inf * -1
        if use_conv_constant_shift:
            log_probability_for_current_alignment_float = log_probability_for_current_alignment_float + conv_constant_total
        if use_threshold:
            log_probability_for_current_alignment_float = max(threshold, log_probability_for_current_alignment_float) ## should not use numpy.max here, because the syntax is not correct
        log_probability_for_all_alignment_float_list.append(log_probability_for_current_alignment_float)

    if return_sequence == True:
        return numpy.asarray(log_probability_for_all_alignment_float_list)
    else:
        return numpy.max(log_probability_for_all_alignment_float_list)




def get_max_log_probs_for_sequences_and_kernels(input_sequence_tensor, PFM_tensor, use_threshold=False, threshold_vector=numpy.nan, use_conv_constant_shift=False, conv_constant_matrix="Not a number"):
    """
    Iterate over all possible pairwise combination of input sequence and PFM, and for each pair of <input sequence, PFM>, calculate the max value of log probabilities across all possible gap-free alignments of the PFM on the input sequence. The PFM must not be longer than the input sequence, and each alignment of the PFM on the sequence can have a probability.
    Args:
        input_sequence_tensor : the input sequence tensor. Must be in the shape [input_num, input_length, 4]. N's are allowed, provided that they are expressed as [0, 0, 0, 0]; they will not be included in probability calculation.
        PFM_tensor : the PFM tensor. Must be in the shape [filter_length, 4, filter_num]
        use_threshold : whether to thresholding the resulting log probabilities. Defaults to False.
        threshold_vector : if use_threshold == True, the value of threshold for each PFM. Log probabilities below this threshold will be replaced with this threshold. Must be in the shape [filter_num] or be numpy.nan (default).
        use_conv_constant_shift : Defaults to False. If set to True, the log probs will be shifted by (i.e., added with) the conv constants (those constants in the equation `conv result = log prob + constant`).
        conv_constant_matrix : Defaults to "Not a number" (which makes the addition err if use_conv_constant_shift==True). The conv constant matrix used for shifting the log probs. Must be in the shape of [filter_num, filter_length]. Because the function knows nothing about convolution, the user is supposed to provide the constants here. Must be a vector of length `filter_num`. The actual shift added to the log prob on some sequence for i-th kernel will be all those i-th row elements in this matrix whose aligned nucleotides are not N. This is applied before thresholding.

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
        if use_conv_constant_shift == True:
            conv_constant_vector = conv_constant_matrix[filter_index, :]
        single_PFM_matrix = PFM_tensor[:, :, filter_index]
        for input_index in numpy.arange(0, input_num):
            single_input_sequence_matrix = input_sequence_tensor[input_index, :, :]
            max_log_prob = ""
            if use_conv_constant_shift == True:
                max_log_prob = get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_matrix, single_PFM_matrix, use_threshold, threshold, return_sequence=False, use_conv_constant_shift=True, conv_constant_vector=conv_constant_vector)
            else:
                max_log_prob = get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_matrix, single_PFM_matrix, use_threshold, threshold, return_sequence=False, use_conv_constant_shift=False)
            result_matrix[input_index, filter_index] = max_log_prob

    return result_matrix




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
