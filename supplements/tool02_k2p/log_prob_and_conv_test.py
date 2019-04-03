from log_prob import *
from k2p import *
import keras

def run_log_prob_and_conv_comparison_test():
    single_input_sequence_matrix = numpy.asarray([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ])
    single_kernel_matrix = numpy.asarray([
       [-0.01, -0.02, -0.03, -0.04],
       [-0.1, -0.2, -0.3, -0.4],
       [-1.0, -2.0, -3.0, -4.0],
    ])
    single_PWM_tensor, constant_matrix = transform(single_kernel_matrix[:, :, None])
    single_PWM_matrix = single_PWM_tensor[:, :, 0]
    log_prob_list = get_max_log_prob_for_single_input_sequence_and_single_kernel(single_input_sequence_matrix, single_PWM_matrix, use_threshold=False, threshold=numpy.inf * -1, return_sequence=True)

    model_input = keras.layers.Input(shape=[single_input_sequence_matrix.shape[0], 4], dtype="float32")
    model_layer_Conv1D =  keras.layers.convolutional.Conv1D(
        input_shape=[single_input_sequence_matrix.shape[0], 4],
        filters=1,
        kernel_size=3,
        padding='valid',
        activation='linear'
    )
    model_output = model_layer_Conv1D(model_input)
    model = keras.models.Model(inputs=[model_input], outputs=[model_output])
    model_layer_Conv1D.set_weights([single_kernel_matrix[:, :, None], numpy.asarray([0.0])])


    conv_result = model.predict(single_input_sequence_matrix[None, :, :])[0, :, 0]

    conv_result - numpy.asarray(log_prob_list)
