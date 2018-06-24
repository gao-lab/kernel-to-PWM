import pandas
import numpy
import h5py
import datetime
import sklearn.cross_validation
import keras
import itertools
import theano
import sklearn.metrics
import joblib

all_PWMs_name_list = []

with open("../data/for.1..2.all.PWMs.name.txt", "r") as all_PWMs_name_filehandle:
    for PWM_name in all_PWMs_name_filehandle:
        all_PWMs_name_list.append(PWM_name.rstrip())


def single_meme_generation_for_single_PWM_function(PWM_name):

    # 1. read the datasets
    
    PWM_dataset_prefix = "../data/for.1..2.sequence.dataset."
    PWM_dataset_filename = PWM_dataset_prefix + "__" + PWM_name + ".hdf5"
    PWM_dataset_filehandle = h5py.File(PWM_dataset_filename, mode="r")
    kernel_length_int = PWM_dataset_filehandle["PWM_normalized_matrix"].shape[1]
    PWM_normalized_matrix = PWM_dataset_filehandle["PWM_normalized_matrix"][:, :]
    X_for_training_tensor = PWM_dataset_filehandle["X_for_training_tensor"][:, :, :]
    X_for_validation_tensor = PWM_dataset_filehandle["X_for_validation_tensor"][:, :, :]
    y_for_training_vector = PWM_dataset_filehandle["y_for_training_vector"][:]
    y_for_validation_vector = PWM_dataset_filehandle["y_for_validation_vector"][:]
    PWM_dataset_filehandle.close()


    # 2. build the model

    training_random_seed = 789
    sequence_length_int = 50
    kernel_count_int = 1

    numpy.random.seed(training_random_seed)
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Convolution1D(
        input_dim=4,
        input_length=sequence_length_int,
        nb_filter=kernel_count_int,
        filter_length=kernel_length_int,
        border_mode='valid',
        activation='relu',
    ))
    model.add(keras.layers.convolutional.MaxPooling1D(
        pool_length=model.layers[0].output_shape[1]
    ))
    model.add(keras.layers.core.Flatten())
    model.add(keras.layers.core.Dense(output_dim=1))
    model.add(keras.layers.core.Activation('sigmoid'))
    modelsave_output_filename = "../data/for.1..3.trained.model.modelsave.for." + PWM_name + ".hdf5"
    model.load_weights(modelsave_output_filename)

    # 3. get the exp-transformed PWM

    exp_transformed_PWM_matrix_list = []
    
    temp_enlargment = 1.0
    temp_CNN_W_and_b_list = model.layers[0].get_weights()
    temp_CNN_W_tensor = temp_CNN_W_and_b_list[0]
    temp_CNN_b_vector = temp_CNN_W_and_b_list[1]
    temp_kernel_count = temp_CNN_W_tensor.shape[0]
    temp_kernel_length = temp_CNN_W_tensor.shape[2]
    to_return_transformed_CNN_W_tensor = numpy.zeros_like(temp_CNN_W_tensor)
    to_return_transformed_CNN_b_vector = numpy.zeros_like(temp_CNN_b_vector)
    
    for temp_kernel_index in numpy.arange(0, temp_kernel_count):
        temp_kernel_W_matrix = temp_CNN_W_tensor[temp_kernel_index, :, :, 0]
        temp_kernel_b_scalar = temp_CNN_b_vector[temp_kernel_index]
        temp_kernel_W_bias_corrected_matrix = temp_kernel_W_matrix + temp_kernel_b_scalar / (temp_kernel_length + 0.0)
        temp_kernel_W_bias_corrected_and_exponentiated_matrix = numpy.exp(temp_kernel_W_bias_corrected_matrix * temp_enlargment)
        temp_kernel_W_bias_corrected_and_exponentiated_and_normalized_matrix = numpy.apply_along_axis(lambda temp_column_vector: temp_column_vector / numpy.sum(temp_column_vector), axis=0, arr=temp_kernel_W_bias_corrected_and_exponentiated_matrix)
        exp_transformed_PWM_matrix = numpy.fliplr(temp_kernel_W_bias_corrected_and_exponentiated_and_normalized_matrix)
        exp_transformed_PWM_matrix_list.append(exp_transformed_PWM_matrix)
        
    # 4. get the Deepbind-transformed PWM

    Deepbind_transformed_PWM_matrix_list = []

    temp_CNN_W_and_b_list = model.layers[0].get_weights()
    temp_kernel_count = temp_CNN_W_and_b_list[0].shape[0]
    temp_kernel_length = temp_CNN_W_and_b_list[0].shape[2]
        
    temp_theano_input_symbol = model.input
    temp_theano_argmax_symbol = theano.tensor.argmax(x=model.layers[0].output, axis=1, keepdims=False)
    temp_theano_max_symbol = theano.tensor.max(x=model.layers[0].output, axis=1, keepdims=False)
    temp_get_argmax_and_max_from_CNN_GlobalMaxPooling_function = theano.function([temp_theano_input_symbol], [temp_theano_argmax_symbol, temp_theano_max_symbol], allow_input_downcast=True)
    
    temp_batch_argmax_list = []
    temp_batch_max_list = []
    temp_X = numpy.concatenate([X_for_training_tensor, X_for_validation_tensor], axis=0)
    temp_X_sample_size = temp_X.shape[0]
    temp_argmax_result_tensor, temp_max_result_tensor = temp_get_argmax_and_max_from_CNN_GlobalMaxPooling_function(temp_X)
        
    temp_PCM_weighted_tensor = numpy.zeros([temp_kernel_count, temp_kernel_length, 4]) +1e-5 # PCM stands for Position Count Matrix. The weight comes from the max score after max pooling.
    for temp_sample_index in range(0, temp_X_sample_size):
        for temp_kernel_index in range(0, temp_kernel_count):
            if (temp_max_result_tensor[temp_sample_index, temp_kernel_index] == 0): # here we choose to ignore ==0 samples, since their internal difference do not make any contribution to the downstream prediction, and thus their sequence fragments should not be taken into consideration when transforming the kernel into PWM
                #print("0 for sample index " + str(temp_sample_index) + " and kernel index " + str(temp_kernel_index))
                continue
            temp_position_index_for_max = temp_argmax_result_tensor[temp_sample_index, temp_kernel_index]
            temp_PCM_weighted_tensor[temp_kernel_index, :, :] = temp_PCM_weighted_tensor[temp_kernel_index, :, :] + temp_max_result_tensor[temp_sample_index, temp_kernel_index] * temp_X[temp_sample_index, temp_position_index_for_max:(temp_position_index_for_max + temp_kernel_length), :]
    temp_PWM_tensor = temp_PCM_weighted_tensor / (temp_PCM_weighted_tensor.sum(axis=2).reshape([temp_kernel_count, temp_kernel_length, 1]) )
    temp_PWM_dimension_swapped_tensor = numpy.transpose(a=temp_PWM_tensor, axes=[0, 2, 1])[:, :, :, None]

    for temp_kernel_index_int in range(0, temp_PWM_dimension_swapped_tensor.shape[0]):
        Deepbind_transformed_PWM_matrix_list.append(temp_PWM_dimension_swapped_tensor[temp_kernel_index_int, :, :, 0])

    # 5. write PWMs to the meme file
        
    all_transformed_PWM_matrix_list = [PWM_normalized_matrix] + exp_transformed_PWM_matrix_list + Deepbind_transformed_PWM_matrix_list
    all_transformed_PWM_name_list = ["original_normalized"] + ["exp_" + str(temp_index_int) for temp_index_int in range(0, len(exp_transformed_PWM_matrix_list))] + ["Deepbind_" + str(temp_index_int) for temp_index_int in range(0, len(Deepbind_transformed_PWM_matrix_list))]

    MEME_HEADER = "MEME version 4.4\n\nALPHABET= ACGT\n\nstrands: +\n\nBackground letter frequencies (from web form):\nA 0.25000 C 0.25000 G 0.25000 T 0.25000 \n\n"
    PWM_output_filename = "../data/for.1..4.meme.for." + PWM_name + ".meme.txt"
    with open(PWM_output_filename, 'w') as PWM_output_filehandle:
        PWM_output_filehandle.write(MEME_HEADER)
        for temp_PWM_name, temp_PWM_matrix in itertools.izip(all_transformed_PWM_name_list, all_transformed_PWM_matrix_list):
            PWM_output_filehandle.write('MOTIF ' + temp_PWM_name + '\n\n')
            PWM_output_filehandle.write('letter-probability matrix: alength= 4 w= ' + str(kernel_length_int) + ' nsites= 20' + ' E= 1e-6\n')
            for position_index in range(0, kernel_length_int):
                PWM_output_filehandle.write(str(temp_PWM_matrix[0, position_index]) + "\t" + str(temp_PWM_matrix[1, position_index]) + "\t" + str(temp_PWM_matrix[2, position_index]) + "\t" + str(temp_PWM_matrix[3, position_index]) + "\n")
            PWM_output_filehandle.write("\n")

    return PWM_name

            
parallel_object_Parallel = joblib.Parallel(n_jobs=20, backend="multiprocessing")
result = parallel_object_Parallel(map(joblib.delayed(single_meme_generation_for_single_PWM_function), all_PWMs_name_list ))
