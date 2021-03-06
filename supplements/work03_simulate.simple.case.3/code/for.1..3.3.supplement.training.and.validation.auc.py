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

# 1. build the model

def single_supplement_for_single_PWM_function(PWM_name):

    # 1. read the datasets
    
    PWM_dataset_prefix = "../data/for.1..2.sequence.dataset."
    PWM_dataset_filename = PWM_dataset_prefix + "__" + PWM_name + ".hdf5"
    PWM_dataset_filehandle = h5py.File(PWM_dataset_filename, mode="r")
    kernel_length_int = PWM_dataset_filehandle["PWM_normalized_matrix"].shape[1]
    X_for_training_and_validation_tensor = PWM_dataset_filehandle["X_for_training_and_validation_tensor"][:, :, :]
    y_for_training_and_validation_vector = PWM_dataset_filehandle["y_for_training_and_validation_vector"][:]
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
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    #model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=["accuracy"])
    
    # 3. load the model

    modelsave_output_filename = "../data/for.1..3.trained.model.modelsave.for." + PWM_name + ".hdf5"
    model.load_weights(modelsave_output_filename)


    # 4. get the auc values

    training_and_validation_auc_float = sklearn.metrics.roc_auc_score(y_true=y_for_training_and_validation_vector, y_score=model.predict(X_for_training_and_validation_tensor))
    
    return PWM_name, [training_and_validation_auc_float], 


parallel_object_Parallel = joblib.Parallel(n_jobs=20, backend="multiprocessing")
result_list = parallel_object_Parallel(map(joblib.delayed(single_supplement_for_single_PWM_function), all_PWMs_name_list ))

unlisted_result_list = [[temp_record[0]] + temp_record[1] for temp_record in result_list]
PWM_name_and_training_and_validation_auc_DataFrame = pandas.DataFrame(unlisted_result_list, columns=["PWM.name", "training.and.validation.auc"])

PWM_name_and_training_and_validation_auc_DataFrame.to_csv(path_or_buf="../data/for.1..3.3.PWM.name.and.training.and.validation.auc.txt", sep="\t", header=True)
