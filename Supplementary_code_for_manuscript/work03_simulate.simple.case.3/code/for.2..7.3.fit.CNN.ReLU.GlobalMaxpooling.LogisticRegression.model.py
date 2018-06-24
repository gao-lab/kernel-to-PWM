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
import os
import stat
import sys

# 1. read CM information

CM_name_str, CM_ACC_str = sys.argv[1:3]
sequence_length_int = int(sys.argv[3])
CM_dir_str = "../data/for.2..7.simulation." + "CM.accession__" + CM_ACC_str + ".CM.name__" + CM_name_str +  ".sequence.length__" + str(sequence_length_int) + "/"

# 2. read the datasets

CM_dataset_filename = CM_dir_str + "for.2..7.2.sequence.tensor.hdf5"
CM_dataset_filehandle = h5py.File(CM_dataset_filename, mode="r")
X_for_training_tensor = CM_dataset_filehandle["training_sequence_tensor"][:, :, :]
X_for_validation_tensor = CM_dataset_filehandle["validation_sequence_tensor"][:, :, :]
X_for_training_and_validation_tensor = CM_dataset_filehandle["training_and_validation_sequence_tensor"][:, :, :]
X_for_testing_tensor = CM_dataset_filehandle["testing_sequence_tensor"][:, :, :]
y_for_training_vector = CM_dataset_filehandle["training_y_vector"][:]
y_for_validation_vector = CM_dataset_filehandle["validation_y_vector"][:]
y_for_training_and_validation_vector = CM_dataset_filehandle["training_and_validation_y_vector"][:]
y_for_testing_vector = CM_dataset_filehandle["testing_y_vector"][:]
CM_dataset_filehandle.close()


# 3. build the model

training_random_seed = 789
kernel_length_int = 80
kernel_count_int = 5

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

# 4. train the model

modelsave_output_filename = CM_dir_str + "for.2..7.3.trained.model.modelsave.hdf5"
checkpointer_output_filename = CM_dir_str + "for.2..7.3.checkpointer.hdf5"
checkpointer = keras.callbacks.ModelCheckpoint(filepath=checkpointer_output_filename, verbose=2, save_best_only=True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(X_for_training_tensor, y_for_training_vector, batch_size=100, nb_epoch=60, shuffle=True, validation_data=(X_for_validation_tensor, y_for_validation_vector), verbose=2, callbacks=[checkpointer, earlystopper])
model.save_weights(modelsave_output_filename, overwrite=True)


training_auc_float = sklearn.metrics.roc_auc_score(y_true=y_for_training_vector, y_score=model.predict(X_for_training_tensor))
validation_auc_float = sklearn.metrics.roc_auc_score(y_true=y_for_validation_vector, y_score=model.predict(X_for_validation_tensor))
training_and_validation_auc_float = sklearn.metrics.roc_auc_score(y_true=y_for_training_and_validation_vector, y_score=model.predict(X_for_training_and_validation_tensor))
testing_auc_float = sklearn.metrics.roc_auc_score(y_true=y_for_testing_vector, y_score=model.predict(X_for_testing_tensor))

auc_filename_str = CM_dir_str + "for.2..7.3.auc.txt"

with open(auc_filename_str, "w") as auc_filehandle:
    auc_filehandle.write(str(training_auc_float) + "\t")
    auc_filehandle.write(str(validation_auc_float) + "\t")
    auc_filehandle.write(str(training_and_validation_auc_float) + "\t")
    auc_filehandle.write(str(testing_auc_float))
    auc_filehandle.write("\n")

