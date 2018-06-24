## Summary of steps
## 8. re-train models and make predictions

import h5py
import numpy
import datetime
import sys
import keras
import theano
import pandas
import sklearn.model_selection

motif_id_str = sys.argv[1]


# 8. re-train models

# 8.1. load the datasets

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


temp_step_7_hdf5_filename = "../data/for.2..2.temp_step_7_for_motif_" + motif_id_str + ".hdf5"
temp_step_7_hdf5_filehandle = h5py.File(temp_step_7_hdf5_filename, "r")
valid_sample_index_int_vector = temp_step_7_hdf5_filehandle["valid_sample_index"].value
valid_kernel_index_int_vector = temp_step_7_hdf5_filehandle["valid_kernel_index"].value
temp_step_7_hdf5_filehandle.close()

temp_step_7_hdf5_output_filename = "../data/for.2..2.2.step.7.output.for.motif." + motif_id_str + ".hdf5"
temp_step_7_hdf5_output_filehandle = h5py.File(temp_step_7_hdf5_output_filename, "r")
exp_PFM_log_likelihood_calculation_result_float_vector = temp_step_7_hdf5_output_filehandle["exp_PFM_log_likelihood"].value
Deepbind_PFM_log_likelihood_calculation_result_float_vector = temp_step_7_hdf5_output_filehandle["Deepbind_PFM_log_likelihood"].value
temp_step_7_hdf5_output_filehandle.close()

temp_real_dataset_hdf5_filename = "../data/for.2..2.3.temp_real_dataset_for_motif_" + motif_id_str + ".hdf5"
temp_real_dataset_hdf5_filehandle = h5py.File(temp_real_dataset_hdf5_filename, "r")
all_sample_index_int_vector = temp_real_dataset_hdf5_filehandle["all_sample_index_int_vector"].value
all_prediction_float_vector = temp_real_dataset_hdf5_filehandle["all_prediction_float_vector"].value
temp_real_dataset_hdf5_filehandle.close()

# 8.2. reshape the datasets into trainable datasets

real_dataset_DataFrame = pandas.DataFrame({
    'sample.index' : all_sample_index_int_vector,
    'real.response' : all_prediction_float_vector
})
real_dataset_DataFrame.set_index("sample.index", inplace=True)

exp_PFM_dataset_long_format_DataFrame = pandas.DataFrame({
    'sample.index' : valid_sample_index_int_vector,
    'kernel.index' : valid_kernel_index_int_vector,
    'exp.PFM.log.likelihood' : exp_PFM_log_likelihood_calculation_result_float_vector
})
exp_PFM_dataset_wide_format_DataFrame = exp_PFM_dataset_long_format_DataFrame.pivot_table(index="sample.index", columns="kernel.index")
exp_PFM_dataset_valid_wide_format_DataFrame = exp_PFM_dataset_wide_format_DataFrame[exp_PFM_dataset_wide_format_DataFrame.isnull().any(axis=1) == False]
exp_PFM_dataset_sample_index_int_vector = exp_PFM_dataset_valid_wide_format_DataFrame.index.values
exp_PFM_dataset_X_float_matrix = numpy.asarray(exp_PFM_dataset_valid_wide_format_DataFrame)
exp_PFM_dataset_y_float_vector = numpy.asarray(real_dataset_DataFrame.ix[exp_PFM_dataset_sample_index_int_vector.tolist()])[:, 0]


Deepbind_PFM_dataset_long_format_DataFrame = pandas.DataFrame({
    'sample.index' : valid_sample_index_int_vector,
    'kernel.index' : valid_kernel_index_int_vector,
    'exp.PFM.log.likelihood' : Deepbind_PFM_log_likelihood_calculation_result_float_vector
})
Deepbind_PFM_dataset_wide_format_DataFrame = Deepbind_PFM_dataset_long_format_DataFrame.pivot_table(index="sample.index", columns="kernel.index")
Deepbind_PFM_dataset_valid_wide_format_DataFrame = Deepbind_PFM_dataset_wide_format_DataFrame[Deepbind_PFM_dataset_wide_format_DataFrame.isnull().any(axis=1) == False]
Deepbind_PFM_dataset_sample_index_int_vector = Deepbind_PFM_dataset_valid_wide_format_DataFrame.index.values
Deepbind_PFM_dataset_X_float_matrix = numpy.asarray(Deepbind_PFM_dataset_valid_wide_format_DataFrame)
Deepbind_PFM_dataset_y_float_vector = numpy.asarray(real_dataset_DataFrame.ix[Deepbind_PFM_dataset_sample_index_int_vector.tolist()])[:, 0]

## Here, exp_PFM_dataset_sample_index_int_vector and Deepbind_PFM_dataset_sample_index_int_vector must be identical, so we used the following way to split the datasets into 3
## Note also that the y vectors are also identical, making the two predictions comparable

numpy.random.seed(seed=123)
shuffled_sample_index_int_vector = numpy.random.choice(a=numpy.arange(0, exp_PFM_dataset_sample_index_int_vector.shape[0]), size=exp_PFM_dataset_sample_index_int_vector.shape[0], replace=False)
## ratio: training:validation:testing = 0.6:0.15:0.25
shuffled_training_sample_index_int_vector, shuffled_validation_sample_index_int_vector, shuffled_testing_sample_index_int_vector = numpy.split(ary=shuffled_sample_index_int_vector, indices_or_sections=[int(0.6 * shuffled_sample_index_int_vector.shape[0]), int(0.75 * shuffled_sample_index_int_vector.shape[0])])


## 8.3. build the model to train

numpy.random.seed(seed=123)
exp_PFM_model = keras.models.Sequential()
if motif_params_dict["num_hidden"] > 0:
    model_layer_dense1 = keras.layers.core.Dense(input_shape=(motif_params_dict["num_detectors"], ), units=motif_params_dict["num_hidden"])
    model_layer_activation_ReLU = keras.layers.core.Activation("relu")
    model_layer_dense2 = keras.layers.core.Dense(units=1)
    exp_PFM_model.add(model_layer_dense1)
    exp_PFM_model.add(model_layer_activation_ReLU)
    exp_PFM_model.add(model_layer_dense2)
else:
    model_layer_dense1 = keras.layers.core.Dense(input_shape=(motif_params_dict["num_detectors"], ), units=1)
    exp_PFM_model.add(model_layer_dense1)

exp_PFM_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss="MSE")
exp_modelsave_output_filename = "../data/for.2..2.4.modelsave_for_exp_PFM_model_with_motif_" + motif_id_str + ".hdf5"
exp_checkpointer_output_filename = "../data/for.2..2.4.checkpointer_for_exp_PFM_model_with_motif" + motif_id_str + ".hdf5"
exp_checkpointer = keras.callbacks.ModelCheckpoint(filepath=exp_checkpointer_output_filename, verbose=1, save_best_only=True)
exp_earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        ## self.losses.append(logs.get('loss'))
        print(logs.get('loss'))
        pdb.set_trace()


exp_PFM_model.fit(exp_PFM_dataset_X_float_matrix[shuffled_training_sample_index_int_vector, :], exp_PFM_dataset_y_float_vector[shuffled_training_sample_index_int_vector], batch_size=100, epochs=360, shuffle=False, validation_data=(exp_PFM_dataset_X_float_matrix[shuffled_validation_sample_index_int_vector, :], exp_PFM_dataset_y_float_vector[shuffled_validation_sample_index_int_vector]), verbose=2, callbacks=[exp_checkpointer, exp_earlystopper])
exp_PFM_model.save_weights(exp_modelsave_output_filename, overwrite=True)











numpy.random.seed(seed=123)
Deepbind_PFM_model = keras.models.Sequential()
if motif_params_dict["num_hidden"] > 0:
    model_layer_dense1 = keras.layers.core.Dense(input_shape=(motif_params_dict["num_detectors"], ), units=motif_params_dict["num_hidden"])
    model_layer_activation_ReLU = keras.layers.core.Activation("relu")
    model_layer_dense2 = keras.layers.core.Dense(units=1)
    Deepbind_PFM_model.add(model_layer_dense1)
    Deepbind_PFM_model.add(model_layer_activation_ReLU)
    Deepbind_PFM_model.add(model_layer_dense2)
else:
    model_layer_dense1 = keras.layers.core.Dense(input_shape=(motif_params_dict["num_detectors"], ), units=1)
    Deepbind_PFM_model.add(model_layer_dense1)

Deepbind_PFM_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss="MSE")
Deepbind_modelsave_output_filename = "../data/for.2..2.4.modelsave_for_Deepbind_PFM_model_with_motif_" + motif_id_str + ".hdf5"
Deepbind_checkpointer_output_filename = "../data/for.2..2.4.checkpointer_for_Deepbind_PFM_model_with_motif" + motif_id_str + ".hdf5"
Deepbind_checkpointer = keras.callbacks.ModelCheckpoint(filepath=Deepbind_checkpointer_output_filename, verbose=1, save_best_only=True)
Deepbind_earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

Deepbind_PFM_model.fit(Deepbind_PFM_dataset_X_float_matrix[shuffled_training_sample_index_int_vector, :], Deepbind_PFM_dataset_y_float_vector[shuffled_training_sample_index_int_vector], batch_size=100, epochs=360, shuffle=False, validation_data=(Deepbind_PFM_dataset_X_float_matrix[shuffled_validation_sample_index_int_vector, :], Deepbind_PFM_dataset_y_float_vector[shuffled_validation_sample_index_int_vector]), verbose=2, callbacks=[Deepbind_checkpointer, Deepbind_earlystopper])
Deepbind_PFM_model.save_weights(Deepbind_modelsave_output_filename, overwrite=True)




exp_PFM_testing_MSE_float = numpy.sum( (exp_PFM_model.predict(exp_PFM_dataset_X_float_matrix[shuffled_testing_sample_index_int_vector, :]).flatten() - exp_PFM_dataset_y_float_vector[shuffled_testing_sample_index_int_vector] ) **2 ) /  shuffled_testing_sample_index_int_vector.shape[0]

exp_PFM_testing_prediction_float_vector = exp_PFM_model.predict(exp_PFM_dataset_X_float_matrix[shuffled_testing_sample_index_int_vector, :]).flatten()

Deepbind_PFM_testing_prediction_float_vector = Deepbind_PFM_model.predict(Deepbind_PFM_dataset_X_float_matrix[shuffled_testing_sample_index_int_vector, :]).flatten()

testing_real_float_vector = exp_PFM_dataset_y_float_vector[shuffled_testing_sample_index_int_vector]

testing_prediction_and_real_DataFrame = pandas.concat([pandas.Series(testing_real_float_vector), pandas.Series(exp_PFM_testing_prediction_float_vector), pandas.Series(Deepbind_PFM_testing_prediction_float_vector)], axis=1)
testing_prediction_and_real_DataFrame.columns = ['real', 'exp.PFM', 'Deepbind.PFM']

testing_prediction_and_real_DataFrame.to_csv(path_or_buf="../data/for.2..2.4.testing.prediction.and.real.for.motif." + motif_id_str + ".txt", sep="\t", header=True, index=False)

