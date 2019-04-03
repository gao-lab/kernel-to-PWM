import keras
import numpy as np
import pickle


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint




model = keras.models.load_model('../external/predict-lab-origin_a4c641c/best_model.h5')

## get the new model (after conv-globalmaxpooling)

train_dna_seqs_onehot = np.load("../data/for.02..3.train_dna_seqs_onehot.npy")
train_pi_labels_onehot = np.load('../external/predict-lab-origin_a4c641c/train_pi_labels_onehot.out.npy')


dna_bp_length = train_dna_seqs_onehot.shape[1]
num_classes = train_pi_labels_onehot.shape[1]


filter_num = 128
num_dense_nodes = 64

new_Deepbind_model_input = Input([1, filter_num])
new_Deepbind_model_flattened = Flatten()(BatchNormalization()(new_Deepbind_model_input))
new_Deepbind_model_before_final_batchnormalization = Activation("relu")(Dense(input_dim=filter_num,output_dim=num_dense_nodes)(new_Deepbind_model_flattened))
new_Deepbind_model_output = Activation("softmax")(Dense(output_dim=num_classes)(BatchNormalization()(new_Deepbind_model_before_final_batchnormalization)))

new_Deepbind_model = keras.models.Model(inputs=new_Deepbind_model_input, outputs=new_Deepbind_model_output)

## copy the weights of old model to this model

new_Deepbind_model.layers[1].set_weights(model.layers[2].get_weights()) ## First batchnormalization
new_Deepbind_model.layers[3].set_weights(model.layers[4].get_weights()) ## First dense
new_Deepbind_model.layers[5].set_weights(model.layers[6].get_weights()) ## Second batchrnomalization
new_Deepbind_model.layers[6].set_weights(model.layers[7].get_weights()) ## Second dense



new_Deepbind_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

checkpointer = ModelCheckpoint(filepath="../data/for.02..7.Deepbind.retrained.checkpoint.hdf5", monitor="val_acc", mode='auto', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, mode='auto', verbose=1)


## ee = np.load("/gpfs/user/dingy/temp/data/for.02..3.train_batches/for.02..3.train_batch_0000.npy")
train_Deepbind_tensor = np.load("../data/for.02..5.train_Deepbind_total_log_prob.npy")[:, None, :]
val_Deepbind_tensor = np.load("../data/for.02..5.val_Deepbind_total_log_prob.npy")[:, None, :]
val_pi_labels_onehot = np.load('../external/predict-lab-origin_a4c641c/val_pi_labels_onehot.out.npy')

cl_weight = pickle.load(open('../external/predict-lab-origin_a4c641c/class_weight.out', 'rb'))

total_epoch = 100
min_batch_size = 8

history = new_Deepbind_model.fit(train_Deepbind_tensor, train_pi_labels_onehot[0:55950, :], batch_size =min_batch_size, validation_data=(val_Deepbind_tensor, val_pi_labels_onehot), nb_epoch=total_epoch, verbose=1, class_weight=cl_weight, callbacks=[checkpointer, early_stopping])





test_pi_labels_onehot = np.load('../external/predict-lab-origin_a4c641c/test_pi_labels_onehot.out.npy')
test_Deepbind_tensor = np.load("../data/for.02..5.test_Deepbind_total_log_prob.npy")[:, None, :]




train_new_Deepbind_scores = new_Deepbind_model.evaluate(train_Deepbind_tensor, train_pi_labels_onehot[0:55950, :], verbose = 1)
val_new_Deepbind_scores = new_Deepbind_model.evaluate(val_Deepbind_tensor, val_pi_labels_onehot, verbose = 1)
test_new_Deepbind_scores = new_Deepbind_model.evaluate(test_Deepbind_tensor, test_pi_labels_onehot, verbose = 1)



with open("../data/for.02..7.Deepbind.performance.txt", "w") as temp_filehandle:
    temp_filehandle.write("dataset\tloss\taccuracy\n")
    temp_filehandle.write("train\t" + str(train_new_Deepbind_scores[0]) + "\t" + str(train_new_Deepbind_scores[1]) + "\n" )
    temp_filehandle.write("val\t" + str(val_new_Deepbind_scores[0]) + "\t" + str(val_new_Deepbind_scores[1]) + "\n" )
    temp_filehandle.write("test\t" + str(test_new_Deepbind_scores[0]) + "\t" + str(test_new_Deepbind_scores[1]) + "\n" )
