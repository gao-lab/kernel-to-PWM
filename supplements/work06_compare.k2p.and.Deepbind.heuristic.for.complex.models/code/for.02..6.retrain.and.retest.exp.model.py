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



model = keras.models.load_model('../external/predict-lab-origin_a4c641c/best_model.h5')

## get the new model (after conv-globalmaxpooling)

train_pi_labels_onehot = np.load('../external/predict-lab-origin_a4c641c/train_pi_labels_onehot.out.npy')
val_pi_labels_onehot = np.load('../external/predict-lab-origin_a4c641c/val_pi_labels_onehot.out.npy')
test_pi_labels_onehot = np.load('../external/predict-lab-origin_a4c641c/test_pi_labels_onehot.out.npy')
train_dna_seqs_onehot = np.load("../data/for.02..3.train_dna_seqs_onehot.npy")
val_dna_seqs_onehot = np.load("../data/for.02..3.val_dna_seqs_onehot.npy")
test_dna_seqs_onehot = np.load("../data/for.02..3.test_dna_seqs_onehot.npy")
train_exp_result = np.load("../data/for.02..5.train_exp_total_log_prob.npy")
val_exp_result = np.load("../data/for.02..5.val_exp_total_log_prob.npy")
test_exp_result = np.load("../data/for.02..5.test_exp_total_log_prob.npy")



dna_bp_length = train_dna_seqs_onehot.shape[1]
num_classes = train_pi_labels_onehot.shape[1]


filter_num = 128
num_dense_nodes = 64

new_model_input = Input([1, filter_num])
new_model_flattened = Flatten()(BatchNormalization()(new_model_input))
new_model_before_final_batchnormalization = Activation("relu")(Dense(input_dim=filter_num,output_dim=num_dense_nodes)(new_model_flattened))
new_model_output = Activation("softmax")(Dense(output_dim=num_classes)(BatchNormalization()(new_model_before_final_batchnormalization)))

new_model = keras.models.Model(inputs=new_model_input, outputs=new_model_output)

## copy the weights of old model to this model

new_model.layers[1].set_weights(model.layers[2].get_weights()) ## First batchnormalization
new_model.layers[3].set_weights(model.layers[4].get_weights()) ## First dense
new_model.layers[5].set_weights(model.layers[6].get_weights()) ## Second batchrnomalization
new_model.layers[6].set_weights(model.layers[7].get_weights()) ## Second dense

new_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])










train_new_exp_scores = new_model.evaluate(train_exp_result[:, None, :], train_pi_labels_onehot[0:55950, :], verbose = 1)

val_new_exp_scores = new_model.evaluate(val_exp_result[:, None, :], val_pi_labels_onehot, verbose = 1)

test_new_exp_scores = new_model.evaluate(test_exp_result[:, None, :], test_pi_labels_onehot, verbose = 1)


with open("../data/for.02..6.exp.performance.txt", "w") as temp_filehandle:
    temp_filehandle.write("dataset\tloss\taccuracy\n")
    temp_filehandle.write("train\t" + str(train_new_exp_scores[0]) + "\t" + str(train_new_exp_scores[1]) + "\n" )
    temp_filehandle.write("val\t" + str(val_new_exp_scores[0]) + "\t" + str(val_new_exp_scores[1]) + "\n" )
    temp_filehandle.write("test\t" + str(test_new_exp_scores[0]) + "\t" + str(test_new_exp_scores[1]) + "\n" )
