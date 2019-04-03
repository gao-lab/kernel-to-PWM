import pickle
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from utils_EC2 import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import time
from keras.optimizers import SGD
from keras.models import load_model

## TODO add training (it has not been evaluated wholly; the previous accuracy is the mean accuracy across batches)

# load data
train_pi_labels_onehot = np.load('../external/predict-lab-origin_a4c641c/train_pi_labels_onehot.out.npy')
train_dna_seqs = pickle.load(open("../external/predict-lab-origin_a4c641c/train_dna_seqs.out", "rb"))
train_dna_seqs_onehot = np.transpose(convert_onehot2D(train_dna_seqs), axes=(0,2,1))

# validation accuracy has been calculated and stored before

test_pi_labels_onehot = np.load('../external/predict-lab-origin_a4c641c/test_pi_labels_onehot.out.npy')
test_dna_seqs = pickle.load(open('../external/predict-lab-origin_a4c641c/test_dna_seqs.out', 'rb'))
test_dna_seqs_onehot = np.transpose(convert_onehot2D(test_dna_seqs), axes=(0,2,1))
# load trained model
model = load_model('../external/predict-lab-origin_a4c641c/best_model.h5')


# compute testing accuracy
train_scores = model.evaluate(train_dna_seqs_onehot[0:55950, :, :], train_pi_labels_onehot[0:55950, :], verbose=1)
test_scores = model.evaluate(test_dna_seqs_onehot, test_pi_labels_onehot, verbose = 1)

with open("../data/for.02..1.model.metrics.txt", "w") as temp_filehandle:
    ## final training accuracy
    final_training_accuracy = np.load("../external/predict-lab-origin_a4c641c/best_model_train_acc.out.npy")[-1]
    ## final validation accuracy
    final_validation_accuracy = np.load("../external/predict-lab-origin_a4c641c/best_model_val_acc.out.npy")[-1][0]
    ## final testing accuracy == the `test_scores[1]` calculated above
    final_testing_accuracy = test_scores[1]
    temp_filehandle.write("training.accuracy\tvalidation.accuracy\ttesting.accuracy\n")
    temp_filehandle.write("\t".join([str(final_training_accuracy), str(final_validation_accuracy), str(final_testing_accuracy)]))
    temp_filehandle.write("\n")
