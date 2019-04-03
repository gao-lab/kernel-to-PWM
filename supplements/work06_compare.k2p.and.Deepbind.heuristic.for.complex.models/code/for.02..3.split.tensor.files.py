import numpy as np
import datetime
import pickle

from utils_EC2 import *

import os

os.makedirs("../data/for.02..3.train_batches")
os.makedirs("../data/for.02..3.val_batches")
os.makedirs("../data/for.02..3.test_batches")

X_train_list = []
for f in range(0, 6):
    X_train = np.load("../external/predict-lab-origin_a4c641c/data" + str(f) + ".npy")
    X_train_list.append(X_train)

del X_train

all_X_train = np.concatenate(X_train_list, axis=0)

del X_train_list

np.save("../data/for.02..3.train_dna_seqs_onehot.npy", all_X_train)

val_dna_seqs = pickle.load(open('../external/predict-lab-origin_a4c641c/val_dna_seqs.out', 'rb'))
val_dna_seqs_onehot = np.transpose(convert_onehot2D(val_dna_seqs), axes=(0,2,1))
np.save("../data/for.02..3.val_dna_seqs_onehot.npy", val_dna_seqs_onehot)


test_dna_seqs = pickle.load(open('../external/predict-lab-origin_a4c641c//test_dna_seqs.out', 'rb'))
test_dna_seqs_onehot = np.transpose(convert_onehot2D(test_dna_seqs), axes=(0,2,1))
np.save("../data/for.02..3.test_dna_seqs_onehot.npy", test_dna_seqs_onehot)



def split_tensor(dataset, input_tensor, batch_size):
    num_batch = np.floor(input_tensor.shape[0] / batch_size) + 1
    for i in range(0, int(num_batch)):
        if i % 100 == 0:
            print(str(datetime.datetime.now()) + " : arrived at batch " + str(i).zfill(4))
        ## np.save("./train_batches/for.02..3.train_batch_" + str(i).zfill(4) + ".npy", all_X_train[i:(i + train_batch_size), :, :])
        np.save("../data/for.02..3." + dataset + "_batches/for.02..3." + dataset + "_batch_" + str(i).zfill(4) + ".npy", input_tensor[(i*batch_size):(i * batch_size + batch_size), :, :])


split_tensor("train", all_X_train, 10)
split_tensor("val", val_dna_seqs_onehot, 10)
split_tensor("test", test_dna_seqs_onehot, 10)
