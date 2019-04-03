import pandas
import keras
import h5py
import numpy as np
import pickle

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("../")))
from tool02_k2p import k2p


model = keras.models.load_model("../external/predict-lab-origin_a4c641c/best_model.h5")

convolution_W_tensor = model.layers[0].get_weights()[0]
convolution_b_vector = model.layers[0].get_weights()[1]

## 1. generate exp PFMs
exp_PFM_tensor, exp_PFM_conv_constant_matrix = k2p.transform(convolution_W_tensor, convolution_b_vector, conv_direction="forward", ln_of_base_of_logarithm=1.0)
np.save('../data/for.02..2.exp_PFM_tensor.npy', exp_PFM_tensor)
np.save('../data/for.02..2.exp_PFM_conv_constant_matrix.npy', exp_PFM_conv_constant_matrix)

## 2. generate Deepbind PFMs
## about 40-50 GB memories in total
## recommended to run using lots of CPU (e.g., 40 cores) and large memory

X_train_list = []
for f in range(0, 6):
    X_train = np.load("../external/predict-lab-origin_a4c641c/data" + str(f) + ".npy")
    X_train_list.append(X_train)

del X_train

all_X_train = np.concatenate(X_train_list, axis=0)

Deepbind_PFM_tensor = k2p.transform_Deepbind(all_X_train, convolution_W_tensor, convolution_b_vector, conv_direction="forward", activation="relu", threshold=0) ## solar: 36 minutes using 40+ CPUs

np.save('../data/for.02..2.Deepbind_PFM_tensor.npy', Deepbind_PFM_tensor)


## exp_result_matrix = k2p.get_max_log_probs_for_sequences_and_kernels(test_seqs_onehot[0:1, :, :], exp_PFM_tensor, use_threshold=True, threshold_vector=exp_PFM_ReLU_threshold_log_probability_vector)
