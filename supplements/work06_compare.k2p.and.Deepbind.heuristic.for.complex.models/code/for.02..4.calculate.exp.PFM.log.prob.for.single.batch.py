import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("../")))
from tool02_k2p import log_prob

import numpy as np

## sys.argv[1] : train/val/test
## sys.argv[2] : 0000

temp_input_tensor = np.load("../data/for.02..3." + sys.argv[1] + "_batches/for.02..3." + sys.argv[1] + "_batch_" + sys.argv[2] + ".npy")
## temp_input_tensor = np.load("../data/for.02..3.train_batches/for.02..3.train_batch_0000.npy")

exp_PFM_tensor = np.load("../data/for.02..2.exp_PFM_tensor.npy")
exp_PFM_conv_constant_matrix = np.load("../data/for.02..2.exp_PFM_conv_constant_matrix.npy")

exp_result_matrix = log_prob.get_max_log_probs_for_sequences_and_kernels(temp_input_tensor, exp_PFM_tensor, use_threshold=True, threshold_vector=np.zeros(exp_PFM_tensor.shape[2]), use_conv_constant_shift=True, conv_constant_matrix=exp_PFM_conv_constant_matrix)

np.save("../data/for.02..4." + sys.argv[1] + "_exp_results/for.02..4." + sys.argv[1] + "_result_" + sys.argv[2] + ".npy", exp_result_matrix)
