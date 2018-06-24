#!/usr/bin/python

# kernel2PWM_MLE.py
#
# Copyright (C) Center for Bioinformatics, Peking University <k2p@mail.cbi.pku.edu.cn>
#
# The software provided herein is free for ACADEMIC INSTRUCTION AND
# RESEARCH USE ONLY. You are free to download, copy, compile, study, and
# refer to the source code for any personal use of yours. Usage by you
# of any work covered by this License should not, directly or
# indirectly, enable its usage by any other individual or organization.
# 
# You are free to make any modifications to the source covered by this
# License. You are also free to compile the source after modifying it
# and using the compiled product obtained thereafter in compliance with
# this License.  You may NOT under any circumstance copy, redistribute
# and/or republish the source or a work based on it (which includes
# binary or object code compiled from it) in part or whole without the
# permission of the authors.
# 
# If you intend to incorporate the source code, in part or whole, into
# any free or proprietary program, you need to explicitly write to the
# original author(s) to ask for permission via e-mail at
# k2p@mail.cbi.pku.edu.cn.
# 
# Commercial licenses are available to legal entities, including
# companies and organizations (both for-profit and non-profit),
# requiring the software for general commercial use. To obtain a
# commercial license, please contact us via e-mail at
# k2p@mail.cbi.pku.edu.cn.
# 
# DISCLAIMER
# 
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# 
# THE ORIGINAL AUTHOR OF THE PROGRAM IS NOT LIABLE TO YOU FOR DAMAGES,
# INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES
# ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT
# NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR
# LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM
# TO OPERATE WITH ANY OTHER PROGRAMS)


import numpy
import sys
import argparse
import warnings
import scipy.signal
import h5py

parser = argparse.ArgumentParser(prog="kernel2PWM_MLE", description='Transform a series of convolutional kernels acting directly on sequences into a PWM, using Maximum Likelihood Estimation and highly scoring sequence fragments across input sequences.')

parser.add_argument("kernel_tensor_hdf5_file", type=argparse.FileType('r'),
                    help='The hdf5 file describing the kernel tensor, which is essentially a list of kernel matrices. For each kernel matrix, its rows are assumed to be the nucleotide positions, and its 4 columns are assumed to be A, C, G, and T. If the kernel is associated with a bias b, add b/L (where L is the number of rows) to each of the kernel elements before the transformation. This kernel tensor is assumed to have the following __three__ dimensions: [the number of kernels, the length of each kernel, 4].')
parser.add_argument("sequence_tensor_hdf5_file", type=argparse.FileType('r'),
                    help='The hdf5 file describing the sequence tensor, which is essentially a list of matrices describing sequence. The sequences are assumed to be one-hot encoded and, similar to the setting of kernel matrices, for each sequence matrix its rows are assumed to be the nucleotide positions, and its 4 columns are assumed to be A, C, G, and T. If the kernel is associated with a bias b, add b/L (where L is the number of rows) to each of the kernel elements before the transformation. In addition, this sequence tensor is assumed to have the following __three__ dimensions: [the number of sequences, the length of each sequence, 4]. Note that the sequences must be no shorter than the kernels.')

parser.add_argument("output_hdf5_file", type=argparse.FileType('w'), 
                    help='The output hdf5 file describing the PWM tensor transformed. The setting for the rows and columns of each PWM matrix, as well as the setting for the dimensions of the PWM tensor, are identical to those of the kernel tensor. The ith PWM matrix in the PWM tensor corresponds to the ith kernel matrix in the kernel tensor.')


parser.add_argument('--kernel-tensor-name', default=None,
                    help='The name of the kernel tensor in the kernel tensor hdf5 file. If not set, the first object in the kernel tensor hdf5 file will be used.')
parser.add_argument('--sequence-tensor-name', default=None,
                    help='The name of the sequence tensor in the sequence tensor hdf5 file. If not set, the first object in the sequence tensor hdf5 file will be used.')
parser.add_argument('--PWM_tensor_name', default="PWM_tensor",
                    help='The name of the PWM tensor in the PWM tensor hdf5 file. Defaults to "PWM_tensor". ')


args = parser.parse_args(sys.argv[1:])
#args = parser.parse_args(["../data/for.3..1.kernel.tensor.hdf5", "../data/for.3..1.sequence.tensor.hdf5", "./aaa"])


args.kernel_tensor_hdf5_file.close()
kernel_tensor_hdf5_filehandle = h5py.File(args.kernel_tensor_hdf5_file.name, "r")
kernel_tensor_name = args.kernel_tensor_name
if kernel_tensor_name is None:
   kernel_tensor_name = kernel_tensor_hdf5_filehandle.keys()[0]

kernel_tensor = kernel_tensor_hdf5_filehandle[kernel_tensor_name][:, :, :]
kernel_tensor_hdf5_filehandle.close()


args.sequence_tensor_hdf5_file.close()
sequence_tensor_hdf5_filehandle = h5py.File(args.sequence_tensor_hdf5_file.name, "r")
sequence_tensor_name = args.sequence_tensor_name
if sequence_tensor_name is None:
   sequence_tensor_name = sequence_tensor_hdf5_filehandle.keys()[0]

sequence_tensor = sequence_tensor_hdf5_filehandle[sequence_tensor_name][:, :, :]
sequence_tensor_hdf5_filehandle.close()

total_score = 0

for sample_index in range(0, sequence_tensor.shape[0]):
    for kernel_index in range(0, kernel_tensor.shape[0]):
        total_score = total_score + numpy.max(scipy.signal.correlate(sequence_tensor[sample_index, :, :], numpy.flipud(kernel_tensor[kernel_index, :, :]), mode="valid")[:, 0])


def MLL_loss_function(ln_b):
    exp_ln_loss_per_kernel_per_position_matrix = numpy.zeros([kernel_tensor.shape[0], kernel_tensor.shape[1]])
    for kernel_index in range(0, kernel_tensor.shape[0]):
        for position_index in range(0, kernel_tensor.shape[1]):
            exp_ln_loss_per_kernel_per_position_matrix[kernel_index, position_index] = numpy.log(numpy.sum(numpy.exp(ln_b * kernel_tensor[kernel_index, position_index, :])))

    exp_ln_loss_across_all_kernels_and_all_positions = exp_ln_loss_per_kernel_per_position_matrix.sum()
    total_loss_ln = -1 * (ln_b * total_score - sequence_tensor.shape[0] * exp_ln_loss_across_all_kernels_and_all_positions)
    return total_loss_ln

optimization_result = scipy.optimize.minimize(MLL_loss_function, 1, method='nelder-mead', bounds=[(0, None)], options={'xtol': 1e-8, 'disp': True})

if (optimization_result["status"] != 0):
    optimization_failure_warning = "Optimization failed."
    warnings.warn(optimization_failure_warning)
    

base = numpy.exp(optimization_result["x"])

PWM_tensor = numpy.zeros(kernel_tensor.shape)

for kernel_index in range(0, kernel_tensor.shape[0]):
    W_matrix = kernel_tensor[kernel_index, :, :]
    W_flipped_matrix = numpy.flipud(W_matrix)
    C_matrix = base ** W_flipped_matrix
    P_matrix = numpy.apply_along_axis(lambda temp_row_vector: temp_row_vector / numpy.sum(temp_row_vector), axis=1, arr=C_matrix)
    if (numpy.sum(P_matrix == 1.0) > 0) | (numpy.sum(P_matrix == 0.0) > 0) | (numpy.sum(numpy.isnan(P_matrix)) > 0) :
        base_too_large_warning = "The base " + str(args.base) + " is too large for the computer to transform the kernel loselessly (i.e., some rows are transformed into a quadruple with 0, 1, or even nan). Please consider using a smaller base instead."
        warnings.warn(base_too_large_warning)
    P_row_bias_sum_wrt_0dot25 = numpy.apply_along_axis(lambda temp_row_vector: numpy.sum((temp_row_vector - 0.25)**2), axis=1, arr=P_matrix)
    if numpy.sum(P_row_bias_sum_wrt_0dot25 < 1e-8) > 0:
        base_too_small_warning = "Some row in the resulting PWM is filled with values that are very close to 0.25. If this row has its corresponding row in the original kernel filled with different elements, then the base " + str(args.base) + " might be too close to 1 for the computer to transform the kernel loselessly. Please consider using a larger base instead if this is the case."
        warnings.warn(base_too_small_warning)
    PWM_tensor[kernel_index, :, :] = P_matrix



args.output_hdf5_file.close()
PWM_tensor_hdf5_filehandle = h5py.File(args.output_hdf5_file.name, "w")
PWM_tensor_name = args.PWM_tensor_name
PWM_tensor_hdf5_filehandle.create_dataset(name=PWM_tensor_name, data=PWM_tensor, compression="gzip")
PWM_tensor_hdf5_filehandle.close()
