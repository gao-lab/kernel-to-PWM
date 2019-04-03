#!/usr/bin/python

# PWM2kernel.py
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
import re
import os.path

parser = argparse.ArgumentParser(prog="PWM2kernel", description='Transform a PWM into a convolutional kernel.')

parser.add_argument("input_file", type=argparse.FileType('r'), nargs="?", default=sys.stdin,
                    help='The input file describing the PWM matrix. Its format is assumed to be the meme motif format (i.e., http://meme-suite.org/doc/meme-format.html). It is assumed to have one motif only. Defaults to the standard input.')
parser.add_argument("output_file", type=argparse.FileType('w'), nargs="?",  default=sys.stdout,
                    help='The output file describing the kernel transformed. Its rows will be the nucleotide positions, and its 4 columns will be A, C, G, and T. Defaults to the standard output.')


parser.add_argument('--base', type=float, default=numpy.exp(1),
                    help='The base used in the logarithm step of the transformation. This base must be larger than 1. Defaults to e, the base of the natural logarithm.')

parser.add_argument('--delta', type=float, default=None,
                    help='If the PWM to transform has any zeroes, this value will be added to the zeroes before the transformation (no normalization will be carried out again, though). This is because it is impossible to find a finite number x that is the logarithm of 0 with a certain base. Defaults to 1/10 of the smallest positive element in the PWM.')
parser.add_argument("--shift", type=float, default=0.0,
                    help='The shift that is added to __every__ element of the kernel transformed. This is useful if the convolution is followed by an ReLU activation. Defaults to 0 (i.e., the kernel transformed is not shifted at all).')
parser.add_argument("--shift-by-minimum", action="store_true",
                    help='Add to __every__ element of the kernel transformed the absolute value of the smallest element of the kernel transformed. This is convenient if one does not bother to predefine the shift. Note that the addition will be done regardless of whether the smallest element of the kernel transformed is positive, or negative, or 0. This overrides --shift.')

args = parser.parse_args(sys.argv[1:])


if (args.base <= 1):
    raise ValueError("The base is not larger than 1.")
    sys.exit(1)

    
try:
    input_raw_lines_list = args.input_file.readlines()
    temp_re = r'^[ ]*[0-9]'
    P_raw_lines_list = [temp_line for temp_line in input_raw_lines_list if re.match(temp_re, temp_line) is not None]
    P_raw_element_list_list = [re.split(pattern="[ ]+", string=temp_line.strip()) for temp_line in P_raw_lines_list]
    P = numpy.asarray(P_raw_element_list_list).astype('float')
except ValueError:
    raise ValueError("The format of the input file is incorrect. Please check again.") 
    sys.exit(2)

if ( numpy.sum(P < 0) > 0 ) | ( numpy.sum(P > 1) > 0 ):
    raise ValueError("The PWM matrix has elements that cannot be probabilities. Please check again.") 
    sys.exit(3)

if ( numpy.sum( numpy.abs(numpy.sum(P, axis=1) - 1.0) > 1e-4 ) > 0 ):
    rowsum_not_1_warning = "Some rows of this PWM do not sum to 1. This might be due to numerical error (which is fine), but it is also possible that there's something wrong with the PWM at the first place. Please check again if needed."
    warnings.warn(rowsum_not_1_warning)
    
    
delta = args.delta
if ( delta is None ):
    delta = numpy.min(P[P > 0]) / 10.0

C = P
if ( numpy.sum(P == 0.0) > 0 ):
    zero_probability_warning = "There are some zero elements in the PWM matrix. They will be increased by " + str(delta) + "."
    warnings.warn(zero_probability_warning)
    C[C == 0.0] += delta


if ( numpy.log(args.base) < 1e-4 ):
    base_too_small_warning = "The base " + str(args.base) + " is too small for the computer to transform the PWM loselessly (i.e., some rows are transformed into a quadruple with -inf only). Please consider using a larger base instead."
    warnings.warn(base_too_small_warning)

W_flipped = numpy.log(C) / numpy.log(args.base)
W = numpy.flipud(W_flipped)

W_shifted = W + args.shift

if (args.shift_by_minimum == True):
    W_shifted = W + numpy.abs(numpy.min(W))

numpy.savetxt(fname=args.output_file, X=W_shifted, delimiter=" ")
args.output_file.close()
