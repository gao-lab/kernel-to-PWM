#!/usr/bin/python

# kernel2PWM.py
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

parser = argparse.ArgumentParser(prog="kernel2PWM", description='Transform a convolutional kernel into a PWM.')

parser.add_argument("input_file", type=argparse.FileType('r'), nargs="?", default=sys.stdin,
                    help='The input file describing the kernel matrix. Its rows are assumed to be the nucleotide positions, and its 4 columns are assumed to be A, C, G, and T. No other types of lines (e.g., headers) are assumed to exist. If the kernel is associated with a bias b, add b/L (where L is the number of rows) to each of the kernel elements before the transformation. Defaults to the standard input.')
parser.add_argument("output_file", type=argparse.FileType('w'), nargs="?",  default=sys.stdout,
                    help='The output file describing the PWM transformed. Its format is the minimal meme motif format (e.g., http://meme-suite.org/doc/examples/sample-dna-motif.meme) without the header lines (i.e., only the "MOTIF" line, the "letter-probability matrix" line, the matrix lines, and blank lines interspersed between them will be written). Defaults to the standard output.')


parser.add_argument('--base', type=float, default=numpy.exp(1),
                    help='The base used in the logarithm step of the transformation. This base must be larger than 1. Defaults to e, the base of the natural logarithm.')
parser.add_argument("--sep", default=None,
                    help='The separator used in the input file. Defaults to a mixture of whitespace and tabs (with arbitrary length).')
parser.add_argument("--motif-name", default="temp_motif",
                    help='The name of the PWM transformed. Defaults to "temp_motif".')
parser.add_argument('--use-meme-header', action="store_true",
                    help='If set, a meme header will be prepended to the output file. Defaults to False.')
parser.add_argument('--save-matrix-only', action="store_true",
                    help='If set, only the matrix lines will be written to the output file. This overrides --use-meme-header. Defaults to False.')

args = parser.parse_args(sys.argv[1:])


if (args.base <= 1):
    raise ValueError("The base is not larger than 1.")
    sys.exit(1)

try:
    W = numpy.genfromtxt(fname=args.input_file, dtype=float, delimiter=args.sep)
except ValueError:
    raise ValueError("The format of the input file is incorrect. Please check again.") 
    sys.exit(2)

    
W_flipped = numpy.flipud(W)
C = (args.base) ** W_flipped
P = numpy.apply_along_axis(lambda temp_row_vector: temp_row_vector / numpy.sum(temp_row_vector), axis=1, arr=C)

if (numpy.sum(P == 1.0) > 0) | (numpy.sum(P == 0.0) > 0) | (numpy.sum(numpy.isnan(P)) > 0) :
    base_too_large_warning = "The base " + str(args.base) + " is too large for the computer to transform the kernel loselessly (i.e., some rows are transformed into a quadruple with 0, 1, or even nan). Please consider using a smaller base instead."
    warnings.warn(base_too_large_warning)

P_row_bias_sum_wrt_0dot25 = numpy.apply_along_axis(lambda temp_row_vector: numpy.sum((temp_row_vector - 0.25)**2), axis=1, arr=P)
    
if numpy.sum(P_row_bias_sum_wrt_0dot25 < 1e-8) > 0:
    base_too_small_warning = "Some row in the resulting PWM is filled with values that are very close to 0.25. If this row has its corresponding row in the original kernel filled with different elements, then the base " + str(args.base) + " might be too close to 1 for the computer to transform the kernel loselessly. Please consider using a larger base instead if this is the case."
    warnings.warn(base_too_small_warning)


output_header_str = "MOTIF " + args.motif_name + "\n" + "letter-probability matrix: alength= 4 w= " + str(W.shape[0]) + " nsites= 20 E= 0" + "\n"

if (args.use_meme_header == True):
    output_header_str = "MEME version 4\n\nALPHABET= ACGT\n\nstrands: +\n\nBackground letter frequencies (from uniform background):\nA 0.25000 C 0.25000 G 0.25000 T 0.25000\n" + output_header_str

if (args.save_matrix_only == True):
    output_header_str = ""

    
numpy.savetxt(fname=args.output_file, X=P, delimiter=" ", header=output_header_str, comments="")
args.output_file.close()
