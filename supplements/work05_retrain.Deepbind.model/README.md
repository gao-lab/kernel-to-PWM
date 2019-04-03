

# How to repeat the results #

- Get the corresponding softwares and initialize the datasets first (see below).
- Run `./code/for.2..__run.for.all.motifs.sh` 
   - Note that the script ./code/for.2..2.2.run.step.7.py is time-consuming when only one CPU core can be used, and a parallelism is recommended to use here. Each run of this script takes a maximum of 20 CPU cores whenever possible.
- The result is the PNG file `./data/for.2..3.MAPE.and.MSE.comparison.result.for.all.motifs.png`, which is Figure 3 of the main text.


# Summary of steps #

1. read the motif parameters
2. read the sequences
3. build the CNN model
4. generate the benchmarking dataset
5. generate the PWM transformed by the exact transformation, with base chosen arbitrarily (not using MLE)
6. generate the PWM transformed by the heuristic transformation
7. calculate the log-likelihoods
8. re-train models and make predictions
9. compare the MAPE and MSE between two transformations, and plot the result


# Initial datasets #

Motif parameter files were downloaded from http://tools.genes.toronto.edu/deepbind/deepbind-v0.11-linux.tgz . Inside this archive there is a directory called `deepbind`. Please put this directory into `./data/` and rename it to `for.0..2.deepbind-v0.11-linux`.

Sequence files were downloaded from http://tools.genes.toronto.edu/deepbind/nbtcode/nbt3300-supplementary-software.zip . Inside this archive there is a directory `nbt3300-code`. Please put its subdirectory `nbt3300-code/data` into `./data/` and rename it to `for.0..3.deepbind.nbt.3300.dataset`; this will create the following directories: `./data/for.0..3.deepbind.nbt.3300.dataset/deepfind`,  `./data/for.0..3.deepbind.nbt.3300.dataset/dream5`,    `./data/for.0..3.deepbind.nbt.3300.dataset/encode`,    `./data/for.0..3.deepbind.nbt.3300.dataset/rnac`,      `./data/for.0..3.deepbind.nbt.3300.dataset/selex`.

# Software version #

- Work05 uses Keras 2(.0.3) with theano backend to support customized layers (specifically, for the reverse-complement layer), which is different from work03 (where Keras 1 is used instead).
- R packages: ggplot2, plyr.
- Other python packages: numpy, pandas, scipy, h5py, joblib, scikit-learn.

# Model structure of Deepbind #

- Overall structure:
  1. direct scanning: input sequence -> convolution (16 kernels, +bias) -> rect (ReLU) -> GlobalMaxPooling
  2. reverse-complementing scanning: input sequence -> reverse complement sequence -> convolution (16 kernels, +bias; parameters are the same to those used in direct scanning) -> rect (ReLU) -> GlobalMaxPooling
  3. final maxpooling result: max of "direct scanning" and "reverse-complementing scanning" per "(sequence, kernel)"
  4. final maxpooling result -> neural network (+bias; with ReLU as the activation)
- valid convolution (note that it is actually (almost) full convolution on the original sequences, as the original sequnces have been padded on both sides with 0.25)
- the neural network has one of the following two structures:
    - 16 -> 1 (linear activation, no further sigmoid)
    - 16 -> 32 -ReLU-> 32 -> 1 (linear activation, no further sigmoid)



# Datasets (training, validation, testing) #

- X:
    - one-hot encoding
    - rows are nucleotide positions
    - columns are A, C, G, and then T
    - 0.25-padded on both sides (extended by the length of motif - 1 on both sides)
	- input might have N which is represented by 0.25 on all four nucleotides. Such input sequences are rare and discarded from our analyses
- y:
    - PBM: the binding affinity measured (not sure)  
    - ChIP or SELEX: 1 for preferred sequences, 0 for presumed background sequences


# Encoding of Deepbind model parameters #

1. For the 'detectors' variable, the index of detector changes the fastest, and then the type of base, and then the position of base. In python the tensor could be established by filling the values into a tensor with shape [num\_detectors, 4, detector\_len] __with order "F"__.
2. For the 'weights1' variable, the index of hidden nodes changes the fastest and then the number of detetors.
