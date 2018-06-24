

# How to repeat the results #

- Run ./code/for.1..2.__run.for.examples.public.version.sh 
   - Note that the script ./code/for.1..2.1.run.step.7.in.parallel.with.smaller.base.py is time-consuming, and a parallelism is recommended to use here. Each run of this script takes a maximum of 20 CPU cores if possible.
- The result is the PNG file ./data/for.1..2.6.MAPE.and.MSE.comparison.result.png, which is Figure 4 of the main text.

# Summary of what each script does #

- See heading comments in each script for the specific steps it serves.
- ./code/for.1..2.0.generate.input.for.step.7.with.smaller.base.py : steps 1, 2, 3, 5, 6, and the start of step 7
- ./code/for.1..2.1.run.step.7.in.parallel.with.smaller.base.py : the rest of step 7. Each run takes 20 CPU cores to maximize its speed.
- ./code/for.1..2.3.generate.real.dataset.with.smaller.base.py : step 4
- ./code/for.1..2.4.run.the.rest.with.smaller.base.py : step 8
- ./code/for.1..2.6.summarize.result.R : step 9

# Summary of steps #

1. read the motif parameters
2. read the sequences
3. build the CNN model
4. generate the benchmarking dataset
5. generate our (exp) PFM, with base chosen arbitrarily (not using MLE)
6. generate Deepbind PFM
7. calculate the log-likelihoods
8. re-train models and make predictions
9. compare the MAPE and MSE between two transformations, and plot the result


# Initial datasets #

The following files were downloaded from http://tools.genes.toronto.edu/deepbind/deepbind-v0.11-linux.tgz .

## motif model meta files ##

- ./data/for.0..2.deepbind-v0.11-linux/db/db.tsv

## motif model files ##

- ./data/for.0..2.deepbind-v0.11-linux/db/params/D00328.003.txt <- this motif has several kernels for which no sequecnes pass the ReLU activation, and thus the Deepbind transformation cannot be applied to them
- ./data/for.0..2.deepbind-v0.11-linux/db/params/D00350.002.txt
- ./data/for.0..2.deepbind-v0.11-linux/db/params/D00350.005.txt
- ./data/for.0..2.deepbind-v0.11-linux/db/params/D00558.002.txt
- ./data/for.0..2.deepbind-v0.11-linux/db/params/D00588.002.txt
- ./data/for.0..2.deepbind-v0.11-linux/db/params/D00600.001.txt
- ./data/for.0..2.deepbind-v0.11-linux/db/params/D00600.004.txt
- ./data/for.0..2.deepbind-v0.11-linux/db/params/D00694.003.txt

## License files ##

- ./data/for.0..2.deepbind-v0.11-linux/LICENSE.TXT
- ./data/for.0..2.deepbind-v0.11-linux/EULA.TXT

# Software version #

- Work05 uses Keras 2(.0.3) to support customized layers (specifically, for the reverse-complement layer), which is different from work03 (where Keras 1 is used instead).

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

# Notes #

- The log probabilities uses the natural logarithm; this will not harm the training much, but when applying the threshold, remember to adjust the threshold as well (the threshold is calculated under the optimal base).
