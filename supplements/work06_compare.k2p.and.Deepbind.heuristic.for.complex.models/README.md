
# Model checked:

https://github.com/VoigtLab/predict-lab-origin

The paper behind this model is on Nature Communications: [https://www.nature.com/articles/s41467-018-05378-z](Deep learning to predict the lab-of-origin of engineered DNA)

# Prerequisites

Python 3 and its packages:

- numpy (>=1.13.0)
- scipy (>=0.19.0)
- tensorflow (>=1.1.0) and tensorflow-gpu
- keras (==2.0.4)
- bayesian-optimization
- pubmed-lookup

# Steps

All codes are assumed to run under the `code` directory. In addition, all relative paths are relative to the `code` directory as well.

## Initialize the project

Modify the enviromental variables in `for.xx..1.setup.shell.environment.sh`. This script is sourced by others.

Specifically, the following variables must be carefully set:

- `PYTHON_EXEC`, the full path to the python executable.
- `PIP_EXEC`, the full path to the pip executable (note that the python package `pubmed-lookup` is available from pip, but not conda, and thus we choose pip to install it).
- `LD_LIBRARY_PATH`: add cuda library path to this to make tensorflow-gpu work.

## Download the model and dataset

The model can be accessed from the GitHub link above. Here, run the following to download the model:

```
bash ./for.00..1.get.model.sh
```

Note that this script clones a specific commit (a4c641c) of the git repository to ensure the reproducibility.

Also, this script will also create a symbolic link to the AddGene dataset `addgene-plasmids-sequences.json`, which is NOT public and must be requested from AddGene itself. 


Now please download the dataset, double-confirm that this dataset is named `addgene-plasmids-sequences.json`, not compressed (i.e., not in .gz or other compressed form), and is put under the `external` directory. The result here is based on the dataset downloaded on February 19th, 2019.

One more thing we should do is to modify the script we just cloned, because:

1. AddGene has modified the data format after the paper published;
2. Some absolute paths in the script are not resolved when ported to other computers.

If you just need a quick check of results without digging into details, run the following to replace the cloned scripts with modified versions:

```
cp utils_EC2_modified.py ../external/predict-lab-origin_a4c641c/utils_EC2.py
cp main1-dna-encoding_modified.py ../external/predict-lab-origin_a4c641c/main1-dna-encoding.py
cp main2-bayesian-optimization_modified.py ../external/predict-lab-origin_a4c641c/main2-bayesian-optimization.py
cp main3-train-best-params_modified.py ../external/predict-lab-origin_a4c641c/main3-train-best-params.py
cp main4-cross-validation_modified.py ../external/predict-lab-origin_a4c641c/main4-cross-validation.py
```

Below we describe what we should modify.

### Modify the data processing codes (sequence item)

In `../external/predict-lab-origin_a4c641c/utils_EC2.py`, the element of following lists are no longer a single sequence, but a dict with sequence embedded:

```
p['sequences']['public_addgene_full_sequences']
p['sequences']['public_user_full_sequences']
p['sequences']['public_addgene_partial_sequences']
p['sequences']['public_user_partial_sequences']
```

But the function `convert_seq_to_atcgn` treat them as a single sequence, which will throw an error. Therefore, we should append to them with the key of sequence, i.e., `["sequence"]`.

There are two functions in `utils_EC2.py` that use them:

- Line 105, 108, 111, 113 in the function `get_num_plasmids_per_pi`
- Line 172, 175, 178, 180, 190, 193, 196, 198 in the function `get_seqs_annotations`

### Modify the data processing codes (pi as int)

The updated "pi" field are integers instead of human names, and are treated by python as integers automatically. This results in an error where the script tries to join these "pi"'s. Our solution is to prepend an underscore to each of these "pi"'s.

The following lines in `utils_EC2.py` are modified accordingly: Line 116, 184, 186

### Resolve the absolute paths

The prefix `/mnt/` is scattered over almost all scripts in this clone. We need to fix it to relative path before running them.

Here, by running our script, all their python scripts will be automatically run in the directory `../external/predict-lab-origin_a4c641c/`. We decided to replace `/mnt/` with `./` in order to keep the datasets in the clone directory itself. The scripts modified here are `main1*` - `main4*` files and their dependency `utils_EC2.py`.

```
main1-dna-encoding.py, Line 47-53, 60, 61
main2-bayesian-optimization_modified.py, Line 62, 63, 69, 70, 95-99
main3-train-best-params_modified.py, Line 17-21, 68, 69, 78, 79
main4-cross-validation_modified.py, Line 17-22
utils_EC2.py, Line 372
```


## Train the CNN model

__NOTE: files generated in this step amount to ~33GB; make sure you have enough disk space first.__

Before we transform the kernels, we need to get the model first. Run the following to prepare the dataset and train the model. Note that, for the version of AddGene dataset used here, the last three sequences in the training dataset are in fact excluded during training. This does not matters much, however, because there are still 55950 sequences -- far more than 3 sequences -- used for calculating the gradient.

```
bash ./for.01..1.train.model.sh
```

Optionally, run the following to check the model's accuracy (stored in `../data/for.02..1.model.metrics.txt`) if you'd like. For the version of AddGene dataset used here, the valiadtion accuracy should be close to 0.474865788189.

## Transform the kernels into PWMs

Run the following scripts to get the PWMs:

```
bash ./for.02..2.generate.exp.and.Deepbind.PFMs.sh
```

The transformed PWMs are:

```
The exact transformation: ../data/for.02..2.exp_PFM_tensor.npy

The Deepbind's heuristic transformation: ../data/for.02..2.Deepbind_PFM_tensor.npy
```

In addition, a constant matrix file is generated for the exact transformation (`../data/for.02..2.exp_PFM_conv_constant_matrix.npy`), which will be used later in calculating the (corrected) log-likelihoods.

## Calculate the (corrected) log-likelihoods

__NOTE: files generated in this step amount to ~62GB; make sure you have enough disk space first.__


Because the calculation itself is extremely memory-intensive, we split the training, validation, and testing tensors into batches, and calculate them one by one.

First, run the following command to split tensors into batches:

```
bash ./for.02..3.split.tensor.files.sh
```

Then run the following command to calculate the log-likelihoods (you may wish to introduce PBS, Slurm, etc. to the loops in this script):

```
bash ./for.02..4.run.tasks.sh
```

Finally, run the following command to concatenate all the batch results:

```
bash ./for.02..5.summarize.log.prob.results.sh
```

The resulting log-likelihood files are:

```
../data/for.02..5.test_Deepbind_total_log_prob.npy
../data/for.02..5.test_exp_total_log_prob.npy
../data/for.02..5.train_Deepbind_total_log_prob.npy
../data/for.02..5.train_exp_total_log_prob.npy
../data/for.02..5.val_Deepbind_total_log_prob.npy
../data/for.02..5.val_exp_total_log_prob.npy
```

The file `../data/for.02..2.exp_PFM_conv_constant_matrix.npy` is used to correct log-likelihoods based on PWMs from the exact transformation. This is because the input sequences in this case have N's (which was encoded as (0, 0, 0, 0)) in the MIDDLE of sequences, which we cannot skip. These N's will make the difference between convolution and log-likelihood not constant on different positions, but the equivalency still holds and thus we can still reconstruct the convolution output from the corresponding log-likelihoods (see Section "Comments on the special case where the sequence tensor is padded with zeroes" in Supplementary Notes for a discussion of effects of N's on the equivalency between convolution and log-likelihoods; although it considers only cases with zero-padding, it can be easily extended to the case here).

## Retrain and retest models

Finally, we retrain and retest models with convolutional outputs replaced by log-likelihoods.

First, run the following command to retrain and retest the model with log-likelihood based on PWMs from the exact transformation. As mentioned in the paper, we did not use stochastic gradient descent; instead, we derived the optimal parameters directly.

```
bash ./for.02..6.retrain.and.retest.exp.model.sh
```

The resulting validation accuracy is available in the file `../data/for.02..6.exp.performance.txt`. For the version of AddGene dataset used here, the validation accuracy is 0.473401659346.

Then run the following command to retrain and retest the model with log-likelihood based on PWMs from the Deepbind's heuristic transformation. Here stochastic gradient descent was used to find the optimal parameters.

```
bash ./for.02..7.retrain.and.retest.Deepbind.model.sh
```

The resulting validation accuracy is available in the file `../data/for.02..7.Deepbind.performance.txt`. For the version of AddGene dataset used here, the validation accuracy is 0.0161054172767.
