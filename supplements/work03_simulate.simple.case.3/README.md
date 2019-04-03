# How to repeat the results #

- Initialize the datasets and get the prerequisites first (see below).
- Run `./code/for.1..__run.sh` and `./code/for.2..__run.sh` to reproduce the Wilcoxon signed rank test results for JASPAR and Rfam motifs, respectively.
   - Note that, due to a large number of motifs (several thousands), parallelism is recommended to use here.
   - For JASPAR motifs, all motifs are run altogether in each step.
   - For Rfam motifs, all major steps are run for each motif separately.
   - Each script will output test results w.r.t. both AUPRC and AUROC.

# Prerequisites #

- Keras 1.0.4 with theano backend
   - Switching to higher versions of Keras will make the model incapable of loading parameters
- Compilation of Infernal 1.1.2 in ./code/infernal-1.1.2 (only `configure` and `make` are needed; `make install` is not needed). The Infernal 1.1.2 was downloaded and `tar -xzvf`-ed from http://eddylab.org/infernal/infernal-1.1.2.tar.gz.
- Some scripts uses parallelism to speed up the calculation. See comments in the bash files for the specific number of CPU cores needed.
- R packages: rhdf5, plyr, stringr, Biostrings, doMC, caret, reshape2, AUC, stepPlr, PerfMeas, ggplot2.
- Other python packages: pandas, numpy, h5py, scikit-learn, joblib, biopython.

# Steps #

1. read known motifs
2. generate artificial sequences (including tensors (for model training) and fasta files (for log-likelihood calculation))
3. train the model
4. get AUC for the model
5. get the PWM transformed by the exact transformation and PWM transformed by the heuristic transformation
6. calculate the maximal log likelihood
7. re-train the models for the two PWMs
8. get AUC for the re-trained models of the two PWMs
9. compare the AUCs

# Script-step mapping for JASPAR motifs #

- ./code/for.1..1.transform.JASPAR.CORE.2016.into.hdf5.R : step 1
- ./code/for.1..2.generate.artificial.sequence.tensors.and.fasta.files.py : step 2
- ./code/for.1..3.fit.CNN.ReLU.GlobalMaxpooling.LogisiticRegression.model.py : step 3
- ./code/for.1..3.3.supplement.training.and.validation.auc.py : step 4
- ./code/for.1..4.generate.meme.py : step 5
- ./code/for.1..5.calculate.MLL.R : step 6
- ./code/for.1..6.4.retrain.logistic.regression.with.auroc.and.auprc.R : steps 7, 8 and 9

# Script-step mapping for Rfam motifs #

- ./code/for.2..1.split.Rfam.into.individual.cm.files.R : step 1
- ./code/for.2..7.simulate.single.cm.sh : steps 2-7
   - step 2 part 1 is some bash commands in this script
   - ./code/for.2..7.2.generate.artificial.sequence.tensor.py : step 2 part 2
   - ./code/for.2..7.3.fit.CNN.ReLU.GlobalMaxpooling.LogisticRegression.model.py : steps 3 and 4
   - ./code/for.2..7.4.generate.meme.py : step 5
   - ./code/for.2..7.5.calculate.MLL.R : step 6
   - ./code/for.2..7.8.retrain.logistic.regression.with.AUPRC.py : step 7
- ./code/for.2..8.2.summarize.AUPRC.and.AUROC.difference.R : step 8
- ./code/for.2..9.2.plot.AUROC.and.AUPRC.difference.R : step 9

# Initial datasets #

- For Figure 3(a)
  - ./data/for.0..1.JASPAR\_CORE\_2016.meme : JASPAR CORE 2016, downloaded from Meme Suite database on May 2nd, 2017
- For Figure 3(b)
  - ./data/for.0..2.Rfam.12.2.cm : Rfam version 12.2, downloaded from Rfam on June 6th, 2017. Link: ftp://ftp.ebi.ac.uk/pub/databases/Rfam/12.2/Rfam.cm.gz . The downloaded file was gunzip-ed and renamed as ./data/for.0..2.Rfam.12.2.cm. The gunzip-ed file is too big to store in GitHub, and an empty file descriptor is used instead. Please download the file by yourself.
