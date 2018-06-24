#!/bin/bash


CM_name=$1
CM_accession=$2
sequence_length=$3

CM_file=../data/for.2..1.Rfam..ACC__${CM_accession}..NAME__${CM_name}.cm
CM_dir=../data/for.2..7.simulation.CM.accession__${CM_accession}.CM.name__${CM_name}.sequence.length__${sequence_length}
rm -fr ${CM_dir}
mkdir -p ${CM_dir}

Rscript_path=Rscript
python_path=python


# step 2 part 1. generate sequence files

echo `date` generating sequence files for setting ${CM_name} ${CM_accession} ${sequence_length}

sequence_count_per_class=2000
./infernal-1.1.2/src/cmemit -o ${CM_dir}/for.2..7.1.positive.sequence.fasta -N ${sequence_count_per_class} -u -e ${sequence_length} -l --seed 123 --dna ${CM_file}
./infernal-1.1.2/src/cmemit -o ${CM_dir}/for.2..7.1.negative.sequence.fasta -N ${sequence_count_per_class} -u -e ${sequence_length} -l --seed 123 --dna --idx $((sequence_count_per_class+1)) --exp 0.01 ${CM_file}

# step 2 part 2. generate sequence tensors and fasta subsets

echo `date` generating sequence tensors and fasta subsets for setting ${CM_name} ${CM_accession} ${sequence_length}

${python_path} ./for.2..7.2.generate.artificial.sequence.tensor.py ${CM_name} ${CM_accession} ${sequence_length}


# step 3 and step 4. fit model and get AUC

echo `date` fitting model for setting ${CM_name} ${CM_accession} ${sequence_length}

OMP_NUM_THREADS=1 ${python_path} ./for.2..7.3.fit.CNN.ReLU.GlobalMaxpooling.LogisticRegression.model.py ${CM_name} ${CM_accession} ${sequence_length}

# step 5. generate meme (PFM)

echo `date` generating meme for setting ${CM_name} ${CM_accession} ${sequence_length}

${python_path} ./for.2..7.4.generate.meme.py ${CM_name} ${CM_accession} ${sequence_length}

# step 6. calculate log-likelihood

echo `date` calculate log-likelihood for setting ${CM_name} ${CM_accession} ${sequence_length}

${Rscript_path} ./for.2..7.5.calculate.MLL.R ${CM_name} ${CM_accession} ${sequence_length}

# step 7. retrain logistic regression

echo `date` retrain logistic regression for setting ${CM_name} ${CM_accession} ${sequence_length}

${python_path} ./for.2..7.8.retrain.logistic.regression.with.AUPRC.py ${CM_name} ${CM_accession} ${sequence_length}
