export ${PYTHON_EXEC}=python
export ${RSCRIPT_EXEC}=Rscript

mkdir -p ../data
mkdir -p ../external

# for.1..1.transform.JASPAR.CORE.2016.into.hdf5.R

${RSCRIPT_EXEC} ./for.1..1.transform.JASPAR.CORE.2016.into.hdf5.R

# for.1..2.generate.artificial.sequence.tensors.and.fasta.files.py
# A maximum of 30 CPU cores will be used in this step.
${PYTHON_EXEC} ./for.1..2.generate.artificial.sequence.tensors.and.fasta.files.py

# for.1..3.fit.CNN.ReLU.GlobalMaxpooling.LogisiticRegression.model.py
# A maximum of 20 CPU cores will be used in this step.
${PYTHON_EXEC} ./for.1..3.fit.CNN.ReLU.GlobalMaxpooling.LogisiticRegression.model.py

# for.1..3.3.supplement.training.and.validation.auc.py
# A maximum of 20 CPU cores will be used in this step.
${PYTHON_EXEC} ./for.1..3.3.supplement.training.and.validation.auc.py

# for.1..4.generate.meme.py
# A maximum of 20 CPU cores will be used in this step.
${PYTHON_EXEC} ./for.1..4.generate.meme.py

# for.1..5.calculate.MLL.R
# A maximum of 20 CPU cores will be used in this step.
${RSCRIPT_EXEC} ./for.1..5.calculate.MLL.R


# for.1..6.4.retrain.logistic.regression.with.auroc.and.auprc.R
# A maximum of 30 CPU cores will be used in this step.
${RSCRIPT_EXEC} ./for.1..6.4.retrain.logistic.regression.with.auroc.and.auprc.R
