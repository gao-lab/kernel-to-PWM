PYTHON_EXEC=python

motif_ID=$1
motif_dataset_filename=$2

# for.2..2.1.generate.input.for.step.7.py

${PYTHON_EXEC} ./for.2..2.1.generate.input.for.step.7.py ${motif_ID} ${motif_dataset_filename}

# for.2..2.2.run.step.7.py

${PYTHON_EXEC} ./for.2..2.2.run.step.7.py ${motif_ID}

# for.2..2.3.generate.real.dataset.py

${PYTHON_EXEC} ./for.2..2.3.generate.real.dataset.py ${motif_ID} ${motif_dataset_filename}

# for.2..2.4.run.the.rest.py

${PYTHON_EXEC} ./for.2..2.4.run.the.rest.py ${motif_ID}

# for.2..2.5.run.the.pseudotraining.rest.py

${PYTHON_EXEC} ./for.2..2.5.run.the.pseudotraining.rest.py ${motif_ID}
