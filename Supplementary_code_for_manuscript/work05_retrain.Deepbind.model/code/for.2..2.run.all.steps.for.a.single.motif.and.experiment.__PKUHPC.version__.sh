#!/bin/bash

PYTHON_EXEC=/home/gaog_pkuhpc/users/dingy/anaconda2_4.3.0/bin/python2.7

motif_ID=$1
motif_dataset_filename=$2

# for.2..2.1.generate.input.for.step.7.py

${PYTHON_EXEC} ./for.2..2.1.generate.input.for.step.7.py ${motif_ID} ${motif_dataset_filename}

if [ -e ../data/for.2..2.1.stop.flag.for.motif.${motif_ID}.txt ]
then
    echo "Give up running this motif due to the following reason:"
    cat ../data/for.2..2.1.stop.flag.for.motif.${motif_ID}.txt
    exit 1
fi


# for.2..2.2.run.step.7.py

${PYTHON_EXEC} ./for.2..2.2.run.step.7.py ${motif_ID}

# for.2..2.3.generate.real.dataset.py

${PYTHON_EXEC} ./for.2..2.3.generate.real.dataset.py ${motif_ID} ${motif_dataset_filename}

# for.2..2.4.run.the.rest.py

${PYTHON_EXEC} ./for.2..2.4.run.the.rest.py ${motif_ID}

# for.2..2.5.run.the.pseudotraining.rest.py

${PYTHON_EXEC} ./for.2..2.5.run.the.pseudotraining.rest.py  ${motif_ID}
