#!/bin/sh

source ./for.xx..1.setup.shell.environment.sh
set -e
export KERAS_BACKEND=tensorflow

## fix all "mnt" to the correct directory in all scripts

cd ../external/predict-lab-origin_a4c641c/
time ${PYTHON_EXEC} ./main1-dna-encoding.py

### for the main2* main3* main4* scripts, import pandas first
time ${PYTHON_EXEC} ./main2-bayesian-optimization.py > ./main2.out 2> ./main2.err


time ${PYTHON_EXEC} ./main3-train-best-params.py > ./main3.out 2> ./main3.err


cd ../../code
