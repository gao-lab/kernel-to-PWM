#!/bin/sh

source ./for.xx..1.setup.shell.environment.sh
set -e
export KERAS_BACKEND=tensorflow

time ${PYTHON_EXEC} ./for.02..7.retrain.and.retest.Deepbind.model.py
