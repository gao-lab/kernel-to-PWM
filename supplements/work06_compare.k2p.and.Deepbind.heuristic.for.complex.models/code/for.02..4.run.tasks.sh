#!/bin/sh

source ./for.xx..1.setup.shell.environment.sh
set -e
export KERAS_BACKEND=tensorflow


mkdir ../data/for.02..4.train_exp_results
mkdir ../data/for.02..4.val_exp_results
mkdir ../data/for.02..4.test_exp_results



for train_batch_num in `seq -f "%04g" 0 5595`
do
    if [ ! -f "../data/for.02..4.train_exp_results/for.02..4.train_result_${train_batch_num}.npy" ]; then
        echo "${train_batch_num}"
        ${PYTHON_EXEC} ./for.02..4.calculate.exp.PFM.log.prob.for.single.batch.py train ${train_batch_num}
    fi
done


for val_batch_num in `seq -f "%04g" 0 409`
do
    if [ ! -f "../data/for.02..4.val_exp_results/for.02..4.val_result_${val_batch_num}.npy" ]; then
        echo "${val_batch_num}"
        ${PYTHON_EXEC} ./for.02..4.calculate.exp.PFM.log.prob.for.single.batch.py val ${val_batch_num}
    fi
done

for test_batch_num in `seq -f "%04g" 0 409`
do
    if [ ! -f "../data/for.02..4.test_exp_results/for.02..4.test_result_${test_batch_num}.npy" ]; then
        echo "${test_batch_num}"
        ${PYTHON_EXEC} ./for.02..4.calculate.exp.PFM.log.prob.for.single.batch.py test ${test_batch_num}
    fi
done


mkdir ../data/for.02..4.train_Deepbind_results
mkdir ../data/for.02..4.val_Deepbind_results
mkdir ../data/for.02..4.test_Deepbind_results


for train_batch_num in `seq -f "%04g" 0 5595`
do
    if [ ! -f "../data/for.02..4.train_Deepbind_results/for.02..4.train_Deepbind_result_${train_batch_num}.npy" ]; then
        echo "${train_batch_num}"
        ${PYTHON_EXEC} ./for.02..4.calculate.Deepbind.PFM.log.prob.for.single.batch.py train ${train_batch_num}
    fi
done


for val_batch_num in `seq -f "%04g" 0 409`
do
    if [ ! -f "../data/for.02..4.val_Deepbind_results/for.02..4.val_Deepbind_result_${val_batch_num}.npy" ]; then
        echo "${val_batch_num}"
        ${PYTHON_EXEC} ./for.02..4.calculate.Deepbind.PFM.log.prob.for.single.batch.py val ${val_batch_num}
    fi
done

for test_batch_num in `seq -f "%04g" 0 409`
do
    if [ ! -f "../data/for.02..4.test_Deepbind_results/for.02..4.test_Deepbind_result_${test_batch_num}.npy" ]; then
        echo "${test_batch_num}"
        ${PYTHON_EXEC} ./for.02..4.calculate.Deepbind.PFM.log.prob.for.single.batch.py test ${test_batch_num}
    fi
done
