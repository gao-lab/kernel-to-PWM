import os
import stat
import sys
import numpy
import pandas
import sklearn.linear_model
import sklearn.metrics
import re

# 1. read CM information

CM_name_str, CM_ACC_str = sys.argv[1:3]
sequence_length_int = int(sys.argv[3])
CM_dir_str = "../data/for.2..7.simulation." + "CM.accession__" + CM_ACC_str + ".CM.name__" + CM_name_str +  ".sequence.length__" + str(sequence_length_int) + "/"

# 2. read training+validation MLL


training_and_validation_MLL_filename_str = CM_dir_str + "for.2..7.5.manual.MLL.txt"
training_and_validation_MLL_DataFrame = pandas.read_csv(filepath_or_buffer=training_and_validation_MLL_filename_str, sep="\t", header=0)
training_and_validation_MLL_DataFrame['y'] = training_and_validation_MLL_DataFrame.apply(lambda row: int(re.sub(pattern=".*__([01]+)$", repl="\\1", string=row["sequence.name"])), axis=1)
training_and_validation_MLL_DataFrame['PWM.type'] = training_and_validation_MLL_DataFrame.apply(lambda row: re.sub(pattern="(.*)__[0-9]+$", repl="\\1", string=row["PWM.name"]), axis=1)


PWM_type_and_auc_list = []

for temp_PWM_type, temp_DataFrame in training_and_validation_MLL_DataFrame.groupby("PWM.type"):
    X_and_y_DataFrame = temp_DataFrame.pivot_table(values='MLL', index=['sequence.name', 'y'], columns='PWM.name')
    X_matrix = numpy.asarray(X_and_y_DataFrame)
    y_vector = numpy.asarray([temp_index[1] for temp_index in X_and_y_DataFrame.index.values])
    temp_model_LogisticRegressionCV = sklearn.linear_model.LogisticRegressionCV(cv=5, penalty='l2', solver='liblinear')
    temp_model_LogisticRegressionCV.fit(X=X_matrix, y=y_vector)
    y_predicted_vector = temp_model_LogisticRegressionCV.predict_proba(X=X_matrix)[:, 1]
    temp_auc = sklearn.metrics.roc_auc_score(y_true=y_vector, y_score=y_predicted_vector)
    PWM_type_and_auc_list.append([temp_PWM_type, temp_auc])


pandas.DataFrame(PWM_type_and_auc_list, columns=['PWM.type', 'AUC']).to_csv(path_or_buf=CM_dir_str + "for.2..7.6.PWM.type.and.auc.txt", sep="\t", header=True, index=False)
