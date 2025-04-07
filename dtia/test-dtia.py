
##########

from dtia import train_cross_validated_trees

##########

# from os import path

import datetime

# datetime.datetime.

from sklearn.datasets import load_iris

SEED = None

iris = load_iris()
data, target = iris.data, iris.target
targets = target


max_depths = [2, 4, 8, 16]
min_samples = [1, 2, 4, 8]
kfolds = [2, 4, 8]

trials, trialname = train_cross_validated_trees(data, targets,
                                              max_depths, min_samples, kfolds,
                                              save_path_prefix=None, overwrite=True)




##########

import os
import pandas as pd
import numpy as np
from glob import glob


cutoff_depth = 4
dataframes = []
for filename in trials.filename:
    csv_filename = os.path.join("results", trialname, "csvs", f"{filename}.csv")
    print(csv_filename)
    tmpdf = pd.read_csv(csv_filename, index_col="node_name")
    tmpdf["trial"] = filename
    dataframes.append(tmpdf)
    pass
df = pd.concat(dataframes)
df = df[df.depth < cutoff_depth]
unique_nodes = pd.unique(df.index)

# for node_name in unique_nodes:
    


# indexDF = pd.read_csv("results/")

# Tr_Precision = precision_score(y_train, y_hat_train, average = 'weighted')
# Tr_Accuracy = accuracy_score(y_train, y_hat_train)
# Tr_F1_Score = f1_score(y_train, y_hat_train, average = 'weighted')
# Tr_Recall = recall_score(y_train, y_hat_train, average = 'weighted')


## Selection Crieteria 1
# average_difference = (np.mean([abs((np.array(Acc_Tr_Precision[m]) - np.array(Acc_Tst_Precision[m]))), 
#                                abs((np.array(Acc_Tr_Accuracy[m])   - np.array(Acc_Tst_Accuracy[m]))), 
#                                abs((np.array(Acc_Tr_F1_Score[m])   - np.array(Acc_Tst_F1_Score[m]))), 
#                                abs((np.array(Acc_Tr_Recall[m])     - np.array(Acc_Tst_Recall[m])))]))
# 
# 
# if average_difference <= metrics_diff:
#     Sel_Fold_Number.append(Fold_Number[m])
#     Sel_Model_Graphviz.append(Model_Graphviz[m])
#     Sel_Min_Samples_Leaf.append(Min_Samples_Leaf[m])
#     Sel_Depth.append(Depth[m])
#     
#     Sel_Tr_Precision.append(Acc_Tr_Precision[m])
#     Sel_Tr_Accuracy.append(Acc_Tr_Accuracy[m])
#     Sel_Tr_F1_Score.append(Acc_Tr_F1_Score[m])
#     Sel_Tr_Recall.append(Acc_Tr_Recall[m])
#     
#     Sel_Tst_Precision.append(Acc_Tst_Precision[m])
#     Sel_Tst_Accuracy.append(Acc_Tst_Accuracy[m])
#     Sel_Tst_F1_Score.append(Acc_Tst_F1_Score[m])
#     Sel_Tst_Recall.append(Acc_Tst_Recall[m])
#     pass

## Selection Crieteria 2
# if len(Sel_Min_Samples_Leaf) > 0:
#     for sm in range(len(Sel_Min_Samples_Leaf)):
#         average_tst_metrics = (np.mean([np.array(Sel_Tst_Precision[sm]), 
#                                         np.array(Sel_Tst_Accuracy[sm]), 
#                                         np.array(Sel_Tst_F1_Score[sm]), 
#                                         np.array(Sel_Tst_Recall[sm])]))
#         
#         if average_tst_metrics >= avg_tst_metrics:
#             final_Sel_Fold_Number.append(Sel_Fold_Number[sm])
#             final_Sel_Model_Graphviz.append(Sel_Model_Graphviz[sm])
#             final_Sel_Min_Samples_Leaf.append(Sel_Min_Samples_Leaf[sm])
#             final_Sel_Depth.append(Sel_Depth[sm])
#             
#             final_Sel_Tr_Precision.append(Sel_Tr_Precision[sm])
#             final_Sel_Tr_Accuracy.append(Sel_Tr_Accuracy[sm])
#             final_Sel_Tr_F1_Score.append(Sel_Tr_F1_Score[sm])
#             final_Sel_Tr_Recall.append(Sel_Tr_Recall[sm])
#             
#             final_Sel_Tst_Precision.append(Sel_Tst_Precision[sm])
#             final_Sel_Tst_Accuracy.append(Sel_Tst_Accuracy[sm])
#             final_Sel_Tst_F1_Score.append(Sel_Tst_F1_Score[sm])
#             final_Sel_Tst_Recall.append(Sel_Tst_Recall[sm])
#             pass
#         pass
#     pass 

