
## IMPORTS
import sklearn

import numpy as np
from numpy.random import choice

import pandas as pd
from pandas import DataFrame

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


SEED = 17

## DATA
iris = load_iris()
data, target = iris.data, iris.target

t0 = np.zeros_like(target)
t0[target==0] = 1
t0[target!=0] = 0

t1 = np.zeros_like(target)
t1[target==1] = 1
t1[target!=1] = 0

t2 = np.zeros_like(target)
t2[target==2] = 1
t2[target!=2] = 0

np.random.seed(SEED)
nSamples = len(target)
targets = target
# targets = np.stack((target, t0, t1, t2)).T
# targets = np.stack((target, choice(target, nSamples), choice(target, nSamples), choice(target, nSamples))).T

## MODEL
clf = DecisionTreeClassifier(random_state=17).fit(data, targets)


## MODEL DETAILS
t = clf.tree_
n_nodes, n_tasks, n_classes = t.value.shape

value_columns = [f"_value_{i}" for i in range(n_classes)]
n_samples_for_class_columns =[f"n_samples_for_class_{i}" for i in range(n_classes)]
internal_columns = ["_weighted_n_node_samples"] + value_columns
node_columns = "id task depth left right feature threshold impurity".split()


task_ids                        = np.tile(range(n_tasks), n_nodes)
node_ids                        = np.repeat(range(n_nodes), n_tasks)
node_depths                     = np.repeat(t.compute_node_depths(), n_tasks)
node_left_ids                   = np.repeat(t.children_left, n_tasks)
node_right_ids                  = np.repeat(t.children_right, n_tasks)
node_features                   = np.repeat(t.feature, n_tasks)
node_thresholds                 = np.repeat(t.threshold, n_tasks)
node_impurities                 = np.repeat(t.impurity, n_tasks)
node_weighted_n_samples         = np.repeat(t.weighted_n_node_samples, n_tasks)

node_values                     = t.value.reshape(-1, n_classes)
n_samples_for_class_value       = node_weighted_n_samples[np.newaxis].T * node_values

# t.weighted_n_node_samples[np.newaxis][np.newaxis].T * t.value

## Everything is ready to go
df = DataFrame(zip(node_ids, task_ids,
                   node_depths,
                   node_left_ids, node_right_ids,
                   node_features, node_thresholds, node_impurities,
                   *n_samples_for_class_value.T, node_weighted_n_samples, *node_values.T),
               columns=node_columns + n_samples_for_class_columns + internal_columns)

df

## Extracting Paths

import collections
from collections import deque

stack = deque([(int(0), -1, [0])])
traversed_tree = []

while len(stack):
    node, parent, path_so_far = stack.pop()
    # print(f"{type(node)=} -> {node=}")
    l, r = df.iloc[node]["left right".split()].astype(int)
    is_leaf = (l == r)
    
    print((node, parent, path_so_far))
    traversed_tree.append((node, parent, path_so_far))
    
    if not is_leaf:
        stack.append((r, node, path_so_far + [r]))
        stack.append((l, node, path_so_far + [l]))
        pass
    pass

print(traversed_tree)

## DT Loops

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score

SEED = 17
max_depths = [2, 4, 8, 16, 32]
min_samples = [1, 2, 4, 8, 16]
kfolds = [2, 4, 8, 16]

from itertools import product

trials = product(max_depths, min_samples, kfolds)

for trial_id, (d, s, k) in enumerate(trials):
    kfold_generator = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for trn_idxs, val_idxs in kfold_generator.split(data, targets):
        clf = DecisionTreeClassifier(random_state=SEED,
                                     max_depth=d,
                                     min_samples_leaf=s)
        clf.fit(data[trn_idxs], targets[trn_idxs])
        trnY_pred = clf.predict(data[trn_idxs])
        valY_pred = clf.predict(data[val_idxs])
        
        trn_prc = precision_score(targets[trn_idxs], trnY_pred, average='weighted')
        val_prc = precision_score(targets[val_idxs], valY_pred, average='weighted')
        print(f"{trial_id=} {trn_prc=} {val_prc=}")
        pass
    pass


# trial_id = 0
# for d in max_depths:
#     for s in min_samples:
#         for k in kfolds:
#             kfold_generator = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
#             for trn_idxs, val_idxs in kfold_generator.split(data, targets):
#                 clf = DecisionTreeClassifier(random_state=SEED,
#                                              max_depth=d,
#                                              min_samples_leaf=s)
#                 clf.fit(data[trn_idxs], targets[trn_idxs])
#                 trnY_pred = clf.predict(data[trn_idxs])
#                 valY_pred = clf.predict(data[val_idxs])
                
#                 trn_prc = precision_score(targets[trn_idxs], trnY_pred, average='weighted')
#                 val_prc = precision_score(targets[val_idxs], valY_pred, average='weighted')
#                 print(f"{trial_id=} {trn_prc=} {val_prc=}")
#                 trial_id += 1
#                 pass
#             pass
#         pass
#     pass

# clf.evaluate

## Saving to H5


import h5py

with h5py.File("file.h5", 'w') as f:
    f.create_dataset("my-data", data=data)
    f.create_dataset("my-target", data=targets)
    # f.create_dataset("my-model", data=clf)
    f.create_dataset("my-fold-trn", data=trn_idxs)
    pass

## Saving to joblib

import joblib

joblib.dump(dict(model=clf,
                 Xs=data, Ys=targets,
                 trn_idxs=trn_idxs, val_idxs=val_idxs,
                 nodeDF=df, pathDF=DataFrame(traversed_tree)), "model.joblib")


## Loading from joblib

import joblib

read_data = joblib.load("model.joblib")

## More Functions
