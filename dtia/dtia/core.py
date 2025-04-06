"""TODO: Add documentation.

Explain everything!!
"""


########################################

import numpy as np
from pandas import DataFrame
from collections import deque
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score

from itertools import product

import os
import joblib

from datetime import datetime

def prepare_tree_dataframe(t):
    """TODO: Add documentation.
    
    Explain everything!!
    """
    n_nodes, n_tasks, n_classes = t.value.shape

    value_columns = [f"_value_{i}" for i in range(n_classes)]
    n_samples_for_class_columns =[f"n_samples_for_class_{i}" for i in range(n_classes)]
    internal_columns = ["_weighted_n_node_samples"] + value_columns
    node_columns = "id task depth left right feature threshold impurity".split()

    # Repeate node_ids with number of outputs (for multi-output training)
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

    ## Packing tree info in a dataframe
    nodesDF = DataFrame(zip(node_ids, task_ids,
                            node_depths,
                            node_left_ids, node_right_ids,
                            node_features, node_thresholds, node_impurities,
                            *n_samples_for_class_value.T,
                            node_weighted_n_samples, *node_values.T),
                        columns=node_columns + n_samples_for_class_columns +
                        internal_columns)

    nodesDF.set_index("id", inplace=True)
    
    ## Extracting Paths (only supports single-output training!!)
    stack = deque([(int(0), -1, [0])])  # [(node, parent, path)] 
    traversed_tree = []                 # [(node, parent, path)]

    while len(stack):
        node, parent, path_so_far = stack.pop()
        l, r = nodesDF.iloc[node]["left right".split()].astype(int)
        is_leaf = (l == r)

        # print((node, parent, path_so_far))
        traversed_tree.append((node, parent, path_so_far))

        if not is_leaf:
            stack.append((r, node, path_so_far + [r]))
            stack.append((l, node, path_so_far + [l]))
            pass
        pass
    pathsDF = DataFrame(traversed_tree, columns="node parent path".split())
    pathsDF.set_index("node", inplace=True)

    return nodesDF.join(pathsDF)


########################################



def train_cross_validated_trees(data, targets, 
                                max_depth_per_tree, min_samples_per_leaf,
                                cross_val_kfolds,
                                verbose=2,
                                random_state=None,
                                score_func=lambda y, yhat: precision_score(y, yhat, average='weighted'),
                                overfitting_threshold=.2,
                                underfitting_threshold=.7,
                                trial_name="trial",
                                save_path_prefix=None, save_source_data=True, overwrite=False,
                                other_dt_kwargs={},
                                other_dt_fit_kwargs={}):
    """TODO: Add documentation.

    Explain everything!!
    """
    records = []
    SEED = random_state
    max_depths = max_depth_per_tree
    min_samples = min_samples_per_leaf
    kfolds = cross_val_kfolds

    now = datetime.now()
    timestamp = now.strftime("%y-%m-%d-%H-%M-%S")
    trial_name = f"{trial_name}_{timestamp}"

    save_path_prefix = save_path_prefix or "./results"
    trial_path = os.path.join(save_path_prefix, trial_name)
    if not os.path.exists(trial_path):
        os.mkdir(trial_path)

    trials = product(max_depths, min_samples, kfolds)
    trial_id = 0
    for d, s, k in trials:
        kfold_generator = StratifiedKFold(n_splits=k,  # Can cause problems if seed not set
                                          shuffle=True, random_state=SEED)
        for trn_idxs, val_idxs in kfold_generator.split(data, targets):
            clf = DecisionTreeClassifier(random_state=SEED,  # Consider having seperate seed
                                         max_depth=d,
                                         min_samples_leaf=s, **other_dt_kwargs)
            clf.fit(data[trn_idxs], targets[trn_idxs], **other_dt_fit_kwargs)
            trnY_pred = clf.predict(data[trn_idxs])
            valY_pred = clf.predict(data[val_idxs])
            
            trn_prc = precision_score(targets[trn_idxs], trnY_pred, average='weighted')
            val_prc = precision_score(targets[val_idxs], valY_pred, average='weighted')
            verbose > 0 and print(f"{trial_id=} {trn_prc=} {val_prc=}")
            
            nodeDF = prepare_tree_dataframe(clf.tree_)
            isOverfitting = abs(trn_prc - val_prc) > overfitting_threshold
            isUnderfitting = val_prc < underfitting_threshold

            isOverfitting = int(isOverfitting)
            isUnderfitting = int(isUnderfitting)
            
            verbose > 1 and isOverfitting and print(f"OVERFITTING {trial_id=} {d=} {s=} {k=} {trn_prc=} {val_prc=} {overfitting_threshold=}")
            verbose > 1 and isUnderfitting and print(f"UNDERFITTING {trial_id=} {d=} {s=} {k=} {trn_prc=} {val_prc=} {underfitting_threshold=}")

            # run_path_prefix = os.path.join(save_path_prefix, f"run_{d:04d}_{s:04d}_{k:04d}")
            # 
            # if not os.path.exists(run_path_prefix):
            #     os.mkdir(run_path_prefix)
            #     pass
            # 
            # def save_model(model, filename):
            #     joblib.dump(clf, filename)
            #     return
            # 
            # def save_data(data, filename):
            #  pass   
            
            if save_path_prefix:
                file_name = f"{trial_id=:04d}_{d=:02d}_{s=:02d}_{k=:02d}_{trn_prc=:.2f}_{val_prc=:.2f}_{isOverfitting=}_{isUnderfitting=}"
                csvs_path = os.path.join(trial_path, "csvs")
                idxs_path = os.path.join(trial_path, "idxs")
                mdls_path = os.path.join(trial_path, "mdls")
                
                file_path = os.path.join(trial_path, file_name)
                
                if os.path.exists(file_path) and not overwrite:
                    raise FileExistsError()

                if not os.path.exists(csvs_path):
                    os.mkdir(csvs_path)

                if not os.path.exists(idxs_path):
                    os.mkdir(idxs_path)
                    
                if not os.path.exists(mdls_path):
                    os.mkdir(mdls_path)

                # Saving files
                joblib.dump(value=clf,
                            filename=os.path.join(mdls_path, f"{file_name}.joblib"))
                nodeDF.to_csv(os.path.join(csvs_path, f"{file_name}.csv"), index=False)

                np.savetxt(os.path.join(idxs_path, f"{file_name}_trnidxs.txt"), trn_idxs, fmt="%d")
                np.savetxt(os.path.join(idxs_path, f"{file_name}_validxs.txt"), val_idxs, fmt="%d")
                
                pass
            
            record = dict(max_depth=d,
                          trial_id=trial_id,
                          min_samples=s,
                          num_folds=k,
                          # nodeDF=nodeDF,
                          # model=clf,
                          # data=data, targets=targets,
                          filename=file_name,
                          # trnXi=trn_idxs, trnYi=trn_idxs,
                          # valXi=val_idxs, valYi=val_idxs,
                          overfitting=isOverfitting, underfitting=isUnderfitting)
            
            records.append(record)
            trial_id += 1
            pass
        pass
    df = DataFrame(records)
    df.set_index('trial_id')
    df.to_csv(os.path.join(trial_path, "index.csv"), index=False)
    return df

