"""This module provides functions related to training trees using cross-validation.

- **train_cross_validated_trees**: Train a list of trees with cross
    validation.

  This function trains one or more decision tree models using k-fold
  cross validation. It uses all available cores to train the ensemble
  of trees in parallel. This function returns a list of trained models
  and a corresponding set of evaluation metrics for each model.

- **filter_trials**: Filter a DataFrame containing trial information
    and concatenate the CSV files from a given path.

  This function reads CSV files from a specified path prefix and
  appends their respective trial data to a single DataFrame, filtering
  out trials with depth values below a certain cutoff. The filtered
  DataFrame is concatenated with other trial information DataFrames.

- **prepare_trial_summary**: Prepare a trial summary by concatenating
    trial-specific data frames.

  Takes a DataFrame of selected trials and creates a summary that
  combines the relevant columns for each node, making it easier to
  analyze trial results side-by-side.

- **generate_trial_summary**: Generate a trial summary, then save it
    as a CSV file with specified options.

  Generates a summary from the filtered trials DataFrames. The
  function concatenates specific data frames into one single DataFrame
  summarizing the trials with chosen columns. This summarized
  DataFrame is saved to a file named 'summary.csv' located at the
  given path prefix.

This module helps streamline the process of preparing and analyzing
trial results from various datasets, aiding in efficient
decision-making based on cross-validated tree models.

"""

from tqdm import tqdm
import pandas as pd
import numpy as np
from pandas import DataFrame
from collections import deque
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

from itertools import product

import os
import joblib

from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def save_tree_as_pdf(mdl, filename):
    """Save a tree as a PDF.

    Parameters:
        - mdl (model): The model to be saved as PDF.
        - filename (str): The name of the output file in pdf.
    """
    plt.figure(figsize=(12, 8))
    plot_tree(mdl, filled=False)
    plt.savefig(filename, format='pdf')
    plt.close()
    return

def avg_diff_metrics(trialsDF):
    """Calculate the average difference in performance metrics between training and validation datasets.
    
    Parameters:
    trialsDF (DataFrame): A DataFrame containing training and validation performance metrics
    
    Returns: diff_avg (Series): A Series object representing the
    average absolute differences between corresponding performance
    metrics in the training and validation sets. The index of this
    Series will match that of the input DataFrame's index.
    
    Note: This function assumes the 'trn_prc trn_acc trn_f1 trn_rec'
    columns are present in trialsDF for training data, and the
    'val_prc val_acc val_f1 val_rec' columns are present in trialsDF
    for validation data. The results will be computed using these
    column names.

    """
    trnDF = trialsDF["trn_prc trn_acc trn_f1 trn_rec".split()]
    valDF = trialsDF["val_prc val_acc val_f1 val_rec".split()]
    diffDF = pd.DataFrame(trnDF.to_numpy() - valDF.to_numpy(),
                          columns="diff_prc diff_acc diff_f1 diff_rec".split()).abs()
    diff_avg = diffDF.mean(axis=1)
    diffDF["diff_avg"] = diff_avg
    return diff_avg




def avg_val_metrics(trialsDF):
    """Calculate the average value of multiple metrics from a DataFrame.
    
    Parameters:
    trialsDF (DataFrame): A DataFrame containing trial data with columns including "val_prc", "val_acc", "val_f1", and "val_rec".
    
    Returns:
    DataFrame: A DataFrame with an additional column representing the average of "val_prc", "val_acc", "val_f1", and "val_rec" metrics for each row.
    
    """
    valDF = trialsDF["val_prc val_acc val_f1 val_rec".split()]
    val_metrics_avg = valDF.mean(axis=1)
    return val_metrics_avg

def prepare_tree_dataframe(t):
    """Prepare a tree dataframe from tree structure.
    
    Parameters:
    t (Tree instance): The Tree instance to be converted into a dataframe
    
    Returns: traversed_nodesDF (DataFrame): DataFrame of the traversed
    nodes, containing all necessary information for analysis and
    visualization.
    
    Notes: This function creates a DataFrame for a given tree
    instance. It includes node id, task id, depth, left child id,
    right child id, feature value, threshold, impurity, values, number
    of samples per class, etc. The 'traversed_nodesDF' contains all
    traversed nodes along with the parent-child relationship. It is
    suitable for use in analysis and visualization tasks. Please note
    that this function currently only supports single-output training
    scenarios.
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
    stack = deque([(int(0), 'R', -1, [0])])  # [(node, parent, path)] 
    traversed_tree = []                 # [(node, parent, path)]

    while len(stack):
        node, name, parent, path_so_far = stack.pop()
        l, r = nodesDF.iloc[node]["left right".split()].astype(int)
        is_leaf = (l == r)

        # print((node, parent, path_so_far))
        traversed_tree.append((node, name, parent, path_so_far))

        if not is_leaf:
            stack.append((r, f"{name}r", node, path_so_far + [r]))
            stack.append((l, f"{name}l", node, path_so_far + [l]))
            pass
        pass
    pathsDF = DataFrame(traversed_tree, columns="id node_name parent path".split())
    pathsDF.set_index("id", inplace=True)
    traversed_nodesDF = nodesDF.join(pathsDF)
    traversed_nodesDF.reset_index(inplace=True)
    traversed_nodesDF.set_index("node_name", inplace=True)
    return traversed_nodesDF

def train_cross_validated_trees(data, targets, 
                                max_depth_per_tree, min_samples_per_leaf,
                                cross_val_kfolds,
                                verbose=2,
                                random_state=None,
                                overfitting_threshold=.05,
                                underfitting_threshold=.95,
                                trial_name="trial",
                                save_figs=True, 
                                save_path_prefix="./results", save_source_data=True,
                                # overwrite=False,
                                other_dt_kwargs={},
                                other_dt_fit_kwargs={}):
    """
    Train cross-validated decision trees with various parameters and save their results.
    
    Parameters:
        - data: Input features for training.
        - targets: Target labels corresponding to the input features.
        - max_depth_per_tree: Maximum depth of each individual tree in the ensemble. 
        - min_samples_per_leaf: Minimum number of samples required to be at a leaf node.
        - cross_val_kfolds: Number of folds used for k-fold cross-validation.
        - verbose: Controls the level of progress output during training.
        - random_state: Seed used by the algorithm to generate reproducible results.
        - overfitting_threshold: Overfitting is considered if the average difference between train and validation metrics exceeds this threshold.
        - underfitting_threshold: Underfitting is considered if the average validation metric falls below this threshold.
        - trial_name: Name for the current trial of experiments, appended with a timestamp.
        - save_figs: If true, save figures showing decision tree structure as PDF files. 
        - save_path_prefix: Path where results are saved (default "./results").
        - save_source_data: If true, save source data used in this experiment.
        - other_dt_kwargs: Additional arguments to pass directly to the DecisionTreeClassifier constructor. 
        - other_dt_fit_kwargs: Additional keyword arguments passed to the fit() method of DecisionTreeClassifier.
    
    Returns:
        - A DataFrame containing records of each model trained and evaluated.
        - The name of the trial run.
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
        os.makedirs(trial_path, exist_ok=True)

    trials = list(product(max_depths, min_samples, kfolds))
    trial_id = 0
    with tqdm(total=len(trials)) as pbar:
        for d, s, k in trials:
            kfold_generator = StratifiedKFold(n_splits=k,  # TODO: Revise if it can cause problems if seed not set
                                              shuffle=True, random_state=SEED)
            for trn_idxs, val_idxs in kfold_generator.split(data, targets):
                clf = DecisionTreeClassifier(random_state=SEED,  # TODO: Consider having seperate seed
                                             max_depth=d,
                                             min_samples_leaf=s, **other_dt_kwargs)
                clf.fit(data[trn_idxs], targets[trn_idxs], **other_dt_fit_kwargs)
                trnY_pred = clf.predict(data[trn_idxs])
                valY_pred = clf.predict(data[val_idxs])
                
                nodeDF = prepare_tree_dataframe(clf.tree_)
                
                trn_Accuracy        = accuracy_score(targets[trn_idxs],     trnY_pred)
                trn_Precision       = precision_score(targets[trn_idxs],    trnY_pred, average='weighted')
                trn_F1_Score        = f1_score(targets[trn_idxs],           trnY_pred, average='weighted')
                trn_Recall          = recall_score(targets[trn_idxs],       trnY_pred, average='weighted')
                
                val_Accuracy        = accuracy_score(targets[val_idxs],     valY_pred)
                val_Precision       = precision_score(targets[val_idxs],    valY_pred, average='weighted')
                val_F1_Score        = f1_score(targets[val_idxs],           valY_pred, average='weighted')
                val_Recall          = recall_score(targets[val_idxs],       valY_pred, average='weighted')
                
                file_name = f"{trial_id=:04d}_{d=:02d}_{s=:02d}_{k=:02d}_{trn_Precision=:.2f}_{val_Precision=:.2f}"
                
                # verbose > 0 and print(file_name)
                
                if save_path_prefix:
                    file_name = f"{trial_id=:04d}_{d=:02d}_{s=:02d}_{k=:02d}_{trn_Precision=:.2f}_{val_Precision=:.2f}"
                    csvs_path = os.path.join(trial_path, "csvs")
                    idxs_path = os.path.join(trial_path, "idxs")
                    mdls_path = os.path.join(trial_path, "mdls")
                    
                    # file_path = os.path.join(trial_path, file_name)
                    
                    # if os.path.exists(file_path) and not overwrite:
                    #     raise FileExistsError()
                    
                    if not os.path.exists(csvs_path):
                        os.mkdir(csvs_path)
                        
                    if not os.path.exists(idxs_path):
                        os.mkdir(idxs_path)
                            
                    if not os.path.exists(mdls_path):
                        os.mkdir(mdls_path)
                                
                    # Saving files
                    joblib.dump(value=clf,
                                filename=os.path.join(mdls_path, f"{file_name}.joblib"))
                    nodeDF.to_csv(os.path.join(csvs_path, f"{file_name}.csv"))
                    
                    np.savetxt(os.path.join(idxs_path, f"{file_name}_trnidxs.txt"), trn_idxs, fmt="%d")
                    np.savetxt(os.path.join(idxs_path, f"{file_name}_validxs.txt"), val_idxs, fmt="%d")

                    if save_figs:
                        figs_path = os.path.join(trial_path, "figs") 
                        if not os.path.exists(figs_path):
                            os.mkdir(figs_path)
                            pass
                        
                        save_tree_as_pdf(clf, os.path.join(figs_path, f"{file_name}.pdf"))
                    
                    pass
                record = dict(max_depth=d,
                              trial_id=trial_id,
                              min_samples=s,
                              num_folds=k,
                              trn_prc=trn_Precision,
                              trn_acc=trn_Accuracy, 
                              trn_f1=trn_F1_Score, 
                              trn_rec=trn_Recall, 
                              val_prc=val_Precision,
                              val_acc=val_Accuracy, 
                              val_f1=val_F1_Score, 
                              val_rec=val_Recall, 
                              filename=file_name)
                            
                records.append(record)
                trial_id += 1
                pass
            pbar.set_description(f"{file_name}")
            pbar.update(1)
            pass
        pass
    df = DataFrame(records)
    df.set_index('trial_id', inplace=True)
    df["avg_diff_metrics"]      = avg_diff_metrics(df)
    df["avg_val_metrics"]       = avg_val_metrics(df)
    overfitting_mask            = df.avg_diff_metrics > overfitting_threshold
    underfitting_mask           = df.avg_val_metrics < underfitting_threshold
    good_models_mask            = (~overfitting_mask & ~underfitting_mask)
    df = df[good_models_mask]
    df.to_csv(os.path.join(trial_path, "index.csv"), index=False)
    return df, trial_name




def filter_trials(trialsDF, path_prefix, verbose=0):
    """Filter a DataFrame containing trial information and concatenates the CSV files from a given path.
    
    Parameters:
        - trialsDF (DataFrame): A pandas DataFrame with information about various trials.
        - path_prefix (str): The prefix path where the csvs folder resides to fetch the trial data.
        - verbose (int, optional): Controls the verbosity of the function. 0 for no output, >0 for debug messages.
    
    Returns:
        - df (pandas DataFrame): A filtered and concatenated DataFrame with all trials below a certain depth cut-off
        - cutoff_depth (int): The minimum depth value among all trial dataframes.
    
    Description: This function reads CSV files from given path_prefix
    and appends the respective trial information into one combined
    pandas DataFrame.  The depth column is used to filter out trials
    beyond a minimum cutoff depth.

    """    
    dataframes = []
    for filename in trialsDF.filename:
        csv_filename = os.path.join(path_prefix, "csvs", f"{filename}.csv")
        verbose > 0 and print(csv_filename)
        tmpdf = pd.read_csv(csv_filename, index_col="node_name")
        tmpdf["trial"] = filename
        dataframes.append(tmpdf)
        pass
    df = pd.concat(dataframes)
    cutoff_depth = min(map(lambda df: int(df.depth.max()), dataframes))
    df = df[df.depth < cutoff_depth]
    return df, cutoff_depth



def prepare_trial_summary(selected_trialsDF,
                          selected_columns="feature threshold impurity".split()):
    """Prepare a trial summary by concatenating trial-specific data frames.
    
    Parameters:
        - selected_trialsDF (DataFrame): A DataFrame containing selected trials' data.
        - selected_columns (list of str, optional): Selected columns to include in the summary. Default is ["feature", "threshold", "impurity"].
    
    Returns: summaryDF (DataFrame): A DataFrame summarizing the
    trial-specific data frames by concatenating them along the columns
    axis. Each column represents a metric or feature with prefixes
    indicating their corresponding nodes.
    """    
    unique_nodes = pd.unique(selected_trialsDF.index)
    unique_nodes = [n for n in unique_nodes if len(selected_trialsDF.loc[n]) == len(selected_trialsDF.loc["R"])]
    unique_nodes.sort()
    
    dataframes = []
    selected_columns = selected_columns or "feature threshold impurity".split()
    for node_name in unique_nodes:
        tmpdf = selected_trialsDF.loc[node_name].reset_index()
        tmpdfidx = tmpdf.trial
        tmpdf = tmpdf[selected_columns]
        tmpdf.rename(columns={col: f"{node_name}_{col}"
                              for col in selected_columns},
                     inplace=True)
        tmpdf.set_index(tmpdfidx, inplace=True)
        dataframes.append(tmpdf)
        pass
    
    summaryDF = pd.concat(dataframes, axis=1)
    return summaryDF


def generate_trial_summary(trialsDF, path_prefix,
                           selected_columns="feature threshold impurity".split(),
                           filename="summary.csv",
                           verbose=1):
    """Generate a trial summary by filtering the given trials dataframe.

    The generation is based on the specified cutoff depth and then
    preparing a summary dataframe with selected columns. It saves this
    summary in a CSV file named 'summary.csv' at the provided path
    prefix.
    
    Args:
        - trialsDF (pd.DataFrame): The input trials dataframe.
        - path_prefix (str): The prefix for the file path where the summary will be saved.
        - selected_columns (list of str, optional): A list of column names to include in the generated summary. Defaults to "feature threshold impurity".
        - filename (str, optional): The name of the output CSV file. Defaults to 'summary.csv'.
        - verbose (int, optional): Controls the verbosity of the function. Higher values increase the level of logging. Defaults to 1.
    
    Returns:
    A tuple containing:
        - summaryDF (pd.DataFrame): A dataframe summarizing the trials with selected columns.
        - cutoff_depth (float): The specified cutoff depth for filtering the trials.

    """
    filename = os.path.join(path_prefix, filename)
    df, cutoff_depth = filter_trials(trialsDF, path_prefix)
    summaryDF = prepare_trial_summary(df, selected_columns=selected_columns)
    summaryDF.to_csv(filename, index=True)
    return summaryDF, cutoff_depth

    

__all__ = ["train_cross_validated_trees",
           "filter_trials",
           "prepare_trial_summary",
           "generate_trial_summary"]


