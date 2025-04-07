
import pandas as pd
from sklearn.datasets import load_breast_cancer
from dtia import train_cross_validated_trees, filter_trials, prepare_trial_summary

# Setting `pandas` display parameters
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)

# Seed left `None` dliberately.
SEED = None

dataset = load_breast_cancer()
data, target = dataset.data, dataset.target
targets = target

max_depths = [4, 8, 16]
min_samples = [1, 2, 4, 8]
kfolds = [2, 4, 8]

trialsDF, trialname = train_cross_validated_trees(data, targets,
                                                  max_depths, min_samples, kfolds,
                                                  random_state=SEED,
                                                  overfitting_threshold=.05,
                                                  underfitting_threshold=.95,
                                                  trial_name="my-trial",
                                                  save_figs=True, # For faster generation
                                                  save_path_prefix="./results")

filteredDF, cutoff_depth = filter_trials(trialsDF, f"results/{trialname}", verbose=1)
summaryDF = prepare_trial_summary(filteredDF, selected_columns="feature threshold impurity".split())

print(filteredDF)
print(summaryDF)



