
import pandas as pd

from dtia import train_cross_validated_trees, generate_trial_summary

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

# Setting `pandas` display parameters
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)

# Seed left `None` dliberately.
SEED = None

iris = load_iris()
data, target = iris.data, iris.target
targets = target

breast_cancer = load_breast_cancer()
data, target = breast_cancer.data, breast_cancer.target
targets = target


max_depths = [4, 8, 16, 32]
min_samples = [1, 2, 4, 8, 16, 32]
kfolds = [2, 4, 8]

trialsDF, trialname = train_cross_validated_trees(data, targets,
                                                  max_depths, min_samples, kfolds,
                                                  # save_figs=False, # For faster generation
                                                  save_path_prefix=None, overwrite=True)




# df, _ = filter_trials(trialsDF, f"results/{trialname}", verbose=1)
# summaryDF = generate_trial_summary(df, selected_columns="feature threshold impurity".split())


generate_trial_summary(trialsDF, f"results/{trialname}",
                       selected_columns="feature threshold impurity".split(),
                       verbose=1)
