# DTIA: Decision Tree Insight Analysis Tool
Decision Tree Insights Analytics (DTIA) is a novel deployment method
for supervised machine learning that shifts the focus from creating
models that differentiate categorical outputs based on input features
to discovering associations between inputs and outputs. DTIA offers an
alternative perspective to traditional machine learning techniques,
such as Random Forest feature importance attributes and K-means
clustering for feature ranking.

DTIA was developed in response to the growing need for methods that
can help uncover hidden patterns and relationships within complex
datasets, particularly in cases where abstract information gain
calculations provide more detailed insights than numerical attribute
importance.  A tutorial for how to use DTIA can be found on YouTube
via the link https://youtu.be/VsPKSYKxeI4

Link to the paper published in Machine Learning: Science and Technology:
[https://iopscience.iop.org/article/10.1088/2632-2153/ad7f23](https://iopscience.iop.org/article/10.1088/2632-2153/ad7f23#:~:text=is%20Open%20access-,Decision%20Tree%20Insights%20Analytics%20(DTIA)%20Tool%3A%20an%20Analytic%20Framework,Records%20Across%20Fields%20of%20Science)

``` bibtex
@article{Hossny_2024,
doi = {10.1088/2632-2153/ad7f23},
url = {https://dx.doi.org/10.1088/2632-2153/ad7f23},
year = {2024},
month = {oct},
publisher = {IOP Publishing},
volume = {5},
number = {4},
pages = {045004},
author = {Hossny, Karim and Hossny, Mohammed and Cougnoux, Antony and Mahmoud, Loay and Villanueva, Walter},
title = {Decision tree insights analytics (DTIA) tool: an analytic framework to identify insights from large data records across fields of science},
journal = {Machine Learning: Science and Technology}}
```

## Installation
Open anaconda prompt PowerShell or CMD. <br />

```powershell
conda create -n dtia-test Python
conda activate dtia-test
conda install pip
pip install git+https://github.com/cmoxiv/DTIA.git
```

To run the `iris` example.

```
python iris-example.py
```

## Examples

### Iris Example
``` python

from dtia import DecisionTreeInsightAnalyser as DTIA

from sklearn.datasets import load_iris

# Seed left `None` dliberately.
SEED = None

dataset = load_iris()
data, target = dataset.data, dataset.target
targets = target

max_depths = [4, 8, 16]
min_samples = [1, 2, 4, 8]
kfolds = [2, 4, 8]


analyser = DTIA(max_depths=max_depths,
                min_samples=min_samples,
                kfolds=kfolds,
                save_figs=True, # False for faster generation
                verbose=1,
                random_state=SEED,
                overfitting_threshold=.05,
                underfitting_threshold=.95,
                columns_to_report="feature threshold impurity".split(),
                other_dt_kwargs={},
                other_dt_fit_kwargs={},
                save_path_prefix="./results")

results = analyser.fit(data, targets)

for k, v in results.items():
    print(f"{k} -> {type(v)}")
    pass

print(results["summaryDF"])
    
```

### Advanced Example: Looking Under the Hood
``` python

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
```

