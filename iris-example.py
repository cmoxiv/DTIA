
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
    
