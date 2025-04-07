
from .core import train_cross_validated_trees, generate_trial_summary


class DecisionTreeInsightAnalyser:
    """TODO: Add docstring.

    Explain everything.
    """
    
    def __init__(self,
                 max_depths=[1, 2],
                 min_samples=[0, 1],
                 kfolds=[2, 4],
                 trial_name="trial",
                 save_figs=False,
                 save_path_prefix="./results",
                 verbose=1,
                 random_state=None,
                 overfitting_threshold=.05,
                 underfitting_threshold=.95,
                 columns_to_report="feature threshold impurity".split(),
                 other_dt_kwargs={},
                 other_dt_fit_kwargs={}):
        self.max_depths = max_depths
        self.min_samples = min_samples
        self.kfolds = kfolds
        self.save_figs = save_figs
        self.save_path_prefix = save_path_prefix
        self.verbose = verbose
        self.random_state = random_state
        self.overfitting_threshold = overfitting_threshold
        self.underfitting_threshold = underfitting_threshold
        self.other_dt_kwargs = other_dt_kwargs
        self.other_dt_fit_kwargs = other_dt_fit_kwargs
        self.columns_to_report = columns_to_report
        self.trial_name = trial_name
        pass

    def fit(self, X, y):
        trialsDF, trialname = train_cross_validated_trees(X, y,
                                                          self.max_depths,
                                                          self.min_samples,
                                                          self.kfolds,
                                                          trial_name=self.trial_name,
                                                          save_figs=self.save_figs,
                                                          random_state=self.random_state,
                                                          overfitting_threshold=self.overfitting_threshold,
                                                          underfitting_threshold=self.underfitting_threshold,
                                                          other_dt_kwargs=self.other_dt_kwargs,
                                                          other_dt_fit_kwargs=self.other_dt_fit_kwargs,
                                                          verbose=self.verbose,
                                                          save_path_prefix=self.save_path_prefix)
        summaryDF, cutoff_depth = generate_trial_summary(trialsDF, f"{self.save_path_prefix}/{trialname}",
                                           selected_columns=self.columns_to_report,
                                           verbose=self.verbose)
        return dict(trialsDF=trialsDF,
                    summaryDF=summaryDF,
                    cutoff_depth=cutoff_depth,
                    trial_name=trialname)
    pass

__all__ = ["DecisionTreeInsightAnalyser"]
