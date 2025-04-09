"""Module for Decision Tree Insight Analyser.

This module provides the `DecisionTreeInsightAnalyser` class, which is
a tool for analyzing decision tree models in machine learning.  It can
be used to tune hyperparameters such as maximum depths and minimum
sample sizes, perform k-fold cross-validation, and generate summary
reports of model performance.

Classes:
    - DecisionTreeInsightAnalyser: A tool for analyzing decision tree models.

Functions:
    - train_cross_validated_trees: Trains cross-validated decision tree models with given parameters.
    - generate_trial_summary: Generates a summary report of decision tree models trained across different depths.

Example Usage:

    ```python
    from insight_analytical_lab import decision_tree_insight_analyser

    insight_analyser = decision_tree_insight_analyser.DecisionTreeInsightAnalyser(max_depths=[3, 5], min_samples=[2, 4])
    results = insight_analyser.fit(X, y)
    ```
"""

from .core import train_cross_validated_trees, generate_trial_summary


class DecisionTreeInsightAnalyser:
    """The DecisionTreeInsightAnalyser is a tool for analyzing decision tree models.

    It allows you to tune hyperparameters such as maximum depths,
    minimum sample sizes, and k-fold cross-validation folds. This
    class can be used with various machine learning libraries, such as
    scikit-learn.

    Attributes:
        max_depths (list): List of possible values for the 'max_depth' hyperparameter.
        min_samples (list): List of possible values for the 'min_samples_split' and 'min_samples_leaf' hyperparameters.
        kfolds (list): List of possible values for the 'n_splits' parameter in k-fold cross-validation.
        trial_name (str): Name of the trial to be used when saving results. Default is "trial".
        save_figs (bool): If True, figures will be saved after each fit. Default is False.
        save_path_prefix (str): The prefix for the path where results are saved. Default is "./results".
        verbose (int): Level of verbosity. Higher values increase logging frequency. Default is 1.
        random_state (int or None): Seed value for the random number generator. If set, ensures reproducible results. Default is None.
        overfitting_threshold (float): Threshold to determine if a model is overfitted based on cross-validation score. Default is .05.
        underfitting_threshold (float): Threshold to determine if a model is underfitted based on cross-validation score. Default is .95.
        columns_to_report (str or list): Columns to include in report when analyzing decision tree models. Default is "feature threshold impurity".
        other_dt_kwargs (dict): Additional keyword arguments for the decision tree model fitting process.
        other_dt_fit_kwargs (dict): Additional keyword arguments for the decision tree model fitting process.

    Example Usage:

    ```python
     decision_tree_insight_analyser = DecisionTreeInsightAnalyser(max_depths=[3, 5], min_samples=[2, 4])
     ```

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
        """The fit method trains cross-validated decision tree models on input data and returns a dictionary containing the results. 

        Parameters:
            - X: Input data for training.
            - y: Output labels corresponding to each instance in X.

        Returns:
            - A dictionary containing the following items:
            - trialsDF: DataFrame object with information about each trial conducted during model training.
            - summaryDF: DataFrame summarizing the performance of decision tree models trained across different depths.
            - cutoff_depth: The depth at which the best performing model was trained.
            - trial_name: Name given to a particular instance of the training process.

        """
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

