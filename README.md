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
journal = {Machine Learning: Science and Technology},
abstract = {Supervised machine learning (SML) techniques have been developed since the 1960s. Most of their applications were oriented towards developing models capable of predicting numerical values or categorical output based on a set of input variables (input features). Recently, SML models’ interpretability and explainability were extensively studied to have confidence in the models’ decisions. In this work, we propose a new deployment method named Decision Tree Insights Analytics (DTIA) that shifts the purpose of using decision tree classification from having a model capable of differentiating the different categorical outputs based on the input features to systematically finding the associations between inputs and outputs. DTIA can reveal interesting areas in the feature space, leading to the development of research questions and the discovery of new associations that might have been overlooked earlier. We applied the method to three case studies: (1) nuclear reactor accident propagation, (2) single-cell RNA sequencing of Niemann-Pick disease type C1 in mice, and (3) bulk RNA sequencing for breast cancer staging in humans. The developed method provided insights into the first two. On the other hand, it showed some of the method’s limitations in the third case study. Finally, we presented how the DTIA’s insights are more agreeable with the abstract information gain calculations and provide more in-depth information that can help derive more profound physical meaning compared to the random forest’s feature importance attribute and K-means clustering for feature ranking.}}
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
python iris_example.py
```

## Examples

### Generic Example
Make a new directory.
```powershell
cd "your\path\to\the\file\to\be\run\in\the\new\directory"
```
Make a new Python file with the following content. <br />
load the input features in a variable called 'x' <br />
load the labels in a variable called 'y' <br />
<br />
|Variable Name                      |Description                                                                                                   |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------|
|test_percent:                      |Percentage of the data taken as a test for the generated decision tree models.                                |
|min_s_leaf_inc:                    |Increment in the minimum number of samples per leaf specified in the generated decision tree models.          |
|min_s_leaf:                        |Maximum number of minimum samples per leaf in the generated decision tree models.                             |
|max_depth_inc:                     |Increment in the maximum depth of the generated decision tree models.                                         |
|max_depth:                         |Maximum depth of the generated decision tree models.                                                          |
|number_of_folds:                   |Number of folds over which each of the generated decision tree models will be trained and tested.             |
|metrics_diff:                      |The difference between the training and test precision for each developed decision tree model.                |
|avg_tst_metrics:                   |The average test metrics for each developed decision tree model.                                              |
|Model_Metrics_Out_Path:            |Path where the selected model metrics and performance file will be saved.                                     |
|Model_Details_Out_Path:            |Path where the files containing details of each selected model will be saved.                                 |
|Imp_Nodes_Path_file:               |Path where the file containing important nodes in all of the selected models is saved.                        |
|N_ID_Feature_Threshold_Path_file:  |Path where the file including node ID, feature number, and feature threshold for all selected models is saved.|
<br />
The following code imports the 'iris' dataset and develops decision tree models using different hyperparameters. The minimum number of samples per leaf ranges from one to 20 with the step of one. The maximum depth of the tree ranges between two and ten with a step of one. Each model was trained and tested over ten folds. The selection criteria for the models to be analyzed were to have an average difference between training and test classification metrics of 0.01, and the average test metrics should be higher than 0.9. Finally, it used the same path where you are in the Anaconda prompt PowerShell to create the results folder. The results folder is named '.#dtia#'. In '.#dtia#' there is a folder named by the time stamp. This folder includes two folders. The 'csvs' and 'joblibs' folders contain the csv files describing the details of each model that passed the selection criteria and the joblib files of all the generated models, respectively.

### iris Example

Specific iris example with default location for output saving.
```python
import logging
logging.basicConfig(level=logging.INFO)

from sklearn.datasets import load_iris
from dtia import DecisionTreeInsightAnalyser


X, y = load_iris(return_X_y=True)
dtia_clf = DecisionTreeInsightAnalyser(
	Model_Metrics_Out_Path="output/joblibs/",
    Model_Details_Out_Path="output/csvs/",
    Imp_Nodes_Path_file=f"./imp_nodes.csv",
    N_ID_Feature_Threshold_Path_file=f"./n_id_feat_thresh.csv")

dtia_clf.fit(X, y)
```
