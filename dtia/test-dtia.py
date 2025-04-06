
##########

from dtia import train_cross_validated_trees

##########

# from os import path

import datetime

# datetime.datetime.

from sklearn.datasets import load_iris

SEED = None

iris = load_iris()
data, target = iris.data, iris.target
targets = target


max_depths = [2, 4, 8, 16]
min_samples = [1, 2, 4, 8]
kfolds = [2, 4, 8]

trials = train_cross_validated_trees(data, targets,
                                     max_depths, min_samples, kfolds,
                                     save_path_prefix=None, overwrite=True)




import pandas as pd
import numpy as np
from glob import glob

# indexDF = pd.read_csv("results/")


# trials
    
##########


# # print(trials)
# 
# 
# from bokeh.io import show
# from bokeh.plotting import figure, curdoc
# from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn
# 
# data = trials[0]['nodeDF'].to_dict
# 
# # data = dict(
# #         dates=[date(2014, 3, i+1) for i in range(10)],
# #         downloads=[randint(0, 100) for i in range(10)],
# #     )
# # source = ColumnDataSource(data)
# 
# # columns = [
# #         TableColumn(field="dates", title="Date", formatter=DateFormatter()),
# #         TableColumn(field="downloads", title="Downloads"),
# #     ]
# # data_table = DataTable(source=source, columns=columns, width=400, height=280)
# 
# # # show(data_table)
# 
# 
# # ########################################
# # curdoc().add_root(data_table)
