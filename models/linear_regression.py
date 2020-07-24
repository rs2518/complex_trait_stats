#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:41:59 2019

@author: raphaelsinclair
"""

# ================================ #
##### LINEAR REGRESSION #####
# ================================ #


# ============= LOAD DATA =============== #

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score as explained_var
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score as R2

import statsmodels.api as sm



# ============= LOAD DATA =============== #

# Set directory and Load dataframes
#hpc_path = '/rdsgpfs/general/project/medbio-berlanga-group/live/projects/ml_trait_prediction'
#os.chdir(hpc_path)
#path = os.path.join(hpc_path, directory)
os.chdir(os.path.expanduser('~'))
home = 'Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/'
directory = 'Data/Processed/'

data = pd.read_pickle(
        os.path.join(home, directory) + 'processed_data_sample.pkl')
int_df = pd.read_pickle(
        os.path.join(home, directory) + 'integrated_data_sample.pkl')

# NOTE:
# data = processed data (scaled and binarised data)
# int_df = integrated data (NOT scaled or binarised)



# ============= METRICS =============== #

# Create table of metrics
def metrics_table(y_true, y_pred):
    
    index = ['Mean Squared Error',
             'Mean Absolute Error',
             'Explained Variance',
             'Mean Squared Log Error',
             'Median Absolute Error',
             'R-sqaured']
    
    data = {'Score' : [MSE(y_true, y_pred),
                        MAE(y_true, y_pred),
                        explained_var(y_true, y_pred),
                        MSLE(y_true, y_pred),
                        median_absolute_error(y_true, y_pred),
                        R2(y_true, y_pred)]}
    
    table = pd.DataFrame(data = data, index = index)
    
    
    return table


##### NOTE: METHODS ARE EQUIVALENT BUT REFERENCE IS NOT STATED FOR METHOD 1

# TO DO: SELECT REFERENCE
# Chr_no = 1, A1_v2 = A, A2_v2 = A



##############################################
# VERSION 1 (Data readily processed)
##############################################

# ============= TRAIN/TEST SPLIT =============== #

# Split processed dataframe
processed_cols = [col for col in 
                  data.columns.to_list() if col not in 
                  int_df.columns.to_list()]
X = data[processed_cols]
X.insert(loc = 0, column = 'intercept', value = np.ones(len(X)))
y = data['p_value']
seed = 21

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = seed)



# ============= TRAIN MODEL =============== #

# Removing colinear variables
drop_cols = ['Chr_no__2','Chr_no__3','Chr_no__4','Chr_no__5','Chr_no__6',
 'Chr_no__7','Chr_no__8','Chr_no__9','Chr_no__11',
 'Chr_no__12','Chr_no__13','Chr_no__14','Chr_no__15','Chr_no__16',
 'Chr_no__17','Chr_no__18','Chr_no__19','Chr_no__20',
 'Chr_no__21','Chr_no__22']

# Train model
lr = sm.OLS(y_train, X_train.drop(drop_cols, axis = 1)).fit()

print('Linear model coefficients :\n', lr.params)
print('Summary :\n', lr.summary())



# ============= MODEL DIAGNOSTICS =============== #

# Predict on the test set
y_pred =  lr.predict(X_test.drop(drop_cols, axis = 1))

# Metrics
lr_metrics = metrics_table(y_test, y_pred)

# Residuals vs index
plt.scatter(x = range(len(X_test)), y = (y_test - y_pred), c = 'g')
plt.show()

# Residuals vs fitted values
plt.scatter(x = y_pred, y = (y_test - y_pred), c = 'g')
plt.show()








##############################################
# VERSION 2 (Formula-based)
##############################################

# ============= TRAIN/TEST SPLIT =============== #

# Define statsmodel formula
input_cols = int_df.drop(['SNP','A1', 'A2', 'p_value', 'log_p_val', 'Position'],
                         axis = 1).columns

formula = 'p_value ~ ' + '{}_scaled'.format(str(input_cols[0]))
for index in range(1, len(input_cols)):
    if data[input_cols[index]].dtype != np.float:
        formula = formula + ' + C({})'.format(str(input_cols[index]))
    else:
        formula = formula + ' + {}_scaled'.format(str(input_cols[index]))


# Split data using dmatrix formula
from patsy import dmatrices

y2, X2 = dmatrices(formula, data = data, return_type = 'dataframe')
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,
                                                    test_size = 0.3,
                                                    random_state = seed)

# Does intercept need to be added here?



# ============= TRAIN MODEL =============== #

# Train model and predict on test set
lr2 = sm.OLS(y2_train, X2_train).fit()

# Metrics
print('Linear model coefficients :\n', lr2.params)
print('Summary :\n', lr2.summary())



# ============= MODEL DIAGNOSTICS =============== #

# Predict on the test set
y2_pred =  lr2.predict(X2_test)

# Metrics
lr2_metrics = metrics_table(y2_test, y2_pred)






### FINISH CODE

