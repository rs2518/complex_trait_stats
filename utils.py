import os

# import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score as explained_var
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score as R2



# ============= DATA LOADER =============== #

# Loads specified file given filname and directory
def pkl_load(filename, directory):
        
    return pd.read_pickle(os.path.join(directory, filename) + '.pkl')



# ============= METRIC TABLE =============== #

# Create table of regression metrics
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



# ============= STABILITY ANALYSIS =============== #
