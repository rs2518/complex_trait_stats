#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:48:01 2019

@author: raphaelsinclair
"""

# ================================ #
##### LASSO #####
# ================================ #


# ============= LOAD DATA =============== #

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score as explained_var
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score as R2

from sklearn.linear_model import Lasso


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

# Cross-validation
folds = 5
seed = 1
max_iter = 10000
l1_space = np.logspace(-5, 5, 11)
parameters = {'alpha': l1_space}
lasso = Lasso(normalize = False,
              random_state = seed,
              max_iter = max_iter)

lasso_cv = GridSearchCV(lasso,
                        parameters,
                        cv = folds,
                        scoring = 'neg_mean_squared_error',
                        return_train_score = True)
lasso_cv.fit(X_train, y_train)

print('Best LASSO (l1) penalty :\n', lasso_cv.best_params_)
print('Best model coefficients :\n', lasso_cv.best_estimator_.coef_)

results = lasso_cv.cv_results_


##### TRAINING

# Plot training scores for each split of cross-validation
fig1 = plt.figure()

sns.set_style('darkgrid')

for split in range(folds):
    train_scores = -results['split%s_train_score' % (split)]
    plt.semilogx(l1_space,
                 train_scores,
                 linestyle = '--',
                 label = 'Fold {0}'.format(split + 1))

plt.xlim([l1_space[0], l1_space[-1]])
plt.ylabel('Mean Squared Error')
plt.xlabel('alpha (L1 penalty)')
plt.title('Training scores by fold', fontweight = 'bold')
plt.legend()
 
  
plt.show()



# Plot mean train scores with confidence intervals
fig2 = plt.figure()

sns.set_style('darkgrid')

mean_train_scores = -results['mean_train_score']
std_train_scores = results['std_train_score']

plt.semilogx(l1_space, mean_train_scores, color = 'navy')
plt.semilogx(l1_space, mean_train_scores + std_train_scores,
             linestyle = '--', color = 'navy')
plt.semilogx(l1_space, mean_train_scores - std_train_scores,
             linestyle = '--', color = 'navy')

plt.ylabel('Mean Squared Error')
plt.xlabel('alpha (L1 penalty)')
plt.axhline(np.min(mean_train_scores), linestyle = ':', color='0.5')
plt.text(l1_space[1],
          np.min(mean_train_scores) - 0.5 * np.max(std_train_scores),
          'Best training score = {:.3g}'.format(np.min(mean_train_scores)),
          color = 'g')
plt.xlim([l1_space[0], l1_space[-1]])

plt.fill_between(l1_space, mean_train_scores + std_train_scores,
                 mean_train_scores - std_train_scores,
                 alpha=0.2, color = 'navy')
plt.title('Mean training scores +/- standard error', fontweight = 'bold')


plt.show()

## NOTE: Training suggests large alpha which implies unrestrained



##### VALIDATION

# Plot training scores for each split of cross-validation
fig3 = plt.figure()

sns.set_style('darkgrid')

for split in range(folds):
    val_scores = -results['split%s_test_score' % (split)]
    plt.semilogx(l1_space,
                 val_scores,
                 linestyle = '--',
                 label = 'Fold {0}'.format(split + 1))

plt.xlim([l1_space[0], l1_space[-1]])
plt.ylabel('Mean Squared Error')
plt.xlabel('alpha (L1 penalty)')
plt.title('Validation scores by fold', fontweight = 'bold')
plt.legend()
 
  
plt.show()


# Plot mean train scores with confidence intervals
fig4 = plt.figure()

sns.set_style('darkgrid')

mean_val_scores = -results['mean_test_score']
std_val_scores = results['std_test_score']

plt.semilogx(l1_space, mean_val_scores, color = 'purple')
plt.semilogx(l1_space, mean_val_scores + std_val_scores,
             linestyle = '--', color = 'purple')
plt.semilogx(l1_space, mean_val_scores - std_val_scores,
             linestyle = '--', color = 'purple')

plt.ylabel('Mean Squared Error')
plt.xlabel('alpha (L1 penalty)')
plt.axhline(np.min(mean_val_scores), linestyle = ':', color = '0.5')
plt.text(l1_space[1],
          np.min(mean_val_scores) - 0.5 * np.max(std_val_scores),
          'Highest validation score = {:.3g}'.format(np.min(mean_val_scores)),
          color = 'g')
plt.xlim([l1_space[0], l1_space[-1]])

plt.fill_between(l1_space, mean_val_scores + std_val_scores,
                 mean_val_scores - std_val_scores,
                 alpha=0.2, color = 'purple')
plt.title('Mean validation scores +/- standard error', fontweight = 'bold')


plt.show()



# ============= MODEL DIAGNOSTICS =============== #

# Plot model coefficients
plt.plot(range(len(X.columns)), lasso_cv.best_estimator_.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=90)

# Predict on test set using model and calculate residuals
y_pred = lasso_cv.predict(X_test)
residuals = y_test - y_pred

# Create metrics table
lr_metrics = metrics_table(y_test, y_pred)


# Residuals vs index
plt.scatter(x = range(len(X_test)), y = residuals, c = 'g')
plt.show()

# Residuals vs fitted values
plt.scatter(x = y_pred, y = residuals, c = 'g')
plt.show()

