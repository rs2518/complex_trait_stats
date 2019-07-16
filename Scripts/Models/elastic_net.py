#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:57:39 2019

@author: raphaelsinclair
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:28:25 2019

@author: raphaelsinclair
"""

# ================================ #
##### ELASTIC-NET #####
# ================================ #


# ============= LOAD DATA =============== #

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score as explained_var
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score as R2

from sklearn.linear_model import ElasticNet


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
l1_ratios = np.linspace(0, 1, 11)
parameters = {'l1_ratio': l1_ratios}
enet = ElasticNet(normalize = False,
              random_state = seed,
              max_iter = max_iter)

enet_cv = GridSearchCV(enet,
                        parameters,
                        cv = folds,
                        scoring = 'neg_mean_squared_error',
                        return_train_score = True)
enet_cv.fit(X_train, y_train)

##### TRAINING

# Plot training scores for each split of cross-validation
results = enet_cv.cv_results_


fig1 = plt.figure()

sns.set_style('darkgrid')

for split in range(folds):
    train_scores = -results['split%s_train_score' % (split)]
    plt.plot(l1_ratios,
                 train_scores,
                 linestyle = '--',
                 label = 'Fold {0}'.format(split + 1))

plt.xlim([l1_ratios[0], l1_ratios[-1]])
plt.ylabel('Mean Squared Error')
plt.xlabel('alpha (l2 penalty)')
plt.title('Training scores by fold', fontweight = 'bold')
plt.legend()
 
  
plt.show()



# Plot mean train scores with confidence intervals
fig2 = plt.figure()

sns.set_style('darkgrid')

mean_train_scores = -results['mean_train_score']
std_train_scores = results['std_train_score']

plt.plot(l1_ratios, mean_train_scores, color = 'navy')
plt.plot(l1_ratios, mean_train_scores + std_train_scores,
             linestyle = '--', color = 'navy')
plt.plot(l1_ratios, mean_train_scores - std_train_scores,
             linestyle = '--', color = 'navy')

plt.ylabel('Mean Squared Error')
plt.xlabel('alpha (l2 penalty)')
plt.axhline(np.min(mean_train_scores), linestyle = ':', color='0.5')
plt.text(l1_ratios[1],
          np.min(mean_train_scores) - 0.5 * np.max(std_train_scores),
          'Best training score = {:.3g}'.format(np.min(mean_train_scores)),
          color = 'g')
plt.xlim([l1_ratios[0], l1_ratios[-1]])

plt.fill_between(l1_ratios, mean_train_scores + std_train_scores,
                 mean_train_scores - std_train_scores,
                 alpha=0.2, color = 'navy')
plt.title('Mean training scores +/- standard error', fontweight = 'bold')


plt.show()

## NOTE: Training suggests large alpha which implies unrestrained



##### VALIDATION

# Plot validation-set ('test') scores for each split of cross-validation
fig3 = plt.figure()

sns.set_style('darkgrid')

for split in range(folds):
    val_scores = -results['split%s_test_score' % (split)]
    plt.plot(l1_ratios,
                 val_scores,
                 linestyle = '--',
                 label = 'Fold {0}'.format(split + 1))

plt.xlim([l1_ratios[0], l1_ratios[-1]])
plt.ylabel('Mean Squared Error')
plt.xlabel('alpha (l2 penalty)')
plt.title('Validation scores by fold', fontweight = 'bold')
plt.legend()
 
  
plt.show()


# Plot mean validation scores with confidence intervals
fig4 = plt.figure()

sns.set_style('darkgrid')

mean_val_scores = -results['mean_test_score']
std_val_scores = results['std_test_score']

plt.plot(l1_ratios, mean_val_scores, color = 'purple')
plt.plot(l1_ratios, mean_val_scores + std_val_scores,
             linestyle = '--', color = 'purple')
plt.plot(l1_ratios, mean_val_scores - std_val_scores,
             linestyle = '--', color = 'purple')

plt.ylabel('Mean Squared Error')
plt.xlabel('alpha (l2 penalty)')
plt.axhline(np.min(mean_val_scores), linestyle = ':', color = '0.5')
plt.text(l1_ratios[1],
          np.min(mean_val_scores) - 0.5 * np.max(std_val_scores),
          'Best validation score = {:.3g}'.format(np.min(mean_val_scores)),
          color = 'g')
plt.xlim([l1_ratios[0], l1_ratios[-1]])

plt.fill_between(l1_ratios, mean_val_scores + std_val_scores,
                 mean_val_scores - std_val_scores,
                 alpha=0.2, color = 'purple')
plt.title('Mean validation scores +/- standard error', fontweight = 'bold')


plt.show()

### NOTE: Indicates LASSO model is best



# ============= MODEL DIAGNOSTICS =============== #

# Display tuned model coefficients and best hyperparameter(s)
print('Best enet (l2) penalty :\n', enet_cv.best_params_)

print(24 * '*')

coefs = np.append(enet_cv.best_estimator_.intercept_,
                  enet_cv.best_estimator_.coef_)
index = ['intercept'] + X.columns.to_list()

coef_table = pd.DataFrame(data = {'Coefficients' : coefs},
                          index = index)

print('Tuned model coefficients :\n', coef_table)


### NOTE: THIS NEEDS TO BE RE_ADJUSTED TO HAVE A BASELINE REFERENCE GROUP
# (AND TO EXCLUDE CORRELATED VARIABLES ACCORDING TO EDA)


# Plot model coefficients with barplot
fig5 = plt.figure()

sns.barplot(x = coefs, y = index)

plt.xlabel('Coefficient')
plt.ylabel('Model parameters')
plt.axvline(0, color = 'black', linewidth = 0.5)
plt.title('Model coefficients', fontweight = 'bold')


plt.show()


# Predict on test set using model and calculate residuals
y_pred = enet_cv.predict(X_test)
residuals = y_test - y_pred


# Create metrics table
enet_metrics = metrics_table(y_test, y_pred)
print(enet_metrics)

# Print linear correlation between prediction and labels
corr = np.corrcoef(x = y_pred, y = y_test, rowvar = False)[0,1]
print('Pearson correlation between model prediction and labels :\n', corr)

# Plot residuals against predicted values
fig6 = plt.figure()

sns.scatterplot(x = y_pred, y = residuals)

plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.xlim((0,1))
plt.axhline(0, color = 'black', linewidth = 0.5)


plt.show()

### NOTE: Not a good predictive model





# CHECK TUNED MODEL MATCHES WITH LASSO!!!
