#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:34:07 2019

@author: raphaelsinclair
"""


# ============= DATA PREPARATION =============== #

# Load dataframes
import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

#hpc_path = '/rdsgpfs/general/project/medbio-berlanga-group/live/projects/ml_trait_prediction'
#os.chdir(hpc_path)
#path = os.path.join(hpc_path, directory)
os.chdir(os.path.expanduser('~'))
home = 'Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Analysis/GeneAtlas/'
directory = 'Data/Processed/'

data = pd.read_csv(
        os.path.join(home, directory) + 'processed_data_sample.csv',
        index_col = 0)
int_df = pd.read_csv(
        os.path.join(home, directory) + 'integrated_data_sample.csv',
        index_col = 0)



# ============= SCIKIT LEARN MODELS =============== #

# Split data into test and train sets
from sklearn.model_selection import train_test_split


# Set input/target values
processed_cols = [col for col in data.columns.to_list() if col not in int_df.columns.to_list()]
X = data[processed_cols]
y = data['p_value']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 21)

log_y_train = -np.log(y_train)
log_y_test = -np.log(y_test)


########## LINEAR MODEL ##########

# Fit model and predict on test set. Return metrics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

modela = linear_model.LinearRegression()
modela.fit(X_train, y_train)
y_pred = modela.predict(X_test)

modela_coefs = pd.DataFrame(np.append(modela.intercept_, modela.coef_),
                           index = ['intercept'] +list(X.columns), 
                           columns = ['Coefficients'])
print('Linear model coefficients :\n', modela_coefs)
print('Mean-squared error :', mean_squared_error(y_test, y_pred))
print('Variance explained (R^2) :' , r2_score(y_test, y_pred))


#plt.scatter(x = X_test.index, y = (y_test - y_pred), c = 'g')
#plt.show()     # Plot not useful

sns.distplot((y_test - y_pred))
plt.show()     # Residuals should be normally distributed


##### Log transformed p-values

modelb = linear_model.LinearRegression()
modelb.fit(X_train, log_y_train)
log_y_pred = modelb.predict(X_test)

modelb_coefs = pd.DataFrame(np.append(modelb.intercept_, modelb.coef_),
                           index = ['intercept'] +list(X.columns), 
                           columns = ['Coefficients'])
print('Linear model coefficients (log-transformed) :\n', modelb_coefs)
print('Mean-squared error :', mean_squared_error(log_y_test, log_y_pred))
print('Variance explained (R^2) :', r2_score(log_y_test, log_y_pred))


#plt.scatter(x = X_test.index, y = (log_y_test - log_y_pred), c = 'g')
#plt.show()     # Plot not useful

sns.distplot((log_y_test - log_y_pred))
plt.show()     # Residuals should be normally distributed



########## LASSO REGRESSION ##########

### NOTE: SET MAX ITERATION BEFORE RUNNING IN HPC (JUST IN CASE)
# maxiter = 10000

### NOTE: MAY WANT TO TRY CV VERSIONS OF RIDGE/LASSO/E-NET BECAUSE OF
# ADDITIONAL SUPPORT FEATURES (STORING SCORES WITHOUT LOOPS)


# Fit model and predict on test set. Return metrics
folds = 5
from sklearn.model_selection import GridSearchCV

# Compute lasso model and plot coefficients
lasso = linear_model.Lasso(normalize = False)
l1_space = np.logspace(-10, 10, 21)
l1_param = {'alpha': l1_space}

lasso_cv = GridSearchCV(lasso, l1_param, cv = folds)
lasso_cv.fit(X_train, y_train)
lasso_pred = lasso_cv.predict(X_test)


# Compute metrics and plots
print('Best LASSO (l1) penalty :\n', lasso_cv.best_params_)
print('Best model coefficients :\n', lasso_cv.best_estimator_.coef_)
print('Mean-squared error :', mean_squared_error(y_test, lasso_pred))
print('Variance explained (R^2) :', r2_score(y_test, lasso_pred))
print('Mean cross-validation scores:', lasso_cv.cv_results_['mean_test_score'])

plt.plot(range(len(X.columns)), lasso_cv.best_estimator_.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=90)
plt.show()

lasso_scores = lasso_cv.cv_results_['mean_test_score']
lasso_se = lasso_cv.cv_results_['std_test_score']
plt.semilogx(l1_space, lasso_scores)
plt.semilogx(l1_space, lasso_scores + lasso_se, 'b--')
plt.semilogx(l1_space, lasso_scores - lasso_se, 'b--')
plt.fill_between(l1_space, lasso_scores + lasso_se,
                 lasso_scores - lasso_se, alpha=0.2)
plt.ylabel('CV score +/- std error')
plt.xlabel('alpha (l1 penalty)')
plt.axhline(np.max(lasso_scores), linestyle='--', color='.5')
plt.xlim([l1_space[0], l1_space[-1]])
plt.show()


##### Log transformed p-values

# Compute log_lasso model and plot coefficients
log_lasso = linear_model.Lasso(normalize = False)
log_lasso_cv = GridSearchCV(log_lasso, l1_param, cv = folds)
log_lasso_cv.fit(X_train, log_y_train)
log_lasso_pred = log_lasso_cv.predict(X_test)

print('Best log_lasso (l1) penalty :\n', log_lasso_cv.best_params_)
print('Best model coefficients :\n', log_lasso_cv.best_estimator_.coef_)
print('Mean-squared error :', mean_squared_error(log_y_test, log_lasso_pred))
print('Variance explained (R^2) :', r2_score(log_y_test, log_lasso_pred))
print('Mean cross-validation scores:', log_lasso_cv.cv_results_['mean_test_score'])

plt.plot(range(len(X.columns)), log_lasso_cv.best_estimator_.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=90)
plt.show()

log_lasso_scores = log_lasso_cv.cv_results_['mean_test_score']
log_lasso_se = log_lasso_cv.cv_results_['std_test_score']
plt.semilogx(l1_space, log_lasso_scores)
plt.semilogx(l1_space, log_lasso_scores + log_lasso_se, 'b--')
plt.semilogx(l1_space, log_lasso_scores - log_lasso_se, 'b--')
plt.fill_between(l1_space, log_lasso_scores + log_lasso_se,
                 log_lasso_scores - log_lasso_se, alpha=0.2)
plt.ylabel('CV score +/- std error')
plt.xlabel('alpha (l1 penalty)')
plt.axhline(np.max(log_lasso_scores), linestyle='--', color='.5')
plt.xlim([l1_space[0], l1_space[-1]])
plt.show()



########## RIDGE REGRESSION ##########

# Compute ridge model and plot coefficients
ridge = linear_model.Ridge(normalize = False)
l2_space = np.logspace(-10, 10, 21)
l2_param = {'alpha': l2_space}

ridge_cv = GridSearchCV(ridge, l2_param, cv = folds)
ridge_cv.fit(X_train, y_train)
ridge_pred = ridge_cv.predict(X_test)


# Compute metrics and plots
print('Best Ridge (l2) penalty :\n', ridge_cv.best_params_)
print('Best model coefficients :\n', ridge_cv.best_estimator_.coef_)
print('Mean-squared error :', mean_squared_error(y_test, ridge_pred))
print('Variance explained (R^2) :', r2_score(y_test, ridge_pred))
print('Mean cross-validation scores:', ridge_cv.cv_results_['mean_test_score'])

plt.plot(range(len(X.columns)), ridge_cv.best_estimator_.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=90)
plt.show()

ridge_scores = ridge_cv.cv_results_['mean_test_score']
ridge_se = ridge_cv.cv_results_['std_test_score']
plt.semilogx(l2_space, ridge_scores)
plt.semilogx(l2_space, ridge_scores + ridge_se, 'b--')
plt.semilogx(l2_space, ridge_scores - ridge_se, 'b--')
plt.fill_between(l2_space, ridge_scores + ridge_se,
                 ridge_scores - ridge_se, alpha=0.2)
plt.ylabel('CV score +/- std error')
plt.xlabel('alpha (l2 penalty)')
plt.axhline(np.max(ridge_scores), linestyle='--', color='.5')
plt.xlim([l2_space[0], l2_space[-1]])
plt.show()


##### Log transformed p-values

# Compute log_ridge model and plot coefficients
log_ridge = linear_model.Ridge(normalize = False)
log_ridge_cv = GridSearchCV(log_ridge, l2_param, cv = folds)
log_ridge_cv.fit(X_train, log_y_train)
log_ridge_pred = log_ridge_cv.predict(X_test)

print('Best log_ridge (l2) penalty :\n', log_ridge_cv.best_params_)
print('Best model coefficients :\n', log_ridge_cv.best_estimator_.coef_)
print('Mean-squared error :', mean_squared_error(log_y_test, log_ridge_pred))
print('Variance explained (R^2) :', r2_score(log_y_test, log_ridge_pred))
print('Mean cross-validation scores:', log_ridge_cv.cv_results_['mean_test_score'])

plt.plot(range(len(X.columns)), log_ridge_cv.best_estimator_.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=90)
plt.show()

log_ridge_scores = log_ridge_cv.cv_results_['mean_test_score']
log_ridge_se = log_ridge_cv.cv_results_['std_test_score']
plt.semilogx(l2_space, log_ridge_scores)
plt.semilogx(l2_space, log_ridge_scores + log_ridge_se, 'b--')
plt.semilogx(l2_space, log_ridge_scores - log_ridge_se, 'b--')
plt.fill_between(l2_space, log_ridge_scores + log_ridge_se,
                 log_ridge_scores - log_ridge_se, alpha=0.2)
plt.ylabel('CV score +/- std error')
plt.xlabel('alpha (l2 penalty)')
plt.axhline(np.max(log_ridge_scores), linestyle='--', color='.5')
plt.xlim([l2_space[0], l2_space[-1]])
plt.show()



########## ELASTIC NET ##########

# Compute Elastic net model and plot coefficients
enet = linear_model.ElasticNet(normalize = False)
l1_ratios = np.linspace(0, 1, 21)
enet_param = {'l1_ratio': l1_ratios}

enet_cv = GridSearchCV(enet, enet_param, cv = folds)
enet_cv.fit(X_train, y_train)
enet_pred = enet_cv.predict(X_test)


# Compute metrics and plots
print('Best Elastic Net penalty (l1 ratio) :\n', enet_cv.best_params_)
print('Best model coefficients :\n', enet_cv.best_estimator_.coef_)
print('Mean-squared error :', mean_squared_error(y_test, enet_pred))
print('Variance explained (R^2) :', r2_score(y_test, enet_pred))
print('Mean cross-validation scores:', enet_cv.cv_results_['mean_test_score'])

plt.plot(range(len(X.columns)), enet_cv.best_estimator_.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=90)
plt.show()

enet_scores = enet_cv.cv_results_['mean_test_score']
enet_se = enet_cv.cv_results_['std_test_score']
plt.plot(l1_ratios, enet_scores)
plt.plot(l1_ratios, enet_scores + enet_se, 'b--')
plt.plot(l1_ratios, enet_scores - enet_se, 'b--')
plt.fill_between(l1_ratios, enet_scores + enet_se,
                 enet_scores - enet_se, alpha=0.2)
plt.ylabel('CV score +/- std error')
plt.xlabel('alpha (l1 ratio)')
plt.axhline(np.max(enet_scores), linestyle='--', color='.5')
plt.xlim([l1_ratios[0], l1_ratios[-1]])
plt.show()


##### Log transformed p-values

# Compute log_enet model and plot coefficients
log_enet = linear_model.ElasticNet(normalize = False)
log_enet_cv = GridSearchCV(log_enet, enet_param, cv = folds)
log_enet_cv.fit(X_train, log_y_train)
log_enet_pred = log_enet_cv.predict(X_test)

print('Best log_enet (l2) penalty :\n', log_enet_cv.best_params_)
print('Best model coefficients :\n', log_enet_cv.best_estimator_.coef_)
print('Mean-squared error :', mean_squared_error(log_y_test, log_enet_pred))
print('Variance explained (R^2) :', r2_score(log_y_test, log_enet_pred))
print('Mean cross-validation scores:', log_enet_cv.cv_results_['mean_test_score'])

plt.plot(range(len(X.columns)), log_enet_cv.best_estimator_.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=90)
plt.show()

log_enet_scores = log_enet_cv.cv_results_['mean_test_score']
log_enet_se = log_enet_cv.cv_results_['std_test_score']
plt.semilogx(l1_ratios, log_enet_scores)
plt.semilogx(l1_ratios, log_enet_scores + log_enet_se, 'b--')
plt.semilogx(l1_ratios, log_enet_scores - log_enet_se, 'b--')
plt.fill_between(l1_ratios, log_enet_scores + log_enet_se,
                 log_enet_scores - log_enet_se, alpha=0.2)
plt.ylabel('CV score +/- std error')
plt.xlabel('alpha (l1 ratio)')
plt.axhline(np.max(log_enet_scores), linestyle='--', color='.5')
plt.xlim([l1_ratios[0], l1_ratios[-1]])
plt.show()



# ============= STATSMODEL =============== #

# Split data into test and train sets
train_data, test_data = train_test_split(data, 
                                         test_size = 0.3,
                                         random_state = 21)


from statsmodels.formula.api import glm
import statsmodels.api as sm

formula = 'p_value ~' + processed_cols[0]
for index in range(1, len(processed_cols)):
    formula = formula + '+' + str(processed_cols[index])

## Test C() function in glm on unprocessed categorical variables!!!
#
#formula = y_train.name + '~' + processed_cols[0]
#for index in range(1, len(processed_cols)):
#    if data[processed_cols[index]].dtype != np.float:
#        formula = formula + '+C(' + str(processed_cols[index] + ')')
#    else:
#        formula = formula + '+' + str(processed_cols[index])

glma = glm(formula = formula, 
          data = train_data, 
          family = sm.families.Gaussian()).fit()

print('Linear model coefficients :\n',glma.params)
print('Summary :\n',glma.summary())


# Log transformed p-values
log_formula = 'log_p_val ~' + processed_cols[0]
for index in range(1, len(processed_cols)):
    log_formula = log_formula + '+' + str(processed_cols[index])

glmb = glm(formula = log_formula, 
          data = train_data, 
          family = sm.families.Gaussian()).fit()

print('Linear model coefficients :\n',glmb.params)
print('Summary :\n',glmb.summary())


# INTERCEPT IS NOT THE SAME AS IN SCIKIT LEARN?!?!


# ============= NEURAL NETWORK =============== #

# Keras neural network
from keras.layers import Input, Dense, Embedding, Flatten
from keras.models import Model
from keras.utils import plot_model


# Create layers
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)


# Build model and plot visual image of model
nn_model = Model(input_tensor, output_tensor)
nn_model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
print(nn_model.summary())

#plot_model(nn_model, to_file=path+'nn_model.png')
#image = plt.imread(path+'nn_model.png')
#plt.imshow(data)
#plt.show()


## Fit and evaluate model
#nn_model.fit(------, y, 
#              epochs=1, batch_size=128, validation_split=0.1, verbose=True)




