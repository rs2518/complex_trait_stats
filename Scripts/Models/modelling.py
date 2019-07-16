#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:34:07 2019

@author: raphaelsinclair
"""


# ============= DATA PREPARATION =============== #

# Set directory and Load dataframes
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

data = pd.read_pickle(
        os.path.join(home, directory) + 'processed_data_sample.pkl')
int_df = pd.read_pickle(
        os.path.join(home, directory) + 'integrated_data_sample.pkl')
# NOTE: int_df more suitable for statsmodel


# Set inputs/targets then split data into test and train sets
# (X, y split)
from sklearn.model_selection import train_test_split

processed_cols = [col for col in data.columns.to_list() if col not in int_df.columns.to_list()]

X = data[processed_cols]
y = data['p_value']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 21)

# Log-transformed targets
log_y_train = -np.log(y_train)
log_y_test = -np.log(y_test)


# (Integrated data split)
train_data, test_data = train_test_split(data, 
                                         test_size = 0.3,
                                         random_state = 21)

train_intdf, test_intdf = train_test_split(int_df, 
                                         test_size = 0.3,
                                         random_state = 21)



# ============= SCIKIT LEARN MODELS =============== #


########## LINEAR REGRESSION ##########

# Scikit-learn linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sk_linear = LinearRegression()
sk_linear.fit(X_train, y_train)
sk_linear_pred = sk_linear.predict(X_test)

sk_linear_coefs = pd.DataFrame(np.append(sk_linear.intercept_,
                                              sk_linear.coef_),
                           index = ['intercept'] +list(X.columns), 
                           columns = ['Coefficients'])
print('Linear model coefficients :\n', sk_linear_coefs)
print('Mean-squared error :', mean_squared_error(y_test, sk_linear_pred))
print('Variance explained (R^2) :' , r2_score(y_test, sk_linear_pred))

#plt.scatter(x = X_test.index, y = (y_test - sk_linear_pred), c = 'g')
#plt.show()     # Plot not useful
sns.distplot((y_test - sk_linear_pred))
plt.show()     # Residuals should be normally distributed



# Log transformed p-values
log_sk_linear = LinearRegression()
log_sk_linear.fit(X_train, log_y_train)
log_sk_lin_pred = log_sk_linear.predict(X_test)

log_sk_lin_coefs = pd.DataFrame(np.append(log_sk_linear.intercept_,
                                      log_sk_linear.coef_),
                           index = ['intercept'] +list(X.columns), 
                           columns = ['Coefficients'])
print('Linear model coefficients (log-transformed) :\n', log_sk_lin_coefs)
print('Mean-squared error :', mean_squared_error(log_y_test, log_sk_lin_pred))
print('Variance explained (R^2) :', r2_score(log_y_test, log_sk_lin_pred))


#plt.scatter(x = X_test.index, y = (log_y_test - log_sk_lin_pred), c = 'g')
#plt.show()     # Plot not useful
sns.distplot((log_y_test - log_sk_lin_pred))
plt.show()     # Residuals should be normally distributed



########## LASSO REGRESSION ##########

### NOTE: SET MAX ITERATION BEFORE RUNNING IN HPC (JUST IN CASE)
# maxiter = 10000

### NOTE: MAY WANT TO TRY CV VERSIONS OF RIDGE/LASSO/E-NET BECAUSE OF
# ADDITIONAL SUPPORT FEATURES (STORING SCORES WITHOUT LOOPS)


# Set folds for crossvalidation
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
folds = 5


# Scikit-learn LASSO regression
lasso = Lasso(normalize = False)
l1_space = np.logspace(-10, 10, 21)
l1_param = {'alpha': l1_space}

lasso_cv = GridSearchCV(lasso, l1_param, cv = folds)
lasso_cv.fit(X_train, y_train)
lasso_pred = lasso_cv.predict(X_test)

print('Best LASSO (l1) penalty :\n', lasso_cv.best_params_)
print('Best model coefficients :\n', lasso_cv.best_estimator_.coef_)
print('Mean-squared error :', mean_squared_error(y_test, lasso_pred))
print('Variance explained (R^2) :', r2_score(y_test, lasso_pred))
print('Mean cross-validation scores:', lasso_cv.cv_results_['mean_test_score'])

plt.plot(range(len(X.columns)), lasso_cv.best_estimator_.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=90)
plt.show()


plt.figure()
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



# Log transformed p-values
log_lasso = Lasso(normalize = False)
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

# Scikit-learn Ridge regression
from sklearn.linear_model import Ridge

ridge = Ridge(normalize = False)
l2_space = np.logspace(-10, 10, 21)
l2_param = {'alpha': l2_space}

ridge_cv = GridSearchCV(ridge, l2_param, cv = folds)
ridge_cv.fit(X_train, y_train)
ridge_pred = ridge_cv.predict(X_test)

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



# Log transformed p-values
log_ridge = Ridge(normalize = False)
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

# Scikit-learn Elastic Net regression
from sklearn.linear_model import ElasticNet

enet = ElasticNet(normalize = False)
l1_ratios = np.linspace(0, 1, 21)
enet_param = {'l1_ratio': l1_ratios}

enet_cv = GridSearchCV(enet, enet_param, cv = folds)
enet_cv.fit(X_train, y_train)
enet_pred = enet_cv.predict(X_test)

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



# Log transformed p-values
log_enet = ElasticNet(normalize = False)
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


########## LINEAR REGRESSION (METHOD 1) ##########
# Categorical variables preprocessed as binary variables


# Statsmodels linear regression glm.Gaussian
from statsmodels.formula.api import glm, ols
import statsmodels.api as sm

formula = 'p_value ~' + processed_cols[0]
for index in range(1, len(processed_cols)):
    formula = formula + '+' + str(processed_cols[index])

glm_linear = glm(formula = formula, 
          data = train_data, 
          family = sm.families.Gaussian()).fit()

print('Linear model coefficients :\n',glm_linear.params)
print('Summary :\n',glm_linear.summary())



# Log transformed p-values
log_formula = 'log_p_val ~' + processed_cols[0]
for index in range(1, len(processed_cols)):
    log_formula = log_formula + '+' + str(processed_cols[index])

log_glm_linear = glm(formula = log_formula, 
          data = train_data, 
          family = sm.families.Gaussian()).fit()

print('Linear model coefficients :\n',log_glm_linear.params)
print('Summary :\n',log_glm_linear.summary())

# INTERCEPT IS NOT THE SAME AS IN SCIKIT LEARN?!?!
# BECAUSE OF DIFFERENT REFERENCES

########## LINEAR REGRESSION (METHOD 2) ##########
# Categorical variables not preprocessed

# NOTE: Method 2 does NOT work with glm function (PerfectSeparation error)

# Statsmodels linear regression ols
input_cols = [col for col in int_df.columns.to_list() if col not in ['SNP',
              'Allele', 'A1', 'A2', 'p_value', 'log_p_val', 'Position']]

formula2 = 'p_value ~' + input_cols[0]
for index in range(1, len(input_cols)):
    if int_df[input_cols[index]].dtype != np.float:
        formula2 = formula2 + '+C(' + str(input_cols[index] + ')')
    else:
        formula2 = formula2 + '+' + str(input_cols[index])

ols_linear = ols(formula = formula2, data = train_intdf).fit()

print('Linear model coefficients :\n', ols_linear.params)
print('Summary :\n', ols_linear.summary())



# Log transformed p-values
log_formula2 = 'log_p_val ~' + input_cols[0]
for index in range(1, len(input_cols)):
    if int_df[input_cols[index]].dtype != np.float:
        log_formula2 = log_formula2 + '+C(' + str(input_cols[index] + ')')
    else:
        log_formula2 = log_formula2 + '+' + str(input_cols[index])

log_ols_linear = ols(formula = log_formula2, data = train_intdf).fit()

print('Linear model coefficients :\n', log_ols_linear.params)
print('Summary :\n', log_ols_linear.summary())



# ============= KERAS (NEURAL NETWORK) =============== #

# Create layers
from keras.layers import Input, Dense, Embedding, Flatten
from keras.models import Sequential, Model

from keras.utils import plot_model

# Categorical embeddings
no_positions = np.unique(int_df['Position']).shape[0]
no_chr = np.unique(int_df['Chr_no']).shape[0]

position_lookup = Embedding(input_dim = no_positions,
                        output_dim = 1,
                        input_length = 1,
                        name = 'Team-Strength')



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




