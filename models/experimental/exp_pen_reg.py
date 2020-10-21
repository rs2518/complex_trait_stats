# =============================================================================
# TEMPORARY IMPORT FOR USE IN AN INTERPRETER
# =============================================================================
import os, sys

directory = "Desktop/Term 3 MSc Project"

path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet

from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import (load_dataframe, process_category,
                                       metrics, plot_coefs, cv_table,
                                       plot_true_vs_pred)

import time

# Load data and log transform p-values
df = load_dataframe(RAW_DATA)
data = process_category(df)

X = data.drop(['p_value'], axis=1)
y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
y.columns = ["p_value", "-log10_p"]
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, test_size = 0.3,
                     random_state = 1010)

# Time model(s)
t0 = time.time()

# Instantiate models and dictionary of parameter grids for each model
classifiers = dict(lasso=Lasso(random_state = 96, max_iter=10000),
                   ridge=Ridge(random_state = 96, max_iter=10000),
                   enet=ElasticNet(random_state = 96, max_iter=10000))

lasso_params = {"alpha":np.logspace(-5, 5, 11)}
ridge_params = {"alpha":np.logspace(-5, 5, 11)}
enet_params = {"alpha":np.logspace(-5, 5, 11),
               "l1_ratio":np.linspace(0, 1, 6)}

params = [lasso_params, ridge_params, enet_params]
param_grids = {model:params for model, params
               in zip(classifiers.keys(), params)}


# Tune alpha for each model using cross-validation. Run models for both the
# raw p-values and log-transformed p-values
folds = 5
max_iter = 10000
models = {}
index = []
scores = pd.DataFrame()

for i in range(y.shape[1]):
    for model, clf in classifiers.items():
        clf_cv = GridSearchCV(estimator=clf, param_grid=param_grids[model],
                              cv=folds, return_train_score=True)
        clf_cv.fit(X_train, y_train[:,i])
        
        model_id = model+" "+y.columns[i]
        models[model_id] = clf_cv
        index.append(model_id)
        
        y_pred = clf_cv.predict(X_test)
        scores = scores.append(metrics(y_test[:,i], y_pred), ignore_index=True)        
        
scores.index = index


t1 = time.time()
print("Running time : {:.2f} seconds".format(t1 - t0))
# ~38 seconds


# Plot model coefficients
for key in models.keys():
    plot_coefs(models[key].best_estimator_.coef_, X.columns)

#### ElasticNet appears to favour LASSO model


# Analyse cross-validation results stability of hyperparameter selection
cv_tables = {key:cv_table(models[key].cv_results_, ordered="ascending")
              for key in models.keys()}



# Perform model diagnostics on each penalised regression model on both the
# raw p-values and log-transformed p-values
for clf in classifiers.keys():
    for col in y.columns:
        y_pred = models[clf+" "+col].predict(X_test)
        title = clf+" (%s)" % (col)
        ind = y.columns.to_list().index(col)
        plot_true_vs_pred(y_test[:,ind], y_pred, title=title)



# =============================================================================
# Hyperparameter testing
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from complex_trait_stats.models._penalised_regression import (lasso_regression,
                                                              ridge_regression,
                                                              enet_regression)
from complex_trait_stats.utils import (process_data, RAW_DATA, load_dataframe,
                                       plot_true_vs_pred, cv_table)



# Load data and add column of ones for intercept
# df = load_dataframe(RAW_DATA)
df = load_dataframe("snp_raw_allchr1000.csv")
data = process_data(df)

X = data.drop(['p_value'], axis=1)
Y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
Y.columns = ["p_value", "-log10_p"]
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=1010)


# y_train = Y_train["p_value"]
# y_test = Y_test["p_value"]
y_train = Y_train["-log10_p"]
y_test = Y_test["-log10_p"]



seed = 1
n_jobs= -2
show_time = True

en_params = dict(alpha=np.logspace(-5, 2, 8),
                 l1_ratio=np.linspace(0, 1, 6))
pr_params = {k:v for k, v in en_params.items() if k == "alpha"}

# -----------------
# lasso = lasso_regression(X_train, y_train, param_grid=pr_params, 
#                          n_jobs=n_jobs, random_state=seed,
#                          return_fit_time=show_time)

lasso = lasso_regression(X_train, y_train, param_grid=pr_params, 
                         n_jobs=n_jobs, random_state=seed,
                         return_fit_time=show_time,
                         warm_start=True)

# print(lasso.cv_results_)
fig = plot_true_vs_pred(y_test, lasso.best_estimator_.predict(X_test))
lasso_tab = cv_table(lasso.cv_results_, ordered="ascending")

# -----------------
ridge = ridge_regression(X_train, y_train, param_grid=pr_params,
                         n_jobs=n_jobs, random_state=seed,
                         return_fit_time=show_time)

# print(ridge.cv_results_)
fig = plot_true_vs_pred(y_test, ridge.best_estimator_.predict(X_test))
ridge_tab = cv_table(ridge.cv_results_, ordered="ascending")

# -----------------
# enet = enet_regression(X_train, y_train, param_grid=en_params,
#                        n_jobs=n_jobs, random_state=seed,
#                        return_fit_time=show_time)

enet = enet_regression(X_train, y_train, param_grid=en_params,
                       n_jobs=n_jobs, random_state=seed,
                       return_fit_time=show_time,
                       warm_start=True)
# Warm start reduces running time but gives different results (could be from
# instability of parameter estimates)
 

# print(enet.cv_results_)
fig = plot_true_vs_pred(y_test, enet.best_estimator_.predict(X_test))
enet_tab = cv_table(enet.cv_results_, ordered="ascending")

# NOTE: Ridge favours larger alpha but LASSO favours lower alpha