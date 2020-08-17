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
                                       metrics, plot_coefs)


# Load data and log transform p-values
df = load_dataframe(RAW_DATA)
data = process_category(df)

X = data.drop(['p_value'], axis=1)
y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
y.columns = ["p_value", "-log10_p"]
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, test_size = 0.3,
                     random_state = 1010)


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


# Plot model coefficients
plot_coefs(models["lasso p_value"].best_estimator_.coef_, X.columns)
plot_coefs(models["ridge p_value"].best_estimator_.coef_, X.columns)
plot_coefs(models["enet p_value"].best_estimator_.coef_, X.columns)
plot_coefs(models["lasso -log10_p"].best_estimator_.coef_, X.columns)
plot_coefs(models["ridge -log10_p"].best_estimator_.coef_, X.columns)
plot_coefs(models["enet -log10_p"].best_estimator_.coef_, X.columns)

#### ElasticNet appears to favour LASSO model