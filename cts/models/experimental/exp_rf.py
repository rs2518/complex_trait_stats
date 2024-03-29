# =============================================================================
# TEMPORARY IMPORT FOR USE IN AN INTERPRETER
# =============================================================================
import os, sys

directory = "Desktop/Term 3 MSc Project"

path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import (load_dataframe, process_category,
                                       metrics, cv_table, plot_true_vs_pred)

import time



# =============================================================================
# Initial model testing
# =============================================================================

# Load data and add column of ones for intercept
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

params = {
    "n_estimators": [100, 500, 1000],
    "max_features": ["auto", "sqrt"],
    "max_depth": [5, 10, 50],
    "min_samples_split": [0.15, 0.25, 0.35],
    "min_samples_leaf": [0.01, 0.1],
    "bootstrap": [True, False]
}

# Tune hyperparameters using cross-validation. Run model for both the raw
# p-values and log-transformed p-values
folds = 5
models = {}
index = []
scores = pd.DataFrame()

for i in range(y.shape[1]):
    rf = RandomForestRegressor(random_state=1010, n_jobs=-2)
    rf_cv = RandomizedSearchCV(estimator=rf,
                               param_distributions=params,
                               n_iter=10,
                               cv=folds,
                               random_state=1010,
                               return_train_score=True)
    rf_cv.fit(X_train, y_train[:,i])
    
    model_id = "rf "+y.columns[i]
    models[model_id] = rf_cv
    index.append(model_id)
    
    y_pred = rf_cv.best_estimator_.predict(X_test)
    scores = scores.append(metrics(y_test[:,i], y_pred), ignore_index=True)        
        
scores.index = index


t1 = time.time()
print("Running time : {:.2f} seconds".format(t1 - t0))
# 300 seconds (5 mins)


# Plot variable importances for raw p-values and log-transformed p-values.
# NOTE: 'feature_importance' is not considered very accurate when dealing with
# high cardinality features. In this case, 'permutation_importance' is usually
# preferred
def plot_feature_importance(importances, feature_names, ordered=None,
                            title=None, **kwargs):
    """Plots bar graph of feature importances
    
    NOTE: Feature importances often do not perform well for high cardinality
    features
    """
    if title is None:
        title = "Random Forest Feature Importances"
    if ordered is None:
        sorted_idx = np.arange(len(importances)-1, -1, step=-1)
    elif ordered == "ascending":
        sorted_idx = importances.argsort()
    elif ordered == "descending":
        sorted_idx = (-importances).argsort()
    
    y_ticks = np.arange(0, len(feature_names))
    
    fig, ax = plt.subplots()
    ax.barh(y_ticks, importances[sorted_idx], **kwargs)
    ax.set_xlim(0,1)
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title(title)


# Use rainbow colormap
cmap = sns.hls_palette(X.shape[1], l=.55)
for i, key in enumerate(models.keys()):
    importances = models[key].best_estimator_.feature_importances_
    title = "Random Forest Feature Importance: '{}'".format(
        key[len("rf "):])
    plot_feature_importance(importances=importances, feature_names=X.columns,
                            title=title, color=cmap)
    
# fig.tight_layout()
# plt.show()


# Analyse cross-validation results stability of hyperparameter selection
cv_tables = {key:cv_table(models[key].cv_results_, ordered="ascending")
              for key in models.keys()}



# Perform model diagnostics on raw p-values and log-transformed p-values
for key in models.keys():
    y_pred = models[key].predict(X_test)
    
    col = key[len("rf")+1:]
    title = " ".join((r"$y_{test}$", "vs",
                     "$y_{predicted}$ (%s)" % (col)))
    ind = y.columns.to_list().index(col)
    plot_true_vs_pred(y_test[:,ind], y_pred, title=title)
    


# =============================================================================
# Hyperparameter testing
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from complex_trait_stats.models._random_forest import random_forest
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


y_train = Y_train["p_value"]
y_test = Y_test["p_value"]


seed = 1
show_time = True

rf_params = dict(n_estimators=[10, 100, 1000],
                 max_features=["auto", "sqrt"],
                 max_depth=[100],
                 min_samples_split=[2, 0.01, 0.1],
                 min_samples_leaf=[1, 0.01, 0.1],
                 bootstrap=[True])

rf = random_forest(X_train, y_train, param_grid=rf_params, n_iter=27,
                   random_state=seed, return_fit_time=show_time)

print(rf.cv_results_)
fig = plot_true_vs_pred(y_test, rf.best_estimator_.predict(X_test))
cv_tab = cv_table(rf.cv_results_, ordered="ascending")


# -------------
# NOTE: small min_samples_split/min_samples_leaf may lead to overfitting for
# log transformed target (high training score, low testing score)

y_train = Y_train["-log10_p"]
y_test = Y_test["-log10_p"]


seed = 1
show_time = True

# rf_params = dict(n_estimators=[10],
#                  max_features=["auto"],
#                  max_depth=[10],
#                  min_samples_split=[0.01],
#                  min_samples_leaf=[0.01],
#                  bootstrap=[True])

rf_params = dict(n_estimators=[250],
                  max_features=["auto"],
                  max_depth=[50],
                  min_samples_split=[0.001],
                  min_samples_leaf=[0.001])

# rf_params = dict(n_estimators=[10, 100, 1000],
#                   max_features=["auto", "sqrt"],
#                   max_depth=[10, 25, 50],
#                   min_samples_split=[0.001, 0.01, 0.1],
#                   min_samples_leaf=[0.001, 0.01, 0.1])

rf = random_forest(X_train, y_train, param_grid=rf_params, n_iter=1,
                   random_state=seed, return_fit_time=show_time, warm_start=True)

print(rf.cv_results_)
fig = plot_true_vs_pred(y_test, rf.best_estimator_.predict(X_test))
cv_tab = cv_table(rf.cv_results_, ordered="ascending")