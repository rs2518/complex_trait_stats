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
                                       metrics)

import time

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
# ccp_alpha=0.0,

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
    
    model_id = "Random Forest: "+y.columns[i]
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
        key[len("Random Forest "):])
    plot_feature_importance(importances=importances, feature_names=X.columns,
                            title=title, color=cmap)
    
# fig.tight_layout()
# plt.show()