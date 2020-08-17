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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression, PLSCanonical

from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import (load_dataframe, process_category,
                                       metrics, plot_coefs)


# Load data and add column of ones for intercept
df = load_dataframe(RAW_DATA)
data = process_category(df)

X = data.drop(['p_value'], axis=1)
y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
y.columns = ["p_value", "-log10_p"]
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, test_size = 0.3,
                     random_state = 1010)


# Instantiate parameter dictionary
params = {"n_components" : np.arange(1, X.shape[1])}


# Tune number of components using cross-validation. Run model for both the
# raw p-values and log-transformed p-values
folds = 5
models = {}
index = []
scores = pd.DataFrame()

for i in range(y.shape[1]):
    pls_cv = GridSearchCV(estimator=PLSRegression(),
                          param_grid=params,
                          cv=folds, return_train_score=True)
    pls_cv.fit(X_train, y_train[:,i])
    
    model_id = "PLSRegression "+y.columns[i]
    models[model_id] = pls_cv
    index.append(model_id)
    
    y_pred = pls_cv.predict(X_test)
    scores = scores.append(metrics(y_test[:,i], y_pred), ignore_index=True)        
        
scores.index = index






# pls_c=PLSCanonical()