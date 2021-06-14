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
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_decomposition import PLSRegression

from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import (load_dataframe, process_category,
                                       metrics, plot_coefs)

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

plot_coefs(pls_cv.best_estimator_.coef_, X.columns, cmap="rainbow")

# PLSCanonical()

t1 = time.time()
print("Running time : {:.2f} seconds".format(t1 - t0))
# ~3 seconds



# =============================================================================
# Hyperparameter testing
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from complex_trait_stats.models._partial_least_sq import pls_regression
from complex_trait_stats.utils import (process_data, RAW_DATA, load_dataframe,
                                       plot_true_vs_pred, cv_table)



# Load data and add column of ones for intercept
df = load_dataframe(RAW_DATA)
# df = load_dataframe("snp_raw_allchr1000.csv")
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
show_time = True

pls_params = dict(n_components=np.arange(1, X.shape[1]+1))

pls = pls_regression(X_train, y_train, param_grid=pls_params,
                     return_fit_time=show_time)

print(pls.cv_results_)
fig = plot_true_vs_pred(y_test, pls.best_estimator_.predict(X_test).ravel())
cv_tab = cv_table(pls.cv_results_, ordered="ascending")