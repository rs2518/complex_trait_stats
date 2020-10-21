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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import (load_dataframe, process_category,
                                       metrics, plot_coefs, plot_true_vs_pred)

import time

# Load data and add column of ones for intercept
df = load_dataframe(RAW_DATA)
data = process_category(df)

X = data.drop(['p_value'], axis=1)
X_int = sm.add_constant(X, prepend=True).rename({"const":"Intercept"}, axis=1)
y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
y.columns = ["p_value", "-log10_p"]
X_train, X_test, y_train, y_test = \
    train_test_split(X_int.values, y.values, test_size=0.3,
                     random_state=1010)


# Time model(s)
t0 = time.time()

# Check that statsmodels and sklearn agree
models = {}
index = []
scores = pd.DataFrame()

for i in range(y.shape[1]):
        
    classifiers = dict(sklearn=LinearRegression(fit_intercept=False),
                       statsmodels=sm.OLS(y_train[:,i], X_train))
    for pkg, clf in classifiers.items():
        if hasattr(clf, "df_model"):
            res = clf.fit()
            coef = res.params            
            conf_int = res.conf_int(alpha=0.05)
        else:
            res = clf.fit(X_train, y_train[:,i])
            coef = res.coef_            
            conf_int = None
        
        # Plot coefficients with error bars (if available) 
        plot_coefs(coef, X_int.columns, conf_int=conf_int, cmap="signif")
        
        model_id = pkg+" "+y.columns[i]
        models[model_id] = res
        index.append(model_id)
        y_pred = res.predict(X_test)
        scores = scores.append(metrics(y_test[:,i], y_pred), ignore_index=True)

scores.index = index


t1 = time.time()
print("Running time : {:.2f} seconds".format(t1 - t0))
# ~0.59 seconds


# # statsmodels results match with sklearn
# np.testing.assert_array_almost_equal(
#     models["statsmodels -log10_p"].params, models["sklearn -log10_p"].coef_)



# Select results from a single package and perform model diagnostics on
# models for raw p-values and log-transformed p-values
for key in models.keys():
    pkg = "sklearn"
    if pkg in key:
        y_pred = models[key].predict(X_test)
        
        col = key[len(pkg)+1:]
        title = " ".join((r"$y_{test}$", "vs",
                         "$y_{predicted}$ (%s)" % (col)))
        ind = y.columns.to_list().index(col)
        plot_true_vs_pred(y_test[:,ind], y_pred, title=title)



# =============================================================================
# Further testing
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from complex_trait_stats.models._linear_regression import linear_regression
from complex_trait_stats.utils import (process_data, RAW_DATA, load_dataframe,
                                       plot_true_vs_pred)



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



show_time = True

lr = linear_regression(X_train, y_train, return_fit_time=show_time)

print(lr.coef_)
fig = plot_true_vs_pred(y_test, lr.predict(X_test))