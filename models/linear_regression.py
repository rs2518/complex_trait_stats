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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import (load_dataframe, process_category,
                                       metrics, plot_coefs)


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

# # statsmodels results match with sklearn
# np.testing.assert_array_almost_equal(
#     models["statsmodels -log10_p"].params, models["sklearn -log10_p"].coef_)



# ax.plot(lims, lims, ls="--", c="red")
# plt.title('True vs. Predicted')
# plt.ylabel('y_pred')
# plt.xlabel('y_true')

# plt.show()