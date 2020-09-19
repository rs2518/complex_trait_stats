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
# from sklearn.linear_model import LinearRegression
from complex_trait_stats.models._linear_regression import linear_regression
from complex_trait_stats.models._partial_least_sq import pls_regression
from complex_trait_stats.models._random_forest import random_forest
from complex_trait_stats.models._neural_network import multilayer_perceptron
from complex_trait_stats.models._penalised_regression import (lasso_regression,
                                                              ridge_regression,
                                                              enet_regression)
    
from complex_trait_stats.utils import (RAW_DATA, load_dataframe,
                                       process_category)
# from complex_trait_stats.utils import metrics, plot_coefs, plot_true_vs_pred



# Load data and add column of ones for intercept
df = load_dataframe(RAW_DATA)
data = process_category(df)

X = data.drop(['p_value'], axis=1)
Y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
Y.columns = ["p_value", "-log10_p"]
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=1010)


y_train = Y_train["p_value"]
y_test = Y_test["p_value"]
seed = 1010



# =============================================================================
# Train models
# =============================================================================

# Linear Regression
# -----------------
lr = linear_regression(X_train, y_train)


# Penalised Regression (LASSO, Ridge, Elastic-Net)
# ------------------------------------------------
en_params = dict(alpha=np.logspace(-5, 5, 11),
                 l1_ratio=np.linspace(0, 1, 6))
pr_params = {k:v for k, v in en_params.items() if k == "alpha"}

lasso = lasso_regression(X_train, y_train, param_grid=pr_params,
                         random_state=seed)
ridge = ridge_regression(X_train, y_train, param_grid=pr_params,
                         random_state=seed)
enet = enet_regression(X_train, y_train, param_grid=en_params,
                       random_state=seed)


# PLS Regression
# --------------
pls_params = dict(n_components=np.arange(X.shape[1]))

pls = pls_regression(X_train, y_train, param_grid=pls_params)


# Random Forest
# -------------
rf_params = dict(n_estimators=[100, 500, 1000],
                 max_features=["auto", "sqrt"],
                 max_depth=[5, 10, 50],
                 min_samples_split=[0.15, 0.25, 0.35],
                 min_samples_leaf=[0.01, 0.1],
                 bootstrap=[True, False])    # 216 possible combinations

rf = random_forest(X_train, y_train, param_grid=rf_params, n_iter=5,
                   random_state=seed)


# Multilayer Perceptron
# ---------------------
mlp_params = dict(hidden_layers=[2, 4, 8],
                  first_neurons=[25, 50, 100],
                  hidden_neurons=[5, 10, 25, 50],
                  activation=["relu", "softplus"],
                  last_activation=[None],
                  dropout=[0, 0.1, 0.2],
                  l1=[None, 1e-02, 1e-04],
                  l2=[None, 1e-02, 1e-04],
                  learning_rate=[1e-01],
                  epochs=[10, 50],
                  batch_size=[20, 80])    # 5832 possible combinations

mlp = multilayer_perceptron(X_train, y_train, param_grid=mlp_params, n_iter=5,
                            random_state=seed)



# =============================================================================
# Stability analysis
# =============================================================================

# Stability heatmap
linear_models = [lr, lasso, ridge, enet, pls]
nonlinear_models = [rf, mlp]
models = linear_models + nonlinear_models