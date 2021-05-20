import os

import numpy as np

from sklearn.model_selection import train_test_split

from complex_trait_stats.models._linear_regression import linear_regression
from complex_trait_stats.models._partial_least_sq import pls_regression
from complex_trait_stats.models._random_forest import random_forest
from complex_trait_stats.models._neural_network import multilayer_perceptron
from complex_trait_stats.models._penalised_regression import (lasso_regression,
                                                              ridge_regression,
                                                              enet_regression)
    
from complex_trait_stats.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
from complex_trait_stats.utils import (load_dataframe,
                                       process_data,
                                       create_directory,
                                       save_models)



# Create directory for figures
path = os.path.join(ROOT, "figures")
create_directory(path)

# Load data and split data into training and testing sets
df = load_dataframe(RAW_DATA)
data = process_data(df)
X = data.drop(['p_value'], axis=1)
y = -np.log10(data["p_value"])
X_train, X_test, y_train, y_test = train_test_split(X, y, **TRAIN_TEST_PARAMS)



# =============================================================================
# Train models
# =============================================================================

# Model tuning
# ------------
# Set seed and n_jobs. Print fit times
seed = 1010
show_time = True
n_jobs = -2


# Linear Regression
# -----------------
lr = linear_regression(X_train, y_train, return_fit_time=show_time)


# Penalised Regression (LASSO, Ridge, Elastic-Net)
# ------------------------------------------------
en_params = dict(alpha=np.logspace(-5, 2, 8),
                 l1_ratio=np.linspace(0, 1, 6))
pr_params = {k:v for k, v in en_params.items() if k == "alpha"}

lasso = lasso_regression(X_train, y_train, param_grid=pr_params, 
                         n_jobs=n_jobs, random_state=seed,
                         return_fit_time=show_time, warm_start=True)
ridge = ridge_regression(X_train, y_train, param_grid=pr_params,
                         n_jobs=n_jobs, random_state=seed,
                         return_fit_time=show_time)
enet = enet_regression(X_train, y_train, param_grid=en_params,
                       n_jobs=n_jobs, random_state=seed,
                       return_fit_time=show_time, warm_start=True)


# PLS Regression
# --------------
pls_params = dict(n_components=np.arange(1, X.shape[1]+1))

pls = pls_regression(X_train, y_train, param_grid=pls_params,
                     n_jobs=n_jobs, return_fit_time=show_time)


# Random Forest
# -------------
rf_params = dict(n_estimators=[10, 100, 250],
                 max_features=["auto", "sqrt"],
                 max_depth=[10, 25, 50],
                 min_samples_split=[0.001, 0.01, 0.1],
                 min_samples_leaf=[0.001, 0.01, 0.1])

rf = random_forest(X_train, y_train, param_grid=rf_params, n_iter=1,
                   n_jobs=n_jobs, random_state=seed,
                   return_fit_time=show_time, warm_start=True)


# Multilayer Perceptron
# ---------------------
one_layer_params = dict(hidden_layers=[1],
                        first_neurons=[1, 10, 25],
                        hidden_neurons=[None],
                        activation=["relu"],
                        last_activation=[None],
                        dropout=[0.01, 0.1],
                        l1=[1e-04],
                        l2=[1e-04],
                        epochs=[20],
                        batch_size=[100])    # Single hidden layer
multi_layer_params = dict(hidden_layers=[2, 3],
                          first_neurons=[1, 10, 25],
                          hidden_neurons=[1, 10, 25],
                          activation=["relu"],
                          last_activation=[None],
                          dropout=[0.01, 0.1],
                          l1=[1e-04],
                          l2=[1e-04],
                          epochs=[20],
                          batch_size=[100])    # Multiple hidden layers

mlp_params = [one_layer_params, multi_layer_params]
mlp = multilayer_perceptron(X_train, y_train, param_grid=mlp_params, n_iter=1,
                            n_jobs=n_jobs, random_state=seed,
                            return_fit_time=show_time)


# Create lists of tuned models
models = [lr,
          lasso.best_estimator_,
          ridge.best_estimator_,
          enet.best_estimator_,
          pls.best_estimator_,
          rf.best_estimator_,
          mlp.best_estimator_]

# Save models
save_models(models)