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
# import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from complex_trait_stats.models._linear_regression import linear_regression
from complex_trait_stats.models._partial_least_sq import pls_regression
from complex_trait_stats.models._random_forest import random_forest
from complex_trait_stats.models._neural_network import (multilayer_perceptron,
                                                        _create_mlp)
from complex_trait_stats.models._penalised_regression import (lasso_regression,
                                                              ridge_regression,
                                                              enet_regression)
    
from complex_trait_stats.utils import (ROOT, RAW_DATA, load_dataframe,
                                       process_data)
from complex_trait_stats.utils import (coef_dict,
                                       coef_stats_dict,
                                       plot_stability,
                                       plot_mean_coef_heatmap,
                                       validate_models,
                                       perm_importances,
                                       plot_coefs,
                                       metrics,
                                       plot_true_vs_pred,
                                       cv_table)



# Define directories (and create if non-existent) to save plots
fig_dir = os.path.join(ROOT, "figures")
stab_figpath = os.path.join(fig_dir, "stability")
eval_figpath = os.path.join(fig_dir, "evaluation")
eda_figpath = os.path.join(fig_dir, "exploratory")
for folder in [fig_dir, stab_figpath, eval_figpath, eda_figpath]:
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("Created '{}' directory!".format(folder[folder.rfind("/")+1:]))


# Load data and add column of ones for intercept
df = load_dataframe(RAW_DATA)
# df = load_dataframe("snp_raw_allchr1000.csv")
data = process_data(df)

X = data.drop(['p_value'], axis=1)
# Y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
# Y.columns = ["p_value", "-log10_p"]
# X_train, X_test, Y_train, Y_test = \
#     train_test_split(X, Y, test_size=0.3, random_state=1010)


y = -np.log10(data["p_value"])
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=1010)




# =============================================================================
# Train models
# =============================================================================
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
# ElasticNet takes VERY long. Using warm_start


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
# 162 possible combinations. Test ~ 25% of hyperparameter space

rf = random_forest(X_train, y_train, param_grid=rf_params, n_iter=5,
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

mlp = multilayer_perceptron(X_train, y_train, param_grid=mlp_params, n_iter=5,
                            n_jobs=n_jobs, random_state=seed,
                            return_fit_time=show_time)


# Get cross-validation results
sort = "ascending"
lasso_cv = cv_table(lasso.cv_results_, ordered=sort)
ridge_cv = cv_table(ridge.cv_results_, ordered=sort)
enet_cv = cv_table(enet.cv_results_, ordered=sort)
pls_cv = cv_table(pls.cv_results_, ordered=sort)
rf_cv = cv_table(rf.cv_results_, ordered=sort)
mlp_cv = cv_table(mlp.cv_results_, ordered=sort)


# Get best model hyperparameters from tuned models
best_lasso_params = lasso.best_params_
best_ridge_params = ridge.best_params_
best_enet_params = enet.best_params_
best_pls_params = pls.best_params_
best_rf_params = rf.best_params_
best_mlp_params = mlp.best_params_

# Create lists of tuned models (fitted and unfitted)
fitted_models = [lr,
                 lasso.best_estimator_,
                 ridge.best_estimator_,
                 enet.best_estimator_,
                 pls.best_estimator_,
                 rf.best_estimator_,
                 mlp.best_estimator_]

unfitted_models = [LinearRegression(),
                   Lasso().set_params(**best_lasso_params),
                   Ridge().set_params(**best_ridge_params),
                   ElasticNet().set_params(**best_enet_params),
                   PLSRegression().set_params(**best_pls_params),
                   RandomForestRegressor(**best_rf_params),
                   KerasRegressor(build_fn=_create_mlp, verbose=0,
                                  **best_mlp_params)]



# =============================================================================
# Stability analysis
# =============================================================================

# Feature stability
# -----------------
# Produce stability plots for linear models
fs_seed = 10
coefs = coef_dict(estimators=unfitted_models[:-2],
                  X=X_train, Y=y_train,
                  n_iters=10, bootstrap=True,
                  random_state=fs_seed)
coef_stats = coef_stats_dict(coefs)

for model, coef in coefs.items():
    fig = plot_stability(coef, title=model)
    
    name = model.lower() + "_stability_plot.png"
    figpath = os.path.join(stab_figpath, name)
    fig.savefig(figpath)

# Plot mean coefficients across all linear models
fig = plot_mean_coef_heatmap(coefs)
fig.savefig(os.path.join(stab_figpath, "mean_coef_heatmap.png"),
            bbox_inches = "tight")



# Positive control and negative control model validation
# X_val, X_testing, Y_val, Y_testing = \
#     train_test_split(X_test, Y_test, test_size=0.8, random_state=1010)

n_repeats = 10
mv_seed = 1

pos_ctrl = validate_models(estimators=unfitted_models, X=X_test, y=y_test,
                           n_repeats=n_repeats, random_state=mv_seed)
 
neg_ctrl = validate_models(estimators=unfitted_models, X=X_test, y=y_test,
                           n_repeats=n_repeats, random_state=mv_seed,
                           control_params={"positive_control":False})


# Permutation importances for each feature
perms = perm_importances(fitted_models, X_test, y_test, n_repeats=n_repeats,
                         random_state=mv_seed)


# Hyperparameter stability
# ------------------------
# # Plot ranked results of cross-validation as a table
# for model_cv in [lasso, ridge, enet, pls, rf, mlp]:
#     table = cv_table(model_cv.cv_results_, ordered="ascending")



# =============================================================================
# Model diagnostics
# =============================================================================

# Get model predictions
predictions = {key:model.predict(X_test).ravel()
               for (key, model) in zip(perms.keys(), fitted_models)}


# Plot true values vs. predictions and metric tables for all models
metric_df = pd.DataFrame()
for key, y_pred in predictions.items():
    
    fig = plot_true_vs_pred(y_test, y_pred, title=key)
    
    name = key.lower() + "_true_vs_pred_plot.png"
    figpath = os.path.join(eval_figpath, name)
    fig.savefig(figpath)
    
    # Metrics table data
    metric_df = metric_df.append(metrics(y_test, y_pred), ignore_index=True)

metric_df.index = predictions.keys()
# metrics_table = plt.table(cellText=metric_df)


# Plot linear model coefficients
for model in fitted_models[:-2]:
    
    model_name = type(model).__name__
    fig = plot_coefs(model.coef_.flatten(),
                     X_train.columns.to_list(),
                     title=model_name,
                     cmap="rainbow")
    name = model_name.lower() + "_coefficient_plot.png"
    figpath = os.path.join(eval_figpath, name)
    fig.savefig(figpath)