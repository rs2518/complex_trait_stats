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
                                       model_validation,
                                       perm_importances,
                                       plot_coefs,
                                       metrics,
                                       plot_true_vs_pred,
                                       cv_table,
                                       tabulate_validation,
                                       tabulate_perm,
                                       plot_perm_importance,
                                       plot_rf_feature_importance,
                                       plot_neg_validation,
                                       plot_pos_validation)



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
data = process_data(df)


# Split data into training and testing sets
X = data.drop(['p_value'], axis=1)
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
fs_seed = 1
coefs = coef_dict(estimators=unfitted_models[:-2],
                  X=X_train, Y=y_train,
                  n_iters=20, bootstrap=True,
                  random_state=fs_seed, train_size=0.3)
coef_stats = coef_stats_dict(coefs)

for model, coef in coefs.items():
    fig = plot_stability(coef, title=model)
    
    name = model.lower() + "_stability_plot.png"
    figpath = os.path.join(stab_figpath, name)
    fig.savefig(figpath)

# Plot mean coefficients across all linear models
fig = plot_mean_coef_heatmap(coefs)
fig.savefig(os.path.join(stab_figpath, "mean_coef_heatmap.png"),
            bbox_inches="tight")

# Plot random forest importances
fig = plot_rf_feature_importance(forest=rf.best_estimator_,
                                 title="Random Forest feature importances",
                                 feature_names=X.columns,
                                 ordered="ascending")
fig.savefig(os.path.join(stab_figpath, "rf_importances.png"),
            bbox_inches="tight")



# Hyperparameter stability
# ------------------------
# Get cross-validation results
sort = "ascending"
lasso_cv = cv_table(lasso.cv_results_, ordered=sort)
ridge_cv = cv_table(ridge.cv_results_, ordered=sort)
enet_cv = cv_table(enet.cv_results_, ordered=sort)
pls_cv = cv_table(pls.cv_results_, ordered=sort)
rf_cv = cv_table(rf.cv_results_, ordered=sort)
mlp_cv = cv_table(mlp.cv_results_, ordered=sort)



# =============================================================================
# Model validation
# =============================================================================

# Negative control (permuted labels)
# ----------------------------------
# Set iterables and create barplot annotation function
n_samples = 3
sample_size = 0.3
n_repeats = 5
mv_seed = 1
scoring = "r2"
correction = "fdr_bh"

# Negative control validation over bootstrapped samples
neg_ctrl = {version:model_validation(estimators=fitted_models,
                                     X=X_test, y=y_test,
                                     scoring=scoring, n_samples=n_samples,
                                     sample_size=sample_size,
                                     n_repeats=n_repeats,
                                     positive_ctrl=False,
                                     random_state=mv_seed,
                                     version=version)
            for version in ["tpr", "fpr"]}

# Plot negative control results
neg_results = tabulate_validation(neg_ctrl, positive_ctrl=False,
                                  method=correction)
fig = plot_neg_validation(neg_results)
figpath = os.path.join(eval_figpath, "negative_control_validation.png")
fig.savefig(figpath)



# Positive control (perfectly correlated control feature)
# -------------------------------------------------------
# Positive control validation vs. noise over bootstrapped samples
noise_params = [0., 3., 5., 10., 15., 25.]

pos_ctrl = {"sigma="+str(noise):model_validation(estimators=unfitted_models,
                                                 X=X_test, y=y_test,
                                                 scoring=scoring,
                                                 n_samples=n_samples,
                                                 n_repeats=n_repeats,
                                                 sample_size=sample_size,
                                                 positive_ctrl=True,
                                                 random_state=mv_seed,
                                                 control_params={
                                                     "sigma":noise})
            for noise in noise_params}

# Plot positive control results
pos_results = tabulate_validation(pos_ctrl, positive_ctrl=True,
                                  method=correction)
fig = plot_pos_validation(pos_results, linestyle="--", marker="x")
figpath = os.path.join(eval_figpath, "positive_control_validation.png")
fig.savefig(figpath)



# Permutation importance
# ----------------------
# Permutation importances for each feature
perms = perm_importances(fitted_models, X_test, y_test, scoring=scoring,
                         n_samples=n_samples, n_repeats=n_repeats,
                         random_state=mv_seed)

# Plot permutation importances
perm_tab = tabulate_perm(perms, feature_names=X.columns, method=correction)
fig = plot_perm_importance(perm_tab, edgecolor="white", alpha=0.75)
figpath = os.path.join(eval_figpath, "permutation_importances.png")
fig.savefig(figpath)



# =============================================================================
# Model diagnostics
# =============================================================================

# Truths vs. predictions
# ----------------------
# Get model predictions
predictions = {key:model.predict(X_test).ravel()
               for (key, model) in zip(perm_tab.columns, fitted_models)}
pred_df = pd.DataFrame(predictions)
pred_df["Truths"] = y_test.values

# Plot true values vs. predictions for all models
fig = plot_true_vs_pred(pred_df,
                        scatter_kws=dict(alpha=0.5, edgecolor="w"),
                        line_kws=dict(linewidth=0.85))
figpath = os.path.join(eval_figpath, "true_vs_pred.png")
fig.savefig(figpath)



# Linear coefficient plots
# ------------------------
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