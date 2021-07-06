import os

import numpy as np

from sklearn.model_selection import train_test_split

from cts.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
from cts.utils import (load_dataframe,
                       process_data,
                       create_directory,
                       load_models,
                       coef_dict,
                       coef_stats_dict,
                       plot_stability,
                       plot_mean_coef_heatmap,
                       plot_coefs,
                       cv_table)



# Create directory for figures
path = os.path.join(ROOT, "figures", "stability")
create_directory(path)

# Load data and split data into training and testing sets
df = load_dataframe(RAW_DATA)
data = process_data(df)
X = data.drop(['p_value'], axis=1)
y = -np.log10(data["p_value"])
X_train, X_test, y_train, y_test = train_test_split(X, y, **TRAIN_TEST_PARAMS)

# Load models
fitted_models = load_models()
unfitted_models = load_models(fitted=False)
linear_fitted_models = {k:v for k, v in fitted_models.items() 
                        if k not in ["Random Forest", "MLP"]}
linear_unfitted_models = {k:v for k, v in unfitted_models.items() 
                          if k not in ["Random Forest", "MLP"]}



# =============================================================================
# Stability analysis
# =============================================================================

# Feature stability
# -----------------
# Set iterables and parameters
sample_size = 0.3
n_iters = 1000
seed = 1
n_jobs = -1

# Produce stability plots for linear models
coefs = coef_dict(estimators=linear_unfitted_models,
                  X=X_train, Y=y_train,
                  n_iters=n_iters, n_jobs=n_jobs,
                  random_state=seed, train_size=sample_size)
coef_stats = coef_stats_dict(coefs)

for model, coef in coefs.items():
    fig = plot_stability(coef, title=model)
    
    name = model.lower().replace(" ", "") + "_stability_plot.png"
    figpath = os.path.join(path, name)
    fig.savefig(figpath)

# Plot mean coefficients across all linear models
fig = plot_mean_coef_heatmap(coefs)
fig.savefig(os.path.join(path, "mean_coef_heatmap.png"),
            bbox_inches="tight")



# # Hyperparameter stability
# # ------------------------
# # Get cross-validation results
# sort = "ascending"
# lasso_cv = cv_table(lasso.cv_results_, ordered=sort)
# ridge_cv = cv_table(ridge.cv_results_, ordered=sort)
# enet_cv = cv_table(enet.cv_results_, ordered=sort)
# pls_cv = cv_table(pls.cv_results_, ordered=sort)
# rf_cv = cv_table(rf.cv_results_, ordered=sort)
# mlp_cv = cv_table(mlp.cv_results_, ordered=sort)



# Linear coefficient plots
# ------------------------
# Plot linear model coefficients
for key, model in linear_fitted_models.items():
    fig = plot_coefs(model.coef_.flatten(),
                     X_train.columns.to_list(),
                     title=key,
                     cmap="rainbow")
    name = key.lower().replace(" ", "") + "_coefficient_plot.png"
    figpath = os.path.join(path, name)
    fig.savefig(figpath)
