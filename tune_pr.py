import os

import numpy as np

from sklearn.model_selection import train_test_split

from cts.models._penalised_regression import (lasso_regression,
                                              ridge_regression,
                                              enet_regression)
    
from cts.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS, CV_FOLDS
from cts.utils import (load_dataframe,
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
n_jobs = -1


# Penalised Regression (LASSO, Ridge, Elastic-Net)
# ------------------------------------------------
en_params = dict(alpha=np.logspace(-8, 8, 17),
                 l1_ratio=np.linspace(0, 1, 41))
pr_params = {k:v for k, v in en_params.items() if k == "alpha"}

lasso = lasso_regression(X_train, y_train, param_grid=pr_params, 
                         folds=CV_FOLDS, n_jobs=n_jobs, random_state=seed,
                         return_fit_time=show_time, warm_start=True)
ridge = ridge_regression(X_train, y_train, param_grid=pr_params,
                         folds=CV_FOLDS, n_jobs=n_jobs, random_state=seed,
                         return_fit_time=show_time)
enet = enet_regression(X_train, y_train, param_grid=en_params,
                       folds=CV_FOLDS, n_jobs=n_jobs, random_state=seed,
                       return_fit_time=show_time, warm_start=True)

# Save model(s)
print(24*"#")
print(lasso.best_estimator_)
print(24*"#")
print(ridge.best_estimator_)
print(24*"#")
print(enet.best_estimator_)
models = [lasso.best_estimator_, ridge.best_estimator_, enet.best_estimator_]
save_models(models)
