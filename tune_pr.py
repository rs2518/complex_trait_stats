import os

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

lasso_cv = lasso_regression(X_train, y_train, param_grid=pr_params, 
                            folds=CV_FOLDS, n_jobs=n_jobs, random_state=seed,
                            return_fit_time=show_time)
print(12*"-", "\n")
lasso = lasso_cv.best_estimator_
print(lasso)
print(12*"-", "\n")
print("Lasso test score (R2) :", r2_score(y_test, lasso.predict(X_test)))
print(36*"=", "\n")

ridge_cv = ridge_regression(X_train, y_train, param_grid=pr_params,
                            folds=CV_FOLDS, n_jobs=n_jobs, random_state=seed,
                            return_fit_time=show_time)
print(12*"-", "\n")
ridge = ridge_cv.best_estimator_
print(ridge)
print(12*"-", "\n")
print("Ridge test score (R2) :", r2_score(y_test, ridge.predict(X_test)))
print(36*"=", "\n")

enet_cv = enet_regression(X_train, y_train, param_grid=en_params,
                          folds=CV_FOLDS, n_jobs=n_jobs, random_state=seed,
                          return_fit_time=show_time)
print(12*"-", "\n")
enet = enet_cv.best_estimator_
print(enet)
print(12*"-", "\n")
print("ElasticNet test score (R2) :", r2_score(y_test, enet.predict(X_test)))
print(36*"=", "\n")

# Save model(s)
save_models([lasso, ridge, enet])
