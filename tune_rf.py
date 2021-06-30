import os

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.config.threading import (set_inter_op_parallelism_threads,
                                         set_intra_op_parallelism_threads)

from cts.models._random_forest import random_forest
    
from cts.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
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

# Set number of threads
num_threads = 80    # Set to match ncpus
set_inter_op_parallelism_threads(num_threads)
set_inter_op_parallelism_threads(num_threads)


# Random Forest
# -------------
rf_params = dict(n_estimators=[10, 25, 50, 100, 250, 500, 1000],
                 max_features=["auto", "sqrt", "log2"],
                 max_depth=[5, 10, 25, 50, 100, 250],
                 min_samples_split=[0.001, 0.01, 0.05, 0.1, 0.15, 0.2],
                 min_samples_leaf=[0.001, 0.01, 0.05, 0.1, 0.15, 0.2])

rf = random_forest(X_train, y_train, param_grid=rf_params, n_iter=2722,
                   n_jobs=n_jobs, random_state=seed,
                   return_fit_time=show_time, warm_start=True)
# Search ~60% of the hyperparameter space

# Save model(s)
save_models(rf.best_estimator_)
