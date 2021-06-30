import os

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.config.threading import (set_inter_op_parallelism_threads,
                                          set_intra_op_parallelism_threads)

from cts.models._linear_regression import linear_regression

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
set_intra_op_parallelism_threads(num_threads)


# Linear Regression
# -----------------
lr = linear_regression(X_train, y_train, n_jobs=n_jobs,
                       return_fit_time=show_time)

# Save model(s)
save_models(lr)