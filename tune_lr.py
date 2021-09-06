import os
import sys

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
n_jobs = int(sys.argv[1])


# Linear Regression
# -----------------
lr = linear_regression(X_train, y_train, n_jobs=n_jobs,
                       return_fit_time=show_time)
print(12*"-", "\n")
print(lr)
print(12*"-", "\n")
print("Linear Regression test score (R2) :", r2_score(y_test,
                                                      lr.predict(X_test)))
print(36*"=", "\n")

# Save model(s)
save_models(lr)
