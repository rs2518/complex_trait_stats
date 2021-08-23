import os

import numpy as np

from sklearn.model_selection import train_test_split

from cts.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
from cts.utils import (load_dataframe,
                       process_data,
                       create_directory,
                       load_models,
                       perm_importances,
                       tabulate_perm)



# Create directory for figures
path = os.path.join(ROOT, "figures", "permutation_importance")
create_directory(path)

# Load data and split data into training and testing sets
df = load_dataframe(RAW_DATA)
data = process_data(df)
X = data.drop(['p_value'], axis=1)
y = -np.log10(data["p_value"])
X_train, X_test, y_train, y_test = train_test_split(X, y, **TRAIN_TEST_PARAMS)

# Load fitted models
models = load_models()

# Set up array job
m = int(os.environ["PBS_ARRAY_INDEX"])
name = list(models.keys())[m-1]
estimator = models[name]    # Adjust for zero-indexing



# =============================================================================
# Permutation importance
# =============================================================================

# Model reliance 
# --------------
# Set iterables and parameters
n_samples = 1000
sample_size = 0.3
n_repeats = 10000
seed = 1
scoring = "r2"
correction = "fdr_bh"

# Permutation importances for each feature
perms = perm_importances(estimator, X_test, y_test, scoring=scoring,
                         n_samples=n_samples, n_repeats=n_repeats,
                         n_jobs=-1, random_state=seed)

# Plot permutation importances
perm_tab = tabulate_perm(perms, index=X.columns, columns=[name],
                         method=correction)
perm_tab.to_csv(os.path.join(path,
                             "tmp_perm_"+name.replace(" ", "_")+".csv"))
