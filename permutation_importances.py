import os

import numpy as np

from sklearn.model_selection import train_test_split
    
from complex_trait_stats.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
from complex_trait_stats.utils import (load_dataframe,
                                       process_data,
                                       create_directory,
                                       load_models,
                                       perm_importances,
                                       tabulate_perm,
                                       plot_perm_importance)



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



# =============================================================================
# Permutation importance
# =============================================================================

# Model reliance 
# --------------
# Set iterables and parameters
n_samples = 3
sample_size = 0.3
n_repeats = 5
seed = 1
scoring = "r2"
correction = "fdr_bh"

# Permutation importances for each feature
perms = perm_importances(models, X_test, y_test, scoring=scoring,
                         n_samples=n_samples, n_repeats=n_repeats,
                         random_state=seed)

# Plot permutation importances
perm_tab = tabulate_perm(perms, feature_names=X.columns, method=correction)
fig = plot_perm_importance(perm_tab, edgecolor="white", alpha=0.75)
figpath = os.path.join(path, "permutation_importances.png")
fig.savefig(figpath)