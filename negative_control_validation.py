import os

import numpy as np

from sklearn.model_selection import train_test_split
    
from cts.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
from cts.utils import (load_dataframe,
                       process_data,
                       create_directory,
                       load_models,
                       model_validation,
                       tabulate_validation,
                       plot_neg_validation)



# Create directory for figures
path = os.path.join(ROOT, "figures", "validation")
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
# Negative control validation
# =============================================================================

# Negative control strategy (permuted labels)
# -------------------------------------------
# Set iterables and parameters
n_samples = 3
sample_size = 0.3
n_repeats = 5
seed = 1
scoring = "r2"
correction = "fdr_bh"

# Negative control validation over bootstrapped samples
neg_ctrl = {version:model_validation(estimators=models,
                                     X=X_test, y=y_test,
                                     scoring=scoring, n_samples=n_samples,
                                     sample_size=sample_size,
                                     n_repeats=n_repeats,
                                     positive_ctrl=False,
                                     random_state=seed,
                                     version=version)
            for version in ["tpr", "fpr"]}

# Plot negative control results
neg_results = tabulate_validation(neg_ctrl, positive_ctrl=False,
                                  method=correction)
fig = plot_neg_validation(neg_results)
figpath = os.path.join(path, "negative_control_validation.png")
fig.savefig(figpath)
