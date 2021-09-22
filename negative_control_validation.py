import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
    
from cts.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
from cts.utils import (load_dataframe,
                       process_data,
                       create_directory,
                       load_models,
                       model_validation)



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

# Set up array job
m = int(os.environ["PBS_ARRAY_INDEX"])
name = list(models.keys())[m-1]
estimator = models[name]    # Adjust for zero-indexing



# =============================================================================
# Negative control validation
# =============================================================================

# Negative control strategy (permuted labels)
# -------------------------------------------
# Set iterables and parameters
n_samples = 1000
sample_size = 0.3
n_repeats = 10000
seed = 1
scoring = "r2"
correction = "fdr_bh"
n_jobs = int(sys.argv[1])
verbose = 10

# Negative control validation over bootstrapped samples
neg_ctrl = {version:model_validation(estimator=estimator,
                                     X=X_test, y=y_test,
                                     scoring=scoring, n_samples=n_samples,
                                     sample_size=sample_size,
                                     n_repeats=n_repeats,
                                     positive_ctrl=False,
                                     random_state=seed,
                                     version=version,
                                     n_jobs=n_jobs,
                                     verbose=verbose)
            for version in ["tpr", "fpr"]}

# Save negative control results
neg_results = pd.DataFrame(neg_ctrl, index=[name])
neg_results.to_csv(os.path.join(path,
                                "tmp_neg_"+name.replace(" ", "_")+".csv"))
