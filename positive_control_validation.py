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

# Load unfitted models
models = load_models(fitted=False)

# Set up array job
m = int(os.environ["PBS_ARRAY_INDEX"])
name = list(models.keys())[m-1]
estimator = models[name]    # Adjust for zero-indexing



# =============================================================================
# Positive control validation
# =============================================================================

# Positive control (perfectly correlated control feature)
# -------------------------------------------------------
# Set iterables and parameters
n_samples = 1000
sample_size = 0.3
n_repeats = 10000
seed = 1
scoring = "r2"
correction = "fdr_bh"
noise_params = [0., 10., 25., 75., 150.]
n_jobs = int(sys.argv[1])
verbose = 10

# Positive control validation vs. noise over bootstrapped samples
pos_ctrl = {float(noise):model_validation(estimator=estimator,
                                          X=X_test, y=y_test,
                                          scoring=scoring,
                                          n_samples=n_samples,
                                          n_repeats=n_repeats,
                                          sample_size=sample_size,
                                          positive_ctrl=True,
                                          random_state=seed,
                                          control_params={"sigma":noise},
                                          n_jobs=n_jobs,
                                          verbose=verbose)
            for noise in noise_params}

# Save positive control results
pos_results = pd.DataFrame(pos_ctrl, index=[name])
pos_results.to_csv(os.path.join(path,
                                "tmp_pos_"+name.replace(" ", "_")+".csv"))
