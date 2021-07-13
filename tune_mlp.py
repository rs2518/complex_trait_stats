import os

import numpy as np

from itertools import chain, product
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from cts.models._neural_network import multilayer_perceptron
    
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


# Multilayer Perceptron
# ---------------------
n_neurons = np.array([1, 5, 10, 25, 50])
max_hidden_layers = 3
hidden_layer_sizes = list(chain(*[list(product(n_neurons, repeat=r+1))
                                  for r in range(max_hidden_layers)]))

mlp_params = dict(hidden_layer_sizes=hidden_layer_sizes,
                  activation=["relu", "identity"],
                  solver=["sgd"],
                  batch_size=[200, 1000, 5000],
                  learning_rate=["constant", "invscaling", "adaptive"],
                  early_stopping=[False, True])
mlp_cv = multilayer_perceptron(X_train, y_train, param_grid=mlp_params,
                               n_iter=3780, folds=CV_FOLDS, n_jobs=n_jobs,
			       random_state=seed, return_fit_time=show_time)
print(12*"-", "\n")
mlp = mlp_cv.best_estimator_
print(mlp)
print(12*"-", "\n")
print("MLP test score (R2) :", r2_score(y_test, mlp.predict(X_test)))
print(36*"=", "\n")

# Save model(s)
save_models(mlp)
