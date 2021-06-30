import os

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.config.threading import (set_inter_op_parallelism_threads,
                                          set_intra_op_parallelism_threads)

from cts.models._neural_network import multilayer_perceptron
    
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


# Multilayer Perceptron
# ---------------------
first_neurons=[1, 5, 10, 15, 25, 50]
hidden_neurons=[1, 5, 10, 15, 25, 50]
activation=["relu"]
last_activation=["relu", None]
dropout=[0.01, 0.1, 0.2]
l1=[1e-02, 1e-04, 1e-06, 1e-08]
l2=[1e-02, 1e-04, 1e-06, 1e-08]
epochs=[10, 500, 1000, 5000, 10000]
batch_size=[20, 100, 500, 1000, 5000]

one_layer_params = dict(hidden_layers=[1],
                        first_neurons=first_neurons,
                        hidden_neurons=[None],
                        activation=activation,
                        last_activation=last_activation,
                        dropout=dropout,
                        l1=l1,
                        l2=l2,
                        epochs=epochs,
                        batch_size=batch_size)    # Single hidden layer
multi_layer_params = dict(hidden_layers=[2, 3],
                          first_neurons=first_neurons,
                          hidden_neurons=hidden_neurons,
                          activation=activation,
                          last_activation=last_activation,
                          dropout=dropout,
                          l1=l1,
                          l2=l2,
                          epochs=epochs,
                          batch_size=batch_size)    # Multiple hidden layers

mlp_params = [one_layer_params, multi_layer_params]
mlp = multilayer_perceptron(X_train, y_train, param_grid=mlp_params,
                            n_iter=50400, n_jobs=n_jobs, random_state=seed,
                            return_fit_time=show_time)
# Search ~50% of the hyperparameter space

# Save model(s)
save_models(mlp.best_estimator_)