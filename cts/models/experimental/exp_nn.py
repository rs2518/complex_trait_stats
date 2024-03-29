# =============================================================================
# TEMPORARY IMPORT FOR USE IN AN INTERPRETER
# =============================================================================
import os, sys

directory = "Desktop/Term 3 MSc Project"

path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================

import os

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import (load_dataframe, process_category,
                                       metrics)

import time

# Load data and log transform p-values
df = load_dataframe(RAW_DATA)
data = process_category(df)

X = data.drop(['p_value'], axis=1)
y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
y.columns = ["p_value", "-log10_p"]
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, test_size=0.3,
                     random_state=1010)

# Time model(s)
t0 = time.time()

# Model generating function
n_features = X.shape[1]
def create_model(hidden_layers=1, first_neurons=5, hidden_neurons=5,
                 activation="relu", last_activation=None,
                 dropout=0, l1=None, l2=None, learning_rate=1e-03):
    """Neural network model generating function
    
    Defines neural network architecture according to given arguments. This
    enables the generated model to pass through the scikit-learn wrapper for
    hyperparameter searches.
    """
    # Create object for kernel_regularization using l1 and l2 values
    kernel_regularizer = l1_l2(l1=l1, l2=l2)
    optimizer = Adam(learning_rate=learning_rate)
    
    # Instantiate model
    model = Sequential()
    
    # Build input layer and subsequent hidden layers. Add dropout onto layers
    first = True
    for n in range(hidden_layers):
        if first:
            model.add(Dense(first_neurons,
                            input_dim=n_features,
                            activation=activation,
                            kernel_regularizer=kernel_regularizer))
            first = False
        else:
            model.add(Dense(hidden_neurons,
                            activation=activation,
                            kernel_regularizer=kernel_regularizer))
        if dropout!=0:
            model.add(Dropout(dropout))
    
    # Build output layer with single neuron (unit). Compile resulting model
    model.add(Dense(1, activation=last_activation))
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    
    return model
    
mlp = KerasRegressor(build_fn=create_model, epochs=50, batch_size=40,
                     verbose=0)
 

# Set up parameter selection for RandomizedSearchCV param_grids
fn_params = dict(epochs=[10, 50],
                 batch_size=[20, 80],
                 hidden_layers=[2, 4, 8],
                 first_neurons=[25, 50, 100],
                 hidden_neurons=[5, 10, 25, 50],
                 activation=["relu", "softplus"],
                 last_activation=[None],
                 dropout=[0, 0.1, 0.2],
                 l1=[None, 1e-02, 1e-04],
                 l2=[None, 1e-02, 1e-04],
                 learning_rate=[1e-01])

# Tune hyperparameters using cross-validation. Run model for both the raw
# p-values and log-transformed p-values
folds = 5
models = {}
index = []
scores = pd.DataFrame()

for i in range(y.shape[1]):
    mlp_cv = RandomizedSearchCV(estimator=mlp,
                                param_distributions=fn_params,
                                n_iter=10,
                                cv=folds,
                                random_state=1010,
                                n_jobs=-2,
                                return_train_score=True)
    
    model_id = "MLP "+y.columns[i]
    models[model_id] = mlp_cv.fit(X_train, y_train[:,i])
    index.append(model_id)
    
    y_pred = mlp_cv.best_estimator_.predict(X_test)
    scores = scores.append(metrics(y_test[:,i], y_pred), ignore_index=True)
    
    # # View properties of network with best hyperparameters
    # p = {k:v for k, v in mlp_cv.best_params_.items()
    #      if k not in ["epochs", "batch_size"]}
    # nn = create_model(**p)
    # nn.summary()
    
scores.index = index


t1 = time.time()
print("Running time : {:.2f} seconds".format(t1 - t0))
# ~900 seconds



# =============================================================================
# Hyperparameter testing
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from complex_trait_stats.models._neural_network import multilayer_perceptron
from complex_trait_stats.utils import (process_data, RAW_DATA, load_dataframe,
                                       plot_true_vs_pred, cv_table)



# Load data and add column of ones for intercept
# df = load_dataframe(RAW_DATA)
df = load_dataframe("snp_raw_allchr1000.csv")
data = process_data(df)

X = data.drop(['p_value'], axis=1)
Y = pd.concat([data["p_value"], -np.log10(data["p_value"])], axis=1)
Y.columns = ["p_value", "-log10_p"]
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=1010)


# y_train = Y_train["p_value"]
# y_test = Y_test["p_value"]
y_train = Y_train["-log10_p"]
y_test = Y_test["-log10_p"]



seed = 1
show_time = True

mlp_params = dict(hidden_layers=[1, 2],
                  first_neurons=[10, 25, 100],
                  hidden_neurons=[5, 10, 50],
                  activation=["relu"],
                  last_activation=[None],
                  dropout=[0.01, 0.1, 0.3],
                  l1=[0.1, 0.0001],
                  l2=[0.1, 0.0001],
                  epochs=[50],
                  batch_size=[50])

# mlp_params = dict(hidden_layers=[1],
#                   first_neurons=[10],
#                   hidden_neurons=[None],
#                   activation=["relu"],
#                   last_activation=[None],
#                   dropout=[0.1, 0.2, 0.3],
#                   l1=[1e-04],
#                   l2=[1e-04],
#                   epochs=[50],
#                   batch_size=[20])

mlp = multilayer_perceptron(X_train, y_train, param_grid=mlp_params, n_iter=25,
                            random_state=seed, return_fit_time=show_time) 

print(mlp.cv_results_)
print(mlp.best_estimator_.predict(X_test.values))
print(mlp.best_estimator_.predict(X_test.values) - y_test.values)
fig = plot_true_vs_pred(y_test, mlp.best_estimator_.predict(X_test.values))
cv_tab = cv_table(mlp.cv_results_, ordered="ascending")



mlp_params = dict(hidden_layers=[2],
                  first_neurons=[15],
                  hidden_neurons=[7],
                  activation=["relu"],
                  last_activation=[None],
                  dropout=[0.01],
                  l1=[0.00001],
                  l2=[0.00001],
                  epochs=[20],
                  batch_size=[100])

mlp = multilayer_perceptron(X_train, y_train, param_grid=mlp_params, n_iter=1,
                            random_state=1, return_fit_time=show_time) 

print(mlp.cv_results_)
# print(mlp.best_estimator_.predict(X_test.values))
# print(mlp.best_estimator_.predict(X_test.values) - y_test.values)
fig = plot_true_vs_pred(y_test, mlp.best_estimator_.predict(X_test.values))
cv_tab = cv_table(mlp.cv_results_, ordered="ascending")