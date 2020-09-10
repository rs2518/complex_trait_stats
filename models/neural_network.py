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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# from tensorflow.keras import Input, Model

from complex_trait_stats.utils import RAW_DATA
from complex_trait_stats.utils import (load_dataframe, process_category,
                                       metrics)

import time

# Load data and add column of ones for intercept
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

def create_model(input_dim, hidden_layers=1,
                 first_neurons=5, hidden_neurons=5,
                 activation="relu", last_activation="linear",
                 dropout=0, l1=None, l2=None,
                 loss="mean_squared_error", optimizer="sgd"):
    """Neural network model generating function
    
    Defines neural network architecture according to given arguments. This
    enables the generated model to pass through the scikit-learn wrapper for
    hyperparameter searches.
    """
    # Create object for kernel_regularization using l1 and l2 values
    kernel_regularizer = l1_l2(l1=l1, l2=l2)
    
    # Instantiate model
    model = Sequential()
    
    # Build input layer and subsequent hidden layers. Add dropout layer if
    # rate > 0
    first = True
    for n in range(hidden_layers):
        if first:
            model.add(Dense(first_neurons,
                            input_dim=input_dim,
                            activation=activation,
                            kernel_regularizer=kernel_regularizer))
            first = False
        else:
            model.add(Dense(hidden_neurons,
                            activation=activation,
                            kernel_regularizer=kernel_regularizer))
        if dropout!=0:
            model.add(Dropout(dropout))
    
    # Build output layer with single neuron (unit) with last_activation
    model.add(Dense(1, activation=last_activation))
    
    # Compile model with given loss functions and optimization algorithm
    model.compile(loss=loss, optimizer=optimizer)
    
    return model
    
# Scikit-learn wrapper for keras regression model
# model = create_model(input_dim=X_train.shape[1])
mlp = KerasRegressor(build_fn=create_model, input_dim=X.shape[1],
                     epochs=2, batch_size=20, verbose=0)
 

# Set up parameter selection for RandomizedSearchCV param_grids
# epochs = [5, 50]
# batch_size = [20, 40, 80]
# l1=[None, 1e-3]
# l2=[None, 1e-3]
param_grid = dict(hidden_layers=[1, 2, 3],
                  first_neurons=[5],
                  hidden_neurons=[5, 20],
                  activation=["relu", "linear"],
                  last_activation=["linear"],
                  dropout=[0, 0.2],
                  optimizer=["sgd", "adam"])

# Tune hyperparameters using cross-validation. Run model for both the raw
# p-values and log-transformed p-values
folds = 3
models = {}
index = []
# scores = pd.DataFrame()

# for i in range(y.shape[1]):
i=0
mlp_cv = RandomizedSearchCV(estimator=mlp,
                            param_distributions=param_grid,
                            n_iter=2,
                            cv=folds,
                            random_state=1010,
                            return_train_score=True)
mlp_cv.fit(X_train, y_train[:,i])

model_id = "MLP "+y.columns[i]
models[model_id] = mlp_cv
index.append(model_id)

# y_pred = mlp_cv.best_estimator_.predict(X_test)
# scores = scores.append(metrics(y_test[:,i], y_pred), ignore_index=True)        
        
# scores.index = index




# # list some properties of the network
# model.summary()
# model.evaluate(X_test, y_test)



# t1 = time.time()
# print("Running time : {:.2f} seconds".format(t1 - t0))
# # ? seconds