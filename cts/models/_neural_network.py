import time
import h5py
import io

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor



class PickleableKerasRegressor(KerasRegressor, BaseEstimator):
    """Wrapper class to enable pickling for KerasRegressor
    
    See https://github.com/keras-team/keras/issues/13168
    """

    def _pickle_model(self):
        bio = io.BytesIO()
        with h5py.File(bio, 'w') as f:
            self.model.save(f)
        return bio

    def _unpickle_model(self, model):
        with h5py.File(model, 'r') as f:
            model = load_model(f)
        return model 

    def __getstate__(self):
        state = BaseEstimator.__getstate__(self)
        if hasattr(self, 'model'):
            state['model'] = self._pickle_model()
        return state

    def __setstate__(self, state):
        if state.get('model', None):
            state['model'] = self._unpickle_model(state['model'])
        return BaseEstimator.__setstate__(self, state)


def _create_mlp(input_dim, hidden_layers=1, first_neurons=5, hidden_neurons=5,
               activation="relu", last_activation=None, dropout=0,
               l1=None, l2=None, learning_rate=1e-02):
    """Multilayer Perceptron neural network generating function
    
    Defines MLP neural network architecture according to given arguments.
    This enables the generated model to pass through the scikit-learn
    wrapper for hyperparameter searches.
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
    
    # Build output layer with single neuron (unit). Compile resulting model
    model.add(Dense(1, activation=last_activation))
    model.compile(loss="mean_squared_error", optimizer=optimizer,
                  metrics=["mse"])
    
    return model


def multilayer_perceptron(X, y, param_grid, folds=5, n_jobs=-2, n_iter=10,
                          random_state=None, return_fit_time=False):
    """Fit Keras Multilayer Perceptron using RandomizedSearchCV
    """     
    # Time model
    t0 = time.time()
    
    # Set input_dim to be the number of features
    # param_grid["input_dim"] = [X.shape[1]]

    # Pass Keras model into scikit-learn wrapper then fit estimator
    clf = PickleableKerasRegressor(build_fn=_create_mlp, input_dim=X.shape[1],
                                   epochs=5, batch_size=20, verbose=0)
    clf_cv = RandomizedSearchCV(estimator=clf,
                                n_iter=n_iter,
                                param_distributions=param_grid,
                                cv=folds, n_jobs=n_jobs,
                                random_state=random_state,
                                return_train_score=True)
    clf_cv.fit(X.values, y.values)
    
    # Print fit time
    t1 = time.time()
    if return_fit_time:
        print("MLP running time : {:.2f} seconds".format(t1 - t0))
        
    return clf_cv