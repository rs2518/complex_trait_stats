import time

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV



def multilayer_perceptron(X, y, param_grid, folds=5, n_jobs=-2, n_iter=10,
                          random_state=None, return_fit_time=False, **kwargs):
    """Fit sklearn Multilayer Perceptron using RandomizedSearchCV
    """
    # Set default keyword arguments
    if kwargs == {}:
        kwargs={"solver":"sgd",
                "early_stopping":True}
    
    # Time model
    t0 = time.time()

    # Pass Keras model into scikit-learn wrapper then fit estimator
    clf = MLPRegressor(**kwargs)
    clf_cv = RandomizedSearchCV(estimator=clf,
                                n_iter=n_iter,
                                param_distributions=param_grid,
                                cv=folds, n_jobs=n_jobs,
                                random_state=random_state,
                                return_train_score=True)
    clf_cv.fit(X, y)
    
    # Print fit time
    t1 = time.time()
    if return_fit_time:
        print("MLP running time : {:.2f} seconds".format(t1 - t0))
        
    return clf_cv
