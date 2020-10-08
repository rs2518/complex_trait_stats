import time

from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression



def pls_regression(X, y, param_grid, folds=5, n_jobs=-2,
                   return_fit_time=False, **kwargs):
    """Fit Partial Least Squares regression model using GridSearchCV
    """
    # Set default PLS keyword arguments
    if kwargs == {}:
        kwargs={"max_iter":1000}
            
    # Time model
    t0 = time.time()
    
    # Fit estimator
    clf = PLSRegression(**kwargs)
    clf_cv = GridSearchCV(estimator=clf, param_grid=param_grid,
                          cv=folds, n_jobs=n_jobs,
                          return_train_score=True)
    clf_cv.fit(X, y)
    
    # Print fit time
    t1 = time.time()
    if return_fit_time:
        print("{} running time : {:.2f} seconds".format(
            type(clf).__name__, (t1 - t0)))
        
    return clf_cv