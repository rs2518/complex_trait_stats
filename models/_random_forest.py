import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor



def random_forest(X, y, param_grid, folds=5, n_jobs=-2, n_iter=10,
                  random_state=None, return_fit_time=False, **kwargs):
    """Fit Random Forest Regressor using RandomizedSearchCV
    """     
    # Time model
    t0 = time.time()
    
    # Fit estimator
    clf = RandomForestRegressor(**kwargs)
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
        print("Running time : {:.2f} seconds".format(t1 - t0))
        
    return clf_cv