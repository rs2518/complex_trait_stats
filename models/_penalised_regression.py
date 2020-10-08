import time

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet



def _penalised_regression(estimator, X, y, param_grid, folds,
                          random_state, n_jobs, return_fit_time,
                          **kwargs):
    """Fit general penalised regression model using GridSearchCV
    """
    # Set default maximum iterations in keyword arguments for all models
    if kwargs == {}:
        kwargs={"max_iter":10000}
            
    # Time model
    t0 = time.time()
    
    # Fit estimator
    clf = estimator(random_state=random_state, **kwargs)
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



def lasso_regression(X, y, param_grid, folds=5, random_state=None,
                     n_jobs=-2, return_fit_time=False, **kwargs):
    """Fit LASSO Regression using cross-validation via GridSearchCV
    """
    clf = _penalised_regression(Lasso, X=X, y=y, param_grid=param_grid,
                                folds=folds, random_state=random_state,
                                n_jobs=n_jobs, return_fit_time=return_fit_time,
                                **kwargs)
    return clf


def ridge_regression(X, y, param_grid, folds=5, random_state=None,
                     n_jobs=-2, return_fit_time=False, **kwargs):
    """Fit Ridge Regression using cross-validation via GridSearchCV
    """
    clf = _penalised_regression(Ridge, X=X, y=y, param_grid=param_grid,
                                folds=folds, random_state=random_state,
                                n_jobs=n_jobs, return_fit_time=return_fit_time,
                                **kwargs)
    return clf


def enet_regression(X, y, param_grid, folds=5, random_state=None,
                    n_jobs=-2, return_fit_time=False, **kwargs):
    """Fit Elastic Net Regression using cross-validation via GridSearchCV
    """
    clf = _penalised_regression(ElasticNet, X=X, y=y, param_grid=param_grid,
                                folds=folds, random_state=random_state,
                                n_jobs=n_jobs, return_fit_time=return_fit_time,
                                **kwargs)
    return clf