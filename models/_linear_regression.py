import time

from sklearn.linear_model import LinearRegression



def linear_regression(X, y, n_jobs=-2, return_fit_time=False):
    """Fit Linear Regression model
    """

    # Time model
    t0 = time.time()
    
    # Fit estimator
    clf = LinearRegression(n_jobs=n_jobs)
    clf.fit(X, y)
    
    # Print fit time
    t1 = time.time()
    if return_fit_time:
        print("Running time : {:.2f} seconds".format(t1 - t0))
        
    return clf