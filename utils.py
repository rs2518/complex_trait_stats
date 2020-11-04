import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score as explained_var
# from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score as R2
# from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

# from sklearn.inspection._permutation_importance import \
#     _calculate_permutation_scores
from sklearn.inspection import permutation_importance
from sklearn.metrics import check_scoring
from sklearn.utils import Bunch, check_random_state



ROOT = os.path.join(os.path.expanduser("~"),
                    "Desktop/Term 3 MSc Project/complex_trait_stats")
RAW_DATA = "snp_raw.csv"
PROCESSED_DATA = "snp_processed.csv"



# Data preprocessing
# ------------------

def load_dataframe(file):
    """Load data from ROOT folder
    """
    filepath = os.path.join(ROOT, "data", file)
    data = pd.read_csv(filepath, index_col=1)
    
    data.drop(data.columns[0], axis=1, inplace=True)
    data = _remove_zero_pval(data)
    
    return _set_categories(data)


def _set_categories(dataframe):
    """Convert columns with 'object' dtype to 'category'
    """
    cols = dataframe.select_dtypes(include=["object"]).columns
    for col in cols:
        dataframe[col] = dataframe[col].astype("category")
        
    return dataframe


def _remove_zero_pval(dataframe):
    """Add constant to zero p_values
    
    Prevents error during log transformations (where p_value = 0) by adding
    half the minimum p_value
    """
    min_p = min(dataframe["p_value"][dataframe["p_value"] != 0])
    dataframe["p_value"].replace(0, min_p/2, inplace=True)
    
    return dataframe


def binarise_category(data):
    """Convert categorical into binary predictors
    """
    # Binarise all categorical variables
    processed_data = pd.get_dummies(data)
    
    # Move label to last column
    cols = processed_data.columns.to_list()
    cols.append(cols.pop(cols.index("p_value")))
    
    return processed_data[cols]


def scale_numeric(data):
    """Apply StandardScaler to numeric predictors
    """
    scaled_data = data.copy()
    
    # Scale with Standard Scaler
    sc = StandardScaler()
    cols = scaled_data.drop(["p_value"], axis=1).select_dtypes(
        exclude=["category", "object"]).columns
    scaled_data[cols] = sc.fit_transform(data[cols])
    
    return scaled_data


def process_data(data):
    """Scale numeric predictors and binarise categorical predictors
    """    
    return binarise_category(scale_numeric(data))



# Model diagnostics
# -----------------

def plot_coefs(coefs, names, title=None, conf_int=None, cmap="default"):
    """Coefficient plot
    
    Plot coefficients against their respective names. If 'conf_int' is given,
    error bars are added to the plot to show 95% confidence interval.
    
    Setting cmap to "signif" highlights significant coefficients in red whilst
    insignificant coefficients are in light grey.
    Setting cmap to "rainbow" returns each coefficient in different colour.
    Otherwise, cmap is set to default or if cmap="default" but no confidence
    interval is given.
    
    """
    # Set title
    if title is None:
        title = "Coefficient plot"
    title_ext = ""
    
    # Set errorbars (if given)
    if conf_int is None:
        yerr = None
    else:
        yerr = np.abs(conf_int.T)
        title_ext = " w/ 95% conf. interval"
        # title = " w/ {:.{a}%} conf. interval".format(
        #     (1-alpha), a=len(str(1-alpha))-4)
    
    
    if cmap == "default":
        cmap = None
    elif cmap == "rainbow":
        cmap = sns.hls_palette(len(names), l=.55)
    elif cmap == "signif":
        if conf_int is None:
            cmap = None
        else:
            sign = np.sign(conf_int)
            inds = sign[:,0] * sign[:,1] > 0
            cmap = ["red" if inds[i] else "silver" for i in range(len(names))]

    
    fig = plt.figure()
    
    plt.errorbar(names, coefs, yerr=yerr, ecolor=cmap, ls="none")
    plt.scatter(names, coefs, c=cmap)
    plt.xticks(rotation=90)
    plt.ylabel("Coefficient "+r'$x_i$')
    plt.title(title + title_ext)
    plt.hlines(0, len(names), 0, colors="grey", linestyles="--")
    
    plt.tight_layout()
    # plt.show()
    
    return fig



def metrics(y_true, y_pred, dp=4):
    """Metrics table
    
    Returns a table of the following regression metrics to the given number 
    of decimal places:
        - Mean Squared Error
        - Mean Absolute Error
        - Explained Variance
        - Median Absolute Error
        - R-sqaured
    """    
    cols = ['Mean Squared Error',
             'Mean Absolute Error',
             'Explained Variance',
             'Median Absolute Error',
             'R-sqaured']    
    fns = [MSE, MAE, explained_var, median_absolute_error, R2]
    
    vals = []
    text = "{:.{dp}f}"
    for fn in fns:
        vals.append(float(text.format(fn(y_true, y_pred), dp=dp)))
    data = np.array(vals).reshape(1,-1)
        
    return pd.DataFrame(data, columns=cols)



def cv_table(cv_results_, ordered=None):
    """Table of cross-validation results
    
    Returns table "mean_test_scores" and "std_test_scores" along with
    respective model parameters from cv_results_. Can sort results according
    to rank_test_score by setting sort="ascending" or sort="descending".
    """    
    dict_keys = ["rank_test_score", "mean_test_score", "std_test_score",
                 "mean_train_score", "std_train_score"]
    dict_keys += [key for key in cv_results_.keys() if "param_" in key]
    d = {key:cv_results_[key] for key in dict_keys}
    
    if ordered is None:
        return pd.DataFrame(data=d)
    elif ordered == "ascending":
        sort = True
    elif ordered == "descending":
        sort = False
        
    return pd.DataFrame(data=d).sort_values("rank_test_score", ascending=sort)



def get_outlier_inds(a, threshold=8):
    """Return indices of a where the abs(z-score) >= threshold
    """
    z = zscore(a)
    return np.arange(len(z))[np.abs(z) >= threshold]



def plot_true_vs_pred(y_true, y_pred, title=None, rm_outliers=True,
                      marker=".", markercolor=None, edgecolor="w",
                      ls="--", linecolor="red", **kwargs):
    """Plot true y values against predicted y values
    """
    if title is None:
        title = "True vs. Predicted"
        
    # Remove outliers
    yt = np.array(y_true)
    yp = np.array(y_pred)
    out_text = ""
    if rm_outliers:
        ind = get_outlier_inds(yt)
        yt = np.delete(yt, ind)
        yp = np.delete(yp, ind)
        
        # Metrics excluding outliers
        out_corr = np.corrcoef(yt, yp)[0,1]
        out_sse = MSE(yt, yp) * len(yt)
        out_text = "\n" + "\n".join((
            r"$\rho\ (excl. outliers) = %.4f$" % (out_corr),
            r"SSE (excl. outliers) = %.2f" % (out_sse)))
        
        if len(ind) > 0:
            title = title + " (%d outliers excluded)" % (len(ind))
        
        
    # Text box
    # corr, p_value = pearsonr(y_true, y_pred)
    # r2 = R2(y_true, y_pred)
    # if np.isnan(corr):
    #     corr = "-"
    corr = np.corrcoef(y_true, y_pred)[0,1]
    sse = MSE(y_true, y_pred) * len(y_true)
    text = "\n".join((
        r"$\rho = %.4f$" % (corr),
        r"SSE = %.2f" % (sse)))
    text = text + out_text
    bbox = dict(facecolor='wheat', alpha=0.5)
    
    
    # Set width of marker edge and line endpoints for better visuals
    # lw = 1/(len(yt)**0.25)
    lim = [min(yt), max(yt)]
    
    fig, ax = plt.subplots()
    
    ax = sns.jointplot(yt, yp, kind="reg")
    ax.ax_joint.plot(lim, lim, c=linecolor)
    ax.ax_joint.set_xlabel("True")
    ax.ax_joint.set_ylabel("Predicted")
    ax.ax_joint.text(0.05, 0.95, text, transform=ax.ax_joint.transAxes,
                     verticalalignment="top", fontsize=12, bbox=bbox)
    ax.ax_joint.set_title(title)
    
    
    # ax.scatter(yt, yp, marker=".", linewidth=lw, edgecolor="w",
    #            **kwargs)
    # ax.plot(lim, lim, c=linecolor)
    # ax.set_xlabel("True")
    # ax.set_ylabel("Predicted")
    # ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment="top",
    #         fontsize=12, bbox=bbox)
    # ax.set_title(title)

    plt.tight_layout()
    # plt.show()
    
    return fig



# Stability analysis
# ------------------

def coef_dict(estimators, X, Y, n_iters=5, bootstrap=False,
              random_state=None, scale_X=False,
              return_scaled_X=False, **split_options):
    """Returns dictionary with array of model coefficients at each iteration
    for a given list of estiamtors
    """
    # Check if estimators is an iterable and instantiate results dictionary
    if not hasattr(estimators, "__iter__"):
        estimators = [estimators]
    coefs = {type(estimator).__name__:np.zeros((X.shape[1], n_iters))
             for estimator in estimators}

    # Break links to original data
    X_s = X.copy()
    Y_s = Y.copy()
    
    # Standard scale numerical features
    if scale_X:
        num_cols = X_s.select_dtypes(
            exclude=["object", "category"]).columns
        scaler = StandardScaler()
        X_s[num_cols] = scaler.fit_transform(X_s[num_cols])

    # Loop over n_iters
    for i in range(n_iters):
        # Take random sample of data
        if bootstrap:
            X_s, _, Y_s, _ = \
                train_test_split(X, Y, random_state=random_state,
                                 **split_options)
            if isinstance(random_state, int):
                random_state += 1
        
        # Fit estimator and store coefficients. Take first column in coef is
        # 2-dimensional
        for estimator in estimators:
            estimator.fit(X_s, Y_s)
            if len(estimator.coef_.shape) == 2:
                coefs[type(estimator).__name__][:,i] = estimator.coef_[:,0]
            else:
                coefs[type(estimator).__name__][:,i] = estimator.coef_
                
    # Store coefficients as a dataframe
    inds = X.columns
    cols = range(1, n_iters+1)
    coefs = {
        k:pd.DataFrame(v, index=inds, columns=cols) for k, v in coefs.items()}

    if return_scaled_X:
        return coefs, X_s
    else:
        return coefs


def _mean_summary(coef_dict, return_std=False):
    """Return means and standard deviations from coef_dict dictionary
    """
    n_models = len(coef_dict.keys())
    features = coef_dict[list(coef_dict.keys())[0]].index
    
    mean = np.zeros((len(features), n_models))
    std = np.zeros_like(mean)
    for i, key in enumerate(coef_dict.keys()):
        mean[:, i] = coef_dict[key].mean(axis=1)
        std[:, i] = coef_dict[key].std(axis=1)    # Default ddof = 1
       
    if return_std:
        return mean, std
    else:
        return mean


def plot_stability(coef_matrix, title=None, vline_kwargs={},
                   bp_kwargs={}, hm_kwargs={}):
    """Plot heatmap and boxplot of model coefficients
    
    Input is dataframe of model coefficients
    """
    # Set default suptitle and kwargs for plots
    if title is None:
        title = ""
    if len(vline_kwargs) == 0:
        vline_kwargs={"ls":":", "c":"g"}
    if len(bp_kwargs) == 0:
        flierprops = dict(markersize=3)
        bp_kwargs={"flierprops":flierprops}
    
    # Set inputs and labels
    features = coef_matrix.index
    coefs = coef_matrix.values
    positions = [i+0.5 for i in range(len(features))]
    xticklabels = np.arange(1, coefs.shape[1]+1)
    vlim = np.amax(np.abs(coefs))*1.05
    
    fig, axes = plt.subplots(1, 2, figsize=(12,8), sharey=True)
    plt.suptitle(title, fontsize=16)
    
    # sns.boxplot(data=coef_matrix.T, orient="h", ax=axes[0], **bp_kwargs)    
    axes[0].boxplot(coefs.T, vert=False, labels=features,
                    positions=positions, **bp_kwargs)
    axes[0].axvline(**vline_kwargs)
    axes[0].set_xlabel("Model coefficients")
    sns.heatmap(data=coefs, vmin=-vlim, vmax=vlim, cmap="vlag", 
                xticklabels=xticklabels, yticklabels=features, ax=axes[1],
                **hm_kwargs)
    axes[1].set_xlabel("Iteration #")
    
    plt.tight_layout()
    # plt.show()
    
    return fig
        

def plot_mean_coef_heatmap(coef_dict, title=None, hm_kwargs={}):
    """Plot heatmap mean model coefficients across dictionary of models
    """
    # Set default suptitle and kwargs for plots
    if title is None:
        title = "Mean coefficients"
    models = list(coef_dict.keys())
    features = coef_dict[list(coef_dict.keys())[0]].index
    
    # Return mean coefficients
    mean_coefs = _mean_summary(coef_dict)
    
    fig, ax = plt.subplots()
    plt.suptitle(title, fontsize=16)
    
    sns.heatmap(data=mean_coefs, vmin=-1, vmax=1, cmap="vlag", 
                xticklabels=models, yticklabels=features, ax=ax,
                **hm_kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    ax.set_xlabel("Model")
    ax.set_ylabel("Features")
    
    plt.tight_layout()
    # plt.show()
    
    return fig



def coef_stats_dict(coef_dict, alpha=0.05):
    """Return dictionary of statistics from coef_dict
    """    
    # n_models = len(coef_dict.keys())
    features = coef_dict[list(coef_dict.keys())[0]].index
    sl = (alpha/2)*100
    
    coef_stats = {}
    for key in coef_dict.keys():
        # Store coefficient stats
        d = {}
        d["Mean"] = coef_dict[key].mean(axis=1)
        d["Standard Deviation"] = coef_dict[key].std(axis=1)
        d["Median"] = np.percentile(coef_dict[key], 0.5, axis=1)
        d["Lower Quartile"] = np.percentile(coef_dict[key], sl, axis=1)
        d["Upper Quartile"] = np.percentile(coef_dict[key], 100-sl, axis=1)
        stats = pd.DataFrame(d, index=features)
        
        coef_stats[key] = stats
    
    return coef_stats



# Model evaluation
# ----------------
    
def _create_control_feature(X, y, positive_control=True, sigma=0.,
                            random_state=None, name="control_variable"):
    """Create a positive or negative control feature
    
    Adds control feature to array of covariates, X.
    Setting positive_control=True creates a 'positive' control feature and
    setting positive_control=False creates a 'negative' control feature
    """
    # If positive control, create a feature highly correlated to y with
    # additional noise defined by sigma.
    # Otherwise (negative control), create a feature uncorrelated with y by
    # shuffling y
    X2 = X.copy()
    y2 = np.array(y).reshape(-1, 1)
    
    if positive_control:
        r = np.random.RandomState(random_state)
        x0 = y2 + r.normal(0, sigma, size=y2.shape)
    else:
        from sklearn.utils import shuffle
        x0 = shuffle(y2, random_state=random_state)
        
    # Add feature onto X (dataframe or array)
    if type(X) == np.ndarray:
        X2 = np.concatenate((X, x0.reshape(-1, 1)), 1)
    elif type(X) == pd.core.frame.DataFrame:
        X2[name] = x0
    
    return X2
       

def _calculate_permutation_scores(estimator, X, y, col_idx,
                                  random_state, n_repeats, scorer,
                                  sample_size=0.2):
    """Calculate score when `col_idx` is permuted
    
    Variation of sklearn's '_calculate_permutation_scores'
    """
    random_state = check_random_state(random_state)

    # Work on a copy of X to to ensure thread-safety in case of threading based
    # parallelism. Furthermore, making a copy is also useful when the joblib
    # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
    # if X is large it will be automatically be backed by a readonly memory map
    # (memmap). X.copy() on the other hand is always guaranteed to return a
    # writable data-structure whose columns can be shuffled inplace.
    X_permuted = X.copy()
    scores = np.zeros(n_repeats)
    shuffling_idx = np.arange(X.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, "iloc"):
            col = X_permuted.iloc[shuffling_idx, col_idx]
            col.index = X_permuted.index
            X_permuted.iloc[:, col_idx] = col
        else:
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        
        # Take a random sample of the data and associated labels
        X_sample, _, y_sample, _ = \
            train_test_split(X_permuted, y, train_size=sample_size,
                             random_state=random_state)
        
        # score on testing
        feature_score = scorer(estimator, X_sample, y_sample)
        scores[n_round] = feature_score

    return scores

    
def validate_models(estimators, X, y, scoring=None, n_repeats=5,
                    random_state=None, return_fitted_estimators=False,
                    control_params={}):
    """Validate list of UNFITTED models using control feature
    
    Returns dataframe of differences between the permuted scores and baseline
    score for the given estimator after fitting on the data with the added
    control feature.
    Optionally returns dictionary of estimators (after fitting with additional
    control feature) and baseline scores if return_fitted_estimators=True
    """    
    if not hasattr(estimators, "__iter__"):
        estimators = [estimators]
    # if random_state is not None:
    #     control_params["random_state"] = random_state
    
    Xs = _create_control_feature(X=X, y=y, random_state=random_state,
                                 **control_params)
    
    # Create arrays to store results
    scores = np.zeros((n_repeats, len(estimators)))
    baseline_scores = np.zeros(len(estimators))
    models = []
    estimators = estimators.copy()    # Break links to original list
        
    for i, estimator in enumerate(estimators):            
        # Get estimator name then calculate permutation importance
        if type(estimator).__name__ == "KerasRegressor":
            models.append("MLP")
            
            # Circumvent annoying Keras warnings
            if hasattr(Xs, "iloc"):
                Xs = Xs.copy().values
            if hasattr(y, "iloc"):
                y = y.copy().values
                
            estimator.set_params(input_dim=Xs.shape[1])
        else:
            models.append(type(estimator).__name__)

        estimator.fit(Xs, y)
        scorer = check_scoring(estimator, scoring=scoring)
        baseline_scores[i] = scorer(estimator, Xs, y)
        
        scores[:,i] = \
            _calculate_permutation_scores(estimator=estimator,
                                          X=Xs, y=y,
                                          col_idx=Xs.shape[1]-1,
                                          random_state=random_state,
                                          n_repeats=n_repeats,
                                          scorer=scorer)
    
        # # Calculate difference in baseline and permuted scores
        # scores[:,i] = baseline_scores[i] - scores[:,i]
        
    scores = pd.DataFrame(scores, index=np.arange(1, n_repeats+1),
                          columns=models)
    model_dict = dict(estimator_name=models,
                      fitted_estimator=estimators)
    
    if return_fitted_estimators:
        return Bunch(scores=scores, baseline_scores=baseline_scores,
                     model_dict=model_dict)
    else:
        return Bunch(scores=scores, baseline_scores=baseline_scores)


def perm_importances(estimators, X, y, scoring=None, n_repeats=5,
                     n_jobs=-2, random_state=None):
    """Get permutation importances across all FITTED estimators
    
    Returns dictionary of all permutations scores
    """    
    if not hasattr(estimators, "__iter__"):
        estimators = [estimators]
       
    # Create dictionary to store results
    importance_dict = {}
    
    for estimator in estimators:
        
        # Get estimator name then calculate permutation importance
        if type(estimator).__name__ == "KerasRegressor":
            name = "MLP"
            jobs = None    # n_jobs not supported for KerasRegressor
            
            # Circumvent annoying Keras warnings
            if hasattr(X, "iloc"):
                X = X.copy().values
            if hasattr(y, "iloc"):
                y = y.copy().values
                
        else:
            name = type(estimator).__name__
            jobs = n_jobs
        
        importance_dict[name] = \
            permutation_importance(estimator=estimator, X=X, y=y,
                                   scoring=scoring, n_repeats=n_repeats,
                                   n_jobs=jobs, random_state=random_state)
            
    return importance_dict


def get_p(n, n_distribution, alpha=0.05):
    """Return p value of entry n given a distribution with sig. level alpha
    """
    sl = (alpha/2)*100
    i = (np.searchsorted(n_distribution, n)+1)/(len(n_distribution)+1)
    if i > np.percentile(np.insert(n_distribution, 0, n), 50):
        p = 1 - i
    else:
        p = i
    lt = np.percentile(n_distribution, sl)
    ut = np.percentile(n_distribution, 100-sl)
    
    return Bunch(p_val=p, lower_tail=lt, upper_tail=ut)