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



ROOT = "~/Desktop/Term 3 MSc Project/complex_trait_stats"
RAW_DATA = "snp_raw.csv"
PROCESSED_DATA = "snp_processed.csv"



def load_dataframe(file):
    """Load data from ROOT folder
    """
    filepath = os.path.join(ROOT, "data", file)
    data = pd.read_csv(filepath, index_col=1)
    
    data.drop(data.columns[0], axis=1, inplace=True)
    
    return _set_categories(data)


def _set_categories(dataframe):
    """Convert columns with 'object' dtype to 'category'
    """
    cols = dataframe.select_dtypes(include=["object"]).columns
    for col in cols:
        dataframe[col] = dataframe[col].astype("category")
        
    return dataframe


def process_category(data):
    """Convert categorical into binary predictors.
    
    Preprocess data for sklearn models that require data to be numerical
    """
    # Binarise all categorical variables
    processed_data = pd.get_dummies(data)
    
    # Move label to last column
    cols = processed_data.columns.to_list()
    cols.append(cols.pop(cols.index("p_value")))
    
    return processed_data[cols]



def plot_coefs(coefs, names, conf_int=None, cmap="default"):
    """Coefficient plot
    
    Plot coefficients against their respective names. If 'conf_int' is given,
    error bars are added to the plot to show 95% confidence interval.
    
    Setting cmap to "signif" highlights significant coefficients in red whilst
    insignificant coefficients are in light grey.
    Setting cmap to "rainbow" returns each coefficient in different colour.
    Otherwise, cmap is set to default or if cmap="default" but no confidence
    interval is given.
    
    """
    # Set errorbars (if given)
    if conf_int is None:
        yerr = None
        title = ""
    else:
        yerr = np.abs(conf_int.T)
        title = " w/ 95% conf. interval"
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

    
    plt.figure()
    
    plt.errorbar(names, coefs, yerr=yerr, ecolor=cmap, ls="none")
    plt.scatter(names, coefs, c=cmap)
    plt.xticks(rotation=90)
    plt.ylabel("Coefficient "+r'$x_i$')
    plt.title("Coefficent plot" + title)
    plt.hlines(0, len(names), 0, colors="grey", linestyles="--")
    
    # plt.show()



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
    dict_keys = ["rank_test_score", "mean_test_score", "std_test_score"]
    dict_keys += [key for key in cv_results_.keys() if "param_" in key]
    d = {key:cv_results_[key] for key in dict_keys}
    
    if ordered is None:
        return pd.DataFrame(data=d)
    elif ordered == "ascending":
        sort = True
    elif ordered == "descending":
        sort = False
        
    return pd.DataFrame(data=d).sort_values("rank_test_score", ascending=sort)



def plot_true_vs_pred(y_true, y_pred, title=None, marker=".", markercolor=None,
                      edgecolor="w", ls="--", linecolor="red", **kwargs):
    """Plot true y values against predicted y values
    """
    if title is None:
        title = "True vs. Predicted"
    
    # Set width of marker edge and line endpoints for better visuals
    lw = 1/(len(y_true)**0.25)
    lim = [min(y_true), max(y_true)]
    
    # Text box
    # corr, p_value = pearsonr(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]
    # sse = MSE(y_true, y_pred) * len(y_true)
    r2 = R2(y_true, y_pred)
    text = "\n".join((
        r"Pearson's $\rho = %.4f$" % (corr),
        r"$\mathit{R}^2 = %.4f$" % (r2)))
    bbox = dict(facecolor='wheat', alpha=0.5)
    
    fig, ax = plt.subplots()
    
    ax.scatter(y_true, y_pred, marker=".", linewidth=lw, edgecolor="w",
               **kwargs)
    ax.plot(lim, lim, c=linecolor)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment="top",
            fontsize=12, bbox=bbox)
    ax.set_title(title)

    # plt.show()



# Stability analysis
# ------------------

def _coef_dict(estimators, X, Y, n_iters=5, bootstrap=False,
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
        
        # Fit estimator and store coefficients
        for estimator in estimators:
            estimator.fit(X_s, Y_s)
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
    """Return means and standard deviations from _coef_dict dictionary
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
    
    fig, axes = plt.subplots(1, 2, figsize=(12,8), sharey=True)
    plt.suptitle(title, fontsize=16)
    
    # sns.boxplot(data=coef_matrix.T, orient="h", ax=axes[0], **bp_kwargs)    
    axes[0].boxplot(coefs.T, vert=False, labels=features,
                    positions=positions, **bp_kwargs)
    axes[0].axvline(**vline_kwargs)
    axes[0].set_xlabel("Model coefficients")
    sns.heatmap(data=coefs, vmin=-1, vmax=1, cmap="vlag", 
                xticklabels=xticklabels, yticklabels=features, ax=axes[1],
                **hm_kwargs)
    axes[1].set_xlabel("Iteration #")
    
    plt.tight_layout()
    # plt.show()
        

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
    
    # plt.show()
    
    

# def _get_boxplot_stats(boxplot, labels):
#     """Get summary stats from boxplot
#     """
#     stats = []
#     for i in range(len(labels)):
#         d = dict(feature=labels,
#                   lower_whisker=boxplot["whiskers"][i*2].get_xdata()[1],
#                   q1=boxplot["boxes"][i].get_xdata()[1],
#                   q2=boxplot["medians"][i].get_xdata()[1],
#                   q3=boxplot["boxes"][i].get_xdata()[2],
#                   upper_whisker=boxplot['whiskers'][(i*2)+1].get_xdata()[1])
#         stats.append(d)

#     return pd.DataFrame(stats)