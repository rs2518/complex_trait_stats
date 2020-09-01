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