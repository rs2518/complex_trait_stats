import os
import math
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

from joblib import Parallel, delayed

from collections import Counter
from sklearn.base import clone
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
from scipy.stats import pearsonr, entropy, combine_pvalues
from statsmodels.stats.multitest import multipletests

from sklearn.inspection._permutation_importance import \
    _calculate_permutation_scores
from sklearn.metrics import check_scoring
from sklearn.utils import Bunch, check_random_state
# from sklearn.utils import Bunch, check_random_state, check_array



#ROOT = os.path.join(os.path.expanduser("~"),
#                    "Desktop/Term 3 MSc Project/complex_trait_stats")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "cts")
RAW_DATA = "geneatlas_data_22chr_1000n.csv"
TOY_DATA = "geneatlas_data_4chr_10n.csv"
ANNOTATED_DATA = "giantanno.hg19_multianno.csv"
# TOY_PROCESSED_DATA = "snp_processed_4chr5000.csv"

MODEL_PATH = os.path.join(PKG, "models")
MODEL_DICT = {"Linear Regression":"LinearRegression_model.gz",
              "Lasso":"Lasso_model.gz",
              "Ridge":"Ridge_model.gz",
              "Elastic Net":"ElasticNet_model.gz",
              "PLS":"PLSRegression_model.gz",
              "Random Forest":"RandomForestRegressor_model.gz",
              "MLP":"MLPRegressor_model.gz"}
#MODEL_DICT = {"Linear Regression":"LinearRegression_model.gz",
#              "Lasso":"Lasso_model.gz",
#              "Ridge":"Ridge_model.gz",
#              "Elastic Net":"ElasticNet_model.gz",
#              "PLS":"PLSRegression_model.gz",
#              "Random Forest":"RandomForestRegressor_model.gz",
#              "MLP":"MLP_model.gz"}

TRAIN_TEST_PARAMS = dict(test_size=0.3, random_state=1010)
CV_FOLDS = 10



# Data preprocessing
# ------------------

def load_dataframe(file):
    """Load data from ROOT folder
    """
    filepath = os.path.join(ROOT, "data", file)
    data = pd.read_csv(filepath, index_col=5)
    # data = pd.read_csv(filepath, index_col=6)    # With start/End position
    
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


def _order_chromosomes(dataframe):
    """Order chromosome columns
    """
    name = "Chr"
    
    chrom = [col for col in dataframe.columns if name in col]
    cols = [col for col in dataframe.columns if col not in chrom]
    new_chrom = [name+str(i+1) for i in range(len(chrom))] 
    
    new_cols = cols[:-1] + new_chrom + [cols[-1]]    # p-value as last column
    dataframe = dataframe[new_cols]
    
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
    
    # Drop prefix for chromosome column
    prefix = "Chromosome_"
    chroms = [col for col in processed_data.columns if prefix in col]
    new_chroms = {col:col[len(prefix):].title() for col in chroms}
    processed_data.rename(columns=new_chroms, inplace=True)
    
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
    """Process features and order chromosome columns
    """    
    return _order_chromosomes(binarise_category(scale_numeric(data)))



# Other utilities
# ---------------

def create_directory(path):
    """Create directory for given path if it doesn't exist
    """
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created '{}' directory!".format(path[path.rfind("/")+1:]))
        

def save_models(models):
    """Save and compress list of models within 'models' directory
    
    Model filenames are determined by '__name__' attribute except for MLP 
    ('PickleableKerasRegressor' is replaced with 'MLP')
    """
    if not isinstance(models, list):
        models = [models]

    # Create 'saved_models' if it doesn't exist
    dir_path = os.path.join(MODEL_PATH, "saved_models")
    create_directory(dir_path)
    
    # Save models
    for model in models:
        name = type(model).__name__
        if name == "PickleableKerasRegressor":
            filename = os.path.join(dir_path, "MLP_model.gz")
        else:
            filename = os.path.join(dir_path, name+"_model.gz")
        
        joblib.dump(model, filename, compress=True)
    

def load_models(fitted=True):
    """Load all saved models to environment in a dictionary
    """
    # Create empty dictionary of models
    models = {}
    
    # Load models to dictionary
    for model, file in MODEL_DICT.items():
        try:
            models[model] = \
                joblib.load(os.path.join(MODEL_PATH, "saved_models", file))
        except FileNotFoundError:
            pass

    if not fitted:
        models = {key:clone(model) for key, model in models.items()}
                
    return models



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



def cv_table(cv_results_, ordered=None, return_train_scores=False):
    """Table of cross-validation results
    
    Returns table "mean_test_scores" and "std_test_scores" along with
    respective model parameters from cv_results_. Can sort results according
    to rank_test_score by setting sort="ascending" or sort="descending".
    """    
    dict_keys = ["rank_test_score", "mean_test_score", "std_test_score"]
    if return_train_scores:
        dict_keys += ["mean_train_score", "std_train_score"]
        
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



# Stability analysis
# ------------------

def coef_dict(estimators, X, Y, n_iters=5, n_jobs=None,
              random_state=None, scale_X=False,
              return_scaled_X=False, **split_options):
    """Returns dictionary with array of model coefficients at each iteration
    for a given dictionary of estiamtors
    """
    # Instantiate results dictionary
    # coefs = {key:np.zeros((X.shape[1], n_iters)) for key in estimators.keys()}
    coefs = {}

    # Break links to original data
    X_s = X.copy()
    Y_s = Y.copy()
    
    # Standard scale numerical features
    if scale_X:
        num_cols = X_s.select_dtypes(
            exclude=["object", "category"]).columns
        scaler = StandardScaler()
        X_s[num_cols] = scaler.fit_transform(X_s[num_cols])

    # Set random seeds and run parallel loop
    random_state = check_random_state(random_state)
    seed = [random_state.randint(np.iinfo(np.int32).max + 1)
            for i in range(n_iters)]
    
    for key, estimator in estimators.items():
        scores = Parallel(n_jobs=n_jobs)(delayed(_extract_coefs)(
            estimator, X_s, Y_s, seed[i], **split_options) for i in range(n_iters))
        
        coefs[key] = pd.DataFrame(np.array(scores).T,
                                  index=X.columns,
                                  columns=range(1, n_iters+1))

    if return_scaled_X:
        return coefs, X_s
    else:
        return coefs


def _extract_coefs(estimator, X, Y, random_state, **split_options):
    """Extract model coefficients
    """
    # Take random sample of data
    X_s, _, Y_s, _ = \
        train_test_split(X, Y, random_state=random_state, **split_options)
    
    # Fit estimator and store coefficients. Take first column in coef is
    # 2-dimensional
    estimator.fit(X_s, Y_s)
            
    return estimator.coef_.ravel()
    

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
    
    uq = np.amax(np.percentile(coefs, 75, axis=1))
    lq = np.amin(np.percentile(coefs, 25, axis=1))
    if uq-lq>20:
        vlim = np.amax(np.abs([lq, uq]))
    else:
        vlim = np.amax(np.abs(coefs))*1.05

    fig, axes = plt.subplots(1, 2, figsize=(12,8), sharey=True)
    plt.suptitle(title, fontsize=16)
    
    # sns.boxplot(data=coef_matrix.T, orient="h", ax=axes[0], **bp_kwargs)    
    axes[0].boxplot(coefs.T, vert=False, labels=features,
                    positions=positions, **bp_kwargs)
    axes[0].axvline(**vline_kwargs)
    axes[0].set_xlabel("Model coefficients")
    sns.heatmap(data=coefs, vmin=-vlim, vmax=vlim, cmap="vlag", 
                yticklabels=features, ax=axes[1],
                **hm_kwargs)
    axes[1].set_xlabel("Iteration #")
    axes[1].tick_params(bottom=False)

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
    
    # Set figure inputs
    a = np.clip(np.amax(np.abs(mean_coefs)), a_min=None, a_max=[1.3])[0]
    vlim = a*1.05
    
    fig, ax = plt.subplots(figsize=(8,8))
    plt.suptitle(title, fontsize=16)
    
    sns.heatmap(data=mean_coefs, vmin=-vlim, vmax=vlim, cmap="vlag", 
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


def plot_rf_feature_importance(forest, feature_names, palette="hls",
                               ordered=None, title=None, **kwargs):
    """Plots bar graph of feature importances for Random Forest
    
    NOTE: Feature importances often do not perform well for high cardinality
    features
    """
    importances = forest.feature_importances_
    
    if title is None:
        title = "Random Forest Feature Importances"
    if ordered is None:
        sorted_idx = np.arange(len(importances)-1, -1, step=-1)
    elif ordered == "ascending":
        sorted_idx = importances.argsort()
    elif ordered == "descending":
        sorted_idx = (-importances).argsort()
    
    importances = forest.feature_importances_
    
    cmap = sns.color_palette(palette=palette,
                             n_colors=len(importances[importances > 0]),
                             desat=.65)

    y_ticks = np.arange(0, len(feature_names))
    err = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    
    tab = np.array([importances[sorted_idx].ravel(), err[sorted_idx].ravel()]).T
    print(pd.DataFrame(tab, index=feature_names[sorted_idx],
                       columns=["mean", "std"]))

    fig, ax = plt.subplots(figsize=(8,8))
    ax.barh(y_ticks, importances[sorted_idx], xerr=err[sorted_idx],
            color=cmap, **kwargs)
    ax.set_xlim(0,1)
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title(title)
    
    plt.tight_layout()
    # plt.show()
    
    return fig



# Model evaluation
# ----------------

def _create_control_feature(X, y, sigma=0., random_state=None,
                            name="control_variable"):
    """Create a positive control feature
    
    Adds control feature with Gaussian distributed noise to array of
    covariates, X
    """
    # For positive control, create a feature highly correlated to y with
    # additional noise defined by sigma.
    X2 = X.copy()
    y2 = np.array(y).reshape(-1, 1)
    r = np.random.RandomState(random_state)
    x0 = y2 + r.normal(0, sigma, size=y2.shape)
    
    # Add feature onto X (dataframe or array)
    if type(X) == np.ndarray:
        X2 = np.concatenate((X, x0.reshape(-1, 1)), 1)
    elif type(X) == pd.core.frame.DataFrame:
        X2[name] = x0
    
    return X2


def _pos_permutation_score(estimator, X, y, col_idx, random_state, scorer):
    """Calculate score when col_id is permuted
    
    Variation of sklearn's '_calculate_permutation_scores'.
    """
    random_state = check_random_state(random_state)
    
    # Break links to original
    X_permuted = X.copy()
    
    # Shuffle labels and return score
    shuffling_idx = np.arange(X.shape[0])
    random_state.shuffle(shuffling_idx)
    if hasattr(X_permuted, "iloc"):
        col = X_permuted.iloc[shuffling_idx, col_idx]
        col.index = X_permuted.index
        X_permuted.iloc[:, col_idx] = col
    else:
        X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        
    return scorer(estimator, X_permuted, y)


def _neg_permutation_score(estimator, X, y, random_state, scorer):
    """Calculate score when labels are permuted
    """
    random_state = check_random_state(random_state)
    
    # Break links to original
    y_permuted = y.copy()
    
    # Shuffle labels and return score
    shuffling_idx = np.arange(X.shape[0])
    random_state.shuffle(shuffling_idx)
    y_permuted = y_permuted[shuffling_idx]

    return scorer(estimator, X, y_permuted)


def _validation_permutation_score(estimator, X, y, col_idx, random_state,
                                  scorer, positive_ctrl):
    """General permutation score for validation methods
    
    If 'positive_ctrl=True', permute X. Otherwise, permute y
    """    
    if positive_ctrl:
        return _pos_permutation_score(estimator=estimator,
                                      X=X, y=y, col_idx=col_idx,
                                      random_state=random_state,
                                      scorer=scorer)
    else:
        return _neg_permutation_score(estimator=estimator, X=X, y=y,
                                      random_state=random_state,
                                      scorer=scorer)

    
def validate_sample(estimator, X, y, n_jobs=None, scoring=None, n_repeats=5,
                    positive_ctrl=True, random_state=None, version="fn",
                    verbose=0, control_params={}):
    """Model validation for models
    
    Used to carry out internal validation on individual samples.
    
    Returns dataframe of differences between the permuted scores and baseline
    score for the given estimator after fitting on the data.
    
    For negative control validation (i.e. positive_ctrl=False), 'version'
    returns information on the true positive rate 'tpr' or the false positive
    rate 'fpr'
    """
    # Initial seed generator using random_state
    r = check_random_state(random_state)
    rng = r.randint(np.iinfo(np.int32).max+1, size=n_repeats)
    
    # Check if bool
    if type(positive_ctrl) != bool:
        raise TypeError("Argument should be boolean")
        
    if positive_ctrl:
        Xs = _create_control_feature(X=X, y=y, random_state=random_state,
                                     **control_params)
    else:
        Xs = X.copy()
    ys = y.copy()
    
    # Create arrays to store results
    scorer = check_scoring(estimator, scoring=scoring)

    # scores = np.zeros((n_repeats, len(estimators)))
    if positive_ctrl:
        estimator.fit(Xs, ys)
        baseline_score = scorer(estimator, Xs, ys)
    else:
        if version == "tpr":
            baseline_score = scorer(estimator, Xs, ys)
        elif version == "fpr":
            init_seed = r.randint(np.iinfo(np.int32).max)
            baseline_score = \
                _neg_permutation_score(estimator=estimator, X=Xs, y=ys,
                                       random_state=init_seed,
                                       scorer=scorer)
    
    scores = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(
        _validation_permutation_score)(
            estimator, Xs, ys, Xs.shape[1]-1, seed, scorer, positive_ctrl)
            for seed in rng)
    scores = np.array(scores)
    
    return Bunch(scores=scores, baseline_score=baseline_score)
    
    
def _bootstrap_wrapper(func, estimator, X, y, n_samples=3,
                       sample_size=0.3, random_state=None, **kwargs):
    """Wrapper function for performing operation over bootstrapped samples
        
    **kwargs are from 'func'
    """
    r = check_random_state(random_state)
    
    # Loop through n_samples
    res = {}
    idxs = np.arange(X.shape[0])
    for n in range(n_samples):

        # Print sample for log file
        print("Running sample ", n+1, " ...")

        sample_idx = \
            train_test_split(
                idxs, train_size=sample_size,
                random_state=r.randint(0, np.iinfo(np.int32).max+1))[0]
        
        # Take a random sample of the data
        ys = y[sample_idx].copy()
        if hasattr(X, "iloc"):
            Xs = X.iloc[sample_idx, :].copy()
        else:
            Xs = X[sample_idx, :].copy()
        
        # Add sample index into Bunch object
        s = func(estimator, Xs, ys, random_state=random_state, **kwargs)
        s.sample = sample_idx
        res["sample_"+str(n+1)] = s
    
    return res


def model_validation(estimator, X, y, n_samples=3, n_repeats=5,
                     sample_size=0.3, positive_ctrl=True,
                     random_state=None, verbose=0, **kwargs):
    """Run validate_sample over _bootstrap_wrapper
    """
    # Set function arguments
    kwargs["n_repeats"] = n_repeats
    kwargs["positive_ctrl"] = positive_ctrl
    
    return _bootstrap_wrapper(validate_sample, estimator=estimator,
                              X=X, y=y, n_samples=n_samples,
                              sample_size=sample_size,
                              random_state=random_state,
                              verbose=verbose,
                              **kwargs)


def permutation_importance(estimator, X, y, scoring=None, n_repeats=5,
                           n_jobs=None, random_state=None, verbose=0):
    """Variation of sklearn's permutation importance
    
    Runs permutation importance as a hypothesis test
    """
    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    sample_weight = None
    scorer = check_scoring(estimator, scoring=scoring)
    baseline_score = scorer(estimator, X, y)

    scores = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(
        _calculate_permutation_scores)(estimator, X, y, sample_weight,
                                       col_idx, random_seed,
                                       n_repeats, scorer)
                                       for col_idx in range(X.shape[1]))

    scores = np.array(scores)
    
    return Bunch(baseline_score=baseline_score, scores=scores)


def perm_importances(estimator, X, y, n_samples=3, n_repeats=5,
                     sample_size=0.3, scoring=None, n_jobs=-2,
                     random_state=None, verbose=0,  **kwargs):
    """Run sklearn permutation_importances over _bootstrap_wrapper
    """
    # Set function arguments
    kwargs["n_repeats"] = n_repeats
    kwargs["scoring"] = scoring
    kwargs["n_jobs"] = n_jobs
    
    return _bootstrap_wrapper(permutation_importance,
                              estimator=estimator,
                              X=X, y=y, n_samples=n_samples,
                              sample_size=sample_size,
                              random_state=random_state,
                              verbose=verbose,
                              **kwargs)


def get_p(n, n_distribution, one_tailed=True, alternative="greater"):
    """Return rank of entry n given a distribution
    
    If one_tailed=True, focus on one tail as defined by 'alternative'
    (i.e. alternative="greater" for upper tail, alternative="less" for lower 
    tail).
    If one_tailed=False, focus on the upper tail when n >= median value and 
    the lower tail when n <= median value. 'alternative' is inactive here.
    """
    a = np.insert(n_distribution, 0, n)
    q2 = np.percentile(a, 50)
    
    if one_tailed:
        if alternative=="greater":
            p = 1 - np.searchsorted(a, n)/len(a)
        elif alternative=="less":
            p = np.searchsorted(a, n)/len(a)
    else:
        if n >= q2:
            p = 1 - np.searchsorted(a, n)/len(a)
        else:
            p = np.searchsorted(a, n)/len(a)
    
    return Bunch(p_val=p)

 
def _compute_proportions(results, alpha=0.05, one_tailed=True,
                         alternative="greater", **kwargs):
    """Compute proportion of significant hypothesis tests
    
    **kwargs are from 'multipletests'
    """
    keys = list(results.keys())
    n_samples = len(keys)
    a = np.array([get_p(n=results[keys[sample]].baseline_score,
                        n_distribution=results[keys[sample]].scores,
                        one_tailed=one_tailed,
                        alternative=alternative).p_val
                  for sample in range(n_samples)])
    
    inds, _, _, _ = multipletests(a, alpha=alpha, **kwargs)
        
    return inds.sum(axis=0)/inds.shape[0]


def _compute_combined_p(results, alpha=0.05, one_tailed=True,
                        alternative="greater", **kwargs):
    """Compute combined p-value for each feature
    
    **kwargs are from 'multipletests'
    """
    keys = list(results.keys())
    n_samples = len(keys)
    n_features = results[list(results.keys())[0]].scores.shape[0]
    
    a = np.array([get_p(n=results[keys[sample]].baseline_score,
                        n_distribution=results[keys[sample]].scores[i, :],
                        one_tailed=one_tailed,
                        alternative=alternative).p_val
                  for sample in range(n_samples)
                  for i in range(n_features)]).reshape(-1, n_features).T
    
    # Combine p-values using Fisher's test across samples for each feature    
    pvals = np.array([combine_pvalues(a[i, :])[1] for i in range(n_features)])
    
    # Multiple p-value correction across features
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, **kwargs)
    
    return pvals_corrected


def tabulate_validation(results, index=None, positive_ctrl=True, **kwargs):
    """Tabulate validation results
    """
    if index is None:
        index = [""]
    
    d = {key:_compute_proportions(results[key], **kwargs)
         for key in results.keys()}
            
    return pd.DataFrame(d, index=index)


def tabulate_perm(results, index=None, columns=None, **kwargs):
    """Tabulate permutation importance results
    """
    d = _compute_combined_p(results, **kwargs)
    
    return pd.DataFrame(d, index=index, columns=columns)


def plot_true_vs_pred(preds, xlabel="Truth", ylabel="Prediction",
                      title=None, figsize=(10, 10), palette="hls",
                      **kwargs):
    """Plot true y values against predicted y values
    """
    # Set plot arguments
    if title is None:
        title = "True vs. Predicted"
    
    models = preds.columns.to_list()[:-1]
    cmap = sns.color_palette(palette=palette, n_colors=len(models), desat=.55)
    markers = [".", "^", "D", "x", "+", "p", "s"]
    lim = [1.05*min(preds.min()), 1.05*max(preds.max())]

    # Plot true values vs. predictions for all models
    fig, ax = plt.subplots(figsize=figsize)
    for i, model in enumerate(models):
        sns.regplot(x="Truths", y=model, data=preds,
                    marker=markers[i], color=cmap[i],
                    ax=ax, label=model, **kwargs)
    plt.plot(lim, lim, color="grey", linestyle="--")
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()   
    # plt.show()
    
    return fig


def plot_neg_validation(results, palette="hls", **kwargs):
    """Bar plot for negative control validation results
    """
    # Set plot arguments
    title = r"Proportion of samples where $H_0$ was rejected"
    sub_titles = {"tpr":"True Positive Rate",
                  "fpr":"False Positive Rate"}
    
    models = results.index.to_list()
    x = np.arange(len(models))
    height = results.values
    cmap = sns.color_palette(palette=palette, n_colors=len(models), desat=.55)
    
    # Barplot for each model
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharey=True,
                           gridspec_kw={"wspace":0.2})
        
    for i, col in enumerate(results.columns.to_list()):
        ax[i].bar(x=x, height=height[:, i], color=cmap, **kwargs) 
        
        def _annotate_bar(bars, dp=2):
            """Annotate bar plot to user-defined no. of decimal places
            """
            for bar in bars:
                height = bar.get_height()
                ax[i].annotate("{:.{dp}g}".format(float(height), dp=dp),
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 1),  # 3 points vertical offset
                               textcoords="offset points",
                               ha="center", va="bottom")
        
        _annotate_bar(ax[i].patches, dp=2)    # Annotate each bar
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(models, rotation=270)
        ax[i].set_ylim([0, 1.05])
        ax[i].set_title(sub_titles[col])
        
    plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    # plt.show()
    
    return fig


def plot_pos_validation(results, palette="hls", title=None, **kwargs):
    """Bar plot for negative control validation results
    """
    # Set plot arguments
    if title is None:
        title = r"Proportion of samples where $H_0$ was rejected"
    
    models = results.index.to_list()
    # noise = results.columns.to_list()
    # x = np.arange(len(noise))
    x = results.columns.to_list()
    cmap = sns.color_palette(palette=palette, n_colors=len(models), desat=.55)
    
    # Lineplot for each model
    fig, ax = plt.subplots(figsize=(8,8))
    
    for i, model in enumerate(models):
        y = results.values[i, :]
        plt.plot(x, y, label=model, color=cmap[i], **kwargs)
    
    ax.set_xticks(x)
    # ax.set_xticklabels(noise)
    ax.set_ylim([0, 1.05])
    plt.xlabel(r"Standard Deviation ($\sigma$)")
    plt.ylabel("Proportion")
    plt.legend(loc="upper right")
    plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    # plt.show()    
    
    return fig


def plot_perm_importance(results, cutoff=0.05, cmap=None, title=None,
                         **kwargs):
    """Plot permutation importance p-values
    
    P-values with a value of zero are assigned a value of 10^30 during
    log-transformation.
    """
    # Set plot arguments
    models = results.columns.to_list()
    features = results.index.to_list()
    n_features = len(features)
    if cmap is None:
        cmap = list(sns.color_palette(palette="hls", n_colors=len(models),
                                      desat=.85))
    fontdict = {"fontsize":16}
        
    # Log-transform p-values. Assign arbitrarily small value to zeroed p-values
    a = results.values.flatten()
    min_p = min(min(a[a > 0]), 2e-30)/2    
    data = -np.log10(results.replace(0, min_p))
                
    # Grid of barplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20), sharey=True)
    
    for i, model in enumerate(models):
        axes[i//3, i%3].scatter(np.arange(n_features), data[model].values,
                                color=cmap[i], **kwargs)
        axes[i//3, i%3].set_title(model, fontdict=fontdict)
        # axes[i//3, i%3].set_ylim(-0.05, 1.05)
        axes[i//3, i%3].set_xticks(np.arange(n_features))
        axes[i//3, i%3].set_xticklabels(features, rotation=90)
        axes[i//3, i%3].axhline(-np.log10(cutoff), ls="--", color="grey")
        axes[i//3, i%3].axhline(0, color="black")
    
    axes[2, 1].remove()
    axes[2, 2].remove()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return fig



# Exploratory Data Analysis
# -------------------------

def _conditional_entropy(x,y):
    """Conditional entropy
    
    Code originally from dython package:
    https://github.com/shakedzy/dython/blob/master/dython/nominal.py
        
    For more information, see:
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9?gi=f05f03ff513a
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    ent = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        ent += p_xy * math.log(p_y/p_xy)
        
    return ent


def theil_u(x,y):
    """Categorical-Categorical interaction
    
    Code originally from dython package:
    https://github.com/shakedzy/dython/blob/master/dython/nominal.py
        
    For more information, see:
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9?gi=f05f03ff513a
    """
    s_xy = _conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = entropy(p_x)
    
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def correlation_ratio(categories, measurements):
    """Continuous-Categorical interaction
    
    Code originally from dython package:
    https://github.com/shakedzy/dython/blob/master/dython/nominal.py
        
    For more information, see:
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9?gi=f05f03ff513a
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


def compute_assoc(dataset, nominal_columns, clustering=False):
    """Compute correlation/associations
    
    Calculate the correlation/strength-of-association of features in data-set
    with both categorical and continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Theil's U for categorical-categorical cases
    
    Code originally from dython package:
    https://github.com/shakedzy/dython/blob/master/dython/nominal.py
        
    For more information, see:
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9?gi=f05f03ff513a
    """
    columns = dataset.columns
    nominal_columns = dataset.select_dtypes(
        include=["object", "category"]).columns.to_list()

    corr = pd.DataFrame(index=columns, columns=columns)
    single_value_columns = []
    
    for c in columns:
        if dataset[c].unique().size == 1:
            single_value_columns.append(c)
    for i in range(0, len(columns)):
        if columns[i] in single_value_columns:
            corr.loc[:, columns[i]] = 0.0
            corr.loc[columns[i], :] = 0.0
            continue
        for j in range(i, len(columns)):
            if columns[j] in single_value_columns:
                continue
            elif i == j:
                corr.loc[columns[i], columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        ji = theil_u(
                            dataset[columns[i]],
                            dataset[columns[j]])
                        ij = theil_u(
                            dataset[columns[j]],
                            dataset[columns[i]])
                    else:
                        cell = correlation_ratio(dataset[columns[i]],
                                                 dataset[columns[j]])
                        ij = cell
                        ji = cell
                else:
                    if columns[j] in nominal_columns:
                        cell = correlation_ratio(dataset[columns[j]],
                                                 dataset[columns[i]])
                        ij = cell
                        ji = cell
                    else:
                        cell, _ = pearsonr(dataset[columns[i]],
                                           dataset[columns[j]])
                        ij = cell
                        ji = cell
                corr.loc[columns[i], columns[j]] = \
                    ij if not np.isnan(ij) and abs(ij) < np.inf else 0.0
                corr.loc[columns[j], columns[i]] = \
                    ji if not np.isnan(ji) and abs(ji) < np.inf else 0.0
    # Convert to numeric array
    corr = pd.DataFrame(corr.values, index=columns, columns=columns,
                        dtype="float")
    if clustering:
        corr, _ = cluster_correlations(corr)
        columns = corr.columns
        
    return corr, columns, nominal_columns, single_value_columns


def cluster_correlations(corr_mat, indices=None):
    """Apply agglomerative clustering in order to sort a correlation matrix.
    
    Originally based on:
    https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
    
    Adapted version taken from:
    https://github.com/shakedzy/dython/blob/master/dython/nominal.py
    
    For more information, see:
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9?gi=f05f03ff513a
    """
    if indices is None:
        X = corr_mat.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method='complete')
        indices = sch.fcluster(L, 0.5 * d.max(), 'distance')
    columns = [corr_mat.columns.tolist()[i]
               for i in list((np.argsort(indices)))]
    corr_mat = corr_mat.reindex(columns=columns).reindex(index=columns)
    return corr_mat, indices


def plot_corr_heatmap(corr, **kwargs):
    """Plot heatmap of correlations/associations across features
    """    
    fig, ax = plt.subplots(figsize=(8,8))
    # plt.suptitle(title, fontsize=16)
    
    sns.heatmap(data=corr, ax=ax, **kwargs)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    # ax.set_xlabel("Model")
    # ax.set_ylabel("Features")
    
    plt.tight_layout()
    # plt.show()
    
    return fig


def plot_log_p_value(data, **kwargs):
    """Plot distribution of p-values
    """
    sub_titles = {"p_value":"Raw p-values",
                  "log_p":"Log-transformed p-values"}
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),
                             gridspec_kw={"wspace":0.4})
    
    for i, col in enumerate(sub_titles.keys()):
        sns.kdeplot(data=data[col], ax=axes[i], legend=False, **kwargs)
        axes[i].set_title(sub_titles[col])
        if col == "p_value":
            axes[i].set_xlim(0, 1)
        else:
            lim = np.amax(data[col])
            axes[i].set_xlim(0, lim*1.03)
    
    plt.tight_layout()
    # plt.show()
    
    return fig
