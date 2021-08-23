import os

import numpy as np
import pandas as pd

from cts.utils import ROOT, RAW_DATA
from cts.utils import load_dataframe
from cts.utils import (compute_assoc,
                       plot_corr_heatmap,
                       plot_log_p_value)


# Define directories (and create if non-existent) to save plots
fig_dir = os.path.join(ROOT, "figures")
eda_figpath = os.path.join(fig_dir, "exploratory")
if not os.path.exists(eda_figpath):
    os.mkdir(eda_figpath)
    print("Created '{}' directory!".format(
        eda_figpath[eda_figpath.rfind("/")+1:]))


# Load data
df = load_dataframe(RAW_DATA)

# Print full dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Log-transform p-values and display distributions
df["log_p"] = -np.log10(df["p_value"])
fig = plot_log_p_value(df, shade=True, linewidth=0, color="r")
figpath = os.path.join(eda_figpath, "pvalue_transformation.png")
fig.savefig(figpath)

df.drop(["p_value"], axis=1, inplace=True)    # Drop raw p-values


# Effect/Reference Allele counts
print(df.value_counts(["A1", "A2"])/df.value_counts(["A1", "A2"]).sum())
print(df.value_counts(["A1"])/df.value_counts(["A1"]).sum())
print(df.value_counts(["A2"])/df.value_counts(["A2"]).sum())

# Hardy-Weinberg p-value
print("HWE_P < 0.05 :", df[df["HWE_P"]<=0.05]["HWE_P"].shape[0]/len(df))
print("HWE_P < 0.005 :", df[df["HWE_P"]<=0.005]["HWE_P"].shape[0]/len(df))

# Descriptive statistics
desc = df.describe(include="all").loc[[
    "mean", "50%", "std", "min", "max", "top", "freq"]].T
cat_cols = df.select_dtypes(include=["object", "category"]).columns.to_list()
num_cols = [col for col in df.columns if col not in cat_cols]
desc = desc.loc[num_cols+cat_cols]    # Reorder columns
print(desc)

# Plot correlation heatmap
corr, _, _, _ = compute_assoc(df, cat_cols, clustering=True)
fig = plot_corr_heatmap(corr, cmap="vlag", vmin=-1, vmax=1,
                        annot=True, fmt='.2f')
figpath = os.path.join(eda_figpath, "association_heatmap.png")
fig.savefig(figpath)
