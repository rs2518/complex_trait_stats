# =============================================================================
# TEMPORARY IMPORT FOR USE IN AN INTERPRETER
# =============================================================================
import os, sys

directory = "Desktop/Term 3 MSc Project"

path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================

import os

from complex_trait_stats.utils import ROOT, RAW_DATA
from complex_trait_stats.utils import load_dataframe, process_data
from complex_trait_stats.utils import compute_assoc, plot_corr_heatmap


# Define directories (and create if non-existent) to save plots
fig_dir = os.path.join(ROOT, "figures")
eda_figpath = os.path.join(fig_dir, "exploratory")
if not os.path.exists(eda_figpath):
    os.mkdir(eda_figpath)
    print("Created '{}' directory!".format(
        eda_figpath[eda_figpath.rfind("/")+1:]))


# Load data and add column of ones for intercept
df = load_dataframe(RAW_DATA)
# data = process_data(df)


# Plot correlation heatmap
cat_cols = df.select_dtypes(include=["object", "category"]).columns.to_list()

corr, _, _, _ = compute_assoc(df, cat_cols, clustering=True)
fig = plot_corr_heatmap(corr)
figpath = os.path.join(eda_figpath, "association_heatmap.png")
fig.savefig(figpath)