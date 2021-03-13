# =============================================================================
# TEMPORARY IMPORT FOR USE IN AN INTERPRETER
# =============================================================================
import os, sys

directory = "Desktop/Term 3 MSc Project"

path = os.path.join(os.path.abspath('.'), directory)
sys.path.append(path)
# =============================================================================

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from complex_trait_stats.utils import ROOT, RAW_DATA
from complex_trait_stats.utils import load_dataframe, process_data



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

# Process categories with label encoder
le = LabelEncoder()
cat_cols = df.select_dtypes(include=["category"]).columns
df[cat_cols] = le.fit_transform(df[cat_cols])

# Split data into features and labels
X = df.drop(['p_value'], axis=1)
y = -np.log10(df["p_value"])

corr = df.iloc[:, :-1].corr()
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)
plt.show()