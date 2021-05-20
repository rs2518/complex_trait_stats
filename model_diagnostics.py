import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
    
from complex_trait_stats.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
from complex_trait_stats.utils import (load_dataframe,
                                       process_data,
                                       create_directory,
                                       load_models,
                                       perm_tab,
                                       plot_true_vs_pred)



# Create directory for figures
path = os.path.join(ROOT, "figures", "evaluation")
create_directory(path)

# Load data and split data into training and testing sets
df = load_dataframe(RAW_DATA)
data = process_data(df)
X = data.drop(['p_value'], axis=1)
y = -np.log10(data["p_value"])
X_train, X_test, y_train, y_test = train_test_split(X, y, **TRAIN_TEST_PARAMS)

# Load fitted models
models = load_models()



# =============================================================================
# Model diagnostics
# =============================================================================

# Truths vs. predictions
# ----------------------
# Get model predictions
predictions = {key:model.predict(X_test).ravel()
               for (key, model) in zip(perm_tab.columns, models)}
pred_df = pd.DataFrame(predictions)
pred_df["Truths"] = y_test.values

# Plot true values vs. predictions for all models
fig = plot_true_vs_pred(pred_df,
                        scatter_kws=dict(alpha=0.5, edgecolor="w"),
                        line_kws=dict(linewidth=0.85))
figpath = os.path.join(path, "true_vs_pred.png")
fig.savefig(figpath)