import os

import pandas as pd

from cts.utils import ROOT
from cts.utils import (load_models,
                       get_array_results,
                       plot_perm_importance,
                       plot_lr_rf_perm_imp)


# Get model names
names = load_models(fitted=False).keys()

# Load temp file data
path = os.path.join(ROOT, "figures/permutation_importance")
results = pd.DataFrame()
prefix = "tmp_perm_"
    
for name in names:
    try:
        if name == "Random Forest":
            d = get_array_results()
        else:
            file = prefix+name.replace(" ", "_")+".csv"
            d = pd.read_csv(os.path.join(path, file), index_col=0)
        results = pd.concat([results, d], axis=1)
    except FileNotFoundError:
        print("Some model results are missing. Loop terminated.")
        break

# Plot results
fig = plot_perm_importance(results, edgecolor="white", alpha=0.75)
figpath = os.path.join(path, "permutation_importances.png")
fig.savefig(figpath)

fig2 = plot_lr_rf_perm_imp(results, edgecolor="white", alpha=0.75)
figpath2 = os.path.join(path, "lr_rf_permutation_importance.png")
fig2.savefig(figpath2)
