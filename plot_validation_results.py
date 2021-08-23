import os
import sys

import pandas as pd

from cts.utils import ROOT
from cts.utils import load_models, plot_neg_validation, plot_pos_validation


# Get model names
names = load_models(fitted=False).keys()

# Pass command line argument
positive_ctrl = sys.argv[1]
if positive_ctrl.lower() == 'true':
    positive_ctrl = True
elif positive_ctrl.lower() == 'false':
    positive_ctrl = False

# Load temp file data
path = os.path.join(ROOT, "figures/validation")
results = pd.DataFrame()
if positive_ctrl:
    prefix = "tmp_pos_"
else:
    prefix = "tmp_neg_"
    
for name in names:
    try:
        file = prefix+name.replace(" ", "_")+".csv"
        d = pd.read_csv(os.path.join(path, file), index_col=0)
        results = pd.concat([results, d], axis=0)
    except FileNotFoundError:
        print("Some model results are missing. Loop terminated.")
        break

# Plot results
if positive_ctrl:
    fig = plot_pos_validation(results)
    figpath = os.path.join(path, "positive_control_validation.png")
else:
    fig = plot_neg_validation(results)
    figpath = os.path.join(path, "negative_control_validation.png")    
fig.savefig(figpath)
