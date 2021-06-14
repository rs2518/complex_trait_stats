import os

import numpy as np

from sklearn.model_selection import train_test_split

from cts.utils import ROOT, RAW_DATA, TRAIN_TEST_PARAMS
from cts.utils import (load_dataframe,
                       process_data,
                       create_directory,
                       load_models,
                       plot_rf_feature_importance)



# Create directory for figures
path = os.path.join(ROOT, "figures", "stability")
create_directory(path)

# Load data and split data into training and testing sets
df = load_dataframe(RAW_DATA)
data = process_data(df)
X = data.drop(['p_value'], axis=1)
y = -np.log10(data["p_value"])
X_train, X_test, y_train, y_test = train_test_split(X, y, **TRAIN_TEST_PARAMS)

# Load models
fitted_models = load_models()



# =============================================================================
# Random forest variable importances
# =============================================================================

# Variable importance plot
# ------------------------
# Plot random forest importances
fig = plot_rf_feature_importance(forest=fitted_models["Random Forest"],
                                 title="Random Forest feature importances",
                                 feature_names=X.columns,
                                 ordered="ascending")
fig.savefig(os.path.join(path, "rf_importances.png"), bbox_inches="tight")
