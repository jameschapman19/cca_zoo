"""
Hyperparameter Selection
===========================

This script will show how to perform hyperparameter selection
"""

# %%
import numpy as np
import pandas as pd
from sklearn.utils.fixes import loguniform

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV, RandomizedSearchCV
from cca_zoo.models import KCCA

# %%
# Data
# ------
np.random.seed(42)
n = 200
p = 100
q = 100
latent_dims = 1
cv = 3

data = LinearSimulatedData(
    view_features=[p, q], latent_dims=latent_dims, correlation=[0.9]
)

(X, Y) = data.sample(n)
(tx, ty) = data.true_features

# %%
# Grid Search
# -------------
# Hyperparameter selection works in a very similar way to in scikit-learn where the main difference is in how we enter the parameter grid.
# We form a parameter grid with the search space for each view for each parameter.
# This search space must be entered as a list but can be any of
#  - a single value (as in "kernel") where this value will be used for each view
#  - a list for each view
#  - a mixture of a single value for one view and a distribution or list for the other
param_grid = {"kernel": ["poly"], "c": [[1e-1], [1e-1, 2e-1]], "degree": [[2], [2, 3]]}
kernel_reg = GridSearchCV(
    KCCA(latent_dims=latent_dims), param_grid=param_grid, cv=cv, verbose=True
).fit([X, Y])
print(pd.DataFrame(kernel_reg.cv_results_))

# %%
# Randomized Search
# --------------------
# With Randomized Search we can additionally use distributions from scikit-learn to define the parameter search space
param_grid = {
    "kernel": ["poly"],
    "c": [loguniform(1e-1, 2e-1), [1e-1]],
    "degree": [[2], [2, 3]],
}
kernel_reg = RandomizedSearchCV(
    KCCA(latent_dims=latent_dims), param_distributions=param_grid, cv=cv, verbose=True
).fit([X, Y])
print(pd.DataFrame(kernel_reg.cv_results_))
