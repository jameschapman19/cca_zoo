"""
Hyperparameter Selection
===========================

This script will show how to perform hyperparameter selection
for kernel CCA using grid search and randomized search methods.
"""

# %%
# Import libraries
import numpy as np
import pandas as pd
from scipy.stats import loguniform

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV, RandomizedSearchCV
from cca_zoo.nonparametric import KCCA

# %%
# Data
# ------
# We set the random seed for reproducibility
np.random.seed(42)

# We generate a linear dataset with 200 samples, 100 features per view,
# 1 latent dimension and a correlation of 0.9 between the views
n = 200
p = 100
q = 100
latent_dims = 1
correlation = 0.9

data = LinearSimulatedData(
    view_features=[p, q], latent_dims=latent_dims, correlation=[correlation]
)

(X, Y) = data.sample(n)

# We use 3-fold cross-validation for model selection
cv = 3

# %%
# Grid Search
# -------------
# Hyperparameter selection works in a very similar way to in scikit-learn where the main difference is in how we enter the parameter grid.
# We form a parameter grid with the search space for each view for each parameter.
# This search space must be entered as a list but can be any of
#  - a single value (as in "kernel") where this value will be used for each view
#  - a list for each view
#  - a mixture of a single value for one view and a distribution or list for the other

# We define a parameter grid with the polynomial kernel and different values for the regularization parameter (c) and the degree of the polynomial
param_grid = {"kernel": ["poly"], "c": [[1e-1], [1e-1, 2e-1]], "degree": [[2], [2, 3]]}

# We use GridSearchCV to find the best KCCA model with the polynomial kernel
kernel_reg = GridSearchCV(
    KCCA(latent_dimensions=latent_dims), param_grid=param_grid, cv=cv, verbose=True
).fit([X, Y])

# We print the results of the grid search as a data frame
print(pd.DataFrame(kernel_reg.cv_results_))

# %%
# Randomized Search
# --------------------
# With Randomized Search we can additionally use distributions from scikit-learn to define the parameter search space

# We define a parameter grid with the polynomial kernel and different values or distributions for the regularization parameter (c) and the degree of the polynomial
param_grid = {
    "kernel": "poly",
    "c": [loguniform(1e-1, 2e-1), [1e-1]],
    "degree": [[2], [2, 3]],
}

# We use RandomizedSearchCV to find the best KCCA model with the polynomial kernel
kernel_reg = RandomizedSearchCV(
    KCCA(latent_dimensions=latent_dims),
    param_distributions=param_grid,
    cv=cv,
    verbose=True,
).fit([X, Y])

# We print the results of the randomized search as a data frame
print(pd.DataFrame(kernel_reg.cv_results_))
