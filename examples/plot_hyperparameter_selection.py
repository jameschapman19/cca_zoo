"""
Kernel CCA Hyperparameter Tuning
================================

This script demonstrates hyperparameter optimization for Kernel Canonical
Correlation Analysis (Kernel CCA) using both grid search and randomized search methods.

Note:
- The grid search approach involves exhaustively trying every combination of provided parameters.
- The randomized search randomly samples from the provided parameter space.
"""

# %%
# Dependencies
# ------------
import numpy as np
import pandas as pd
from scipy.stats import loguniform

from cca_zoo.datasets import JointData
from cca_zoo.model_selection import GridSearchCV, RandomizedSearchCV
from cca_zoo.nonparametric import KCCA

# %%
# Dataset Preparation
# -------------------
# Fixing a seed for reproducibility.
np.random.seed(42)

# Creating a linear dataset having 200 samples, 100 features per view,
# a single latent dimension, and a 0.9 correlation between the representations.
n = 200
p = 100
q = 100
latent_dimensions = 1
correlation = 0.9

data = JointData(
    view_features=[p, q], latent_dimensions=latent_dimensions, correlation=[correlation]
)
(X, Y) = data.sample(n)

# Setting up 3-fold cross-validation.
cv = 3

# %%
# Grid Search Hyperparameter Tuning
# ---------------------------------
# Parameter selection is similar to scikit-learn, with slight variations in parameter grid format.
# The search space can include:
#  - Single values, which will apply to each view.
#  - Lists per view.
#  - Mixtures of single values for one view and distributions or lists for the other.

# Here, we'll try out the polynomial kernel, varying regularization (c) and polynomial degree.
param_grid = {"kernel": ["poly"], "c": [[1e-1], [1e-1, 2e-1]], "degree": [[2], [2, 3]]}

# Using GridSearchCV to optimize KCCA with the polynomial kernel.
kernel_reg_grid = GridSearchCV(
    KCCA(latent_dimensions=latent_dimensions),
    param_grid=param_grid,
    cv=cv,
    verbose=True,
).fit([X, Y])

# Displaying the grid search results.
print(pd.DataFrame(kernel_reg_grid.cv_results_))

# %%
# Randomized Search Hyperparameter Tuning
# ---------------------------------------
# With Randomized Search, we can also use distributions (like loguniform) from scikit-learn.

# Again, we're defining parameters for the polynomial kernel.
param_grid_random = {
    "c": [loguniform(1e-1, 2e-1), [1e-1]],
    "degree": [[2], [2, 3]],
}

# Using RandomizedSearchCV for optimization.
kernel_reg_random = RandomizedSearchCV(
    KCCA(latent_dimensions=latent_dimensions, kernel="poly"),
    param_distributions=param_grid_random,
    cv=cv,
    verbose=True,
).fit([X, Y])

# Displaying the randomized search results.
print(pd.DataFrame(kernel_reg_random.cv_results_))
