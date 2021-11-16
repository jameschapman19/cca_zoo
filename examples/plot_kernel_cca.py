"""
Kernel CCA and Nonparametric CCA
===================================

This script demonstrates how to use kernel and nonparametric methods
"""

# %%
import numpy as np

from cca_zoo.data import generate_covariance_data
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.models import KCCA

# %%
np.random.seed(42)
n = 200
p = 100
q = 100
latent_dims = 1
cv = 3

(X, Y), (tx, ty) = generate_covariance_data(
    n, view_features=[p, q], latent_dims=latent_dims, correlation=[0.9]
)

# %%
# Linear
c1 = [0.9, 0.99]
c2 = [0.9, 0.99]
param_grid = {"kernel": ["linear"], "c": [c1, c2]}
kernel_reg = (
    GridSearchCV(
        KCCA(latent_dims=latent_dims), param_grid=param_grid, cv=cv, verbose=True
    )
        .fit([X, Y])
        .best_estimator_
)

# %%
# Polynomial
degree1 = [2, 3]
degree2 = [2, 3]
param_grid = {"kernel": ["poly"], "degree": [degree1, degree2], "c": [c1, c2]}
kernel_poly = (
    GridSearchCV(
        KCCA(latent_dims=latent_dims), param_grid=param_grid, cv=cv, verbose=True
    )
        .fit([X, Y])
        .best_estimator_
)

# %%
# kernel cca (gaussian/rbf)
gamma1 = [1e1, 1e2, 1e3]
gamma2 = [1e1, 1e2, 1e3]
param_grid = {"kernel": ["rbf"], "gamma": [gamma1, gamma2], "c": [c1, c2]}
kernel_poly = (
    GridSearchCV(
        KCCA(latent_dims=latent_dims), param_grid=param_grid, cv=cv, verbose=True
    )
        .fit([X, Y])
        .best_estimator_
)
