"""
Kernel CCA and Nonparametric CCA
===================================

This script demonstrates how to use kernel and nonparametric methods
"""

# %%
import numpy as np

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.models import KCCA

# %%
# Data
# -----
np.random.seed(42)
n = 200
p = 100
q = 100
latent_dims = 1
cv = 3

(X, Y) = LinearSimulatedData(
    view_features=[p, q], latent_dims=latent_dims, correlation=[0.9]
).sample(n)


# %%
# Custom Kernel
def my_kernel(X, Y, param=0):
    """
    We create a custom kernel:

    """

    return np.random.normal(0, param)


kernel_custom = KCCA(
    latent_dims=latent_dims,
    kernel=[my_kernel, my_kernel],
    kernel_params=[{"param": 1}, {"param": 1}],
).fit([X, Y])

# %%
# Linear
c1 = [0.9, 0.99]
c2 = [0.9, 0.99]
param_grid = {"kernel": ["linear"], "c": [c1, c2]}
kernel_reg = GridSearchCV(
    KCCA(latent_dims=latent_dims), param_grid=param_grid, cv=cv, verbose=True
).fit([X, Y])

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
gamma1 = [1e-1, 1e-2]
gamma2 = [1e-1, 1e-2]
param_grid = {"kernel": ["rbf"], "gamma": [gamma1, gamma2], "c": [c1, c2]}
kernel_poly = (
    GridSearchCV(
        KCCA(latent_dims=latent_dims), param_grid=param_grid, cv=cv, verbose=True
    )
    .fit([X, Y])
    .best_estimator_
)


# %%
# Custom Kernel
def my_kernel(X, Y, param=0):
    """
    We create a custom kernel:

    """
    M = np.random.rand(X.shape[0], X.shape[0]) + param
    return X @ M @ M.T @ Y.T


kernel_custom = KCCA(
    latent_dims=latent_dims,
    kernel=[my_kernel, my_kernel],
    kernel_params=[{"param": 1}, {"param": 1}],
).fit([X, Y])
