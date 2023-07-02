"""
Kernel CCA and Nonparametric CCA
===================================

This script demonstrates how to use kernel and nonparametric methods
to perform canonical correlation analysis (CCA) on simulated data.
"""

# %%
# Import libraries
import numpy as np

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

# %%
# Data
# -----
# We set the random seed for reproducibility
np.random.seed(42)

# We generate a linear dataset with 200 samples, 100 features per view,
# 1 latent dimension and a correlation of 0.9 between the views
n = 200
p = 100
q = 100
latent_dims = 1
correlation = 0.9

(X, Y) = LinearSimulatedData(
    view_features=[p, q], latent_dims=latent_dims, correlation=[correlation]
).sample(n)

# We use 3-fold cross-validation for model selection
cv = 3


# %%
# Custom Kernel
def my_kernel(X, Y, param=0, **kwargs):
    """
    We create a custom kernel that adds some random noise to the linear kernel:

    """
    # We generate a random matrix with values between 0 and 1
    M = np.random.rand(X.shape[0], X.shape[0])
    # We add the parameter to the matrix to control the noise level
    M += param
    # We return the product of X, M, M.T and Y.T as the kernel matrix
    return X @ M @ M.T @ Y.T


# We create a KCCA object with our custom kernel and a parameter of 1 for both views
kernel_custom = KCCA(
    latent_dimensions=latent_dims,
    kernel=[my_kernel, my_kernel],
    kernel_params=[{"param": 1}, {"param": 1}],
).fit([X, Y])

# %%
# Linear Kernel
# We define a list of regularization parameters to try for both views
c1 = [0.9, 0.99]
c2 = [0.9, 0.99]

# We create a parameter grid with the linear kernel and the regularization parameters
param_grid = {"kernel": ["linear"], "c": [c1, c2]}

# We use GridSearchCV to find the best KCCA model with the linear kernel
kernel_linear = GridSearchCV(
    KCCA(latent_dimensions=latent_dims), param_grid=param_grid, cv=cv, verbose=True
).fit([X, Y])

# %%
# Polynomial Kernel
# We define a list of polynomial degrees to try for both views
degree1 = [2, 3]
degree2 = [2, 3]

# We create a parameter grid with the polynomial kernel, the degrees and the regularization parameters
param_grid = {"kernel": ["poly"], "degree": [degree1, degree2], "c": [c1, c2]}

# We use GridSearchCV to find the best KCCA model with the polynomial kernel
kernel_poly = (
    GridSearchCV(
        KCCA(latent_dimensions=latent_dims), param_grid=param_grid, cv=cv, verbose=True
    )
    .fit([X, Y])
    .best_estimator_
)

# %%
# Gaussian/RBF Kernel
# We define a list of gamma values to try for both views
gamma1 = [1e-1, 1e-2]
gamma2 = [1e-1, 1e-2]

# We create a parameter grid with the Gaussian/RBF kernel, the gammas and the regularization parameters
param_grid = {"kernel": ["rbf"], "gamma": [gamma1, gamma2], "c": [c1, c2]}

# We use GridSearchCV to find the best KCCA model with the Gaussian/RBF kernel
kernel_rbf = (
    GridSearchCV(
        KCCA(latent_dimensions=latent_dims), param_grid=param_grid, cv=cv, verbose=True
    )
    .fit([X, Y])
    .best_estimator_
)
