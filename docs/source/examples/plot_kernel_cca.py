"""
Exploring Canonical Correlation Analysis (CCALoss) with Kernel & Nonparametric Methods
=================================================================================

This script provides a walkthrough on using kernel and nonparametric techniques
to perform Canonical Correlation Analysis (CCALoss) on a simulated dataset.
"""

# %%
# Dependencies
# ------------
import numpy as np
from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

# %%
# Dataset Generation
# ------------------
# Setting a seed to ensure reproducibility.
np.random.seed(42)

# Configuring and generating a simulated dataset with given specifications.
n, p, q, latent_dims, correlation = 200, 100, 100, 1, 0.9
(X, Y) = LinearSimulatedData(
    view_features=[p, q], latent_dims=latent_dims, correlation=[correlation]
).sample(n)

# Specifying the number of folds for cross-validation.
cv = 3


# %%
# Custom Kernel Definition
# ------------------------
def my_kernel(X, Y, param=0, **kwargs):
    """
    Custom kernel function that introduces controlled random noise
    to the linear kernel computation.
    """
    M = np.random.rand(X.shape[0], X.shape[0])
    M += param
    return X @ M @ M.T @ Y.T


# Initializing the KCCA model with the custom kernel and specified parameters.
kernel_custom = KCCA(
    latent_dimensions=latent_dims,
    kernel=[my_kernel, my_kernel],
    kernel_params=[{"param": 1}, {"param": 1}],
).fit([X, Y])

# %%
# Linear Kernel-based CCALoss
# -----------------------
c_values = [0.9, 0.99]
param_grid_linear = {"kernel": ["linear"], "c": [c_values, c_values]}

# Tuning hyperparameters using GridSearchCV for the linear kernel.
kernel_linear = GridSearchCV(
    KCCA(latent_dimensions=latent_dims),
    param_grid=param_grid_linear,
    cv=cv,
    verbose=True,
).fit([X, Y])

# %%
# Polynomial Kernel-based CCALoss
# ---------------------------
degrees = [2, 3]
param_grid_poly = {
    "kernel": ["poly"],
    "degree": [degrees, degrees],
    "c": [c_values, c_values],
}

# Tuning hyperparameters using GridSearchCV for the polynomial kernel.
kernel_poly = (
    GridSearchCV(
        KCCA(latent_dimensions=latent_dims),
        param_grid=param_grid_poly,
        cv=cv,
        verbose=True,
    )
    .fit([X, Y])
    .best_estimator_
)

# %%
# Gaussian/RBF Kernel-based CCALoss
# -----------------------------
gammas = [1e-1, 1e-2]
param_grid_rbf = {
    "kernel": ["rbf"],
    "gamma": [gammas, gammas],
    "c": [c_values, c_values],
}

# Tuning hyperparameters using GridSearchCV for the Gaussian/RBF kernel.
kernel_rbf = (
    GridSearchCV(
        KCCA(latent_dimensions=latent_dims),
        param_grid=param_grid_rbf,
        cv=cv,
        verbose=True,
    )
    .fit([X, Y])
    .best_estimator_
)
