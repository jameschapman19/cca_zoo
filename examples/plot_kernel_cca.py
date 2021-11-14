"""
Kernel CCA
===============================
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

(X, Y), (tx, ty) = generate_covariance_data(n, view_features=[p, q], latent_dims=latent_dims,
                                            correlation=[0.9])

# %%
c1 = [0.9, 0.99]
c2 = [0.9, 0.99]
param_grid = {'kernel': ['linear'], 'c': [c1, c2]}
kernel_reg = GridSearchCV(KCCA(latent_dims=latent_dims), param_grid=param_grid,
                          cv=cv,
                          verbose=True).fit([X, Y]).best_estimator_

# %%
c1 = [0.9, 0.99]
c2 = [0.9, 0.99]
param_grid = {'kernel': ['linear'], 'c': [c1, c2]}
kernel_reg = GridSearchCV(KCCA(latent_dims=latent_dims), param_grid=param_grid,
                          cv=cv,
                          verbose=True).fit([X, Y]).best_estimator_

# %%
c1 = [0.9, 0.99]
c2 = [0.9, 0.99]
param_grid = {'kernel': ['linear'], 'c': [c1, c2]}
kernel_reg = GridSearchCV(KCCA(latent_dims=latent_dims), param_grid=param_grid,
                          cv=cv,
                          verbose=True).fit([X, Y]).best_estimator_
