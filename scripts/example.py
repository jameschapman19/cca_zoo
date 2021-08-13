# %%

import numpy as np
# Imports
from ccagame.ccagame import calc_numpy, calc_sklearn, calc_eigengame
from jax import random

# %%

# Parameters
random_state = 0
n = 50
p = 8
q = 9
latent_dims = 5
max_iter = 90
riemannian_projection = False
lr = 5e-1
initialization = 'random'

# %%

# Data Generation
key = random.PRNGKey(random_state)
key, subkey = random.split(key)
X = random.normal(key, (n, p))
Y = random.normal(subkey, (n, q))

Xnp = np.array(X)
Ynp = np.array(Y)

# %%

# Model
corr_sk, U1sk, V1sk = calc_sklearn(X, Y, n=latent_dims)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
corr_np, U1np, V1np = calc_numpy(X, Y, n=latent_dims)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
corr_r, U1_r, V1_r = calc_eigengame(X, Y, latent_dims, lr=lr, iterations=max_iter,
                                    riemannian_projection=False, random_state=random_state,
                                    initialization=initialization, simultaneous=True)
corr, U1, V1 = calc_eigengame(X, Y, latent_dims, lr=lr, iterations=max_iter,
                              riemannian_projection=False, random_state=random_state,
                              initialization=initialization, simultaneous=False)
print("\n Eigenvalues calculated using numpy are :\n", corr_sk)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
print("\n Eigenvalues calculate using the Eigengame are :\n", corr)
print("\n Left Eigenvectors calculated using numpy are :\n", U1np)
print("\n Left Eigenvectors calculated using the Eigengame are :\n", U1)
print("\n Right Eigenvectors calculated using numpy are :\n", V1np)
print("\n Right Eigenvectors calculated using the Eigengame are :\n", V1)
print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",
      np.sum((np.abs(U1np) - np.abs(U1)) ** 2, axis=0))
