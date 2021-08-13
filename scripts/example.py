# %%

import numpy as np
# Imports
from ccagame.cca import calc_numpy,calc_sklearn,calc_game, calc_lscca, calc_lscca_exact
from jax import random

# %%

# Parameters
random_state = 0
n = 50
p = 8
q = 9
latent_dims = 5
max_iter = 200
riemannian_projection = False
lr = 5e-1
initialization = 'uniform'

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
corr_lse, U1_lse, V1_lse = calc_lscca_exact(X, Y, latent_dims, iterations=max_iter,
                                    random_state=random_state,
                                    initialization=initialization)
print("\n Eigenvalues calculated using lsccae are :\n", corr_lse)
corr_ls, U1_ls, V1_ls = calc_lscca(X, Y, latent_dims, iterations=max_iter,
                                    random_state=random_state,
                                    initialization=initialization)
print("\n Eigenvalues calculated using lscca are :\n", corr_ls)
corr, U1, V1 = calc_game(X, Y, latent_dims, lr=lr, iterations=max_iter,
                                    riemannian_projection=False, random_state=random_state,
                                    initialization=initialization, simultaneous=True)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
print("\n Eigenvalues calculated using lscca are :\n", corr_ls)
print("\n Eigenvalues calculate using the game are :\n", corr)

