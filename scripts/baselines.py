# %%

import numpy as np
# Imports
from ccagame.cca import calc_numpy, calc_sklearn, calc_game, \
    calc_lscca, calc_lscca_exact, calc_genoja, calc_ccalin
from jax import random

# %%

# Parameters
random_state = 0
n = 1000
p = 200
q = 200
latent_dims = 2
max_iter = 300
riemannian_projection = False
lr = 1
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
corr_sk, U1sk, V1sk = calc_sklearn(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
corr, U1, V1 = calc_ccalin(X, Y, latent_dims, iterations=max_iter,
                           random_state=random_state, verbose=True
                           )
print("\n Eigenvalues calculated using ccalin are :\n", corr)
corr, U1, V1 = calc_genoja(X, Y, latent_dims, iterations=max_iter,
                           random_state=random_state,
                           initialization=initialization)
print("\n Eigenvalues calculated using genoja are :\n", corr)
corr, U1, V1 = calc_game(X, Y, latent_dims, lr=lr, iterations=max_iter,
                         riemannian_projection=riemannian_projection, random_state=random_state,
                         initialization=initialization, simultaneous=True)
print("\n Eigenvalues calculated using game are :\n", corr)
corr_lse, U1_lse, V1_lse = calc_lscca_exact(X, Y, latent_dims, iterations=max_iter,
                                            random_state=random_state,
                                            initialization=initialization)
print("\n Eigenvalues calculated using lsccae are :\n", corr_lse)
corr_ls, U1_ls, V1_ls = calc_lscca(X, Y, latent_dims, iterations=max_iter,
                                   random_state=random_state,
                                   initialization=initialization)
print("\n Eigenvalues calculated using lscca are :\n", corr_ls)
corr_np, U1np, V1np = calc_numpy(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
print("\n Eigenvalues calculated using lscca are :\n", corr_ls)
print("\n Eigenvalues calculate using the game are :\n", corr)