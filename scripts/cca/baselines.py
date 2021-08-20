# %%

import numpy as np
# Imports
from ccagame.cca import calc_numpy, calc_sklearn, calc_game, \
    calc_lscca, calc_lscca_exact, calc_genoja, calc_ccalin, calc_lagrangeminmax
from jax import random
import jax.numpy as jnp

# %%

# Parameters
random_state = 0
n = 1000
p = 100
q = 110
latent_dims = 5
max_iter = 300
riemannian_projection = True
initialization = 'random'
lr=1

# %%

# Data Generation
key = random.PRNGKey(random_state)
key, subkey = random.split(key)
X = random.normal(key, (n, p))
X = X / jnp.linalg.norm(X, axis=0)
Y = random.normal(subkey, (n, q))
Y = Y / jnp.linalg.norm(Y, axis=0)

# %%

# Model
corr_sk, Usk, Vsk = calc_sklearn(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
print("\n Sum :\n", jnp.sum(corr_sk))

corr, U, V = calc_game(X, Y, latent_dims, lr=lr, iterations=max_iter,
                         riemannian_projection=riemannian_projection,
                         simultaneous=True)
print("\n Eigenvalues calculated using game are :\n", corr)
print("\n Sum :\n", jnp.sum(corr))

corr_go, Ugo, Vgo = calc_genoja(X, Y, latent_dims, iterations=max_iter, alpha=lr,beta_0=lr)
print("\n Eigenvalues calculated using genoja are :\n", corr)
print("\n Sum :\n", jnp.sum(corr_go))

corr_l, Ul, Vl = calc_lagrangeminmax(X, Y, latent_dims, iterations=max_iter)
print("\n Eigenvalues calculated using lagrangeminmax are :\n", corr)
print("\n Sum :\n", jnp.sum(corr_l))

corr_cl, Ucl, Vcl = calc_ccalin(X, Y, latent_dims, iterations=max_iter,
                           random_state=random_state, verbose=True
                           )
print("\n Eigenvalues calculated using ccalin are :\n", corr)
print("\n Sum :\n", jnp.sum(corr_cl))

corr_lse, U_lse, V_lse = calc_lscca_exact(X, Y, latent_dims, iterations=max_iter,
                                            random_state=random_state,
                                            initialization=initialization)
print("\n Eigenvalues calculated using lsccae are :\n", corr_lse)
print("\n Sum :\n", jnp.sum(corr_lse))

corr_ls, U_ls, V_ls = calc_lscca(X, Y, latent_dims, iterations=max_iter,
                                   random_state=random_state,
                                   initialization=initialization)
print("\n Eigenvalues calculated using lscca are :\n", corr_ls)
print("\n Sum :\n", jnp.sum(corr_ls))
