# %%

import numpy as np
# Imports
from ccagame.pls import calc_numpy, calc_sklearn, calc_game, calc_sgd
from jax import random
import jax.numpy as jnp
# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 11
latent_dims = 3
max_iter = 500
riemannian_projection = False
#EIGENGAME IS SUPER SENSITIVE TO LR IT LIKES A BIG ONE
lr = 1e-1
initialization = 'uniform'

# %%

# Data Generation
key = random.PRNGKey(random_state)
key, subkey = random.split(key)
X = random.normal(key, (n, p))
X = X/jnp.linalg.norm(X,axis=0)
Y = random.normal(subkey, (n, q))
Y = Y/jnp.linalg.norm(Y,axis=0)

# %%

# Model
corr_sk, U1sk, V1sk = calc_sklearn(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
print("\n Sum :\n", jnp.sum(corr_sk))

corr_np, U1np, V1np = calc_numpy(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
print("\n Sum :\n", jnp.sum(corr_np))

corr_inc, U1inc, V1inc = calc_sgd(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using numpy are :\n", corr_inc)
print("\n Sum :\n", jnp.sum(corr_inc))

corr_sg, U1sg, V1sg = calc_sgd(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using sgd are :\n", corr_sg)
print("\n Sum :\n", jnp.sum(corr_sg))

corr, U1, V1 = calc_game(X, Y, latent_dims, lr=lr, iterations=max_iter,
                         riemannian_projection=riemannian_projection, random_state=random_state,
                         simultaneous=True)
print("\n Eigenvalues calculated using game are :\n", corr)
print("\n Sum :\n", jnp.sum(corr))
