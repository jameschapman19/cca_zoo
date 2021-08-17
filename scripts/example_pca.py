import jax.numpy as jnp
from jax import random

from ccagame import pca


n = 100
p = 10
dims = 3
lr=1e-3

key = random.PRNGKey(0)
X = random.normal(key, (n, p))

vals_np, vecs_np = pca.calc_numpy(X,dims)
print("\n Eigenvalues calculated using numpy are :\n", vals_np)
print("\n Sum :\n", jnp.sum(vals_np))
vals, vecs, obj = pca.calc_game(X, dims, lr=1e-1, simultaneous=True)
print("\n Eigenvalues calculate using the Eigengame are :\n", vals)
print("\n Sum :\n", jnp.sum(vals))
vals_kr, vecs_kr, obj_kr = pca.calc_krasulina(X, dims, lr=lr)
print("\n Eigenvalues calculated using krasulina are :\n", vals_kr)
print("\n Sum :\n", jnp.sum(vals_kr))
vals_oj, vecs_oj, obj_oj = pca.calc_oja(X, dims, lr=lr)
print("\n Eigenvalues calculated using oja are :\n", vals_oj)
print("\n Sum :\n", jnp.sum(vals_oj))
vals_gha, vecs_gha, obj_gha = pca.calc_gha(X, dims, lr=lr)
print("\n Eigenvalues calculated using gha are :\n", vals_gha)
print("\n Sum :\n", jnp.sum(vals_gha))
