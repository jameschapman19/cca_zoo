import jax.numpy as jnp
from jax import random

from ccagame import pca


n = 100
p = 10
dims = 3
lr=1e-1

key = random.PRNGKey(0)
X = random.normal(key, (n, p))

vals_kr, vecs_kr, obj_kr = pca.calc_krasulina(X, dims, lr=lr)
print("\n Eigenvalues calculated using krasulina are :\n", vals_kr)
print("\n Sum :\n", jnp.sum(vals_kr))
print("\n Eigenvectors calculated using krasulina are :\n", vecs_kr)
vals_oj, vecs_oj, obj_oj = pca.calc_oja(X, dims, lr=lr)
print("\n Eigenvalues calculated using oja are :\n", vals_oj)
print("\n Sum :\n", jnp.sum(vals_oj))
print("\n Eigenvectors calculated using oja are :\n", vecs_oj)
vals, vecs, obj = pca.calc_game(X, dims, lr=lr, simultaneous=True)
print("\n Eigenvalues calculate using the Eigengame are :\n", vals)
print("\n Sum :\n", jnp.sum(vals))
print("\n Eigenvectors calculated using the Eigengame are :\n", vecs)
vals_gha, vecs_gha, obj_gha = pca.calc_gha(X, dims, lr=lr)
print("\n Eigenvalues calculated using gha are :\n", vals_gha)
print("\n Sum :\n", jnp.sum(vals_gha))
print("\n Eigenvectors calculated using gha are :\n", vecs_gha)
vals_np, vecs_np = pca.calc_numpy(X)
print("\n Eigenvalues calculated using numpy are :\n", vals_np)
print("\n Eigenvectors calculated using numpy are :\n", vecs_np)
