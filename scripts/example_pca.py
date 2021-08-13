import jax.numpy as jnp
from jax import random

from ccagame import pca

# Matrix X for which we want to find the PCA
# X = jnp.array([[7., 4., 5., 2.],
#               [2., 19., 6., 13.],
#               [34., 23., 67., 23.],
#               [1., 7., 8., 4.]])

n = 100
p = 20
dims = 10

key = random.PRNGKey(0)
X = random.normal(key, (n, p))

vals_oj, vecs_oj, obj = pca.calc_krasulina(X, dims, lr=1e-1)
print("\n Eigenvalues calculated using oja are :\n", vals_oj)
print("\n Sum :\n", jnp.sum(vals_oj))
print("\n Eigenvectors calculated using oja are :\n", vecs_oj)
vals_oj, vecs_oj, obj = pca.calc_oja(X, dims, lr=1e-1)
print("\n Eigenvalues calculated using oja are :\n", vals_oj)
print("\n Sum :\n", jnp.sum(vals_oj))
print("\n Eigenvectors calculated using oja are :\n", vecs_oj)
vals, vecs, obj = pca.calc_alphaeigengame(X, dims, lr=1e-1, simultaneous=True)
print("\n Eigenvalues calculate using the Eigengame are :\n", vals)
print("\n Eigenvectors calculated using the Eigengame are :\n", vecs)
vals, vecs, obj = pca.calc_alphaeigengame(X, dims, lr=1e-1, simultaneous=False)
print("\n Eigenvalues calculate using the Eigengame are :\n", vals)
print("\n Eigenvectors calculated using the Eigengame are :\n", vecs)
vals_gha, vecs_gha, obj = pca.calc_gha(X, dims, lr=1e-1)
print("\n Eigenvalues calculated using gha are :\n", vals_gha)
print("\n Eigenvectors calculated using gha are :\n", vecs_gha)
vals_np, vecs_np = pca.calc_numpy(X)
print("\n Eigenvalues calculated using numpy are :\n", vals_np)
print("\n Eigenvectors calculated using numpy are :\n", vecs_np)
