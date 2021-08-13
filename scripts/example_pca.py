from ccagame import pca
import jax.numpy as jnp
import numpy as np


X=np.random.rand(100,100)
X=X/np.linalg.norm(X,axis=0)

a=np.linalg.eigvals(X)


# Matrix X for which we want to find the PCA
X = jnp.array([[7., 4., 5., 2.],
               [2., 19., 6., 13.],
               [34., 23., 67., 23.],
               [1., 7., 8., 4.]])

vals, vecs = pca.calc_alphaeigengame(X, 3,lr=1e-3, simultaneous=True)
print("\n Eigenvalues calculate using the Eigengame are :\n", vals)
print("\n Eigenvectors calculated using the Eigengame are :\n", vecs)
vals, vecs = pca.calc_alphaeigengame(X, 3,lr=1e-3, simultaneous=False)
print("\n Eigenvalues calculate using the Eigengame are :\n", vals)
print("\n Eigenvectors calculated using the Eigengame are :\n", vecs)
vals_gha, vecs_gha = pca.calc_gha(X,3,lr=1e-4)
print("\n Eigenvalues calculated using gha are :\n", vals_gha)
print("\n Eigenvectors calculated using gha are :\n", vecs_gha)
vals_oj, vecs_oj = pca.calc_oja(X,3,lr=1e-3)
print("\n Eigenvalues calculated using oja are :\n", vals_oj)
print("\n Eigenvectors calculated using oja are :\n", vecs_oj)
vals_np, vecs_np = pca.calc_numpy(X)
print("\n Eigenvalues calculated using numpy are :\n", vals_np)
print("\n Eigenvectors calculated using numpy are :\n", vecs_np)

print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",
      np.sum((np.abs(vecs_np) - np.abs(vecs)) ** 2, axis=0))
