from ccagame import pca
import jax.numpy as jnp
import numpy as np

# Matrix X for which we want to find the PCA
X = jnp.array([[7.,4.,5.,2.],
            [2.,19.,6.,13.],
            [34.,23.,67.,23.],
            [1.,7.,8.,4.]])

vals_np, vecs_np = pca.calc_numpy(X)
print("\n Eigenvalues calculated using numpy are :\n",vals_np)
print("\n Eigenvectors calculated using numpy are :\n",vecs_np)
vals, vecs = pca.calc_alphaeigengame(X,4)
print("\n Eigenvalues calculate using the Eigengame are :\n",vals)
print("\n Eigenvectors calculated using the Eigengame are :\n",vecs)
print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",np.sum((np.abs(vecs_np)-np.abs(vecs))**2,axis=0))