from ccagame import pca
import jax.numpy as jnp
import numpy as np

# Matrix X for which we want to find the PCA
X = jnp.array([[7.,4.,5.,2.],
            [2.,19.,6.,13.],
            [34.,23.,67.,23.],
            [1.,7.,8.,4.]])

# X = jnp.array([[9.,0.,0.,0.],
#             [0.,8.,0.,0.],
#             [0.,0.,7.,0.],
#             [0.,0.,0.,1.]])

# Centre the data
# X = X-jnp.mean(X,axis=0)
# print(X)

p,q = pca.calc_numpy(X)
V1 = pca.calc_alphaeigengame(X,4)
print("\n Eigenvalues calculated using numpy are :\n",p)
print("\n Eigenvectors calculated using numpy are :\n",q)
print("\n Eigenvalues calculate using the Eigengame are :\n",calc_eigengame_eigenvalues(X,V1))
print("\n Eigenvectors calculated using the Eigengame are :\n",V1)
print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",np.sum((np.abs(q)-np.abs(V1))**2,axis=0))