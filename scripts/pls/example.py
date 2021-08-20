# %%

import numpy as np
# Imports
from ccagame.pls import calc_numpy, calc_sklearn
from jax import random

# %%

# Parameters
random_state = 0
n = 100
p = 10
q = 10
latent_dims = 3
max_iter = 500
riemannian_projection = False
#EIGENGAME IS SUPER SENSITIVE TO LR IT LIKES A BIG ONE
lr = 1e-3
initialization = 'uniform'

# %%

# Data Generation
key = random.PRNGKey(random_state)
key, subkey = random.split(key)
X = random.normal(key, (n, p))
Y = random.normal(subkey, (n, q))

# %%

# Model
corr_sk, U1sk, V1sk = calc_sklearn(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using scikit are :\n", corr_sk)
corr_np, U1np, V1np = calc_numpy(X, Y, k=latent_dims)
print("\n Eigenvalues calculated using numpy are :\n", corr_np)
corr, U1, V1 = calc_game(X, Y, latent_dims, lr=1e-1, iterations=max_iter,
                         riemannian_projection=riemannian_projection, random_state=random_state,
                         initialization=initialization,simultaneous=True)
print("\n Eigenvalues calculated using game are :\n", corr)
print("\n Left Eigenvectors calculated using numpy are :\n", U1np)
print("\n Left Eigenvectors calculated using the Eigengame are :\n", U1)
print("\n Right Eigenvectors calculated using numpy are :\n", V1np)
print("\n Right Eigenvectors calculated using the Eigengame are :\n", V1)
print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",
      np.sum((np.abs(U1np) - np.abs(U1)) ** 2, axis=0))
