# Importing necessary libraries
import jax.numpy as jnp
import numpy as np


# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
# @partial(jit, static_argnums=3)
def calc_numpy(X, Y, k):
    C = X.T @ Y
    U, S, Vt = jnp.linalg.svd(C)
    idx = np.argsort(S, axis=0)[::-1][:k]
    U = U[:, idx]
    Vt = Vt[:, idx]
    S = S[idx]
    return S, U, Vt.T
