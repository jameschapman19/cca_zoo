# Importing necessary libraries
from functools import partial

import jax.numpy as jnp
from jax import jit, lax


# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
@partial(jit, static_argnums=(2))
def calc_numpy(X, Y, k):
    C = X.T @ Y
    U, S, Vt = jnp.linalg.svd(C)
    U = lax.dynamic_slice(U, (0, 0), (U.shape[0], k))
    Vt = lax.dynamic_slice(Vt, (0, 0), (Vt.shape[0], k))
    S = lax.dynamic_slice(S.reshape(1, C.shape[0]), (0, 0), (1, k))
    return S, U, Vt.T
