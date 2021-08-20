# Importing necessary libraries

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from .utils import TV, initialize


# Update rule to be used for calculating eigenvectors
# @partial(jit, static_argnums=(2))
def update(X, Y, U, S, V, k):
    uhat = X @ U
    u_orth = X - X @ U @ U.T
    vhat = Y @ V
    v_orth = Y - Y @ V @ V.T
    Q = jnp.vstack((jnp.hstack((jnp.diag(S) + uhat.T @ vhat, jnp.linalg.norm(v_orth) * uhat.T)),
                   jnp.hstack((jnp.linalg.norm(u_orth) * vhat, jnp.atleast_2d(jnp.linalg.norm(u_orth) * jnp.linalg.norm(v_orth))))))
    U_, S, V_ = jnp.linalg.svd(Q)
    U = jnp.hstack((U, u_orth.T / jnp.linalg.norm(u_orth)))@U_[:,:k]
    V = jnp.hstack((V, v_orth.T / jnp.linalg.norm(v_orth)))@V_[:,:k]
    return U, S[:k], V


# Run the update step iteratively across all eigenvectors
def calc_incremental(X, Y, k, iterations=100,
                     random_state=0):
    U, V = initialize(X, Y, k, 'random', random_state)
    S = np.zeros(k)
    for i in range(iterations):
        for idx in range(X.shape[0]):
            U, S, V = update(X[idx, None], Y[idx, None], U, S, V, k)
            print(f'iteration {i}: {TV(X, Y, U, V)}')
    return TV(X, Y, U, V), U, V
